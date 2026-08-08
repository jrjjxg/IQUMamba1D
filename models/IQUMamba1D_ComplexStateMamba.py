"""Stage-4 IQUMamba with complex-state (oscillatory) selective SSMs.

Stage 295.  Only the diagonal state transition of each Mamba block changes:
it gains learnable imaginary parts, so the discretized transition

    a_t = exp(dt_t * (-exp(a_log) + i * theta))
        = exp(-dt_t * exp(a_log)) * (cos(dt_t * theta) + i sin(dt_t * theta))

is a decaying *rotation* instead of a pure decay.  Because ``dt_t`` is
data-dependent, the rotation angles are input-controlled — the selective
analogue of S4D-Lin initialization (Gu et al., 2022) and the Mamba-3
"complex state update as data-dependent rotation" formulation.  Oscillatory
state dynamics directly model carrier frequency offsets, symbol-rate
cyclostationarity and other narrowband rotations in RF mixtures, which a
non-negative real decay bank can only imitate indirectly.

Everything else — the Mamba macro-block (projections, causal depthwise conv,
SiLU gating), the surrounding Stage-4 U-Net, ASC and output head — remains
real-valued and identical to the Stage-4 baseline, so ``stage4 vs stage295``
isolates exactly one variable: real versus complex state dynamics.  This
variant is intentionally *not* phase-equivariant; it is the physics-dynamics
track, parallel to the strict-complex/equivariant C1-C5 track (290-294).

The recurrence runs as a Hillis-Steele parallel prefix scan in pure PyTorch
(complex arithmetic as paired real tensors, no custom kernels).  The scan is
wrapped in gradient checkpointing so activation memory stays at
O(B*L*D*N) rather than O(B*L*D*N*log L).
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from models.complex_scan_cuda import native_complex_scan

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn as _mamba_selective_scan_fn,
    )
except Exception:  # CUDA extension absent or ABI-incompatible: use torch fallback.
    _mamba_selective_scan_fn = None

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


def _shift_right(x: torch.Tensor, offset: int, fill: float) -> torch.Tensor:
    """Shift along dim 1 by ``offset`` steps, filling the head with ``fill``."""

    pad = x.new_full((x.shape[0], offset, *x.shape[2:]), fill)
    return torch.cat((pad, x[:, :-offset]), dim=1)


def complex_prefix_scan(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    u_real: torch.Tensor,
    u_imag: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive scan of ``h_t = a_t * h_{t-1} + u_t`` along dim 1.

    ``a`` and ``u`` are represented by paired real tensors. ``u_imag`` may be
    omitted for the Stage-295 real-valued state input. All inputs have shape
    (B, L, D, N); the complex
    state trajectory ``h`` is returned as two real tensors of that shape.
    The scan composes first-order recurrences with the associative operator
    ``(a2, u2) o (a1, u1) = (a2*a1, a2*u1 + u2)`` in log2(L) parallel rounds.
    """

    if u_imag is None:
        u_imag = torch.zeros_like(u_real)
    if not (
        a_real.shape == a_imag.shape == u_real.shape == u_imag.shape
    ):
        raise ValueError("scan inputs must share shape [B,L,D,N]")

    acc_ar, acc_ai = a_real, a_imag
    acc_ur = u_real
    acc_ui = u_imag
    length = u_real.shape[1]
    offset = 1
    while offset < length:
        shifted_ar = _shift_right(acc_ar, offset, 1.0)
        shifted_ai = _shift_right(acc_ai, offset, 0.0)
        shifted_ur = _shift_right(acc_ur, offset, 0.0)
        shifted_ui = _shift_right(acc_ui, offset, 0.0)
        new_ur = acc_ur + acc_ar * shifted_ur - acc_ai * shifted_ui
        new_ui = acc_ui + acc_ar * shifted_ui + acc_ai * shifted_ur
        new_ar = acc_ar * shifted_ar - acc_ai * shifted_ai
        new_ai = acc_ar * shifted_ai + acc_ai * shifted_ar
        acc_ar, acc_ai, acc_ur, acc_ui = new_ar, new_ai, new_ur, new_ui
        offset *= 2
    return acc_ur, acc_ui


_COMPILED_COMPLEX_SCAN = None
_COMPILED_COMPLEX_SCAN_ERROR: Exception | None = None


def compiled_complex_prefix_scan(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    u_real: torch.Tensor,
    u_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile the reference scan as a secondary CUDA optimization path."""

    global _COMPILED_COMPLEX_SCAN, _COMPILED_COMPLEX_SCAN_ERROR
    if _COMPILED_COMPLEX_SCAN_ERROR is not None:
        raise RuntimeError("compiled complex scan is unavailable") from _COMPILED_COMPLEX_SCAN_ERROR
    if _COMPILED_COMPLEX_SCAN is None:
        try:
            _COMPILED_COMPLEX_SCAN = torch.compile(
                complex_prefix_scan,
                fullgraph=True,
                dynamic=False,
            )
        except Exception as exc:
            _COMPILED_COMPLEX_SCAN_ERROR = exc
            raise RuntimeError("failed to initialize compiled complex scan") from exc
    try:
        return _COMPILED_COMPLEX_SCAN(a_real, a_imag, u_real, u_imag)
    except Exception as exc:
        _COMPILED_COMPLEX_SCAN_ERROR = exc
        raise RuntimeError("compiled complex scan execution failed") from exc


def _pack_complex_sequence(
    real: torch.Tensor,
    imag: torch.Tensor,
) -> torch.Tensor:
    """Pack [B,L,N] real/imag as the Mamba CUDA kernel's [B,N,2L]."""

    if real.shape != imag.shape or real.ndim != 3:
        raise ValueError("complex sequence parts must have equal [B,L,N] shapes")
    return (
        torch.stack((real, imag), dim=-1)
        .permute(0, 2, 1, 3)
        .reshape(real.shape[0], real.shape[2], 2 * real.shape[1])
        .contiguous()
    )


class ComplexStateSelectiveSSM(nn.Module):
    """Selective SSM (S6 macro-block) with a complex diagonal state.

    Identical to Mamba except: the state transition has learnable rotation
    frequencies ``theta`` (S4D-Lin init: pi * n), the state is complex, and
    the readout ``C`` is complex so each oscillator can be read at an
    arbitrary phase.  Input projection, causal depthwise conv, SiLU gating,
    ``dt``/``B`` selectivity and the skip ``D`` all stay real, matching the
    reference Mamba block.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        decay_init: float = 0.5,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        require_mamba_fused_scan: bool = False,
        discretization: str = "exponential_euler",
        trapezoid_lambda_init: float = 0.5,
        cyclic_theta_enable: bool = False,
        cyclic_frequencies=(),
        cyclic_max_frequency_delta: float = 0.01,
        token_stride: int = 1,
        reliability_enable: bool = False,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.d_inner = int(expand) * self.d_model
        self.dt_rank = (
            math.ceil(self.d_model / 16) if dt_rank is None else int(dt_rank)
        )
        self.scan_checkpoint = bool(scan_checkpoint)
        self.scan_backend = str(scan_backend).lower()
        self.require_mamba_fused_scan = bool(require_mamba_fused_scan)
        if self.scan_backend not in {"auto", "cuda", "torch"}:
            raise ValueError(
                "scan_backend must be one of {'auto', 'cuda', 'torch'}"
            )
        self.last_scan_backend = "uninitialized"
        self._fallback_warned = False
        self._native_fallback_warned = False
        discretization_aliases = {
            "euler": "exponential_euler",
            "exponential_euler": "exponential_euler",
            "trapezoidal": "exponential_trapezoidal",
            "exponential_trapezoidal": "exponential_trapezoidal",
        }
        try:
            self.discretization = discretization_aliases[str(discretization).lower()]
        except KeyError as exc:
            raise ValueError(
                "discretization must be exponential_euler or "
                "exponential_trapezoidal"
            ) from exc
        self.cyclic_theta_enable = bool(cyclic_theta_enable)
        self.token_stride = max(1, int(token_stride))
        self.reliability_enable = bool(reliability_enable)
        self.last_reliability = None
        self.last_trapezoid_lambda = None

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=True,
        )
        # dt (real), B (real), C (complex: real and imaginary parts).
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 3 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        if self.discretization == "exponential_trapezoidal":
            if not 0.0 < float(trapezoid_lambda_init) < 1.0:
                raise ValueError("trapezoid_lambda_init must be in (0, 1)")
            initial_logit = math.log(
                float(trapezoid_lambda_init)
                / (1.0 - float(trapezoid_lambda_init))
            )
            self.trapezoid_lambda_weight = nn.Parameter(
                torch.zeros(self.d_inner)
            )
            self.trapezoid_lambda_bias = nn.Parameter(
                torch.full((self.d_inner,), initial_logit)
            )
        else:
            self.register_parameter("trapezoid_lambda_weight", None)
            self.register_parameter("trapezoid_lambda_bias", None)

        if self.reliability_enable:
            if not 0.0 <= float(reliability_floor) < 1.0:
                raise ValueError("reliability_floor must be in [0, 1)")
            if not float(reliability_floor) < float(reliability_init) < 1.0:
                raise ValueError(
                    "reliability_init must be between reliability_floor and 1"
                )
            self.reliability_floor = float(reliability_floor)
            hidden = max(1, int(reliability_hidden))
            self.reliability_net = nn.Sequential(
                nn.Linear(3, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            normalized_init = (
                (float(reliability_init) - self.reliability_floor)
                / (1.0 - self.reliability_floor)
            )
            nn.init.zeros_(self.reliability_net[-1].weight)
            nn.init.constant_(
                self.reliability_net[-1].bias,
                math.log(normalized_init / (1.0 - normalized_init)),
            )
        else:
            self.reliability_floor = 1.0
            self.reliability_net = None

        # Mamba's dt initialization: weight uniform, bias = softplus^{-1}(dt)
        # with dt log-uniform in [dt_min, dt_max].
        dt_init_std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D-Lin: uniform decay -1/2, oscillation frequencies pi * n.
        self.a_log = nn.Parameter(
            torch.full((self.d_inner, self.d_state), math.log(decay_init))
        )
        if self.cyclic_theta_enable:
            frequencies = [float(value) for value in cyclic_frequencies]
            if not frequencies:
                raise ValueError(
                    "cyclic_frequencies must be provided when cyclic theta is enabled"
                )
            repeated = [
                frequencies[index % len(frequencies)]
                for index in range(self.d_state)
            ]
            token_frequencies = torch.tensor(repeated, dtype=torch.float32)
            token_frequencies = torch.remainder(
                token_frequencies * self.token_stride + 0.5, 1.0
            ) - 0.5
            angular_anchor = (
                2.0
                * math.pi
                * token_frequencies.unsqueeze(0)
            ).expand(self.d_inner, -1).clone()
            angular_limit = torch.full_like(
                angular_anchor,
                2.0
                * math.pi
                * float(cyclic_max_frequency_delta)
                * self.token_stride,
            )
            self.register_buffer(
                "theta_anchor_angular", angular_anchor, persistent=False
            )
            self.register_buffer(
                "theta_residual_angular_limit",
                angular_limit,
                persistent=False,
            )
            self.theta = nn.Parameter(torch.zeros_like(angular_anchor))
        else:
            self.register_buffer(
                "theta_anchor_angular", torch.empty(0), persistent=False
            )
            self.register_buffer(
                "theta_residual_angular_limit",
                torch.empty(0),
                persistent=False,
            )
            self.theta = nn.Parameter(
                (math.pi * torch.arange(self.d_state, dtype=torch.float32))
                .repeat(self.d_inner, 1)
            )
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _scan(
        self,
        a_real: torch.Tensor,
        a_imag: torch.Tensor,
        u_real: torch.Tensor,
        u_imag: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        needs_grad = torch.is_grad_enabled() and (
            a_real.requires_grad
            or u_real.requires_grad
            or (u_imag is not None and u_imag.requires_grad)
        )
        if self.scan_checkpoint and needs_grad:
            if u_imag is None:
                u_imag = torch.zeros_like(u_real)
            return checkpoint(
                complex_prefix_scan, a_real, a_imag, u_real, u_imag,
                use_reentrant=False,
            )
        return complex_prefix_scan(a_real, a_imag, u_real, u_imag)

    def _theta_values(self, theta_sign: float = 1.0) -> torch.Tensor:
        """Return pole rotations, optionally conjugated for reverse scanning.

        ``theta_sign=-1`` changes only the imaginary part of the diagonal
        transition.  It therefore evaluates the exact complex-conjugate state
        dynamics while sharing every learned real-valued projection and decay
        parameter with the forward direction.
        """

        sign = float(theta_sign)
        if sign not in (-1.0, 1.0):
            raise ValueError("theta_sign must be +1 or -1")
        if not self.cyclic_theta_enable:
            return sign * self.theta
        dt_reference = F.softplus(self.dt_proj.bias.detach()).clamp_min(1e-6)
        angular_frequency = (
            self.theta_anchor_angular
            + self.theta_residual_angular_limit * torch.tanh(self.theta)
        )
        return sign * angular_frequency / dt_reference.unsqueeze(1)

    def effective_cyclic_frequencies(self) -> torch.Tensor:
        """Return the initialized/learned pole frequencies in input cycles/sample."""

        dt_reference = F.softplus(self.dt_proj.bias.detach()).clamp_min(1e-6)
        return (
            self._theta_values().detach()
            * dt_reference.unsqueeze(1)
            / (2.0 * math.pi * self.token_stride)
        )

    def _reliability(self, x_conv: torch.Tensor) -> torch.Tensor:
        if self.reliability_net is None:
            return torch.ones(
                (*x_conv.shape[:2], 1),
                dtype=x_conv.dtype,
                device=x_conv.device,
            )
        previous = F.pad(x_conv[:, :-1], (0, 0, 1, 0))
        power = x_conv.square().mean(dim=-1).clamp_min(1e-8)
        flux = (x_conv - previous).square().mean(dim=-1).clamp_min(1e-8)
        correlation = (x_conv * previous).mean(dim=-1) / torch.sqrt(
            power * previous.square().mean(dim=-1).clamp_min(1e-8)
        )
        log_power = torch.log(power)
        log_flux = torch.log(flux)
        evidence = torch.stack(
            (
                log_power - log_power.mean(dim=1, keepdim=True),
                log_flux - log_flux.mean(dim=1, keepdim=True),
                correlation.clamp(-1.0, 1.0),
            ),
            dim=-1,
        )
        probability = torch.sigmoid(self.reliability_net(evidence))
        return self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * probability

    def _select_scan_backend(self, x: torch.Tensor) -> str:
        """Select official fused CUDA, native CUDA, or the reference scan."""

        if self.scan_backend == "torch":
            return "torch"
        if not x.is_cuda:
            if self.scan_backend == "cuda":
                raise RuntimeError("scan_backend='cuda' requires a CUDA tensor")
            return "torch"
        if (
            self.discretization == "exponential_euler"
            and _mamba_selective_scan_fn is not None
        ):
            # Reliability only rescales delta before this point. The official
            # kernel already supports an arbitrary per-token delta tensor.
            return "mamba_cuda"
        if self.require_mamba_fused_scan:
            if self.discretization != "exponential_euler":
                raise RuntimeError(
                    "The official fused selective scan requires exponential_euler "
                    "input discretization"
                )
            raise RuntimeError(
                "This stage requires the official mamba_ssm fused "
                "selective_scan_fn on CUDA; no fallback is enabled"
            )
        return "native_cuda"

    def _should_use_fused_scan(self, x: torch.Tensor) -> bool:
        """Backward-compatible predicate used by older diagnostics/tests."""

        return self._select_scan_backend(x) == "mamba_cuda"

    def _fused_scan(
        self,
        x_conv: torch.Tensor,
        z: torch.Tensor,
        dt: torch.Tensor,
        b_in: torch.Tensor,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        theta_sign: float = 1.0,
    ) -> torch.Tensor:
        """Run the mathematically equivalent complex selective scan in CUDA."""

        # Official selective_scan supports complex A. Variable complex B/C are
        # represented as real tensors with interleaved real/imag samples on
        # the last axis. Our input injection is real, while the official
        # complex readout returns 2*Re(h*C); C is scaled/conjugated so it
        # exactly matches h_real*c_real + h_imag*c_imag.
        A = torch.complex(
            -torch.exp(self.a_log.float()),
            self._theta_values(theta_sign).float(),
        )
        packed_B = _pack_complex_sequence(b_in, torch.zeros_like(b_in))
        packed_C = _pack_complex_sequence(0.5 * c_real, -0.5 * c_imag)
        output = _mamba_selective_scan_fn(
            x_conv.transpose(1, 2).contiguous(),
            dt.transpose(1, 2).contiguous(),
            A,
            packed_B,
            packed_C,
            self.D.float(),
            z=z.transpose(1, 2).contiguous(),
            delta_softplus=False,
        )
        return output.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        theta_sign: float = 1.0,
    ) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""

        length = x.shape[1]
        x_in, z = torch.chunk(self.in_proj(x), 2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[..., :length]
        x_conv = F.silu(x_conv.transpose(1, 2))

        projected = self.x_proj(x_conv)
        dt_raw, b_in, c_real, c_imag = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state, self.d_state],
            dim=-1,
        )
        # Conjugating the reverse state-space system flips both Im(A) and
        # Im(C).  The default +1 path is bit-for-bit identical to the original
        # Stage-295 implementation.
        c_imag = float(theta_sign) * c_imag
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, D)
        reliability = self._reliability(x_conv)
        if self.reliability_enable:
            dt = dt * reliability
        self.last_reliability = reliability.detach()

        selected_backend = self._select_scan_backend(x_conv)
        if selected_backend == "mamba_cuda":
            try:
                y = self._fused_scan(
                    x_conv,
                    z,
                    dt,
                    b_in,
                    c_real,
                    c_imag,
                    theta_sign=theta_sign,
                )
                self.last_scan_backend = "mamba_cuda"
            except Exception as exc:
                if self.require_mamba_fused_scan:
                    raise RuntimeError(
                        "Official mamba_ssm fused selective scan failed and "
                        "this stage forbids fallback"
                    ) from exc
                selected_backend = "native_cuda"
                if not self._fallback_warned:
                    warnings.warn(
                        "Official mamba_ssm complex selective-scan failed; "
                        "trying the native CUDA scan. "
                        f"Reason: {exc}",
                        RuntimeWarning,
                    )
                    self._fallback_warned = True
        if selected_backend != "mamba_cuda":
            decay = torch.exp(
                -dt.unsqueeze(-1) * torch.exp(self.a_log)
            )  # (B, L, D, N)
            angle = dt.unsqueeze(-1) * self._theta_values(theta_sign)
            a_real = decay * torch.cos(angle)
            a_imag = decay * torch.sin(angle)
            state_input = x_conv.unsqueeze(-1) * b_in.unsqueeze(2)
            if self.discretization == "exponential_trapezoidal":
                trapezoid_lambda = torch.sigmoid(
                    x_conv * self.trapezoid_lambda_weight
                    + self.trapezoid_lambda_bias
                )
                previous_input = F.pad(
                    state_input[:, :-1], (0, 0, 0, 0, 1, 0)
                )
                current_weight = (trapezoid_lambda * dt).unsqueeze(-1)
                previous_weight = (
                    (1.0 - trapezoid_lambda) * dt
                ).unsqueeze(-1)
                u_real = (
                    current_weight * state_input
                    + previous_weight * a_real * previous_input
                )
                u_imag = previous_weight * a_imag * previous_input
                self.last_trapezoid_lambda = trapezoid_lambda.detach()
            else:
                u_real = dt.unsqueeze(-1) * state_input
                u_imag = None
                self.last_trapezoid_lambda = None

            if selected_backend == "native_cuda":
                if u_imag is None:
                    u_imag = torch.zeros_like(u_real)
                try:
                    h_real, h_imag = native_complex_scan(
                        a_real, a_imag, u_real, u_imag
                    )
                    self.last_scan_backend = "native_cuda"
                except Exception as native_exc:
                    try:
                        h_real, h_imag = compiled_complex_prefix_scan(
                            a_real, a_imag, u_real, u_imag
                        )
                        self.last_scan_backend = "compiled_cuda"
                        if not self._native_fallback_warned:
                            warnings.warn(
                                "Native complex CUDA scan unavailable; using "
                                "torch.compile scan instead. "
                                f"Native reason: {native_exc}",
                                RuntimeWarning,
                            )
                            self._native_fallback_warned = True
                    except Exception as compiled_exc:
                        if self.scan_backend == "cuda":
                            raise RuntimeError(
                                "scan_backend='cuda' could not execute either "
                                "native or torch.compile complex scan"
                            ) from compiled_exc
                        if not self._native_fallback_warned:
                            warnings.warn(
                                "Optimized complex CUDA scans unavailable; "
                                "falling back to the much slower PyTorch scan. "
                                f"Native reason: {native_exc}; compiled reason: "
                                f"{compiled_exc}",
                                RuntimeWarning,
                            )
                            self._native_fallback_warned = True
                        h_real, h_imag = self._scan(
                            a_real, a_imag, u_real, u_imag
                        )
                        self.last_scan_backend = "torch"
            else:
                h_real, h_imag = self._scan(a_real, a_imag, u_real, u_imag)
                self.last_scan_backend = "torch"
            y = (
                (h_real * c_real.unsqueeze(2)).sum(dim=-1)
                + (h_imag * c_imag.unsqueeze(2)).sum(dim=-1)
            )
            y = y + self.D * x_conv
            y = y * F.silu(z)
        return self.out_proj(y)


class ComplexStateMambaLayer(nn.Module):
    """Drop-in replacement for the Stage-4 ``MambaLayer`` (same interface)."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        require_mamba_fused_scan: bool = False,
        **ssm_kwargs,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.ssm = ComplexStateSelectiveSSM(
            self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            require_mamba_fused_scan=require_mamba_fused_scan,
            **ssm_kwargs,
        )
        self.channel_token = bool(channel_token)

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        dims = x.shape[2:]
        x_flat = x.reshape(batch, d_model, n_tokens).transpose(-1, -2)
        out = self.ssm(self.norm(x_flat))
        return out.transpose(-1, -2).reshape(batch, d_model, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        out = self.ssm(self.norm(x_flat))
        return out.reshape(batch, n_tokens, *dims)

    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class IQUMamba1DComplexStateMamba(nn.Module):
    """Stage 295/299: Stage-4 with complex-state selective SSMs.

    Builds the unchanged Stage-4 backbone, then swaps every ``MambaLayer``
    in the encoder for a ``ComplexStateMambaLayer`` at the same position,
    width and token mode.  Stage 299 additionally enables the strict-complex
    C1 stem from Stage 290 through ``complex_stem_enable``.
    """

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        require_mamba_fused_scan: bool = False,
        mamba_discretization: str = "exponential_euler",
        trapezoid_lambda_init: float = 0.5,
        cyclic_theta_enable: bool = False,
        cyclic_frequencies=(),
        cyclic_max_frequency_delta: float = 0.01,
        reliability_enable: bool = False,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
        complex_stem_enable: bool = False,
        complex_norm_eps: float = 1e-6,
        **_: object,
    ) -> None:
        super().__init__()
        from models.IQUMamba1D import IQUMamba1D, MambaLayer

        self.backbone = IQUMamba1D(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=list(features_per_stage),
            conv_op=nn.Conv1d,
            kernel_sizes=list(kernel_sizes),
            strides=list(strides),
            n_conv_per_stage=list(n_conv_per_stage),
            num_classes=num_classes,
            n_conv_per_stage_decoder=list(n_conv_per_stage_decoder),
            deep_supervision=False,
        )
        self.complex_stem_enable = bool(complex_stem_enable)
        if self.complex_stem_enable:
            if input_channels != 2:
                raise ValueError("The strict-complex stem expects one I/Q mixture")
            from models.IQUMamba1D_ComplexStage4 import ComplexStem1d

            self.backbone.encoder.stem = ComplexStem1d(
                int(features_per_stage[0]),
                blocks=int(n_conv_per_stage[0]),
                kernel_size=int(kernel_sizes[0]),
                norm_eps=float(complex_norm_eps),
            )
        replaced = 0
        mamba_layers = self.backbone.encoder.mamba_layers
        cumulative_stride = 1
        for index, layer in enumerate(mamba_layers):
            stride = strides[index]
            if isinstance(stride, (list, tuple)):
                stride = stride[0]
            cumulative_stride *= int(stride)
            if isinstance(layer, MambaLayer):
                mamba_layers[index] = ComplexStateMambaLayer(
                    dim=layer.dim,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    channel_token=layer.channel_token,
                    scan_checkpoint=scan_checkpoint,
                    scan_backend=scan_backend,
                    require_mamba_fused_scan=require_mamba_fused_scan,
                    discretization=mamba_discretization,
                    trapezoid_lambda_init=trapezoid_lambda_init,
                    cyclic_theta_enable=cyclic_theta_enable,
                    cyclic_frequencies=cyclic_frequencies,
                    cyclic_max_frequency_delta=cyclic_max_frequency_delta,
                    token_stride=cumulative_stride,
                    reliability_enable=reliability_enable,
                    reliability_hidden=reliability_hidden,
                    reliability_floor=reliability_floor,
                    reliability_init=reliability_init,
                )
                replaced += 1
        if replaced == 0:
            raise ValueError(
                "Stage-4 backbone exposed no MambaLayer to complexify"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def effective_cyclic_frequencies(self) -> dict[str, torch.Tensor]:
        """Return per-layer complex-pole frequencies in input cycles/sample."""

        return {
            f"encoder.mamba_layers.{index}": layer.ssm.effective_cyclic_frequencies()
            for index, layer in enumerate(self.backbone.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
            and layer.ssm.cyclic_theta_enable
        }

    def scan_backend_status(self) -> dict[str, str]:
        """Expose the actual backend selected by every complex Mamba layer."""

        return {
            f"encoder.mamba_layers.{index}": layer.ssm.last_scan_backend
            for index, layer in enumerate(self.backbone.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
        }

    def diagnostics(self) -> dict[str, str]:
        return {
            f"scan_backend_{index}": backend
            for index, backend in self.scan_backend_status().items()
        }

    def no_weight_decay(self) -> set[str]:
        names = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".a_log", ".theta")) or name.endswith(".D")
        }
        if self.complex_stem_enable:
            from models.IQUMamba1D_ComplexStage4 import (
                ComplexModReLU,
                ComplexRMSNorm1d,
            )

            for module_name, module in self.named_modules():
                if isinstance(module, ComplexRMSNorm1d):
                    names.add(f"{module_name}.log_scale")
                if isinstance(module, ComplexModReLU):
                    names.add(f"{module_name}.bias")
        return names


class IQUMamba1DRealStateTrapReliability(nn.Module):
    """Stage 347: Real-state Mamba with trapezoidal discretization + reliability gating.

    Identical to Stage 333 (RF Mamba-3 Combined) **except** the complex rotation
    poles are removed:

    * ``cyclic_theta_enable=False`` and ``theta`` is initialised to **all-zeros**
      and immediately frozen (``requires_grad=False``).

    With ``theta=0`` the angle term ``dt * theta`` is always zero, so:

        a_real = exp(-dt * alpha) * cos(0) = exp(-dt * alpha)   [pure real decay]
        a_imag = exp(-dt * alpha) * sin(0) = 0                  [no imaginary part]

    The complex prefix scan therefore reduces to an ordinary real-valued scan.
    Trapezoidal input integration and reliability-conditioned ``dt`` scaling are
    still active, giving a clean ablation that isolates those two mechanisms from
    the complex-pole contribution.

    Stage 333 ablation matrix position:
        [trapezoidal=True] + [complex_poles=False] + [reliability=True]
    """

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        trapezoid_lambda_init: float = 0.5,
        reliability_enable: bool = True,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
        **_: object,
    ) -> None:
        super().__init__()
        # Build the model via IQUMamba1DComplexStateMamba with:
        #   - trapezoidal discretization ON
        #   - cyclic_theta_enable=False  (theta initialised to pi*n by default)
        #   - reliability_enable as specified
        # Then freeze theta to zero, removing all oscillatory behaviour.
        self._inner = IQUMamba1DComplexStateMamba(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            mamba_discretization="exponential_trapezoidal",
            trapezoid_lambda_init=trapezoid_lambda_init,
            cyclic_theta_enable=False,
            reliability_enable=reliability_enable,
            reliability_hidden=reliability_hidden,
            reliability_floor=reliability_floor,
            reliability_init=reliability_init,
            complex_stem_enable=False,
        )
        # Freeze theta to zero: removes all oscillatory (imaginary) dynamics.
        # With theta=0 => cos(dt*theta)=1, sin(dt*theta)=0 => pure real decay.
        for layer in self._inner.backbone.encoder.mamba_layers:
            if isinstance(layer, ComplexStateMambaLayer):
                with torch.no_grad():
                    layer.ssm.theta.zero_()
                layer.ssm.theta.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner(x)

    @property
    def backbone(self) -> nn.Module:
        return self._inner.backbone

    @property
    def encoder(self) -> nn.Module:
        return self._inner.backbone.encoder

    @property
    def decoder(self) -> nn.Module:
        return self._inner.backbone.decoder

    def scan_backend_status(self) -> dict[str, str]:
        return self._inner.scan_backend_status()

    def diagnostics(self) -> dict[str, str]:
        return self._inner.diagnostics()

    def no_weight_decay(self) -> set[str]:
        # The wrapper adds an ``_inner.`` prefix to every named parameter.
        return {
            f"_inner.{name}"
            for name in self._inner.no_weight_decay()
            if not name.endswith(".theta")
        }
