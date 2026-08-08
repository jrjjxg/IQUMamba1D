"""Stage-4 memory-range reproductions and role-specific RF experiments.

Paper reproductions:
* Mamba-2/SSD uses the official ``mamba_ssm.Mamba2`` implementation.
* S4D follows the Apache-2.0 standalone implementation from state-spaces/s4,
  ``models/s4/s4d.py`` (Gu, Gupta, Goel, Re, NeurIPS 2022).

The role-RF variants are project innovations. They preserve the Mamba S6
recurrence while changing only the context used to generate delta, B and C.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast

from models.icassp_complex_wavenet import (
    ComplexConv1d,
    ComplexModReLU,
    ComplexRMSNorm1d,
)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    _SELECTIVE_SCAN_IMPORT_ERROR = None
except Exception as exc:  # CPU/slim test environments use the recurrence below.
    selective_scan_fn = None
    _SELECTIVE_SCAN_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


HALF_DTYPES = (torch.float16, torch.bfloat16)


def _use_fused_selective_scan(tensor: torch.Tensor) -> bool:
    """Require the official fused scan on CUDA; allow reference scan on CPU."""
    if not tensor.is_cuda:
        return False
    if selective_scan_fn is None:
        detail = _SELECTIVE_SCAN_IMPORT_ERROR or "unknown import failure"
        raise RuntimeError(
            "CUDA Role-RF execution requires the fused mamba_ssm "
            "selective_scan_fn; refusing to fall back to the sequential "
            f"PyTorch reference scan. Import error: {detail}"
        )
    return True


def _stage4_backbone(
    *,
    input_size,
    input_channels,
    n_stages,
    features_per_stage,
    kernel_sizes,
    strides,
    n_conv_per_stage,
    num_classes,
    n_conv_per_stage_decoder,
    deep_supervision=False,
):
    from models.IQUMamba1D import IQUMamba1D

    return IQUMamba1D(
        input_size=input_size,
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv1d,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=nn.InstanceNorm1d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=bool(deep_supervision),
    )


def _replace_stage4_mamba(backbone: nn.Module, factory) -> int:
    from models.IQUMamba1D import MambaLayer

    replaced = 0
    layers = backbone.encoder.mamba_layers
    for index, layer in enumerate(layers):
        if isinstance(layer, MambaLayer):
            layers[index] = factory(layer.dim, layer.channel_token)
            replaced += 1
    if replaced == 0:
        raise ValueError("Stage-4 backbone exposed no MambaLayer")
    return replaced


def _valid_headdim(d_inner: int, preferred: int) -> int:
    candidates = [int(preferred), 64, 32, 16, 8, 4, 2, 1]
    for value in candidates:
        if value > 0 and value <= d_inner and d_inner % value == 0:
            return value
    return 1


class Mamba2SSDLayer(nn.Module):
    """Drop-in Stage-4 adapter around the official Mamba-2/SSD block."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 32,
        ngroups: int = 1,
        chunk_size: int = 256,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        from mamba_ssm import Mamba2

        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.last_scan_backend = "uninitialized"
        inner = int(expand) * self.dim
        actual_headdim = _valid_headdim(inner, int(headdim))
        self.mamba = Mamba2(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            headdim=actual_headdim,
            ngroups=int(ngroups),
            chunk_size=int(chunk_size),
            use_mem_eff_path=True,
        )

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        tokens = x.reshape(batch, channels, -1).transpose(1, 2)
        output = self.mamba(self.norm(tokens))
        return output.transpose(1, 2).reshape(batch, channels, *dims)

    def _channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        flat = x.flatten(2)
        return self.mamba(self.norm(flat)).reshape(batch, tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_DTYPES:
            x = x.float()
        output = self._channel_tokens(x) if self.channel_token else self._patch_tokens(x)
        self.last_scan_backend = (
            "mamba2_ssd_mem_eff_cuda" if x.is_cuda else "mamba2_official_cpu"
        )
        return output


class S4DKernel(nn.Module):
    """Official standalone S4D-Lin diagonal kernel, adapted without einops."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        if int(d_state) < 2 or int(d_state) % 2:
            raise ValueError("S4D d_state must be an even integer >= 2")
        hidden = int(d_model)
        half_state = int(d_state) // 2
        log_dt = torch.rand(hidden) * (
            math.log(float(dt_max)) - math.log(float(dt_min))
        ) + math.log(float(dt_min))
        coefficients = torch.randn(hidden, half_state, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(coefficients))
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(hidden, half_state))
        )
        self.A_imag = nn.Parameter(
            math.pi
            * torch.arange(half_state, dtype=torch.float32)
            .unsqueeze(0)
            .expand(hidden, -1)
            .clone()
        )
        for parameter in (self.log_dt, self.log_A_real, self.A_imag):
            parameter._optim = {"weight_decay": 0.0}

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        coefficients = torch.view_as_complex(self.C)
        poles = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt_poles = poles * dt.unsqueeze(-1)
        steps = torch.arange(length, device=poles.device, dtype=dt.dtype)
        vandermonde = torch.exp(dt_poles.unsqueeze(-1) * steps)
        discretized_c = coefficients * (torch.exp(dt_poles) - 1.0) / poles
        return 2.0 * torch.einsum("hn,hnl->hl", discretized_c, vandermonde).real


class ChannelDropout1d(nn.Module):
    """Drop entire feature channels, matching the official S4 DropoutNd."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        mask = torch.empty(
            x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype
        ).bernoulli_(keep)
        return x * mask / keep


class S4DLayer(nn.Module):
    """Stage-4 adapter reproducing the standalone S4D FFT convolution layer."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.last_scan_backend = "uninitialized"
        self.D = nn.Parameter(torch.randn(self.dim))
        self.kernel = S4DKernel(
            self.dim, d_state=int(d_state), dt_min=dt_min, dt_max=dt_max
        )
        self.activation = nn.GELU()
        self.dropout = ChannelDropout1d(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.dim, 2 * self.dim, kernel_size=1),
            nn.GLU(dim=1),
        )

    def _sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: B,L,H; official S4D operates on B,H,L.
        u = self.norm(tokens).transpose(1, 2).float()
        length = u.size(-1)
        kernel = self.kernel(length)
        kernel_fft = torch.fft.rfft(kernel, n=2 * length)
        input_fft = torch.fft.rfft(u, n=2 * length)
        y = torch.fft.irfft(input_fft * kernel_fft, n=2 * length)[..., :length]
        y = y + u * self.D.unsqueeze(-1)
        y = self.output_linear(self.dropout(self.activation(y)))
        return y.transpose(1, 2)

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        output = self._sequence(x.reshape(batch, channels, -1).transpose(1, 2))
        return output.transpose(1, 2).reshape(batch, channels, *dims)

    def _channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        return self._sequence(x.flatten(2)).reshape(batch, tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_DTYPES:
            x = x.float()
        output = self._channel_tokens(x) if self.channel_token else self._patch_tokens(x)
        self.last_scan_backend = "torch_fft_cuda" if x.is_cuda else "torch_fft_cpu"
        return output


class StrictComplexS4DKernel(nn.Module):
    r"""ZOH-discretized diagonal S4D kernel over complex feature channels.

    Unlike the real-output S4D kernel, this operator does not construct an
    implicit conjugate copy and does not take ``2 * real(K)``. For each complex
    feature channel and state, it evaluates

    ``K[l] = C * ((exp(A*dt)-1)/A) * exp(A*dt*l)``.

    ``d_state`` counts complex states per complex feature. Because the model
    has half as many complex features as real feature channels, choosing the
    same numeric value as Stage 336 preserves its total real recurrent-state
    budget. Frequencies duplicate Stage 336's S4D-Lin half-spectrum at
    initialization; the two copies are independently trainable afterwards.
    """

    def __init__(
        self,
        complex_channels: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ) -> None:
        super().__init__()
        if int(complex_channels) < 1:
            raise ValueError("complex_channels must be positive")
        if int(d_state) < 2 or int(d_state) % 2:
            raise ValueError("strict-complex S4D d_state must be even and >= 2")
        if not 0.0 < float(dt_min) < float(dt_max):
            raise ValueError("expected 0 < dt_min < dt_max")
        hidden = int(complex_channels)
        state = int(d_state)
        log_dt = torch.rand(hidden) * (
            math.log(float(dt_max)) - math.log(float(dt_min))
        ) + math.log(float(dt_min))
        coefficients = torch.randn(hidden, state, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(coefficients))
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(hidden, state)))
        base_frequencies = math.pi * torch.arange(
            state // 2, dtype=torch.float32
        )
        frequencies = base_frequencies.repeat_interleave(2)
        self.A_imag = nn.Parameter(
            frequencies.unsqueeze(0).expand(hidden, -1).clone()
        )
        for parameter in (self.log_dt, self.log_A_real, self.A_imag):
            parameter._optim = {"weight_decay": 0.0}

    def continuous_parameters(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(A, C, dt)`` in their mathematical complex form."""

        poles = -torch.exp(self.log_A_real) + 1j * self.A_imag
        coefficients = torch.view_as_complex(self.C)
        return poles, coefficients, torch.exp(self.log_dt)

    def forward(self, length: int) -> torch.Tensor:
        if int(length) < 1:
            raise ValueError("kernel length must be positive")
        poles, coefficients, dt = self.continuous_parameters()
        dt_poles = poles * dt.unsqueeze(-1)
        steps = torch.arange(length, device=poles.device, dtype=dt.dtype)
        vandermonde = torch.exp(dt_poles.unsqueeze(-1) * steps)
        input_gain = torch.expm1(dt_poles) / poles
        return torch.einsum(
            "hn,hnl->hl",
            coefficients * input_gain,
            vandermonde,
        )


class ComplexPairDropout1d(nn.Module):
    """Drop one complex feature as a pair, preserving phase equivariance."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= float(probability) < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        real, imag = torch.chunk(x, 2, dim=1)
        keep = 1.0 - self.probability
        mask = torch.empty(
            real.size(0), real.size(1), 1, device=x.device, dtype=x.dtype
        ).bernoulli_(keep)
        return torch.cat((real * mask / keep, imag * mask / keep), dim=1)


class StrictComplexS4DLayer(nn.Module):
    """Strict complex-linear S4D memory with phase-equivariant wrappers."""

    def __init__(
        self,
        real_channels: int,
        *,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(real_channels) < 2 or int(real_channels) % 2:
            raise ValueError("strict-complex S4D requires an even real width")
        self.dim = int(real_channels)
        self.complex_channels = self.dim // 2
        self.channel_token = False
        self.last_scan_backend = "uninitialized"
        self.norm = ComplexRMSNorm1d(self.complex_channels, eps=norm_eps)
        self.kernel = StrictComplexS4DKernel(
            self.complex_channels,
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        direct = torch.randn(self.complex_channels, dtype=torch.cfloat)
        self.D = nn.Parameter(torch.view_as_real(direct))
        self.activation = ComplexModReLU(self.complex_channels)
        self.dropout = (
            ComplexPairDropout1d(dropout) if dropout > 0 else nn.Identity()
        )
        self.output_linear = ComplexConv1d(
            self.complex_channels,
            self.complex_channels,
            kernel_size=1,
        )

    @staticmethod
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        real, imag = torch.chunk(x, 2, dim=1)
        return torch.complex(real.float(), imag.float())

    @staticmethod
    def _to_real_layout(x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.real, x.imag), dim=1)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.dim:
            raise ValueError(
                "strict-complex S4D expects [B, 2*C, L], got "
                f"{tuple(x.shape)}"
            )
        if x.dtype in HALF_DTYPES:
            x = x.float()
        normalized = self._to_complex(self.norm(x))
        length = normalized.size(-1)
        kernel = self.kernel(length)
        kernel_fft = torch.fft.fft(kernel, n=2 * length)
        input_fft = torch.fft.fft(normalized, n=2 * length)
        y = torch.fft.ifft(input_fft * kernel_fft, n=2 * length)[..., :length]
        direct = torch.view_as_complex(self.D).unsqueeze(-1)
        y = y + normalized * direct
        output = self.output_linear(
            self.dropout(self.activation(self._to_real_layout(y)))
        )
        self.last_scan_backend = (
            "torch_complex_fft_cuda" if x.is_cuda else "torch_complex_fft_cpu"
        )
        return output


class ReliabilitySelectiveS4DLayer(nn.Module):
    """S4D-Lin poles with Stage-333-style reliability-controlled state time.

    A token-dependent time step makes the recurrence time varying, so this
    layer cannot use Stage 336's single FFT kernel.  CUDA execution therefore
    uses the official complex ``selective_scan_fn``.  Its input discretization
    is exponential Euler (the fused scan convention), while the diagonal
    poles, readout, skip and output head retain the S4D-Lin parameterization.
    """

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        dropout: float = 0.0,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.last_scan_backend = "uninitialized"
        self.last_reliability = None
        self.last_delta = None
        self.D = nn.Parameter(torch.randn(self.dim))
        self.kernel = S4DKernel(
            self.dim, d_state=int(d_state), dt_min=dt_min, dt_max=dt_max
        )
        self.activation = nn.GELU()
        self.dropout = ChannelDropout1d(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.dim, 2 * self.dim, kernel_size=1),
            nn.GLU(dim=1),
        )

        if not 0.0 <= float(reliability_floor) < 1.0:
            raise ValueError("reliability_floor must be in [0, 1)")
        if not float(reliability_floor) < float(reliability_init) < 1.0:
            raise ValueError(
                "reliability_init must be between reliability_floor and 1"
            )
        self.reliability_floor = float(reliability_floor)
        self.reliability_init = float(reliability_init)
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

    def _reliability(self, tokens: torch.Tensor) -> torch.Tensor:
        previous = F.pad(tokens[:, :-1], (0, 0, 1, 0))
        power = tokens.square().mean(dim=-1).clamp_min(1e-8)
        previous_power = previous.square().mean(dim=-1).clamp_min(1e-8)
        flux = (tokens - previous).square().mean(dim=-1).clamp_min(1e-8)
        correlation = (tokens * previous).mean(dim=-1) / torch.sqrt(
            power * previous_power
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

    def _reference_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        poles: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        state = poles.new_zeros(u.size(0), self.dim, poles.size(1))
        outputs = []
        for index in range(u.size(-1)):
            step = delta[..., index]
            state = torch.exp(step.unsqueeze(-1) * poles.unsqueeze(0)) * state
            state = state + step.unsqueeze(-1) * u[..., index].unsqueeze(-1)
            output = 2.0 * torch.einsum(
                "bdn,dn->bd", state, coefficients
            ).real
            outputs.append(output)
        return torch.stack(outputs, dim=-1) + u * self.D.float().view(1, -1, 1)

    def _scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        poles: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        if _use_fused_selective_scan(u):
            output = selective_scan_fn(
                u.contiguous(),
                delta.contiguous(),
                poles.contiguous(),
                torch.ones_like(poles),
                coefficients.contiguous(),
                self.D.float(),
                delta_softplus=False,
            )
            self.last_scan_backend = "mamba_complex_selective_scan_cuda"
            return output
        self.last_scan_backend = "torch_reference"
        return self._reference_scan(u, delta, poles, coefficients)

    def _sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens).float()
        reliability = self._reliability(normalized)
        # Center the gate at one so Stage 348 starts from Stage 336's time
        # scale instead of silently shortening every state step at init.
        time_scale = reliability / self.reliability_init
        delta = (
            self.kernel.log_dt.exp().view(1, 1, self.dim) * time_scale
        )
        self.last_reliability = reliability.detach()
        self.last_delta = delta.detach()

        poles = torch.complex(
            -self.kernel.log_A_real.exp(), self.kernel.A_imag
        )
        base_dt = self.kernel.log_dt.exp().unsqueeze(-1)
        dt_poles = base_dt * poles
        # selective_scan injects dt*u (exponential Euler), whereas S4D-Lin
        # uses exact ZOH input integration.  Calibrating C by phi_1(dt*A)
        # makes the fused recurrence exactly match Stage 336 when the centered
        # reliability time scale is one, without materializing B*D*N*L.
        zoh_correction = torch.expm1(dt_poles) / dt_poles
        coefficients = torch.view_as_complex(self.kernel.C) * zoh_correction
        output = self._scan(
            normalized.transpose(1, 2).contiguous(),
            delta.transpose(1, 2).contiguous(),
            poles,
            coefficients,
        )
        output = self.output_linear(self.dropout(self.activation(output)))
        return output.transpose(1, 2)

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        output = self._sequence(x.reshape(batch, channels, -1).transpose(1, 2))
        return output.transpose(1, 2).reshape(batch, channels, *dims)

    def _channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        return self._sequence(x.flatten(2)).reshape(batch, tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_DTYPES:
            x = x.float()
        return self._channel_tokens(x) if self.channel_token else self._patch_tokens(x)


class RoleRFSelectiveSSM(nn.Module):
    """Mamba S6 with shared, fixed-role, or routed multi-scale contexts."""

    VARIANTS = {"shared", "fixed_role", "routed_role"}

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        expand: int = 2,
        context_kernels: Sequence[int] = (3, 15, 63),
        variant: str = "shared",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(expand) * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16)
        self.variant = str(variant)
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}")
        kernels = tuple(int(value) for value in context_kernels)
        if len(kernels) < 2 or any(value < 1 or value % 2 == 0 for value in kernels):
            raise ValueError("context_kernels must contain at least two positive odd values")
        self.context_kernels = kernels

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.context_convs = nn.ModuleList(
            nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=kernel,
                padding=kernel - 1,
                groups=self.d_inner,
                bias=True,
            )
            for kernel in kernels
        )
        self.dt_in_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        self.b_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.c_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        state_index = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(state_index).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3))
            + math.log(1e-3)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        nn.init.uniform_(
            self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5
        )

        router_count = 1 if self.variant == "shared" else 3
        self.routers = nn.ModuleList(
            nn.Conv1d(self.d_inner, len(kernels), kernel_size=1)
            for _ in range(router_count)
        ) if self.variant != "fixed_role" else nn.ModuleList()
        if self.variant == "routed_role":
            preferences = (0, len(kernels) // 2, len(kernels) - 1)
            for router, preferred in zip(self.routers, preferences):
                nn.init.zeros_(router.weight)
                nn.init.constant_(router.bias, -1.0)
                with torch.no_grad():
                    router.bias[preferred] = 2.0
        self.last_role_weights: dict[str, torch.Tensor] = {}
        self.last_scan_backend = "uninitialized"

    def _contexts(self, x: torch.Tensor) -> list[torch.Tensor]:
        length = x.size(1)
        channels_first = x.transpose(1, 2)
        return [
            F.silu(conv(channels_first)[..., :length].transpose(1, 2))
            for conv in self.context_convs
        ]

    @staticmethod
    def _mix(contexts: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(contexts, dim=2)  # B,L,S,D
        return (stacked * weights.unsqueeze(-1)).sum(dim=2)

    def _role_contexts(
        self, contexts: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        names = ("B", "C", "delta")
        if self.variant == "fixed_role":
            indices = (0, len(contexts) // 2, len(contexts) - 1)
            self.last_role_weights = {
                name: F.one_hot(
                    torch.tensor(index, device=contexts[0].device),
                    num_classes=len(contexts),
                ).float()
                for name, index in zip(names, indices)
            }
            return tuple(contexts[index] for index in indices)

        evidence = sum(contexts) / float(len(contexts))
        router_outputs = self.routers if self.variant == "routed_role" else (
            self.routers[0], self.routers[0], self.routers[0]
        )
        mixed = []
        diagnostics = {}
        for name, router in zip(names, router_outputs):
            weights = torch.softmax(router(evidence.transpose(1, 2)), dim=1)
            weights = weights.transpose(1, 2)
            mixed.append(self._mix(contexts, weights))
            diagnostics[name] = weights.detach()
        self.last_role_weights = diagnostics
        return tuple(mixed)

    def _reference_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())
        state = x.new_zeros(x.size(0), self.d_inner, self.d_state)
        outputs = []
        for index in range(x.size(1)):
            step = dt[:, index]
            transition = torch.exp(step.unsqueeze(-1) * A.unsqueeze(0))
            state_input = (
                step.unsqueeze(-1)
                * x[:, index].unsqueeze(-1)
                * b[:, index].unsqueeze(1)
            )
            state = transition * state + state_input
            outputs.append(
                (state * c[:, index].unsqueeze(1)).sum(dim=-1)
                + self.D * x[:, index]
            )
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        x_in, gate = torch.chunk(self.in_proj(x), 2, dim=-1)
        contexts = self._contexts(x_in)
        b_context, c_context, dt_context = self._role_contexts(contexts)
        b = self.b_proj(b_context)
        c = self.c_proj(c_context)
        dt = F.softplus(self.dt_proj(self.dt_in_proj(dt_context)))
        scan_input = contexts[0]

        if _use_fused_selective_scan(scan_input):
            y = selective_scan_fn(
                scan_input.transpose(1, 2).contiguous(),
                dt.transpose(1, 2).contiguous(),
                -torch.exp(self.A_log.float()),
                b.transpose(1, 2).contiguous(),
                c.transpose(1, 2).contiguous(),
                self.D.float(),
                z=gate.transpose(1, 2).contiguous(),
                delta_softplus=False,
            ).transpose(1, 2)
            self.last_scan_backend = "mamba_fused_cuda_selective_scan"
        else:
            y = self._reference_scan(scan_input, dt, b, c) * F.silu(gate)
            self.last_scan_backend = "torch_reference"
        if y.size(1) != length:
            raise RuntimeError("role-RF scan changed sequence length")
        return self.out_proj(y)


class RoleRFMambaLayer(nn.Module):
    def __init__(self, dim: int, *, channel_token: bool = False, **ssm_kwargs) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.ssm = RoleRFSelectiveSSM(self.dim, **ssm_kwargs)

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        tokens = x.reshape(batch, channels, -1).transpose(1, 2)
        output = self.ssm(self.norm(tokens))
        return output.transpose(1, 2).reshape(batch, channels, *dims)

    def _channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        return self.ssm(self.norm(x.flatten(2))).reshape(batch, tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_DTYPES:
            x = x.float()
        return self._channel_tokens(x) if self.channel_token else self._patch_tokens(x)


class _Stage4Replacement(nn.Module):
    def __init__(self, backbone: nn.Module, factory) -> None:
        super().__init__()
        self.backbone = backbone
        self.replaced_layers = _replace_stage4_mamba(self.backbone, factory)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


class IQUMamba1DMamba2SSD(_Stage4Replacement):
    def __init__(self, *, d_state=64, d_conv=4, expand=2, headdim=32,
                 ngroups=1, chunk_size=256, **backbone_kwargs) -> None:
        backbone = _stage4_backbone(**backbone_kwargs)
        super().__init__(
            backbone,
            lambda dim, channel: Mamba2SSDLayer(
                dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                chunk_size=chunk_size,
                channel_token=channel,
            ),
        )


class IQUMamba1DS4D(_Stage4Replacement):
    def __init__(self, *, d_state=64, dropout=0.0, dt_min=1e-3, dt_max=1e-1,
                 complex_stem_enable=False, complex_norm_eps=1e-6,
                 **backbone_kwargs) -> None:
        backbone = _stage4_backbone(**backbone_kwargs)
        uses_complex_stem = bool(complex_stem_enable)
        if uses_complex_stem:
            if int(backbone_kwargs["input_channels"]) != 2:
                raise ValueError("The strict-complex stem expects one I/Q mixture")
            from models.IQUMamba1D_ComplexStage4 import ComplexStem1d

            backbone.encoder.stem = ComplexStem1d(
                int(backbone_kwargs["features_per_stage"][0]),
                blocks=int(backbone_kwargs["n_conv_per_stage"][0]),
                kernel_size=int(backbone_kwargs["kernel_sizes"][0]),
                norm_eps=float(complex_norm_eps),
            )
        super().__init__(
            backbone,
            lambda dim, channel: S4DLayer(
                dim,
                d_state=d_state,
                dropout=dropout,
                dt_min=dt_min,
                dt_max=dt_max,
                channel_token=channel,
            ),
        )
        self.uses_complex_stem = uses_complex_stem

    def no_weight_decay(self) -> set[str]:
        names = {
            name
            for name, parameter in self.named_parameters()
            if bool(getattr(parameter, "_no_weight_decay", False))
            or bool(getattr(parameter, "_optim", {}).get("weight_decay") == 0.0)
        }
        if self.uses_complex_stem:
            from models.IQUMamba1D_ComplexStage4 import (
                ComplexModReLU,
                ComplexRMSNorm1d,
            )

            for module_name, module in self.named_modules():
                if isinstance(module, ComplexRMSNorm1d):
                    names.add(f"{module_name}.log_scale")
                elif isinstance(module, ComplexModReLU):
                    names.add(f"{module_name}.bias")
        return names


class IQUMamba1DStrictComplexS4D(nn.Module):
    """Stage 357: fully strict-complex Stage-4 U-Net with complex S4D memory."""

    def __init__(
        self,
        *,
        input_size,
        input_channels,
        n_stages,
        features_per_stage,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes,
        n_conv_per_stage_decoder,
        deep_supervision=False,
        d_state=64,
        dropout=0.0,
        dt_min=1e-3,
        dt_max=1e-1,
        complex_norm_eps=1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("strict-complex Stage 357 expects one I/Q mixture")
        if int(num_classes) % 2:
            raise ValueError("strict-complex output requires complete I/Q pairs")
        if bool(deep_supervision):
            raise ValueError("Stage 357 exposes one strict-complex output only")
        if int(n_stages) != len(features_per_stage):
            raise ValueError("n_stages does not match features_per_stage")
        if any(int(width) % 2 for width in features_per_stage):
            raise ValueError("all Stage 357 feature widths must be even")

        from models.IQUMamba1D_ComplexStage4 import (
            ComplexDecoder1d,
            ComplexEncoder1d,
        )

        self.input_size = int(input_size)
        self.uses_complex_stem = True
        self.strict_complex_output = True
        self.encoder = ComplexEncoder1d(
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            blocks_per_stage=n_conv_per_stage,
            use_equivariant_mamba=False,
            norm_eps=float(complex_norm_eps),
            mamba_d_state=1,
            mamba_d_conv=1,
            mamba_expand=1,
            mamba_max_gain_delta=0.0,
            memory_factory=lambda width, _stage: StrictComplexS4DLayer(
                width,
                d_state=int(d_state),
                dropout=float(dropout),
                dt_min=float(dt_min),
                dt_max=float(dt_max),
                norm_eps=float(complex_norm_eps),
            ),
        )
        self.decoder = ComplexDecoder1d(
            self.encoder,
            num_classes=int(num_classes),
            blocks_per_stage=n_conv_per_stage_decoder,
            strict_complex_output=True,
            norm_eps=float(complex_norm_eps),
        )
        self.replaced_layers = sum(
            isinstance(layer, StrictComplexS4DLayer)
            for layer in self.encoder.mamba_layers
        )
        if self.replaced_layers == 0:
            raise ValueError("Stage 357 constructed no strict-complex S4D layers")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def no_weight_decay(self) -> set[str]:
        names = {
            name
            for name, parameter in self.named_parameters()
            if bool(getattr(parameter, "_optim", {}).get("weight_decay") == 0.0)
        }
        for module_name, module in self.named_modules():
            if isinstance(module, ComplexRMSNorm1d):
                names.add(f"{module_name}.log_scale")
            elif isinstance(module, ComplexModReLU):
                names.add(f"{module_name}.bias")
        return names


class IQUMamba1DS4DUniRepLK(IQUMamba1DS4D):
    """Stage 353: strict-complex Stage-290 stem + Stage-336 S4D + UniRepLK."""

    def __init__(
        self,
        *,
        rf_residual_scale_init=0.05,
        unireplk_large_kernel=17,
        unireplk_ffn_factor=4,
        unireplk_layer_scale=1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from models.IQUMamba1D_RecentRFModules import (
            FeatureResidualAdapter,
            build_recent_rf_operator,
        )

        channels = self.backbone.encoder.output_channels
        if len(channels) < 3:
            raise ValueError("S4D + UniRepLK requires at least three encoder stages")
        operator_config = {
            "rf_large_kernel": int(unireplk_large_kernel),
            "rf_ffn_factor": int(unireplk_ffn_factor),
            "rf_layer_scale": float(unireplk_layer_scale),
        }
        self.stage_rf = nn.ModuleDict({
            str(stage): FeatureResidualAdapter(
                int(channels[stage]),
                build_recent_rf_operator(
                    "unireplk", int(channels[stage]), operator_config
                ),
                float(rf_residual_scale_init),
            )
            for stage in (0, 1, 2)
        })

    @property
    def encoder(self) -> nn.Module:
        return self.backbone.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.backbone.decoder

    def forward(self, x: torch.Tensor):
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage, (conv_stage, memory) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            x = memory(conv_stage(x))
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](x)
            skips.append(x)
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        return super().no_weight_decay() | {
            f"stage_rf.{stage}.residual_scale" for stage in self.stage_rf
        }


class IQUMamba1DReliabilityS4D(_Stage4Replacement):
    """Stage 348: S4D-Lin complex poles plus reliability-controlled time."""

    def __init__(self, *, d_state=64, dropout=0.0, dt_min=1e-3, dt_max=1e-1,
                 reliability_hidden=8, reliability_floor=0.05,
                 reliability_init=0.995, **backbone_kwargs) -> None:
        backbone = _stage4_backbone(**backbone_kwargs)
        super().__init__(
            backbone,
            lambda dim, channel: ReliabilitySelectiveS4DLayer(
                dim,
                d_state=d_state,
                dropout=dropout,
                dt_min=dt_min,
                dt_max=dt_max,
                reliability_hidden=reliability_hidden,
                reliability_floor=reliability_floor,
                reliability_init=reliability_init,
                channel_token=channel,
            ),
        )


class IQUMamba1DRoleRF(_Stage4Replacement):
    def __init__(self, *, variant="shared", d_state=16, expand=2,
                 context_kernels=(3, 15, 63), **backbone_kwargs) -> None:
        backbone = _stage4_backbone(**backbone_kwargs)
        super().__init__(
            backbone,
            lambda dim, channel: RoleRFMambaLayer(
                dim,
                channel_token=channel,
                variant=variant,
                d_state=d_state,
                expand=expand,
                context_kernels=context_kernels,
            ),
        )
