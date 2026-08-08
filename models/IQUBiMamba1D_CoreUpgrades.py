"""Strict Stage-12 BiMamba-core upgrades and their RF combination.

Stages 361--364 keep the four-stage Stage-12 U-Net, decoder, skip paths and
BiMamba placement (encoder stages 1 and 3).  Only the token mixer installed at
those two positions changes:

* Hydra: the official Hydra data-dependent quasiseparable bidirectional SSM
  topology, with a pure-PyTorch reference scan when the fused SSD kernel is
  unavailable.
* ComplexState: one shared complex selective SSM evaluated with conjugate pole
  rotations in the two directions, followed by adaptive token/channel fusion.
* MultiScaleBiMamba: the original pair of global Mamba scans plus non-causal
  local depthwise branches with widths 3, 7 and 15.
* IndependentComplexState: two fully independent Stage-295 complex SSMs with
  the same adaptive fusion used by the shared-conjugate ablation.
* IndependentComplexStateUniRepLK: the Stage-364 core plus parallel UniRepLK
  residual deltas sourced from encoder stages 0/1/2.

The Hydra equations and projection layout follow the authors' MIT-licensed
reference implementation:
https://github.com/goombalab/hydra/blob/main/hydra/modules/hydra.py
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from mamba_ssm import Mamba

from models.IQUMamba1D import UNetResDecoder
from models.IQUMamba1D_ComplexStateMamba import ComplexStateSelectiveSSM
from models.IQUMamba1D_RecentRFModules import (
    ParallelFeatureDeltaAdapter,
    build_recent_rf_operator,
)
from models.IQUBiMamba1D_SafeAllStages import (
    HALF_PRECISION_DTYPES,
    ResidualSafeAllStageBiMambaEncoder,
)

try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_chunk_scan_combined,
    )
except Exception:  # Mamba-2/Triton is optional; CPU/reference scan stays valid.
    _mamba_chunk_scan_combined = None


def _inverse_tanh(value: float) -> float:
    value = min(max(float(value), -0.999), 0.999)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


class _TokenLayerBase(nn.Module):
    """Patch/channel token reshaping shared by all three drop-in layers."""

    def __init__(self, dim: int, channel_token: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        tokens = int(np.prod(dims))
        flat = x.reshape(batch, channels, tokens).transpose(-1, -2)
        return self._mix(flat).transpose(-1, -2).reshape(batch, channels, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        flat = x.flatten(2)
        return self._mix(flat).reshape(batch, tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class HydraQuasiseparableMixer(nn.Module):
    """Hydra's data-dependent bidirectional quasiseparable matrix mixer.

    The fused Mamba-2 SSD scan is used on CUDA when available.  The reference
    path evaluates the same diagonal SSD recurrence in ordinary PyTorch, which
    makes the stage importable and testable without adding a hard Hydra/Triton
    dependency.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        prefer_fused_scan: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.headdim = int(headdim)
        self.ngroups = int(ngroups)
        self.chunk_size = int(chunk_size)
        self.prefer_fused_scan = bool(prefer_fused_scan)
        self.last_scan_backend = "uninitialized"

        if self.d_conv < 1 or self.d_conv % 2 == 0:
            raise ValueError("Hydra d_conv must be a positive odd integer")
        if self.headdim < 1 or self.d_inner % self.headdim != 0:
            raise ValueError(
                f"Hydra d_inner={self.d_inner} must be divisible by headdim={self.headdim}"
            )
        self.nheads = self.d_inner // self.headdim
        if self.ngroups < 1 or self.nheads % self.ngroups != 0:
            raise ValueError("Hydra nheads must be divisible by ngroups")

        # Official order: [z, x, B_fw, C_fw, B_bw, C_bw, dt_fw, dt_bw].
        projected_dim = (
            2 * self.d_inner
            + 4 * self.ngroups * self.d_state
            + 2 * self.nheads
        )
        self.in_proj = nn.Linear(self.d_model, projected_dim, bias=bias)
        conv_dim = self.d_inner + 4 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=self.d_conv,
            padding=self.d_conv // 2,
            groups=conv_dim,
            bias=conv_bias,
        )

        dt = torch.exp(
            torch.rand(self.nheads)
            * (math.log(float(dt_max)) - math.log(float(dt_min)))
            + math.log(float(dt_min))
        ).clamp_min(float(dt_init_floor))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.zeros(self.nheads, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.fc_D = nn.Linear(self.d_inner, self.nheads, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
        self.norm_eps = 1e-5
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        for parameter in (self.dt_bias, self.A_log, self.D):
            parameter._no_weight_decay = True

    def _reference_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Inclusive diagonal SSD scan used when the fused kernel is absent."""

        batch, length, nheads, headdim = x.shape
        heads_per_group = nheads // self.ngroups
        head_to_group = torch.arange(nheads, device=x.device) // heads_per_group
        B_heads = B.index_select(2, head_to_group)
        C_heads = C.index_select(2, head_to_group)
        state = x.new_zeros(batch, nheads, headdim, self.d_state)
        A = A.to(dtype=x.dtype, device=x.device).view(1, nheads, 1, 1)

        def scan_chunk(
            initial_state: torch.Tensor,
            x_chunk: torch.Tensor,
            dt_chunk: torch.Tensor,
            B_chunk: torch.Tensor,
            C_chunk: torch.Tensor,
            transition_rate: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            current = initial_state
            chunk_outputs = []
            for local_index in range(x_chunk.shape[1]):
                dt_t = (
                    dt_chunk[:, local_index]
                    .to(dtype=x_chunk.dtype)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                transition = torch.exp(dt_t * transition_rate)
                drive = (
                    dt_t
                    * x_chunk[:, local_index].unsqueeze(-1)
                    * B_chunk[:, local_index].unsqueeze(-2)
                )
                current = transition * current + drive
                chunk_outputs.append(
                    (current * C_chunk[:, local_index].unsqueeze(-2)).sum(dim=-1)
                )
            return torch.stack(chunk_outputs, dim=1), current

        outputs = []
        reference_chunk = max(1, min(self.chunk_size, length))
        needs_checkpoint = torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (x, dt, A, B_heads, C_heads)
        )
        for start in range(0, length, reference_chunk):
            stop = min(start + reference_chunk, length)
            arguments = (
                state,
                x[:, start:stop],
                dt[:, start:stop],
                B_heads[:, start:stop],
                C_heads[:, start:stop],
                A,
            )
            if needs_checkpoint:
                chunk_output, state = checkpoint(
                    scan_chunk, *arguments, use_reentrant=False
                )
            else:
                chunk_output, state = scan_chunk(*arguments)
            outputs.append(chunk_output)
        return torch.cat(outputs, dim=1)

    def _scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        if self.prefer_fused_scan and x.is_cuda and _mamba_chunk_scan_combined is not None:
            try:
                output = _mamba_chunk_scan_combined(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    chunk_size=self.chunk_size,
                    D=None,
                    z=None,
                )
                self.last_scan_backend = "mamba_ssd_cuda"
                return output
            except Exception:
                # ABI/version mismatches must not make the stage unusable.
                pass
        self.last_scan_backend = "torch_reference"
        return self._reference_scan(x=x, dt=dt, A=A, B=B, C=C)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        if u.ndim != 3 or u.shape[-1] != self.d_model:
            raise ValueError(f"Hydra expects [B,L,{self.d_model}], got {tuple(u.shape)}")
        batch, length, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xbc, dt = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 4 * self.ngroups * self.d_state,
                2 * self.nheads,
            ],
            dim=-1,
        )
        xbc = F.silu(self.conv1d(xbc.transpose(1, 2)).transpose(1, 2))
        x_original, bc = torch.split(
            xbc,
            [self.d_inner, 4 * self.ngroups * self.d_state],
            dim=-1,
        )

        # Batch-concatenate the lower and upper quasiseparable halves, exactly
        # as the official Hydra implementation does before its single SSD scan.
        x_scan = torch.cat((x_original, x_original.flip(1)), dim=0)
        bc_forward, bc_backward = torch.split(
            bc, 2 * self.ngroups * self.d_state, dim=-1
        )
        bc_scan = torch.cat((bc_forward, bc_backward.flip(1)), dim=0)
        B, C = torch.split(bc_scan, self.ngroups * self.d_state, dim=-1)
        dt_forward, dt_backward = torch.split(dt, self.nheads, dim=-1)
        dt_scan = torch.cat((dt_forward, dt_backward.flip(1)), dim=0)
        dt_scan = F.softplus(dt_scan + self.dt_bias)

        x_heads = x_scan.reshape(
            2 * batch, length, self.nheads, self.headdim
        )
        B = B.reshape(2 * batch, length, self.ngroups, self.d_state)
        C = C.reshape(2 * batch, length, self.ngroups, self.d_state)
        A = -torch.exp(self.A_log.float())
        y = self._scan(x=x_heads, dt=dt_scan, A=A, B=B, C=C)
        y = y.reshape(2 * batch, length, self.d_inner)

        # The two scans represent strictly lower/upper triangular terms; the
        # data-dependent diagonal term is added separately.
        y = torch.roll(y, shifts=1, dims=1)
        y[:, 0] = 0.0
        y_forward = y[:batch]
        y_backward = y[batch:].flip(1)
        diagonal = F.linear(x_original, self.fc_D.weight, bias=self.D)
        diagonal = diagonal.repeat_interleave(self.headdim, dim=-1)
        y = y_forward + y_backward + x_original * diagonal

        # Pure-PyTorch equivalent of Hydra's RMSNormGated(norm_before_gate=True).
        variance = y.float().square().mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(variance + self.norm_eps).to(dtype=y.dtype)
        y = y * self.norm_weight * F.silu(z)
        return self.out_proj(y)


class HydraBiMambaLayer(_TokenLayerBase):
    """Stage-12 residual wrapper around the formal Hydra mixer."""

    def __init__(
        self,
        dim: int,
        *,
        channel_token: bool = False,
        d_state: int = 64,
        d_conv: int = 7,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        residual_scale_init: float = 1.0,
        prefer_fused_scan: bool = True,
    ) -> None:
        super().__init__(dim=dim, channel_token=channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.hydra = HydraQuasiseparableMixer(
            self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            prefer_fused_scan=prefer_fused_scan,
        )
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.hydra(self.norm(x))


class ComplexStateBiMambaLayer(_TokenLayerBase):
    """Shared complex SSM, conjugate reverse dynamics and adaptive fusion."""

    def __init__(
        self,
        dim: int,
        *,
        channel_token: bool = False,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        fusion_hidden: int = 64,
        residual_scale_init: float = 1.0,
        **ssm_kwargs,
    ) -> None:
        super().__init__(dim=dim, channel_token=channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.ssm = ComplexStateSelectiveSSM(
            self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            **ssm_kwargs,
        )
        hidden = max(4, int(fusion_hidden))
        self.direction_router = nn.Sequential(
            nn.Linear(3 * self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )
        # Begin as an unbiased symmetric fusion, then learn token/channel
        # direction preferences from RF evidence.
        nn.init.zeros_(self.direction_router[-1].weight)
        nn.init.zeros_(self.direction_router[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_direction_gate: torch.Tensor | None = None
        self.last_scan_backends: Tuple[str, str] = (
            "uninitialized",
            "uninitialized",
        )
        self.conjugate_directions = True

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        forward = self.ssm(x_norm, theta_sign=1.0)
        forward_backend = self.ssm.last_scan_backend
        backward = self.ssm(x_norm.flip(1), theta_sign=-1.0).flip(1)
        backward_backend = self.ssm.last_scan_backend
        symmetric = 0.5 * (forward + backward)
        disagreement = (forward - backward).abs()
        gate = torch.sigmoid(
            self.direction_router(torch.cat((x_norm, symmetric, disagreement), dim=-1))
        )
        self.last_direction_gate = gate.detach()
        self.last_scan_backends = (forward_backend, backward_backend)
        fused = gate * forward + (1.0 - gate) * backward
        return x + self.res_scale * fused


class IndependentComplexStateBiMambaLayer(_TokenLayerBase):
    """Two fully independent Stage-295 complex SSMs with adaptive fusion.

    Unlike :class:`ComplexStateBiMambaLayer`, no projection, decay, rotation
    frequency or readout parameter is shared or conjugate-tied between the
    two directions.  Keeping the same router makes Stage 362 vs Stage 364 a
    controlled comparison of shared-conjugate and independent complex states.
    """

    def __init__(
        self,
        dim: int,
        *,
        channel_token: bool = False,
        d_state: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        fusion_hidden: int = 64,
        residual_scale_init: float = 1.0,
        **ssm_kwargs,
    ) -> None:
        super().__init__(dim=dim, channel_token=channel_token)
        self.norm = nn.LayerNorm(self.dim)
        ssm_config = dict(
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            **ssm_kwargs,
        )
        self.ssm_fwd = ComplexStateSelectiveSSM(self.dim, **ssm_config)
        self.ssm_bwd = ComplexStateSelectiveSSM(self.dim, **ssm_config)
        hidden = max(4, int(fusion_hidden))
        self.direction_router = nn.Sequential(
            nn.Linear(3 * self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
        )
        nn.init.zeros_(self.direction_router[-1].weight)
        nn.init.zeros_(self.direction_router[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_direction_gate: torch.Tensor | None = None
        self.last_scan_backends: Tuple[str, str] = (
            "uninitialized",
            "uninitialized",
        )
        self.conjugate_directions = False
        self.independent_directions = True

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        forward = self.ssm_fwd(x_norm)
        forward_backend = self.ssm_fwd.last_scan_backend
        backward = self.ssm_bwd(x_norm.flip(1)).flip(1)
        backward_backend = self.ssm_bwd.last_scan_backend
        symmetric = 0.5 * (forward + backward)
        disagreement = (forward - backward).abs()
        gate = torch.sigmoid(
            self.direction_router(torch.cat((x_norm, symmetric, disagreement), dim=-1))
        )
        self.last_direction_gate = gate.detach()
        self.last_scan_backends = (forward_backend, backward_backend)
        fused = gate * forward + (1.0 - gate) * backward
        return x + self.res_scale * fused


class MultiScaleBiMambaLayer(_TokenLayerBase):
    """Original global BiMamba plus routed d_conv=3/7/15 local context."""

    def __init__(
        self,
        dim: int,
        *,
        channel_token: bool = False,
        d_state: int = 16,
        global_d_conv: int = 4,
        expand: int = 2,
        local_kernel_sizes: Sequence[int] = (3, 7, 15),
        local_scale_init: float = 0.1,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__(dim=dim, channel_token=channel_token)
        kernels = tuple(int(kernel) for kernel in local_kernel_sizes)
        if kernels != (3, 7, 15):
            raise ValueError(
                "Stage12-MultiScaleBiMamba requires local d_conv widths [3, 7, 15]"
            )
        self.local_kernel_sizes = kernels
        self.norm = nn.LayerNorm(self.dim)
        self.mamba_fwd = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(global_d_conv),
            expand=int(expand),
        )
        self.mamba_bwd = Mamba(
            d_model=self.dim,
            d_state=int(d_state),
            d_conv=int(global_d_conv),
            expand=int(expand),
        )
        self.global_out_proj = nn.Linear(2 * self.dim, self.dim)
        self.local_convs = nn.ModuleList(
            nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=kernel,
                padding=kernel // 2,
                groups=self.dim,
                bias=True,
            )
            for kernel in kernels
        )
        self.local_projections = nn.ModuleList(
            nn.Linear(self.dim, self.dim, bias=False) for _ in kernels
        )
        self.local_router = nn.Linear(self.dim, len(kernels))
        self.local_gate = nn.Linear(2 * self.dim, self.dim)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(self.local_gate.bias, _inverse_sigmoid(0.5))
        self.local_scale_raw = nn.Parameter(
            torch.tensor(_inverse_tanh(local_scale_init))
        )
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_local_weights: torch.Tensor | None = None

    def _mix(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        forward = self.mamba_fwd(x_norm)
        backward = self.mamba_bwd(x_norm.flip(1)).flip(1)
        global_context = self.global_out_proj(torch.cat((forward, backward), dim=-1))

        channels_first = x_norm.transpose(1, 2)
        local_contexts = [
            projection(F.silu(conv(channels_first)).transpose(1, 2))
            for conv, projection in zip(self.local_convs, self.local_projections)
        ]
        local_weights = torch.softmax(self.local_router(x_norm), dim=-1)
        self.last_local_weights = local_weights.detach()
        local_context = sum(
            local_weights[..., index : index + 1] * context
            for index, context in enumerate(local_contexts)
        )
        local_gate = torch.sigmoid(
            self.local_gate(torch.cat((global_context, local_context), dim=-1))
        )
        delta = global_context + torch.tanh(self.local_scale_raw) * local_gate * local_context
        return x + self.res_scale * delta


class _CoreUpgradeEncoder(ResidualSafeAllStageBiMambaEncoder):
    """Build unchanged Stage-12 geometry, then swap only active token mixers."""

    layer_kind = ""

    def _active_indices(self) -> Tuple[int, ...]:
        return tuple(
            index
            for index, layer in enumerate(self.mamba_layers)
            if not isinstance(layer, nn.Identity)
        )


class HydraStage12Encoder(_CoreUpgradeEncoder):
    layer_kind = "hydra"

    def __init__(
        self,
        *args,
        hydra_d_state: int = 64,
        hydra_d_conv: int = 7,
        hydra_expand: int = 2,
        hydra_headdim: int = 64,
        hydra_ngroups: int = 1,
        hydra_chunk_size: int = 256,
        hydra_prefer_fused_scan: bool = True,
        bimamba_residual_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **kwargs,
        )
        for index in self._active_indices():
            old = self.mamba_layers[index]
            self.mamba_layers[index] = HydraBiMambaLayer(
                dim=old.dim,
                channel_token=old.channel_token,
                d_state=hydra_d_state,
                d_conv=hydra_d_conv,
                expand=hydra_expand,
                headdim=hydra_headdim,
                ngroups=hydra_ngroups,
                chunk_size=hydra_chunk_size,
                residual_scale_init=bimamba_residual_scale_init,
                prefer_fused_scan=hydra_prefer_fused_scan,
            )


class ComplexStateStage12Encoder(_CoreUpgradeEncoder):
    layer_kind = "complex_state"

    def __init__(
        self,
        *args,
        complex_state_d_state: int = 8,
        complex_state_d_conv: int = 4,
        complex_state_expand: int = 2,
        complex_state_scan_checkpoint: bool = True,
        complex_state_scan_backend: str = "auto",
        complex_state_fusion_hidden: int = 64,
        bimamba_residual_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **kwargs,
        )
        for index in self._active_indices():
            old = self.mamba_layers[index]
            self.mamba_layers[index] = ComplexStateBiMambaLayer(
                dim=old.dim,
                channel_token=old.channel_token,
                d_state=complex_state_d_state,
                d_conv=complex_state_d_conv,
                expand=complex_state_expand,
                scan_checkpoint=complex_state_scan_checkpoint,
                scan_backend=complex_state_scan_backend,
                fusion_hidden=complex_state_fusion_hidden,
                residual_scale_init=bimamba_residual_scale_init,
            )


class IndependentComplexStateStage12Encoder(_CoreUpgradeEncoder):
    layer_kind = "complex_state_independent"

    def __init__(
        self,
        *args,
        complex_state_d_state: int = 8,
        complex_state_d_conv: int = 4,
        complex_state_expand: int = 2,
        complex_state_scan_checkpoint: bool = True,
        complex_state_scan_backend: str = "auto",
        complex_state_fusion_hidden: int = 64,
        bimamba_residual_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **kwargs,
        )
        for index in self._active_indices():
            old = self.mamba_layers[index]
            self.mamba_layers[index] = IndependentComplexStateBiMambaLayer(
                dim=old.dim,
                channel_token=old.channel_token,
                d_state=complex_state_d_state,
                d_conv=complex_state_d_conv,
                expand=complex_state_expand,
                scan_checkpoint=complex_state_scan_checkpoint,
                scan_backend=complex_state_scan_backend,
                fusion_hidden=complex_state_fusion_hidden,
                residual_scale_init=bimamba_residual_scale_init,
            )


class MultiScaleStage12Encoder(_CoreUpgradeEncoder):
    layer_kind = "multiscale"

    def __init__(
        self,
        *args,
        multiscale_d_state: int = 16,
        multiscale_global_d_conv: int = 4,
        multiscale_expand: int = 2,
        multiscale_local_kernels: Sequence[int] = (3, 7, 15),
        multiscale_local_scale_init: float = 0.1,
        bimamba_residual_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **kwargs,
        )
        for index in self._active_indices():
            old = self.mamba_layers[index]
            self.mamba_layers[index] = MultiScaleBiMambaLayer(
                dim=old.dim,
                channel_token=old.channel_token,
                d_state=multiscale_d_state,
                global_d_conv=multiscale_global_d_conv,
                expand=multiscale_expand,
                local_kernel_sizes=multiscale_local_kernels,
                local_scale_init=multiscale_local_scale_init,
                residual_scale_init=bimamba_residual_scale_init,
            )


class _Stage12CoreUpgradeModel(nn.Module):
    encoder_cls: Type[_CoreUpgradeEncoder]

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[nn.Conv1d],
        kernel_sizes: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        bimamba_apply_stages: Sequence[int] = (1, 3),
        bimamba_residual_scale_init: float = 1.0,
        **core_kwargs,
    ) -> None:
        super().__init__()
        if int(n_stages) != 4:
            raise ValueError("Stage12 core upgrades require exactly four U-Net stages")
        apply_stages = tuple(sorted(int(index) for index in bimamba_apply_stages))
        if apply_stages != (1, 3):
            raise ValueError(
                "Stage12 core upgrades must keep BiMamba placement at stages [1, 3]"
            )
        norm_op_kwargs = (
            {"eps": 1e-5, "affine": True}
            if norm_op_kwargs is None
            else norm_op_kwargs
        )
        nonlin_kwargs = (
            {"inplace": True} if nonlin_kwargs is None else nonlin_kwargs
        )
        self.encoder = self.encoder_cls(
            input_size=(int(input_size),),
            input_channels=int(input_channels),
            n_stages=4,
            features_per_stage=list(features_per_stage),
            conv_op=conv_op,
            kernel_sizes=[[int(value)] for value in kernel_sizes],
            strides=[[int(value)] for value in strides],
            n_blocks_per_stage=list(n_conv_per_stage),
            conv_bias=bool(conv_bias),
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            bimamba_apply_stages=apply_stages,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **core_kwargs,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=int(num_classes),
            n_conv_per_stage=list(n_conv_per_stage_decoder),
            deep_supervision=bool(deep_supervision),
        )
        self.stage12_active_indices = apply_stages

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class IQUBiMamba1D_Hydra(_Stage12CoreUpgradeModel):
    """Stage 361: formal Hydra mixers at the original Stage-12 positions."""

    encoder_cls = HydraStage12Encoder

    def no_weight_decay(self) -> set[str]:
        return {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".A_log", ".dt_bias", ".D"))
        }


class IQUBiMamba1D_ComplexState(_Stage12CoreUpgradeModel):
    """Stage 362: conjugate complex-state scans with adaptive fusion."""

    encoder_cls = ComplexStateStage12Encoder

    def no_weight_decay(self) -> set[str]:
        return {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".a_log", ".theta", ".D"))
        }


class IQUBiMamba1D_MultiScale(_Stage12CoreUpgradeModel):
    """Stage 363: global BiMamba plus local d_conv 3/7/15 branches."""

    encoder_cls = MultiScaleStage12Encoder


class IQUBiMamba1D_IndependentComplexState(_Stage12CoreUpgradeModel):
    """Stage 364: two independent Stage-295 complex SSM directions."""

    encoder_cls = IndependentComplexStateStage12Encoder

    def no_weight_decay(self) -> set[str]:
        return {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".a_log", ".theta", ".D"))
        }


class IQUBiMamba1D_IndependentComplexStateUniRepLK(
    IQUBiMamba1D_IndependentComplexState
):
    """Stage 365: Stage 364 plus parallel UniRepLK deltas at stages 0/1/2."""

    def __init__(
        self,
        *args,
        rf_apply_stages: Sequence[int] = (0, 1, 2),
        rf_residual_scale_init: float = 0.05,
        rf_large_kernel: int = 17,
        rf_ffn_factor: int = 4,
        rf_layer_scale: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        stages = tuple(int(stage) for stage in rf_apply_stages)
        if len(set(stages)) != len(stages):
            raise ValueError("rf_apply_stages must be unique")
        for stage in stages:
            if not 0 <= stage < len(self.encoder.stages):
                raise ValueError(
                    f"UniRepLK stage {stage} is outside the "
                    f"{len(self.encoder.stages)}-stage encoder"
                )

        operator_config = {
            "rf_large_kernel": int(rf_large_kernel),
            "rf_ffn_factor": int(rf_ffn_factor),
            "rf_layer_scale": float(rf_layer_scale),
        }
        self.rf_apply_stages = stages
        self.stage_rf = nn.ModuleDict(
            {
                str(stage): ParallelFeatureDeltaAdapter(
                    int(self.encoder.output_channels[stage]),
                    build_recent_rf_operator(
                        "unireplk",
                        int(self.encoder.output_channels[stage]),
                        operator_config,
                    ),
                    float(rf_residual_scale_init),
                )
                for stage in stages
            }
        )

    def forward(self, x: torch.Tensor):
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)

        skips = []
        for stage, (conv_stage, memory) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            stage_features = conv_stage(x)
            x = memory(stage_features)
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](stage_features, x)
            skips.append(x)
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        names = super().no_weight_decay()
        names.update(
            f"stage_rf.{stage}.residual_scale" for stage in self.stage_rf
        )
        return names


__all__ = [
    "HydraQuasiseparableMixer",
    "HydraBiMambaLayer",
    "ComplexStateBiMambaLayer",
    "IndependentComplexStateBiMambaLayer",
    "MultiScaleBiMambaLayer",
    "IQUBiMamba1D_Hydra",
    "IQUBiMamba1D_ComplexState",
    "IQUBiMamba1D_MultiScale",
    "IQUBiMamba1D_IndependentComplexState",
    "IQUBiMamba1D_IndependentComplexStateUniRepLK",
]
