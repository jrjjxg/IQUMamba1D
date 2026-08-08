"""Low-overhead robust bidirectional Mamba variants for IQ separation."""

from __future__ import annotations

import math
from typing import List, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast

from mamba_ssm import Mamba

from models.IQUMamba1D import UNetResDecoder
from models.IQUBiMamba1D_SafeAllStages import (
    HALF_PRECISION_DTYPES,
    ResidualSafeAllStageBiMambaEncoder,
)


def _inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class _TokenLayerBase(nn.Module):
    def __init__(self, dim: int, channel_token: bool = False) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        tokens = int(np.prod(dims))
        flat = x.reshape(batch, channels, tokens).transpose(-1, -2)
        return self._fuse(flat).transpose(-1, -2).reshape(batch, channels, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        flat = x.flatten(2)
        return self._fuse(flat).reshape(batch, tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class TimeReversalSharedBiMambaLayer(_TokenLayerBase):
    """Shared-core BiMamba with reversal-equivariant uncertainty shrinkage."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        residual_scale_init: float = 1.0,
        boundary_tau_init: float = 16.0,
        shrinkage_init: float = 0.02,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(dim=dim, channel_token=channel_token)
        self.norm = nn.LayerNorm(self.dim)
        self.shared_mamba = Mamba(
            d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.log_tau = nn.Parameter(torch.tensor(_inverse_softplus(boundary_tau_init - 1.0)))
        self.shrinkage_raw = nn.Parameter(torch.tensor(_inverse_softplus(shrinkage_init)))
        self.eps = float(eps)

    def _boundary_weight(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=dtype)
        tau = F.softplus(self.log_tau).to(dtype=dtype) + 1.0
        forward_reliability = -torch.expm1(-(positions + 1.0) / tau)
        backward_reliability = -torch.expm1(-(float(length) - positions) / tau)
        return (
            forward_reliability / (forward_reliability + backward_reliability).clamp_min(self.eps)
        ).view(1, length, 1)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        forward = self.shared_mamba(x_norm)
        backward = self.shared_mamba(x_norm.flip(1)).flip(1)
        forward_weight = self._boundary_weight(x.size(1), x.device, x.dtype)
        consensus = forward_weight * forward + (1.0 - forward_weight) * backward

        disagreement = (forward - backward).square().mean(dim=-1, keepdim=True)
        energy = (forward.square() + backward.square()).mean(dim=-1, keepdim=True)
        uncertainty = disagreement / energy.clamp_min(self.eps)
        confidence = 1.0 / (1.0 + F.softplus(self.shrinkage_raw) * uncertainty)
        return x + self.res_scale * confidence * consensus


class AlternatingGlobalLocalMambaLayer(_TokenLayerBase):
    """One global scan plus cheap local context from the opposite direction."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        global_reverse: bool = False,
        local_kernel_size: int = 5,
        local_gate_init: float = 0.1,
    ) -> None:
        super().__init__(dim=dim, channel_token=channel_token)
        kernel_size = max(1, int(local_kernel_size))
        self.global_reverse = bool(global_reverse)
        self.local_kernel_size = kernel_size
        self.norm = nn.LayerNorm(self.dim)
        self.global_mamba = Mamba(
            d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.local_conv = nn.Conv1d(
            self.dim,
            self.dim,
            kernel_size=kernel_size,
            groups=self.dim,
            bias=True,
        )
        local_gate_init = min(max(float(local_gate_init), 1e-4), 1.0 - 1e-4)
        self.local_gate_logit = nn.Parameter(
            torch.full((self.dim,), math.log(local_gate_init / (1.0 - local_gate_init)))
        )

    def _causal_local(self, sequence: torch.Tensor) -> torch.Tensor:
        channels_first = sequence.transpose(1, 2)
        channels_first = F.pad(channels_first, (self.local_kernel_size - 1, 0))
        return F.silu(self.local_conv(channels_first)).transpose(1, 2)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        global_input = x_norm.flip(1) if self.global_reverse else x_norm
        global_context = self.global_mamba(global_input)
        if self.global_reverse:
            global_context = global_context.flip(1)

        local_reverse = not self.global_reverse
        local_input = x_norm.flip(1) if local_reverse else x_norm
        local_context = self._causal_local(local_input)
        if local_reverse:
            local_context = local_context.flip(1)
        gate = torch.sigmoid(self.local_gate_logit).view(1, 1, self.dim)
        return global_context + gate * local_context


class _RobustFusionEncoder(ResidualSafeAllStageBiMambaEncoder):
    layer_kind = "shared"

    def __init__(
        self,
        *args,
        bimamba_residual_scale_init: float = 1.0,
        bimamba_boundary_tau_init: float = 16.0,
        bimamba_shrinkage_init: float = 0.02,
        bimamba_fusion_eps: float = 1e-6,
        bimamba_local_kernel_size: int = 5,
        bimamba_local_gate_init: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, bimamba_residual_scale_init=1.0, **kwargs)
        active_indices = [i for i, layer in enumerate(self.mamba_layers) if not isinstance(layer, nn.Identity)]
        for rank, index in enumerate(active_indices):
            # Direction has physical meaning only along time; never inherit the
            # base encoder's optional channel-token scan at short resolutions.
            common = {"dim": int(self.output_channels[index]), "channel_token": False}
            if self.layer_kind == "shared":
                replacement = TimeReversalSharedBiMambaLayer(
                    **common,
                    residual_scale_init=bimamba_residual_scale_init,
                    boundary_tau_init=bimamba_boundary_tau_init,
                    shrinkage_init=bimamba_shrinkage_init,
                    eps=bimamba_fusion_eps,
                )
            else:
                replacement = AlternatingGlobalLocalMambaLayer(
                    **common,
                    global_reverse=bool(rank % 2),
                    local_kernel_size=bimamba_local_kernel_size,
                    local_gate_init=bimamba_local_gate_init,
                )
            self.mamba_layers[index] = replacement


class TimeReversalSharedBiMambaEncoder(_RobustFusionEncoder):
    layer_kind = "shared"


class AlternatingGlobalLocalMambaEncoder(_RobustFusionEncoder):
    layer_kind = "alternating"


class _RobustFusionModel(nn.Module):
    encoder_cls = TimeReversalSharedBiMambaEncoder

    def __init__(
        self,
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
        bimamba_apply_stages: List[int] | Tuple[int, ...] = (1, 3),
        bimamba_residual_scale_init: float = 1.0,
        bimamba_boundary_tau_init: float = 16.0,
        bimamba_shrinkage_init: float = 0.02,
        bimamba_fusion_eps: float = 1e-6,
        bimamba_local_kernel_size: int = 5,
        bimamba_local_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        norm_op_kwargs = {"eps": 1e-5, "affine": True} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {"inplace": True} if nonlin_kwargs is None else nonlin_kwargs
        self.encoder = self.encoder_cls(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            bimamba_apply_stages=bimamba_apply_stages,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            bimamba_boundary_tau_init=bimamba_boundary_tau_init,
            bimamba_shrinkage_init=bimamba_shrinkage_init,
            bimamba_fusion_eps=bimamba_fusion_eps,
            bimamba_local_kernel_size=bimamba_local_kernel_size,
            bimamba_local_gate_init=bimamba_local_gate_init,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))


class IQUBiMamba1D_TimeReversalShared(_RobustFusionModel):
    """Stage 217: shared-core, reversal-equivariant full BiMamba."""

    encoder_cls = TimeReversalSharedBiMambaEncoder


class IQUBiMamba1D_AlternatingGlobalLocal(_RobustFusionModel):
    """Stage 218: alternating global scan with opposite local context."""

    encoder_cls = AlternatingGlobalLocalMambaEncoder
