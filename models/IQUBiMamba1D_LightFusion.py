"""Lightweight BiMamba direction-fusion variants.

The variants keep the stage-12/stage-4 BiMamba U-Net geometry and replace only
the forward/backward fusion rule inside selected encoder Mamba layers.
"""

from __future__ import annotations

from typing import List, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.amp import autocast

from mamba_ssm import Mamba

from models.IQUMamba1D import UNetResDecoder
from models.IQUBiMamba1D_SafeAllStages import (
    HALF_PRECISION_DTYPES,
    ResidualSafeAllStageBiMambaEncoder,
)


def _default_bimamba_stages(n_stages: int) -> Tuple[int, ...]:
    return tuple(stage for stage in range(int(n_stages)) if bool(stage % 2) ^ bool(n_stages % 2))


class DifferenceFusionBiMambaLayer(nn.Module):
    """Bidirectional Mamba with symmetric context plus controlled direction difference."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        residual_scale_init: float = 0.01,
        diff_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba_fwd = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.diff_scale = nn.Parameter(torch.tensor(float(diff_scale_init)))
        self.channel_token = bool(channel_token)

    def _mix_delta(self, h_fwd: torch.Tensor, h_bwd: torch.Tensor) -> torch.Tensor:
        symmetric_context = 0.5 * (h_fwd + h_bwd)
        direction_difference = 0.5 * (h_fwd - h_bwd)
        return symmetric_context + torch.tanh(self.diff_scale) * direction_difference

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        h_fwd = self.mamba_fwd(x_norm)
        h_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        delta = self._mix_delta(h_fwd=h_fwd, h_bwd=h_bwd)
        return x + self.res_scale * delta

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = int(np.prod(dims))
        x_flat = x.reshape(batch, channels, n_tokens).transpose(-1, -2)
        out = self._fuse(x_flat)
        return out.transpose(-1, -2).reshape(batch, channels, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        out = self._fuse(x_flat)
        return out.reshape(batch, n_tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class AdaptiveDifferenceFusionBiMambaLayer(DifferenceFusionBiMambaLayer):
    """Difference fusion with a tiny token/channel reliability gate."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        residual_scale_init: float = 0.01,
        diff_scale_init: float = 0.25,
        gate_logit_init: float = -1.5,
        gate_token_scale_init: float = 1.0,
        gate_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            channel_token=channel_token,
            residual_scale_init=residual_scale_init,
            diff_scale_init=diff_scale_init,
        )
        self.channel_gate_logit = nn.Parameter(torch.full((self.dim,), float(gate_logit_init)))
        self.token_gate_scale = nn.Parameter(torch.tensor(float(gate_token_scale_init)))
        self.gate_eps = float(gate_eps)

    def _direction_reliability(self, direction_difference: torch.Tensor) -> torch.Tensor:
        direction_reliability = direction_difference.square().mean(dim=-1, keepdim=True)
        mean = direction_reliability.mean(dim=1, keepdim=True)
        std = direction_reliability.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.gate_eps)
        return (direction_reliability - mean) / std

    def _mix_delta(self, h_fwd: torch.Tensor, h_bwd: torch.Tensor) -> torch.Tensor:
        symmetric_context = 0.5 * (h_fwd + h_bwd)
        direction_difference = 0.5 * (h_fwd - h_bwd)
        direction_reliability = self._direction_reliability(direction_difference)
        gate_logits = (
            self.channel_gate_logit.view(1, 1, self.dim)
            + self.token_gate_scale * direction_reliability
        )
        direction_gate = torch.sigmoid(gate_logits)
        return symmetric_context + torch.tanh(self.diff_scale) * direction_gate * direction_difference


class ResidualLightFusionBiMambaEncoder(ResidualSafeAllStageBiMambaEncoder):
    """Safe encoder geometry with lightweight direction-fusion layers."""

    layer_cls = DifferenceFusionBiMambaLayer

    def __init__(
        self,
        *args,
        bimamba_residual_scale_init: float = 0.01,
        bimamba_diff_scale_init: float = 0.0,
        bimamba_gate_logit_init: float = -1.5,
        bimamba_gate_token_scale_init: float = 1.0,
        bimamba_gate_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **kwargs,
        )
        for idx, layer in enumerate(self.mamba_layers):
            if isinstance(layer, nn.Identity):
                continue
            layer_kwargs = {
                "dim": layer.dim,
                "channel_token": layer.channel_token,
                "residual_scale_init": bimamba_residual_scale_init,
                "diff_scale_init": bimamba_diff_scale_init,
            }
            if self.layer_cls is AdaptiveDifferenceFusionBiMambaLayer:
                layer_kwargs.update(
                    gate_logit_init=bimamba_gate_logit_init,
                    gate_token_scale_init=bimamba_gate_token_scale_init,
                    gate_eps=bimamba_gate_eps,
                )
            self.mamba_layers[idx] = self.layer_cls(**layer_kwargs)


class ResidualAdaptiveLightFusionBiMambaEncoder(ResidualLightFusionBiMambaEncoder):
    layer_cls = AdaptiveDifferenceFusionBiMambaLayer


class _IQUBiMamba1D_LightFusionBase(nn.Module):
    encoder_cls = ResidualLightFusionBiMambaEncoder

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
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        bimamba_apply_stages: List[int] | Tuple[int, ...] | None = None,
        bimamba_residual_scale_init: float = 0.01,
        bimamba_diff_scale_init: float = 0.0,
        bimamba_gate_logit_init: float = -1.5,
        bimamba_gate_token_scale_init: float = 1.0,
        bimamba_gate_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if bimamba_apply_stages is None:
            bimamba_apply_stages = _default_bimamba_stages(n_stages)
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
            bimamba_diff_scale_init=bimamba_diff_scale_init,
            bimamba_gate_logit_init=bimamba_gate_logit_init,
            bimamba_gate_token_scale_init=bimamba_gate_token_scale_init,
            bimamba_gate_eps=bimamba_gate_eps,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)


class IQUBiMamba1D_DiffFusion(_IQUBiMamba1D_LightFusionBase):
    """Stage 206: minimal symmetric/difference BiMamba fusion."""

    encoder_cls = ResidualLightFusionBiMambaEncoder


class IQUBiMamba1D_AdaptiveDiffFusion(_IQUBiMamba1D_LightFusionBase):
    """Stage 207: lightweight reliability-gated difference BiMamba fusion."""

    encoder_cls = ResidualAdaptiveLightFusionBiMambaEncoder
