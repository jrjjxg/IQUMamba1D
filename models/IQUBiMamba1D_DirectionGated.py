"""Direction-gated all-stage BiMamba variant for IQUMamba1D.

Stage200 follows the stage198 all-stage placement, but changes the BiMamba
fusion rule. Instead of concatenating forward/backward states with a static
linear projection, each token learns a direction gate:

    y = g * forward + (1 - g) * backward

This keeps the model 4-stage aligned with stage4/stage12 while testing whether
IQ separation benefits from adaptive directional context.
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


class DirectionGatedBiMambaLayer(nn.Module):
    """Bidirectional Mamba with token-wise forward/backward direction gating."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba_fwd = Mamba(
            d_model=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=self.dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.direction_gate = nn.Sequential(
            nn.Linear(self.dim * 3, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.channel_token = bool(channel_token)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        h_fwd = self.mamba_fwd(x_norm)
        h_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        gate = self.direction_gate(torch.cat([x_norm, h_fwd, h_bwd], dim=-1))
        mixed = gate * h_fwd + (1.0 - gate) * h_bwd
        delta = self.out_proj(mixed)
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


class ResidualDirectionGatedBiMambaEncoder(ResidualSafeAllStageBiMambaEncoder):
    """Stage198 encoder geometry with direction-gated BiMamba layers."""

    def __init__(
        self,
        *args,
        bimamba_residual_scale_init: float = 0.01,
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
            self.mamba_layers[idx] = DirectionGatedBiMambaLayer(
                dim=layer.dim,
                channel_token=layer.channel_token,
                residual_scale_init=bimamba_residual_scale_init,
            )


class IQUBiMamba1D_DirectionGated(nn.Module):
    """4-stage IQ U-Net with all-stage direction-gated BiMamba."""

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
    ) -> None:
        super().__init__()
        self.encoder = ResidualDirectionGatedBiMambaEncoder(
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
