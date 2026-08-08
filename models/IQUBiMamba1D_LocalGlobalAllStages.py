"""Local-global all-stage BiMamba variant for IQUMamba1D.

Stage201 follows the stage198 all-stage BiMamba placement, but each BiMamba
block has two residual branches:

  - a local depthwise-convolution branch for symbol-neighborhood detail
  - a global bidirectional Mamba branch for long-range context

The two branches are fused by a learned gate and added back with a small
learnable residual scale. This borrows the hybrid CNN/SSM lesson from U-Mamba
style designs while staying 4-stage aligned with stage4/stage12.
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


class _GlobalBiMambaBranch(nn.Module):
    """Bidirectional Mamba branch without residual addition."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba_fwd = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(self.dim * 2, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        h_fwd = self.mamba_fwd(x_norm)
        h_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        return self.out_proj(torch.cat([h_fwd, h_bwd], dim=-1))


class LocalGlobalAllStageBiMambaLayer(nn.Module):
    """Fuse local convolutional detail with global bidirectional Mamba context."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        local_kernel_size: int = 7,
        gate_hidden: int = 64,
        channel_token: bool = False,
        residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if int(local_kernel_size) % 2 == 0:
            raise ValueError(f"local_kernel_size must be odd, got {local_kernel_size}")
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.global_branch = _GlobalBiMambaBranch(self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.local_branch = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=int(local_kernel_size),
                padding=int(local_kernel_size) // 2,
                groups=self.dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=True),
        )
        hidden = max(4, int(gate_hidden))
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _local(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local_branch[0](x).transpose(1, 2)
        y = self.local_branch[1](y)
        y = self.local_branch[2](y)
        y = self.local_branch[3](y)
        return y.transpose(1, 2)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        local_delta = self._local(x)
        global_delta = self.global_branch(x)
        weights = torch.softmax(self.fusion_gate(x.mean(dim=1)), dim=-1).view(x.size(0), 2, 1, 1)
        branches = torch.stack([local_delta, global_delta], dim=1)
        delta = (branches * weights).sum(dim=1)
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


class ResidualLocalGlobalBiMambaEncoder(ResidualSafeAllStageBiMambaEncoder):
    """Stage198 encoder geometry with local-global BiMamba layers."""

    def __init__(
        self,
        *args,
        bimamba_residual_scale_init: float = 0.01,
        local_kernel_size: int = 7,
        local_global_gate_hidden: int = 64,
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
            self.mamba_layers[idx] = LocalGlobalAllStageBiMambaLayer(
                dim=layer.dim,
                channel_token=layer.channel_token,
                residual_scale_init=bimamba_residual_scale_init,
                local_kernel_size=local_kernel_size,
                gate_hidden=local_global_gate_hidden,
            )


class IQUBiMamba1D_LocalGlobalAllStages(nn.Module):
    """4-stage IQ U-Net with all-stage local-global BiMamba fusion."""

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
        local_kernel_size: int = 7,
        local_global_gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ResidualLocalGlobalBiMambaEncoder(
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
            local_kernel_size=local_kernel_size,
            local_global_gate_hidden=local_global_gate_hidden,
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
