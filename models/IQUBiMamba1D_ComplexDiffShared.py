"""Bottleneck-only heterogeneous BiMamba with a shared Mamba core."""

from __future__ import annotations

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


class ComplexDiffSharedBiMambaLayer(nn.Module):
    """Scan absolute tokens and phase-invariant differential tokens with one core."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        channel_token: bool = False,
        residual_scale_init: float = 0.1,
        gate_init: float = 0.2,
        differential_stride: int = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.eps = float(eps)
        self.differential_stride = max(1, int(differential_stride))
        self.absolute_norm = nn.LayerNorm(self.dim)
        self.difference_norm = nn.LayerNorm(self.dim)
        self.shared_mamba = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(torch.full((self.dim,), float(np.log(gate_init / (1.0 - gate_init)))))
        self.res_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _complex_differential_tokens(self, x: torch.Tensor) -> torch.Tensor:
        original_dim = x.size(-1)
        if original_dim % 2:
            x = torch.nn.functional.pad(x, (0, 1))
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs[..., 0], pairs[..., 1]
        prev_real = torch.cat([real[:, :1], real[:, :-1]], dim=1)
        prev_imag = torch.cat([imag[:, :1], imag[:, :-1]], dim=1)
        prod_real = real * prev_real + imag * prev_imag
        prod_imag = imag * prev_real - real * prev_imag
        magnitude = torch.sqrt(prod_real.square() + prod_imag.square() + self.eps)
        scale = torch.log1p(magnitude)
        differential = torch.stack(
            [scale * prod_real / magnitude, scale * prod_imag / magnitude],
            dim=-1,
        ).flatten(-2)
        return differential[..., :original_dim]

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        absolute = self.shared_mamba(self.absolute_norm(x))
        differential_tokens = self._complex_differential_tokens(x)
        differential_tokens = differential_tokens[:, ::self.differential_stride, :]
        differential = self.shared_mamba(
            self.difference_norm(differential_tokens).flip(dims=[1])
        ).flip(dims=[1])
        if differential.size(1) != x.size(1):
            differential = F.interpolate(
                differential.transpose(1, 2), size=x.size(1), mode="linear", align_corners=False
            ).transpose(1, 2)
        gate = torch.sigmoid(self.gate_logit).view(1, 1, self.dim)
        delta = (1.0 - gate) * absolute + gate * differential
        return x + self.res_scale * delta

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


class ResidualComplexDiffSharedEncoder(ResidualSafeAllStageBiMambaEncoder):
    def __init__(
        self,
        *args,
        bimamba_residual_scale_init: float = 0.1,
        bimamba_complex_diff_gate_init: float = 0.2,
        bimamba_complex_diff_stride: int = 2,
        bimamba_complex_diff_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            **kwargs,
        )
        for index, layer in enumerate(self.mamba_layers):
            if isinstance(layer, nn.Identity):
                continue
            self.mamba_layers[index] = ComplexDiffSharedBiMambaLayer(
                dim=layer.dim,
                channel_token=layer.channel_token,
                residual_scale_init=bimamba_residual_scale_init,
                gate_init=bimamba_complex_diff_gate_init,
                differential_stride=bimamba_complex_diff_stride,
                eps=bimamba_complex_diff_eps,
            )


class IQUBiMamba1D_ComplexDiffShared(nn.Module):
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
        deep_supervision: bool = False,
        bimamba_apply_stages: List[int] | Tuple[int, ...] = (3,),
        bimamba_residual_scale_init: float = 0.1,
        bimamba_complex_diff_gate_init: float = 0.2,
        bimamba_complex_diff_stride: int = 2,
        bimamba_complex_diff_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.encoder = ResidualComplexDiffSharedEncoder(
            input_size=(input_size,), input_channels=input_channels, n_stages=n_stages,
            features_per_stage=features_per_stage, conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes], strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage, conv_bias=True,
            norm_op=nn.InstanceNorm1d, norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True}, return_skips=True,
            bimamba_apply_stages=bimamba_apply_stages,
            bimamba_residual_scale_init=bimamba_residual_scale_init,
            bimamba_complex_diff_gate_init=bimamba_complex_diff_gate_init,
            bimamba_complex_diff_stride=bimamba_complex_diff_stride,
            bimamba_complex_diff_eps=bimamba_complex_diff_eps,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder, num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))
