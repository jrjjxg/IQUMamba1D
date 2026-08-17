"""Stage380: Stage377 with lightweight temporal mask estimation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.IQUResUNet1D_ComplexStateUniRepLK_LatentMask import (
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
)


class LightweightTemporalSeparatorBlock(nn.Module):
    """Depthwise-separable temporal residual block for mask estimation."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        residual_scale_init: float,
    ) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        if channels < 1:
            raise ValueError("separator channels must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("separator kernel_size must be a positive odd integer")
        if dilation < 1:
            raise ValueError("separator dilation must be positive")

        padding = dilation * (kernel_size - 1) // 2
        self.dilation = dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm1d(channels, eps=1.0e-5, affine=True)
        self.activation = nn.GELU()
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(x)
        update = self.pointwise(self.activation(self.norm(update)))
        return x + self.residual_scale * update


class LightweightTemporalMaskEstimator(nn.Module):
    """A short dilated temporal separator followed by source mask logits."""

    def __init__(
        self,
        channels: int,
        num_sources: int,
        kernel_size: int = 5,
        dilations: Sequence[int] = (1, 2),
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        channels = int(channels)
        num_sources = int(num_sources)
        dilations = tuple(int(value) for value in dilations)
        if not dilations:
            raise ValueError("separator dilations must not be empty")
        if any(value < 1 for value in dilations):
            raise ValueError("separator dilations must be positive")

        self.blocks = nn.ModuleList(
            [
                LightweightTemporalSeparatorBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    residual_scale_init=residual_scale_init,
                )
                for dilation in dilations
            ]
        )
        self.output = nn.Conv1d(
            channels,
            num_sources * channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator(
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask
):
    """Stage377 with two local temporal blocks in every mask estimator."""

    def __init__(
        self,
        *args,
        separator_kernel_size: int = 5,
        separator_dilations: Sequence[int] = (1, 2),
        separator_residual_scale_init: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.separator_kernel_size = int(separator_kernel_size)
        self.separator_dilations = tuple(
            int(value) for value in separator_dilations
        )
        self.separator_residual_scale_init = float(
            separator_residual_scale_init
        )
        self.latent_mask_heads = nn.ModuleList(
            [
                LightweightTemporalMaskEstimator(
                    channels=int(channels),
                    num_sources=self.latent_mask_num_sources,
                    kernel_size=self.separator_kernel_size,
                    dilations=self.separator_dilations,
                    residual_scale_init=self.separator_residual_scale_init,
                )
                for channels in self.encoder.output_channels
            ]
        )


__all__ = [
    "LightweightTemporalSeparatorBlock",
    "LightweightTemporalMaskEstimator",
    "IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator",
]
