"""IQUBiMamba1D_CSB - BiMamba with a complex-valued stem and bottleneck bridge.

Design rationale:
  - Wireless IQ mixtures are inherently complex-valued, so the earliest feature
    extraction stage should preserve I/Q coupling instead of treating I and Q
    as unrelated real channels.
  - The original IQUBiMamba1D backbone is already fairly rich. Converting the
    full U-Net to complex-valued modules would make the model heavy and invasive.
  - A pragmatic compromise is therefore:
      1. use a complex-valued stem to build phase-aware low-level features;
      2. keep the mature BiMamba U-Net encoder/decoder intact in the middle;
      3. add a complex-valued bottleneck bridge at the deepest stage where the
         temporal length is shortest and multi-scale complex refinement is cheap.

This follows the spirit of complex-domain wireless separation models such as
CTDCRN and C2ESDNet, while remaining architecture-compatible with the existing
IQUBiMamba1D training pipeline.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUBiMamba1D import BiMambaLayer
from models.IQUMamba1D import BasicResBlock, UNetResDecoder
from models.ctdcrn import (
    ComplexConv1d,
    ComplexDilatedConvModule,
    ComplexGlobalLayerNorm,
    ComplexLayerNorm,
    ComplexLeakyReLU,
)


def _split_complex(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4 or x.size(1) != 2:
        raise ValueError(f"Expected (B, 2, C, T), got {tuple(x.shape)}")
    return x[:, 0, ...], x[:, 1, ...]


def _stack_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    if real.shape != imag.shape:
        raise ValueError(
            f"Complex tensor shape mismatch: real={tuple(real.shape)} imag={tuple(imag.shape)}"
        )
    return torch.stack([real, imag], dim=1)


class RealToComplexProjection1D(nn.Module):
    """Project real-valued feature maps to paired complex channels."""

    def __init__(self, in_channels: int, out_complex_channels: int) -> None:
        super().__init__()
        self.real_proj = nn.Conv1d(in_channels, out_complex_channels, kernel_size=1, bias=True)
        self.imag_proj = nn.Conv1d(in_channels, out_complex_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real_proj(x)
        imag = self.imag_proj(x)
        return _stack_complex(real, imag)


class ComplexToRealProjection1D(nn.Module):
    """Collapse paired complex channels back to a real-valued backbone width."""

    def __init__(self, in_complex_channels: int, out_channels: int, zero_init: bool = False) -> None:
        super().__init__()
        self.proj = nn.Conv1d(2 * in_complex_channels, out_channels, kernel_size=1, bias=True)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        return self.proj(torch.cat([real, imag], dim=1))


class ComplexStemBlock1D(nn.Module):
    """Complex-valued stem that converts raw IQ into a real-valued backbone tensor.

    The stem follows the "complex hierarchical encoder" idea from CTDCRN:
      ComplexConv -> complex norm -> complex activation -> ComplexConv
    and then projects the resulting complex representation back to the real
    BiMamba width with a residual shortcut from the raw IQ input.
    """

    def __init__(
        self,
        output_channels: int,
        hidden_complex_channels: int = 32,
        kernel_size: int = 5,
        eps: float = 1e-8,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        self.conv1 = ComplexConv1d(1, hidden_complex_channels, kernel_size=kernel_size, padding="same", bias=True)
        self.norm1 = ComplexLayerNorm(hidden_complex_channels, eps=eps, affine=True)
        self.act1 = ComplexLeakyReLU(negative_slope=negative_slope)

        self.conv2 = ComplexConv1d(
            hidden_complex_channels,
            hidden_complex_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=True,
        )
        self.norm2 = ComplexLayerNorm(hidden_complex_channels, eps=eps, affine=True)
        self.act2 = ComplexLeakyReLU(negative_slope=negative_slope)

        self.to_real = ComplexToRealProjection1D(hidden_complex_channels, output_channels, zero_init=False)
        self.input_skip = nn.Conv1d(2, output_channels, kernel_size=1, bias=True)
        self.out_norm = nn.InstanceNorm1d(output_channels, eps=1e-5, affine=True)
        self.out_act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected IQ input (B, 2, T), got {tuple(x.shape)}")
        z = x.unsqueeze(2)  # (B, 2, 1, T)
        z = self.conv1(z)
        z = self.norm1(z)
        z = self.act1(z)
        z = self.conv2(z)
        z = self.norm2(z)
        z = self.act2(z)
        y = self.to_real(z)
        y = y + self.input_skip(x)
        return self.out_act(self.out_norm(y))


class ComplexBottleneckBridge1D(nn.Module):
    """Complex-valued refinement bridge applied at the deepest encoder stage.

    The block projects the real-valued bottleneck feature map into a complex
    latent space, applies several complex dilated residual modules with growing
    dilation factors, and then projects back to the real backbone width through
    a zero-initialized output layer so the whole bridge starts as an identity.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 128,
        num_blocks: int = 3,
        kernel_size: int = 5,
        dilation_growth: int = 2,
        eps: float = 1e-8,
        negative_slope: float = 0.01,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        if dilation_growth < 1:
            raise ValueError(f"dilation_growth must be >= 1, got {dilation_growth}")

        self.to_complex = RealToComplexProjection1D(channels, channels)
        self.pre_norm = ComplexGlobalLayerNorm(channels, eps=eps, affine=True)
        self.blocks = nn.ModuleList()

        dilation = 1
        for _ in range(num_blocks):
            self.blocks.append(
                ComplexDilatedConvModule(
                    channels=channels,
                    bottleneck_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    eps=eps,
                    leaky_relu_slope=negative_slope,
                )
            )
            dilation *= dilation_growth

        self.to_real = ComplexToRealProjection1D(channels, channels, zero_init=zero_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_complex(x)
        z = self.pre_norm(z)
        for block in self.blocks:
            z = block(z)
        return x + self.to_real(z)


class ResidualBiMambaEncoder_CSB(nn.Module):
    """ResidualBiMambaEncoder with a complex stem and a complex bottleneck bridge."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        return_skips: bool = False,
        stem_channels: int | None = None,
        pool_type: str = "conv",
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
    ) -> None:
        super().__init__()
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i / j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]

        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = ComplexStemBlock1D(
            output_channels=stem_channels,
            hidden_complex_channels=int(complex_stem_hidden_channels),
            kernel_size=int(complex_stem_kernel_size),
        )

        input_channels = stem_channels
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                    for _ in range(n_blocks_per_stage[s] - 1)
                ],
            )

            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    BiMambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s],
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.complex_bottleneck = ComplexBottleneckBridge1D(
            channels=int(features_per_stage[-1]),
            hidden_channels=int(complex_bottleneck_hidden_channels),
            num_blocks=int(complex_bottleneck_num_blocks),
            kernel_size=int(complex_bottleneck_kernel_size),
            dilation_growth=int(complex_bottleneck_dilation_growth),
            zero_init=bool(complex_bottleneck_zero_init),
        )

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            if s == len(self.stages) - 1:
                x = self.complex_bottleneck(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUBiMamba1D_CSB(nn.Module):
    """BiMamba U-Net with a complex stem and a complex bottleneck bridge."""

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
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ResidualBiMambaEncoder_CSB(
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
            complex_stem_hidden_channels=complex_stem_hidden_channels,
            complex_stem_kernel_size=complex_stem_kernel_size,
            complex_bottleneck_hidden_channels=complex_bottleneck_hidden_channels,
            complex_bottleneck_num_blocks=complex_bottleneck_num_blocks,
            complex_bottleneck_kernel_size=complex_bottleneck_kernel_size,
            complex_bottleneck_dilation_growth=complex_bottleneck_dilation_growth,
            complex_bottleneck_zero_init=complex_bottleneck_zero_init,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)
