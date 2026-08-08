"""IQUComplexUNet1D - pure complex-convolutional 1D U-Net baseline.

Contract:
  input : (B, 2, T) where channels are I/Q
  output: (B, 2*K, T) as [I1,Q1,I2,Q2,...]

Internal tensors use paired complex channels: (B, 2, C, T), where dim 1 is
real/imag.  There are no Mamba, attention, or receiver-prior blocks here; this
is intended as a clean complex-CNN U-Net baseline.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from models.ctdcrn import ComplexConv1d, ComplexGlobalLayerNorm, ComplexLayerNorm, ComplexLeakyReLU


def _split_complex(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4 or x.size(1) != 2:
        raise ValueError(f"Expected complex tensor (B, 2, C, T), got {tuple(x.shape)}")
    return x[:, 0, ...], x[:, 1, ...]


def _stack_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    if real.shape != imag.shape:
        raise ValueError(
            f"Complex tensor shape mismatch: real={tuple(real.shape)} imag={tuple(imag.shape)}"
        )
    return torch.stack([real, imag], dim=1)


def _cat_complex(xs: List[torch.Tensor], dim: int = 1) -> torch.Tensor:
    reals, imags = zip(*[_split_complex(x) for x in xs])
    return _stack_complex(torch.cat(reals, dim=dim), torch.cat(imags, dim=dim))


def _interpolate_complex(x: torch.Tensor, size: int | None = None, scale_factor: int | None = None) -> torch.Tensor:
    real, imag = _split_complex(x)
    kwargs = {"mode": "linear", "align_corners": False}
    if size is None:
        real = F.interpolate(real, scale_factor=scale_factor, **kwargs)
        imag = F.interpolate(imag, scale_factor=scale_factor, **kwargs)
    else:
        real = F.interpolate(real, size=size, **kwargs)
        imag = F.interpolate(imag, size=size, **kwargs)
    return _stack_complex(real, imag)


def _complex_sources_to_channels(x: torch.Tensor) -> torch.Tensor:
    real, imag = _split_complex(x)
    return torch.stack([real, imag], dim=2).flatten(1, 2)


class ComplexResBlock1D(nn.Module):
    """Complex residual block with two complex convolutions."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        eps: float = 1e-8,
        negative_slope: float = 0.01,
        use_1x1conv: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = ComplexConv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.norm1 = ComplexLayerNorm(output_channels, eps=eps, affine=True)
        self.act1 = ComplexLeakyReLU(negative_slope=negative_slope)
        self.conv2 = ComplexConv1d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.norm2 = ComplexLayerNorm(output_channels, eps=eps, affine=True)
        self.act2 = ComplexLeakyReLU(negative_slope=negative_slope)
        if use_1x1conv or input_channels != output_channels or stride != 1:
            self.shortcut = ComplexConv1d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        y = y + self.shortcut(x)
        return self.act2(y)


class ComplexStemBlock1D(nn.Module):
    def __init__(self, output_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.block = ComplexResBlock1D(
            input_channels=1,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            use_1x1conv=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected IQ input (B, 2, T), got {tuple(x.shape)}")
        return self.block(x.unsqueeze(2))


class ComplexUpsampleLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.proj = ComplexConv1d(input_channels, output_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(_interpolate_complex(x, scale_factor=self.scale_factor))


class ComplexSkipConnectionProcessor(nn.Module):
    """Lightweight complex skip alignment and fusion."""

    def __init__(self, skip_channels: int, upsampled_channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.align_skip = nn.Sequential(
            ComplexConv1d(skip_channels, skip_channels, kernel_size=1, padding=0, bias=True),
            ComplexGlobalLayerNorm(skip_channels, eps=eps, affine=True),
            ComplexLeakyReLU(),
        )
        self.fuse = nn.Sequential(
            ComplexConv1d(skip_channels + upsampled_channels, skip_channels, kernel_size=1, padding=0, bias=True),
            ComplexGlobalLayerNorm(skip_channels, eps=eps, affine=True),
            ComplexLeakyReLU(),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, skip_features: torch.Tensor, upsampled_features: torch.Tensor) -> torch.Tensor:
        identity = skip_features
        skip = self.align_skip(skip_features)
        if upsampled_features.size(-1) != skip.size(-1):
            upsampled_features = _interpolate_complex(upsampled_features, size=skip.size(-1))
        fused = self.fuse(_cat_complex([skip, upsampled_features], dim=1))
        return self.residual_weight * fused + (1.0 - self.residual_weight) * identity


class ComplexConvEncoder(nn.Module):
    """Pure complex-convolutional encoder matching the project U-Net interface."""

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
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        return_skips: bool = False,
        stem_channels: int | None = None,
        pool_type: str = "conv",
        complex_stem_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        del input_size, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, pool_type
        if input_channels != 2:
            raise ValueError(f"ComplexUNet1D expects input_channels=2, got {input_channels}")

        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = ComplexStemBlock1D(stem_channels, kernel_size=int(complex_stem_kernel_size))

        stages = []
        input_complex_channels = stem_channels
        for s in range(n_stages):
            blocks = [
                ComplexResBlock1D(
                    input_channels=input_complex_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                )
            ]
            blocks.extend(
                ComplexResBlock1D(
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s][0],
                    stride=1,
                )
                for _ in range(n_blocks_per_stage[s] - 1)
            )
            stages.append(nn.Sequential(*blocks))
            input_complex_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class ComplexConvUNetDecoder(nn.Module):
    """Pure complex-convolutional decoder producing standard IQ source channels."""

    def __init__(self, encoder: ComplexConvEncoder, num_classes: int, n_conv_per_stage, deep_supervision: bool):
        super().__init__()
        if num_classes % 2 != 0:
            raise ValueError(f"ComplexUNet1D expects even num_classes=2*num_sources, got {num_classes}")
        self.deep_supervision = bool(deep_supervision)
        self.num_complex_sources = num_classes // 2

        n_stages_encoder = len(encoder.output_channels)
        n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1) if isinstance(n_conv_per_stage, int) else n_conv_per_stage
        stages = []
        upsample_layers = []
        seg_layers = []
        skip_processors = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s][0]

            upsample_layers.append(
                ComplexUpsampleLayer(
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    scale_factor=stride_for_upsampling,
                )
            )
            skip_processors.append(
                ComplexSkipConnectionProcessor(
                    skip_channels=input_features_skip,
                    upsampled_channels=input_features_skip,
                )
            )

            blocks = [
                ComplexResBlock1D(
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=True,
                )
            ]
            blocks.extend(
                ComplexResBlock1D(
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                )
                for _ in range(n_conv_per_stage[s - 1] - 1)
            )
            stages.append(nn.Sequential(*blocks))
            seg_layers.append(ComplexConv1d(input_features_skip, self.num_complex_sources, kernel_size=1, padding=0))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.skip_processors = nn.ModuleList(skip_processors)

    def forward(self, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            processed_skip = self.skip_processors[s](skips[-(s + 2)], x)
            x = _cat_complex([x, processed_skip], dim=1)
            x = self.stages[s](x)
            seg_outputs.append(_complex_sources_to_channels(self.seg_layers[s](x)))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQUComplexUNet1D(nn.Module):
    """Pure complex-convolutional U-Net baseline for IQ separation."""

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
        complex_stem_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = ComplexConvEncoder(
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
            complex_stem_kernel_size=complex_stem_kernel_size,
        )
        self.decoder = ComplexConvUNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)
