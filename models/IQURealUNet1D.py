"""IQURealUNet1D - strict real-valued mirror of IQUComplexUNet1D.

Contract:
  input : (B, 2, T) where channels are I/Q
  output: (B, num_classes, T)

This baseline keeps the stage layout, residual blocks, layer norms, linear
upsampling, and skip processor used by IQUComplexUNet1D, but operates on
standard real-valued feature maps throughout.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list


def _interpolate_real(
    x: torch.Tensor,
    size: int | None = None,
    scale_factor: int | None = None,
) -> torch.Tensor:
    kwargs = {"mode": "linear", "align_corners": False}
    if size is None:
        return F.interpolate(x, scale_factor=scale_factor, **kwargs)
    return F.interpolate(x, size=size, **kwargs)


class RealLayerNorm(nn.Module):
    """LayerNorm over channel dim at each time step."""

    def __init__(self, channels: int, eps: float = 1e-8, affine: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2).contiguous()


class RealGlobalLayerNorm(nn.Module):
    """Global LN over (C, T) per sample."""

    def __init__(self, channels: int, eps: float = 1e-8, affine: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            y = y * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return y


class RealResBlock1D(nn.Module):
    """Real-valued residual block mirroring ComplexResBlock1D."""

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
        self.conv1 = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.norm1 = RealLayerNorm(output_channels, eps=eps, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope)
        self.conv2 = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=True,
        )
        self.norm2 = RealLayerNorm(output_channels, eps=eps, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope)
        if use_1x1conv or input_channels != output_channels or stride != 1:
            self.shortcut = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=True,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        y = y + self.shortcut(x)
        return self.act2(y)


class RealStemBlock1D(nn.Module):
    def __init__(self, output_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.block = RealResBlock1D(
            input_channels=2,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=1,
            use_1x1conv=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected IQ input (B, 2, T), got {tuple(x.shape)}")
        return self.block(x)


class RealUpsampleLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.proj = nn.Conv1d(input_channels, output_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(_interpolate_real(x, scale_factor=self.scale_factor))


class RealSkipConnectionProcessor(nn.Module):
    """Real-valued mirror of the complex skip alignment and fusion block."""

    def __init__(self, skip_channels: int, upsampled_channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.align_skip = nn.Sequential(
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1, padding=0, bias=True),
            RealGlobalLayerNorm(skip_channels, eps=eps, affine=True),
            nn.LeakyReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(skip_channels + upsampled_channels, skip_channels, kernel_size=1, padding=0, bias=True),
            RealGlobalLayerNorm(skip_channels, eps=eps, affine=True),
            nn.LeakyReLU(),
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, skip_features: torch.Tensor, upsampled_features: torch.Tensor) -> torch.Tensor:
        identity = skip_features
        skip = self.align_skip(skip_features)
        if upsampled_features.size(-1) != skip.size(-1):
            upsampled_features = _interpolate_real(upsampled_features, size=skip.size(-1))
        fused = self.fuse(torch.cat([skip, upsampled_features], dim=1))
        return self.residual_weight * fused + (1.0 - self.residual_weight) * identity


class RealConvEncoder(nn.Module):
    """Pure real-convolutional encoder matching the complex U-Net interface."""

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
        stem_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        del input_size, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, pool_type
        if input_channels != 2:
            raise ValueError(f"RealUNet1D expects input_channels=2, got {input_channels}")

        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = RealStemBlock1D(stem_channels, kernel_size=int(stem_kernel_size))

        stages = []
        input_real_channels = stem_channels
        for s in range(n_stages):
            blocks = [
                RealResBlock1D(
                    input_channels=input_real_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                )
            ]
            blocks.extend(
                RealResBlock1D(
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s][0],
                    stride=1,
                )
                for _ in range(n_blocks_per_stage[s] - 1)
            )
            stages.append(nn.Sequential(*blocks))
            input_real_channels = features_per_stage[s]

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


class RealConvUNetDecoder(nn.Module):
    """Pure real-convolutional decoder producing standard IQ source channels."""

    def __init__(self, encoder: RealConvEncoder, num_classes: int, n_conv_per_stage, deep_supervision: bool):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)

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
                RealUpsampleLayer(
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    scale_factor=stride_for_upsampling,
                )
            )
            skip_processors.append(
                RealSkipConnectionProcessor(
                    skip_channels=input_features_skip,
                    upsampled_channels=input_features_skip,
                )
            )

            blocks = [
                RealResBlock1D(
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=True,
                )
            ]
            blocks.extend(
                RealResBlock1D(
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                )
                for _ in range(n_conv_per_stage[s - 1] - 1)
            )
            stages.append(nn.Sequential(*blocks))
            seg_layers.append(nn.Conv1d(input_features_skip, num_classes, kernel_size=1, padding=0))

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
            x = torch.cat([x, processed_skip], dim=1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQURealUNet1D(nn.Module):
    """Strict real-valued control model for IQUComplexUNet1D."""

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
        stem_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = RealConvEncoder(
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
            stem_kernel_size=stem_kernel_size,
        )
        self.decoder = RealConvUNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)
