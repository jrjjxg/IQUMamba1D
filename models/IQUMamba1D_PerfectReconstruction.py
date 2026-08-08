"""Communication-aware U-Net sampling and skip variants for IQ separation."""

from __future__ import annotations

import math
from typing import List, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUMamba1D import (
    BasicResBlock,
    MambaLayer,
    SkipConnectionProcessor,
)


class HaarPolyphaseDownsample1D(nn.Module):
    """Lossless even/odd polyphase analysis with an orthonormal Haar basis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) % 2:
            x = F.pad(x, (0, 1), mode="replicate")
        even = x[..., 0::2]
        odd = x[..., 1::2]
        scale = math.sqrt(0.5)
        low = (even + odd) * scale
        high = (even - odd) * scale
        return torch.cat((low, high), dim=1)


class HaarPolyphaseUpsample1D(nn.Module):
    """Exact inverse of :class:`HaarPolyphaseDownsample1D`."""

    def forward(self, x: torch.Tensor, target_length: int | None = None) -> torch.Tensor:
        if x.size(1) % 2:
            raise ValueError(f"Haar synthesis requires an even channel count, got {x.size(1)}")
        low, high = x.chunk(2, dim=1)
        scale = math.sqrt(0.5)
        even = (low + high) * scale
        odd = (low - high) * scale
        output = x.new_empty(x.size(0), low.size(1), 2 * x.size(-1))
        output[..., 0::2] = even
        output[..., 1::2] = odd
        if target_length is not None:
            output = output[..., : int(target_length)]
        return output


class PolyphaseTransition1D(nn.Module):
    """Haar analysis followed by an optional channel adapter."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        expanded = 2 * int(input_channels)
        self.analysis = HaarPolyphaseDownsample1D()
        self.adapter = (
            nn.Identity()
            if expanded == int(output_channels)
            else nn.Conv1d(expanded, int(output_channels), kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.analysis(x))


class PolyphaseUpsample1D(nn.Module):
    """Optional channel adapter followed by Haar synthesis."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        synthesis_channels = 2 * int(output_channels)
        self.adapter = (
            nn.Identity()
            if int(input_channels) == synthesis_channels
            else nn.Conv1d(int(input_channels), synthesis_channels, kernel_size=1, bias=False)
        )
        self.synthesis = HaarPolyphaseUpsample1D()

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        return self.synthesis(self.adapter(x), target_length=target_length)


class PerfectReconstructionMambaEncoder(nn.Module):
    """Stage-4 encoder with lossless polyphase transitions at strides of two."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes,
        strides,
        n_blocks_per_stage,
        conv_bias: bool = False,
        norm_op=None,
        norm_op_kwargs: dict | None = None,
        nonlin=None,
        nonlin_kwargs: dict | None = None,
        return_skips: bool = False,
        stem_channels: int | None = None,
        pool_type: str = "conv",
    ) -> None:
        super().__init__()
        del pool_type
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, value) for value in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, value) for value in strides]
        features = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else list(features_per_stage)
        blocks = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else list(n_blocks_per_stage)
        if any(int(stride[0]) not in (1, 2) for stride in strides):
            raise ValueError("Perfect-reconstruction encoder supports only strides 1 and 2")

        self.conv_pad_sizes = [[kernel // 2 for kernel in kernels] for kernels in kernel_sizes]
        stem_channels = features[0] if stem_channels is None else int(stem_channels)
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op, input_channels=input_channels, output_channels=stem_channels,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0], stride=1, nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs, use_1x1conv=True,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op, input_channels=stem_channels, output_channels=stem_channels,
                    kernel_size=kernel_sizes[0], stride=1, conv_bias=conv_bias,
                    norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(blocks[0] - 1)
            ],
        )

        current_channels = stem_channels
        current_length = int(input_size[0])
        stages, mamba_layers = [], []
        feature_lengths = []
        for stage_index in range(n_stages):
            stride = int(strides[stage_index][0])
            transition = nn.Identity()
            if stride == 2:
                transition = PolyphaseTransition1D(current_channels, features[stage_index])
                current_length = (current_length + 1) // 2
                block_input = features[stage_index]
            else:
                block_input = current_channels
            stage = nn.Sequential(
                transition,
                BasicResBlock(
                    conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    input_channels=block_input, output_channels=features[stage_index],
                    kernel_size=kernel_sizes[stage_index], padding=self.conv_pad_sizes[stage_index][0],
                    stride=1, use_1x1conv=True, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op, input_channels=features[stage_index],
                        output_channels=features[stage_index], kernel_size=kernel_sizes[stage_index],
                        stride=1, conv_bias=conv_bias, norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs, nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                    for _ in range(blocks[stage_index] - 1)
                ],
            )
            feature_lengths.append(current_length)
            use_channel_tokens = current_length <= features[stage_index]
            if bool(stage_index % 2) ^ bool(n_stages % 2):
                mamba_layers.append(MambaLayer(
                    dim=current_length if use_channel_tokens else features[stage_index],
                    channel_token=use_channel_tokens,
                ))
            else:
                mamba_layers.append(nn.Identity())
            stages.append(stage)
            current_channels = features[stage_index]

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.output_channels = features
        self.feature_lengths = feature_lengths
        self.strides = strides
        self.return_skips = bool(return_skips)
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        outputs = []
        for stage, mamba in zip(self.stages, self.mamba_layers):
            x = mamba(stage(x))
            outputs.append(x)
        return outputs if self.return_skips else outputs[-1]


class PerfectReconstructionDecoder(nn.Module):
    """Stage-4 decoder using Haar synthesis instead of interpolation."""

    def __init__(
        self,
        encoder: PerfectReconstructionMambaEncoder,
        num_classes: int,
        n_conv_per_stage,
        deep_supervision: bool,
        restrict_shallow_skip: bool = False,
        shallow_skip_init: float = 0.25,
        shallow_skip_drop_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.restrict_shallow_skip = bool(restrict_shallow_skip)
        self.shallow_skip_drop_probability = min(max(float(shallow_skip_drop_probability), 0.0), 1.0)
        stage_count = len(encoder.output_channels)
        blocks = [n_conv_per_stage] * (stage_count - 1) if isinstance(n_conv_per_stage, int) else list(n_conv_per_stage)
        shallow_skip_init = min(max(float(shallow_skip_init), 1e-4), 1.0 - 1e-4)
        self.shallow_skip_logit = nn.Parameter(torch.tensor(math.log(shallow_skip_init / (1.0 - shallow_skip_init))))

        upsamplers, skip_processors, stages, heads = [], [], [], []
        for decoder_index in range(1, stage_count):
            below = encoder.output_channels[-decoder_index]
            skip_channels = encoder.output_channels[-(decoder_index + 1)]
            upsamplers.append(PolyphaseUpsample1D(below, skip_channels))
            skip_processors.append(SkipConnectionProcessor(
                skip_channels=skip_channels, upsampled_channels=skip_channels,
                conv_op=encoder.conv_op, norm_op=encoder.norm_op,
                norm_op_kwargs=encoder.norm_op_kwargs, nonlin=encoder.nonlin,
                nonlin_kwargs=encoder.nonlin_kwargs,
            ))
            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op, norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs, nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs, input_channels=2 * skip_channels,
                    output_channels=skip_channels, kernel_size=encoder.kernel_sizes[-(decoder_index + 1)][0],
                    padding=encoder.conv_pad_sizes[-(decoder_index + 1)][0], stride=1,
                    use_1x1conv=True,
                ),
                *[
                    BasicBlockD(
                        conv_op=encoder.conv_op, input_channels=skip_channels,
                        output_channels=skip_channels, kernel_size=encoder.kernel_sizes[-(decoder_index + 1)][0],
                        stride=1, conv_bias=encoder.conv_bias, norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs, nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                    )
                    for _ in range(blocks[decoder_index - 1] - 1)
                ],
            ))
            heads.append(encoder.conv_op(skip_channels, num_classes, 1))

        self.upsample_layers = nn.ModuleList(upsamplers)
        self.skip_processors = nn.ModuleList(skip_processors)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(heads)

    @property
    def shallow_skip_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.shallow_skip_logit)

    def _restricted_skip(self, skip: torch.Tensor) -> torch.Tensor:
        scaled = self.shallow_skip_scale * skip
        if not self.training or self.shallow_skip_drop_probability <= 0:
            return scaled
        keep_probability = 1.0 - self.shallow_skip_drop_probability
        if keep_probability <= 0:
            return torch.zeros_like(scaled)
        keep = torch.empty(skip.size(0), 1, 1, device=skip.device, dtype=skip.dtype).bernoulli_(keep_probability)
        return scaled * keep / keep_probability

    def forward(self, skips):
        x = skips[-1]
        outputs = []
        final_decoder_index = len(self.stages) - 1
        for index, (upsample, processor, stage, head) in enumerate(zip(
            self.upsample_layers, self.skip_processors, self.stages, self.seg_layers
        )):
            skip = skips[-(index + 2)]
            x = upsample(x, target_length=skip.size(-1))
            if self.restrict_shallow_skip and index == final_decoder_index:
                processed_skip = self._restricted_skip(skip)
            else:
                processed_skip = processor(skip, x)
            x = stage(torch.cat((x, processed_skip), dim=1))
            outputs.append(head(x))
        return outputs[::-1] if self.deep_supervision else outputs[-1]


class IQUMamba1D_PerfectReconstruction(nn.Module):
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
        training_only_deep_supervision: bool = True,
        restrict_shallow_skip: bool = False,
        shallow_skip_init: float = 0.25,
        shallow_skip_drop_probability: float = 0.0,
    ) -> None:
        super().__init__()
        norm_op_kwargs = {"eps": 1e-5, "affine": True} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {"inplace": True} if nonlin_kwargs is None else nonlin_kwargs
        self.training_only_deep_supervision = bool(training_only_deep_supervision)
        self.encoder = PerfectReconstructionMambaEncoder(
            input_size=(input_size,), input_channels=input_channels, n_stages=n_stages,
            features_per_stage=features_per_stage, conv_op=conv_op,
            kernel_sizes=[[value] for value in kernel_sizes], strides=[[value] for value in strides],
            n_blocks_per_stage=n_conv_per_stage, conv_bias=conv_bias, norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
        )
        self.decoder = PerfectReconstructionDecoder(
            encoder=self.encoder, num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder, deep_supervision=deep_supervision,
            restrict_shallow_skip=restrict_shallow_skip, shallow_skip_init=shallow_skip_init,
            shallow_skip_drop_probability=shallow_skip_drop_probability,
        )

    def forward(self, x: torch.Tensor):
        outputs = self.decoder(self.encoder(x))
        if not self.training and self.training_only_deep_supervision and isinstance(outputs, list):
            return outputs[0]
        return outputs


class IQUMamba1D_RestrictedShallowSkip(IQUMamba1D_PerfectReconstruction):
    """Stage 221: PR U-Net with a bounded, stochastically dropped shallow skip."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["restrict_shallow_skip"] = True
        super().__init__(*args, **kwargs)

