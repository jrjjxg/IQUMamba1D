"""IQUBiMamba1D_CSB_CMASC - CSB with communication-aware ASC.

CMASC replaces the generic adaptive skip processor in the U-Net decoder with a
complex mixture-consistent skip fusion block:
  - pseudo-complex encoder skips are phase-aligned to decoder features;
  - a feature-domain mixture residual controls how much skip detail returns;
  - the rest of the CSB encoder and BiMamba stages are unchanged.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUBiMamba1D_CSB import ResidualBiMambaEncoder_CSB
from models.IQUMamba1D import BasicResBlock, UpsampleLayer


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.size(-1) == ref.size(-1):
        return x
    return F.interpolate(x, size=ref.size(-1), mode="linear", align_corners=False)


class ComplexMixtureConsistentSkipProcessor(nn.Module):
    """Phase-aligned, residual-gated skip fusion for IQ separation."""

    def __init__(
        self,
        skip_channels: int,
        upsampled_channels: int,
        conv_op: Type[_ConvNd],
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict | None,
        nonlin: Type[nn.Module],
        nonlin_kwargs: dict | None,
        gate_hidden: int = 64,
        residual_scale_init: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        norm_op_kwargs = {} if norm_op_kwargs is None else norm_op_kwargs
        nonlin_kwargs = {} if nonlin_kwargs is None else nonlin_kwargs

        self.skip_channels = int(skip_channels)
        self.upsampled_channels = int(upsampled_channels)
        self.eps = float(eps)
        hidden = max(4, int(gate_hidden))

        self.feature_align = nn.Sequential(
            conv_op(skip_channels, skip_channels, kernel_size=1),
            norm_op(skip_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs),
        )
        self.decoder_proj = conv_op(upsampled_channels, skip_channels, kernel_size=1)

        self.phase_gate = nn.Sequential(
            conv_op(skip_channels, skip_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_gate = nn.Sequential(
            conv_op(skip_channels * 3, hidden, kernel_size=1),
            nonlin(**nonlin_kwargs),
            conv_op(hidden, skip_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.cross_interaction = nn.Sequential(
            conv_op(skip_channels + upsampled_channels, skip_channels, kernel_size=1),
            norm_op(skip_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs),
        )
        self.feature_refine = nn.Sequential(
            conv_op(skip_channels, skip_channels, kernel_size=3, padding=1),
            norm_op(skip_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs),
            conv_op(skip_channels, skip_channels, kernel_size=1),
            norm_op(skip_channels, **norm_op_kwargs),
        )
        self.residual_scale = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def _phase_align(self, skip_features: torch.Tensor, decoder_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if skip_features.size(1) % 2 == 1:
            skip_pair = F.pad(skip_features, (0, 0, 0, 1))
            dec_pair = F.pad(decoder_features, (0, 0, 0, 1))
        else:
            skip_pair = skip_features
            dec_pair = decoder_features

        sr = skip_pair[:, 0::2, :]
        si = skip_pair[:, 1::2, :]
        dr = dec_pair[:, 0::2, :]
        di = dec_pair[:, 1::2, :]

        dot = dr * sr + di * si
        cross = di * sr - dr * si
        denom = torch.sqrt(dot.square() + cross.square() + self.eps)
        cos_theta = dot / denom
        sin_theta = cross / denom

        aligned_r = sr * cos_theta - si * sin_theta
        aligned_i = sr * sin_theta + si * cos_theta
        aligned = torch.empty_like(skip_pair)
        aligned[:, 0::2, :] = aligned_r
        aligned[:, 1::2, :] = aligned_i
        aligned = aligned[:, : self.skip_channels, :]

        phase_map = cos_theta.repeat_interleave(2, dim=1)[:, : self.skip_channels, :]
        return aligned, phase_map

    def forward(self, skip_features: torch.Tensor, upsampled_features: torch.Tensor) -> torch.Tensor:
        identity = skip_features
        skip = self.feature_align(skip_features)
        upsampled = _resize_like(upsampled_features, skip)
        decoder_context = self.decoder_proj(upsampled)

        phase_aligned_skip, phase_map = self._phase_align(skip, decoder_context)
        phase_weight = self.phase_gate(phase_map)

        mixture_residual = torch.abs(phase_aligned_skip - decoder_context)
        residual_weight = self.residual_gate(
            torch.cat([phase_aligned_skip, decoder_context, mixture_residual], dim=1)
        )
        gated_skip = phase_aligned_skip * phase_weight * residual_weight

        fused = self.cross_interaction(torch.cat([gated_skip, upsampled], dim=1))
        refined = self.feature_refine(fused)
        return self.residual_scale * refined + (1 - self.residual_scale) * identity


class ComplexMixtureConsistentUNetResDecoder(nn.Module):
    """UNet decoder using ComplexMixtureConsistentSkipProcessor instead of ASC."""

    def __init__(
        self,
        encoder,
        num_classes: int,
        n_conv_per_stage: Union[int, List[int]],
        deep_supervision: bool,
        cmasc_gate_hidden: int = 64,
        cmasc_residual_scale_init: float = 0.5,
        cmasc_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
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
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode="linear" if encoder.conv_op == nn.Conv1d else "nearest",
                )
            )

            skip_processors.append(
                ComplexMixtureConsistentSkipProcessor(
                    skip_channels=input_features_skip,
                    upsampled_channels=input_features_skip,
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    gate_hidden=cmasc_gate_hidden,
                    residual_scale_init=cmasc_residual_scale_init,
                    eps=cmasc_eps,
                )
            )

            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=encoder.conv_op,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        input_channels=2 * input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                        padding=encoder.conv_pad_sizes[-(s + 1)][0],
                        stride=1,
                        use_1x1conv=True,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=encoder.conv_op,
                            input_channels=input_features_skip,
                            output_channels=input_features_skip,
                            kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                            stride=1,
                            conv_bias=encoder.conv_bias,
                            norm_op=encoder.norm_op,
                            norm_op_kwargs=encoder.norm_op_kwargs,
                            nonlin=encoder.nonlin,
                            nonlin_kwargs=encoder.nonlin_kwargs,
                        )
                        for _ in range(n_conv_per_stage[s - 1] - 1)
                    ],
                )
            )
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1))

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
            x = torch.cat((x, processed_skip), dim=1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQUBiMamba1D_CSB_CMASC(nn.Module):
    """CSB model with complex mixture-consistent adaptive skip connections."""

    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = nn.InstanceNorm1d,
        norm_op_kwargs: dict | None = {"eps": 1e-5, "affine": True},
        nonlin: Union[None, Type[torch.nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = {"inplace": True},
        deep_supervision: bool = False,
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
        cmasc_gate_hidden: int = 64,
        cmasc_residual_scale_init: float = 0.5,
        cmasc_eps: float = 1e-6,
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
        self.decoder = ComplexMixtureConsistentUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            cmasc_gate_hidden=cmasc_gate_hidden,
            cmasc_residual_scale_init=cmasc_residual_scale_init,
            cmasc_eps=cmasc_eps,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)
