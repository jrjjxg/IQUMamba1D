"""IQUMamba1D with Mamba layers in the decoder path."""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import (
    BasicBlockD,
    BasicResBlock,
    MambaLayer,
    ResidualMambaEncoder,
    SkipConnectionProcessor,
    UNetResDecoder,
    UpsampleLayer,
)


class UNetResDecoderMamba(UNetResDecoder):
    """U-Net decoder with optional Mamba refinement after decoder blocks."""

    def __init__(
        self,
        encoder: ResidualMambaEncoder,
        num_classes: int,
        n_conv_per_stage: Union[int, List[int]],
        deep_supervision: bool,
        decoder_mamba_stages: Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage,
            deep_supervision=deep_supervision,
        )

        active_stages = self._normalize_decoder_mamba_stages(
            decoder_mamba_stages,
            num_decoder_stages=len(self.stages),
        )
        self.decoder_mamba_stages = tuple(sorted(active_stages))

        mamba_layers = []
        for decoder_idx in range(len(self.stages)):
            if decoder_idx in active_stages:
                output_channels = encoder.output_channels[-(decoder_idx + 2)]
                mamba_layers.append(MambaLayer(dim=output_channels, channel_token=False))
            else:
                mamba_layers.append(nn.Identity())
        self.decoder_mamba_layers = nn.ModuleList(mamba_layers)

    @staticmethod
    def _normalize_decoder_mamba_stages(
        stages: Sequence[int] | None,
        num_decoder_stages: int,
    ) -> set[int]:
        if stages is None:
            return {0}
        active = {int(stage) for stage in stages}
        invalid = [stage for stage in active if stage < 0 or stage >= num_decoder_stages]
        if invalid:
            raise ValueError(
                f"decoder_mamba_stages contains invalid stage(s) {invalid}; "
                f"valid range is [0, {num_decoder_stages - 1}]"
            )
        return active

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            processed_skip = self.skip_processors[s](skips[-(s + 2)], x)
            x = torch.cat((x, processed_skip), 1)
            x = self.stages[s](x)
            x = self.decoder_mamba_layers[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQUMamba1D_DecoderMamba(nn.Module):
    """IQUMamba1D baseline with selected decoder stages augmented by Mamba."""

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
        decoder_mamba_stages: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ResidualMambaEncoder(
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
        )
        self.decoder = UNetResDecoderMamba(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            decoder_mamba_stages=decoder_mamba_stages,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        return self.decoder(skips)
