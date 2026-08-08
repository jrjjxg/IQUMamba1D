"""IQUResUNet1D - stage4 IQUMamba ablation without Mamba layers.

This is intentionally a thin ablation of ``IQUMamba1D``:
- keep the stage4 residual encoder blocks;
- keep the stage4 ASC skip decoder;
- remove only the encoder Mamba calls and parameters.
"""

from typing import List, Type

import torch
from torch import nn

from models.IQUMamba1D import BasicResBlock, ResidualMambaEncoder, UNetResDecoder


class ResidualConvEncoder(ResidualMambaEncoder):
    """Stage4 encoder with the Mamba layers removed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.mamba_layers

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUResUNet1D(nn.Module):
    """Stage42: stage4 IQUMamba with only encoder Mamba layers ablated."""

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
    ):
        super().__init__()
        self.encoder = ResidualConvEncoder(
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
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)
