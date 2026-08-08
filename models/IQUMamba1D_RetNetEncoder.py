from typing import List, Type

import torch.nn as nn

from models.IQUMamba1D_MambaReplacementBase import IQUMamba1D_MambaReplacementBase, Retention1D


class IQUMamba1D_RetNetEncoder(IQUMamba1D_MambaReplacementBase):
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
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        num_heads: int = 8,
        retention_kernel_size: int = 128,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            replacement_factory=lambda channels: Retention1D(
                channels=channels,
                num_heads=num_heads,
                retention_kernel_size=retention_kernel_size,
                dropout=dropout,
                residual_scale_init=residual_scale_init,
            ),
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
