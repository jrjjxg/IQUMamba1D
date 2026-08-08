"""IQUBiMamba1D_CSB_URIC - CSB BiMamba with an unrolled residual IC head."""

from __future__ import annotations

from typing import List, Sequence, Union

import torch
from torch import nn

from models.IQUBiMamba1D_CSB import IQUBiMamba1D_CSB
from models.unrolled_residual_ic import UnrolledResidualInterferenceCancellationHead


class IQUBiMamba1D_CSB_URIC(IQUBiMamba1D_CSB):
    """Complex-stem/complex-bottleneck BiMamba followed by URIC refinement."""

    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: type[nn.Conv1d],
        kernel_sizes: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool = True,
        norm_op: type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
        ric_num_steps: int = 3,
        ric_hidden_channels: int = 48,
        ric_kernel_size: int = 7,
        ric_dropout: float = 0.0,
        ric_tied_steps: bool = True,
        ric_step_init: float = 0.5,
        ric_return_intermediate: bool = False,
        ric_update_block_type: str = "conv",
        ric_dilations: tuple[int, ...] = (1, 2, 4),
        ric_num_heads: int = 4,
        ric_attention_stride: int = 1,
        ric_ffn_multiplier: int = 2,
    ) -> None:
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
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            complex_stem_hidden_channels=complex_stem_hidden_channels,
            complex_stem_kernel_size=complex_stem_kernel_size,
            complex_bottleneck_hidden_channels=complex_bottleneck_hidden_channels,
            complex_bottleneck_num_blocks=complex_bottleneck_num_blocks,
            complex_bottleneck_kernel_size=complex_bottleneck_kernel_size,
            complex_bottleneck_dilation_growth=complex_bottleneck_dilation_growth,
            complex_bottleneck_zero_init=complex_bottleneck_zero_init,
        )
        if num_classes % 2 != 0:
            raise ValueError(
                f"num_classes must be even for I/Q source pairs, got {num_classes}"
            )
        self.num_sources = num_classes // 2
        self.ric_return_intermediate = bool(ric_return_intermediate)
        self.ric_head = UnrolledResidualInterferenceCancellationHead(
            num_sources=self.num_sources,
            num_steps=ric_num_steps,
            hidden_channels=ric_hidden_channels,
            kernel_size=ric_kernel_size,
            dropout=ric_dropout,
            tied_steps=ric_tied_steps,
            step_init=ric_step_init,
            update_block_type=ric_update_block_type,
            dilations=ric_dilations,
            num_heads=ric_num_heads,
            attention_stride=ric_attention_stride,
            ffn_multiplier=ric_ffn_multiplier,
        )

    def _refine_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            refined = self.ric_head(
                outputs[-1],
                mixture,
                return_intermediate=self.training and self.ric_return_intermediate,
            )
            if isinstance(refined, tuple):
                outputs[-1] = refined[0]
                return outputs[-1], refined[1]
            outputs[-1] = refined
            return outputs
        return self.ric_head(
            outputs,
            mixture,
            return_intermediate=self.training and self.ric_return_intermediate,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)
