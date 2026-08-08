"""Stage 235 wrapped with the Stage-79 estimated Cyclo-FRESH adapter."""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUBiMamba1D_CrossScaleAttention import IQUBiMamba1D_CrossScaleAttention
from models.IQUMamba1D_EstimatedCycloFRESH import EstimatedCycloFRESHAdapter1D


class IQUBiMamba1D_CrossScaleEstimatedCycloFRESH(nn.Module):
    """Stage-235 separator with a metadata-free cyclic-frequency front end."""

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
        cross_scale_query_stages: Sequence[int] = (2,),
        cross_scale_global_stage: int = 3,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 0.01,
        estimated_cyclofresh_min_freq: float = 1.0 / 64.0,
        estimated_cyclofresh_max_freq: float = 1.0 / 8.0,
        estimated_cyclofresh_default_freq: float = 1.0 / 32.0,
        estimated_cyclofresh_momentum: float = 0.05,
        estimated_cyclofresh_hidden_channels: int = 8,
        estimated_cyclofresh_kernel_size: int = 9,
        estimated_cyclofresh_scale_init: float = 0.01,
        estimated_cyclofresh_gate_hidden: int = 8,
        estimated_cyclofresh_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.estimated_cyclofresh_adapter = EstimatedCycloFRESHAdapter1D(
            input_channels=input_channels,
            min_freq=estimated_cyclofresh_min_freq,
            max_freq=estimated_cyclofresh_max_freq,
            default_freq=estimated_cyclofresh_default_freq,
            momentum=estimated_cyclofresh_momentum,
            hidden_channels=estimated_cyclofresh_hidden_channels,
            kernel_size=estimated_cyclofresh_kernel_size,
            scale_init=estimated_cyclofresh_scale_init,
            gate_hidden=estimated_cyclofresh_gate_hidden,
            zero_init=estimated_cyclofresh_zero_init,
        )
        self.backbone = IQUBiMamba1D_CrossScaleAttention(
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
            cross_scale_query_stages=cross_scale_query_stages,
            cross_scale_global_stage=cross_scale_global_stage,
            cross_scale_kv_tokens=cross_scale_kv_tokens,
            cross_scale_num_heads=cross_scale_num_heads,
            cross_scale_dropout=cross_scale_dropout,
            cross_scale_residual_scale_init=cross_scale_residual_scale_init,
            cross_scale_evidence_gate=False,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(self.estimated_cyclofresh_adapter(x))
