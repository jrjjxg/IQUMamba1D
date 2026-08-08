"""Stage-4 IQUMamba with compressed global cross-scale skip attention.

Stage 300 is the unidirectional control for Stage 235. The Stage-4 encoder
and decoder remain unchanged; immediately before decoding, selected
higher-resolution encoder skips query a compact token bank built from the
deepest encoder feature.
"""

from __future__ import annotations

from typing import List, Sequence, Type

import torch
from torch import nn

from models.IQUBiMamba1D_CrossScaleAttention import (
    CompressedGlobalCrossAttention,
    MixturePhysicalEvidenceGate,
)
from models.IQUMamba1D import IQUMamba1D


class IQUMamba1D_CrossScaleAttention(IQUMamba1D):
    """Original Stage-4 backbone plus configurable cross-scale skip fusion."""

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
        cross_scale_evidence_gate: bool = False,
        cross_scale_evidence_hidden: int = 32,
        cross_scale_evidence_eps: float = 1e-6,
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
        )

        self.cross_scale_global_stage = int(cross_scale_global_stage)
        self.cross_scale_query_stages = tuple(
            int(stage) for stage in cross_scale_query_stages
        )
        if not 0 <= self.cross_scale_global_stage < int(n_stages):
            raise ValueError(
                f"Invalid global stage {self.cross_scale_global_stage}"
            )
        if not self.cross_scale_query_stages:
            raise ValueError("cross_scale_query_stages must not be empty")
        if len(set(self.cross_scale_query_stages)) != len(
            self.cross_scale_query_stages
        ):
            raise ValueError("cross_scale_query_stages must be unique")
        for stage in self.cross_scale_query_stages:
            if (
                not 0 <= stage < int(n_stages)
                or stage == self.cross_scale_global_stage
            ):
                raise ValueError(
                    f"Query stage {stage} must be valid and differ from global "
                    f"stage {self.cross_scale_global_stage}"
                )

        global_channels = int(
            features_per_stage[self.cross_scale_global_stage]
        )
        self.cross_scale_blocks = nn.ModuleDict(
            {
                str(stage): CompressedGlobalCrossAttention(
                    query_channels=int(features_per_stage[stage]),
                    global_channels=global_channels,
                    kv_tokens=cross_scale_kv_tokens,
                    num_heads=cross_scale_num_heads,
                    dropout=cross_scale_dropout,
                    residual_scale_init=cross_scale_residual_scale_init,
                )
                for stage in self.cross_scale_query_stages
            }
        )
        self.evidence_gate = (
            MixturePhysicalEvidenceGate(
                num_gates=len(self.cross_scale_query_stages),
                hidden_channels=cross_scale_evidence_hidden,
                eps=cross_scale_evidence_eps,
            )
            if bool(cross_scale_evidence_gate)
            else None
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        global_feature = skips[self.cross_scale_global_stage]
        gates = self.evidence_gate(x) if self.evidence_gate is not None else None

        enhanced_skips = list(skips)
        for gate_index, stage in enumerate(self.cross_scale_query_stages):
            gate = None if gates is None else gates[:, gate_index]
            enhanced_skips[stage] = self.cross_scale_blocks[str(stage)](
                enhanced_skips[stage],
                global_feature,
                gate=gate,
            )
        return self.decoder(enhanced_skips)
