"""UniRepLK receptive-field enhancement for the two cross-scale controls.

Stages 359 and 360 keep the four-level U-Net hierarchy unchanged.  They add
the exact Stage-310 UniRepLK residual blocks after encoder stages 0/1/2, then
apply the existing Stage-235/300 compressed global cross-scale attention
before decoding.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.IQUBiMamba1D_CrossScaleAttention import (
    IQUBiMamba1D_CrossScaleAttention,
)
from models.IQUMamba1D_CrossScaleAttention import (
    IQUMamba1D_CrossScaleAttention,
)
from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    build_recent_rf_operator,
)


class _UniRepLKCrossScaleMixin:
    """Shared Stage-310 insertion and Stage-235/300 fusion data flow."""

    def _init_unireplk(
        self,
        rf_apply_stages: Sequence[int] = (0, 1, 2),
        rf_residual_scale_init: float = 0.05,
        rf_large_kernel: int = 17,
        rf_ffn_factor: int = 4,
        rf_layer_scale: float = 1e-6,
    ) -> None:
        stages = tuple(int(stage) for stage in rf_apply_stages)
        if len(set(stages)) != len(stages):
            raise ValueError("rf_apply_stages must be unique")
        for stage in stages:
            if not 0 <= stage < len(self.encoder.stages):
                raise ValueError(
                    f"UniRepLK stage {stage} is outside the "
                    f"{len(self.encoder.stages)}-stage encoder"
                )

        operator_config = {
            "rf_large_kernel": int(rf_large_kernel),
            "rf_ffn_factor": int(rf_ffn_factor),
            "rf_layer_scale": float(rf_layer_scale),
        }
        self.rf_apply_stages = stages
        self.stage_rf = nn.ModuleDict(
            {
                str(stage): FeatureResidualAdapter(
                    int(self.encoder.output_channels[stage]),
                    build_recent_rf_operator(
                        "unireplk",
                        int(self.encoder.output_channels[stage]),
                        operator_config,
                    ),
                    float(rf_residual_scale_init),
                )
                for stage in stages
            }
        )

    def _forward_unireplk_cross_scale(self, x: torch.Tensor):
        mixture = x
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)

        skips = []
        for stage, (conv_stage, memory) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            x = memory(conv_stage(x))
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](x)
            skips.append(x)

        global_feature = skips[self.cross_scale_global_stage]
        gates = (
            self.evidence_gate(mixture)
            if self.evidence_gate is not None
            else None
        )
        enhanced_skips = list(skips)
        for gate_index, stage in enumerate(self.cross_scale_query_stages):
            gate = None if gates is None else gates[:, gate_index]
            enhanced_skips[stage] = self.cross_scale_blocks[str(stage)](
                enhanced_skips[stage],
                global_feature,
                gate=gate,
            )
        return self.decoder(enhanced_skips)


class IQUBiMamba1D_UniRepLKCrossScale(
    _UniRepLKCrossScaleMixin,
    IQUBiMamba1D_CrossScaleAttention,
):
    """Stage 359: Stage 235 BiMamba Cross-Scale plus Stage 310 UniRepLK."""

    def __init__(
        self,
        *args,
        rf_apply_stages: Sequence[int] = (0, 1, 2),
        rf_residual_scale_init: float = 0.05,
        rf_large_kernel: int = 17,
        rf_ffn_factor: int = 4,
        rf_layer_scale: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_unireplk(
            rf_apply_stages=rf_apply_stages,
            rf_residual_scale_init=rf_residual_scale_init,
            rf_large_kernel=rf_large_kernel,
            rf_ffn_factor=rf_ffn_factor,
            rf_layer_scale=rf_layer_scale,
        )

    def forward(self, x: torch.Tensor):
        return self._forward_unireplk_cross_scale(x)


class IQUMamba1D_UniRepLKCrossScale(
    _UniRepLKCrossScaleMixin,
    IQUMamba1D_CrossScaleAttention,
):
    """Stage 360: Stage 300 uni-Mamba Cross-Scale plus Stage 310 UniRepLK."""

    def __init__(
        self,
        *args,
        rf_apply_stages: Sequence[int] = (0, 1, 2),
        rf_residual_scale_init: float = 0.05,
        rf_large_kernel: int = 17,
        rf_ffn_factor: int = 4,
        rf_layer_scale: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_unireplk(
            rf_apply_stages=rf_apply_stages,
            rf_residual_scale_init=rf_residual_scale_init,
            rf_large_kernel=rf_large_kernel,
            rf_ffn_factor=rf_ffn_factor,
            rf_layer_scale=rf_layer_scale,
        )

    def forward(self, x: torch.Tensor):
        return self._forward_unireplk_cross_scale(x)
