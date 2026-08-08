"""IQUBiMamba1D_MC - BiMamba separator with exact mixture-consistency projection.

This variant keeps the original IQUBiMamba1D encoder/decoder unchanged and
adds a post-decoder projection layer that enforces:

    sum_k s_hat_k = x_mix

for every sample and time step.

Why this design:
  - It preserves the representational capacity of the original separator.
  - It injects a hard physical prior at the output instead of only as a loss.
  - The correction is distributed with confidence-aware weights derived from
    the estimated source energies, which is less destructive than uniform
    residual sharing when one source clearly dominates a time region.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import torch
from torch import nn

from models.IQUBiMamba1D import IQUBiMamba1D
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


class IQUBiMamba1D_MC(IQUBiMamba1D):
    """IQUBiMamba1D with a post-decoder mixture-consistency projection layer."""

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
        mc_weight_mode: str = "energy",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 1e-3,
        mc_eps: float = 1e-8,
        mc_detach_weights: bool = False,
        mc_project_deep_supervision: bool = True,
        mc_apply_train: bool = True,
        mc_apply_eval: bool = True,
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
        if num_classes % 2 != 0:
            raise ValueError(
                f"num_classes must be even for I/Q source pairs, got {num_classes}"
            )

        self.num_sources = num_classes // 2
        self.mc_project_deep_supervision = bool(mc_project_deep_supervision)
        self.mc_apply_train = bool(mc_apply_train)
        self.mc_apply_eval = bool(mc_apply_eval)
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=mc_eps,
            detach_weights=mc_detach_weights,
        )

    def _should_project(self) -> bool:
        return self.mc_apply_train if self.training else self.mc_apply_eval

    def _project_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            if not self.mc_project_deep_supervision:
                outputs = list(outputs)
                # outputs[0] is the full-resolution output (UNetResDecoder
                # returns seg_outputs[::-1], so index 0 = highest resolution).
                outputs[0] = self.mc_projection(outputs[0], mixture)
                return outputs
            return [self.mc_projection(out, mixture) for out in outputs]
        return self.mc_projection(outputs, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = super().forward(x)
        if not self._should_project():
            return outputs
        return self._project_outputs(outputs, x)
