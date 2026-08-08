"""Complex-mask IQUMamba with structured mixture-consistent output.

The decoder predicts complex masks instead of free waveform samples.  Source
estimates are produced by complex multiplication with the input mixture, then
optionally passed through the existing exact mixture-consistency projection.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


class ComplexMaskMixtureHead(nn.Module):
    """Convert decoder logits into complex masks and apply them to the mixture."""

    def __init__(
        self,
        num_sources: int,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_logit_scale_init: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        self.num_sources = int(num_sources)
        self.mask_bound = float(mask_bound)
        self.mask_sum_constraint = bool(mask_sum_constraint)
        self.eps = float(eps)
        self.logit_scale = nn.Parameter(torch.tensor(float(mask_logit_scale_init)))

    def _resize_mixture(self, mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)

    def _bounded_masks(self, mask_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mask_logits.dim() != 3:
            raise ValueError(f"mask_logits must have shape (B, 2K, L), got {tuple(mask_logits.shape)}")
        if mask_logits.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} mask channels, got {mask_logits.size(1)}"
            )

        batch, _, length = mask_logits.shape
        masks = mask_logits.reshape(batch, self.num_sources, 2, length)
        masks = masks * self.logit_scale
        if self.mask_bound > 0:
            masks = torch.tanh(masks) * self.mask_bound

        mask_real = masks[:, :, 0, :]
        mask_imag = masks[:, :, 1, :]
        if self.mask_sum_constraint:
            sum_real = mask_real.sum(dim=1, keepdim=True)
            sum_imag = mask_imag.sum(dim=1, keepdim=True)
            mask_real = mask_real - (sum_real - 1.0) / self.num_sources
            mask_imag = mask_imag - sum_imag / self.num_sources
        return mask_real, mask_imag

    def forward(self, mask_logits: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}")

        target_length = mask_logits.size(-1)
        mixture = self._resize_mixture(mixture, target_length).to(dtype=mask_logits.dtype)
        mask_real, mask_imag = self._bounded_masks(mask_logits)

        mix_real = mixture[:, 0:1, :]
        mix_imag = mixture[:, 1:2, :]
        source_real = mask_real * mix_real - mask_imag * mix_imag
        source_imag = mask_real * mix_imag + mask_imag * mix_real
        sources = torch.stack([source_real, source_imag], dim=2)
        return sources.reshape(mask_logits.size(0), 2 * self.num_sources, target_length)


class IQUMamba1D_ComplexMaskMC(IQUMamba1D):
    """IQUMamba whose final output is constrained complex-mask source estimates."""

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
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_apply_projection: bool = True,
        mask_project_deep_supervision: bool = True,
        mask_logit_scale_init: float = 0.1,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_eps: float = 1e-8,
        mc_detach_weights: bool = False,
        mc_apply_train: bool = True,
        mc_apply_eval: bool = True,
    ) -> None:
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")
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
        self.num_sources = num_classes // 2
        self.mask_apply_projection = bool(mask_apply_projection)
        self.mask_project_deep_supervision = bool(mask_project_deep_supervision)
        self.mc_apply_train = bool(mc_apply_train)
        self.mc_apply_eval = bool(mc_apply_eval)
        self.mask_head = ComplexMaskMixtureHead(
            num_sources=self.num_sources,
            mask_bound=mask_bound,
            mask_sum_constraint=mask_sum_constraint,
            mask_logit_scale_init=mask_logit_scale_init,
            eps=mc_eps,
        )
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=mc_eps,
            detach_weights=mc_detach_weights,
        )

    def _should_project(self) -> bool:
        if not self.mask_apply_projection:
            return False
        return self.mc_apply_train if self.training else self.mc_apply_eval

    def _apply_mask_and_projection(
        self,
        mask_logits: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        estimates = self.mask_head(mask_logits, mixture)
        if self._should_project():
            estimates = self.mc_projection(estimates, mixture)
        return estimates

    def _convert_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            converted = [self.mask_head(out, mixture) for out in outputs]
            if not self._should_project():
                return converted
            if self.mask_project_deep_supervision:
                return [self.mc_projection(out, mixture) for out in converted]
            converted[0] = self.mc_projection(converted[0], mixture)
            return converted
        return self._apply_mask_and_projection(outputs, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        mask_logits = super().forward(x)
        return self._convert_outputs(mask_logits, x)
