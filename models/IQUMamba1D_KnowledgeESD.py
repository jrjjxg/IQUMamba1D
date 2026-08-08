"""Knowledge-embedded IQUMamba with source-slot residual refinement.

This variant keeps the stage-4 IQUMamba encoder/decoder as the main separator.
The communication prior is only applied at the output:

1. Treat the decoder output as K complex source slots.
2. Refine each slot with a shared residual block conditioned on the observed
   mixture and the additive-mixture residual.
3. Project the refined slots onto the exact additive mixture constraint.

The model does not consume metadata at inference.  All conditioning is computed
from the observed IQ mixture and the current source estimates.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


class SourceSlotResidualRefiner(nn.Module):
    """Shared source-slot residual adapter conditioned by mixture consistency."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        residual_scale_init: float = 0.01,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1, got {hidden_channels}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.num_sources = int(num_sources)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        padding = kernel_size // 2

        # Per slot features:
        #   estimate IQ, mixture IQ, additive residual IQ, estimate energy,
        #   residual energy.
        in_channels = 8
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 3, kernel_size=1),
        )
        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _resize_mixture(self, mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if estimates.dim() != 3:
            raise ValueError(f"estimates must have shape (B, 2K, L), got {tuple(estimates.shape)}")
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}")
        if estimates.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} estimate channels, got {estimates.size(1)}"
            )

        batch, _, length = estimates.shape
        mixture = self._resize_mixture(mixture, length).to(dtype=estimates.dtype)
        base_sources = estimates.reshape(batch, self.num_sources, 2, length)
        mixture_residual = mixture - base_sources.sum(dim=1)

        mix_slots = mixture.unsqueeze(1).expand(-1, self.num_sources, -1, -1)
        residual_slots = mixture_residual.unsqueeze(1).expand(-1, self.num_sources, -1, -1)
        estimate_energy = base_sources.pow(2).sum(dim=2, keepdim=True).clamp_min(1e-12).sqrt()
        residual_energy = residual_slots.pow(2).sum(dim=2, keepdim=True).clamp_min(1e-12).sqrt()
        features = torch.cat(
            [base_sources, mix_slots, residual_slots, estimate_energy, residual_energy],
            dim=2,
        )

        flat_features = features.reshape(batch * self.num_sources, features.size(2), length)
        raw = self.net(flat_features).reshape(batch, self.num_sources, 3, length)
        gate = torch.sigmoid(raw[:, :, 2:3, :])
        correction = raw[:, :, 0:2, :] * gate
        scale = torch.tanh(self.residual_scale)
        refined = base_sources + scale * correction
        return refined.reshape(batch, 2 * self.num_sources, length)


class IQUMamba1D_KnowledgeESD(IQUMamba1D):
    """Stage-4 IQUMamba with blind source-slot refinement and mixture projection."""

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
        source_slot_hidden_channels: int = 32,
        source_slot_kernel_size: int = 7,
        source_slot_residual_scale_init: float = 0.01,
        source_slot_zero_init: bool = True,
        source_slot_refine_deep_supervision: bool = True,
        source_slot_apply_train: bool = True,
        source_slot_apply_eval: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_eps: float = 1e-8,
        mc_detach_weights: bool = False,
        mc_project_deep_supervision: bool = True,
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
        self.source_slot_refine_deep_supervision = bool(source_slot_refine_deep_supervision)
        self.source_slot_apply_train = bool(source_slot_apply_train)
        self.source_slot_apply_eval = bool(source_slot_apply_eval)
        self.mc_project_deep_supervision = bool(mc_project_deep_supervision)
        self.mc_apply_train = bool(mc_apply_train)
        self.mc_apply_eval = bool(mc_apply_eval)

        self.source_refiner = SourceSlotResidualRefiner(
            num_sources=self.num_sources,
            hidden_channels=source_slot_hidden_channels,
            kernel_size=source_slot_kernel_size,
            residual_scale_init=source_slot_residual_scale_init,
            zero_init=source_slot_zero_init,
        )
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=mc_eps,
            detach_weights=mc_detach_weights,
        )

    def _should_refine(self) -> bool:
        return self.source_slot_apply_train if self.training else self.source_slot_apply_eval

    def _should_project(self) -> bool:
        return self.mc_apply_train if self.training else self.mc_apply_eval

    def _refine_one(self, output: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if self._should_refine():
            output = self.source_refiner(output, mixture)
        if self._should_project():
            output = self.mc_projection(output, mixture)
        return output

    def _refine_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not isinstance(outputs, (list, tuple)):
            return self._refine_one(outputs, mixture)

        outputs = list(outputs)
        if self.source_slot_refine_deep_supervision or self.mc_project_deep_supervision:
            refined = []
            for index, output in enumerate(outputs):
                apply_refine = self.source_slot_refine_deep_supervision or index == 0
                apply_project = self.mc_project_deep_supervision or index == 0
                if apply_refine and self._should_refine():
                    output = self.source_refiner(output, mixture)
                if apply_project and self._should_project():
                    output = self.mc_projection(output, mixture)
                refined.append(output)
            return refined

        outputs[0] = self._refine_one(outputs[0], mixture)
        return outputs

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)
