"""IQUMamba with learned complex feature masks.

This variant keeps the original IQUMamba separator path for mask prediction,
but changes the output parameterization:

    mixture IQ -> learned complex feature bank -> complex masks -> complex decoder

The mask is applied to learned complex features, not directly to raw IQ samples.
No modulation label, symbol timing, cyclic frequency, or receiver-side metadata
is used.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.ctdcrn import ComplexConv1d
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


class ComplexFeatureMaskHead(nn.Module):
    """Apply source complex masks on learned complex features and decode to IQ."""

    def __init__(
        self,
        num_sources: int,
        feature_channels: int = 8,
        kernel_size: int = 9,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_logit_scale_init: float = 0.05,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if feature_channels < 1:
            raise ValueError(f"feature_channels must be >= 1, got {feature_channels}")
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")

        self.num_sources = int(num_sources)
        self.feature_channels = int(feature_channels)
        self.mask_bound = float(mask_bound)
        self.mask_sum_constraint = bool(mask_sum_constraint)
        self.logit_scale = nn.Parameter(torch.tensor(float(mask_logit_scale_init)))

        self.feature_encoder = ComplexConv1d(
            1,
            self.feature_channels,
            kernel_size=int(kernel_size),
            padding="same",
            bias=True,
        )
        self.feature_decoder = ComplexConv1d(
            self.feature_channels,
            1,
            kernel_size=int(kernel_size),
            padding="same",
            bias=True,
        )

        if identity_init:
            self._init_identity_feature_bank()

    def _init_identity_feature_bank(self) -> None:
        """Initialize channel 0 as an IQ pass-through and other channels as spare."""
        with torch.no_grad():
            for module in (self.feature_encoder, self.feature_decoder):
                module.conv_re.weight.zero_()
                module.conv_im.weight.zero_()
                if module.bias_re is not None:
                    module.bias_re.zero_()
                    module.bias_im.zero_()

            enc_center = self.feature_encoder.conv_re.weight.size(-1) // 2
            dec_center = self.feature_decoder.conv_re.weight.size(-1) // 2
            self.feature_encoder.conv_re.weight[0, 0, enc_center] = 1.0
            self.feature_decoder.conv_re.weight[0, 0, dec_center] = 1.0

    def _resize_mask_logits(self, mask_logits: torch.Tensor, target_length: int) -> torch.Tensor:
        if mask_logits.size(-1) == target_length:
            return mask_logits
        return F.interpolate(mask_logits, size=target_length, mode="linear", align_corners=False)

    def encode_features(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}")
        return self.feature_encoder(mixture.unsqueeze(2))

    def features_to_real_channels(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4 or features.size(1) != 2 or features.size(2) != self.feature_channels:
            raise ValueError(f"Expected (B, 2, C, L) features, got {tuple(features.shape)}")
        batch, _, channels, length = features.shape
        return features.reshape(batch, 2 * channels, length)

    def _bounded_feature_masks(self, mask_logits: torch.Tensor, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if mask_logits.dim() != 3:
            raise ValueError(f"mask_logits must have shape (B, 2*K*C, L), got {tuple(mask_logits.shape)}")
        expected_channels = 2 * self.num_sources * self.feature_channels
        if mask_logits.size(1) != expected_channels:
            raise ValueError(f"Expected {expected_channels} mask channels, got {mask_logits.size(1)}")

        mask_logits = self._resize_mask_logits(mask_logits, length)
        batch = mask_logits.size(0)
        masks = mask_logits.reshape(batch, self.num_sources, 2, self.feature_channels, length)
        masks = masks * self.logit_scale
        if self.mask_bound > 0:
            masks = torch.tanh(masks) * self.mask_bound

        mask_real = masks[:, :, 0, :, :]
        mask_imag = masks[:, :, 1, :, :]
        if self.mask_sum_constraint:
            sum_real = mask_real.sum(dim=1, keepdim=True)
            sum_imag = mask_imag.sum(dim=1, keepdim=True)
            mask_real = mask_real - (sum_real - 1.0) / self.num_sources
            mask_imag = mask_imag - sum_imag / self.num_sources
        return mask_real, mask_imag

    def forward(
        self,
        mask_logits: torch.Tensor,
        mixture: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            features = self.encode_features(mixture)
        features = features.to(dtype=mask_logits.dtype)
        batch, _, _, length = features.shape
        mask_real, mask_imag = self._bounded_feature_masks(mask_logits, length)

        feat_real = features[:, 0, :, :].unsqueeze(1)
        feat_imag = features[:, 1, :, :].unsqueeze(1)
        source_real = mask_real * feat_real - mask_imag * feat_imag
        source_imag = mask_real * feat_imag + mask_imag * feat_real

        source_features = torch.stack([source_real, source_imag], dim=2)
        source_features = source_features.reshape(
            batch * self.num_sources,
            2,
            self.feature_channels,
            length,
        )
        decoded = self.feature_decoder(source_features).squeeze(2)
        decoded = decoded.reshape(batch, self.num_sources, 2, length)
        return decoded.reshape(batch, 2 * self.num_sources, length)


class IQUMamba1D_FeatureComplexMask(nn.Module):
    """Predict complex masks on a learned complex feature bank."""

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
        feature_mask_channels: int = 8,
        feature_mask_kernel_size: int = 9,
        feature_mask_bound: float = 4.0,
        feature_mask_sum_constraint: bool = True,
        feature_mask_apply_projection: bool = True,
        feature_mask_project_deep_supervision: bool = True,
        feature_mask_logit_scale_init: float = 0.05,
        feature_mask_identity_init: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_eps: float = 1e-8,
        mc_detach_weights: bool = False,
        mc_apply_train: bool = True,
        mc_apply_eval: bool = True,
    ) -> None:
        super().__init__()
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")
        self.num_sources = num_classes // 2
        self.feature_mask_apply_projection = bool(feature_mask_apply_projection)
        self.feature_mask_project_deep_supervision = bool(feature_mask_project_deep_supervision)
        self.mc_apply_train = bool(mc_apply_train)
        self.mc_apply_eval = bool(mc_apply_eval)

        self.mask_net = IQUMamba1D(
            input_size=input_size,
            input_channels=2 * int(feature_mask_channels),
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=2 * self.num_sources * int(feature_mask_channels),
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.feature_mask_head = ComplexFeatureMaskHead(
            num_sources=self.num_sources,
            feature_channels=int(feature_mask_channels),
            kernel_size=int(feature_mask_kernel_size),
            mask_bound=float(feature_mask_bound),
            mask_sum_constraint=bool(feature_mask_sum_constraint),
            mask_logit_scale_init=float(feature_mask_logit_scale_init),
            identity_init=bool(feature_mask_identity_init),
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
        if not self.feature_mask_apply_projection:
            return False
        return self.mc_apply_train if self.training else self.mc_apply_eval

    def _project(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if self._should_project():
            return self.mc_projection(estimates, mixture)
        return estimates

    def _convert_outputs(
        self,
        mask_outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
        features: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(mask_outputs, (list, tuple)):
            converted = [self.feature_mask_head(mask_logits, mixture, features) for mask_logits in mask_outputs]
            if not self._should_project():
                return converted
            if self.feature_mask_project_deep_supervision:
                return [self.mc_projection(out, mixture) for out in converted]
            converted[0] = self.mc_projection(converted[0], mixture)
            return converted

        estimates = self.feature_mask_head(mask_outputs, mixture, features)
        return self._project(estimates, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        features = self.feature_mask_head.encode_features(x)
        mask_inputs = self.feature_mask_head.features_to_real_channels(features)
        mask_outputs = self.mask_net(mask_inputs)
        return self._convert_outputs(mask_outputs, x, features)
