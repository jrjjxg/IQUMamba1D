"""Noise-aware mixture consistency for IQUMamba.

The dataset uses noisy mixtures but clean source targets.  Hard projection
``sum_k source_k = mixture`` therefore pushes residual noise into the clean
sources.  This variant adds a noise head and projects onto:

    sum_k source_k + noise = mixture

Only the projected clean sources are returned, so existing PIT losses and BER
evaluation remain unchanged.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D


class NoiseAwareMixtureConsistencyProjection1D(nn.Module):
    """Project clean sources and residual noise onto noisy-mixture consistency."""

    def __init__(
        self,
        num_sources: int,
        source_weight: float = 0.25,
        noise_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if source_weight < 0:
            raise ValueError(f"source_weight must be >= 0, got {source_weight}")
        if noise_weight < 0:
            raise ValueError(f"noise_weight must be >= 0, got {noise_weight}")
        if source_weight == 0 and noise_weight == 0:
            raise ValueError("At least one projection weight must be positive")
        self.num_sources = int(num_sources)
        self.source_weight = float(source_weight)
        self.noise_weight = float(noise_weight)
        self.eps = float(eps)

    def _resize_mixture(self, mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)

    def forward(
        self,
        sources: torch.Tensor,
        noise_est: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sources.dim() != 3:
            raise ValueError(f"sources must have shape (B, 2K, L), got {tuple(sources.shape)}")
        if sources.size(1) != 2 * self.num_sources:
            raise ValueError(f"Expected {2 * self.num_sources} source channels, got {sources.size(1)}")
        if noise_est.dim() != 3 or noise_est.size(1) != 2:
            raise ValueError(f"noise_est must have shape (B, 2, L), got {tuple(noise_est.shape)}")
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}")

        target_length = sources.size(-1)
        mixture = self._resize_mixture(mixture, target_length).to(dtype=sources.dtype)
        if noise_est.size(-1) != target_length:
            noise_est = F.interpolate(noise_est, size=target_length, mode="linear", align_corners=False)

        b = sources.size(0)
        source_view = sources.reshape(b, self.num_sources, 2, target_length)
        source_sum = source_view.sum(dim=1)
        residual = mixture - source_sum - noise_est

        denom = self.num_sources * self.source_weight + self.noise_weight + self.eps
        source_residual = (self.source_weight / denom) * residual
        noise_residual = (self.noise_weight / denom) * residual
        projected_sources = source_view + source_residual.unsqueeze(1)
        projected_noise = noise_est + noise_residual
        return projected_sources.reshape(b, 2 * self.num_sources, target_length), projected_noise


class NoiseResidualHead(nn.Module):
    """Predict residual mixture noise from IQUMamba's clean-source estimates."""

    def __init__(
        self,
        num_source_channels: int,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        hidden_channels = max(4, int(hidden_channels))
        in_channels = int(num_source_channels) + 4
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, groups=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, sources: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.size(-1) != sources.size(-1):
            mixture = F.interpolate(mixture, size=sources.size(-1), mode="linear", align_corners=False)
        source_sum_i = sources[:, 0::2, :].sum(dim=1, keepdim=True)
        source_sum_q = sources[:, 1::2, :].sum(dim=1, keepdim=True)
        source_sum = torch.cat([source_sum_i, source_sum_q], dim=1)
        residual_hint = mixture - source_sum
        return self.net(torch.cat([sources, mixture, residual_hint], dim=1))


class IQUMamba1D_NoiseAwareMC(IQUMamba1D):
    """IQUMamba with residual-noise head and noise-aware consistency projection."""

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
        noise_mc_apply_projection: bool = True,
        noise_mc_project_during_train: bool = True,
        noise_mc_project_during_eval: bool = True,
        noise_mc_source_weight: float = 0.25,
        noise_mc_noise_weight: float = 1.0,
        noise_head_hidden_channels: int = 32,
        noise_head_kernel_size: int = 7,
        noise_head_zero_init: bool = True,
        noise_mc_eps: float = 1e-8,
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
        self.noise_mc_apply_projection = bool(noise_mc_apply_projection)
        self.noise_mc_project_during_train = bool(noise_mc_project_during_train)
        self.noise_mc_project_during_eval = bool(noise_mc_project_during_eval)
        self.noise_head = NoiseResidualHead(
            num_source_channels=num_classes,
            hidden_channels=noise_head_hidden_channels,
            kernel_size=noise_head_kernel_size,
            zero_init=noise_head_zero_init,
        )
        self.noise_projection = NoiseAwareMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            source_weight=noise_mc_source_weight,
            noise_weight=noise_mc_noise_weight,
            eps=noise_mc_eps,
        )

    def _should_project(self) -> bool:
        if not self.noise_mc_apply_projection:
            return False
        return self.noise_mc_project_during_train if self.training else self.noise_mc_project_during_eval

    def _project_one(self, sources: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        noise_est = self.noise_head(sources, mixture)
        if not self._should_project():
            return sources
        projected_sources, _ = self.noise_projection(sources, noise_est, mixture)
        return projected_sources

    def _project_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            return [self._project_one(out, mixture) for out in outputs]
        return self._project_one(outputs, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        sources = super().forward(x)
        return self._project_outputs(sources, x)
