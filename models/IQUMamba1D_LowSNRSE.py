"""Stage-4 IQUMamba with a low-SNR enhancement front-end.

This variant borrows the noisy speech separation idea of unifying enhancement
and separation: first estimate a cleaner mixture representation, then run the
unchanged separator.  Unlike prior cyclic-prior stages, this module uses no
frequency-peak estimate, modulation labels, symbol timing, or metadata.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_NoiseAwareMC import NoiseAwareMixtureConsistencyProjection1D


class LowSNRMixtureEnhancer1D(nn.Module):
    """Near-identity I/Q enhancement block for noisy mixtures."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 24,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"LowSNRMixtureEnhancer1D expects one I/Q mixture with 2 channels, got {input_channels}")

        hidden_channels = max(4, int(hidden_channels))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, input_channels, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        delta = torch.tanh(self.net(x))
        clean_mixture_hat = x + self.scale * delta
        noise_hat = x - clean_mixture_hat
        aux = {
            "clean_mixture_hat": clean_mixture_hat,
            "noise_hat": noise_hat,
            "enhancement_delta": delta,
        }
        return clean_mixture_hat, aux


class IQUMamba1D_LowSNRSE(nn.Module):
    """Low-SNR enhancement front-end plus the original stage-4 IQUMamba."""

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
        low_snr_se_hidden_channels: int = 24,
        low_snr_se_kernel_size: int = 9,
        low_snr_se_scale_init: float = 0.01,
        low_snr_se_zero_init: bool = True,
        low_snr_se_use_projection: bool = True,
        low_snr_se_project_during_train: bool = True,
        low_snr_se_project_during_eval: bool = True,
        low_snr_se_source_weight: float = 0.25,
        low_snr_se_noise_weight: float = 1.0,
        low_snr_se_return_aux: bool = False,
        low_snr_se_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")

        self.enhancer = LowSNRMixtureEnhancer1D(
            input_channels=input_channels,
            hidden_channels=low_snr_se_hidden_channels,
            kernel_size=low_snr_se_kernel_size,
            scale_init=low_snr_se_scale_init,
            zero_init=low_snr_se_zero_init,
        )
        self.backbone = IQUMamba1D(
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
        self.low_snr_se_use_projection = bool(low_snr_se_use_projection)
        self.low_snr_se_project_during_train = bool(low_snr_se_project_during_train)
        self.low_snr_se_project_during_eval = bool(low_snr_se_project_during_eval)
        self.low_snr_se_return_aux = bool(low_snr_se_return_aux)
        self.noise_projection = NoiseAwareMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            source_weight=low_snr_se_source_weight,
            noise_weight=low_snr_se_noise_weight,
            eps=low_snr_se_eps,
        )

    def _should_project(self) -> bool:
        if not self.low_snr_se_use_projection:
            return False
        return self.low_snr_se_project_during_train if self.training else self.low_snr_se_project_during_eval

    def _project_one(self, sources: torch.Tensor, mixture: torch.Tensor, noise_hat: torch.Tensor) -> torch.Tensor:
        if not self._should_project():
            return sources
        projected_sources, _ = self.noise_projection(sources, noise_hat, mixture)
        return projected_sources

    def _project_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
        noise_hat: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            return [self._project_one(out, mixture, noise_hat) for out in outputs]
        return self._project_one(outputs, mixture, noise_hat)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], tuple[Union[torch.Tensor, List[torch.Tensor]], dict[str, torch.Tensor]]]:
        clean_mixture_hat, aux = self.enhancer(x)
        sources = self.backbone(clean_mixture_hat)
        projected = self._project_outputs(sources, x, aux["noise_hat"])
        if self.low_snr_se_return_aux:
            return projected, aux
        return projected
