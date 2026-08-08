"""Neural Wiener low-SNR front-end for stage-4 IQUMamba.

The front-end predicts local signal/noise power proxies and applies a
Wiener-like gain before the unchanged separator. It is initialized close to
identity so it cannot aggressively erase weak sources at the start of training.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_NoiseAwareMC import NoiseAwareMixtureConsistencyProjection1D


class NeuralWienerMixtureEnhancer1D(nn.Module):
    """Predict signal/noise powers and apply a Wiener gain to raw I/Q mixtures."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 16,
        kernel_size: int = 9,
        signal_bias_init: float = 3.0,
        noise_bias_init: float = -3.0,
        log_power_clip: float = 8.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"NeuralWienerMixtureEnhancer1D expects one I/Q mixture with 2 channels, got {input_channels}")

        hidden_channels = max(4, int(hidden_channels))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        self.eps = float(eps)
        self.log_power_clip = float(log_power_clip)
        self.power_net = nn.Sequential(
            nn.Conv1d(input_channels + 1, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        final = self.power_net[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias[0], float(signal_bias_init))
        nn.init.constant_(final.bias[1], float(noise_bias_init))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        mixture_power = x.square().sum(dim=1, keepdim=True).clamp_min(self.eps)
        log_power = torch.log(mixture_power)
        log_power = log_power - log_power.mean(dim=-1, keepdim=True)
        logits = self.power_net(torch.cat([x, log_power], dim=1)).clamp(
            min=-self.log_power_clip,
            max=self.log_power_clip,
        )
        signal_power = mixture_power * torch.exp(logits[:, 0:1, :])
        noise_power = mixture_power * torch.exp(logits[:, 1:2, :])
        wiener_gain = signal_power / (signal_power + noise_power + self.eps)
        clean_mixture_hat = wiener_gain * x
        noise_hat = x - clean_mixture_hat
        aux = {
            "clean_mixture_hat": clean_mixture_hat,
            "noise_hat": noise_hat,
            "signal_power": signal_power,
            "noise_power": noise_power,
            "wiener_gain": wiener_gain,
        }
        return clean_mixture_hat, aux


class IQUMamba1D_NeuralWienerSE(nn.Module):
    """Neural Wiener front-end plus the original stage-4 IQUMamba."""

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
        wiener_hidden_channels: int = 16,
        wiener_kernel_size: int = 9,
        wiener_signal_bias_init: float = 3.0,
        wiener_noise_bias_init: float = -3.0,
        wiener_log_power_clip: float = 8.0,
        wiener_use_projection: bool = True,
        wiener_project_during_train: bool = False,
        wiener_project_during_eval: bool = True,
        wiener_source_weight: float = 0.25,
        wiener_noise_weight: float = 1.0,
        wiener_return_aux: bool = False,
        wiener_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")

        self.enhancer = NeuralWienerMixtureEnhancer1D(
            input_channels=input_channels,
            hidden_channels=wiener_hidden_channels,
            kernel_size=wiener_kernel_size,
            signal_bias_init=wiener_signal_bias_init,
            noise_bias_init=wiener_noise_bias_init,
            log_power_clip=wiener_log_power_clip,
            eps=wiener_eps,
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
        self.wiener_use_projection = bool(wiener_use_projection)
        self.wiener_project_during_train = bool(wiener_project_during_train)
        self.wiener_project_during_eval = bool(wiener_project_during_eval)
        self.wiener_return_aux = bool(wiener_return_aux)
        self.noise_projection = NoiseAwareMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            source_weight=wiener_source_weight,
            noise_weight=wiener_noise_weight,
            eps=wiener_eps,
        )

    def _should_project(self) -> bool:
        if not self.wiener_use_projection:
            return False
        return self.wiener_project_during_train if self.training else self.wiener_project_during_eval

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

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor], tuple[Union[torch.Tensor, List[torch.Tensor]], dict[str, torch.Tensor]]]:
        clean_mixture_hat, aux = self.enhancer(x)
        sources = self.backbone(clean_mixture_hat)
        projected = self._project_outputs(sources, x, aux["noise_hat"])
        if self.wiener_return_aux:
            return projected, aux
        return projected
