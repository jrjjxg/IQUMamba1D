"""Lightweight low-SNR conditioned front-ends for stage-4 IQUMamba.

The goal is to improve low-SNR robustness without adding sequence blocks to the
U-Net backbone.  Both variants keep the original separator unchanged and only
learn a near-identity mixture enhancement before the backbone.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_NoiseAwareMC import NoiseAwareMixtureConsistencyProjection1D


class _ConditionedEnhancerBase(nn.Module):
    """Near-identity residual enhancer with a sample-wise residual scale."""

    def __init__(
        self,
        input_channels: int,
        stats_dim: int,
        hidden_channels: int = 24,
        kernel_size: int = 9,
        gate_hidden: int = 12,
        scale_init: float = 0.01,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"Conditioned low-SNR enhancer expects one I/Q mixture with 2 channels, got {input_channels}")

        hidden_channels = max(4, int(hidden_channels))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        gate_hidden = max(4, int(gate_hidden))

        self.eps = float(eps)
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv1d(hidden_channels, input_channels, kernel_size=1),
        )
        self.scale_base = nn.Parameter(torch.tensor(float(scale_init)))
        self.scale_gate = nn.Sequential(
            nn.Linear(stats_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

        if zero_init:
            final_conv = self.net[-1]
            nn.init.zeros_(final_conv.weight)
            nn.init.zeros_(final_conv.bias)
            final_gate = self.scale_gate[-1]
            nn.init.zeros_(final_gate.weight)
            nn.init.zeros_(final_gate.bias)

    def _envelope_power(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0].pow(2) + x[:, 1].pow(2)

    def _base_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        power = self._envelope_power(x)
        mean_power = power.mean(dim=-1).clamp_min(self.eps)
        centered = power - power.mean(dim=-1, keepdim=True)
        var_power = centered.pow(2).mean(dim=-1).clamp_min(self.eps)
        spectrum = torch.fft.rfft(centered, dim=-1)
        spectral_power = spectrum.abs().pow(2)
        total_spectral_power = spectral_power[:, 1:].sum(dim=-1).clamp_min(self.eps)
        split = max(1, spectral_power.size(-1) // 3)
        high_power = spectral_power[:, split:].sum(dim=-1)
        high_freq_ratio = high_power / total_spectral_power
        return power, spectral_power, torch.log(mean_power), torch.log(var_power), high_freq_ratio

    def _compute_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        stats, aux_stats = self._compute_stats(x)
        gate = torch.sigmoid(self.scale_gate(stats)).view(x.size(0), 1, 1)
        scale = self.scale_base * (2.0 * gate)
        delta = torch.tanh(self.net(x))
        clean_mixture_hat = x + scale * delta
        noise_hat = x - clean_mixture_hat
        aux = {
            "clean_mixture_hat": clean_mixture_hat,
            "noise_hat": noise_hat,
            "enhancement_delta": delta,
            "enhancement_scale": scale,
            "scale_gate": gate,
        }
        aux.update(aux_stats)
        return clean_mixture_hat, aux


class SNRConditionedLowSNRMixtureEnhancer1D(_ConditionedEnhancerBase):
    """Condition enhancement strength on simple per-sample SNR proxy statistics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, stats_dim=3, **kwargs)

    def _compute_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, _, log_mean_power, log_var_power, high_freq_ratio = self._base_stats(x)
        stats = torch.stack([log_mean_power, log_var_power, high_freq_ratio], dim=-1)
        return stats, {
            "log_mean_power": log_mean_power,
            "log_var_power": log_var_power,
            "high_freq_ratio": high_freq_ratio,
        }


class CyclicReliabilityLowSNRMixtureEnhancer1D(_ConditionedEnhancerBase):
    """Condition enhancement strength on cyclic-envelope peak reliability."""

    def __init__(
        self,
        *args,
        min_freq: float = 0.01,
        max_freq: float = 0.45,
        **kwargs,
    ) -> None:
        super().__init__(*args, stats_dim=4, **kwargs)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

    def _compute_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, spectral_power, log_mean_power, log_var_power, high_freq_ratio = self._base_stats(x)
        device = x.device
        freq_grid = torch.linspace(0.0, 0.5, spectral_power.size(-1), device=device, dtype=spectral_power.dtype)
        band_mask = (freq_grid >= self.min_freq) & (freq_grid <= self.max_freq)
        if not torch.any(band_mask):
            peak_ratio = torch.zeros(x.size(0), device=device, dtype=x.dtype)
            dominant_freq = torch.zeros_like(peak_ratio)
        else:
            band_power = spectral_power[:, band_mask]
            band_freq = freq_grid[band_mask]
            peak_power, peak_idx = band_power.max(dim=-1)
            mean_band_power = band_power.mean(dim=-1).clamp_min(self.eps)
            peak_ratio = torch.log1p(peak_power / mean_band_power)
            dominant_freq = band_freq[peak_idx].to(dtype=x.dtype)

        stats = torch.stack([log_mean_power, log_var_power, high_freq_ratio, peak_ratio], dim=-1)
        return stats, {
            "log_mean_power": log_mean_power,
            "log_var_power": log_var_power,
            "high_freq_ratio": high_freq_ratio,
            "peak_ratio": peak_ratio,
            "dominant_cyclic_freq": dominant_freq,
        }


class _IQUMamba1D_LowSNRConditionedBase(nn.Module):
    """Conditioned low-SNR front-end plus the original stage-4 IQUMamba."""

    enhancer_cls: Type[_ConditionedEnhancerBase]

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
        low_snr_cond_hidden_channels: int = 24,
        low_snr_cond_kernel_size: int = 9,
        low_snr_cond_gate_hidden: int = 12,
        low_snr_cond_scale_init: float = 0.01,
        low_snr_cond_zero_init: bool = True,
        low_snr_cond_use_projection: bool = True,
        low_snr_cond_project_during_train: bool = False,
        low_snr_cond_project_during_eval: bool = True,
        low_snr_cond_source_weight: float = 0.25,
        low_snr_cond_noise_weight: float = 1.0,
        low_snr_cond_return_aux: bool = False,
        low_snr_cond_eps: float = 1e-8,
        **enhancer_kwargs,
    ) -> None:
        super().__init__()
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")

        self.enhancer = self.enhancer_cls(
            input_channels=input_channels,
            hidden_channels=low_snr_cond_hidden_channels,
            kernel_size=low_snr_cond_kernel_size,
            gate_hidden=low_snr_cond_gate_hidden,
            scale_init=low_snr_cond_scale_init,
            zero_init=low_snr_cond_zero_init,
            eps=low_snr_cond_eps,
            **enhancer_kwargs,
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
        self.low_snr_cond_use_projection = bool(low_snr_cond_use_projection)
        self.low_snr_cond_project_during_train = bool(low_snr_cond_project_during_train)
        self.low_snr_cond_project_during_eval = bool(low_snr_cond_project_during_eval)
        self.low_snr_cond_return_aux = bool(low_snr_cond_return_aux)
        self.noise_projection = NoiseAwareMixtureConsistencyProjection1D(
            num_sources=self.num_sources,
            source_weight=low_snr_cond_source_weight,
            noise_weight=low_snr_cond_noise_weight,
            eps=low_snr_cond_eps,
        )

    def _should_project(self) -> bool:
        if not self.low_snr_cond_use_projection:
            return False
        return self.low_snr_cond_project_during_train if self.training else self.low_snr_cond_project_during_eval

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
        if self.low_snr_cond_return_aux:
            return projected, aux
        return projected


class IQUMamba1D_LowSNRSNRConditioned(_IQUMamba1D_LowSNRConditionedBase):
    enhancer_cls = SNRConditionedLowSNRMixtureEnhancer1D


class IQUMamba1D_LowSNRCyclicConditioned(_IQUMamba1D_LowSNRConditionedBase):
    enhancer_cls = CyclicReliabilityLowSNRMixtureEnhancer1D
