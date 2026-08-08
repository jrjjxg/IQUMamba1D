"""Modulation-prior input adapters for stage-4 IQUMamba.

These variants keep the IQUMamba backbone unchanged and add one small
near-identity residual adapter before it.  Each adapter uses only the received
I/Q mixture and is intended as a standalone ablation for non-8PSK-A datasets:
phase-step structure for PSK, lattice proximity for QAM, and radius-ring
structure for APSK.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D


def _validate_iq_mixture(x: torch.Tensor) -> None:
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected raw I/Q mixture shaped (B, 2, L), got {tuple(x.shape)}")


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = max(1, int(kernel_size))
    return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size


def _complex_view(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0, :].float(), x[:, 1, :].float())


def _safe_axis_normalize(axis: torch.Tensor, eps: float) -> torch.Tensor:
    scale = axis.detach().abs().amax(dim=-1, keepdim=True).clamp_min(float(eps))
    return (axis / scale).clamp(-1.5, 1.5)


def psk_phase_reliability(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a scalar confidence for repeated phase-step structure."""
    _validate_iq_mixture(x)
    z = _complex_view(x)
    if z.size(-1) < 2:
        return x.new_tensor(0.0)

    phase_step = torch.angle(z[:, 1:] * torch.conj(z[:, :-1]))
    unit_step = torch.exp(1j * phase_step)
    concentration = torch.abs(unit_step.mean(dim=-1))
    power = x.float().square().sum(dim=1)
    envelope = torch.sqrt(power + eps)
    envelope_cv = envelope.std(dim=-1, unbiased=False) / envelope.mean(dim=-1).clamp_min(eps)
    constant_envelope_score = torch.exp(-envelope_cv)
    return (0.65 * concentration + 0.35 * constant_envelope_score).clamp(0.0, 1.0).mean().to(dtype=x.dtype)


def _axis_level_scores(
    axis: torch.Tensor,
    axis_level_bank: Sequence[int],
    temperature: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis_norm = _safe_axis_normalize(axis.float(), eps)
    score_bank = []
    distance_bank = []
    for level_count in axis_level_bank:
        level_count = max(2, int(level_count))
        levels = torch.linspace(-1.0, 1.0, level_count, device=axis.device, dtype=axis_norm.dtype)
        distances = (axis_norm.unsqueeze(1) - levels.view(1, -1, 1)).abs()
        min_distance = distances.amin(dim=1)
        score = torch.exp(-float(temperature) * min_distance.square())
        score_bank.append(score)
        distance_bank.append(min_distance)
    return torch.stack(score_bank, dim=1), torch.stack(distance_bank, dim=1)


def qam_lattice_reliability(
    x: torch.Tensor,
    axis_level_bank: Sequence[int] = (4, 8, 12, 16),
    temperature: float = 24.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a scalar confidence for rectangular QAM-like axis levels."""
    _validate_iq_mixture(x)
    real_scores, _ = _axis_level_scores(x[:, 0, :], axis_level_bank, temperature, eps)
    imag_scores, _ = _axis_level_scores(x[:, 1, :], axis_level_bank, temperature, eps)
    grid_scores = real_scores * imag_scores
    return grid_scores.amax(dim=1).mean().clamp(0.0, 1.0).to(dtype=x.dtype)


def apsk_ring_reliability(
    x: torch.Tensor,
    ring_radii: Sequence[float] = (0.40, 1.13),
    temperature: float = 18.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return a scalar confidence for APSK-like radius rings."""
    _validate_iq_mixture(x)
    radius = torch.sqrt(x[:, 0, :].float().square() + x[:, 1, :].float().square() + eps)
    radii = torch.as_tensor(tuple(float(r) for r in ring_radii), device=x.device, dtype=radius.dtype)
    if radii.numel() == 0:
        return x.new_tensor(0.0)
    distances = (radius.unsqueeze(1) - radii.view(1, -1, 1)).abs()
    min_distance = distances.amin(dim=1)
    return torch.exp(-float(temperature) * min_distance.square()).mean().clamp(0.0, 1.0).to(dtype=x.dtype)


class PhaseDifferencePriorAdapter1D(nn.Module):
    """Near-identity adapter for PSK-like phase-step structure."""

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 8,
        harmonics: Sequence[int] = (1, 2, 4, 8),
        kernel_size: int = 9,
        scale_init: float = 0.01,
        reliability_floor: float = 0.05,
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError(f"PhaseDifferencePriorAdapter1D expects I/Q input with 2 channels, got {input_channels}")
        self.harmonics = tuple(max(1, int(h)) for h in harmonics)
        if not self.harmonics:
            raise ValueError("at least one phase harmonic is required")
        self.eps = float(eps)
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))

        feature_channels = 2 * len(self.harmonics) + 2
        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(feature_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        z = _complex_view(x)
        if z.size(-1) > 1:
            phase_step = torch.angle(z[:, 1:] * torch.conj(z[:, :-1]))
            phase_step = torch.cat([phase_step[:, :1], phase_step], dim=-1)
        else:
            phase_step = torch.zeros_like(x[:, 0, :].float())

        features = []
        for harmonic in self.harmonics:
            scaled = float(harmonic) * phase_step
            features.extend([torch.sin(scaled), torch.cos(scaled)])

        power = x.float().square().sum(dim=1)
        envelope = torch.sqrt(power + self.eps)
        envelope_norm = envelope / envelope.mean(dim=-1, keepdim=True).clamp_min(self.eps) - 1.0
        diff_power = torch.zeros_like(envelope_norm)
        if x.size(-1) > 1:
            diff = x[:, :, 1:].float() - x[:, :, :-1].float()
            diff_power[:, 1:] = torch.sqrt(diff.square().sum(dim=1) + self.eps)
        features.extend([envelope_norm, diff_power])
        return torch.stack(features, dim=1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_iq_mixture(x)
        reliability = psk_phase_reliability(x, eps=self.eps)
        reliability_scale = self.reliability_floor + (1.0 - self.reliability_floor) * reliability
        delta = self.net(self._features(x))
        return x + self.scale * reliability_scale.view(1, 1, 1) * torch.tanh(delta)


class QAMLatticePriorAdapter1D(nn.Module):
    """Near-identity adapter for QAM-like I/Q lattice proximity."""

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 8,
        axis_level_bank: Sequence[int] = (4, 8, 12, 16),
        temperature: float = 24.0,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        reliability_floor: float = 0.05,
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError(f"QAMLatticePriorAdapter1D expects I/Q input with 2 channels, got {input_channels}")
        self.axis_level_bank = tuple(max(2, int(levels)) for levels in axis_level_bank)
        if not self.axis_level_bank:
            raise ValueError("at least one QAM axis level count is required")
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))

        feature_channels = 2 + 3 * len(self.axis_level_bank)
        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(feature_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        real_norm = _safe_axis_normalize(x[:, 0, :], self.eps)
        imag_norm = _safe_axis_normalize(x[:, 1, :], self.eps)
        real_scores, real_distances = _axis_level_scores(x[:, 0, :], self.axis_level_bank, self.temperature, self.eps)
        imag_scores, imag_distances = _axis_level_scores(x[:, 1, :], self.axis_level_bank, self.temperature, self.eps)
        grid_scores = real_scores * imag_scores
        features = [real_norm, imag_norm]
        features.extend([real_distances[:, idx, :] for idx in range(len(self.axis_level_bank))])
        features.extend([imag_distances[:, idx, :] for idx in range(len(self.axis_level_bank))])
        features.extend([grid_scores[:, idx, :] for idx in range(len(self.axis_level_bank))])
        return torch.stack(features, dim=1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_iq_mixture(x)
        reliability = qam_lattice_reliability(
            x,
            axis_level_bank=self.axis_level_bank,
            temperature=self.temperature,
            eps=self.eps,
        )
        reliability_scale = self.reliability_floor + (1.0 - self.reliability_floor) * reliability
        delta = self.net(self._features(x))
        return x + self.scale * reliability_scale.view(1, 1, 1) * torch.tanh(delta)


class APSKRingPriorAdapter1D(nn.Module):
    """Near-identity adapter for APSK-like ring-radius structure."""

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 8,
        ring_radii: Sequence[float] = (0.40, 1.13),
        temperature: float = 18.0,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        reliability_floor: float = 0.05,
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError(f"APSKRingPriorAdapter1D expects I/Q input with 2 channels, got {input_channels}")
        self.ring_radii = tuple(float(radius) for radius in ring_radii)
        if not self.ring_radii:
            raise ValueError("at least one APSK ring radius is required")
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))

        feature_channels = 3 + 2 * len(self.ring_radii)
        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(feature_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0, :].float()
        imag = x[:, 1, :].float()
        radius = torch.sqrt(real.square() + imag.square() + self.eps)
        phase = torch.atan2(imag, real)
        radii = torch.as_tensor(self.ring_radii, device=x.device, dtype=radius.dtype)
        distances = (radius.unsqueeze(1) - radii.view(1, -1, 1)).abs()
        scores = torch.exp(-self.temperature * distances.square())
        radius_norm = radius / radius.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        features = [radius_norm, torch.sin(phase), torch.cos(phase)]
        features.extend([distances[:, idx, :] for idx in range(len(self.ring_radii))])
        features.extend([scores[:, idx, :] for idx in range(len(self.ring_radii))])
        return torch.stack(features, dim=1).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_iq_mixture(x)
        reliability = apsk_ring_reliability(
            x,
            ring_radii=self.ring_radii,
            temperature=self.temperature,
            eps=self.eps,
        )
        reliability_scale = self.reliability_floor + (1.0 - self.reliability_floor) * reliability
        delta = self.net(self._features(x))
        return x + self.scale * reliability_scale.view(1, 1, 1) * torch.tanh(delta)


class _IQUMambaPriorWrapper(nn.Module):
    def _build_backbone(
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
        conv_bias: bool,
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        nonlin: Type[nn.Module],
        nonlin_kwargs: dict,
        deep_supervision: bool,
    ) -> IQUMamba1D:
        return IQUMamba1D(
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


class IQUMamba1D_PSKPhasePrior(_IQUMambaPriorWrapper):
    """Stage-4 IQUMamba with a PSK phase-step input adapter."""

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
        psk_prior_hidden_channels: int = 8,
        psk_prior_harmonics: Sequence[int] = (1, 2, 4, 8),
        psk_prior_kernel_size: int = 9,
        psk_prior_scale_init: float = 0.01,
        psk_prior_reliability_floor: float = 0.05,
        psk_prior_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_adapter = PhaseDifferencePriorAdapter1D(
            input_channels=input_channels,
            hidden_channels=psk_prior_hidden_channels,
            harmonics=psk_prior_harmonics,
            kernel_size=psk_prior_kernel_size,
            scale_init=psk_prior_scale_init,
            reliability_floor=psk_prior_reliability_floor,
            zero_init=psk_prior_zero_init,
        )
        self.backbone = self._build_backbone(
            input_size, input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_conv_per_stage, num_classes,
            n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(self.input_adapter(x))


class IQUMamba1D_QAMLatticePrior(_IQUMambaPriorWrapper):
    """Stage-4 IQUMamba with a QAM lattice input adapter."""

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
        qam_prior_hidden_channels: int = 8,
        qam_prior_axis_level_bank: Sequence[int] = (4, 8, 12, 16),
        qam_prior_temperature: float = 24.0,
        qam_prior_kernel_size: int = 9,
        qam_prior_scale_init: float = 0.01,
        qam_prior_reliability_floor: float = 0.05,
        qam_prior_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_adapter = QAMLatticePriorAdapter1D(
            input_channels=input_channels,
            hidden_channels=qam_prior_hidden_channels,
            axis_level_bank=qam_prior_axis_level_bank,
            temperature=qam_prior_temperature,
            kernel_size=qam_prior_kernel_size,
            scale_init=qam_prior_scale_init,
            reliability_floor=qam_prior_reliability_floor,
            zero_init=qam_prior_zero_init,
        )
        self.backbone = self._build_backbone(
            input_size, input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_conv_per_stage, num_classes,
            n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(self.input_adapter(x))


class IQUMamba1D_APSKRingPrior(_IQUMambaPriorWrapper):
    """Stage-4 IQUMamba with an APSK radius-ring input adapter."""

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
        apsk_prior_hidden_channels: int = 8,
        apsk_prior_ring_radii: Sequence[float] = (0.40, 1.13),
        apsk_prior_temperature: float = 18.0,
        apsk_prior_kernel_size: int = 9,
        apsk_prior_scale_init: float = 0.01,
        apsk_prior_reliability_floor: float = 0.05,
        apsk_prior_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_adapter = APSKRingPriorAdapter1D(
            input_channels=input_channels,
            hidden_channels=apsk_prior_hidden_channels,
            ring_radii=apsk_prior_ring_radii,
            temperature=apsk_prior_temperature,
            kernel_size=apsk_prior_kernel_size,
            scale_init=apsk_prior_scale_init,
            reliability_floor=apsk_prior_reliability_floor,
            zero_init=apsk_prior_zero_init,
        )
        self.backbone = self._build_backbone(
            input_size, input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_conv_per_stage, num_classes,
            n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(self.input_adapter(x))
