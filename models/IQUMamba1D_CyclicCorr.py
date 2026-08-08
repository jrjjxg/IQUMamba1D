"""Mixture-estimated cyclic-correlation input adapter for stage-4 IQUMamba.

The adapter keeps the original IQUMamba backbone unchanged.  It estimates a
dominant cyclic frequency from the received mixture, computes compact
second-order cyclic-correlation statistics at several lags, and uses those
statistics to condition a small complex residual adapter.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d
from models.IQUMamba1D_EstimatedCycloFRESH import estimate_cyclic_frequency


def _parse_lags(lags: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(lags, str):
        parts = [part.strip() for part in lags.split(",") if part.strip()]
        parsed = tuple(int(part) for part in parts)
    else:
        parsed = tuple(int(lag) for lag in lags)
    parsed = tuple(lag for lag in parsed if lag >= 0)
    return parsed if parsed else (0,)


def compute_cyclic_correlation_features(
    x: torch.Tensor,
    base_freq: torch.Tensor,
    lags: Sequence[int] = (0, 1, 2, 4, 8),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute compact cyclic-correlation statistics from raw I/Q mixtures.

    Args:
        x: Mixture tensor shaped (B, 2, L).
        base_freq: Normalized cyclic frequency in cycles/sample.
        lags: Non-negative sample lags used for R_x^alpha(tau).
        eps: Numerical guard for amplitude normalization.

    Returns:
        Tensor shaped (B, 4 * len(lags)): real/imag cyclic-correlation features
        for alpha = base_freq and alpha = 2 * base_freq.
    """
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

    lags = _parse_lags(lags)
    real = x[:, 0, :]
    imag = x[:, 1, :]
    z = torch.complex(real.float(), imag.float())
    length = int(z.size(-1))
    if length < 2:
        return x.new_zeros((x.size(0), 4 * len(lags)))

    base = base_freq.detach().to(device=x.device, dtype=torch.float32).clamp(min=0.0, max=0.5)
    alphas = torch.stack([base, (2.0 * base).clamp(max=0.5)])
    n = torch.arange(length, device=x.device, dtype=torch.float32)
    feats: list[torch.Tensor] = []

    power = z.abs().square().mean(dim=-1, keepdim=True).clamp_min(eps)
    for alpha in alphas:
        phasor = torch.exp(-1j * (2.0 * math.pi * alpha * n))
        for lag in lags:
            if lag >= length:
                corr = z.new_zeros((z.size(0),))
            elif lag == 0:
                corr = (z * z.conj() * phasor).mean(dim=-1)
            else:
                corr = (z[:, lag:] * z[:, :-lag].conj() * phasor[lag:]).mean(dim=-1)
            corr = corr / power.squeeze(-1)
            feats.append(corr.real)
            feats.append(corr.imag)

    return torch.stack(feats, dim=1).to(dtype=x.dtype)


class EstimatedCyclicCorrelationAdapter1D(nn.Module):
    """Residual complex adapter conditioned by mixture cyclic-correlation stats."""

    def __init__(
        self,
        input_channels: int,
        min_freq: float = 1.0 / 64.0,
        max_freq: float = 1.0 / 8.0,
        default_freq: float = 1.0 / 32.0,
        momentum: float = 0.05,
        lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        hidden_channels: int = 8,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        gate_hidden: int = 16,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"EstimatedCyclicCorrelationAdapter1D expects one complex mixture (2 channels), got {input_channels}")

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.default_freq = float(default_freq)
        self.momentum = float(min(max(momentum, 0.0), 1.0))
        self.lags = _parse_lags(lags)
        self.register_buffer("freq_ema", torch.tensor(float(default_freq), dtype=torch.float32))

        hidden_channels = max(1, int(hidden_channels))
        gate_hidden = max(1, int(gate_hidden))
        stat_dim = 4 * len(self.lags)

        self.in_proj = ComplexTiedConv1d(
            in_complex_channels=1,
            out_complex_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=hidden_channels,
            out_complex_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.stat_gate = nn.Sequential(
            nn.Linear(stat_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, hidden_channels),
            nn.Sigmoid(),
        )
        self.stat_bias = nn.Sequential(
            nn.Linear(stat_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 2 * hidden_channels),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    def current_base_frequency(self, x: torch.Tensor) -> torch.Tensor:
        estimated = estimate_cyclic_frequency(
            x.detach(),
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            default_freq=self.default_freq,
        )
        if self.training:
            with torch.no_grad():
                estimate_f32 = estimated.detach().to(device=self.freq_ema.device, dtype=self.freq_ema.dtype)
                self.freq_ema.mul_(1.0 - self.momentum).add_(estimate_f32 * self.momentum)
            return estimated
        return self.freq_ema.to(device=x.device, dtype=x.dtype)

    def cyclic_stats(self, x: torch.Tensor) -> torch.Tensor:
        base = self.current_base_frequency(x).clamp(min=self.min_freq, max=self.max_freq)
        return compute_cyclic_correlation_features(x.detach(), base, self.lags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        stats = self.cyclic_stats(x)
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        hidden_real, hidden_imag = self.in_proj(real, imag)

        gate = self.stat_gate(stats).unsqueeze(-1)
        bias = self.stat_bias(stats).unsqueeze(-1)
        bias_real, bias_imag = bias.chunk(2, dim=1)
        hidden_real = hidden_real * gate + bias_real
        hidden_imag = hidden_imag * gate + bias_imag

        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        return x + self.scale * delta


class IQUMamba1D_CyclicCorr(nn.Module):
    """Stage-4 IQUMamba wrapped with a cyclic-correlation input adapter."""

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
        cycliccorr_min_freq: float = 1.0 / 64.0,
        cycliccorr_max_freq: float = 1.0 / 8.0,
        cycliccorr_default_freq: float = 1.0 / 32.0,
        cycliccorr_momentum: float = 0.05,
        cycliccorr_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        cycliccorr_hidden_channels: int = 8,
        cycliccorr_kernel_size: int = 9,
        cycliccorr_scale_init: float = 0.01,
        cycliccorr_gate_hidden: int = 16,
        cycliccorr_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.cycliccorr_adapter = EstimatedCyclicCorrelationAdapter1D(
            input_channels=input_channels,
            min_freq=cycliccorr_min_freq,
            max_freq=cycliccorr_max_freq,
            default_freq=cycliccorr_default_freq,
            momentum=cycliccorr_momentum,
            lags=cycliccorr_lags,
            hidden_channels=cycliccorr_hidden_channels,
            kernel_size=cycliccorr_kernel_size,
            scale_init=cycliccorr_scale_init,
            gate_hidden=cycliccorr_gate_hidden,
            zero_init=cycliccorr_zero_init,
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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.cycliccorr_adapter(x)
        return self.backbone(x)
