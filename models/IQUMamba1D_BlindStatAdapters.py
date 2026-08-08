"""Blind-statistic adapters for stage-4 IQUMamba.

Both variants keep the original IQUMamba path intact and use only statistics
computed from the received mixture.  Stage 85 conditions feature maps after the
stem and at the bottleneck.  Stage 86 applies a small near-identity input
residual before the unchanged backbone.
"""

from __future__ import annotations

import math
from typing import List, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_EstimatedCycloFRESH import estimate_cyclic_frequency


class BlindSignalStats(nn.Module):
    """Compute compact mixture-only RF statistics for conditioning."""

    num_stats = 10

    def __init__(
        self,
        cyclic_min_freq: float = 1.0 / 64.0,
        cyclic_max_freq: float = 1.0 / 8.0,
        cyclic_default_freq: float = 1.0 / 32.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cyclic_min_freq = float(cyclic_min_freq)
        self.cyclic_max_freq = float(cyclic_max_freq)
        self.cyclic_default_freq = float(cyclic_default_freq)
        self.eps = float(eps)

    def _cyclic_stats(self, envelope_power: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(envelope_power.size(-1))
        if length < 8:
            freq = envelope_power.new_full((envelope_power.size(0),), self.cyclic_default_freq)
            return freq, torch.zeros_like(freq)

        centered = envelope_power - envelope_power.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered.float(), dim=-1)
        power = spectrum.abs().square()
        freqs = torch.fft.rfftfreq(length, d=1.0).to(device=envelope_power.device)
        mask = (freqs >= self.cyclic_min_freq) & (freqs <= self.cyclic_max_freq)
        if not bool(mask.any()):
            freq = envelope_power.new_full((envelope_power.size(0),), self.cyclic_default_freq)
            return freq, torch.zeros_like(freq)

        masked_power = power[:, mask]
        masked_freqs = freqs[mask].to(device=envelope_power.device)
        peak_idx = torch.argmax(masked_power, dim=-1)
        peak_power = masked_power.gather(1, peak_idx.unsqueeze(1)).squeeze(1)
        peak_freq = masked_freqs[peak_idx].to(dtype=envelope_power.dtype)
        mean_power = masked_power.mean(dim=-1).to(dtype=envelope_power.dtype)
        reliability = torch.log1p(peak_power.to(dtype=envelope_power.dtype) / (mean_power + self.eps))
        return peak_freq, reliability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"BlindSignalStats expects one I/Q mixture shaped (B, 2, L), got {tuple(x.shape)}")

        real = x[:, 0, :].float()
        imag = x[:, 1, :].float()
        z = torch.complex(real, imag)
        envelope_power = real.square() + imag.square()
        envelope = torch.sqrt(envelope_power + self.eps)
        mean_power = envelope_power.mean(dim=-1) + self.eps
        rms = torch.sqrt(mean_power)

        log_rms = torch.log1p(rms)
        papr = torch.log1p(envelope_power.amax(dim=-1) / mean_power)
        envelope_mean = envelope.mean(dim=-1) + self.eps
        envelope_cv = envelope.std(dim=-1, unbiased=False) / envelope_mean
        envelope_kurtosis = torch.log1p(envelope_power.square().mean(dim=-1) / mean_power.square())
        circularity = torch.abs((z * z).mean(dim=-1)) / mean_power

        spectrum = torch.fft.fft(z, dim=-1)
        one_sided = spectrum[:, : spectrum.size(-1) // 2 + 1]
        spec_power = one_sided.abs().square() + self.eps
        prob = spec_power / spec_power.sum(dim=-1, keepdim=True)
        spec_entropy = -(prob * torch.log(prob)).sum(dim=-1) / math.log(max(2, spec_power.size(-1)))
        split = max(1, spec_power.size(-1) // 3)
        low_energy = spec_power[:, :split].sum(dim=-1)
        high_energy = spec_power[:, -split:].sum(dim=-1)
        high_low_ratio = torch.log1p(high_energy / (low_energy + self.eps))

        if z.size(-1) > 1:
            phase_step = torch.angle(z[:, 1:] * torch.conj(z[:, :-1]))
            phase_diff_var = phase_step.var(dim=-1, unbiased=False) / (math.pi * math.pi)
        else:
            phase_diff_var = torch.zeros_like(log_rms)

        cyclic_freq, cyclic_reliability = self._cyclic_stats(envelope_power)
        cyclic_freq = cyclic_freq.to(device=x.device, dtype=log_rms.dtype)
        cyclic_reliability = cyclic_reliability.to(device=x.device, dtype=log_rms.dtype)

        stats = torch.stack(
            [
                log_rms,
                papr,
                envelope_cv,
                envelope_kurtosis,
                circularity,
                spec_entropy,
                high_low_ratio,
                phase_diff_var,
                cyclic_freq,
                cyclic_reliability,
            ],
            dim=-1,
        )
        return torch.nan_to_num(stats, nan=0.0, posinf=10.0, neginf=-10.0).to(dtype=x.dtype)


class BlindStatFiLM(nn.Module):
    """Small FiLM block driven by mixture-level blind statistics."""

    def __init__(self, stat_dim: int, channels: int, hidden: int = 32, scale_init: float = 0.01, zero_init: bool = True) -> None:
        super().__init__()
        hidden = max(1, int(hidden))
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * int(channels)),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(stats.float()).to(dtype=x.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1)
        return x * (1.0 + self.scale * gamma) + self.scale * beta


class BlindStatFiLMEncoder(nn.Module):
    """Wrap an IQUMamba encoder with stem and bottleneck FiLM conditioning."""

    def __init__(self, encoder: nn.Module, stat_dim: int, hidden: int = 32, scale_init: float = 0.01, zero_init: bool = True) -> None:
        super().__init__()
        self.encoder = encoder
        self.stem_film = BlindStatFiLM(stat_dim, encoder.output_channels[0], hidden, scale_init, zero_init)
        self.bottleneck_film = BlindStatFiLM(stat_dim, encoder.output_channels[-1], hidden, scale_init, zero_init)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)

    def forward(self, x: torch.Tensor, stats: torch.Tensor | None = None):
        if stats is None:
            raise ValueError("BlindStatFiLMEncoder requires precomputed blind statistics")

        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
            x = self.stem_film(x, stats)

        ret = []
        last_stage = len(self.encoder.stages) - 1
        for stage_idx in range(len(self.encoder.stages)):
            x = self.encoder.stages[stage_idx](x)
            x = self.encoder.mamba_layers[stage_idx](x)
            if stage_idx == last_stage:
                x = self.bottleneck_film(x, stats)
            ret.append(x)
        return ret if self.encoder.return_skips else ret[-1]


class BlindStatInputAdapter1D(nn.Module):
    """Near-identity input residual conditioned by blind mixture statistics."""

    def __init__(
        self,
        input_channels: int,
        stat_dim: int,
        hidden: int = 16,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        zero_init: bool = True,
        cyclic_min_freq: float = 1.0 / 64.0,
        cyclic_max_freq: float = 1.0 / 8.0,
        cyclic_default_freq: float = 1.0 / 32.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"BlindStatInputAdapter1D expects one I/Q mixture with 2 channels, got {input_channels}")
        hidden = max(1, int(hidden))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.eps = float(eps)
        self.cyclic_min_freq = float(cyclic_min_freq)
        self.cyclic_max_freq = float(cyclic_max_freq)
        self.cyclic_default_freq = float(cyclic_default_freq)
        self.smooth_kernel_size = kernel_size
        self.view_proj = nn.Sequential(
            nn.Conv1d(10, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
        )
        self.stat_gate = nn.Sequential(
            nn.Linear(stat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv1d(hidden, 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def _cyclic_shift_view(self, x: torch.Tensor) -> torch.Tensor:
        base = estimate_cyclic_frequency(
            x.detach(),
            min_freq=self.cyclic_min_freq,
            max_freq=self.cyclic_max_freq,
            default_freq=self.cyclic_default_freq,
        ).to(device=x.device, dtype=torch.float32)
        n = torch.arange(x.size(-1), device=x.device, dtype=torch.float32)
        phase = -2.0 * math.pi * base * n
        cos = torch.cos(phase).to(dtype=x.dtype).view(1, 1, -1)
        sin = torch.sin(phase).to(dtype=x.dtype).view(1, 1, -1)
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return torch.cat([shifted_real, shifted_imag], dim=1)

    def _views(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.sqrt(x[:, 0:1, :].square() + x[:, 1:2, :].square() + self.eps)
        unit = x / (mag + self.eps)
        local_mean = F.avg_pool1d(x, kernel_size=self.smooth_kernel_size, stride=1, padding=self.smooth_kernel_size // 2, count_include_pad=False)
        highpass = x - local_mean
        amp_centered = mag - F.avg_pool1d(mag, kernel_size=self.smooth_kernel_size, stride=1, padding=self.smooth_kernel_size // 2, count_include_pad=False)
        amp_residual = amp_centered * unit
        shifted = self._cyclic_shift_view(x)
        return torch.cat([x, highpass, unit, amp_residual, shifted], dim=1)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"BlindStatInputAdapter1D expects one I/Q mixture shaped (B, 2, L), got {tuple(x.shape)}")
        views = self._views(x)
        hidden = self.view_proj(views)
        gate = self.stat_gate(stats.float()).to(dtype=x.dtype).unsqueeze(-1)
        delta = self.out_proj(hidden * gate)
        return x + self.scale * torch.tanh(delta)


class IQUMamba1D_BlindStatFiLM(nn.Module):
    """Stage-4 IQUMamba with blind-statistic feature FiLM conditioning."""

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
        blindstat_hidden: int = 32,
        blindstat_scale_init: float = 0.01,
        blindstat_cyclic_min_freq: float = 1.0 / 64.0,
        blindstat_cyclic_max_freq: float = 1.0 / 8.0,
        blindstat_cyclic_default_freq: float = 1.0 / 32.0,
        blindstat_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.stats = BlindSignalStats(
            cyclic_min_freq=blindstat_cyclic_min_freq,
            cyclic_max_freq=blindstat_cyclic_max_freq,
            cyclic_default_freq=blindstat_cyclic_default_freq,
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
        self.encoder = BlindStatFiLMEncoder(
            self.backbone.encoder,
            stat_dim=self.stats.num_stats,
            hidden=blindstat_hidden,
            scale_init=blindstat_scale_init,
            zero_init=blindstat_zero_init,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        stats = self.stats(x)
        skips = self.encoder(x, stats)
        return self.backbone.decoder(skips)


class IQUMamba1D_BlindStatInput(nn.Module):
    """Stage-4 IQUMamba with a blind-statistic near-identity input adapter."""

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
        blindstat_hidden: int = 16,
        blindstat_kernel_size: int = 9,
        blindstat_scale_init: float = 0.01,
        blindstat_cyclic_min_freq: float = 1.0 / 64.0,
        blindstat_cyclic_max_freq: float = 1.0 / 8.0,
        blindstat_cyclic_default_freq: float = 1.0 / 32.0,
        blindstat_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.stats = BlindSignalStats(
            cyclic_min_freq=blindstat_cyclic_min_freq,
            cyclic_max_freq=blindstat_cyclic_max_freq,
            cyclic_default_freq=blindstat_cyclic_default_freq,
        )
        self.input_adapter = BlindStatInputAdapter1D(
            input_channels=input_channels,
            stat_dim=self.stats.num_stats,
            hidden=blindstat_hidden,
            kernel_size=blindstat_kernel_size,
            scale_init=blindstat_scale_init,
            zero_init=blindstat_zero_init,
            cyclic_min_freq=blindstat_cyclic_min_freq,
            cyclic_max_freq=blindstat_cyclic_max_freq,
            cyclic_default_freq=blindstat_cyclic_default_freq,
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
        stats = self.stats(x)
        x = self.input_adapter(x, stats)
        return self.backbone(x)
