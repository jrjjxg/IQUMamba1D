"""Cyclostationary front-end variants built on the successful stage-79 idea.

All variants keep the original stage-4 IQUMamba backbone unchanged.  They only
add mixture-derived input adapters and do not read dataset metadata, labels, or
clean reference signals.
"""

from __future__ import annotations

import math
from typing import List, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d
from models.IQUMamba1D_EstimatedCycloFRESH import EstimatedCycloFRESHAdapter1D


def _validate_iq_mixture(x: torch.Tensor) -> None:
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")


def _default_frequency_vector(
    x: torch.Tensor,
    num_peaks: int,
    min_freq: float,
    max_freq: float,
    default_freq: float,
) -> torch.Tensor:
    values = [float(default_freq) * float(i + 1) for i in range(max(1, int(num_peaks)))]
    values = [min(max(value, float(min_freq)), float(max_freq)) for value in values]
    return x.new_tensor(values)


def _envelope_power_spectrum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_iq_mixture(x)
    length = int(x.size(-1))
    real = x[:, 0, :]
    imag = x[:, 1, :]
    envelope_power = real.square() + imag.square()
    envelope_power = envelope_power - envelope_power.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(envelope_power.float(), dim=-1)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(device=x.device)
    return spectrum.abs().square(), freqs


def _pick_top_frequencies_from_power(
    power: torch.Tensor,
    freqs: torch.Tensor,
    num_peaks: int,
    min_freq: float,
    max_freq: float,
    default_freq: float,
    guard_bins: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_peaks = max(1, int(num_peaks))
    min_freq = max(0.0, float(min_freq))
    max_freq = min(0.5, float(max_freq))
    defaults = _default_frequency_vector(power, num_peaks, min_freq, max_freq, default_freq)

    if min_freq >= max_freq or power.numel() == 0:
        return defaults, power.new_tensor(0.0)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not bool(mask.any()):
        return defaults, power.new_tensor(0.0)

    masked_power = power[mask].float()
    masked_freqs = freqs[mask].to(device=power.device)
    if masked_power.numel() == 0:
        return defaults, power.new_tensor(0.0)

    work_power = masked_power.clone()
    selected: list[torch.Tensor] = []
    selected_power: list[torch.Tensor] = []
    guard_bins = max(0, int(guard_bins))
    invalid_value = work_power.new_tensor(float("-inf"))

    for index in range(num_peaks):
        finite_mask = torch.isfinite(work_power)
        if not bool(finite_mask.any()):
            selected.append(defaults[index].to(device=power.device, dtype=masked_freqs.dtype))
            selected_power.append(work_power.new_tensor(0.0))
            continue

        peak_power, peak_idx_tensor = torch.max(work_power, dim=0)
        peak_idx = int(peak_idx_tensor.item())
        if (not bool(torch.isfinite(peak_power))) or float(peak_power.detach().cpu()) <= eps:
            selected.append(defaults[index].to(device=power.device, dtype=masked_freqs.dtype))
            selected_power.append(work_power.new_tensor(0.0))
        else:
            selected.append(masked_freqs[peak_idx])
            selected_power.append(peak_power)

        start = max(0, peak_idx - guard_bins)
        stop = min(int(work_power.numel()), peak_idx + guard_bins + 1)
        work_power[start:stop] = invalid_value

    freqs_out = torch.stack(selected).to(device=power.device, dtype=power.dtype)
    peak0 = selected_power[0] if selected_power else power.new_tensor(0.0)
    floor = torch.median(masked_power).to(device=power.device)
    ratio = peak0 / (floor + eps)
    reliability = ((ratio - 1.0) / (ratio + 1.0 + eps)).clamp(0.0, 1.0)
    return freqs_out, reliability.to(device=power.device, dtype=power.dtype)


def estimate_cyclic_frequencies(
    x: torch.Tensor,
    num_peaks: int = 2,
    min_freq: float = 1.0 / 64.0,
    max_freq: float = 1.0 / 8.0,
    default_freq: float = 1.0 / 32.0,
    guard_bins: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate top-K normalized cyclic frequencies from the mixture batch."""
    _validate_iq_mixture(x)
    if int(x.size(-1)) < 8:
        defaults = _default_frequency_vector(x, num_peaks, min_freq, max_freq, default_freq)
        return defaults, x.new_tensor(0.0)

    power_per_sample, freqs = _envelope_power_spectrum(x.detach())
    batch_power = power_per_sample.mean(dim=0)
    return _pick_top_frequencies_from_power(
        batch_power,
        freqs,
        num_peaks=num_peaks,
        min_freq=min_freq,
        max_freq=max_freq,
        default_freq=default_freq,
        guard_bins=guard_bins,
        eps=eps,
    )


def estimate_samplewise_cyclic_frequencies(
    x: torch.Tensor,
    num_peaks: int = 1,
    min_freq: float = 1.0 / 64.0,
    max_freq: float = 1.0 / 8.0,
    default_freq: float = 1.0 / 32.0,
    guard_bins: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate top-K cyclic frequencies independently for each mixture item."""
    _validate_iq_mixture(x)
    if int(x.size(-1)) < 8:
        defaults = _default_frequency_vector(x, num_peaks, min_freq, max_freq, default_freq)
        return defaults.unsqueeze(0).expand(x.size(0), -1), x.new_zeros(x.size(0))

    power_per_sample, freqs = _envelope_power_spectrum(x.detach())
    freq_list = []
    reliability_list = []
    for sample_power in power_per_sample:
        sample_freqs, sample_reliability = _pick_top_frequencies_from_power(
            sample_power,
            freqs,
            num_peaks=num_peaks,
            min_freq=min_freq,
            max_freq=max_freq,
            default_freq=default_freq,
            guard_bins=guard_bins,
            eps=eps,
        )
        freq_list.append(sample_freqs)
        reliability_list.append(sample_reliability)
    return torch.stack(freq_list, dim=0), torch.stack(reliability_list, dim=0)


class MultiPeakCycloFRESHAdapter1D(nn.Module):
    """Residual FRESH adapter using multiple batch-level mixture-estimated peaks."""

    def __init__(
        self,
        input_channels: int,
        min_freq: float = 1.0 / 64.0,
        max_freq: float = 1.0 / 8.0,
        default_freq: float = 1.0 / 32.0,
        momentum: float = 0.05,
        num_peaks: int = 2,
        guard_bins: int = 3,
        hidden_channels: int = 8,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        gate_hidden: int = 8,
        reliability_floor: float = 0.25,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"MultiPeakCycloFRESHAdapter1D expects one complex mixture (2 channels), got {input_channels}")

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.default_freq = float(default_freq)
        self.momentum = float(min(max(momentum, 0.0), 1.0))
        self.num_peaks = max(1, int(num_peaks))
        self.guard_bins = max(0, int(guard_bins))
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))
        self.num_branches = 1 + 4 * self.num_peaks

        default_freqs = _default_frequency_vector(
            torch.empty(1),
            self.num_peaks,
            self.min_freq,
            self.max_freq,
            self.default_freq,
        ).float()
        self.register_buffer("freq_ema", default_freqs)
        self.register_buffer("reliability_ema", torch.tensor(1.0, dtype=torch.float32))

        hidden_channels = max(1, int(hidden_channels))
        gate_hidden = max(1, int(gate_hidden))
        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=self.num_branches,
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
        self.gate = nn.Sequential(
            nn.Conv1d(2 * self.num_branches, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            self._zero_output_projection()

    def _zero_output_projection(self) -> None:
        nn.init.zeros_(self.out_proj.real.weight)
        nn.init.zeros_(self.out_proj.imag.weight)
        if self.out_proj.bias_real is not None:
            nn.init.zeros_(self.out_proj.bias_real)
            nn.init.zeros_(self.out_proj.bias_imag)

    def current_base_frequencies(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs, reliability = estimate_cyclic_frequencies(
            x.detach(),
            num_peaks=self.num_peaks,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            default_freq=self.default_freq,
            guard_bins=self.guard_bins,
        )
        if self.training:
            with torch.no_grad():
                self.freq_ema.mul_(1.0 - self.momentum).add_(
                    freqs.detach().to(device=self.freq_ema.device, dtype=self.freq_ema.dtype) * self.momentum
                )
                self.reliability_ema.mul_(1.0 - self.momentum).add_(
                    reliability.detach().to(device=self.reliability_ema.device, dtype=self.reliability_ema.dtype) * self.momentum
                )
            return freqs.to(device=x.device, dtype=x.dtype), reliability.to(device=x.device, dtype=x.dtype)
        return (
            self.freq_ema.to(device=x.device, dtype=x.dtype),
            self.reliability_ema.to(device=x.device, dtype=x.dtype),
        )

    def _branch_frequencies(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bases, reliability = self.current_base_frequencies(x)
        bases = bases.clamp(min=self.min_freq, max=self.max_freq)
        branches = [x.new_tensor(0.0)]
        for base in bases:
            branches.extend([base, -base, 2.0 * base, -2.0 * base])
        return torch.stack(branches).clamp(min=-0.5, max=0.5), reliability

    def _phasors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = x.size(-1)
        n = torch.arange(length, device=x.device, dtype=torch.float32)
        freqs, reliability = self._branch_frequencies(x)
        phase = -2.0 * math.pi * freqs.to(dtype=torch.float32).unsqueeze(1) * n.unsqueeze(0)
        cos = torch.cos(phase).to(dtype=x.dtype).unsqueeze(0)
        sin = torch.sin(phase).to(dtype=x.dtype).unsqueeze(0)
        return cos, sin, reliability

    def _shift_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        cos, sin, reliability = self._phasors(x)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return shifted_real, shifted_imag, reliability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_iq_mixture(x)
        shifted_real, shifted_imag, reliability = self._shift_branches(x)
        hidden_real, hidden_imag = self.branch_filter(shifted_real, shifted_imag)
        gate_input = torch.cat([shifted_real, shifted_imag], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        reliability_scale = self.reliability_floor + (1.0 - self.reliability_floor) * reliability.clamp(0.0, 1.0)
        return x + self.scale * reliability_scale.view(1, 1, 1) * delta


class SampleAdaptiveCycloFRESHAdapter1D(nn.Module):
    """Residual FRESH adapter with per-item cyclic-frequency estimates."""

    def __init__(
        self,
        input_channels: int,
        min_freq: float = 1.0 / 64.0,
        max_freq: float = 1.0 / 8.0,
        default_freq: float = 1.0 / 32.0,
        num_peaks: int = 1,
        guard_bins: int = 3,
        hidden_channels: int = 8,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        gate_hidden: int = 8,
        reliability_floor: float = 0.25,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"SampleAdaptiveCycloFRESHAdapter1D expects one complex mixture (2 channels), got {input_channels}")

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.default_freq = float(default_freq)
        self.num_peaks = max(1, int(num_peaks))
        self.guard_bins = max(0, int(guard_bins))
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))
        self.num_branches = 1 + 4 * self.num_peaks

        hidden_channels = max(1, int(hidden_channels))
        gate_hidden = max(1, int(gate_hidden))
        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=self.num_branches,
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
        self.gate = nn.Sequential(
            nn.Conv1d(2 * self.num_branches, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    def _branch_frequencies(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bases, reliability = estimate_samplewise_cyclic_frequencies(
            x.detach(),
            num_peaks=self.num_peaks,
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            default_freq=self.default_freq,
            guard_bins=self.guard_bins,
        )
        bases = bases.to(device=x.device, dtype=x.dtype).clamp(min=self.min_freq, max=self.max_freq)
        branches = [x.new_zeros(x.size(0), 1)]
        for index in range(self.num_peaks):
            base = bases[:, index:index + 1]
            branches.extend([base, -base, 2.0 * base, -2.0 * base])
        return torch.cat(branches, dim=1).clamp(min=-0.5, max=0.5), reliability.to(device=x.device, dtype=x.dtype)

    def _phasors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = x.size(-1)
        n = torch.arange(length, device=x.device, dtype=torch.float32)
        freqs, reliability = self._branch_frequencies(x)
        phase = -2.0 * math.pi * freqs.to(dtype=torch.float32).unsqueeze(-1) * n.view(1, 1, -1)
        cos = torch.cos(phase).to(dtype=x.dtype)
        sin = torch.sin(phase).to(dtype=x.dtype)
        return cos, sin, reliability

    def _shift_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        cos, sin, reliability = self._phasors(x)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return shifted_real, shifted_imag, reliability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_iq_mixture(x)
        shifted_real, shifted_imag, reliability = self._shift_branches(x)
        hidden_real, hidden_imag = self.branch_filter(shifted_real, shifted_imag)
        gate_input = torch.cat([shifted_real, shifted_imag], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        reliability_scale = self.reliability_floor + (1.0 - self.reliability_floor) * reliability.clamp(0.0, 1.0)
        return x + self.scale * reliability_scale.view(-1, 1, 1) * delta


class FrequencyBiasCompensationAdapter1D(nn.Module):
    """Small high-pass residual adapter to counter low-frequency SSM bias."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 8,
        kernel_size: int = 9,
        lowpass_kernel_size: int = 17,
        scale_init: float = 0.01,
        gate_hidden: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"FrequencyBiasCompensationAdapter1D expects one complex mixture (2 channels), got {input_channels}")

        hidden_channels = max(1, int(hidden_channels))
        gate_hidden = max(1, int(gate_hidden))
        lowpass_kernel_size = max(3, int(lowpass_kernel_size))
        if lowpass_kernel_size % 2 == 0:
            lowpass_kernel_size += 1
        self.lowpass_kernel_size = lowpass_kernel_size

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
        self.gate = nn.Sequential(
            nn.Conv1d(2, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    def _highpass(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.lowpass_kernel_size // 2
        low = F.avg_pool1d(x, kernel_size=self.lowpass_kernel_size, stride=1, padding=pad)
        if low.size(-1) != x.size(-1):
            low = low[..., :x.size(-1)]
        return x - low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_iq_mixture(x)
        hp = self._highpass(x)
        real = hp[:, 0:1, :]
        imag = hp[:, 1:2, :]
        hidden_real, hidden_imag = self.in_proj(real, imag)
        gate_input = torch.cat([
            torch.sqrt(x[:, 0:1, :].square() + x[:, 1:2, :].square() + 1e-8),
            torch.sqrt(real.square() + imag.square() + 1e-8),
        ], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        return x + self.scale * delta


class _IQUMambaBackboneWrapper(nn.Module):
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


class IQUMamba1D_MultiPeakCycloFRESH(_IQUMambaBackboneWrapper):
    """Stage-4 IQUMamba with multi-peak mixture-estimated Cyclo-FRESH adapter."""

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
        multipeak_cyclofresh_min_freq: float = 1.0 / 64.0,
        multipeak_cyclofresh_max_freq: float = 1.0 / 8.0,
        multipeak_cyclofresh_default_freq: float = 1.0 / 32.0,
        multipeak_cyclofresh_momentum: float = 0.05,
        multipeak_cyclofresh_num_peaks: int = 2,
        multipeak_cyclofresh_guard_bins: int = 3,
        multipeak_cyclofresh_hidden_channels: int = 8,
        multipeak_cyclofresh_kernel_size: int = 9,
        multipeak_cyclofresh_scale_init: float = 0.01,
        multipeak_cyclofresh_gate_hidden: int = 8,
        multipeak_cyclofresh_reliability_floor: float = 0.25,
        multipeak_cyclofresh_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.cyclofresh_adapter = MultiPeakCycloFRESHAdapter1D(
            input_channels=input_channels,
            min_freq=multipeak_cyclofresh_min_freq,
            max_freq=multipeak_cyclofresh_max_freq,
            default_freq=multipeak_cyclofresh_default_freq,
            momentum=multipeak_cyclofresh_momentum,
            num_peaks=multipeak_cyclofresh_num_peaks,
            guard_bins=multipeak_cyclofresh_guard_bins,
            hidden_channels=multipeak_cyclofresh_hidden_channels,
            kernel_size=multipeak_cyclofresh_kernel_size,
            scale_init=multipeak_cyclofresh_scale_init,
            gate_hidden=multipeak_cyclofresh_gate_hidden,
            reliability_floor=multipeak_cyclofresh_reliability_floor,
            zero_init=multipeak_cyclofresh_zero_init,
        )
        self.backbone = self._build_backbone(
            input_size, input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_conv_per_stage, num_classes,
            n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(self.cyclofresh_adapter(x))


class IQUMamba1D_SampleCycloFRESH(_IQUMambaBackboneWrapper):
    """Stage-4 IQUMamba with sample-adaptive mixture-estimated Cyclo-FRESH adapter."""

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
        sample_cyclofresh_min_freq: float = 1.0 / 64.0,
        sample_cyclofresh_max_freq: float = 1.0 / 8.0,
        sample_cyclofresh_default_freq: float = 1.0 / 32.0,
        sample_cyclofresh_num_peaks: int = 1,
        sample_cyclofresh_guard_bins: int = 3,
        sample_cyclofresh_hidden_channels: int = 8,
        sample_cyclofresh_kernel_size: int = 9,
        sample_cyclofresh_scale_init: float = 0.01,
        sample_cyclofresh_gate_hidden: int = 8,
        sample_cyclofresh_reliability_floor: float = 0.25,
        sample_cyclofresh_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.cyclofresh_adapter = SampleAdaptiveCycloFRESHAdapter1D(
            input_channels=input_channels,
            min_freq=sample_cyclofresh_min_freq,
            max_freq=sample_cyclofresh_max_freq,
            default_freq=sample_cyclofresh_default_freq,
            num_peaks=sample_cyclofresh_num_peaks,
            guard_bins=sample_cyclofresh_guard_bins,
            hidden_channels=sample_cyclofresh_hidden_channels,
            kernel_size=sample_cyclofresh_kernel_size,
            scale_init=sample_cyclofresh_scale_init,
            gate_hidden=sample_cyclofresh_gate_hidden,
            reliability_floor=sample_cyclofresh_reliability_floor,
            zero_init=sample_cyclofresh_zero_init,
        )
        self.backbone = self._build_backbone(
            input_size, input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_conv_per_stage, num_classes,
            n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(self.cyclofresh_adapter(x))


class IQUMamba1D_CycloFRESHFreqBias(_IQUMambaBackboneWrapper):
    """Stage-4 IQUMamba with stage-79 FRESH plus high-frequency input adapter."""

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
        estimated_cyclofresh_min_freq: float = 1.0 / 64.0,
        estimated_cyclofresh_max_freq: float = 1.0 / 8.0,
        estimated_cyclofresh_default_freq: float = 1.0 / 32.0,
        estimated_cyclofresh_momentum: float = 0.05,
        estimated_cyclofresh_hidden_channels: int = 8,
        estimated_cyclofresh_kernel_size: int = 9,
        estimated_cyclofresh_scale_init: float = 0.01,
        estimated_cyclofresh_gate_hidden: int = 8,
        estimated_cyclofresh_zero_init: bool = True,
        freqbias_hidden_channels: int = 8,
        freqbias_kernel_size: int = 9,
        freqbias_lowpass_kernel_size: int = 17,
        freqbias_scale_init: float = 0.01,
        freqbias_gate_hidden: int = 8,
        freqbias_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.estimated_cyclofresh_adapter = EstimatedCycloFRESHAdapter1D(
            input_channels=input_channels,
            min_freq=estimated_cyclofresh_min_freq,
            max_freq=estimated_cyclofresh_max_freq,
            default_freq=estimated_cyclofresh_default_freq,
            momentum=estimated_cyclofresh_momentum,
            hidden_channels=estimated_cyclofresh_hidden_channels,
            kernel_size=estimated_cyclofresh_kernel_size,
            scale_init=estimated_cyclofresh_scale_init,
            gate_hidden=estimated_cyclofresh_gate_hidden,
            zero_init=estimated_cyclofresh_zero_init,
        )
        self.freqbias_adapter = FrequencyBiasCompensationAdapter1D(
            input_channels=input_channels,
            hidden_channels=freqbias_hidden_channels,
            kernel_size=freqbias_kernel_size,
            lowpass_kernel_size=freqbias_lowpass_kernel_size,
            scale_init=freqbias_scale_init,
            gate_hidden=freqbias_gate_hidden,
            zero_init=freqbias_zero_init,
        )
        self.backbone = self._build_backbone(
            input_size, input_channels, n_stages, features_per_stage, conv_op,
            kernel_sizes, strides, n_conv_per_stage, num_classes,
            n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.estimated_cyclofresh_adapter(x)
        x = self.freqbias_adapter(x)
        return self.backbone(x)
