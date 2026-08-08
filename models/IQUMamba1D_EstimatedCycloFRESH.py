"""Estimated-cyclic-frequency FRESH input adapter for stage-4 IQUMamba.

This variant avoids reading samples-per-symbol metadata.  It estimates a
dominant cyclic frequency from the received mixture itself using a simple
second-order statistic: the spectrum of the centered complex-envelope power.
The adapter then uses [0, +/-f_hat, +/-2*f_hat] shifted views before the
unchanged IQUMamba backbone.
"""

from __future__ import annotations

import math
from typing import List, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d


def estimate_cyclic_frequency(
    x: torch.Tensor,
    min_freq: float = 1.0 / 64.0,
    max_freq: float = 1.0 / 8.0,
    default_freq: float = 1.0 / 32.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate one normalized cyclic frequency from raw I/Q mixtures.

    Args:
        x: Mixture tensor shaped (B, 2, L).
        min_freq: Lower search bound in cycles/sample.
        max_freq: Upper search bound in cycles/sample.
        default_freq: Fallback when no valid FFT bin exists.
        eps: Numerical guard.

    Returns:
        Scalar tensor containing the batch-level frequency estimate.
    """
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

    length = int(x.size(-1))
    if length < 8:
        return x.new_tensor(float(default_freq))

    min_freq = max(0.0, float(min_freq))
    max_freq = min(0.5, float(max_freq))
    if min_freq >= max_freq:
        return x.new_tensor(float(default_freq))

    real = x[:, 0, :]
    imag = x[:, 1, :]
    envelope_power = real.square() + imag.square()
    envelope_power = envelope_power - envelope_power.mean(dim=-1, keepdim=True)

    spectrum = torch.fft.rfft(envelope_power.float(), dim=-1)
    power = spectrum.abs().square().mean(dim=0)
    freqs = torch.fft.rfftfreq(length, d=1.0).to(device=x.device)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not bool(mask.any()):
        return x.new_tensor(float(default_freq))

    masked_power = power[mask]
    masked_freqs = freqs[mask]
    peak = int(torch.argmax(masked_power).item())
    peak_freq = masked_freqs[peak]
    if not bool(torch.isfinite(peak_freq)):
        return x.new_tensor(float(default_freq))
    if float(masked_power[peak].detach().cpu()) <= eps:
        return x.new_tensor(float(default_freq))
    return peak_freq.to(device=x.device, dtype=x.dtype)


def estimate_cyclic_frequency_with_confidence(
    x: torch.Tensor,
    min_freq: float = 1.0 / 64.0,
    max_freq: float = 1.0 / 8.0,
    default_freq: float = 1.0 / 32.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate one cyclic frequency and peak confidence per mixture sample."""

    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(
            f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}"
        )
    batch, _, length = x.shape
    fallback = x.new_full((batch,), float(default_freq))
    no_confidence = x.new_zeros((batch,))
    if int(length) < 8:
        return fallback, no_confidence

    min_freq = max(0.0, float(min_freq))
    max_freq = min(0.5, float(max_freq))
    if min_freq >= max_freq:
        return fallback, no_confidence

    envelope_power = x[:, 0].square() + x[:, 1].square()
    envelope_power = envelope_power - envelope_power.mean(dim=-1, keepdim=True)
    spectrum_power = torch.fft.rfft(
        envelope_power.float(), dim=-1
    ).abs().square()
    frequencies = torch.fft.rfftfreq(
        int(length), d=1.0, device=x.device
    )
    mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    if not bool(mask.any()):
        return fallback, no_confidence

    candidates = spectrum_power[:, mask]
    candidate_frequencies = frequencies[mask]
    peak_power, peak_index = candidates.max(dim=-1)
    estimated = candidate_frequencies[peak_index]
    mean_power = candidates.mean(dim=-1)
    # Peak prominence is zero for a flat spectrum and approaches one for a
    # dominant cyclic line. It is bounded and comparable across FFT lengths.
    confidence = ((peak_power - mean_power) / (peak_power + mean_power + eps))
    confidence = confidence.clamp(0.0, 1.0)
    valid = torch.isfinite(estimated) & torch.isfinite(confidence) & (peak_power > eps)
    estimated = torch.where(valid, estimated, fallback.float())
    confidence = torch.where(valid, confidence, no_confidence.float())
    return estimated.to(dtype=x.dtype), confidence.to(dtype=x.dtype)


class EstimatedCycloFRESHAdapter1D(nn.Module):
    """Residual FRESH adapter using mixture-estimated cyclic frequency."""

    def __init__(
        self,
        input_channels: int,
        min_freq: float = 1.0 / 64.0,
        max_freq: float = 1.0 / 8.0,
        default_freq: float = 1.0 / 32.0,
        momentum: float = 0.05,
        hidden_channels: int = 8,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        gate_hidden: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"EstimatedCycloFRESHAdapter1D expects one complex mixture (2 channels), got {input_channels}")

        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.default_freq = float(default_freq)
        self.momentum = float(min(max(momentum, 0.0), 1.0))
        self.num_branches = 5
        self.register_buffer("freq_ema", torch.tensor(float(default_freq), dtype=torch.float32))

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
        self.last_frequency: torch.Tensor | None = None
        self.last_confidence: torch.Tensor | None = None

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

    def _branch_frequencies(self, x: torch.Tensor) -> torch.Tensor:
        base = self.current_base_frequency(x).clamp(min=self.min_freq, max=self.max_freq)
        return self._branch_frequencies_from_base(x, base)

    def _branch_frequencies_from_base(
        self, x: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        base = base.to(device=x.device, dtype=x.dtype).clamp(
            min=self.min_freq, max=self.max_freq
        )
        if base.ndim == 0:
            stack_dim = 0
        elif base.ndim == 1 and base.shape[0] == x.shape[0]:
            stack_dim = 1
        else:
            raise ValueError(
                "Conditioned FRESH frequency must be scalar or shaped (batch,)"
            )
        freqs = torch.stack([
            torch.zeros_like(base),
            base,
            -base,
            2.0 * base,
            -2.0 * base,
        ], dim=stack_dim)
        return freqs.clamp(min=-0.5, max=0.5)

    def _phasors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._phasors_from_base(x, self.current_base_frequency(x))

    def _phasors_from_base(
        self, x: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.size(-1)
        n = torch.arange(length, device=x.device, dtype=torch.float32)
        freqs = self._branch_frequencies_from_base(x, base).float()
        if freqs.ndim == 1:
            phase = -2.0 * math.pi * freqs[:, None] * n[None, :]
            phase = phase.unsqueeze(0)
        else:
            phase = -2.0 * math.pi * freqs[:, :, None] * n[None, None, :]
        cos = torch.cos(phase).to(dtype=x.dtype)
        sin = torch.sin(phase).to(dtype=x.dtype)
        return cos, sin

    def _shift_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._shift_branches_from_base(x, self.current_base_frequency(x))

    def _shift_branches_from_base(
        self, x: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        cos, sin = self._phasors_from_base(x, base)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return shifted_real, shifted_imag

    def forward_conditioned(
        self,
        x: torch.Tensor,
        frequency: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply FRESH branches driven by a shared external estimate."""

        shifted_real, shifted_imag = self._shift_branches_from_base(x, frequency)
        hidden_real, hidden_imag = self.branch_filter(shifted_real, shifted_imag)
        gate_input = torch.cat([shifted_real, shifted_imag], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        if confidence is not None:
            if confidence.ndim != 1 or confidence.shape[0] != x.shape[0]:
                raise ValueError("FRESH confidence must be shaped (batch,)")
            delta = delta * confidence[:, None, None].to(delta.dtype)
        self.last_frequency = frequency.detach()
        self.last_confidence = (
            None if confidence is None else confidence.detach()
        )
        return x + self.scale * delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        shifted_real, shifted_imag = self._shift_branches(x)
        hidden_real, hidden_imag = self.branch_filter(shifted_real, shifted_imag)
        gate_input = torch.cat([shifted_real, shifted_imag], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        return x + self.scale * delta


class IQUMamba1D_EstimatedCycloFRESH(nn.Module):
    """Stage-4 IQUMamba wrapped with a mixture-estimated Cyclo-FRESH input adapter."""

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
        complex_stem_enable: bool = False,
        complex_norm_eps: float = 1e-6,
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
        self.complex_stem_enable = bool(complex_stem_enable)
        if self.complex_stem_enable:
            from models.IQUMamba1D_ComplexStage4 import ComplexStem1d

            self.backbone.encoder.stem = ComplexStem1d(
                int(features_per_stage[0]),
                blocks=int(n_conv_per_stage[0]),
                kernel_size=int(kernel_sizes[0]),
                norm_eps=float(complex_norm_eps),
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.estimated_cyclofresh_adapter(x)
        return self.backbone(x)

    def no_weight_decay(self) -> set[str]:
        """Keep the optional Stage-290 stem's scale/threshold parameters unregularized."""
        if not self.complex_stem_enable:
            return set()

        from models.IQUMamba1D_ComplexStage4 import (
            ComplexModReLU,
            ComplexRMSNorm1d,
        )

        names = set()
        for module_name, module in self.named_modules():
            if isinstance(module, ComplexRMSNorm1d):
                names.add(f"{module_name}.log_scale")
            if isinstance(module, ComplexModReLU):
                names.add(f"{module_name}.bias")
        return names
