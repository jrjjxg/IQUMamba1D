"""Stage 226: evidence-routed multi-view communication prior adapter.

The adapter is deliberately mixture-only at inference time.  It exposes a
small set of views that cover different modulation families:

* a null/identity view;
* a multi-lag cyclic/FRESH view;
* an amplitude/phase-increment view;
* a learnable pulse/multiscale filterbank view.

The views are fused as bounded residual corrections before an unchanged
IQUMamba Stage-4 backbone.  The identity path and zero-initialized output
heads make the model an exact Stage-4 model at initialization.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d
from models.IQU_UniversalPriorAdapter import (
    MultiBranchFRESH,
    MultiLagCyclicEstimator,
    RRCBank,
)


def _zero_complex_projection(layer: ComplexTiedConv1d) -> None:
    """Zero the final complex projection while keeping its input trainable."""
    nn.init.zeros_(layer.real.weight)
    nn.init.zeros_(layer.imag.weight)
    if layer.bias_real is not None:
        nn.init.zeros_(layer.bias_real)
        nn.init.zeros_(layer.bias_imag)


def _zero_real_projection(layer: nn.Conv1d) -> None:
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class AdaptiveViewEvidence(nn.Module):
    """Extract low-dimensional, mixture-only reliability evidence.

    The evidence is not a modulation classifier.  It measures whether a view
    is likely to be useful for the current mixture and is detached before it
    enters the router.  The feature set intentionally combines amplitude,
    phase, lag-correlation, cyclic and spectral-flatness cues.
    """

    evidence_dim = 8
    num_experts = 3

    def __init__(
        self,
        lag_bank: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        min_freq: float = 1.0 / 128.0,
        max_freq: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        lags = tuple(sorted({int(lag) for lag in lag_bank if int(lag) > 0}))
        self.register_buffer("lag_bank", torch.tensor(lags, dtype=torch.long), persistent=False)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.eps = float(eps)

    def _normalized_lag_correlation(self, signal: torch.Tensor, lag: int) -> torch.Tensor:
        if lag <= 0 or signal.size(-1) <= lag:
            return signal.new_zeros(signal.size(0))
        left = signal[..., lag:]
        right = signal[..., :-lag]
        cross = (left * torch.conj(right)).mean(dim=-1).abs()
        left_power = left.abs().square().mean(dim=-1)
        right_power = right.abs().square().mean(dim=-1)
        denominator = (left_power * right_power).clamp_min(self.eps).sqrt()
        return (cross / denominator).clamp(0.0, 1.0)

    def _cyclic_statistics(self, power: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        length = power.size(-1)
        centered = power - power.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(centered, dim=-1).abs().square()
        frequencies = torch.fft.rfftfreq(length, d=1.0).to(device=power.device)
        mask = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)

        if not bool(mask.any()):
            zeros = power.new_zeros(power.size(0))
            return zeros, zeros, zeros, torch.ones_like(zeros)

        selected = spectrum[:, mask]
        floor = selected.mean(dim=-1).clamp_min(self.eps)
        top_k = min(2, selected.size(-1))
        values = torch.topk(selected, k=top_k, dim=-1).values
        peak = values[:, 0]
        second = values[:, 1] if top_k > 1 else torch.zeros_like(peak)
        sharpness = torch.log1p(peak / floor)
        multiplicity = (second / peak.clamp_min(self.eps)).clamp(0.0, 1.0)
        confidence = torch.tanh(sharpness / 3.0).clamp(0.0, 1.0)
        flatness = torch.exp(torch.log(selected.clamp_min(self.eps)).mean(dim=-1)) / floor
        return sharpness, multiplicity, confidence, flatness.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected mixture IQ with shape [B, 2, L], got {tuple(x.shape)}")

        # Statistics are evaluated in fp32 so that phase and FFT evidence do
        # not become unstable under mixed-precision training.
        real_input = torch.nan_to_num(x.float())
        signal = torch.complex(real_input[:, 0], real_input[:, 1])
        envelope = signal.abs()
        power = envelope.square()
        envelope_mean = envelope.mean(dim=-1).clamp_min(self.eps)
        envelope_cv = envelope.std(dim=-1, unbiased=False) / envelope_mean

        phase_product = signal[..., 1:] * torch.conj(signal[..., :-1])
        phase_magnitude = phase_product.abs().clamp_min(self.eps)
        phase_cos = phase_product.real / phase_magnitude
        phase_sin = phase_product.imag / phase_magnitude
        phase_coherence = self._normalized_lag_correlation(signal, 1)
        if phase_cos.size(-1) > 1:
            phase_roughness = torch.atan2(phase_sin, phase_cos).diff(dim=-1).abs().mean(dim=-1)
        else:
            phase_roughness = real_input.new_zeros(real_input.size(0))

        if self.lag_bank.numel() > 0:
            lag_values = torch.stack(
                [self._normalized_lag_correlation(signal, int(lag.item())) for lag in self.lag_bank],
                dim=1,
            )
            lag_peak = lag_values.max(dim=1).values
        else:
            lag_peak = real_input.new_zeros(real_input.size(0))

        cyclic_sharpness, cyclic_multiplicity, cyclic_confidence, spectral_flatness = self._cyclic_statistics(power)

        evidence = torch.stack(
            [
                torch.tanh(envelope_cv),
                phase_coherence,
                lag_peak,
                torch.tanh(cyclic_sharpness / 3.0),
                cyclic_multiplicity,
                spectral_flatness,
                torch.tanh(phase_roughness / (math.pi / 2.0)),
                torch.tanh(power.mean(dim=-1).sqrt()),
            ],
            dim=-1,
        )

        phase_confidence = phase_coherence.clamp(0.05, 1.0)
        pulse_confidence = (0.5 * lag_peak + 0.5 * (1.0 - spectral_flatness)).clamp(0.05, 1.0)
        confidence = torch.stack(
            [cyclic_confidence.clamp(0.05, 1.0), phase_confidence, pulse_confidence],
            dim=-1,
        )

        return {
            "evidence": torch.nan_to_num(evidence, nan=0.0, posinf=1.0, neginf=0.0),
            "confidence": torch.nan_to_num(confidence, nan=0.05, posinf=1.0, neginf=0.05),
            "cyclic_confidence": cyclic_confidence.detach(),
            "phase_coherence": phase_coherence.detach(),
            "lag_peak": lag_peak.detach(),
            "spectral_flatness": spectral_flatness.detach(),
        }


class EvidenceRouter(nn.Module):
    """Soft evidence router with an explicit identity/null candidate."""

    def __init__(
        self,
        evidence_dim: int,
        hidden_channels: int = 16,
        identity_bias: float = 2.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = max(float(temperature), 1e-3)
        self.identity_bias = float(identity_bias)
        self.network = nn.Sequential(
            nn.Linear(int(evidence_dim), int(hidden_channels)),
            nn.SiLU(),
            nn.Linear(int(hidden_channels), 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, evidence: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 2 or confidence.ndim != 2 or confidence.size(1) != 3:
            raise ValueError("EvidenceRouter expects [B, E] evidence and [B, 3] confidence")
        expert_logits = self.network(evidence)
        expert_logits = expert_logits + torch.log(confidence.clamp(0.05, 1.0))
        identity = torch.full(
            (evidence.size(0), 1),
            self.identity_bias,
            dtype=expert_logits.dtype,
            device=expert_logits.device,
        )
        return torch.softmax(torch.cat([identity, expert_logits], dim=-1) / self.temperature, dim=-1)


class _BoundedRealPriorExpert(nn.Module):
    def __init__(self, hidden_channels: int, input_channels: int, max_delta: float) -> None:
        super().__init__()
        hidden_channels = max(1, int(hidden_channels))
        self.max_delta = float(max_delta)
        self.in_proj = nn.Conv1d(input_channels, hidden_channels, kernel_size=5, padding=2)
        self.gate = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        _zero_real_projection(self.out_proj)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.in_proj(features))
        hidden = hidden * self.gate(features)
        return self.max_delta * torch.tanh(self.out_proj(hidden))


class PhaseFrequencyPriorExpert(_BoundedRealPriorExpert):
    """A phase-increment view for frequency/phase trajectory modulations."""

    def __init__(self, hidden_channels: int = 8, max_delta: float = 0.15, eps: float = 1e-8) -> None:
        super().__init__(hidden_channels=hidden_channels, input_channels=6, max_delta=max_delta)
        self.eps = float(eps)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        signal = torch.complex(x[:, 0].float(), x[:, 1].float())
        amplitude = signal.abs()
        amplitude_norm = amplitude / amplitude.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        amplitude_diff = F.pad(amplitude[..., 1:] - amplitude[..., :-1], (1, 0))

        if signal.size(-1) > 1:
            product = signal[..., 1:] * torch.conj(signal[..., :-1])
            magnitude = product.abs().clamp_min(self.eps)
            phase_cos = product.real / magnitude
            phase_sin = product.imag / magnitude
        else:
            phase_cos = signal.real.new_zeros(signal.size(0), 0)
            phase_sin = signal.real.new_zeros(signal.size(0), 0)

        ones = signal.real.new_ones(signal.size(0), 1)
        zeros = signal.real.new_zeros(signal.size(0), 1)
        phase_cos = torch.cat([ones, phase_cos], dim=-1)
        phase_sin = torch.cat([zeros, phase_sin], dim=-1)
        phase_cos_diff = F.pad(phase_cos[..., 1:] - phase_cos[..., :-1], (1, 0))
        phase_sin_diff = F.pad(phase_sin[..., 1:] - phase_sin[..., :-1], (1, 0))
        return torch.stack(
            [amplitude_norm, amplitude_diff, phase_cos, phase_sin, phase_cos_diff, phase_sin_diff],
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.project(self._features(x))
        return delta.to(dtype=x.dtype)


class PulseScalePriorExpert(_BoundedRealPriorExpert):
    """Learnable RRC-initialized and multiscale pulse/filterbank view."""

    def __init__(
        self,
        hidden_channels: int = 8,
        max_delta: float = 0.15,
        rolloffs: Sequence[float] = (0.2, 0.35, 0.5),
        kernel_size: int = 31,
    ) -> None:
        rolloffs = tuple(float(value) for value in rolloffs)
        super().__init__(
            hidden_channels=hidden_channels,
            input_channels=2 * len(rolloffs),
            max_delta=max_delta,
        )
        self.rolloffs = rolloffs
        self.rrc_bank = RRCBank(rolloffs=list(self.rolloffs), kernel_size=int(kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self.rrc_bank(x.float())
        features = torch.cat([real, imag], dim=1)
        delta = self.project(features)
        return delta.to(dtype=x.dtype)


class CyclicPriorExpert(nn.Module):
    """Dynamic multi-lag FRESH residual expert with a bounded output."""

    def __init__(
        self,
        hidden_channels: int = 8,
        max_delta: float = 0.15,
        top_k: int = 2,
        lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        min_freq: float = 1.0 / 128.0,
        max_freq: float = 0.25,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.top_k = max(1, int(top_k))
        self.max_delta = float(max_delta)
        self.estimator = MultiLagCyclicEstimator(
            lags=list(lags),
            min_freq=float(min_freq),
            max_freq=float(max_freq),
            top_k=self.top_k,
        )
        self.fresh = MultiBranchFRESH(top_k=self.top_k)
        num_branches = 1 + 4 * self.top_k
        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=num_branches,
            out_complex_channels=max(1, int(hidden_channels)),
            kernel_size=kernel_size,
            bias=True,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(2 * num_branches, max(1, int(hidden_channels)), kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(max(1, int(hidden_channels)), max(1, int(hidden_channels)), kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=max(1, int(hidden_channels)),
            out_complex_channels=1,
            kernel_size=1,
            bias=True,
        )
        _zero_complex_projection(self.out_proj)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            alphas, confidence = self.estimator(x.float())
        # Keep the phase arithmetic in fp32; the FRESH helper casts the
        # resulting phasors back to the input feature dtype.
        real, imag = self.fresh(x, alphas)
        hidden_real, hidden_imag = self.branch_filter(real, imag)
        gate = self.gate(torch.cat([real, imag], dim=1))
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = self.max_delta * torch.tanh(torch.cat([delta_real, delta_imag], dim=1))
        return delta, {
            "alphas": alphas.detach(),
            "confidence": confidence.detach(),
        }


class AdaptiveMultiViewPriorAdapter1D(nn.Module):
    """Fuse the prior experts through evidence-conditioned residual routing."""

    def __init__(
        self,
        input_channels: int = 2,
        evidence_lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        cyclic_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        cyclic_top_k: int = 2,
        cyclic_min_freq: float = 1.0 / 128.0,
        cyclic_max_freq: float = 0.25,
        pulse_rolloffs: Sequence[float] = (0.2, 0.35, 0.5),
        pulse_kernel_size: int = 31,
        hidden_channels: int = 8,
        router_hidden_channels: int = 16,
        identity_bias: float = 2.0,
        router_temperature: float = 1.0,
        max_delta: float = 0.15,
        max_scale: float = 0.2,
        scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("AdaptiveMultiViewPriorAdapter1D expects one I/Q mixture with 2 channels")
        self.max_scale = max(float(max_scale), 1e-6)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.evidence_extractor = AdaptiveViewEvidence(
            lag_bank=evidence_lags,
            min_freq=cyclic_min_freq,
            max_freq=cyclic_max_freq,
        )
        self.router = EvidenceRouter(
            evidence_dim=AdaptiveViewEvidence.evidence_dim,
            hidden_channels=router_hidden_channels,
            identity_bias=identity_bias,
            temperature=router_temperature,
        )
        self.cyclic_expert = CyclicPriorExpert(
            hidden_channels=hidden_channels,
            max_delta=max_delta,
            top_k=cyclic_top_k,
            lags=cyclic_lags,
            min_freq=cyclic_min_freq,
            max_freq=cyclic_max_freq,
        )
        self.phase_expert = PhaseFrequencyPriorExpert(
            hidden_channels=hidden_channels,
            max_delta=max_delta,
        )
        self.pulse_expert = PulseScalePriorExpert(
            hidden_channels=hidden_channels,
            max_delta=max_delta,
            rolloffs=pulse_rolloffs,
            kernel_size=pulse_kernel_size,
        )
        self.last_aux: dict[str, torch.Tensor] = {}

    def _bounded_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.residual_scale / self.max_scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected mixture IQ with shape [B, 2, L], got {tuple(x.shape)}")

        with torch.no_grad():
            evidence_info = self.evidence_extractor(x)
        route_weights = self.router(evidence_info["evidence"], evidence_info["confidence"])
        cyclic_delta, cyclic_aux = self.cyclic_expert(x)
        phase_delta = self.phase_expert(x)
        pulse_delta = self.pulse_expert(x)
        deltas = torch.stack([cyclic_delta, phase_delta, pulse_delta], dim=1)
        # [B, 3] -> [B, 3, 1, 1] so each expert weight broadcasts over
        # both the I/Q channel and time dimensions of [B, 3, 2, L].
        expert_weights = route_weights[:, 1:].to(dtype=deltas.dtype)[:, :, None, None]
        fused_delta = (deltas * expert_weights).sum(dim=1)
        scale = self._bounded_scale().to(dtype=fused_delta.dtype)
        adapted = x + scale * fused_delta.to(dtype=x.dtype)

        aux = {
            "route_weights": route_weights.detach(),
            "evidence": evidence_info["evidence"].detach(),
            "confidence": evidence_info["confidence"].detach(),
            "cyclic_alphas": cyclic_aux["alphas"].detach(),
            "cyclic_confidence": cyclic_aux["confidence"].detach(),
            "residual_scale": scale.detach().reshape(1),
        }
        self.last_aux = aux
        return adapted, aux


class PaddedStage4Backbone(nn.Module):
    """Pad odd-length inputs to the encoder stride and crop outputs back."""

    def __init__(self, backbone: nn.Module, nominal_length: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.nominal_length = int(nominal_length)

    def forward(self, x: torch.Tensor):
        original_length = x.size(-1)
        if original_length > self.nominal_length:
            raise ValueError(
                f"input length {original_length} exceeds configured length {self.nominal_length}"
            )
        if original_length < self.nominal_length:
            x = F.pad(x, (0, self.nominal_length - original_length), mode="replicate")
        output = self.backbone(x)
        if isinstance(output, (list, tuple)):
            return type(output)(item[..., :original_length] for item in output)
        return output[..., :original_length]


class IQUMamba1D_AdaptiveMultiViewPrior(nn.Module):
    """Stage-4 IQUMamba with a modulation-agnostic multi-view input adapter."""

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
        norm_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        adaptive_view_evidence_lags: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        adaptive_view_cyclic_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        adaptive_view_cyclic_top_k: int = 2,
        adaptive_view_cyclic_min_freq: float = 1.0 / 128.0,
        adaptive_view_cyclic_max_freq: float = 0.25,
        adaptive_view_pulse_rolloffs: Sequence[float] = (0.2, 0.35, 0.5),
        adaptive_view_pulse_kernel_size: int = 31,
        adaptive_view_hidden_channels: int = 8,
        adaptive_view_router_hidden_channels: int = 16,
        adaptive_view_identity_bias: float = 2.0,
        adaptive_view_router_temperature: float = 1.0,
        adaptive_view_max_delta: float = 0.15,
        adaptive_view_max_scale: float = 0.2,
        adaptive_view_scale_init: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("IQUMamba1D_AdaptiveMultiViewPrior expects input_channels=2")
        if int(num_classes) % 2 != 0:
            raise ValueError("num_classes must be even because each source has I/Q channels")
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        total_stride = 1
        for stride in strides:
            total_stride *= max(1, int(stride))
        backbone_input_size = ((int(input_size) + total_stride - 1) // total_stride) * total_stride

        self.adapter = AdaptiveMultiViewPriorAdapter1D(
            input_channels=input_channels,
            evidence_lags=adaptive_view_evidence_lags,
            cyclic_lags=adaptive_view_cyclic_lags,
            cyclic_top_k=adaptive_view_cyclic_top_k,
            cyclic_min_freq=adaptive_view_cyclic_min_freq,
            cyclic_max_freq=adaptive_view_cyclic_max_freq,
            pulse_rolloffs=adaptive_view_pulse_rolloffs,
            pulse_kernel_size=adaptive_view_pulse_kernel_size,
            hidden_channels=adaptive_view_hidden_channels,
            router_hidden_channels=adaptive_view_router_hidden_channels,
            identity_bias=adaptive_view_identity_bias,
            router_temperature=adaptive_view_router_temperature,
            max_delta=adaptive_view_max_delta,
            max_scale=adaptive_view_max_scale,
            scale_init=adaptive_view_scale_init,
        )
        raw_backbone = IQUMamba1D(
            input_size=backbone_input_size,
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
        self.backbone = PaddedStage4Backbone(raw_backbone, backbone_input_size)
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        adapted, aux = self.adapter(x)
        output = self.backbone(adapted)
        self.last_aux = aux
        return output
