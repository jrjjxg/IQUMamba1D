"""Communication-scale WaveNet experiments derived from Stage 261.

The modules in this file deliberately separate four hypotheses:

* anti-aliased low-rate Mamba context;
* ordered, phase-aware physical chunk tokens;
* time-varying Mamba controls instead of one frame-level control vector;
* symbol-clock-conditioned dilation routing, optionally preceded by a
  widely-linear complex stem.

Every model preserves the ordinary ``(B, 2, L) -> (B, 2*K, L)`` contract.
No transmitter metadata, source labels, modulation labels, or target-side
features are consumed by the model.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icassp_baseline_wavenet import (
    ICASPBaselineWaveNet,
    WaveNetResidualBlock,
    _kaiming_conv1d,
)
from models.icassp_wavenet_mamba import MambaSequenceMixer


def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.float().square().mean().add(eps).sqrt()


def _chunk_padding(length: int, chunk_size: int, chunk_hop: int) -> int:
    if length < chunk_size:
        return chunk_size - length
    remainder = (length - chunk_size) % chunk_hop
    return (chunk_hop - remainder) % chunk_hop


def _windowed_sinc_lowpass(
    downsample_factor: int,
    taps_per_phase: int = 8,
    cutoff_ratio: float = 0.90,
) -> torch.Tensor:
    """Return a unit-DC low-pass FIR suitable for decimation.

    ``cutoff_ratio`` is relative to the post-decimation Nyquist frequency.
    A Hann-windowed sinc is deterministic, differentiable with respect to its
    input, and avoids the unconstrained analysis filter in Stage 261 creating
    an aliased low-rate representation early in training.
    """

    factor = int(downsample_factor)
    if factor < 2:
        raise ValueError("downsample_factor must be at least 2")
    half = factor * int(taps_per_phase)
    positions = torch.arange(-half, half + 1, dtype=torch.float32)
    cutoff = float(cutoff_ratio) / factor
    kernel = cutoff * torch.sinc(cutoff * positions)
    window = torch.hann_window(kernel.numel(), periodic=False)
    kernel = kernel * window
    return kernel / kernel.sum().clamp_min(1e-8)


class WidelyLinearComplexStem(nn.Module):
    """Map one complex input to complex feature pairs using ``Wz + Vz*``.

    Output channels are laid out as ``[real_0..real_C, imag_0..imag_C]``.
    The conjugate branch starts at zero, so initialization is an ordinary
    phase-equivariant complex projection. It can subsequently learn mirror
    leakage and I/Q imbalance through ``V``.
    """

    def __init__(self, output_channels: int) -> None:
        super().__init__()
        if int(output_channels) < 2 or int(output_channels) % 2:
            raise ValueError("output_channels must be a positive even integer")
        self.complex_channels = int(output_channels) // 2
        shape = (self.complex_channels, 1, 1)
        self.w_real = nn.Parameter(torch.empty(shape))
        self.w_imag = nn.Parameter(torch.empty(shape))
        self.v_real = nn.Parameter(torch.zeros(shape))
        self.v_imag = nn.Parameter(torch.zeros(shape))
        self.bias_real = nn.Parameter(torch.zeros(self.complex_channels))
        self.bias_imag = nn.Parameter(torch.zeros(self.complex_channels))
        nn.init.kaiming_normal_(self.w_real)
        nn.init.kaiming_normal_(self.w_imag)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(
                "WidelyLinearComplexStem expects (B, 2, L), "
                f"got {tuple(mixture.shape)}"
            )
        i = mixture[:, 0:1]
        q = mixture[:, 1:2]
        # Wz
        real = F.conv1d(i, self.w_real) - F.conv1d(q, self.w_imag)
        imag = F.conv1d(i, self.w_imag) + F.conv1d(q, self.w_real)
        # Vz*
        real = real + F.conv1d(i, self.v_real) + F.conv1d(q, self.v_imag)
        imag = imag + F.conv1d(i, self.v_imag) - F.conv1d(q, self.v_real)
        real = real + self.bias_real.view(1, -1, 1)
        imag = imag + self.bias_imag.view(1, -1, 1)
        return torch.cat((real, imag), dim=1)

    def conjugate_ratio(self, eps: float = 1e-8) -> torch.Tensor:
        numerator = self.v_real.square().sum() + self.v_imag.square().sum()
        denominator = self.w_real.square().sum() + self.w_imag.square().sum()
        return (numerator / denominator.clamp_min(eps)).sqrt()


class ComplexModReLU(nn.Module):
    """Phase-equivariant magnitude activation for paired complex channels."""

    def __init__(self, channels: int, bias_init: float = 0.0) -> None:
        super().__init__()
        if int(channels) % 2:
            raise ValueError("channels must be even for paired complex activation")
        self.complex_channels = int(channels) // 2
        self.bias = nn.Parameter(torch.full((self.complex_channels,), float(bias_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = torch.chunk(x, 2, dim=1)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        scale = F.relu(magnitude + self.bias.view(1, -1, 1)) / magnitude
        return torch.cat((real * scale, imag * scale), dim=1)


class AntiAliasedMambaContext(nn.Module):
    """Low-rate Mamba residual with fixed anti-alias analysis and interpolation.

    Unlike Stage 261, this module does not use a randomly initialized strided
    convolution followed by a transposed convolution. The fixed FIR makes the
    low-rate signal well-defined, while a pointwise analysis/synthesis pair
    learns only channel mixing.
    """

    def __init__(
        self,
        channels: int,
        *,
        mamba_channels: int = 64,
        downsample_factor: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        scale_init: float = 0.01,
        taps_per_phase: int = 8,
        cutoff_ratio: float = 0.90,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.downsample_factor = int(downsample_factor)
        kernel = _windowed_sinc_lowpass(
            self.downsample_factor,
            taps_per_phase=taps_per_phase,
            cutoff_ratio=cutoff_ratio,
        )
        self.register_buffer("analysis_kernel", kernel.view(1, 1, -1))
        self.input_projection = nn.Conv1d(
            self.channels, int(mamba_channels), kernel_size=1
        )
        self.mamba = MambaSequenceMixer(
            int(mamba_channels),
            bidirectional=False,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.output_projection = nn.Conv1d(
            int(mamba_channels), self.channels, kernel_size=1
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.register_buffer(
            "last_context_ratio", torch.tensor(float("nan")), persistent=False
        )
        self.register_buffer(
            "last_scaled_context_ratio", torch.tensor(float("nan")), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_length = int(x.shape[-1])
        z = self.input_projection(x)
        channels = int(z.shape[1])
        kernel = self.analysis_kernel.to(device=z.device, dtype=z.dtype).expand(
            channels, 1, -1
        )
        padding = (kernel.shape[-1] - 1) // 2
        z = F.conv1d(
            z,
            kernel,
            stride=self.downsample_factor,
            padding=padding,
            groups=channels,
        )
        z = self.mamba(z)
        z = F.interpolate(z, size=original_length, mode="linear", align_corners=False)
        delta = self.dropout(self.output_projection(z))
        with torch.no_grad():
            base = _rms(x)
            ratio = _rms(delta) / base
            self.last_context_ratio.copy_(ratio.detach().to(self.last_context_ratio))
            self.last_scaled_context_ratio.copy_(
                (self.residual_scale.detach().abs() * ratio)
                .to(self.last_scaled_context_ratio)
            )
        return x + self.residual_scale * delta

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return {
            "context_residual_scale": self.residual_scale.detach(),
            "context_delta_to_local_rms": self.last_context_ratio.detach(),
            "scaled_context_to_local_rms": self.last_scaled_context_ratio.detach(),
        }


class ICASPAntiAliasedInterleavedMamba(ICASPBaselineWaveNet):
    """Stage 278: Stage 261 with an explicitly anti-aliased context path."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 128,
        residual_layers: int = 20,
        dilation_cycle_length: int = 10,
        *,
        mamba_insert_after_block: int = 10,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_scale_init: float = 0.01,
        antialias_taps_per_phase: int = 8,
        antialias_cutoff_ratio: float = 0.90,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.mamba_insert_after_block = int(mamba_insert_after_block)
        if not 1 <= self.mamba_insert_after_block < self.num_layers:
            raise ValueError(
                "mamba_insert_after_block must be between 1 and residual_layers - 1"
            )
        self.context = AntiAliasedMambaContext(
            residual_channels,
            mamba_channels=mamba_channels,
            downsample_factor=mamba_downsample_factor,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            scale_init=mamba_scale_init,
            taps_per_phase=antialias_taps_per_phase,
            cutoff_ratio=antialias_cutoff_ratio,
        )
        self.register_buffer(
            "last_pre_context_skip_fraction",
            torch.tensor(float("nan")),
            persistent=False,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(mixture))
        skip_sum = None
        pre_skip_energy = x.new_zeros(())
        total_skip_energy = x.new_zeros(())
        for block_index, block in enumerate(self.residual_blocks, start=1):
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
            energy = skip.float().square().mean()
            total_skip_energy = total_skip_energy + energy
            if block_index <= self.mamba_insert_after_block:
                pre_skip_energy = pre_skip_energy + energy
            if block_index == self.mamba_insert_after_block:
                x = self.context(x)
        with torch.no_grad():
            fraction = pre_skip_energy / total_skip_energy.clamp_min(1e-8)
            self.last_pre_context_skip_fraction.copy_(
                fraction.detach().to(self.last_pre_context_skip_fraction)
            )
        x = skip_sum / math.sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        values = self.context.diagnostics()
        values["pre_context_skip_energy_fraction"] = (
            self.last_pre_context_skip_fraction.detach()
        )
        return values


class OrderedPhysicalTokenizer(nn.Module):
    """Create ordered hidden-state and modulation-agnostic RF tokens.

    Physical features are computed only from the received mixture:

    * normalized complex autocorrelation at candidate lags;
    * centered envelope autocorrelation at the same lags;
    * RMS energy, normalized I/Q mean, fourth moment, and difference energy.

    These retain phase progression and symbol-clock evidence that mean/std
    chunk tokenizers discard.
    """

    def __init__(
        self,
        feature_channels: int,
        token_channels: int = 64,
        *,
        chunk_size: int = 64,
        chunk_hop: int = 32,
        physical_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.token_channels = int(token_channels)
        self.chunk_size = int(chunk_size)
        self.chunk_hop = int(chunk_hop)
        self.physical_lags = tuple(int(lag) for lag in physical_lags)
        self.eps = float(eps)
        if self.chunk_size < 4:
            raise ValueError("chunk_size must be at least 4")
        if not 1 <= self.chunk_hop <= self.chunk_size:
            raise ValueError("chunk_hop must be in [1, chunk_size]")
        if not self.physical_lags or min(self.physical_lags) < 1:
            raise ValueError("physical_lags must contain positive integers")
        if max(self.physical_lags) >= self.chunk_size:
            raise ValueError("physical_lags must be smaller than chunk_size")

        physical_dim = 5 + 3 * len(self.physical_lags)
        input_dim = 2 * self.feature_channels + physical_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Conv1d(input_dim, self.token_channels, kernel_size=1)

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        right_padding = _chunk_padding(
            int(x.shape[-1]), self.chunk_size, self.chunk_hop
        )
        if right_padding:
            x = F.pad(x, (0, right_padding))
        return x.unfold(-1, self.chunk_size, self.chunk_hop)

    def _physical_features(self, mixture_chunks: torch.Tensor) -> torch.Tensor:
        i = mixture_chunks[:, 0]
        q = mixture_chunks[:, 1]
        energy = i.square() + q.square()
        mean_energy = energy.mean(dim=-1).clamp_min(self.eps)
        rms = mean_energy.sqrt()
        centered_energy = energy - mean_energy.unsqueeze(-1)
        centered_norm = centered_energy.square().mean(dim=-1).clamp_min(self.eps)

        features = [
            mean_energy.log1p().unsqueeze(1),
            (i.mean(dim=-1) / rms).unsqueeze(1),
            (q.mean(dim=-1) / rms).unsqueeze(1),
            (
                energy.square().mean(dim=-1)
                / mean_energy.square().clamp_min(self.eps)
            ).unsqueeze(1),
            (
                (i[..., 1:] - i[..., :-1]).square().add(
                    (q[..., 1:] - q[..., :-1]).square()
                ).mean(dim=-1)
                / mean_energy
            ).unsqueeze(1),
        ]
        for lag in self.physical_lags:
            current_i, previous_i = i[..., lag:], i[..., :-lag]
            current_q, previous_q = q[..., lag:], q[..., :-lag]
            corr_real = (
                current_i * previous_i + current_q * previous_q
            ).mean(dim=-1) / mean_energy
            corr_imag = (
                current_q * previous_i - current_i * previous_q
            ).mean(dim=-1) / mean_energy
            envelope_corr = (
                centered_energy[..., lag:] * centered_energy[..., :-lag]
            ).mean(dim=-1) / centered_norm
            features.extend(
                (
                    corr_real.unsqueeze(1),
                    corr_imag.unsqueeze(1),
                    envelope_corr.unsqueeze(1),
                )
            )
        return torch.cat(features, dim=1)

    def forward(
        self, feature: torch.Tensor, mixture: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(
                "OrderedPhysicalTokenizer expects mixture (B, 2, L), "
                f"got {tuple(mixture.shape)}"
            )
        feature_chunks = self._unfold(feature)
        mixture_chunks = self._unfold(mixture)
        feature_mean = feature_chunks.mean(dim=-1)
        feature_std = feature_chunks.float().var(
            dim=-1, unbiased=False
        ).add(self.eps).sqrt().to(feature.dtype)
        physical = self._physical_features(mixture_chunks).to(feature.dtype)
        combined = torch.cat((feature_mean, feature_std, physical), dim=1)
        combined = self.input_norm(combined.transpose(1, 2)).transpose(1, 2)
        return self.projection(combined), physical


class TemporalPhysicalMambaController(nn.Module):
    """Return ordered late-block controls and symbol-period probabilities."""

    def __init__(
        self,
        feature_channels: int,
        controlled_blocks: int,
        *,
        token_channels: int = 64,
        chunk_size: int = 64,
        chunk_hop: int = 32,
        physical_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        candidate_periods: Sequence[int] = (2, 4, 8, 16, 32),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        evidence_strength: float = 1.0,
        router_temperature: float = 0.35,
    ) -> None:
        super().__init__()
        self.controlled_blocks = int(controlled_blocks)
        self.candidate_periods = tuple(int(x) for x in candidate_periods)
        self.evidence_strength = float(evidence_strength)
        self.router_temperature = float(router_temperature)
        if self.controlled_blocks < 1:
            raise ValueError("controlled_blocks must be positive")
        if not self.candidate_periods:
            raise ValueError("candidate_periods cannot be empty")
        if max(self.candidate_periods) >= int(chunk_size):
            raise ValueError("candidate_periods must be smaller than chunk_size")

        union_lags = tuple(
            sorted(set(int(x) for x in physical_lags) | set(self.candidate_periods))
        )
        self.tokenizer = OrderedPhysicalTokenizer(
            feature_channels,
            token_channels,
            chunk_size=chunk_size,
            chunk_hop=chunk_hop,
            physical_lags=union_lags,
        )
        self.physical_lags = union_lags
        self.mamba = MambaSequenceMixer(
            int(token_channels),
            bidirectional=False,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        # Four sample-specific scalar controls per late block:
        # residual gate, skip gate, FiLM gamma, FiLM beta.
        self.control_head = nn.Conv1d(
            int(token_channels), 4 * self.controlled_blocks, kernel_size=1
        )
        self.period_head = nn.Conv1d(
            int(token_channels), len(self.candidate_periods), kernel_size=1
        )
        nn.init.zeros_(self.control_head.weight)
        nn.init.zeros_(self.control_head.bias)
        nn.init.zeros_(self.period_head.weight)
        nn.init.zeros_(self.period_head.bias)
        lag_to_index = {lag: index for index, lag in enumerate(self.physical_lags)}
        self.period_feature_indices = tuple(
            lag_to_index[period] for period in self.candidate_periods
        )
        self.register_buffer(
            "last_period_probabilities",
            torch.full((len(self.candidate_periods),), 1.0 / len(self.candidate_periods)),
            persistent=False,
        )
        self.register_buffer(
            "last_router_entropy", torch.tensor(float("nan")), persistent=False
        )
        self.register_buffer(
            "last_control_magnitude", torch.tensor(float("nan")), persistent=False
        )

    def forward(
        self,
        feature: torch.Tensor,
        mixture: torch.Tensor,
        *,
        output_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, physical = self.tokenizer(feature, mixture)
        tokens = self.dropout(self.mamba(tokens))
        controls = self.control_head(tokens)
        controls = F.interpolate(
            controls, size=int(output_length), mode="linear", align_corners=False
        )
        controls = controls.view(
            controls.shape[0], self.controlled_blocks, 4, int(output_length)
        )

        # Each lag contributes [complex-real, complex-imag, envelope-corr].
        evidence = []
        base = 5
        for lag_index in self.period_feature_indices:
            offset = base + 3 * lag_index
            corr_real = physical[:, offset]
            corr_imag = physical[:, offset + 1]
            envelope_corr = physical[:, offset + 2]
            evidence.append(
                torch.sqrt(corr_real.square() + corr_imag.square() + 1e-8)
                + envelope_corr.abs()
            )
        evidence_logits = torch.stack(evidence, dim=1)
        learned_logits = self.period_head(tokens)
        logits = learned_logits + self.evidence_strength * evidence_logits
        period_probabilities = F.softmax(
            logits / max(self.router_temperature, 1e-4), dim=1
        )
        period_probabilities = F.interpolate(
            period_probabilities,
            size=int(output_length),
            mode="linear",
            align_corners=False,
        )
        period_probabilities = period_probabilities / period_probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)

        with torch.no_grad():
            mean_probs = period_probabilities.mean(dim=(0, 2))
            entropy = -(
                period_probabilities.clamp_min(1e-8)
                * period_probabilities.clamp_min(1e-8).log()
            ).sum(dim=1).mean()
            entropy = entropy / math.log(len(self.candidate_periods))
            self.last_period_probabilities.copy_(
                mean_probs.detach().to(self.last_period_probabilities)
            )
            self.last_router_entropy.copy_(
                entropy.detach().to(self.last_router_entropy)
            )
            self.last_control_magnitude.copy_(
                controls.detach().abs().mean().to(self.last_control_magnitude)
            )
        return controls, period_probabilities

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        values: Dict[str, torch.Tensor] = {
            "symbol_router_entropy": self.last_router_entropy.detach(),
            "temporal_control_abs_mean": self.last_control_magnitude.detach(),
        }
        for period, probability in zip(
            self.candidate_periods, self.last_period_probabilities
        ):
            values[f"period_probability_sps_{period}"] = probability.detach()
        return values


def _apply_temporal_controls(
    previous: torch.Tensor,
    candidate: torch.Tensor,
    skip: torch.Tensor,
    controls: torch.Tensor,
    *,
    gate_max_delta: float,
    film_max_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual_raw, skip_raw, gamma_raw, beta_raw = controls.unbind(dim=1)
    residual_gate = 1.0 + float(gate_max_delta) * torch.tanh(residual_raw)
    skip_gate = 1.0 + float(gate_max_delta) * torch.tanh(skip_raw)
    gamma = float(film_max_delta) * torch.tanh(gamma_raw)
    beta = float(film_max_delta) * torch.tanh(beta_raw)
    residual_update = math.sqrt(2.0) * candidate - previous
    output = (
        previous + residual_gate.unsqueeze(1) * residual_update
    ) / math.sqrt(2.0)
    local_rms = previous.detach().float().square().mean(
        dim=1, keepdim=True
    ).add(1e-8).sqrt().to(previous.dtype)
    output = output * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1) * local_rms
    return output, skip * skip_gate.unsqueeze(1)


class ICASPTemporalPhysicalControllerWaveNet(ICASPBaselineWaveNet):
    """Stage 279: fixed-dilation WaveNet with ordered time-varying controls."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 128,
        residual_layers: int = 20,
        dilation_cycle_length: int = 10,
        *,
        controller_insert_after_block: int = 10,
        token_channels: int = 64,
        chunk_size: int = 64,
        chunk_hop: int = 32,
        physical_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        candidate_periods: Sequence[int] = (2, 4, 8, 16, 32),
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        control_gate_max_delta: float = 0.5,
        control_film_max_delta: float = 0.1,
        evidence_strength: float = 1.0,
        router_temperature: float = 0.35,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.controller_insert_after_block = int(controller_insert_after_block)
        controlled_blocks = self.num_layers - self.controller_insert_after_block
        if not 1 <= self.controller_insert_after_block < self.num_layers:
            raise ValueError(
                "controller_insert_after_block must be within the residual stack"
            )
        self.controller = TemporalPhysicalMambaController(
            residual_channels,
            controlled_blocks,
            token_channels=token_channels,
            chunk_size=chunk_size,
            chunk_hop=chunk_hop,
            physical_lags=physical_lags,
            candidate_periods=candidate_periods,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            evidence_strength=evidence_strength,
            router_temperature=router_temperature,
        )
        self.control_gate_max_delta = float(control_gate_max_delta)
        self.control_film_max_delta = float(control_film_max_delta)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(mixture))
        skip_sum = None
        controls = None
        for block_index, block in enumerate(self.residual_blocks, start=1):
            previous = x
            candidate, skip = block(x)
            if block_index == self.controller_insert_after_block:
                controls, _ = self.controller(
                    candidate, mixture, output_length=mixture.shape[-1]
                )
                x = candidate
            elif block_index > self.controller_insert_after_block:
                controlled_index = (
                    block_index - self.controller_insert_after_block - 1
                )
                x, skip = _apply_temporal_controls(
                    previous,
                    candidate,
                    skip,
                    controls[:, controlled_index],
                    gate_max_delta=self.control_gate_max_delta,
                    film_max_delta=self.control_film_max_delta,
                )
            else:
                x = candidate
            skip_sum = skip if skip_sum is None else skip_sum + skip
        x = skip_sum / math.sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return self.controller.diagnostics()


class DepthwiseMultiDilationResidualBlock(nn.Module):
    """Efficient WaveNet block with soft routing across physical dilations."""

    def __init__(self, channels: int, dilations: Iterable[int]) -> None:
        super().__init__()
        self.channels = int(channels)
        self.dilations = tuple(max(1, int(dilation)) for dilation in dilations)
        if not self.dilations:
            raise ValueError("dilations cannot be empty")
        self.depthwise = nn.ModuleList(
            [
                _kaiming_conv1d(
                    self.channels,
                    self.channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=self.channels,
                )
                for dilation in self.dilations
            ]
        )
        self.gate_projections = nn.ModuleList(
            [
                _kaiming_conv1d(
                    self.channels, 2 * self.channels, kernel_size=1
                )
                for _ in self.dilations
            ]
        )
        self.output_projections = nn.ModuleList(
            [
                _kaiming_conv1d(
                    self.channels, 2 * self.channels, kernel_size=1
                )
                for _ in self.dilations
            ]
        )

    def forward(
        self, x: torch.Tensor, route_probabilities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if route_probabilities.shape[1] != len(self.dilations):
            raise ValueError(
                f"Expected {len(self.dilations)} route probabilities, "
                f"got {route_probabilities.shape[1]}"
            )
        residual_sum = torch.zeros_like(x)
        skip_sum = torch.zeros_like(x)
        for index, (depthwise, gate_projection, output_projection) in enumerate(
            zip(self.depthwise, self.gate_projections, self.output_projections)
        ):
            hidden = gate_projection(depthwise(x))
            gate, filt = torch.chunk(hidden, 2, dim=1)
            hidden = torch.sigmoid(gate) * torch.tanh(filt)
            residual, skip = torch.chunk(output_projection(hidden), 2, dim=1)
            weight = route_probabilities[:, index : index + 1]
            residual_sum = residual_sum + weight * residual
            skip_sum = skip_sum + weight * skip
        return (x + residual_sum) / math.sqrt(2.0), skip_sum


class ICASPSymbolClockWaveNet(nn.Module):
    """Stages 280-282: physical-clock routing with optional complex/control paths."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 128,
        *,
        pre_residual_layers: int = 10,
        pre_dilation_cycle_length: int = 10,
        adaptive_layers: int = 5,
        candidate_periods: Sequence[int] = (2, 4, 8, 16, 32),
        dilation_multipliers: Sequence[float] = (1, 2, 4, 8, 16),
        max_dilation: int = 512,
        use_widely_linear_stem: bool = False,
        use_temporal_controls: bool = False,
        token_channels: int = 64,
        chunk_size: int = 64,
        chunk_hop: int = 32,
        physical_lags: Sequence[int] = (1, 2, 4, 8, 16, 32),
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        control_gate_max_delta: float = 0.5,
        control_film_max_delta: float = 0.1,
        evidence_strength: float = 1.0,
        router_temperature: float = 0.35,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("ICASPSymbolClockWaveNet currently expects raw I/Q input")
        self.num_layers = int(pre_residual_layers) + int(adaptive_layers)
        self.use_widely_linear_stem = bool(use_widely_linear_stem)
        self.use_temporal_controls = bool(use_temporal_controls)
        self.control_gate_max_delta = float(control_gate_max_delta)
        self.control_film_max_delta = float(control_film_max_delta)
        self.candidate_periods = tuple(int(x) for x in candidate_periods)
        if int(adaptive_layers) < 1 or int(pre_residual_layers) < 1:
            raise ValueError("pre_residual_layers and adaptive_layers must be positive")
        multipliers = tuple(float(x) for x in dilation_multipliers)
        if len(multipliers) < int(adaptive_layers):
            raise ValueError("dilation_multipliers must cover every adaptive layer")

        if self.use_widely_linear_stem:
            self.input_projection = WidelyLinearComplexStem(residual_channels)
            self.input_activation = ComplexModReLU(residual_channels)
        else:
            self.input_projection = _kaiming_conv1d(
                input_channels, residual_channels, kernel_size=1
            )
            self.input_activation = nn.ReLU()
        self.pre_blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(
                    residual_channels,
                    dilation=2 ** (index % int(pre_dilation_cycle_length)),
                )
                for index in range(int(pre_residual_layers))
            ]
        )
        adaptive_blocks = []
        for layer_index in range(int(adaptive_layers)):
            multiplier = multipliers[layer_index]
            dilations = [
                min(
                    int(max_dilation),
                    max(1, int(round(period * multiplier))),
                )
                for period in self.candidate_periods
            ]
            adaptive_blocks.append(
                DepthwiseMultiDilationResidualBlock(
                    residual_channels, dilations=dilations
                )
            )
        self.adaptive_blocks = nn.ModuleList(adaptive_blocks)
        self.controller = TemporalPhysicalMambaController(
            residual_channels,
            int(adaptive_layers),
            token_channels=token_channels,
            chunk_size=chunk_size,
            chunk_hop=chunk_hop,
            physical_lags=physical_lags,
            candidate_periods=self.candidate_periods,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            evidence_strength=evidence_strength,
            router_temperature=router_temperature,
        )
        self.skip_projection = _kaiming_conv1d(
            residual_channels, residual_channels, kernel_size=1
        )
        self.output_projection = _kaiming_conv1d(
            residual_channels, num_classes, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        x = self.input_activation(self.input_projection(mixture))
        skip_sum = None
        for block in self.pre_blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        controls, period_probabilities = self.controller(
            x, mixture, output_length=mixture.shape[-1]
        )
        for block_index, block in enumerate(self.adaptive_blocks):
            previous = x
            candidate, skip = block(x, period_probabilities)
            if self.use_temporal_controls:
                x, skip = _apply_temporal_controls(
                    previous,
                    candidate,
                    skip,
                    controls[:, block_index],
                    gate_max_delta=self.control_gate_max_delta,
                    film_max_delta=self.control_film_max_delta,
                )
            else:
                x = candidate
            skip_sum = skip_sum + skip
        x = skip_sum / math.sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        values = self.controller.diagnostics()
        if isinstance(self.input_projection, WidelyLinearComplexStem):
            values["widely_linear_conjugate_ratio"] = (
                self.input_projection.conjugate_ratio().detach()
            )
        return values

    def no_weight_decay(self) -> set[str]:
        return set()
