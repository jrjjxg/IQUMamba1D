"""Stage-12 Mamba/attention ablations with RF-aware compact context.

The attention layouts follow the public reference implementations rather than
copying task-specific model code:

* MambaVision: pre-norm QKV attention in later/deeper stages.
* Hymba: Mamba and QKV attention consume the same feature sequence in parallel.
* complex-valued-transformer: bias-free complex Q/K/V projections and the real
  Hermitian inner product for real-valued attention weights.

The RF token extractors are local adaptations.  They reuse the same cyclic
frequency statistic and complex-IQ conventions as the existing Stage-79/80
implementations, while keeping the KV bank compact.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUBiMamba1D import IQUBiMamba1D


def _require_iq(x: torch.Tensor) -> None:
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected one complex I/Q mixture shaped (B, 2, L), got {tuple(x.shape)}")


def _as_complex_iq(x: torch.Tensor) -> torch.Tensor:
    _require_iq(x)
    return torch.complex(x[:, 0].float(), x[:, 1].float())


class ComplexLinearPair(nn.Module):
    """Bias-free complex linear projection implemented with paired real weights."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight_real = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_imag = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.weight_real, gain=gain)
        nn.init.xavier_uniform_(self.weight_imag, gain=gain)

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = F.linear(real, self.weight_real) - F.linear(imag, self.weight_imag)
        out_imag = F.linear(real, self.weight_imag) + F.linear(imag, self.weight_real)
        return out_real, out_imag


class ComplexRMSNormPair(nn.Module):
    """Magnitude-only normalization, preserving a common complex rotation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(int(dim)))

    def forward(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_rms = (real.square() + imag.square()).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(inv_rms + self.eps)
        gain = self.gain.to(dtype=real.dtype)
        return real * inv_rms * gain, imag * inv_rms * gain


class GlobalPhaseCanonicalizer1D(nn.Module):
    """Canonicalize a common I/Q phase and restore it on source-pair outputs.

    The maximum-magnitude sample is invariant to a common phase rotation, so
    its unit phasor is a deterministic covariant anchor.  An arbitrary real
    backbone can therefore operate in the canonical frame while the complete
    wrapped model remains equivariant to a global phase rotation.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def canonicalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_iq(x)
        z = torch.complex(x[:, 0].float(), x[:, 1].float())
        anchor_index = z.abs().argmax(dim=-1, keepdim=True)
        anchor = z.gather(1, anchor_index).squeeze(1)
        unit_phase = anchor / anchor.abs().clamp_min(self.eps)
        canonical = z * unit_phase.conj().unsqueeze(-1)
        canonical_iq = torch.stack([canonical.real, canonical.imag], dim=1)
        return canonical_iq.to(dtype=x.dtype), unit_phase

    def restore(self, output: torch.Tensor, unit_phase: torch.Tensor) -> torch.Tensor:
        if output.dim() != 3 or output.size(1) % 2 != 0:
            raise ValueError(
                "Phase restoration expects source outputs shaped (B, 2 * num_sources, L)"
            )
        batch, channels, length = output.shape
        pairs = output.reshape(batch, channels // 2, 2, length).float()
        sources = torch.complex(pairs[:, :, 0], pairs[:, :, 1])
        restored = sources * unit_phase.unsqueeze(1).unsqueeze(2)
        restored_pairs = torch.stack([restored.real, restored.imag], dim=2)
        return restored_pairs.reshape(batch, channels, length).to(dtype=output.dtype)


class ComplexHermitianInputFusion1D(nn.Module):
    """Phase-equivariant compact-KV attention over a raw complex waveform.

    For a common rotation ``x -> exp(j theta) x``, bias-free complex Q/K/V
    projections rotate identically.  ``Re(Q K^H)`` is therefore invariant,
    while the value-weighted output is equivariant.
    """

    def __init__(
        self,
        kv_tokens: int = 32,
        num_heads: int = 4,
        head_dim: int = 8,
        dropout: float = 0.0,
        residual_scale_init: float | None = 0.01,
    ) -> None:
        super().__init__()
        if int(kv_tokens) < 1:
            raise ValueError("kv_tokens must be positive")
        if int(num_heads) < 1 or int(head_dim) < 1:
            raise ValueError("num_heads and head_dim must be positive")
        self.kv_tokens = int(kv_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = float(dropout)

        self.to_q = ComplexLinearPair(1, self.inner_dim)
        self.to_k = ComplexLinearPair(1, self.inner_dim)
        self.to_v = ComplexLinearPair(1, self.inner_dim)
        self.q_norm = ComplexRMSNormPair(self.head_dim)
        self.k_norm = ComplexRMSNormPair(self.head_dim)
        self.to_out = ComplexLinearPair(self.inner_dim, 1)
        if residual_scale_init is None:
            self.register_parameter("residual_scale", None)
        else:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_iq(x)
        dtype = x.dtype
        real = x[:, 0].float()
        imag = x[:, 1].float()
        query_real = real.unsqueeze(-1)
        query_imag = imag.unsqueeze(-1)

        token_count = min(self.kv_tokens, int(x.size(-1)))
        context_real = F.adaptive_avg_pool1d(real.unsqueeze(1), token_count).transpose(1, 2)
        context_imag = F.adaptive_avg_pool1d(imag.unsqueeze(1), token_count).transpose(1, 2)

        q_real, q_imag = self.to_q(query_real, query_imag)
        k_real, k_imag = self.to_k(context_real, context_imag)
        v_real, v_imag = self.to_v(context_real, context_imag)
        q_real, q_imag = self._heads(q_real), self._heads(q_imag)
        k_real, k_imag = self._heads(k_real), self._heads(k_imag)
        v_real, v_imag = self._heads(v_real), self._heads(v_imag)
        q_real, q_imag = self.q_norm(q_real, q_imag)
        k_real, k_imag = self.k_norm(k_real, k_imag)

        scores = torch.matmul(q_real, k_real.transpose(-2, -1))
        scores = scores + torch.matmul(q_imag, k_imag.transpose(-2, -1))
        weights = torch.softmax(scores * (self.head_dim ** -0.5), dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        out_real = torch.matmul(weights, v_real)
        out_imag = torch.matmul(weights, v_imag)
        out_real = out_real.transpose(1, 2).reshape(x.size(0), x.size(-1), self.inner_dim)
        out_imag = out_imag.transpose(1, 2).reshape(x.size(0), x.size(-1), self.inner_dim)
        out_real, out_imag = self.to_out(out_real, out_imag)
        delta = torch.cat([out_real.transpose(1, 2), out_imag.transpose(1, 2)], dim=1)
        scale = self.residual_scale.to(device=x.device, dtype=dtype)
        return x + scale * delta.to(dtype=dtype)


def _parse_ints(values: Sequence[int] | str, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(values, str):
        parsed = tuple(int(part.strip()) for part in values.split(",") if part.strip())
    else:
        parsed = tuple(int(value) for value in values)
    parsed = tuple(value for value in parsed if value >= 0)
    return parsed or fallback


class TimeDomainPhysicalTokenExtractor(nn.Module):
    """Build compact cyclic, CFO, polyphase and symbol-phase tokens from I/Q."""

    token_dim = 8

    def __init__(
        self,
        cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        polyphase_branches: int = 8,
        symbol_orders: Sequence[int] | str = (2, 4, 8),
        min_cyclic_freq: float = 1.0 / 64.0,
        max_cyclic_freq: float = 1.0 / 8.0,
        cyclic_temperature: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.cyclic_lags = _parse_ints(cyclic_lags, (0, 1, 2, 4, 8))
        self.polyphase_branches = int(polyphase_branches)
        self.symbol_orders = tuple(
            value for value in _parse_ints(symbol_orders, (2, 4, 8)) if value > 0
        )
        if self.polyphase_branches < 1:
            raise ValueError("polyphase_branches must be positive")
        if not self.symbol_orders:
            raise ValueError("symbol_orders must contain a positive order")
        self.min_cyclic_freq = float(min_cyclic_freq)
        self.max_cyclic_freq = float(max_cyclic_freq)
        self.cyclic_temperature = max(float(cyclic_temperature), 1e-3)
        self.eps = float(eps)
        self.token_count = (
            2 * len(self.cyclic_lags)
            + self.polyphase_branches
            + len(self.symbol_orders)
            + 2
        )

    def _soft_cyclic_frequency(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(z.size(-1))
        if length < 8:
            default = z.real.new_full((z.size(0),), 1.0 / 32.0)
            return default, torch.zeros_like(default)
        envelope = z.abs().square()
        envelope = envelope - envelope.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(envelope, dim=-1)
        power = spectrum.abs().square()
        freqs = torch.fft.rfftfreq(length, d=1.0, device=z.device)
        mask = (freqs >= self.min_cyclic_freq) & (freqs <= self.max_cyclic_freq)
        if not bool(mask.any()):
            default = z.real.new_full((z.size(0),), 1.0 / 32.0)
            return default, torch.zeros_like(default)
        selected = power[:, mask]
        selected_freqs = freqs[mask]
        logits = torch.log(selected + self.eps) / self.cyclic_temperature
        weights = torch.softmax(logits, dim=-1)
        estimate = (weights * selected_freqs).sum(dim=-1)
        reliability = selected.max(dim=-1).values / selected.sum(dim=-1).clamp_min(self.eps)
        return estimate, reliability.clamp(0.0, 1.0)

    def _common_statistics(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        power = z.abs().square().mean(dim=-1).clamp_min(self.eps)
        envelope = z.abs().square()
        envelope_cv = envelope.std(dim=-1, unbiased=False) / power
        if z.size(-1) > 1:
            lag_product = z[:, 1:] * z[:, :-1].conj()
            lag_denom = (z[:, 1:].abs() * z[:, :-1].abs()).mean(dim=-1).clamp_min(self.eps)
            lag_corr = lag_product.mean(dim=-1) / lag_denom
        else:
            lag_corr = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        cfo_unit = lag_corr / lag_corr.abs().clamp_min(self.eps)
        return {
            "power": power,
            "envelope_cv": envelope_cv.clamp(0.0, 4.0) / 4.0,
            "cfo_real": cfo_unit.real,
            "cfo_imag": cfo_unit.imag,
            "cfo_coherence": lag_corr.abs().clamp(0.0, 1.0),
            "iq_balance": (
                x[:, 0].float().square().mean(dim=-1)
                - x[:, 1].float().square().mean(dim=-1)
            ) / power,
            "iq_corr": 2.0 * (x[:, 0].float() * x[:, 1].float()).mean(dim=-1) / power,
        }

    def _cyclic_tokens(
        self,
        z: torch.Tensor,
        base_freq: torch.Tensor,
        reliability: torch.Tensor,
        stats: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        length = int(z.size(-1))
        n = torch.arange(length, device=z.device, dtype=torch.float32)
        max_lag = max(max(self.cyclic_lags), 1)
        for multiplier in (1.0, 2.0):
            alpha = (multiplier * base_freq).clamp(0.0, 0.5)
            phasor = torch.exp(-1j * 2.0 * math.pi * alpha.unsqueeze(1) * n.unsqueeze(0))
            for lag in self.cyclic_lags:
                if lag >= length:
                    corr = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
                elif lag == 0:
                    corr = (z * z.conj() * phasor).mean(dim=-1)
                else:
                    corr = (
                        z[:, lag:] * z[:, :-lag].conj() * phasor[:, lag:]
                    ).mean(dim=-1)
                corr = corr / stats["power"]
                tokens.append(torch.stack([
                    corr.real,
                    corr.imag,
                    corr.abs(),
                    2.0 * alpha,
                    z.real.new_full((z.size(0),), float(lag) / float(max_lag)),
                    reliability,
                    stats["cfo_real"],
                    stats["cfo_imag"],
                ], dim=-1))
        return tokens

    def _polyphase_tokens(
        self,
        z: torch.Tensor,
        stats: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        period = self.polyphase_branches
        for offset in range(period):
            branch = z[:, offset::period]
            branch_power = branch.abs().square().mean(dim=-1).clamp_min(self.eps)
            unit_mean = (branch / branch.abs().clamp_min(self.eps)).mean(dim=-1)
            if branch.size(-1) > 1:
                lag = (branch[:, 1:] * branch[:, :-1].conj()).mean(dim=-1)
                lag = lag / branch_power
            else:
                lag = torch.zeros(branch.size(0), device=z.device, dtype=z.dtype)
            phase = 2.0 * math.pi * float(offset) / float(period)
            tokens.append(torch.stack([
                torch.log((branch_power / stats["power"]).clamp_min(self.eps)).clamp(-4.0, 4.0) / 4.0,
                unit_mean.real,
                unit_mean.imag,
                unit_mean.abs(),
                lag.real,
                lag.imag,
                z.real.new_full((z.size(0),), math.sin(phase)),
                z.real.new_full((z.size(0),), math.cos(phase)),
            ], dim=-1))
        return tokens

    def _symbol_tokens(
        self,
        z: torch.Tensor,
        reliability: torch.Tensor,
        stats: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        unit = z / z.abs().clamp_min(self.eps)
        max_order = max(self.symbol_orders)
        tokens: list[torch.Tensor] = []
        for order in self.symbol_orders:
            moment = unit.pow(order).mean(dim=-1)
            tokens.append(torch.stack([
                moment.real,
                moment.imag,
                moment.abs(),
                z.real.new_full((z.size(0),), float(order) / float(max_order)),
                stats["cfo_real"],
                stats["cfo_imag"],
                stats["envelope_cv"],
                reliability,
            ], dim=-1))
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        z = _as_complex_iq(x)
        stats = self._common_statistics(x, z)
        base_freq, reliability = self._soft_cyclic_frequency(z)
        tokens = self._cyclic_tokens(z, base_freq, reliability, stats)
        tokens.extend(self._polyphase_tokens(z, stats))
        tokens.extend(self._symbol_tokens(z, reliability, stats))
        tokens.append(torch.stack([
            torch.log(stats["power"]).clamp(-12.0, 12.0) / 12.0,
            stats["envelope_cv"],
            stats["iq_balance"],
            stats["iq_corr"],
            stats["cfo_real"],
            stats["cfo_imag"],
            stats["cfo_coherence"],
            reliability,
        ], dim=-1))
        tokens.append(torch.stack([
            2.0 * base_freq,
            reliability,
            stats["cfo_real"],
            stats["cfo_imag"],
            stats["cfo_coherence"],
            stats["envelope_cv"],
            stats["iq_balance"],
            stats["iq_corr"],
        ], dim=-1))
        output = torch.stack(tokens, dim=1)
        if output.size(1) != self.token_count:
            raise RuntimeError(f"Physical token count changed: {output.size(1)} != {self.token_count}")
        return output.to(dtype=original_dtype)


class RFMultiDomainTokenExtractor(nn.Module):
    """Add compact complex-STFT subband and local phase-difference tokens."""

    token_dim = TimeDomainPhysicalTokenExtractor.token_dim

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        num_subbands: int = 8,
        temporal_tokens: int = 4,
        cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        polyphase_branches: int = 8,
        symbol_orders: Sequence[int] | str = (2, 4, 8),
        min_cyclic_freq: float = 1.0 / 64.0,
        max_cyclic_freq: float = 1.0 / 8.0,
        cyclic_temperature: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.num_subbands = int(num_subbands)
        self.temporal_tokens = int(temporal_tokens)
        self.eps = float(eps)
        if self.n_fft < 2 or self.hop_length < 1:
            raise ValueError("n_fft must be >= 2 and hop_length must be positive")
        if not 1 <= self.win_length <= self.n_fft:
            raise ValueError("win_length must be in [1, n_fft]")
        if self.num_subbands < 1 or self.temporal_tokens < 1:
            raise ValueError("num_subbands and temporal_tokens must be positive")
        self.time_tokens = TimeDomainPhysicalTokenExtractor(
            cyclic_lags=cyclic_lags,
            polyphase_branches=polyphase_branches,
            symbol_orders=symbol_orders,
            min_cyclic_freq=min_cyclic_freq,
            max_cyclic_freq=max_cyclic_freq,
            cyclic_temperature=cyclic_temperature,
            eps=eps,
        )
        self.token_count = self.time_tokens.token_count + self.num_subbands + self.temporal_tokens
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=True))

    def _stft_tokens(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(-1) < self.n_fft:
            z = F.pad(z, (0, self.n_fft - int(z.size(-1))))
        window = self.window.to(device=z.device, dtype=z.real.dtype)
        spectrum = torch.stft(
            z,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            onesided=False,
            return_complex=True,
        )
        spectrum = torch.fft.fftshift(spectrum, dim=1)
        global_power = spectrum.abs().square().mean(dim=(1, 2)).clamp_min(self.eps)
        frequencies = torch.fft.fftshift(
            torch.fft.fftfreq(self.n_fft, d=1.0, device=z.device)
        )
        band_tokens: list[torch.Tensor] = []
        for indices in torch.tensor_split(torch.arange(self.n_fft, device=z.device), self.num_subbands):
            band = spectrum.index_select(1, indices)
            band_power_map = band.abs().square()
            band_power = band_power_map.mean(dim=(1, 2)).clamp_min(self.eps)
            variation = band_power_map.std(dim=(1, 2), unbiased=False) / band_power
            flatness = torch.exp(torch.log(band_power_map + self.eps).mean(dim=(1, 2))) / band_power
            if band.size(-1) > 1:
                phase_step = (band[:, :, 1:] * band[:, :, :-1].conj()).mean(dim=(1, 2))
                phase_step = phase_step / phase_step.abs().clamp_min(self.eps)
            else:
                phase_step = torch.zeros(band.size(0), device=z.device, dtype=z.dtype)
            phase_concentration = (band / band.abs().clamp_min(self.eps)).mean(dim=(1, 2))
            center_frequency = frequencies.index_select(0, indices).mean()
            band_tokens.append(torch.stack([
                torch.log((band_power / global_power).clamp_min(self.eps)).clamp(-6.0, 6.0) / 6.0,
                variation.clamp(0.0, 4.0) / 4.0,
                flatness.clamp(0.0, 1.0),
                phase_step.real,
                phase_step.imag,
                phase_concentration.real,
                phase_concentration.imag,
                center_frequency.to(dtype=z.real.dtype).expand(z.size(0)),
            ], dim=-1))
        return torch.stack(band_tokens, dim=1)

    def _temporal_tokens(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(-1) < 2:
            z = F.pad(z, (0, 2 - int(z.size(-1))))
        current = z[:, 1:]
        previous = z[:, :-1]
        amplitude = current.abs().clamp_min(self.eps)
        mean_amplitude = amplitude.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        phase_step = current * previous.conj()
        phase_unit = phase_step / phase_step.abs().clamp_min(self.eps)
        signal_unit = current / amplitude
        features = torch.stack([
            torch.log((amplitude / mean_amplitude).clamp_min(self.eps)).clamp(-4.0, 4.0) / 4.0,
            (amplitude - previous.abs()) / mean_amplitude,
            phase_unit.real,
            phase_unit.imag,
            torch.angle(phase_unit) / math.pi,
            signal_unit.real,
            signal_unit.imag,
            amplitude / mean_amplitude,
        ], dim=1)
        return F.adaptive_avg_pool1d(features, self.temporal_tokens).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        z = _as_complex_iq(x)
        tokens = torch.cat([
            self.time_tokens(x).float(),
            self._stft_tokens(z),
            self._temporal_tokens(z),
        ], dim=1)
        if tokens.size(1) != self.token_count:
            raise RuntimeError(f"RF token count changed: {tokens.size(1)} != {self.token_count}")
        return tokens.to(dtype=original_dtype)


class CompactPhysicalCrossAttention(nn.Module):
    """Standard QKV cross-attention with a small typed physical-token bank."""

    def __init__(
        self,
        query_channels: int,
        token_dim: int,
        token_count: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float | None = 0.01,
    ) -> None:
        super().__init__()
        query_channels = int(query_channels)
        if query_channels % int(num_heads) != 0:
            raise ValueError("query_channels must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = query_channels // self.num_heads
        self.token_count = int(token_count)
        self.dropout = float(dropout)
        self.query_norm = nn.LayerNorm(query_channels)
        self.token_norm = nn.LayerNorm(int(token_dim))
        self.q_proj = nn.Linear(query_channels, query_channels, bias=False)
        self.k_proj = nn.Linear(int(token_dim), query_channels, bias=False)
        self.v_proj = nn.Linear(int(token_dim), query_channels, bias=False)
        self.out_proj = nn.Linear(query_channels, query_channels, bias=False)
        self.output_norm = nn.LayerNorm(query_channels)
        self.token_bias = nn.Parameter(torch.zeros(self.num_heads, self.token_count))
        if residual_scale_init is None:
            self.register_parameter("residual_scale", None)
        else:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        return x.view(batch, tokens, self.num_heads, channels // self.num_heads).transpose(1, 2)

    def compute_delta(
        self,
        query_feature: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.size(1) != self.token_count:
            raise ValueError(f"Expected {self.token_count} physical tokens, got {tokens.size(1)}")
        query = self.query_norm(query_feature.transpose(1, 2))
        tokens = self.token_norm(tokens.to(dtype=query.dtype))
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(tokens))
        v = self._heads(self.v_proj(tokens))
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + self.token_bias.to(dtype=scores.dtype).view(
            1, self.num_heads, 1, self.token_count
        )
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        delta = torch.matmul(weights, v).transpose(1, 2).reshape(
            query.size(0), query.size(1), query.size(2)
        )
        return self.output_norm(self.out_proj(delta)).transpose(1, 2)

    def forward(self, query_feature: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        delta = self.compute_delta(query_feature, tokens)
        scale = (
            delta.new_tensor(1.0)
            if self.residual_scale is None
            else self.residual_scale.to(device=delta.device, dtype=delta.dtype)
        )
        return query_feature + scale * delta


class BottleneckSelfAttention1D(nn.Module):
    """MambaVision-style pre-norm QKV residual block for one 1-D feature map."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels % int(num_heads) != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.residual_scale = nn.Parameter(
            torch.full((channels,), float(residual_scale_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        normalized = self.norm(tokens)
        delta, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.residual_scale.to(dtype=delta.dtype) * delta
        return tokens.transpose(1, 2)


class HymbaParallelSequenceLayer1D(nn.Module):
    """Run an existing Stage-12 BiMamba layer and QKV attention in parallel."""

    def __init__(
        self,
        mamba_branch: nn.Module,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        mamba_scale_init: float = 1.0,
        attention_scale_init: float = 0.01,
        attention_scale_max: float = 1.0,
    ) -> None:
        super().__init__()
        channels = int(channels)
        if channels % int(num_heads) != 0:
            raise ValueError("channels must be divisible by num_heads")
        if not 0.0 < float(attention_scale_init) < float(attention_scale_max):
            raise ValueError("attention_scale_init must be in (0, attention_scale_max)")
        self.mamba_branch = mamba_branch
        self.attention_scale_max = float(attention_scale_max)
        self.attention_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.attention_output_norm = nn.LayerNorm(channels)
        self.mamba_scale = nn.Parameter(torch.full((channels,), float(mamba_scale_init)))
        ratio = float(attention_scale_init) / float(attention_scale_max)
        initial_raw = math.atanh(ratio)
        self.attention_scale_raw = nn.Parameter(torch.full((channels,), initial_raw))

    def attention_scale_values(self) -> torch.Tensor:
        return self.attention_scale_max * torch.tanh(self.attention_scale_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mamba_output = self.mamba_branch(x)
        tokens = x.transpose(1, 2)
        normalized = self.attention_norm(tokens)
        attention_delta, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        attention_delta = self.attention_output_norm(attention_delta).transpose(1, 2)
        mamba_delta = mamba_output - x
        return (
            x
            + self.mamba_scale.to(dtype=x.dtype).view(1, -1, 1) * mamba_delta
            + self.attention_scale_values().to(dtype=x.dtype).view(1, -1, 1) * attention_delta
        )


class IQUBiMamba1D_PhaseEquivariantFusion(IQUBiMamba1D):
    """Stage 12 in a canonical phase frame plus Hermitian complex fusion."""

    def __init__(
        self,
        *args,
        phase_fusion_kv_tokens: int = 32,
        phase_fusion_num_heads: int = 4,
        phase_fusion_head_dim: int = 8,
        phase_fusion_dropout: float = 0.0,
        phase_fusion_scale_init: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.phase_canonicalizer = GlobalPhaseCanonicalizer1D()
        self.phase_equivariant_fusion = ComplexHermitianInputFusion1D(
            kv_tokens=phase_fusion_kv_tokens,
            num_heads=phase_fusion_num_heads,
            head_dim=phase_fusion_head_dim,
            dropout=phase_fusion_dropout,
            residual_scale_init=phase_fusion_scale_init,
        )

    def forward(self, x: torch.Tensor):
        canonical, unit_phase = self.phase_canonicalizer.canonicalize(x)
        output = super().forward(self.phase_equivariant_fusion(canonical))
        if isinstance(output, tuple):
            return tuple(self.phase_canonicalizer.restore(item, unit_phase) for item in output)
        if isinstance(output, list):
            return [self.phase_canonicalizer.restore(item, unit_phase) for item in output]
        return self.phase_canonicalizer.restore(output, unit_phase)


class IQUBiMamba1D_PhysicalTokenCrossAttention(IQUBiMamba1D):
    """Stage 12 with time-domain RF physical tokens as explicit K/V."""

    def __init__(
        self,
        *args,
        physical_query_stage: int = 2,
        physical_num_heads: int = 4,
        physical_dropout: float = 0.0,
        physical_residual_scale_init: float = 0.01,
        physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.physical_query_stage = int(physical_query_stage)
        if not 0 <= self.physical_query_stage < len(self.encoder.output_channels):
            raise ValueError(f"Invalid physical_query_stage {self.physical_query_stage}")
        self.physical_token_extractor = TimeDomainPhysicalTokenExtractor(
            cyclic_lags=physical_cyclic_lags,
            polyphase_branches=physical_polyphase_branches,
            symbol_orders=physical_symbol_orders,
            min_cyclic_freq=physical_min_cyclic_freq,
            max_cyclic_freq=physical_max_cyclic_freq,
            cyclic_temperature=physical_cyclic_temperature,
        )
        self.physical_cross_attention = CompactPhysicalCrossAttention(
            query_channels=int(self.encoder.output_channels[self.physical_query_stage]),
            token_dim=self.physical_token_extractor.token_dim,
            token_count=self.physical_token_extractor.token_count,
            num_heads=physical_num_heads,
            dropout=physical_dropout,
            residual_scale_init=physical_residual_scale_init,
        )

    def forward(self, x: torch.Tensor):
        tokens = self.physical_token_extractor(x)
        skips = self.encoder(x)
        skips = list(skips)
        stage = self.physical_query_stage
        skips[stage] = self.physical_cross_attention(skips[stage], tokens)
        return self.decoder(skips)


class IQUBiMamba1D_BottleneckSelfAttention(IQUBiMamba1D):
    """Stage 12 plus self-attention after the deepest BiMamba feature."""

    def __init__(
        self,
        *args,
        bottleneck_attention_stage: int = 3,
        bottleneck_attention_num_heads: int = 4,
        bottleneck_attention_dropout: float = 0.0,
        bottleneck_attention_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.bottleneck_attention_stage = int(bottleneck_attention_stage)
        if not 0 <= self.bottleneck_attention_stage < len(self.encoder.output_channels):
            raise ValueError(f"Invalid bottleneck_attention_stage {self.bottleneck_attention_stage}")
        self.bottleneck_attention = BottleneckSelfAttention1D(
            channels=int(self.encoder.output_channels[self.bottleneck_attention_stage]),
            num_heads=bottleneck_attention_num_heads,
            dropout=bottleneck_attention_dropout,
            residual_scale_init=bottleneck_attention_scale_init,
        )

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        stage = self.bottleneck_attention_stage
        skips[stage] = self.bottleneck_attention(skips[stage])
        return self.decoder(skips)


class IQUBiMamba1D_HymbaParallel(IQUBiMamba1D):
    """Stage 12 with same-resolution parallel BiMamba and QKV attention."""

    def __init__(
        self,
        *args,
        hymba_stage: int = 3,
        hymba_num_heads: int = 4,
        hymba_dropout: float = 0.0,
        hymba_mamba_scale_init: float = 1.0,
        hymba_attention_scale_init: float = 0.01,
        hymba_attention_scale_max: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hymba_stage = int(hymba_stage)
        if not 0 <= self.hymba_stage < len(self.encoder.mamba_layers):
            raise ValueError(f"Invalid hymba_stage {self.hymba_stage}")
        original = self.encoder.mamba_layers[self.hymba_stage]
        if isinstance(original, nn.Identity):
            raise ValueError("hymba_stage must select a Stage-12 BiMamba layer, not an identity stage")
        self.encoder.mamba_layers[self.hymba_stage] = HymbaParallelSequenceLayer1D(
            mamba_branch=original,
            channels=int(self.encoder.output_channels[self.hymba_stage]),
            num_heads=hymba_num_heads,
            dropout=hymba_dropout,
            mamba_scale_init=hymba_mamba_scale_init,
            attention_scale_init=hymba_attention_scale_init,
            attention_scale_max=hymba_attention_scale_max,
        )

    @property
    def hymba_block(self) -> HymbaParallelSequenceLayer1D:
        return self.encoder.mamba_layers[self.hymba_stage]


class IQUBiMamba1D_RFPhysicalKVCrossAttention(IQUBiMamba1D):
    """Stage 12 with compact STFT and time-domain RF tokens as K/V."""

    def __init__(
        self,
        *args,
        rf_physical_query_stage: int = 2,
        rf_physical_num_heads: int = 4,
        rf_physical_dropout: float = 0.0,
        rf_physical_residual_scale_init: float = 0.01,
        rf_physical_stft_n_fft: int = 256,
        rf_physical_stft_hop_length: int = 64,
        rf_physical_stft_win_length: int = 256,
        rf_physical_num_subbands: int = 8,
        rf_physical_temporal_tokens: int = 4,
        rf_physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        rf_physical_polyphase_branches: int = 8,
        rf_physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        rf_physical_min_cyclic_freq: float = 1.0 / 64.0,
        rf_physical_max_cyclic_freq: float = 1.0 / 8.0,
        rf_physical_cyclic_temperature: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rf_physical_query_stage = int(rf_physical_query_stage)
        if not 0 <= self.rf_physical_query_stage < len(self.encoder.output_channels):
            raise ValueError(f"Invalid rf_physical_query_stage {self.rf_physical_query_stage}")
        self.rf_physical_token_extractor = RFMultiDomainTokenExtractor(
            n_fft=rf_physical_stft_n_fft,
            hop_length=rf_physical_stft_hop_length,
            win_length=rf_physical_stft_win_length,
            num_subbands=rf_physical_num_subbands,
            temporal_tokens=rf_physical_temporal_tokens,
            cyclic_lags=rf_physical_cyclic_lags,
            polyphase_branches=rf_physical_polyphase_branches,
            symbol_orders=rf_physical_symbol_orders,
            min_cyclic_freq=rf_physical_min_cyclic_freq,
            max_cyclic_freq=rf_physical_max_cyclic_freq,
            cyclic_temperature=rf_physical_cyclic_temperature,
        )
        self.rf_physical_cross_attention = CompactPhysicalCrossAttention(
            query_channels=int(self.encoder.output_channels[self.rf_physical_query_stage]),
            token_dim=self.rf_physical_token_extractor.token_dim,
            token_count=self.rf_physical_token_extractor.token_count,
            num_heads=rf_physical_num_heads,
            dropout=rf_physical_dropout,
            residual_scale_init=rf_physical_residual_scale_init,
        )

    def forward(self, x: torch.Tensor):
        tokens = self.rf_physical_token_extractor(x)
        skips = list(self.encoder(x))
        stage = self.rf_physical_query_stage
        skips[stage] = self.rf_physical_cross_attention(skips[stage], tokens)
        return self.decoder(skips)
