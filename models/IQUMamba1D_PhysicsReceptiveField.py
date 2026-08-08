"""Stage-4 communication-specific identity and receptive-field variants.

Stage 306 gives two-source outputs a deterministic physical identity by sorting
them with an observable waveform signature. Stage 307 expands the input
receptive field in symbol-normalized coordinates and phase-transports remote IQ
samples before aggregation. Neither variant replaces the Stage-4 backbone.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D


def _split_iq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 3 or x.size(1) != 2:
        raise ValueError(f"Expected complex IQ tensor shaped (B, 2, L), got {tuple(x.shape)}")
    return x[:, 0], x[:, 1]


def phase_increment(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return the mean one-sample complex phase increment in radians."""
    real, imag = _split_iq(x.float())
    cross_real = (real[:, 1:] * real[:, :-1] + imag[:, 1:] * imag[:, :-1]).mean(-1)
    cross_imag = (imag[:, 1:] * real[:, :-1] - real[:, 1:] * imag[:, :-1]).mean(-1)
    magnitude = torch.sqrt(cross_real.square() + cross_imag.square()).clamp_min(eps)
    return torch.atan2(cross_imag / magnitude, cross_real / magnitude)


def normalized_bandwidth_proxy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant temporal bandwidth proxy used only as an optional tie-breaker."""
    real, imag = _split_iq(x.float())
    power = (real.square() + imag.square()).mean(-1).clamp_min(eps)
    derivative = (
        (real[:, 1:] - real[:, :-1]).square()
        + (imag[:, 1:] - imag[:, :-1]).square()
    ).mean(-1)
    return torch.sqrt(derivative / power)


def carrier_phase_increment(
    x: torch.Tensor,
    symbol_orders: Sequence[int] = (2, 4, 8),
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate carrier phase step with modulation-removing power transforms."""
    if float(temperature) <= 0:
        raise ValueError("carrier score temperature must be positive")
    real, imag = _split_iq(x.float())
    unit = torch.complex(real, imag)
    unit = unit / unit.abs().clamp_min(eps)
    estimates = []
    reliabilities = []
    for order_value in symbol_orders:
        order = int(order_value)
        if order < 1:
            raise ValueError("canonical symbol orders must be positive")
        powered = unit.pow(order)
        correlation = (powered[:, 1:] * powered[:, :-1].conj()).mean(dim=-1)
        estimates.append(torch.angle(correlation) / float(order))
        reliabilities.append(correlation.abs())
    reliability = torch.stack(reliabilities, dim=1)
    weights = torch.softmax(reliability / float(temperature), dim=1)
    return (weights * torch.stack(estimates, dim=1)).sum(dim=1)


class PhysicalSourceCanonicalizer(nn.Module):
    """Sort two separated sources by CFO proxy, then optional bandwidth proxy.

    This operation defines a reproducible physical output identity. It cannot
    recover arbitrary dataset indices when two sources are physically
    exchangeable; such datasets must use PIT metrics or define a canonical rule.
    """

    def __init__(
        self,
        cfo_weight: float = 1.0,
        bandwidth_weight: float = 0.0,
        ascending: bool = True,
        symbol_orders: Sequence[int] = (2, 4, 8),
        order_temperature: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if float(cfo_weight) == 0.0 and float(bandwidth_weight) == 0.0:
            raise ValueError("At least one physical canonicalization weight must be non-zero")
        self.cfo_weight = float(cfo_weight)
        self.bandwidth_weight = float(bandwidth_weight)
        self.ascending = bool(ascending)
        self.symbol_orders = tuple(int(order) for order in symbol_orders)
        self.order_temperature = float(order_temperature)
        self.eps = float(eps)
        self.last_scores = None

    def source_scores(self, sources: torch.Tensor) -> torch.Tensor:
        if sources.dim() != 4 or sources.size(1) != 2 or sources.size(2) != 2:
            raise ValueError(
                "PhysicalSourceCanonicalizer requires sources shaped (B, 2, 2, L)"
            )
        scores = []
        for source_index in range(2):
            source = sources[:, source_index]
            score = self.cfo_weight * carrier_phase_increment(
                source,
                symbol_orders=self.symbol_orders,
                temperature=self.order_temperature,
                eps=self.eps,
            )
            if self.bandwidth_weight != 0.0:
                score = score + self.bandwidth_weight * normalized_bandwidth_proxy(
                    source, self.eps
                )
            scores.append(score)
        return torch.stack(scores, dim=1)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        if output.dim() != 3 or output.size(1) != 4:
            raise ValueError(
                f"Stage 306 supports exactly two complex sources (4 channels), got {tuple(output.shape)}"
            )
        sources = output.reshape(output.size(0), 2, 2, output.size(-1))
        scores = self.source_scores(sources)
        self.last_scores = scores.detach()
        swap = scores[:, 0] > scores[:, 1]
        if not self.ascending:
            swap = ~swap
        swap = swap[:, None, None]
        first = torch.where(swap, sources[:, 1], sources[:, 0])
        second = torch.where(swap, sources[:, 0], sources[:, 1])
        return torch.stack((first, second), dim=1).reshape_as(output)


class IQUMamba1DPhysicalCanonical(IQUMamba1D):
    """Stage 306: unchanged Stage-4 separator plus physical output identity."""

    def __init__(self, *args, canonical_cfo_weight: float = 1.0,
                 canonical_bandwidth_weight: float = 0.0,
                 canonical_ascending: bool = True,
                 canonical_symbol_orders=(2, 4, 8),
                 canonical_order_temperature: float = 0.1,
                 canonical_eps: float = 1e-8, **kwargs) -> None:
        if int(kwargs.get("num_classes", 4)) != 4:
            raise ValueError("Stage 306 currently supports exactly two complex sources")
        super().__init__(*args, **kwargs)
        self.canonicalizer = PhysicalSourceCanonicalizer(
            cfo_weight=canonical_cfo_weight,
            bandwidth_weight=canonical_bandwidth_weight,
            ascending=canonical_ascending,
            symbol_orders=canonical_symbol_orders,
            order_temperature=canonical_order_temperature,
            eps=canonical_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.canonicalizer(super().forward(x))


class SymbolNormalizedDelayDopplerRF(nn.Module):
    """Sparse phase-coherent IQ context in estimated symbol coordinates."""

    def __init__(
        self,
        sps_candidates: Sequence[float] = (8, 10, 16, 20, 32, 40),
        symbol_spans: Sequence[float] = (1, 2, 4, 8, 16, 32, 64),
        default_sps: float = 20.0,
        sps_temperature: float = 0.25,
        max_phase_step: float = 0.25,
        max_doppler_offset: float = 0.05,
        gate_hidden: int = 24,
        residual_scale_init: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        candidates = torch.as_tensor(tuple(float(v) for v in sps_candidates))
        spans = torch.as_tensor(tuple(float(v) for v in symbol_spans))
        if candidates.numel() < 2 or torch.any(candidates <= 1):
            raise ValueError("sps_candidates must contain at least two values greater than one")
        if spans.numel() < 1 or torch.any(spans <= 0):
            raise ValueError("symbol_spans must be positive")
        if float(sps_temperature) <= 0:
            raise ValueError("sps_temperature must be positive")
        self.register_buffer("sps_candidates", candidates, persistent=True)
        self.register_buffer("symbol_spans", spans, persistent=True)
        self.sps_temperature = float(sps_temperature)
        self.max_phase_step = float(max_phase_step)
        self.max_doppler_offset = float(max_doppler_offset)
        self.eps = float(eps)

        nearest_default = int(torch.argmin(torch.abs(candidates - float(default_sps))))
        bias = torch.full((candidates.numel(),), -2.0)
        bias[nearest_default] = 2.0
        self.sps_logits = nn.Parameter(bias)
        self.evidence_scale = nn.Parameter(torch.tensor(1.0))
        evidence_dim = int(candidates.numel()) + 3
        self.branch_gate = nn.Sequential(
            nn.Linear(evidence_dim, int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), int(spans.numel())),
        )
        nn.init.zeros_(self.branch_gate[-1].weight)
        nn.init.zeros_(self.branch_gate[-1].bias)

        branch_count = int(spans.numel())
        self.weight_real = nn.Parameter(torch.full((branch_count,), 1.0 / branch_count))
        self.weight_imag = nn.Parameter(torch.zeros(branch_count))
        self.raw_doppler_offsets = nn.Parameter(torch.zeros(branch_count))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_sps = None
        self.last_phase_step = None

    def _lag_evidence(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_iq(x.float())
        power = (real.square() + imag.square()).mean(-1).clamp_min(self.eps)
        evidence = []
        length = int(x.size(-1))
        for candidate in self.sps_candidates:
            lag = min(max(1, int(round(float(candidate.item())))), length - 1)
            corr_real = (
                real[:, lag:] * real[:, :-lag] + imag[:, lag:] * imag[:, :-lag]
            ).mean(-1)
            corr_imag = (
                imag[:, lag:] * real[:, :-lag] - real[:, lag:] * imag[:, :-lag]
            ).mean(-1)
            evidence.append(torch.sqrt(corr_real.square() + corr_imag.square()) / power)
        return torch.stack(evidence, dim=1)

    def estimate_sps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        evidence = self._lag_evidence(x)
        logits = self.sps_logits[None, :] + self.evidence_scale * evidence
        weights = torch.softmax(logits / self.sps_temperature, dim=1)
        sps = (weights * self.sps_candidates[None, :]).sum(dim=1)
        return sps, evidence

    @staticmethod
    def _fractional_shift(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Sample x[t-offset] with linear interpolation and zero boundaries."""
        batch, channels, length = x.shape
        branch_count = offsets.size(1)
        position = (
            torch.arange(length, device=x.device, dtype=offsets.dtype)[None, None, :]
            - offsets[:, :, None]
        )
        lower = torch.floor(position)
        upper = lower + 1.0
        upper_weight = position - lower
        lower_weight = 1.0 - upper_weight
        lower_index = lower.long().clamp(0, length - 1)
        upper_index = upper.long().clamp(0, length - 1)
        expanded = x[:, :, None, :].expand(batch, channels, branch_count, length)
        lower_value = torch.gather(
            expanded, 3, lower_index[:, None, :, :].expand_as(expanded)
        )
        upper_value = torch.gather(
            expanded, 3, upper_index[:, None, :, :].expand_as(expanded)
        )
        valid_lower = ((lower >= 0) & (lower < length)).to(x.dtype)
        valid_upper = ((upper >= 0) & (upper < length)).to(x.dtype)
        return (
            lower_value * lower_weight[:, None] * valid_lower[:, None]
            + upper_value * upper_weight[:, None] * valid_upper[:, None]
        )

    @staticmethod
    def _rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        cosine = torch.cos(angle)[:, None, :, None]
        sine = torch.sin(angle)[:, None, :, None]
        real, imag = x[:, 0:1], x[:, 1:2]
        return torch.cat(
            (cosine * real - sine * imag, sine * real + cosine * imag), dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < 3:
            return x
        original_dtype = x.dtype
        work = x.float()
        sps, lag_evidence = self.estimate_sps(work)
        phase_step = phase_increment(work, self.eps).clamp(
            -self.max_phase_step, self.max_phase_step
        )
        delays = sps[:, None] * self.symbol_spans[None, :]
        delays = delays.clamp(max=max(1.0, float(x.size(-1) - 2)))
        doppler = phase_step[:, None] + self.max_doppler_offset * torch.tanh(
            self.raw_doppler_offsets
        )[None, :]
        angle = doppler * delays

        past = self._rotate(self._fractional_shift(work, delays), angle)
        future = self._rotate(self._fractional_shift(work, -delays), -angle)
        aligned = 0.5 * (past + future)
        local = work[:, :, None, :]
        context_delta = aligned - local

        real, imag = _split_iq(work)
        log_power = torch.log(
            (real.square() + imag.square()).mean(-1).clamp_min(self.eps)
        ).clamp(-12.0, 12.0) / 12.0
        stats = torch.cat(
            (
                lag_evidence,
                (sps / self.sps_candidates.max()).unsqueeze(1),
                (phase_step / max(self.max_phase_step, self.eps)).unsqueeze(1),
                log_power.unsqueeze(1),
            ),
            dim=1,
        )
        gates = 2.0 * torch.sigmoid(self.branch_gate(stats.float()))
        weighted_real = (
            self.weight_real[None, :, None] * context_delta[:, 0]
            - self.weight_imag[None, :, None] * context_delta[:, 1]
        )
        weighted_imag = (
            self.weight_real[None, :, None] * context_delta[:, 1]
            + self.weight_imag[None, :, None] * context_delta[:, 0]
        )
        delta = torch.stack(
            (
                (gates[:, :, None] * weighted_real).sum(dim=1),
                (gates[:, :, None] * weighted_imag).sum(dim=1),
            ),
            dim=1,
        )
        self.last_sps = sps.detach()
        self.last_phase_step = phase_step.detach()
        return (work + self.residual_scale * delta).to(original_dtype)

    def no_weight_decay(self) -> set[str]:
        return {"sps_logits", "residual_scale", "raw_doppler_offsets"}


class IQUMamba1DSymbolDelayDopplerRF(IQUMamba1D):
    """Stage 307: Stage 4 with phase-coherent symbol-normalized input memory."""

    def __init__(self, *args, rf_sps_candidates=(8, 10, 16, 20, 32, 40),
                 rf_symbol_spans=(1, 2, 4, 8, 16, 32, 64), rf_default_sps=20.0,
                 rf_sps_temperature=0.25, rf_max_phase_step=0.25,
                 rf_max_doppler_offset=0.05, rf_gate_hidden=24,
                 rf_residual_scale_init=0.01, rf_eps=1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.symbol_delay_doppler_rf = SymbolNormalizedDelayDopplerRF(
            sps_candidates=rf_sps_candidates,
            symbol_spans=rf_symbol_spans,
            default_sps=rf_default_sps,
            sps_temperature=rf_sps_temperature,
            max_phase_step=rf_max_phase_step,
            max_doppler_offset=rf_max_doppler_offset,
            gate_hidden=rf_gate_hidden,
            residual_scale_init=rf_residual_scale_init,
            eps=rf_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.symbol_delay_doppler_rf(x))

    def no_weight_decay(self) -> set[str]:
        return {
            f"symbol_delay_doppler_rf.{name}"
            for name in self.symbol_delay_doppler_rf.no_weight_decay()
        }
