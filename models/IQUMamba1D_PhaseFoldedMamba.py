"""Blind multi-hypothesis phase-folded Mamba residual for Stage 334.

The branch estimates candidate cyclic frequencies from the received mixture,
warps each candidate to an integer canonical period, and runs one shared
complex-state SSM along each fixed-phase trajectory. A compact low-rank router
communicates across phases before the trajectories are unfolded and warped
back to the sample clock. A null route and zero-initialized output projection
make the branch conservative at initialization.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D_ComplexStateMamba import ComplexStateSelectiveSSM


def _sample_time_axis(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Linearly sample ``x`` at per-example fractional sample positions."""

    if x.dim() != 3 or positions.dim() != 2 or x.size(0) != positions.size(0):
        raise ValueError("Expected x=(B,C,L) and positions=(B,T)")
    length = int(x.size(-1))
    if length == 1:
        return x.expand(-1, -1, positions.size(1))
    normalized = 2.0 * positions / float(length - 1) - 1.0
    grid = torch.stack([normalized, torch.zeros_like(normalized)], dim=-1)
    sampled = F.grid_sample(
        x.unsqueeze(2),
        grid.unsqueeze(1),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(2)


class CrossPhaseRouter(nn.Module):
    """Low-rank phase-to-router-to-phase communication at every cycle."""

    def __init__(self, dim: int, num_routers: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_routers = max(1, int(num_routers))
        self.norm = nn.LayerNorm(self.dim)
        self.router_tokens = nn.Parameter(
            torch.randn(self.num_routers, self.dim) * (self.dim**-0.5)
        )
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process phase trajectories shaped ``(B, P, cycles, D)``."""

        batch, phases, cycles, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected feature dim {self.dim}, got {dim}")
        tokens = self.norm(x).permute(0, 2, 1, 3).reshape(
            batch * cycles, phases, dim
        )
        routers = self.router_tokens.unsqueeze(0).expand(tokens.size(0), -1, -1)
        scale = dim**-0.5
        gather = torch.softmax(
            torch.matmul(routers, tokens.transpose(1, 2)) * scale, dim=-1
        )
        routed = torch.matmul(gather, tokens)
        distribute = torch.softmax(
            torch.matmul(tokens, routed.transpose(1, 2)) * scale, dim=-1
        )
        update = self.out_proj(torch.matmul(distribute, routed))
        update = update.reshape(batch, cycles, phases, dim).permute(0, 2, 1, 3)
        return x + update


class PhaseFoldedMambaResidual(nn.Module):
    """Mixture-only phase-folded residual with soft period routing."""

    def __init__(
        self,
        input_channels: int = 2,
        output_channels: int = 4,
        hidden_channels: int = 16,
        candidate_periods: Sequence[int] = (8, 12, 16, 24, 32),
        local_frequency_radius: float = 0.20,
        evidence_temperature: float = 0.10,
        frequency_temperature: float = 0.25,
        null_logit_init: float = 1.0,
        num_routers: int = 2,
        d_state: int = 4,
        d_conv: int = 3,
        expand: int = 1,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        candidate_top_k: int | None = 3,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("Stage 334 expects one complex I/Q mixture")
        periods = tuple(int(period) for period in candidate_periods)
        if not periods or any(period < 2 for period in periods):
            raise ValueError("candidate_periods must contain integers >= 2")
        if len(set(periods)) != len(periods):
            raise ValueError("candidate_periods must be unique")

        self.hidden_channels = max(2, int(hidden_channels))
        self.candidate_periods = periods
        self.local_frequency_radius = float(local_frequency_radius)
        self.evidence_temperature = max(float(evidence_temperature), 1e-3)
        self.frequency_temperature = max(float(frequency_temperature), 1e-3)
        self.eps = float(eps)
        self.candidate_top_k = (
            len(periods)
            if candidate_top_k is None
            else max(1, min(int(candidate_top_k), len(periods)))
        )
        self.register_buffer(
            "nominal_frequencies",
            torch.tensor([1.0 / period for period in periods], dtype=torch.float32),
        )
        self.null_logit = nn.Parameter(torch.tensor(float(null_logit_init)))
        self.candidate_bias = nn.Parameter(torch.zeros(len(periods)))

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, self.hidden_channels, kernel_size=5, padding=2),
            nn.SiLU(),
        )
        self.phase_ssm = ComplexStateSelectiveSSM(
            self.hidden_channels,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            scan_checkpoint=bool(scan_checkpoint),
            scan_backend=str(scan_backend),
        )
        self.phase_router = CrossPhaseRouter(
            self.hidden_channels, num_routers=int(num_routers)
        )
        self.output_proj = nn.Conv1d(
            self.hidden_channels, int(output_channels), kernel_size=1
        )
        if zero_init:
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

        self.last_candidate_frequencies: torch.Tensor | None = None
        self.last_route_weights: torch.Tensor | None = None
        self.last_selected_candidates: tuple[int, ...] = ()

    def _period_evidence(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-example continuous frequencies and null/candidate routes."""

        work = x.float()
        envelope = work[:, 0].square() + work[:, 1].square()
        envelope = envelope - envelope.mean(dim=-1, keepdim=True)
        power = torch.fft.rfft(envelope, dim=-1).abs().square()
        frequencies = torch.fft.rfftfreq(
            x.size(-1), d=1.0, device=x.device, dtype=power.dtype
        )
        total_power = power[:, 1:].sum(dim=-1).clamp_min(self.eps)

        estimates = []
        concentrations = []
        for nominal in self.nominal_frequencies.to(device=x.device):
            radius = nominal * self.local_frequency_radius
            mask = (frequencies >= nominal - radius) & (
                frequencies <= nominal + radius
            )
            if not bool(mask.any()):
                estimates.append(nominal.expand(x.size(0)))
                concentrations.append(power.new_zeros(x.size(0)))
                continue
            local_power = power[:, mask]
            local_freqs = frequencies[mask]
            logits = torch.log(local_power + self.eps) / self.frequency_temperature
            weights = torch.softmax(logits, dim=-1)
            estimates.append((weights * local_freqs).sum(dim=-1))
            concentrations.append(local_power.sum(dim=-1) / total_power)

        estimated = torch.stack(estimates, dim=-1)
        concentration = torch.stack(concentrations, dim=-1).clamp(0.0, 1.0)
        candidate_logits = (
            concentration / self.evidence_temperature
            + self.candidate_bias.unsqueeze(0)
        )
        null_logits = self.null_logit.expand(x.size(0), 1)
        routes = torch.softmax(torch.cat([null_logits, candidate_logits], dim=-1), dim=-1)
        return estimated.to(dtype=x.dtype), routes.to(dtype=x.dtype)

    @staticmethod
    def fold(x: torch.Tensor, period: int) -> tuple[torch.Tensor, int]:
        """Circular-pad and fold ``(B,D,L)`` into ``(B,P,cycles,D)``."""

        period = int(period)
        length = int(x.size(-1))
        padded_length = int(math.ceil(length / period) * period)
        pad = padded_length - length
        if pad:
            repeats = int(math.ceil(pad / length))
            x = torch.cat([x, x.repeat(1, 1, repeats)[..., :pad]], dim=-1)
        cycles = padded_length // period
        folded = x.reshape(x.size(0), x.size(1), cycles, period)
        return folded.permute(0, 3, 2, 1).contiguous(), length

    @staticmethod
    def unfold(x: torch.Tensor, length: int) -> torch.Tensor:
        """Invert :meth:`fold` and crop circular padding."""

        unfolded = x.permute(0, 3, 2, 1).contiguous().flatten(2)
        return unfolded[..., : int(length)]

    def _canonical_warp(
        self,
        x: torch.Tensor,
        estimated_frequency: torch.Tensor,
        nominal_period: int,
    ) -> torch.Tensor:
        length = int(x.size(-1))
        positions = torch.arange(length, device=x.device, dtype=torch.float32)
        estimated_period = estimated_frequency.float().clamp_min(self.eps).reciprocal()
        source_positions = positions.unsqueeze(0) * (
            estimated_period / float(nominal_period)
        ).unsqueeze(1)
        return _sample_time_axis(x, source_positions)

    def _inverse_warp(
        self,
        x: torch.Tensor,
        estimated_frequency: torch.Tensor,
        nominal_period: int,
    ) -> torch.Tensor:
        length = int(x.size(-1))
        positions = torch.arange(length, device=x.device, dtype=torch.float32)
        estimated_period = estimated_frequency.float().clamp_min(self.eps).reciprocal()
        canonical_positions = positions.unsqueeze(0) * (
            float(nominal_period) / estimated_period
        ).unsqueeze(1)
        return _sample_time_axis(x, canonical_positions)

    def _process_candidate(
        self, x: torch.Tensor, frequency: torch.Tensor, period: int
    ) -> torch.Tensor:
        canonical = self._canonical_warp(x, frequency, period)
        features = self.input_proj(canonical)
        folded, length = self.fold(features, period)
        batch, phases, cycles, dim = folded.shape
        trajectories = folded.reshape(batch * phases, cycles, dim)
        trajectories = self.phase_ssm(trajectories)
        folded = trajectories.reshape(batch, phases, cycles, dim)
        folded = self.phase_router(folded)
        canonical_features = self.unfold(folded, length)
        return self._inverse_warp(canonical_features, frequency, period)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected mixture shaped (B,2,L), got {tuple(x.shape)}")
        frequencies, routes = self._period_evidence(x)
        aggregate = x.new_zeros(x.size(0), self.hidden_channels, x.size(-1))
        candidate_mass = routes[:, 1:].detach().mean(dim=0)
        selected = torch.topk(
            candidate_mass, k=self.candidate_top_k, sorted=False
        ).indices.tolist()
        for index in selected:
            period = self.candidate_periods[index]
            candidate = self._process_candidate(x, frequencies[:, index], period)
            aggregate = aggregate + routes[:, index + 1, None, None] * candidate
        self.last_candidate_frequencies = frequencies.detach()
        self.last_route_weights = routes.detach()
        self.last_selected_candidates = tuple(sorted(int(index) for index in selected))
        return self.output_proj(aggregate)


class IQUMamba1DPhaseFoldedMamba(nn.Module):
    """Stage-4 backbone plus the conservative Stage-334 phase branch."""

    def __init__(
        self,
        input_size,
        input_channels,
        n_stages,
        features_per_stage,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes,
        n_conv_per_stage_decoder,
        deep_supervision=False,
        **phase_kwargs,
    ) -> None:
        super().__init__()
        from models.IQUMamba1D import IQUMamba1D

        self.backbone = IQUMamba1D(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv1d,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm1d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=bool(deep_supervision),
        )
        self.phase_branch = PhaseFoldedMambaResidual(
            input_channels=input_channels,
            output_channels=num_classes,
            **phase_kwargs,
        )

    def forward(self, x: torch.Tensor):
        baseline = self.backbone(x)
        correction = self.phase_branch(x)
        if isinstance(baseline, (list, tuple)):
            corrected = []
            for output in baseline:
                residual = correction
                if output.size(-1) != correction.size(-1):
                    residual = F.interpolate(
                        correction, size=output.size(-1), mode="linear", align_corners=False
                    )
                corrected.append(output + residual)
            return type(baseline)(corrected)
        return baseline + correction

    def scan_backend_status(self) -> dict[str, str]:
        return {"phase_branch.phase_ssm": self.phase_branch.phase_ssm.last_scan_backend}

    def diagnostics(self) -> dict[str, object]:
        values: dict[str, object] = {
            "scan_backend_phase": self.phase_branch.phase_ssm.last_scan_backend,
            "phase_selected_candidates": ",".join(
                str(index) for index in self.phase_branch.last_selected_candidates
            ),
        }
        if self.phase_branch.last_route_weights is not None:
            mean_routes = self.phase_branch.last_route_weights.float().mean(dim=0)
            values["phase_null_route"] = mean_routes[0]
            values["phase_max_candidate_route"] = mean_routes[1:].max()
        return values
