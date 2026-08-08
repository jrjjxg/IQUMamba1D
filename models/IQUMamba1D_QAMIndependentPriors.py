"""Independent QAM prior variants built on the Stage-4 IQUMamba backbone.

This file intentionally contains three separate output-side priors so that
experiments can compare mechanisms instead of stacking them:

* ``IQUMamba1D_QAMMMAUnrolled``: source-wise multi-modulus equalization;
* ``IQUMamba1D_QAMDensityPrior``: source-wise constellation-density/topology
  conditioning;
* ``IQUMamba1D_QAMTimingPrior``: blind symbol-rate/offset hypothesis routing
  with an RRC matched-filter bank.

All three use the Stage-4 separator first and only then apply a bounded,
near-identity source refinement.  They use mixture/output tensors only and do
not require modulation labels or target metadata during inference.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D


def _odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _complex_rms(source: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(
        source.square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True) + eps
    )


def _rotate(source: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    theta = theta.to(device=source.device, dtype=source.dtype)
    c = torch.cos(theta)
    s = torch.sin(theta)
    real, imag = source[:, 0], source[:, 1]
    return torch.stack((real * c - imag * s, real * s + imag * c), dim=1)


def _square_points(axis_levels: int) -> torch.Tensor:
    levels = torch.arange(-(int(axis_levels) - 1), int(axis_levels), 2, dtype=torch.float32)
    ii, qq = torch.meshgrid(levels, levels, indexing="ij")
    points = torch.stack((ii.reshape(-1), qq.reshape(-1)), dim=-1)
    return points / points.square().sum(dim=-1).mean().sqrt().clamp_min(1e-8)


def _cross_128_points() -> torch.Tensor:
    points: list[tuple[float, float]] = []
    for i in range(-7, 8):
        for q in range(-7, 8):
            if abs(i) + abs(q) <= 11:
                points.append((float(i), float(q)))
            if len(points) >= 128:
                break
        if len(points) >= 128:
            break
    values = torch.tensor(points[:128], dtype=torch.float32)
    return values / values.square().sum(dim=-1).mean().sqrt().clamp_min(1e-8)


class _QAMPointBank(nn.Module):
    """Fixed QAM point clouds, phase hypotheses and branch metadata."""

    def __init__(
        self,
        axis_level_bank: Sequence[int] = (4, 8),
        include_cross_128: bool = True,
        phase_bank: Sequence[float] = (-math.pi / 8.0, 0.0, math.pi / 8.0, math.pi / 4.0),
    ) -> None:
        super().__init__()
        levels = tuple(sorted({max(2, int(value)) for value in axis_level_bank}))
        geometries = [_square_points(value) for value in levels]
        names = [f"{value * value}QAM" for value in levels]
        if include_cross_128:
            geometries.append(_cross_128_points())
            names.append("128QAM-cross")
        if not geometries:
            raise ValueError("QAM point bank must contain at least one geometry")
        phases = tuple(float(value) for value in phase_bank)
        if not phases:
            raise ValueError("QAM point bank must contain at least one phase")
        for index, points in enumerate(geometries):
            self.register_buffer(f"points_{index}", points, persistent=False)
            # MMA constants E[a^4]/E[a^2] for the two axes.
            moments = torch.stack(
                (
                    points[:, 0].square().square().mean() / points[:, 0].square().mean().clamp_min(1e-8),
                    points[:, 1].square().square().mean() / points[:, 1].square().mean().clamp_min(1e-8),
                )
            )
            self.register_buffer(f"moments_{index}", moments, persistent=False)
        self.geometry_names = tuple(names)
        self.num_geometries = len(geometries)
        self.phase_bank = phases
        self.register_buffer("phase_tensor", torch.tensor(phases), persistent=False)
        self.num_branches = self.num_geometries * len(phases)

    def points(self, index: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return getattr(self, f"points_{index}").to(device=device, dtype=dtype)

    def moments(self, index: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return getattr(self, f"moments_{index}").to(device=device, dtype=dtype)


def _build_stage4(
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


class _Stage4QAMWrapper(nn.Module):
    """Shared output handling; the three priors remain separate modules."""

    def _forward_backbone_and_prior(self, x: torch.Tensor):
        output = self.backbone(x)
        if isinstance(output, (list, tuple)):
            refined = [self.prior(item) for item in output]
            self.last_aux = getattr(self.prior, "last_aux", {})
            return type(output)(refined)
        refined = self.prior(output)
        self.last_aux = getattr(self.prior, "last_aux", {})
        return refined


class QAMMMAUnrolledPrior1D(nn.Module):
    """Source-wise QAM multi-modulus equalization unfolded into T updates."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 16,
        axis_level_bank: Sequence[int] = (4, 8),
        include_cross_128: bool = True,
        phase_bank: Sequence[float] = (-math.pi / 8.0, 0.0, math.pi / 8.0, math.pi / 4.0),
        num_unroll_steps: int = 3,
        step_init: float = 0.05,
        route_temperature: float = 8.0,
        null_bias: float = 0.0,
        reliability_floor: float = 0.05,
        max_scale: float = 0.12,
        scale_init: float = 0.005,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.num_sources = max(1, int(num_sources))
        self.bank = _QAMPointBank(axis_level_bank, include_cross_128, phase_bank)
        self.route_temperature = max(float(route_temperature), 1e-3)
        self.null_bias = float(null_bias)
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))
        self.max_scale = max(float(max_scale), 1e-6)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.step_size = nn.Parameter(
            torch.full((max(1, int(num_unroll_steps)),), float(step_init))
        )
        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        self.update_net = nn.Sequential(
            nn.Conv1d(4, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        nn.init.zeros_(self.update_net[-1].weight)
        nn.init.zeros_(self.update_net[-1].bias)
        self.last_aux: dict[str, torch.Tensor] = {}

    def _bounded_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.residual_scale / self.max_scale)

    def _branch_terms(self, state: torch.Tensor):
        costs: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        phase_tensor = self.bank.phase_tensor.to(device=state.device, dtype=state.dtype)
        for theta in phase_tensor:
            rotated = _rotate(state, theta)
            c, s = torch.cos(theta), torch.sin(theta)
            for geometry_index in range(self.bank.num_geometries):
                moments = self.bank.moments(geometry_index, state.device, state.dtype)
                real_error = rotated[:, 0].square() - moments[0]
                imag_error = rotated[:, 1].square() - moments[1]
                cost = real_error.square() + imag_error.square()
                gradient_rotated = torch.stack(
                    (rotated[:, 0] * real_error, rotated[:, 1] * imag_error), dim=1
                )
                gradient = torch.stack(
                    (
                        gradient_rotated[:, 0] * c + gradient_rotated[:, 1] * s,
                        -gradient_rotated[:, 0] * s + gradient_rotated[:, 1] * c,
                    ),
                    dim=1,
                )
                costs.append(cost)
                residuals.append(gradient)
        return torch.stack(costs, dim=1), torch.stack(residuals, dim=1)

    def _route(self, costs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = -self.route_temperature * costs.mean(dim=-1)
        null = costs.new_full((costs.size(0), 1), self.null_bias)
        weights = torch.softmax(torch.cat((null, logits), dim=1), dim=1)
        best = costs.min(dim=1).values
        confidence = torch.exp(-best).clamp(0.0, 1.0)
        return weights, confidence

    def _refine_one(self, source: torch.Tensor):
        scale = _complex_rms(source)
        state = source.float() / scale.float().clamp_min(1e-6)
        route_weights = None
        confidence = None
        for step_parameter in self.step_size:
            costs, gradients = self._branch_terms(state)
            route_weights, confidence = self._route(costs)
            weighted_gradient = (
                route_weights[:, 1:].view(source.size(0), -1, 1, 1) * gradients
            ).sum(dim=1)
            gate = self.reliability_floor + (1.0 - self.reliability_floor) * confidence.unsqueeze(1)
            learned = self.update_net(torch.cat((state, weighted_gradient), dim=1))
            step = 0.2 * torch.tanh(step_parameter)
            state = state - step * gate * torch.tanh(weighted_gradient)
            state = state + 0.25 * step * torch.tanh(learned)

        bounded_scale = self._bounded_scale().to(dtype=source.dtype)
        refined = source + bounded_scale * (state - source.float() / scale.float().clamp_min(1e-6)).to(source.dtype) * scale.to(source.dtype)
        return refined, {
            "route_weights": route_weights.detach(),
            "confidence": confidence.detach(),
            "residual_scale": bounded_scale.detach().reshape(1),
        }

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 3 or outputs.size(1) != 2 * self.num_sources:
            raise ValueError(f"QAMMMAUnrolledPrior1D expects [B, {2 * self.num_sources}, L], got {tuple(outputs.shape)}")
        batch, _, length = outputs.shape
        sources = outputs.reshape(batch, self.num_sources, 2, length)
        refined, routes, confidences = [], [], []
        residual_scale = None
        for source_index in range(self.num_sources):
            value, aux = self._refine_one(sources[:, source_index])
            refined.append(value)
            routes.append(aux["route_weights"])
            confidences.append(aux["confidence"])
            residual_scale = aux["residual_scale"]
        self.last_aux = {
            "source_route_weights": torch.stack(routes, dim=1),
            "source_confidence": torch.stack(confidences, dim=1),
            "residual_scale": residual_scale,
        }
        return torch.stack(refined, dim=1).reshape(batch, 2 * self.num_sources, length)


class QAMDensityPrior1D(nn.Module):
    """Source-wise density/topology encoder without hard constellation snapping."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 16,
        axis_level_bank: Sequence[int] = (4, 8),
        include_cross_128: bool = True,
        phase_bank: Sequence[float] = (-math.pi / 8.0, 0.0, math.pi / 8.0, math.pi / 4.0),
        density_temperature: float = 12.0,
        route_temperature: float = 4.0,
        null_bias: float = 0.0,
        max_scale: float = 0.12,
        scale_init: float = 0.02,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.num_sources = max(1, int(num_sources))
        self.bank = _QAMPointBank(axis_level_bank, include_cross_128, phase_bank)
        self.density_temperature = max(float(density_temperature), 1e-3)
        self.route_temperature = max(float(route_temperature), 1e-3)
        self.null_bias = float(null_bias)
        self.max_scale = max(float(max_scale), 1e-6)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))
        branch_features = 4 * self.bank.num_branches
        hidden_channels = max(1, int(hidden_channels))
        self.router = nn.Sequential(
            nn.Linear(branch_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, self.bank.num_branches + 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.context = nn.Linear(branch_features, hidden_channels)
        conv_kernel = _odd_kernel(kernel_size)
        self.temporal_in = nn.Conv1d(6, hidden_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.temporal_norm = nn.InstanceNorm1d(hidden_channels, affine=True)
        self.temporal_out = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        nn.init.zeros_(self.temporal_out.weight)
        nn.init.zeros_(self.temporal_out.bias)
        self.last_aux: dict[str, torch.Tensor] = {}

    def _density_terms(self, normalized: torch.Tensor):
        local_scores, summaries = [], []
        phase_tensor = self.bank.phase_tensor.to(device=normalized.device, dtype=normalized.dtype)
        for theta in phase_tensor:
            rotated = _rotate(normalized, theta)
            for geometry_index in range(self.bank.num_geometries):
                points = self.bank.points(geometry_index, normalized.device, normalized.dtype)
                distance = (
                    rotated[:, 0].unsqueeze(-1) - points[:, 0].view(1, 1, -1)
                ).square() + (
                    rotated[:, 1].unsqueeze(-1) - points[:, 1].view(1, 1, -1)
                ).square()
                rbf = torch.exp(-self.density_temperature * distance)
                local_scores.append(rbf.amax(dim=-1))
                occupancy = rbf.mean(dim=1)
                probability = occupancy / occupancy.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
                summaries.append(
                    torch.stack(
                        (
                            distance.amin(dim=-1).mean(dim=-1),
                            rbf.amax(dim=-1).mean(dim=-1),
                            entropy,
                            probability.amax(dim=-1),
                        ),
                        dim=-1,
                    )
                )
        return torch.stack(local_scores, dim=1), torch.stack(summaries, dim=1)

    def _refine_one(self, source: torch.Tensor):
        scale = _complex_rms(source)
        normalized = source.float() / scale.float().clamp_min(1e-6)
        local_scores, summaries = self._density_terms(normalized)
        router_input = summaries.reshape(source.size(0), -1)
        route_logits = self.router(router_input)
        route_logits = torch.cat(
            (route_logits[:, :1] + self.null_bias, route_logits[:, 1:]), dim=1
        )
        route_weights = torch.softmax(route_logits / self.route_temperature, dim=-1)
        expert_weights = route_weights[:, 1:].unsqueeze(-1)
        local_score = (expert_weights * local_scores).sum(dim=1)
        local_error = 1.0 - local_score
        context = self.context(router_input).unsqueeze(-1)
        amplitude = normalized.square().sum(dim=1, keepdim=True).sqrt()
        local_change = F.pad(
            (normalized[..., 1:] - normalized[..., :-1]).square().sum(dim=1, keepdim=True).sqrt(),
            (1, 0),
        )
        features = torch.cat(
            (normalized, amplitude, local_change, local_score.unsqueeze(1), local_error.unsqueeze(1)), dim=1
        ).to(dtype=source.dtype)
        hidden = F.silu(self.temporal_norm(self.temporal_in(features)) + context.to(dtype=features.dtype))
        correction = 0.10 * torch.tanh(self.temporal_out(hidden))
        confidence = (1.0 - route_weights[:, 0]).detach()
        bounded_scale = self.max_scale * torch.tanh(self.residual_scale / self.max_scale)
        refined = source + bounded_scale.to(source.dtype) * correction * scale.to(source.dtype)
        return refined, {
            "route_weights": route_weights.detach(),
            "density_summary": summaries.detach(),
            "confidence": confidence,
            "residual_scale": bounded_scale.detach().reshape(1),
        }

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 3 or outputs.size(1) != 2 * self.num_sources:
            raise ValueError(f"QAMDensityPrior1D expects [B, {2 * self.num_sources}, L], got {tuple(outputs.shape)}")
        batch, _, length = outputs.shape
        sources = outputs.reshape(batch, self.num_sources, 2, length)
        refined, routes, confidences, residual_scale = [], [], [], None
        for source_index in range(self.num_sources):
            value, aux = self._refine_one(sources[:, source_index])
            refined.append(value)
            routes.append(aux["route_weights"])
            confidences.append(aux["confidence"])
            residual_scale = aux["residual_scale"]
        self.last_aux = {
            "source_route_weights": torch.stack(routes, dim=1),
            "source_confidence": torch.stack(confidences, dim=1),
            "residual_scale": residual_scale,
        }
        return torch.stack(refined, dim=1).reshape(batch, 2 * self.num_sources, length)


def _rrc_taps(sps: int, rolloff: float, span: int, device, dtype) -> torch.Tensor:
    """Create a unit-energy root-raised-cosine matched-filter kernel."""
    sps = max(1, int(sps))
    span = max(2, int(span))
    beta = min(max(float(rolloff), 1e-4), 0.9999)
    count = span * sps + 1
    t = (torch.arange(count, device=device, dtype=dtype) - count // 2) / float(sps)
    numerator = torch.sin(math.pi * t * (1.0 - beta)) + 4.0 * beta * t * torch.cos(
        math.pi * t * (1.0 + beta)
    )
    denominator = math.pi * t * (1.0 - (4.0 * beta * t).square())
    taps = numerator / denominator
    at_zero = t.abs() < 1e-7
    zero_value = 1.0 + beta * (4.0 / math.pi - 1.0)
    taps = torch.where(at_zero, taps.new_tensor(zero_value), taps)
    singular = (t.abs() - 1.0 / (4.0 * beta)).abs() < 1e-6
    singular_value = beta / math.sqrt(2.0) * (
        (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
        + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
    )
    taps = torch.where(singular, taps.new_tensor(singular_value), taps)
    return taps / taps.square().sum().sqrt().clamp_min(1e-8)


class QAMTimingPrior1D(nn.Module):
    """Blind SPS/offset routing with a differentiable matched-filter view."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 16,
        sps_candidates: Sequence[int] = (10, 20),
        rrc_rolloff: float = 0.35,
        rrc_span: int = 12,
        axis_level_bank: Sequence[int] = (4, 8),
        route_temperature: float = 12.0,
        null_bias: float = 0.0,
        reliability_floor: float = 0.05,
        max_scale: float = 0.12,
        scale_init: float = 0.02,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.num_sources = max(1, int(num_sources))
        self.sps_candidates = tuple(sorted({max(1, int(value)) for value in sps_candidates}))
        if not self.sps_candidates:
            raise ValueError("sps_candidates must not be empty")
        self.rrc_rolloff = float(rrc_rolloff)
        self.rrc_span = max(2, int(rrc_span))
        self.axis_level_bank = tuple(sorted({max(2, int(value)) for value in axis_level_bank}))
        self.route_temperature = max(float(route_temperature), 1e-3)
        self.null_bias = float(null_bias)
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))
        self.max_scale = max(float(max_scale), 1e-6)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))
        hidden_channels = max(1, int(hidden_channels))
        conv_kernel = _odd_kernel(kernel_size)
        self.temporal_in = nn.Conv1d(7, hidden_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)
        self.temporal_norm = nn.InstanceNorm1d(hidden_channels, affine=True)
        self.temporal_out = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        nn.init.zeros_(self.temporal_out.weight)
        nn.init.zeros_(self.temporal_out.bias)
        self.last_aux: dict[str, torch.Tensor] = {}

    def _best_qam_error(self, symbols: torch.Tensor) -> torch.Tensor:
        best = None
        for levels_count in self.axis_level_bank:
            points = _square_points(levels_count).to(device=symbols.device, dtype=symbols.dtype)
            distance = (
                symbols[:, 0].unsqueeze(-1) - points[:, 0].view(1, 1, -1)
            ).square() + (
                symbols[:, 1].unsqueeze(-1) - points[:, 1].view(1, 1, -1)
            ).square()
            error = distance.amin(dim=-1).mean(dim=-1)
            best = error if best is None else torch.minimum(best, error)
        return best

    def _timing_one(self, source: torch.Tensor):
        scale = _complex_rms(source)
        normalized = source.float() / scale.float().clamp_min(1e-6)
        filtered_bank: list[torch.Tensor] = []
        branch_costs: list[torch.Tensor] = []
        branch_meta: list[tuple[int, int]] = []
        for sps in self.sps_candidates:
            taps = _rrc_taps(sps, self.rrc_rolloff, self.rrc_span, normalized.device, normalized.dtype)
            kernel = taps.view(1, 1, -1).repeat(2, 1, 1)
            filtered = F.conv1d(normalized, kernel, padding=taps.numel() // 2, groups=2)
            filtered_bank.append(filtered)
            for offset in range(sps):
                symbols = filtered[..., offset::sps]
                branch_costs.append(self._best_qam_error(symbols))
                branch_meta.append((sps, offset))

        costs = torch.stack(branch_costs, dim=1)
        null = costs.new_full((costs.size(0), 1), self.null_bias)
        route_weights = torch.softmax(
            torch.cat((null, -self.route_temperature * costs), dim=1), dim=1
        )
        sample_index = torch.arange(normalized.size(-1), device=normalized.device)
        timing_gate = normalized.new_zeros(normalized.size(0), normalized.size(-1))
        selected_filtered = normalized.new_zeros(normalized.shape)
        sps_weight: dict[int, torch.Tensor] = {
            sps: normalized.new_zeros(normalized.size(0)) for sps in self.sps_candidates
        }
        for branch_index, (sps, offset) in enumerate(branch_meta):
            weight = route_weights[:, branch_index + 1]
            mask = (sample_index.remainder(sps) == offset).to(dtype=normalized.dtype)
            timing_gate = timing_gate + weight.unsqueeze(-1) * mask.view(1, -1)
            sps_weight[sps] = sps_weight[sps] + weight
        for sps, filtered in zip(self.sps_candidates, filtered_bank):
            selected_filtered = selected_filtered + sps_weight[sps].view(-1, 1, 1) * filtered
        smooth_kernel_size = max(self.sps_candidates) * 2 + 1
        smooth_gate = F.avg_pool1d(
            timing_gate.unsqueeze(1), kernel_size=smooth_kernel_size, stride=1, padding=smooth_kernel_size // 2
        ).squeeze(1).clamp(0.0, 1.0)
        confidence = (1.0 - route_weights[:, 0]).clamp(0.0, 1.0)
        return normalized, selected_filtered, smooth_gate, confidence, route_weights

    def _refine_one(self, source: torch.Tensor):
        scale = _complex_rms(source)
        normalized, filtered, gate, confidence, route_weights = self._timing_one(source)
        amplitude = normalized.square().sum(dim=1, keepdim=True).sqrt()
        local_change = F.pad(
            (normalized[..., 1:] - normalized[..., :-1]).square().sum(dim=1, keepdim=True).sqrt(),
            (1, 0),
        )
        qam_gate = confidence.view(-1, 1, 1) * (
            self.reliability_floor + (1.0 - self.reliability_floor) * gate.unsqueeze(1)
        )
        features = torch.cat((normalized, filtered, amplitude, local_change, gate.unsqueeze(1)), dim=1).to(dtype=source.dtype)
        hidden = F.silu(self.temporal_norm(self.temporal_in(features)))
        correction = 0.10 * torch.tanh(self.temporal_out(hidden))
        bounded_scale = self.max_scale * torch.tanh(self.residual_scale / self.max_scale)
        refined = source + bounded_scale.to(source.dtype) * qam_gate.to(source.dtype) * correction * scale.to(source.dtype)
        return refined, {
            "route_weights": route_weights.detach(),
            "timing_gate": gate.detach(),
            "confidence": confidence.detach(),
            "residual_scale": bounded_scale.detach().reshape(1),
        }

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 3 or outputs.size(1) != 2 * self.num_sources:
            raise ValueError(f"QAMTimingPrior1D expects [B, {2 * self.num_sources}, L], got {tuple(outputs.shape)}")
        batch, _, length = outputs.shape
        sources = outputs.reshape(batch, self.num_sources, 2, length)
        refined, routes, gates, confidences, residual_scale = [], [], [], [], None
        for source_index in range(self.num_sources):
            value, aux = self._refine_one(sources[:, source_index])
            refined.append(value)
            routes.append(aux["route_weights"])
            gates.append(aux["timing_gate"])
            confidences.append(aux["confidence"])
            residual_scale = aux["residual_scale"]
        self.last_aux = {
            "source_route_weights": torch.stack(routes, dim=1),
            "source_timing_gate": torch.stack(gates, dim=1),
            "source_confidence": torch.stack(confidences, dim=1),
            "residual_scale": residual_scale,
        }
        return torch.stack(refined, dim=1).reshape(batch, 2 * self.num_sources, length)


class IQUMamba1D_QAMMMAUnrolled(_Stage4QAMWrapper):
    """Stage-4 + source-wise unfolded QAM MMA prior (Stage 228)."""

    def __init__(self, input_size: int, input_channels: int, n_stages: int, features_per_stage: List[int], conv_op: Type[nn.Conv1d], kernel_sizes: List[int], strides: List[int], n_conv_per_stage: List[int], num_classes: int, n_conv_per_stage_decoder: List[int], conv_bias: bool = True, norm_op: Type[nn.Module] = nn.InstanceNorm1d, norm_op_kwargs: dict = {"eps": 1e-5, "affine": True}, nonlin: Type[nn.Module] = nn.LeakyReLU, nonlin_kwargs: dict = {"inplace": True}, deep_supervision: bool = False, qam_mma_hidden_channels: int = 16, qam_mma_axis_level_bank: Sequence[int] = (4, 8), qam_mma_include_cross_128: bool = True, qam_mma_phase_bank: Sequence[float] = (-math.pi / 8.0, 0.0, math.pi / 8.0, math.pi / 4.0), qam_mma_num_unroll_steps: int = 3, qam_mma_step_init: float = 0.05, qam_mma_route_temperature: float = 8.0, qam_mma_null_bias: float = 0.0, qam_mma_reliability_floor: float = 0.05, qam_mma_max_scale: float = 0.12, qam_mma_scale_init: float = 0.005, qam_mma_kernel_size: int = 7, **kwargs) -> None:
        super().__init__()
        self.backbone = _build_stage4(input_size, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, deep_supervision)
        self.prior = QAMMMAUnrolledPrior1D(num_classes // 2, qam_mma_hidden_channels, qam_mma_axis_level_bank, qam_mma_include_cross_128, qam_mma_phase_bank, qam_mma_num_unroll_steps, qam_mma_step_init, qam_mma_route_temperature, qam_mma_null_bias, qam_mma_reliability_floor, qam_mma_max_scale, qam_mma_scale_init, qam_mma_kernel_size)
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        return self._forward_backbone_and_prior(x)


class IQUMamba1D_QAMDensityPrior(_Stage4QAMWrapper):
    """Stage-4 + source-wise constellation-density prior (Stage 229)."""

    def __init__(self, input_size: int, input_channels: int, n_stages: int, features_per_stage: List[int], conv_op: Type[nn.Conv1d], kernel_sizes: List[int], strides: List[int], n_conv_per_stage: List[int], num_classes: int, n_conv_per_stage_decoder: List[int], conv_bias: bool = True, norm_op: Type[nn.Module] = nn.InstanceNorm1d, norm_op_kwargs: dict = {"eps": 1e-5, "affine": True}, nonlin: Type[nn.Module] = nn.LeakyReLU, nonlin_kwargs: dict = {"inplace": True}, deep_supervision: bool = False, qam_density_hidden_channels: int = 16, qam_density_axis_level_bank: Sequence[int] = (4, 8), qam_density_include_cross_128: bool = True, qam_density_phase_bank: Sequence[float] = (-math.pi / 8.0, 0.0, math.pi / 8.0, math.pi / 4.0), qam_density_temperature: float = 12.0, qam_density_route_temperature: float = 4.0, qam_density_null_bias: float = 0.0, qam_density_max_scale: float = 0.12, qam_density_scale_init: float = 0.02, qam_density_kernel_size: int = 9, **kwargs) -> None:
        super().__init__()
        self.backbone = _build_stage4(input_size, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, deep_supervision)
        self.prior = QAMDensityPrior1D(num_classes // 2, qam_density_hidden_channels, qam_density_axis_level_bank, qam_density_include_cross_128, qam_density_phase_bank, qam_density_temperature, qam_density_route_temperature, qam_density_null_bias, qam_density_max_scale, qam_density_scale_init, qam_density_kernel_size)
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        return self._forward_backbone_and_prior(x)


class IQUMamba1D_QAMTimingPrior(_Stage4QAMWrapper):
    """Stage-4 + blind RRC timing/SPS prior (Stage 230)."""

    def __init__(self, input_size: int, input_channels: int, n_stages: int, features_per_stage: List[int], conv_op: Type[nn.Conv1d], kernel_sizes: List[int], strides: List[int], n_conv_per_stage: List[int], num_classes: int, n_conv_per_stage_decoder: List[int], conv_bias: bool = True, norm_op: Type[nn.Module] = nn.InstanceNorm1d, norm_op_kwargs: dict = {"eps": 1e-5, "affine": True}, nonlin: Type[nn.Module] = nn.LeakyReLU, nonlin_kwargs: dict = {"inplace": True}, deep_supervision: bool = False, qam_timing_hidden_channels: int = 16, qam_timing_sps_candidates: Sequence[int] = (10, 20), qam_timing_rrc_rolloff: float = 0.35, qam_timing_rrc_span: int = 12, qam_timing_axis_level_bank: Sequence[int] = (4, 8), qam_timing_route_temperature: float = 12.0, qam_timing_null_bias: float = 0.0, qam_timing_reliability_floor: float = 0.05, qam_timing_max_scale: float = 0.12, qam_timing_scale_init: float = 0.02, qam_timing_kernel_size: int = 9, **kwargs) -> None:
        super().__init__()
        self.backbone = _build_stage4(input_size, input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder, conv_bias, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, deep_supervision)
        self.prior = QAMTimingPrior1D(num_classes // 2, qam_timing_hidden_channels, qam_timing_sps_candidates, qam_timing_rrc_rolloff, qam_timing_rrc_span, qam_timing_axis_level_bank, qam_timing_route_temperature, qam_timing_null_bias, qam_timing_reliability_floor, qam_timing_max_scale, qam_timing_scale_init, qam_timing_kernel_size)
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        return self._forward_backbone_and_prior(x)


__all__ = [
    "QAMMMAUnrolledPrior1D",
    "QAMDensityPrior1D",
    "QAMTimingPrior1D",
    "IQUMamba1D_QAMMMAUnrolled",
    "IQUMamba1D_QAMDensityPrior",
    "IQUMamba1D_QAMTimingPrior",
]
