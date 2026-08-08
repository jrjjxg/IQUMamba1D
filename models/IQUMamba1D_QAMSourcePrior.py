"""Stage-227: source-wise QAM geometry refinement on top of Stage-4.

The earlier QAM prior adapters operate on the received mixture.  A mixture of
two QAM signals is generally not itself a QAM constellation, so that placement
can turn a useful source prior into a harmful input distortion.  This module
keeps the Stage-4 IQUMamba separator unchanged and applies a small, bounded
QAM prior after the coarse source estimates are produced.

The prior is blind at inference time:

* every estimated source is normalized independently;
* a soft bank contains square 16/64-QAM geometries and a cross-128-QAM
  geometry;
* phase candidates and a null candidate are routed by projection consistency;
* a per-sample confidence gate avoids pulling pulse transitions toward a
  constellation point;
* the final correction is a small residual, initialized close to identity.

No modulation label, source name, target, or symbol timing metadata is used in
the forward path.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D


def _odd_kernel(kernel_size: int) -> int:
    value = max(1, int(kernel_size))
    return value if value % 2 == 1 else value + 1


def _square_qam_points(axis_levels: int) -> torch.Tensor:
    """Return unit-average-energy square-QAM points as ``[M, 2]``."""
    axis_levels = max(2, int(axis_levels))
    levels = torch.arange(
        -(axis_levels - 1), axis_levels, 2, dtype=torch.float32
    )
    ii, qq = torch.meshgrid(levels, levels, indexing="ij")
    points = torch.stack((ii.reshape(-1), qq.reshape(-1)), dim=-1)
    power = points.square().sum(dim=-1).mean().clamp_min(1e-8)
    return points / power.sqrt()


def _cross_128_qam_points() -> torch.Tensor:
    """Return the project-compatible cross-shaped 128-QAM geometry."""
    points: list[tuple[float, float]] = []
    for i in range(-7, 8):
        for q in range(-7, 8):
            if abs(i) + abs(q) <= 11:
                points.append((float(i), float(q)))
            if len(points) >= 128:
                break
        if len(points) >= 128:
            break
    if len(points) < 128:
        raise RuntimeError("Unable to construct the 128-QAM cross constellation")
    values = torch.tensor(points[:128], dtype=torch.float32)
    power = values.square().sum(dim=-1).mean().clamp_min(1e-8)
    return values / power.sqrt()


def _complex_rms(source: torch.Tensor, eps: float) -> torch.Tensor:
    """Return complex RMS with shape ``[B, 1, 1]`` for ``[B, 2, L]``."""
    return torch.sqrt(source.square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True) + eps)


def _rotate_iq(source: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate ``[B, 2, L]`` by a scalar angle tensor."""
    cosine = torch.cos(theta).to(device=source.device, dtype=source.dtype)
    sine = torch.sin(theta).to(device=source.device, dtype=source.dtype)
    real = source[:, 0]
    imag = source[:, 1]
    return torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), dim=1)


class SourceWiseSoftQAMGeometry(nn.Module):
    """Differentiable source-wise QAM projection and soft geometry routing."""

    def __init__(
        self,
        axis_level_bank: Sequence[int] = (4, 8),
        include_cross_128: bool = True,
        phase_bank: Sequence[float] = (
            -math.pi / 8.0,
            -math.pi / 16.0,
            0.0,
            math.pi / 16.0,
            math.pi / 8.0,
            math.pi / 4.0,
        ),
        projection_temperature: float = 24.0,
        route_temperature: float = 10.0,
        null_bias: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        axis_bank = tuple(sorted({max(2, int(value)) for value in axis_level_bank}))
        if not axis_bank and not include_cross_128:
            raise ValueError("At least one QAM geometry is required")
        phases = tuple(float(value) for value in phase_bank)
        if not phases:
            raise ValueError("At least one QAM phase candidate is required")
        self.axis_level_bank = axis_bank
        self.include_cross_128 = bool(include_cross_128)
        self.phase_bank = phases
        self.projection_temperature = max(float(projection_temperature), 1e-3)
        self.route_temperature = max(float(route_temperature), 1e-3)
        self.null_bias = float(null_bias)
        self.eps = float(eps)

        geometries: list[torch.Tensor] = []
        geometry_names: list[str] = []
        for levels in axis_bank:
            geometries.append(_square_qam_points(levels))
            geometry_names.append(f"{levels * levels}QAM")
        if include_cross_128:
            geometries.append(_cross_128_qam_points())
            geometry_names.append("128QAM-cross")
        for index, points in enumerate(geometries):
            self.register_buffer(f"geometry_{index}", points, persistent=False)
        self.geometry_names = tuple(geometry_names)
        self.num_geometries = len(geometries)
        self.num_branches = self.num_geometries * len(phases)
        self.register_buffer(
            "phase_tensor", torch.tensor(phases, dtype=torch.float32), persistent=False
        )
        # This bias is deliberately small: projection consistency is the main
        # routing signal, while the parameter can absorb dataset-specific
        # frequency of 16/64/128-QAM branches.
        self.branch_bias = nn.Parameter(torch.zeros(self.num_branches))

    @property
    def branch_names(self) -> tuple[str, ...]:
        return tuple(
            f"{geometry_name}@{phase:.5f}"
            for phase in self.phase_bank
            for geometry_name in self.geometry_names
        )

    def _geometry(self, index: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return getattr(self, f"geometry_{index}").to(device=device, dtype=dtype)

    def _soft_project(
        self,
        rotated: torch.Tensor,
        points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project a rotated source to a point cloud.

        Returns a projected point sequence and a per-sample projection error.
        The temperature is a multiplier on squared Euclidean distance, making
        the same implementation work for square and cross QAM geometries.
        """
        real = rotated[:, 0].unsqueeze(-1)
        imag = rotated[:, 1].unsqueeze(-1)
        point_real = points[:, 0].view(1, 1, -1)
        point_imag = points[:, 1].view(1, 1, -1)
        distance = (real - point_real).square() + (imag - point_imag).square()
        weights = torch.softmax(-self.projection_temperature * distance, dim=-1)
        projected_real = (weights * point_real).sum(dim=-1)
        projected_imag = (weights * point_imag).sum(dim=-1)
        projected = torch.stack((projected_real, projected_imag), dim=1)
        error = (weights * distance).sum(dim=-1)
        return projected, error

    def forward(self, normalized_sources: torch.Tensor) -> dict[str, torch.Tensor]:
        if normalized_sources.ndim != 3 or normalized_sources.size(1) != 2:
            raise ValueError(
                "SourceWiseSoftQAMGeometry expects [B, 2, L], got "
                f"{tuple(normalized_sources.shape)}"
            )

        source = normalized_sources.float()
        deltas: list[torch.Tensor] = []
        errors: list[torch.Tensor] = []
        phase_tensor = self.phase_tensor.to(device=source.device, dtype=source.dtype)
        branch_index = 0
        for theta in phase_tensor:
            rotated = _rotate_iq(source, theta)
            cosine = torch.cos(theta)
            sine = torch.sin(theta)
            for geometry_index in range(self.num_geometries):
                projected, error = self._soft_project(
                    rotated, self._geometry(geometry_index, source.device, source.dtype)
                )
                # Rotate the projected point back to the original source frame.
                projected_back = torch.stack(
                    (
                        projected[:, 0] * cosine + projected[:, 1] * sine,
                        -projected[:, 0] * sine + projected[:, 1] * cosine,
                    ),
                    dim=1,
                )
                deltas.append(projected_back - source)
                errors.append(error)
                branch_index += 1

        branch_errors = torch.stack(errors, dim=1)  # [B, N, L]
        branch_deltas = torch.stack(deltas, dim=1)  # [B, N, 2, L]
        mean_errors = branch_errors.mean(dim=-1)
        logits = -self.route_temperature * mean_errors + self.branch_bias.to(
            device=source.device, dtype=source.dtype
        ).view(1, -1)
        null = source.new_full((source.size(0), 1), self.null_bias)
        route_weights = torch.softmax(torch.cat((null, logits), dim=1), dim=1)

        expert_weights = route_weights[:, 1:].view(source.size(0), -1, 1, 1)
        prior_delta = (expert_weights * branch_deltas).sum(dim=1)
        best_error, _ = branch_errors.min(dim=1)
        confidence = torch.exp(-self.projection_temperature * best_error).clamp(0.0, 1.0)
        return {
            "prior_delta": prior_delta,
            "confidence": confidence,
            "route_weights": route_weights,
            "branch_errors": branch_errors,
        }


class QAMSourcePriorAdapter1D(nn.Module):
    """Bounded QAM residual adapter operating independently on each source."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 16,
        axis_level_bank: Sequence[int] = (4, 8),
        include_cross_128: bool = True,
        phase_bank: Sequence[float] = (
            -math.pi / 8.0,
            -math.pi / 16.0,
            0.0,
            math.pi / 16.0,
            math.pi / 8.0,
            math.pi / 4.0,
        ),
        projection_temperature: float = 24.0,
        route_temperature: float = 10.0,
        null_bias: float = 0.0,
        reliability_floor: float = 0.05,
        max_scale: float = 0.15,
        scale_init: float = 0.0,
        max_refine: float = 0.10,
        kernel_size: int = 9,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_sources = max(1, int(num_sources))
        self.reliability_floor = float(min(max(reliability_floor, 0.0), 1.0))
        self.max_scale = max(float(max_scale), 1e-6)
        self.max_refine = max(float(max_refine), 0.0)
        self.eps = float(eps)
        self.geometry = SourceWiseSoftQAMGeometry(
            axis_level_bank=axis_level_bank,
            include_cross_128=include_cross_128,
            phase_bank=phase_bank,
            projection_temperature=projection_temperature,
            route_temperature=route_temperature,
            null_bias=null_bias,
            eps=eps,
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        padding = kernel_size // 2
        # [source, projected point, projection delta, confidence, local change]
        self.refine_net = nn.Sequential(
            nn.Conv1d(8, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm1d(hidden_channels, affine=True),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        nn.init.zeros_(self.refine_net[-1].weight)
        nn.init.zeros_(self.refine_net[-1].bias)
        self.last_aux: dict[str, torch.Tensor] = {}

    def _bounded_scale(self) -> torch.Tensor:
        return self.max_scale * torch.tanh(self.residual_scale / self.max_scale)

    def _refine_one(self, source: torch.Tensor) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scale = _complex_rms(source, self.eps)
        normalized = source.float() / scale.float().clamp_min(self.eps)
        geometry = self.geometry(normalized)
        prior_delta = geometry["prior_delta"]
        projected = normalized + prior_delta
        confidence = geometry["confidence"].unsqueeze(1)
        local_change = F.pad(
            (normalized[..., 1:] - normalized[..., :-1]).square().sum(dim=1, keepdim=True).sqrt(),
            (1, 0),
        )
        features = torch.cat(
            [normalized, projected, prior_delta, confidence, local_change], dim=1
        ).to(dtype=source.dtype)
        learned_refine = self.max_refine * torch.tanh(self.refine_net(features))

        # Confidence is per time sample.  A nonzero floor keeps the prior
        # trainable at low SNR but prevents a hard projection of every sample.
        gate = self.reliability_floor + (1.0 - self.reliability_floor) * confidence
        normalized_delta = prior_delta.to(dtype=source.dtype) + learned_refine
        bounded_scale = self._bounded_scale().to(dtype=source.dtype)
        refined = source + bounded_scale * gate * normalized_delta * scale.to(dtype=source.dtype)
        aux = {
            "route_weights": geometry["route_weights"].detach(),
            "confidence": confidence.squeeze(1).detach(),
            "gate": gate.squeeze(1).detach(),
            "residual_scale": bounded_scale.detach().reshape(1),
        }
        return refined, aux

    def forward(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.ndim != 3 or outputs.size(1) != 2 * self.num_sources:
            raise ValueError(
                "QAMSourcePriorAdapter1D expects [B, 2K, L] with K=num_sources, got "
                f"{tuple(outputs.shape)}"
            )
        batch, _, length = outputs.shape
        sources = outputs.reshape(batch, self.num_sources, 2, length)
        refined_sources: list[torch.Tensor] = []
        route_weights: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        residual_scale = None
        for source_index in range(self.num_sources):
            refined, aux = self._refine_one(sources[:, source_index])
            refined_sources.append(refined)
            route_weights.append(aux["route_weights"])
            confidences.append(aux["confidence"])
            gates.append(aux["gate"])
            residual_scale = aux["residual_scale"]
        refined = torch.stack(refined_sources, dim=1).reshape(batch, 2 * self.num_sources, length)
        self.last_aux = {
            "source_route_weights": torch.stack(route_weights, dim=1),
            "source_confidence": torch.stack(confidences, dim=1),
            "source_gate": torch.stack(gates, dim=1),
            "residual_scale": residual_scale,
        }
        return refined


class IQUMamba1D_QAMSourcePrior(nn.Module):
    """Stage-4 IQUMamba followed by a source-wise QAM prior refinement."""

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
        qam_source_hidden_channels: int = 16,
        qam_source_axis_level_bank: Sequence[int] = (4, 8),
        qam_source_include_cross_128: bool = True,
        qam_source_phase_bank: Sequence[float] = (
            -math.pi / 8.0,
            -math.pi / 16.0,
            0.0,
            math.pi / 16.0,
            math.pi / 8.0,
            math.pi / 4.0,
        ),
        qam_source_projection_temperature: float = 24.0,
        qam_source_route_temperature: float = 10.0,
        qam_source_null_bias: float = 0.0,
        qam_source_reliability_floor: float = 0.05,
        qam_source_max_scale: float = 0.15,
        qam_source_scale_init: float = 0.0,
        qam_source_max_refine: float = 0.10,
        qam_source_kernel_size: int = 9,
        **kwargs,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("IQUMamba1D_QAMSourcePrior expects a 2-channel I/Q mixture")
        if int(num_classes) % 2 != 0:
            raise ValueError(f"num_classes must be even for source-wise I/Q output, got {num_classes}")
        self.num_sources = int(num_classes) // 2
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
        self.qam_prior = QAMSourcePriorAdapter1D(
            num_sources=self.num_sources,
            hidden_channels=qam_source_hidden_channels,
            axis_level_bank=qam_source_axis_level_bank,
            include_cross_128=qam_source_include_cross_128,
            phase_bank=qam_source_phase_bank,
            projection_temperature=qam_source_projection_temperature,
            route_temperature=qam_source_route_temperature,
            null_bias=qam_source_null_bias,
            reliability_floor=qam_source_reliability_floor,
            max_scale=qam_source_max_scale,
            scale_init=qam_source_scale_init,
            max_refine=qam_source_max_refine,
            kernel_size=qam_source_kernel_size,
        )
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor):
        output = self.backbone(x)
        if isinstance(output, (list, tuple)):
            refined = [self.qam_prior(item) for item in output]
            if self.qam_prior.last_aux:
                self.last_aux = self.qam_prior.last_aux
            return type(output)(refined)
        refined = self.qam_prior(output)
        self.last_aux = self.qam_prior.last_aux
        return refined


__all__ = [
    "SourceWiseSoftQAMGeometry",
    "QAMSourcePriorAdapter1D",
    "IQUMamba1D_QAMSourcePrior",
]
