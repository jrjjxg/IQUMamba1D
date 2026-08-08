"""Stage 238: joint QAM receiver and interference-cancellation unfolding.

Unlike output-side constellation regularizers, this module treats separation as
an alternating receiver problem.  A Stage-4 separator supplies the initial
sources and every unfolded iteration performs four coupled operations:

1. blind RRC/polyphase timing and soft QAM demapping;
2. differentiable short complex-channel estimation by ridge regression;
3. mixture reconstruction and source-wise interference cancellation;
4. a shared learned proximal update driven by both receiver and data views.

The channel solve is important for the project datasets: mixtures and target
sources are normalized independently, so directly enforcing ``sum(s_k) = x``
is generally incorrect even when the underlying RF mixing is additive.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D


def _logit(value: float) -> float:
    value = min(max(float(value), 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def _complex_rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x.float().square().sum(dim=-2, keepdim=True).mean(dim=-1, keepdim=True).add(eps).sqrt()


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[..., 0, :].float(), x[..., 1, :].float())


def _from_complex(z: torch.Tensor) -> torch.Tensor:
    return torch.stack((z.real, z.imag), dim=-2)


def rectangular_qam_points(order: int) -> torch.Tensor:
    """Return unit-power rectangular QAM geometry used by MATLAB ``qammod``.

    Odd-bit constellations are rectangular.  In particular, 128-QAM is a
    16-by-8 grid, not the asymmetric truncated cross used by older project
    priors and BER helpers.
    """
    order = int(order)
    bits = int(round(math.log2(order)))
    if order < 4 or (1 << bits) != order:
        raise ValueError(f"QAM order must be a power of two >= 4, got {order}")
    i_count = 1 << ((bits + 1) // 2)
    q_count = 1 << (bits // 2)
    i_levels = torch.arange(-(i_count - 1), i_count, 2, dtype=torch.float32)
    q_levels = torch.arange(-(q_count - 1), q_count, 2, dtype=torch.float32)
    ii, qq = torch.meshgrid(i_levels, q_levels, indexing="ij")
    points = torch.stack((ii.reshape(-1), qq.reshape(-1)), dim=-1)
    return points / points.square().sum(dim=-1).mean().sqrt().clamp_min(1e-8)


class QAMConstellationBank(nn.Module):
    def __init__(self, orders: Sequence[int] = (16, 64, 128)) -> None:
        super().__init__()
        normalized_orders = tuple(dict.fromkeys(int(order) for order in orders))
        if not normalized_orders:
            raise ValueError("At least one QAM order is required")
        self.orders = normalized_orders
        for index, order in enumerate(self.orders):
            points = rectangular_qam_points(order)
            z = torch.complex(points[:, 0], points[:, 1])
            fourth_angle = torch.angle(z.pow(4).mean())
            self.register_buffer(f"points_{index}", points, persistent=False)
            self.register_buffer(f"fourth_angle_{index}", fourth_angle, persistent=False)

    def points(self, index: int, reference: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"points_{index}").to(device=reference.device, dtype=reference.dtype)

    def fourth_angle(self, index: int, reference: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"fourth_angle_{index}").to(device=reference.device, dtype=reference.dtype)


def _rrc_taps(sps: int, rolloff: float, span: int) -> torch.Tensor:
    sps = max(1, int(sps))
    span = max(2, int(span))
    beta = min(max(float(rolloff), 1e-4), 0.9999)
    count = span * sps + 1
    if count % 2 == 0:
        count += 1
    t = (torch.arange(count, dtype=torch.float64) - count // 2) / float(sps)
    taps = torch.empty_like(t)
    zero = t.abs() < 1e-10
    singular = (t.abs() - 1.0 / (4.0 * beta)).abs() < 1e-10
    regular = ~(zero | singular)
    tr = t[regular]
    taps[regular] = (
        torch.sin(math.pi * tr * (1.0 - beta))
        + 4.0 * beta * tr * torch.cos(math.pi * tr * (1.0 + beta))
    ) / (math.pi * tr * (1.0 - (4.0 * beta * tr).square()))
    taps[zero] = 1.0 + beta * (4.0 / math.pi - 1.0)
    taps[singular] = beta / math.sqrt(2.0) * (
        (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
        + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
    )
    taps = taps / taps.square().sum().sqrt().clamp_min(1e-12)
    return taps.float()


class DifferentiableQAMReceiver(nn.Module):
    """Blind timing/carrier recovery and soft QAM posterior reconstruction."""

    def __init__(
        self,
        qam_orders: Sequence[int] = (16, 64, 128),
        samples_per_symbol: int = 20,
        rrc_rolloff: float = 0.35,
        rrc_span: int = 20,
        posterior_temperature: float = 18.0,
        route_temperature: float = 10.0,
        route_entropy_weight: float = 0.02,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.bank = QAMConstellationBank(qam_orders)
        self.sps = max(1, int(samples_per_symbol))
        self.posterior_temperature = max(float(posterior_temperature), 1e-3)
        self.route_temperature = max(float(route_temperature), 1e-3)
        self.route_entropy_weight = max(float(route_entropy_weight), 0.0)
        self.eps = float(eps)
        self.register_buffer(
            "rrc_taps",
            _rrc_taps(self.sps, rrc_rolloff, rrc_span),
            persistent=False,
        )

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        taps = self.rrc_taps.to(device=x.device, dtype=x.dtype)
        kernel = taps.view(1, 1, -1).repeat(2, 1, 1)
        return F.conv1d(x, kernel, padding=taps.numel() // 2, groups=2)

    def _scatter_symbols(
        self,
        symbols: torch.Tensor,
        offset: int,
        output_length: int,
    ) -> torch.Tensor:
        indices = torch.arange(offset, output_length, self.sps, device=symbols.device)
        count = min(indices.numel(), symbols.size(-1))
        base = symbols.new_zeros(symbols.size(0), 2, output_length)
        if count == 0:
            return base
        return base.index_copy(2, indices[:count], symbols[..., :count])

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if source.ndim != 3 or source.size(1) != 2:
            raise ValueError(f"QAM receiver expects [B, 2, L], got {tuple(source.shape)}")
        original_dtype = source.dtype
        source_float = source.float()
        source_scale = _complex_rms(source_float, self.eps)
        normalized = source_float / source_scale
        matched = self._filter(normalized)
        branch_costs: list[torch.Tensor] = []
        branch_symbols: list[torch.Tensor] = []
        branch_offsets: list[int] = []
        branch_modulations: list[int] = []

        for offset in range(self.sps):
            sampled = matched[..., offset::self.sps]
            sampled_complex = _to_complex(sampled)
            symbol_scale = sampled_complex.abs().square().mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            symbols = sampled_complex / symbol_scale
            symbol_index = torch.arange(symbols.size(-1), device=symbols.device, dtype=symbols.real.dtype)
            if symbols.size(-1) > 1:
                fourth_step = symbols[:, 1:].pow(4) * torch.conj(symbols[:, :-1].pow(4))
                slope = torch.angle(fourth_step.mean(dim=-1).add(self.eps)) / 4.0
            else:
                slope = symbols.real.new_zeros(symbols.size(0))
            slope_track = slope.unsqueeze(-1) * symbol_index.unsqueeze(0)
            detrended = symbols * torch.exp(torch.complex(torch.zeros_like(slope_track), -slope_track))
            observed_fourth_angle = torch.angle(detrended.pow(4).mean(dim=-1).add(self.eps))

            for geometry_index, order in enumerate(self.bank.orders):
                reference_angle = self.bank.fourth_angle(geometry_index, symbols.real)
                phase = (observed_fourth_angle - reference_angle) / 4.0
                bits = int(round(math.log2(int(order))))
                # Odd-bit rectangular QAM has a 90-degree long/short-axis
                # ambiguity after fourth-power carrier recovery.
                orientation_offsets = (0.0, math.pi / 2.0) if bits % 2 else (0.0,)
                for orientation_offset in orientation_offsets:
                    phase_track = phase.unsqueeze(-1) + float(orientation_offset) + slope_track
                    canonical = symbols * torch.exp(
                        torch.complex(torch.zeros_like(phase_track), -phase_track)
                    )
                    points = self.bank.points(geometry_index, canonical.real)
                    point_complex = torch.complex(points[:, 0], points[:, 1])
                    distance = (
                        canonical.unsqueeze(-1) - point_complex.view(1, 1, -1)
                    ).abs().square()
                    posterior = torch.softmax(-self.posterior_temperature * distance, dim=-1)
                    posterior_mean = (
                        posterior * point_complex.view(1, 1, -1)
                    ).sum(dim=-1)
                    expected_distance = (posterior * distance).sum(dim=-1)
                    entropy = -(
                        posterior * posterior.clamp_min(self.eps).log()
                    ).sum(dim=-1)
                    normalized_entropy = entropy / math.log(float(points.size(0)))
                    branch_costs.append(
                        expected_distance.mean(dim=-1)
                        + self.route_entropy_weight * normalized_entropy.mean(dim=-1)
                    )
                    restored = posterior_mean * torch.exp(
                        torch.complex(torch.zeros_like(phase_track), phase_track)
                    ) * symbol_scale
                    branch_symbols.append(_from_complex(restored))
                    branch_offsets.append(offset)
                    branch_modulations.append(geometry_index)

        costs = torch.stack(branch_costs, dim=-1)
        route_weights = torch.softmax(-self.route_temperature * costs, dim=-1)
        impulse = normalized.new_zeros(normalized.shape)
        for branch_index, (symbols, offset) in enumerate(zip(branch_symbols, branch_offsets)):
            scattered = self._scatter_symbols(symbols, offset, normalized.size(-1))
            impulse = impulse + route_weights[:, branch_index].view(-1, 1, 1) * scattered
        synthesized = self._filter(impulse)

        prior_complex = _to_complex(synthesized)
        source_complex = _to_complex(normalized)
        gain = (
            (torch.conj(prior_complex) * source_complex).sum(dim=-1)
            / prior_complex.abs().square().sum(dim=-1).clamp_min(self.eps)
        ).detach()
        synthesized = _from_complex(prior_complex * gain.unsqueeze(-1))

        modulation_weights = torch.stack(
            [
                route_weights[:, [i for i, value in enumerate(branch_modulations) if value == mod]].sum(dim=-1)
                for mod in range(len(self.bank.orders))
            ],
            dim=-1,
        )
        timing_weights = torch.stack(
            [
                route_weights[:, [i for i, value in enumerate(branch_offsets) if value == offset]].sum(dim=-1)
                for offset in range(self.sps)
            ],
            dim=-1,
        )
        best_cost = costs.min(dim=-1).values
        confidence = torch.exp(-best_cost).clamp(0.0, 1.0)
        route_entropy_denominator = max(math.log(float(route_weights.size(-1))), self.eps)
        route_entropy = -(
            route_weights * route_weights.clamp_min(self.eps).log()
        ).sum(dim=-1) / route_entropy_denominator
        expected_cost = (route_weights * costs).sum(dim=-1)
        prior = synthesized * source_scale
        return prior.to(original_dtype), {
            "qam_expected_distance": expected_cost,
            "route_entropy": route_entropy,
            "confidence": confidence,
            "modulation_weights": modulation_weights,
            "timing_weights": timing_weights,
        }


def _shift_complex(x: torch.Tensor, lag: int) -> torch.Tensor:
    if lag == 0:
        return x
    zeros = x.new_zeros(x.size(0), abs(lag))
    if lag > 0:
        return torch.cat((zeros, x[:, :-lag]), dim=-1)
    return torch.cat((x[:, -lag:], zeros), dim=-1)


class ComplexChannelRidgeEstimator(nn.Module):
    """Alternating short complex-channel solve and matched inverse views."""

    def __init__(
        self,
        num_sources: int,
        num_taps: int = 3,
        ridge: float = 1e-3,
        detach_solution: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        num_taps = max(1, int(num_taps))
        if num_taps % 2 == 0:
            num_taps += 1
        radius = num_taps // 2
        self.lags = tuple(range(-radius, radius + 1))
        self.ridge = max(float(ridge), 1e-8)
        self.detach_solution = bool(detach_solution)
        self.eps = float(eps)

    def forward(
        self,
        sources: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if sources.ndim != 4 or sources.size(1) != self.num_sources or sources.size(2) != 2:
            raise ValueError("sources must have shape [B, K, 2, L]")
        source_complex = _to_complex(sources)
        mixture_complex = _to_complex(mixture)
        columns = [
            _shift_complex(source_complex[:, source_index], lag)
            for source_index in range(self.num_sources)
            for lag in self.lags
        ]
        design = torch.stack(columns, dim=-1)
        length = max(1, design.size(1))
        gram = torch.matmul(torch.conj(design).transpose(1, 2), design) / float(length)
        rhs = torch.matmul(torch.conj(design).transpose(1, 2), mixture_complex.unsqueeze(-1)) / float(length)
        trace_scale = gram.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(self.eps)
        identity = torch.eye(gram.size(-1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
        regularized = gram + self.ridge * trace_scale.view(-1, 1, 1) * identity
        coefficients = torch.linalg.solve(regularized, rhs).squeeze(-1)
        if self.detach_solution:
            coefficients = coefficients.detach()
        coefficients = coefficients.reshape(sources.size(0), self.num_sources, len(self.lags))

        contributions = []
        for source_index in range(self.num_sources):
            contribution = sum(
                coefficients[:, source_index, tap_index].unsqueeze(-1)
                * _shift_complex(source_complex[:, source_index], lag)
                for tap_index, lag in enumerate(self.lags)
            )
            contributions.append(contribution)
        contributions_complex = torch.stack(contributions, dim=1)
        reconstruction = contributions_complex.sum(dim=1)

        data_views = []
        for source_index in range(self.num_sources):
            source_observation = mixture_complex - (reconstruction - contributions_complex[:, source_index])
            numerator = sum(
                torch.conj(coefficients[:, source_index, tap_index]).unsqueeze(-1)
                * _shift_complex(source_observation, -lag)
                for tap_index, lag in enumerate(self.lags)
            )
            denominator = coefficients[:, source_index].abs().square().sum(dim=-1).clamp_min(self.eps)
            data_views.append(numerator / denominator.unsqueeze(-1))
        data_views_complex = torch.stack(data_views, dim=1)
        return (
            _from_complex(data_views_complex),
            _from_complex(reconstruction),
            _from_complex(contributions_complex),
            coefficients,
        )


class QAMTurboProximal(nn.Module):
    feature_channels = 13

    def __init__(self, hidden_channels: int = 32, kernel_size: int = 7) -> None:
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.input = nn.Conv1d(
            self.feature_channels,
            hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(1, hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(1, hidden_channels),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.output = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        current: torch.Tensor,
        data_view: torch.Tensor,
        qam_prior: torch.Tensor,
        mixture_residual: torch.Tensor,
        confidence: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        scale = _complex_rms(current, eps)
        features = torch.cat(
            (
                current / scale,
                data_view / scale,
                qam_prior / scale,
                mixture_residual / scale,
                (data_view - current) / scale,
                (qam_prior - current) / scale,
                confidence.view(-1, 1, 1).expand(-1, 1, current.size(-1)),
            ),
            dim=1,
        )
        hidden = self.input(features)
        for block in self.blocks:
            hidden = hidden + 0.5 * block(hidden)
        return torch.tanh(self.output(F.silu(hidden))) * scale


class IQUMamba1D_QAMTurboUnfold(nn.Module):
    """Stage-4 initialization followed by joint receiver/channel/IC unfolding."""

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
        qam_turbo_iterations: int = 3,
        qam_turbo_hidden_channels: int = 32,
        qam_turbo_kernel_size: int = 7,
        qam_turbo_orders: Sequence[int] = (16, 64, 128),
        qam_turbo_sps: int = 20,
        qam_turbo_rrc_rolloff: float = 0.35,
        qam_turbo_rrc_span: int = 20,
        qam_turbo_posterior_temperature: float = 18.0,
        qam_turbo_route_temperature: float = 10.0,
        qam_turbo_channel_taps: int = 3,
        qam_turbo_channel_ridge: float = 1e-3,
        qam_turbo_detach_channel_solve: bool = True,
        qam_turbo_data_step_init: float = 0.15,
        qam_turbo_prior_step_init: float = 0.08,
        qam_turbo_learned_step_init: float = 0.05,
        qam_turbo_eps: float = 1e-6,
        **_kwargs,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("QAMTurboUnfold requires one complex mixture with two I/Q channels")
        if num_classes % 2 != 0:
            raise ValueError("num_classes must contain complete I/Q source pairs")
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        self.num_sources = num_classes // 2
        self.num_iterations = max(1, int(qam_turbo_iterations))
        self.eps = float(qam_turbo_eps)
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
        self.receiver = DifferentiableQAMReceiver(
            qam_orders=qam_turbo_orders,
            samples_per_symbol=qam_turbo_sps,
            rrc_rolloff=qam_turbo_rrc_rolloff,
            rrc_span=qam_turbo_rrc_span,
            posterior_temperature=qam_turbo_posterior_temperature,
            route_temperature=qam_turbo_route_temperature,
            eps=self.eps,
        )
        self.channel_estimator = ComplexChannelRidgeEstimator(
            num_sources=self.num_sources,
            num_taps=qam_turbo_channel_taps,
            ridge=qam_turbo_channel_ridge,
            detach_solution=qam_turbo_detach_channel_solve,
            eps=self.eps,
        )
        self.proximal = QAMTurboProximal(
            hidden_channels=qam_turbo_hidden_channels,
            kernel_size=qam_turbo_kernel_size,
        )
        self.data_step_logits = nn.Parameter(torch.full(
            (self.num_iterations,),
            _logit(float(qam_turbo_data_step_init) / 0.5),
        ))
        self.prior_step_logits = nn.Parameter(torch.full(
            (self.num_iterations,),
            _logit(float(qam_turbo_prior_step_init) / 0.3),
        ))
        self.learned_step_logits = nn.Parameter(torch.full(
            (self.num_iterations,),
            _logit(float(qam_turbo_learned_step_init) / 0.2),
        ))
        self.last_aux: dict[str, torch.Tensor | list[torch.Tensor]] = {}

    def _bounded_delta(self, target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        scale = _complex_rms(current, self.eps)
        return 2.0 * scale * torch.tanh((target - current) / (2.0 * scale))

    def forward(self, mixture: torch.Tensor):
        coarse = self.backbone(mixture)
        if isinstance(coarse, (list, tuple)):
            coarse = coarse[0]
        batch, channels, length = coarse.shape
        expected_channels = 2 * self.num_sources
        if channels != expected_channels:
            raise ValueError(f"Expected Stage-4 output [B, {expected_channels}, L], got {tuple(coarse.shape)}")
        states = coarse.reshape(batch, self.num_sources, 2, length)
        intermediate_outputs: list[torch.Tensor] = []
        receiver_aux = None

        for iteration in range(self.num_iterations):
            priors, source_aux = [], []
            for source_index in range(self.num_sources):
                prior, aux = self.receiver(states[:, source_index])
                priors.append(prior)
                source_aux.append(aux)
            qam_priors = torch.stack(priors, dim=1)
            data_views, reconstruction, _contributions, _coefficients = self.channel_estimator(
                states,
                mixture,
            )
            mixture_residual = mixture.float() - reconstruction
            next_states = []
            data_step = 0.5 * torch.sigmoid(self.data_step_logits[iteration])
            prior_step = 0.3 * torch.sigmoid(self.prior_step_logits[iteration])
            learned_step = 0.2 * torch.sigmoid(self.learned_step_logits[iteration])
            for source_index in range(self.num_sources):
                confidence = source_aux[source_index]["confidence"]
                learned_delta = self.proximal(
                    states[:, source_index],
                    data_views[:, source_index],
                    qam_priors[:, source_index],
                    mixture_residual,
                    confidence,
                    self.eps,
                )
                next_state = (
                    states[:, source_index]
                    + data_step * self._bounded_delta(data_views[:, source_index], states[:, source_index])
                    + prior_step
                    * confidence.view(-1, 1, 1)
                    * self._bounded_delta(qam_priors[:, source_index], states[:, source_index])
                    + learned_step * learned_delta
                )
                next_states.append(next_state)
            states = torch.stack(next_states, dim=1)
            intermediate_outputs.append(states.reshape(batch, expected_channels, length))
            receiver_aux = source_aux

        data_views, reconstruction, contributions, coefficients = self.channel_estimator(states, mixture)
        final_output = states.reshape(batch, expected_channels, length).to(coarse.dtype)
        mixture_residual = mixture.float() - reconstruction
        assert receiver_aux is not None
        self.last_aux = {
            "intermediate_outputs": intermediate_outputs,
            "mixture_reconstruction": reconstruction,
            "mixture_residual": mixture_residual,
            "source_contributions": contributions,
            "mixing_coefficients_real": coefficients.real,
            "mixing_coefficients_imag": coefficients.imag,
            "qam_expected_distance": torch.stack(
                [aux["qam_expected_distance"] for aux in receiver_aux], dim=1
            ),
            "qam_route_entropy": torch.stack(
                [aux["route_entropy"] for aux in receiver_aux], dim=1
            ),
            "source_confidence": torch.stack(
                [aux["confidence"] for aux in receiver_aux], dim=1
            ),
            "modulation_weights": torch.stack(
                [aux["modulation_weights"] for aux in receiver_aux], dim=1
            ),
            "timing_weights": torch.stack(
                [aux["timing_weights"] for aux in receiver_aux], dim=1
            ),
            "data_views": data_views,
        }
        return final_output, self.last_aux


__all__ = [
    "rectangular_qam_points",
    "QAMConstellationBank",
    "DifferentiableQAMReceiver",
    "ComplexChannelRidgeEstimator",
    "QAMTurboProximal",
    "IQUMamba1D_QAMTurboUnfold",
]
