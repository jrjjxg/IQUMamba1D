"""Stage 222: evidence-routed residual experts on top of the Stage-4 IQUMamba.

The backbone remains unchanged. A small, permutation-equivariant residual MoE
uses mixture/estimate evidence to select among identity, local, periodic and
leakage-correction refinements. The refinement heads are zero initialized so
the new model starts exactly at the Stage-4 solution.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D


def _complex_correlation(a: torch.Tensor, b: torch.Tensor, eps: float) -> torch.Tensor:
    """Return normalized complex correlation magnitude for ``[B, 2, T]`` tensors."""
    if a.size(-1) == 0 or b.size(-1) == 0:
        return a.new_zeros(a.size(0))
    real = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).mean(dim=-1)
    imag = (a[:, 1] * b[:, 0] - a[:, 0] * b[:, 1]).mean(dim=-1)
    power_a = a.square().mean(dim=(1, 2))
    power_b = b.square().mean(dim=(1, 2))
    denominator = (power_a * power_b).clamp_min(eps).sqrt()
    return torch.sqrt(real.square() + imag.square() + eps) / denominator


def _lagged_complex_correlation(x: torch.Tensor, lag: int, eps: float) -> torch.Tensor:
    if lag <= 0:
        return _complex_correlation(x, x, eps)
    if lag >= x.size(-1):
        return x.new_zeros(x.size(0))
    return _complex_correlation(x[..., lag:], x[..., :-lag], eps)


class UniversalEvidenceExtractor(nn.Module):
    """Extract modulation-agnostic evidence from a mixture and preliminary estimates.

    The features are deliberately low dimensional: closure residual, energy
    imbalance, source correlation, residual whiteness, lag-bank periodicity,
    phase continuity and envelope variation. They are used only for routing,
    and are detached from the separation loss to avoid a second optimization
    path through the evidence statistics.
    """

    evidence_dim = 8

    def __init__(self, lag_bank: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128), eps: float = 1e-8):
        super().__init__()
        lags = tuple(sorted({int(lag) for lag in lag_bank if int(lag) > 0}))
        self.register_buffer("lag_bank", torch.tensor(lags, dtype=torch.long), persistent=False)
        self.eps = float(eps)

    def forward(self, mixture: torch.Tensor, estimates: torch.Tensor) -> torch.Tensor:
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(f"mixture must have shape [B, 2, L], got {tuple(mixture.shape)}")
        if estimates.ndim != 3 or estimates.size(1) % 2 != 0:
            raise ValueError(
                f"estimates must have shape [B, 2*K, L], got {tuple(estimates.shape)}"
            )
        if mixture.size(0) != estimates.size(0) or mixture.size(-1) != estimates.size(-1):
            raise ValueError("mixture and estimates must agree in batch and sequence dimensions")

        batch, _, length = mixture.shape
        num_sources = estimates.size(1) // 2
        sources = estimates.reshape(batch, num_sources, 2, length)
        reconstruction = sources.sum(dim=1)
        residual = mixture - reconstruction

        mixture_rms = mixture.square().mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt()
        residual_ratio = (
            residual.square().mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt() / mixture_rms
        ).flatten(1)

        source_power = sources.square().mean(dim=(2, 3)).add(self.eps)
        source_cv = source_power.std(dim=1, unbiased=False) / source_power.mean(dim=1).clamp_min(self.eps)

        if num_sources > 1:
            pairwise = []
            for source_index in range(num_sources):
                for other_index in range(source_index + 1, num_sources):
                    pairwise.append(
                        _complex_correlation(
                            sources[:, source_index],
                            sources[:, other_index],
                            self.eps,
                        )
                    )
            source_correlation = torch.stack(pairwise, dim=0).mean(dim=0)
        else:
            source_correlation = mixture.new_zeros(batch)

        residual_whiteness = _lagged_complex_correlation(residual, 1, self.eps)

        if self.lag_bank.numel() > 0:
            periodicity = torch.stack(
                [_lagged_complex_correlation(mixture, int(lag.item()), self.eps) for lag in self.lag_bank],
                dim=1,
            )
            periodicity_peak = periodicity.max(dim=1).values
            periodicity_contrast = (periodicity_peak - periodicity.mean(dim=1)).relu()
        else:
            periodicity_peak = mixture.new_zeros(batch)
            periodicity_contrast = mixture.new_zeros(batch)

        phase_continuity = _lagged_complex_correlation(mixture, 1, self.eps)

        envelope = mixture.square().sum(dim=1).add(self.eps).sqrt()
        envelope_cv = envelope.std(dim=1, unbiased=False) / envelope.mean(dim=1).clamp_min(self.eps)

        evidence = torch.cat(
            [
                residual_ratio,
                source_cv[:, None],
                source_correlation[:, None],
                residual_whiteness[:, None],
                periodicity_peak[:, None],
                periodicity_contrast[:, None],
                phase_continuity[:, None],
                envelope_cv[:, None],
            ],
            dim=1,
        )
        return torch.nan_to_num(evidence, nan=0.0, posinf=1.0, neginf=0.0)


class SharedResidualExpert(nn.Module):
    """A small source-shared residual head with a distinct temporal receptive field."""

    def __init__(self, hidden_channels: int, dilation: int, max_delta: float):
        super().__init__()
        self.input_channels = 6
        self.hidden_channels = int(hidden_channels)
        self.dilation = int(dilation)
        self.max_delta = float(max_delta)
        padding = 2 * self.dilation
        self.in_proj = nn.Conv1d(self.input_channels, self.hidden_channels, 5, padding=padding, dilation=self.dilation)
        self.out_proj = nn.Conv1d(self.hidden_channels, 2, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, features: torch.Tensor, rms: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.in_proj(features))
        correction = torch.tanh(self.out_proj(hidden))
        return self.max_delta * rms * correction


class EvidenceRouter(nn.Module):
    def __init__(self, evidence_dim: int, hidden_channels: int, identity_bias: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(evidence_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, 0.0)
        with torch.no_grad():
            self.network[-1].bias[0] = float(identity_bias)

    def forward(self, evidence: torch.Tensor, temperature: float) -> torch.Tensor:
        temperature = max(float(temperature), 1e-3)
        return F.softmax(self.network(evidence) / temperature, dim=-1)


class EvidenceRoutedResidualMoE(nn.Module):
    """Permutation-equivariant mixture of identity and three shared refinements."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 12,
        max_delta: float = 0.15,
        identity_bias: float = 1.5,
        router_temperature: float = 1.0,
        route_hard_eval: bool = True,
        lag_bank: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        eps: float = 1e-8,
    ):
        super().__init__()
        if int(num_sources) < 2:
            raise ValueError("EvidenceRoutedResidualMoE requires at least two sources")
        self.num_sources = int(num_sources)
        self.eps = float(eps)
        self.router_temperature = float(router_temperature)
        self.route_hard_eval = bool(route_hard_eval)
        self.evidence = UniversalEvidenceExtractor(lag_bank=lag_bank, eps=eps)
        self.router = EvidenceRouter(self.evidence.evidence_dim, int(hidden_channels), identity_bias)
        self.experts = nn.ModuleList(
            [
                SharedResidualExpert(int(hidden_channels), dilation=1, max_delta=max_delta),
                SharedResidualExpert(int(hidden_channels), dilation=4, max_delta=max_delta),
                SharedResidualExpert(int(hidden_channels), dilation=16, max_delta=max_delta),
            ]
        )

    def forward(self, mixture: torch.Tensor, estimates: torch.Tensor):
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(f"mixture must have shape [B, 2, L], got {tuple(mixture.shape)}")
        expected_channels = 2 * self.num_sources
        if estimates.ndim != 3 or estimates.size(1) != expected_channels:
            raise ValueError(
                f"estimates must have shape [B, {expected_channels}, L], got {tuple(estimates.shape)}"
            )
        if mixture.size(0) != estimates.size(0) or mixture.size(-1) != estimates.size(-1):
            raise ValueError("mixture and estimates must agree in batch and sequence dimensions")

        batch, _, length = mixture.shape
        sources = estimates.reshape(batch, self.num_sources, 2, length)
        reconstruction = sources.sum(dim=1)
        residual = mixture - reconstruction
        if self.num_sources > 1:
            other_sources = (reconstruction[:, None] - sources) / float(self.num_sources - 1)
        else:
            other_sources = torch.zeros_like(sources)

        mixture_rms = mixture.square().mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt()
        normalized_mixture = mixture / mixture_rms
        normalized_sources = sources / mixture_rms[:, None]
        normalized_other = other_sources / mixture_rms[:, None]
        normalized_residual = residual / mixture_rms
        features = torch.cat(
            [normalized_sources, normalized_other, normalized_residual[:, None].expand(-1, self.num_sources, -1, -1)],
            dim=2,
        ).reshape(batch * self.num_sources, 6, length)
        source_rms = mixture_rms[:, None].expand(batch, self.num_sources, 1, 1).reshape(
            batch * self.num_sources, 1, 1
        )

        candidates = [estimates]
        for expert in self.experts:
            correction = expert(features, source_rms).reshape(batch, self.num_sources, 2, length)
            candidates.append((sources + correction).reshape(batch, expected_channels, length))

        with torch.no_grad():
            evidence = self.evidence(mixture.detach(), estimates.detach())
        soft_weights = self.router(evidence, self.router_temperature)
        route_weights = soft_weights
        if not self.training and self.route_hard_eval:
            selected = soft_weights.argmax(dim=-1)
            route_weights = F.one_hot(selected, num_classes=soft_weights.size(-1)).to(soft_weights.dtype)

        # Difference form preserves exact Stage-4 identity when every residual
        # head is zero initialized, independent of the router probabilities.
        output = candidates[0]
        for route_index, candidate in enumerate(candidates[1:], start=1):
            output = output + route_weights[:, route_index, None, None] * (candidate - candidates[0])
        auxiliary = {
            "candidate_outputs": candidates,
            "route_weights": route_weights,
            "soft_route_weights": soft_weights,
            "evidence": evidence,
        }
        return output, auxiliary


class PaddedStage4Backbone(nn.Module):
    """Keep the legacy Stage-4 encoder well-defined for odd-length crops."""

    def __init__(self, backbone: nn.Module, nominal_length: int):
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


class IQUMamba1D_EvidenceRoutedMoE(nn.Module):
    """Stage-4 IQUMamba plus a low-cost, evidence-routed residual MoE."""

    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        conv_bias: bool = True,
        norm_op=nn.InstanceNorm1d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision: bool = False,
        evidence_moe_hidden_channels: int = 12,
        evidence_moe_max_delta: float = 0.15,
        evidence_moe_identity_bias: float = 1.5,
        evidence_moe_router_temperature: float = 1.0,
        evidence_moe_route_hard_eval: bool = True,
        evidence_moe_lag_bank: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        evidence_moe_return_route_aux: bool = True,
    ):
        super().__init__()
        if int(num_classes) % 2 != 0:
            raise ValueError("num_classes must be even because each source has I/Q channels")
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        # The legacy encoder computes its channel-token width from the nominal
        # input length. Round that nominal width up to the encoder stride so an
        # odd-length inference crop still reaches the correct bottleneck width.
        total_stride = 1
        for stride in strides:
            total_stride *= max(1, int(stride))
        backbone_input_size = ((int(input_size) + total_stride - 1) // total_stride) * total_stride
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
        self.refiner = EvidenceRoutedResidualMoE(
            num_sources=int(num_classes) // 2,
            hidden_channels=evidence_moe_hidden_channels,
            max_delta=evidence_moe_max_delta,
            identity_bias=evidence_moe_identity_bias,
            router_temperature=evidence_moe_router_temperature,
            route_hard_eval=evidence_moe_route_hard_eval,
            lag_bank=evidence_moe_lag_bank,
        )
        self.return_route_aux = bool(evidence_moe_return_route_aux)

    def forward(self, x: torch.Tensor):
        base_output = self.backbone(x)
        if isinstance(base_output, (list, tuple)):
            base_estimates = base_output[-1]
        else:
            base_estimates = base_output
        output, auxiliary = self.refiner(x, base_estimates)
        if self.training and self.return_route_aux:
            return output, auxiliary
        return output
