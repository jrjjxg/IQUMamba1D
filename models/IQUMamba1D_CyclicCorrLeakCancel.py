"""Stage-4 IQUMamba with cyclic-correlation leakage cancellation.

This wraps the original stage-4 IQUMamba with a conservative output-side head.
The head uses only the mixture and predicted sources:

1. estimate source-to-source leakage from predicted sources with a closed-form
   complex covariance ratio;
2. subtract a small, clipped pairwise leakage estimate;
3. optionally add the older learned cyclic-correlation leakage branch for
   ablations;
4. optionally apply soft mixture-consistency projection.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_EstimatedCycloFRESH import estimate_cyclic_frequency


def _parse_lags(lags: Sequence[int] | str) -> tuple[int, ...]:
    if isinstance(lags, str):
        parts = [part.strip() for part in lags.split(",") if part.strip()]
        parsed = tuple(int(part) for part in parts)
    else:
        parsed = tuple(int(lag) for lag in lags)
    parsed = tuple(lag for lag in parsed if lag >= 0)
    return parsed if parsed else (0,)


def _split_iq_sources(estimates: torch.Tensor) -> torch.Tensor:
    if estimates.dim() != 3 or estimates.size(1) % 2 != 0:
        raise ValueError(f"Expected source I/Q tensor (B, 2S, L), got {tuple(estimates.shape)}")
    return torch.complex(estimates[:, 0::2, :].float(), estimates[:, 1::2, :].float())


def _merge_iq_sources(sources: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    real = sources.real.to(dtype=dtype)
    imag = sources.imag.to(dtype=dtype)
    return torch.stack([real, imag], dim=2).flatten(1, 2)


def compute_cross_cyclic_correlation_features(
    estimates: torch.Tensor,
    base_freq: torch.Tensor,
    lags: Sequence[int] = (0, 1, 2, 4, 8),
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cross cyclic-correlation features for predicted source pairs.

    Returns:
        Tensor shaped (B, S, S, 4 * len(lags)).  The last dimension contains
        real/imag features for alpha = base_freq and alpha = 2*base_freq.
    """
    z = _split_iq_sources(estimates)
    batch, num_sources, length = z.shape
    lags = _parse_lags(lags)
    if length < 2:
        return estimates.new_zeros((batch, num_sources, num_sources, 4 * len(lags)))

    base = base_freq.detach().to(device=estimates.device, dtype=torch.float32).clamp(min=0.0, max=0.5)
    alphas = torch.stack([base, (2.0 * base).clamp(max=0.5)])
    n = torch.arange(length, device=estimates.device, dtype=torch.float32)
    power = z.abs().square().mean(dim=-1).clamp_min(eps)
    features: list[torch.Tensor] = []

    for alpha in alphas:
        phasor = torch.exp(-1j * (2.0 * math.pi * alpha * n))
        for lag in lags:
            if lag >= length:
                corr = z.new_zeros((batch, num_sources, num_sources))
            elif lag == 0:
                left = z.unsqueeze(2)
                right = z.conj().unsqueeze(1)
                corr = (left * right * phasor.view(1, 1, 1, -1)).mean(dim=-1)
            else:
                left = z[:, :, lag:].unsqueeze(2)
                right = z[:, :, :-lag].conj().unsqueeze(1)
                corr = (left * right * phasor[lag:].view(1, 1, 1, -1)).mean(dim=-1)
            denom = torch.sqrt(power.unsqueeze(2) * power.unsqueeze(1)).clamp_min(eps)
            corr = corr / denom
            features.append(corr.real)
            features.append(corr.imag)

    return torch.stack(features, dim=-1).to(dtype=estimates.dtype)


def soft_mixture_consistency_projection(
    estimates: torch.Tensor,
    mixture: torch.Tensor,
    mc_scale: torch.Tensor,
    weight_mode: str = "uniform",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply a differentiable soft mixture-consistency correction."""
    if mixture.dim() != 3 or mixture.size(1) != 2:
        raise ValueError(f"Expected mixture I/Q tensor (B, 2, L), got {tuple(mixture.shape)}")
    z = _split_iq_sources(estimates)
    mix = torch.complex(mixture[:, 0, :].float(), mixture[:, 1, :].float())
    residual = mix - z.sum(dim=1)

    if str(weight_mode).lower() == "energy":
        weights = z.abs().square().mean(dim=-1, keepdim=True).clamp_min(eps)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    else:
        weights = z.new_full((z.size(0), z.size(1), 1), 1.0 / float(z.size(1)))

    scale = torch.clamp(mc_scale.to(device=estimates.device, dtype=estimates.dtype), min=0.0, max=1.0)
    corrected = z + scale.float() * weights * residual.unsqueeze(1)
    return _merge_iq_sources(corrected, estimates.dtype)


def _clip_complex_magnitude(coeffs: torch.Tensor, limit: float, eps: float = 1e-8) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0.0:
        return coeffs
    magnitude = coeffs.abs()
    factor = torch.clamp(limit / magnitude.clamp_min(eps), max=1.0)
    return coeffs * factor


def compute_closed_form_leakage_coefficients(
    estimates: torch.Tensor,
    coeff_limit: float = 0.25,
    center: bool = True,
    detach_coeffs: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate pairwise complex leakage coefficients from predicted sources.

    If a predicted source has the form y_i = s_i + c_ij s_j and sources are
    weakly correlated, <y_i, y_j>/<y_j, y_j> estimates c_ij.  This is a
    deterministic BSS post-step, not a learned receiver or metadata prior.

    Returns:
        Complex tensor shaped (B, S, S), where coeffs[:, i, j] estimates
        leakage from source j into source i.  Diagonal entries are zeroed.
    """
    z = _split_iq_sources(estimates)
    if detach_coeffs:
        z_stat = z.detach()
    else:
        z_stat = z
    if center:
        z_stat = z_stat - z_stat.mean(dim=-1, keepdim=True)

    numerator = torch.einsum("bil,bjl->bij", z_stat, z_stat.conj()) / float(z_stat.size(-1))
    denominator = z_stat.abs().square().mean(dim=-1).clamp_min(eps)
    coeffs = numerator / denominator.unsqueeze(1)

    eye = torch.eye(z_stat.size(1), device=estimates.device, dtype=torch.bool).view(1, z_stat.size(1), z_stat.size(1))
    coeffs = coeffs.masked_fill(eye, 0.0)
    return _clip_complex_magnitude(coeffs, coeff_limit, eps=eps)


def apply_closed_form_leakage_cancellation(
    estimates: torch.Tensor,
    scale: torch.Tensor | float,
    coeff_limit: float = 0.25,
    center: bool = True,
    detach_coeffs: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply one small covariance-domain leakage cancellation step."""
    z = _split_iq_sources(estimates)
    coeffs = compute_closed_form_leakage_coefficients(
        estimates,
        coeff_limit=coeff_limit,
        center=center,
        detach_coeffs=detach_coeffs,
        eps=eps,
    )
    leakage = torch.einsum("bij,bjl->bil", coeffs, z)
    if not torch.is_tensor(scale):
        scale = estimates.new_tensor(float(scale))
    scale = torch.clamp(scale.to(device=estimates.device, dtype=estimates.dtype), min=0.0, max=1.0).float()
    corrected = z - scale * leakage
    return _merge_iq_sources(corrected, estimates.dtype)


class CyclicLeakageCancellationHead(nn.Module):
    """Output-side leakage cancellation head.

    Default mode is a deterministic covariance post-step.  The learned
    cyclic-correlation MLP is retained only for ablation through
    leakcancel_mode=learned or hybrid.
    """

    def __init__(
        self,
        num_sources: int,
        min_freq: float = 1.0 / 64.0,
        max_freq: float = 1.0 / 8.0,
        default_freq: float = 1.0 / 32.0,
        lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        hidden: int = 16,
        scale_init: float = 0.2,
        mc_scale_init: float = 0.0,
        mc_weight_mode: str = "uniform",
        mode: str = "covariance",
        coeff_limit: float = 0.25,
        zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        if self.num_sources < 2:
            raise ValueError("CyclicLeakageCancellationHead requires at least two sources")
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.default_freq = float(default_freq)
        self.lags = _parse_lags(lags)
        self.mc_weight_mode = str(mc_weight_mode)
        self.mode = str(mode).lower().strip()
        if self.mode not in {"covariance", "learned", "hybrid"}:
            raise ValueError(f"Unsupported leakcancel mode: {mode}")
        self.coeff_limit = float(coeff_limit)

        hidden = max(1, int(hidden))
        feat_dim = 4 * len(self.lags)
        self.coeff_head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        scale_init = float(min(max(scale_init, 1e-6), 1.0 - 1e-6))
        self.cancel_scale_logit = nn.Parameter(torch.logit(torch.tensor(scale_init)))
        self.mc_scale_logit = nn.Parameter(torch.logit(torch.tensor(float(min(max(mc_scale_init, 1e-6), 1.0 - 1e-6)))))

        if zero_init:
            last = self.coeff_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def _base_frequency(self, mixture: torch.Tensor) -> torch.Tensor:
        return estimate_cyclic_frequency(
            mixture.detach(),
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            default_freq=self.default_freq,
        ).clamp(min=self.min_freq, max=self.max_freq)

    def _current_cancel_scale(self, estimates: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.cancel_scale_logit).to(device=estimates.device, dtype=estimates.dtype)

    def _apply_learned_cyclic_cancellation(
        self,
        estimates: torch.Tensor,
        mixture: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        z = _split_iq_sources(estimates)
        base_freq = self._base_frequency(mixture)
        features = compute_cross_cyclic_correlation_features(estimates.detach(), base_freq, self.lags)
        coeffs = self.coeff_head(features).to(dtype=estimates.dtype)
        coeff_complex = torch.complex(coeffs[..., 0].float(), coeffs[..., 1].float())

        eye = torch.eye(self.num_sources, device=estimates.device, dtype=torch.bool).view(1, self.num_sources, self.num_sources)
        coeff_complex = coeff_complex.masked_fill(eye, 0.0)
        coeff_complex = _clip_complex_magnitude(coeff_complex, self.coeff_limit)
        leakage = torch.einsum("bij,bjl->bil", coeff_complex, z)
        corrected = z - scale.float() * leakage
        return _merge_iq_sources(corrected, estimates.dtype)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        cancel_scale = self._current_cancel_scale(estimates)
        corrected_tensor = estimates

        if self.mode in {"covariance", "hybrid"}:
            corrected_tensor = apply_closed_form_leakage_cancellation(
                corrected_tensor,
                scale=cancel_scale,
                coeff_limit=self.coeff_limit,
                center=True,
                detach_coeffs=True,
            )

        if self.mode in {"learned", "hybrid"}:
            corrected_tensor = self._apply_learned_cyclic_cancellation(
                corrected_tensor,
                mixture,
                scale=cancel_scale,
            )

        mc_scale = torch.sigmoid(self.mc_scale_logit)
        return soft_mixture_consistency_projection(
            corrected_tensor,
            mixture,
            mc_scale=mc_scale,
            weight_mode=self.mc_weight_mode,
        )


class IQUMamba1D_CyclicCorrLeakCancel(nn.Module):
    """Original IQUMamba plus output-side cyclic leakage cancellation."""

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
        cycliccorr_min_freq: float = 1.0 / 64.0,
        cycliccorr_max_freq: float = 1.0 / 8.0,
        cycliccorr_default_freq: float = 1.0 / 32.0,
        leakcancel_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        leakcancel_hidden: int = 16,
        leakcancel_scale_init: float = 0.2,
        leakcancel_mc_scale_init: float = 0.05,
        leakcancel_mc_weight_mode: str = "uniform",
        leakcancel_mode: str = "covariance",
        leakcancel_coeff_limit: float = 0.25,
        leakcancel_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be paired I/Q source channels, got {num_classes}")
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
        self.leakage_head = CyclicLeakageCancellationHead(
            num_sources=num_classes // 2,
            min_freq=cycliccorr_min_freq,
            max_freq=cycliccorr_max_freq,
            default_freq=cycliccorr_default_freq,
            lags=leakcancel_lags,
            hidden=leakcancel_hidden,
            scale_init=leakcancel_scale_init,
            mc_scale_init=leakcancel_mc_scale_init,
            mc_weight_mode=leakcancel_mc_weight_mode,
            mode=leakcancel_mode,
            coeff_limit=leakcancel_coeff_limit,
            zero_init=leakcancel_zero_init,
        )

    def _adapt_output(self, output: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        return self.leakage_head(output, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = self.backbone(x)
        if isinstance(outputs, (list, tuple)):
            return [self._adapt_output(out, x) for out in outputs]
        return self._adapt_output(outputs, x)
