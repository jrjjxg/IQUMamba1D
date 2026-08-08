"""Modulation-agnostic higher-order cumulant prior for complex IQ sources."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from util.low_snr_training import _pit_reorder_to_targets, _split_iq_sources


def complex_fourth_cumulants(
    signal: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gain-normalized complex C40 and real C42 along the last axis."""
    if not torch.is_complex(signal):
        raise ValueError("signal must be a complex tensor")
    centered = signal - signal.mean(dim=-1, keepdim=True)
    moment20 = centered.square().mean(dim=-1)
    power = centered.abs().square().mean(dim=-1)
    moment40 = centered.pow(4).mean(dim=-1)
    moment42 = centered.abs().pow(4).mean(dim=-1)
    denominator = power.square().clamp_min(float(eps))
    c40 = (moment40 - 3.0 * moment20.square()) / denominator
    c42 = (moment42 - moment20.abs().square() - 2.0 * power.square()) / denominator
    return c40, c42


def _to_complex_sources(x: torch.Tensor, num_sources: int) -> torch.Tensor:
    iq = _split_iq_sources(x, num_sources)
    return torch.complex(iq[:, :, 0].float(), iq[:, :, 1].float())


def _windowed(signal: torch.Tensor, window_size: int) -> torch.Tensor:
    length = signal.size(-1)
    window = min(max(8, int(window_size)), length)
    count = max(1, length // window)
    trimmed = signal[..., : count * window]
    return trimmed.reshape(*trimmed.shape[:-1], count, window)


def _signature(blocks: torch.Tensor, eps: float) -> torch.Tensor:
    c40, c42 = complex_fourth_cumulants(blocks, eps=eps)
    return torch.stack((c40.real, c40.imag, c42), dim=-1)


def _signature_confidence(signature: torch.Tensor, floor: float, eps: float) -> torch.Tensor:
    mean_energy = signature.mean(dim=-2).square().mean(dim=-1)
    variance = signature.var(dim=-2, unbiased=False).mean(dim=-1)
    reliability = mean_energy / (mean_energy + variance + float(eps))
    floor = min(max(float(floor), 0.0), 1.0)
    return floor + (1.0 - floor) * reliability


def _normalized_cross_cumulant(
    first: torch.Tensor,
    second: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    first = first - first.mean(dim=-1, keepdim=True)
    second = second - second.mean(dim=-1, keepdim=True)
    p_first = first.abs().square().mean(dim=-1)
    p_second = second.abs().square().mean(dim=-1)
    mixed_energy = (first.abs().square() * second.abs().square()).mean(dim=-1)
    covariance = (first * second.conj()).mean(dim=-1)
    pseudo_covariance = (first * second).mean(dim=-1)
    cumulant = (
        mixed_energy
        - p_first * p_second
        - covariance.abs().square()
        - pseudo_covariance.abs().square()
    )
    return cumulant / (p_first * p_second).clamp_min(float(eps))


def cumulant_prior_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    window_sizes: Sequence[int] = (256, 512, 1024),
    self_weight: float = 1.0,
    cross_weight: float = 0.25,
    confidence_floor: float = 0.1,
    beta: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Match source and cross-source fourth-order cumulants after PIT alignment."""
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[-1]
    if outputs.shape != targets.shape or outputs.ndim != 3 or outputs.size(1) % 2:
        raise ValueError(f"Expected matching (B, 2K, L), got {outputs.shape} and {targets.shape}")
    num_sources = outputs.size(1) // 2
    aligned = _pit_reorder_to_targets(outputs, targets, num_sources).reshape_as(outputs)
    pred = _to_complex_sources(aligned, num_sources)
    ref = _to_complex_sources(targets, num_sources)
    sizes = tuple(sorted({max(8, int(size)) for size in window_sizes}))
    if not sizes:
        raise ValueError("window_sizes must not be empty")

    self_terms = []
    cross_terms = []
    for size in sizes:
        pred_blocks = _windowed(pred, size)
        ref_blocks = _windowed(ref, size)
        pred_signature = _signature(pred_blocks, eps)
        with torch.no_grad():
            ref_signature = _signature(ref_blocks, eps)
            confidence = _signature_confidence(ref_signature, confidence_floor, eps)
        source_error = F.smooth_l1_loss(
            pred_signature,
            ref_signature,
            beta=float(beta),
            reduction="none",
        ).mean(dim=(-2, -1))
        self_terms.append((confidence * source_error).mean())

        if num_sources > 1 and float(cross_weight) > 0.0:
            pair_errors = []
            for first in range(num_sources):
                for second in range(first + 1, num_sources):
                    pred_cross = _normalized_cross_cumulant(
                        pred_blocks[:, first], pred_blocks[:, second], eps
                    )
                    with torch.no_grad():
                        ref_cross = _normalized_cross_cumulant(
                            ref_blocks[:, first], ref_blocks[:, second], eps
                        )
                        pair_confidence = 0.5 * (confidence[:, first] + confidence[:, second])
                    pair_error = F.smooth_l1_loss(
                        pred_cross,
                        ref_cross,
                        beta=float(beta),
                        reduction="none",
                    ).mean(dim=-1)
                    pair_errors.append((pair_confidence * pair_error).mean())
            cross_terms.append(torch.stack(pair_errors).mean())

    self_loss = torch.stack(self_terms).mean()
    if cross_terms:
        cross_loss = torch.stack(cross_terms).mean()
    else:
        cross_loss = self_loss.new_zeros(())
    return float(self_weight) * self_loss + float(cross_weight) * cross_loss


def gaussian_residual_prior_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
    *,
    window_sizes: Sequence[int] = (256, 512, 1024),
    residual_weight: float = 1.0,
    cross_weight: float = 0.25,
    confidence_floor: float = 0.1,
    beta: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize non-Gaussian residuals after PIT source alignment.

    The separator is allowed to leave receiver noise in the residual:

        residual = mixture - sum(predicted_sources)

    For circular Gaussian receiver noise, normalized fourth-order cumulants
    should be close to zero.  Source-residual cross-cumulants should also be
    close to zero because the clean sources and receiver noise are independent.
    This is a training-only constraint and adds no inference path.
    """
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[-1]
    if (
        outputs.ndim != 3
        or targets.ndim != 3
        or mixture.ndim != 3
        or outputs.shape != targets.shape
        or outputs.size(1) % 2 != 0
        or mixture.size(1) != 2
        or mixture.size(0) != outputs.size(0)
        or mixture.size(-1) != outputs.size(-1)
    ):
        raise ValueError(
            "Expected outputs/targets with shape (B, 2K, L) and mixture "
            f"with shape (B, 2, L), got {tuple(outputs.shape)}, "
            f"{tuple(targets.shape)}, and {tuple(mixture.shape)}"
        )

    num_sources = outputs.size(1) // 2
    # PIT returns the structured source view [B, K, 2, L].  Convert it back
    # to the public separator contract before passing it to helpers that
    # validate and split [B, 2K, L] tensors.
    aligned = _pit_reorder_to_targets(outputs, targets, num_sources).reshape_as(outputs)
    predicted_sources = _to_complex_sources(aligned, num_sources)
    mixture_complex = torch.complex(
        mixture[:, 0].float(),
        mixture[:, 1].float(),
    )
    residual = mixture_complex - predicted_sources.sum(dim=1)
    reference_sources = _to_complex_sources(targets, num_sources)

    sizes = tuple(sorted({max(8, int(size)) for size in window_sizes}))
    if not sizes:
        raise ValueError("window_sizes must not be empty")

    residual_terms = []
    cross_terms = []
    for size in sizes:
        residual_blocks = _windowed(residual, size)
        residual_signature = _signature(residual_blocks, eps)
        residual_error = F.smooth_l1_loss(
            residual_signature,
            torch.zeros_like(residual_signature),
            beta=float(beta),
            reduction="none",
        ).mean(dim=(-2, -1))

        with torch.no_grad():
            reference_blocks = _windowed(reference_sources, size)
            reference_signature = _signature(reference_blocks, eps)
            source_confidence = _signature_confidence(
                reference_signature,
                confidence_floor,
                eps,
            )
            residual_confidence = source_confidence.mean(dim=-1)
        residual_terms.append((residual_confidence * residual_error).mean())

        if num_sources > 0 and float(cross_weight) > 0.0:
            predicted_blocks = _windowed(predicted_sources, size)
            pair_errors = []
            for source_idx in range(num_sources):
                cross_cumulant = _normalized_cross_cumulant(
                    predicted_blocks[:, source_idx],
                    residual_blocks,
                    eps,
                )
                cross_cumulant = torch.nan_to_num(
                    cross_cumulant,
                    nan=0.0,
                    posinf=10.0,
                    neginf=-10.0,
                ).clamp(-10.0, 10.0)
                cross_error = F.smooth_l1_loss(
                    cross_cumulant,
                    torch.zeros_like(cross_cumulant),
                    beta=float(beta),
                    reduction="none",
                ).mean(dim=-1)
                pair_errors.append(
                    (source_confidence[:, source_idx] * cross_error).mean()
                )
            cross_terms.append(torch.stack(pair_errors).mean())

    residual_loss = torch.stack(residual_terms).mean()
    if cross_terms:
        cross_loss = torch.stack(cross_terms).mean()
    else:
        cross_loss = residual_loss.new_zeros(())
    return float(residual_weight) * residual_loss + float(cross_weight) * cross_loss
