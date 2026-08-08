"""Training-only counterfactual routing losses for evidence-routed MoE stages."""

from __future__ import annotations

import itertools

import torch
from torch.nn import functional as F


def _pit_smooth_l1_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("prediction and target must have shape [B, 2*K, L]")
    if prediction.shape != target.shape or prediction.size(1) % 2 != 0:
        raise ValueError("prediction and target must have the same [B, 2*K, L] shape")
    num_sources = prediction.size(1) // 2
    target_sources = target.reshape(target.size(0), num_sources, 2, target.size(-1))
    losses = []
    for permutation in itertools.permutations(range(num_sources)):
        ordered = prediction.reshape(prediction.size(0), num_sources, 2, prediction.size(-1))[:, permutation]
        losses.append(F.smooth_l1_loss(ordered, target_sources, reduction="none").mean(dim=(1, 2, 3)))
    return torch.stack(losses, dim=1).min(dim=1).values


def _pit_si_snr_huber_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
) -> torch.Tensor:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("prediction and target must have shape [B, 2*K, L]")
    if prediction.shape != target.shape or prediction.size(1) % 2 != 0:
        raise ValueError("prediction and target must have the same [B, 2*K, L] shape")
    from util.metrics import _si_snr_complex_per_item

    num_sources = prediction.size(1) // 2
    predictions = prediction.reshape(prediction.size(0), num_sources, 2, prediction.size(-1))
    targets = target.reshape(target.size(0), num_sources, 2, target.size(-1))
    permutation_losses = []
    for permutation in itertools.permutations(range(num_sources)):
        source_losses = []
        for target_index, prediction_index in enumerate(permutation):
            predicted_source = predictions[:, prediction_index]
            target_source = targets[:, target_index]
            si_snr = _si_snr_complex_per_item(
                predicted_source, target_source, zero_mean=True
            )
            huber = F.huber_loss(
                predicted_source, target_source, delta=float(delta), reduction="none"
            ).mean(dim=(1, 2))
            source_losses.append(-float(alpha) * si_snr + float(beta) * huber)
        permutation_losses.append(torch.stack(source_losses, dim=1).mean(dim=1))
    return torch.stack(permutation_losses, dim=1).min(dim=1).values


def counterfactual_route_loss(
    candidate_outputs,
    target: torch.Tensor,
    route_weights: torch.Tensor,
    temperature: float = 0.25,
    quality_loss: str = "smooth_l1",
    si_snr_alpha: float = 0.1,
    huber_beta: float = 1.0,
    huber_delta: float = 0.5,
):
    """Supervise the router using detached per-candidate PIT losses.

    Every candidate still receives the ordinary separator loss. This auxiliary
    term only teaches the router to prefer the candidate that is currently
    better for the sample, so it cannot force a modulation-specific expert.
    """
    if not isinstance(candidate_outputs, (list, tuple)) or not candidate_outputs:
        raise ValueError("candidate_outputs must be a non-empty list or tuple")
    if route_weights.ndim != 2 or route_weights.size(1) != len(candidate_outputs):
        raise ValueError("route_weights must have one column per candidate")
    if quality_loss == "smooth_l1":
        loss_fn = _pit_smooth_l1_per_sample
    elif quality_loss == "pit_si_snr_huber":
        loss_fn = lambda candidate, expected: _pit_si_snr_huber_per_sample(
            candidate,
            expected,
            alpha=si_snr_alpha,
            beta=huber_beta,
            delta=huber_delta,
        )
    else:
        raise ValueError(f"Unsupported counterfactual route quality loss: {quality_loss}")
    candidate_losses = torch.stack(
        [loss_fn(candidate, target) for candidate in candidate_outputs], dim=1
    )
    target_temperature = max(float(temperature), 1e-3)
    target_route = F.softmax(-candidate_losses.detach() / target_temperature, dim=1)
    route_loss = -(
        target_route * route_weights.clamp_min(1e-8).log()
    ).sum(dim=1).mean()
    return route_loss, candidate_losses.detach(), target_route
