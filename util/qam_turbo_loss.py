"""Auxiliary objectives for Stage 238 QAM turbo unfolding."""

from __future__ import annotations

import torch

from util.loss import pit_si_snr_huber_loss


def normalized_mixture_reconstruction_loss(
    mixture_residual: torch.Tensor,
    mixture: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    numerator = mixture_residual.float().square().mean(dim=(1, 2))
    denominator = mixture.float().square().mean(dim=(1, 2)).clamp_min(eps)
    return (numerator / denominator).mean()


def complex_source_independence_loss(
    outputs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if outputs.ndim != 3 or outputs.size(1) % 2 != 0:
        raise ValueError(f"Expected [B, 2K, L], got {tuple(outputs.shape)}")
    batch, channels, length = outputs.shape
    num_sources = channels // 2
    if num_sources < 2:
        return outputs.new_zeros(())
    sources = outputs.float().reshape(batch, num_sources, 2, length)
    z = torch.complex(sources[:, :, 0], sources[:, :, 1])
    z = z - z.mean(dim=-1, keepdim=True)
    power = z.abs().square().mean(dim=-1).clamp_min(eps)
    terms = []
    for left in range(num_sources):
        for right in range(left + 1, num_sources):
            correlation = (z[:, left] * torch.conj(z[:, right])).mean(dim=-1)
            terms.append(
                correlation.abs().square() / (power[:, left] * power[:, right])
            )
    return torch.stack(terms, dim=-1).mean()


def qam_turbo_auxiliary_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
    auxiliary: dict,
    mixture_weight: float = 0.15,
    qam_weight: float = 0.03,
    independence_weight: float = 0.02,
    intermediate_weight: float = 0.20,
    route_entropy_weight: float = 0.002,
    intermediate_alpha: float = 1.0,
    intermediate_beta: float = 0.5,
    intermediate_delta: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the Stage-238 joint objective excluding the caller's base loss."""
    zero = outputs.new_zeros(())
    mixture_term = zero
    qam_term = zero
    independence_term = zero
    intermediate_term = zero
    entropy_term = zero

    mixture_residual = auxiliary.get("mixture_residual")
    if mixture_residual is not None and mixture_weight > 0.0:
        mixture_term = normalized_mixture_reconstruction_loss(
            mixture_residual,
            mixture,
            eps=eps,
        )

    qam_distance = auxiliary.get("qam_expected_distance")
    if qam_distance is not None and qam_weight > 0.0:
        qam_term = qam_distance.float().mean()

    if independence_weight > 0.0:
        independence_term = complex_source_independence_loss(outputs, eps=eps)

    intermediates = auxiliary.get("intermediate_outputs")
    if intermediate_weight > 0.0 and isinstance(intermediates, (list, tuple)):
        # The final iterate is already supervised by the caller's base loss.
        supervised = list(intermediates[:-1])
        if supervised:
            terms = []
            weights = []
            for index, estimate in enumerate(supervised):
                weight = 1.0 / float(len(supervised) - index)
                terms.append(
                    pit_si_snr_huber_loss(
                        estimate,
                        targets,
                        alpha=intermediate_alpha,
                        beta=intermediate_beta,
                        delta=intermediate_delta,
                    )
                )
                weights.append(weight)
            weight_tensor = outputs.new_tensor(weights)
            intermediate_term = (
                torch.stack(terms) * weight_tensor
            ).sum() / weight_tensor.sum().clamp_min(eps)

    route_entropy = auxiliary.get("qam_route_entropy")
    if route_entropy is not None and route_entropy_weight > 0.0:
        entropy_term = route_entropy.float().mean()

    total = (
        float(mixture_weight) * mixture_term
        + float(qam_weight) * qam_term
        + float(independence_weight) * independence_term
        + float(intermediate_weight) * intermediate_term
        + float(route_entropy_weight) * entropy_term
    )
    return total, {
        "mixture": mixture_term.detach(),
        "qam": qam_term.detach(),
        "independence": independence_term.detach(),
        "intermediate": intermediate_term.detach(),
        "route_entropy": entropy_term.detach(),
    }


__all__ = [
    "normalized_mixture_reconstruction_loss",
    "complex_source_independence_loss",
    "qam_turbo_auxiliary_loss",
]
