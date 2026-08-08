"""Shared-permutation auxiliary supervision for multi-scale IQ separation."""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import torch
from torch.nn import functional as F


def _split_sources(x: torch.Tensor, num_sources: int) -> torch.Tensor:
    if x.ndim != 3 or x.size(1) != 2 * num_sources:
        raise ValueError(f"Expected (B,{2 * num_sources},L), got {tuple(x.shape)}")
    return x.reshape(x.size(0), num_sources, 2, x.size(-1))


def _pairwise_average_to_length(target: torch.Tensor, length: int) -> torch.Tensor:
    output = target
    while output.size(-1) > length and (output.size(-1) + 1) // 2 >= length:
        if output.size(-1) % 2:
            output = F.pad(output, (0, 1), mode="replicate")
        output = 0.5 * (output[..., 0::2] + output[..., 1::2])
    if output.size(-1) != length:
        output = F.adaptive_avg_pool1d(output, length)
    return output


def _best_final_permutation(final: torch.Tensor, target: torch.Tensor, num_sources: int) -> torch.Tensor:
    pred_sources = _split_sources(final.detach(), num_sources)
    target_sources = _split_sources(target.detach(), num_sources)
    scores = []
    permutation_bank = list(permutations(range(num_sources)))
    for permutation in permutation_bank:
        aligned = pred_sources[:, list(permutation)]
        scores.append(F.smooth_l1_loss(aligned, target_sources, reduction="none").mean(dim=(1, 2, 3)))
    best_index = torch.stack(scores, dim=0).argmin(dim=0)
    bank = torch.tensor(permutation_bank, device=final.device, dtype=torch.long)
    return bank.index_select(0, best_index)


def _align_with_permutation(prediction: torch.Tensor, permutation: torch.Tensor, num_sources: int) -> torch.Tensor:
    sources = _split_sources(prediction, num_sources)
    gather_index = permutation[:, :, None, None].expand(-1, -1, 2, sources.size(-1))
    return torch.gather(sources, dim=1, index=gather_index)


def shared_permutation_auxiliary_loss(
    outputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    weights: Sequence[float] | None = None,
    include_final: bool = True,
) -> torch.Tensor:
    """Apply one final-output PIT assignment to every decoder scale."""
    if not isinstance(outputs, (list, tuple)) or not outputs:
        raise ValueError("outputs must be a non-empty list of decoder predictions")
    final = outputs[0]
    if final.ndim != 3 or target.ndim != 3 or final.size(1) != target.size(1):
        raise ValueError(f"Invalid final/target shapes: {tuple(final.shape)}, {tuple(target.shape)}")
    if target.size(1) % 2:
        raise ValueError("IQ targets require an even channel count")
    num_sources = target.size(1) // 2
    permutation = _best_final_permutation(final, target, num_sources)
    if weights is None:
        weights = [1.0 / (2 ** index) for index in range(len(outputs))]
    if len(weights) != len(outputs):
        raise ValueError("weights must match the number of decoder outputs")

    total = final.new_zeros(())
    normalizer = 0.0
    start = 0 if include_final else 1
    for prediction, weight in zip(outputs[start:], weights[start:]):
        weight = float(weight)
        if weight <= 0:
            continue
        reduced_target = _pairwise_average_to_length(target, prediction.size(-1))
        aligned = _align_with_permutation(prediction, permutation, num_sources)
        target_sources = _split_sources(reduced_target, num_sources)
        total = total + weight * F.smooth_l1_loss(aligned, target_sources)
        normalizer += weight
    if normalizer == 0:
        return total
    return total / normalizer

