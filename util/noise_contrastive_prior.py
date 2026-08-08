"""Training-only residual-noise contrastive prior for IQ separation.

The separator still predicts only the K sources.  The omitted component is
formed as a residual, and local source patches are trained to be closer to
their clean targets than to that residual.  The prior is deliberately
modulation agnostic and can be used with either two or three sources.
"""

from __future__ import annotations

from itertools import permutations
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.nn import functional as F


_PERMUTATION_CACHE: Dict[int, Tuple[Tuple[int, ...], ...]] = {}


def _get_permutations(num_sources: int) -> Tuple[Tuple[int, ...], ...]:
    num_sources = int(num_sources)
    if num_sources not in _PERMUTATION_CACHE:
        _PERMUTATION_CACHE[num_sources] = tuple(permutations(range(num_sources)))
    return _PERMUTATION_CACHE[num_sources]


def _validate_iq_tensor(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value) or value.ndim != 3:
        raise ValueError(f"{name} must have shape [B, 2*K, L], got {getattr(value, 'shape', None)}")
    if value.size(1) % 2 != 0:
        raise ValueError(f"{name} must have an even number of IQ channels, got {value.size(1)}")


def align_sources_for_prior(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample PIT-aligned predictions and source indices.

    The permutation is selected from detached pairwise MSE scores.  Gathering
    the original prediction afterward preserves gradients while preventing the
    discrete PIT decision from becoming a second gradient path.
    """
    _validate_iq_tensor("prediction", prediction)
    _validate_iq_tensor("target", target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {tuple(prediction.shape)} "
            f"and {tuple(target.shape)}"
        )

    batch, channels, length = prediction.shape
    num_sources = channels // 2
    pred_sources = prediction.reshape(batch, num_sources, 2, length)
    target_sources = target.reshape(batch, num_sources, 2, length)

    with torch.no_grad():
        pairwise_mse = (
            pred_sources.detach()[:, :, None] - target_sources.detach()[:, None, :]
        ).square().mean(dim=(3, 4))
        permutation_scores = []
        for permutation in _get_permutations(num_sources):
            permutation_scores.append(
                torch.stack(
                    [pairwise_mse[:, pred_idx, target_idx] for target_idx, pred_idx in enumerate(permutation)],
                    dim=1,
                ).mean(dim=1)
            )
        best_permutation = torch.stack(permutation_scores, dim=1).argmin(dim=1)
        permutation_table = torch.tensor(
            _get_permutations(num_sources), device=prediction.device, dtype=torch.long
        )
        selected = permutation_table[best_permutation]

    gather_index = selected[:, :, None, None].expand(-1, -1, 2, length)
    aligned = pred_sources.gather(dim=1, index=gather_index)
    return aligned, selected


def _patchify(value: torch.Tensor, patch_size: int, patch_stride: int) -> torch.Tensor:
    """Convert [N, 2, L] into [N, P, 2, patch_size], padding only at the tail."""
    if value.ndim != 3 or value.size(1) != 2:
        raise ValueError(f"patchify expects [N, 2, L], got {tuple(value.shape)}")
    patch_size = int(patch_size)
    patch_stride = int(patch_stride)
    if patch_size <= 0 or patch_stride <= 0:
        raise ValueError("patch_size and patch_stride must be positive")
    length = value.size(-1)
    if length <= 0:
        raise ValueError("the sequence length must be positive")
    if length < patch_size:
        value = F.pad(value, (0, patch_size - length), mode="replicate")
    patches = value.unfold(dimension=-1, size=patch_size, step=patch_stride)
    return patches.permute(0, 2, 1, 3).contiguous()


def _fixed_patch_embedding(patches: torch.Tensor, eps: float) -> torch.Tensor:
    """Build a compact deterministic IQ+local-difference patch embedding."""
    if patches.ndim < 4 or patches.size(-2) != 2:
        raise ValueError(f"expected [..., P, 2, S] patches, got {tuple(patches.shape)}")
    scale = patches.square().mean(dim=(-2, -1), keepdim=True).add(eps).sqrt()
    normalized = patches / scale
    delta = F.pad(normalized[..., 1:] - normalized[..., :-1], (1, 0))
    embedding = torch.cat([normalized, delta], dim=-2).flatten(start_dim=-2)
    return F.normalize(embedding, dim=-1, eps=eps)


def _project_patches(
    patches: torch.Tensor,
    projector: Optional[Callable[[torch.Tensor], torch.Tensor]],
    eps: float,
) -> torch.Tensor:
    flat = patches.reshape(-1, 2, patches.size(-1))
    if projector is None:
        return _fixed_patch_embedding(patches, eps)
    projected = projector(flat)
    if projected.ndim != 2 or projected.size(0) != flat.size(0):
        raise ValueError(
            "noise prior projector must return [N, embedding], "
            f"got {tuple(projected.shape)} for input {tuple(flat.shape)}"
        )
    return F.normalize(projected.reshape(*patches.shape[:3], -1), dim=-1, eps=eps)


def residual_noise_contrastive_prior_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    *,
    patch_size: int = 64,
    patch_stride: int = 32,
    temperature: float = 0.2,
    residual_weight: float = 0.1,
    gate_floor: float = 0.1,
    projector: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a reliability-gated source-vs-residual patch contrastive loss.

    ``mixture - sum(target)`` is the available clean noise target in the
    synthetic supervised setting.  ``mixture - sum(prediction)`` is its model
    counterpart.  The residual reconstruction term is intentionally small; the
    contrastive term supplies the separation pressure that ordinary pointwise
    PIT losses lack at low SNR.
    """
    _validate_iq_tensor("prediction", prediction)
    _validate_iq_tensor("target", target)
    if mixture.ndim != 3 or mixture.size(1) != 2:
        raise ValueError(f"mixture must have shape [B, 2, L], got {tuple(mixture.shape)}")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.size(0) != mixture.size(0) or prediction.size(-1) != mixture.size(-1):
        raise ValueError("prediction/target and mixture must agree in batch and length")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if residual_weight < 0.0:
        raise ValueError(f"residual_weight must be non-negative, got {residual_weight}")

    aligned, selected = align_sources_for_prior(prediction, target)
    batch, num_sources, _, length = aligned.shape
    target_sources = target.reshape(batch, num_sources, 2, length)
    target_noise = mixture - target_sources.sum(dim=1)
    predicted_noise = mixture - aligned.sum(dim=1)

    source_pred_patches = _patchify(
        aligned.reshape(batch * num_sources, 2, length), patch_size, patch_stride
    ).reshape(batch, num_sources, -1, 2, int(patch_size))
    source_target_patches = _patchify(
        target_sources.reshape(batch * num_sources, 2, length), patch_size, patch_stride
    ).reshape(batch, num_sources, -1, 2, int(patch_size))
    noise_target_patches = _patchify(target_noise, patch_size, patch_stride)
    noise_pred_patches = _patchify(predicted_noise, patch_size, patch_stride)

    # The same residual patch is the negative for every source at that time.
    noise_target_patches = noise_target_patches[:, None].expand_as(source_target_patches)
    noise_pred_patches = noise_pred_patches[:, None].expand_as(source_pred_patches)

    pred_embedding = _project_patches(source_pred_patches, projector, eps)
    target_embedding = _project_patches(source_target_patches, projector, eps)
    noise_embedding = _project_patches(noise_target_patches, projector, eps)

    positive_similarity = (pred_embedding * target_embedding).sum(dim=-1)
    negative_similarity = (pred_embedding * noise_embedding).sum(dim=-1)
    logits = torch.stack([positive_similarity, negative_similarity], dim=-1)
    contrastive_per_sample = F.cross_entropy(
        logits.reshape(-1, 2) / float(temperature),
        torch.zeros(logits.reshape(-1, 2).size(0), dtype=torch.long, device=logits.device),
        reduction="none",
    ).reshape(batch, num_sources, -1).mean(dim=(1, 2))

    mixture_power = mixture.square().mean(dim=(1, 2)).add(eps)
    noise_power = target_noise.square().mean(dim=(1, 2))
    noise_ratio = noise_power / mixture_power
    gate_floor = max(float(gate_floor), eps)
    reliability_gate = (noise_ratio / (noise_ratio + gate_floor)).clamp(0.0, 1.0).detach()

    residual_error = F.smooth_l1_loss(
        predicted_noise,
        target_noise,
        beta=0.5,
        reduction="none",
    ).mean(dim=(1, 2)) / mixture_power
    total_per_sample = reliability_gate * (
        contrastive_per_sample + float(residual_weight) * residual_error
    )
    total = torch.nan_to_num(total_per_sample.mean(), nan=0.0, posinf=1e4, neginf=0.0)
    diagnostics = {
        "contrastive_loss": torch.nan_to_num(contrastive_per_sample.mean(), nan=0.0),
        "residual_loss": torch.nan_to_num(residual_error.mean(), nan=0.0),
        "reliability_gate": reliability_gate,
        "noise_ratio": torch.nan_to_num(noise_ratio, nan=0.0, posinf=1.0, neginf=0.0),
        "permutation": selected,
    }
    return total, diagnostics
