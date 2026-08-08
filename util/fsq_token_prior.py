"""Stage 296: FSQ token cross-entropy prior for separator training.

This module turns a frozen, pretrained FSQ tokenizer
(models/FSQTokenizer1D.py, trained by pretrain_fsq_tokenizer.py) into a
training-only loss for any waveform separator with [B, 2K, L] outputs:

    ce = fsq_token_ce_loss(tokenizer, separation, targets)

Mechanism: the clean target of each source is encoded by the frozen tokenizer
into discrete token indices (its "communication content" at symbol-ish rate);
the *predicted* source is passed through the same frozen encoder to a
continuous bounded latent, converted to per-level logits by (negative squared)
distance to the FSQ grid, and trained with cross entropy against the target
indices. Gradients flow through the frozen encoder into the separator output,
so the separator is pushed to produce waveforms whose discrete communication
content matches the target - a BER-aligned objective in the spirit of the RF
Transformer's token cross entropy (arXiv:2603.09201), but non-autoregressive
and with zero inference-time cost.

The wrapper in main.py (_wrap_fsq_token_ce) applies this only when gradients
are enabled, so validation/test losses remain identical to the base criterion
and stay comparable with earlier stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "load_frozen_tokenizer",
    "fsq_token_ce_loss",
    "fsq_token_accuracy",
]

_TOKENIZER_CACHE: Dict[str, torch.nn.Module] = {}


def load_frozen_tokenizer(path: str, device: Optional[torch.device] = None):
    """Load (once) and cache a frozen FSQ tokenizer checkpoint."""
    key = str(Path(path).expanduser().resolve())
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        from models.FSQTokenizer1D import load_fsq_tokenizer

        tokenizer = load_fsq_tokenizer(key, map_location="cpu")
        tokenizer.eval()
        for parameter in tokenizer.parameters():
            parameter.requires_grad_(False)
        _TOKENIZER_CACHE[key] = tokenizer
    if device is not None:
        tokenizer.to(device)
    return tokenizer


def _split_sources(x: torch.Tensor) -> torch.Tensor:
    """[B, 2K, L] with [I0,Q0,I1,Q1,...] channel layout -> [B*K, 2, L]."""
    if x.dim() != 3 or x.size(1) % 2 != 0:
        raise ValueError(f"expected [B, 2K, L] tensor, got {tuple(x.shape)}")
    batch, channels, length = x.shape
    num_sources = channels // 2
    return x.reshape(batch, num_sources, 2, length).reshape(
        batch * num_sources, 2, length
    )


def _per_dim_logits(
    tokenizer, bounded: torch.Tensor, dim: int, temperature: float
) -> torch.Tensor:
    """Distance logits of bounded latent against grid of one FSQ dim.

    bounded: [N, D, T] continuous bounded latent. Returns [N, T, levels[dim]].
    """
    positions = tokenizer.fsq.positions(
        dim, device=bounded.device, dtype=bounded.dtype
    )
    diff = bounded[:, dim, :].unsqueeze(-1) - positions.view(1, 1, -1)
    return -(diff.pow(2)) / max(float(temperature), 1e-6)


def fsq_token_ce_loss(
    tokenizer,
    separation: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Cross entropy between predicted-source tokens and clean-target tokens.

    Args:
        tokenizer: frozen FSQTokenizer1D.
        separation: separator output [B, 2K, L] (grad flows from here).
        targets: clean sources [B, 2K, L], same channel layout.
        temperature: distance-to-logit temperature in units of the (unit)
            level spacing; smaller = sharper classification.
    Returns:
        scalar loss (mean CE over sources, latent dims, and token frames).
    """
    if separation.shape != targets.shape:
        raise ValueError(
            f"separation {tuple(separation.shape)} and targets "
            f"{tuple(targets.shape)} must match"
        )
    predicted_sources = _split_sources(separation.float())
    target_sources = _split_sources(targets.float())

    with torch.no_grad():
        target_indices = tokenizer.encode_indices(target_sources)  # [N, D, T]
    predicted_bounded = tokenizer.encode_bounded(predicted_sources)  # [N, D, T]

    losses = []
    for dim in range(tokenizer.fsq.num_dims):
        logits = _per_dim_logits(tokenizer, predicted_bounded, dim, temperature)
        losses.append(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_indices[:, dim, :].reshape(-1),
            )
        )
    return torch.stack(losses).mean()


@torch.no_grad()
def fsq_token_accuracy(
    tokenizer, separation: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Fraction of (source, dim, frame) tokens predicted exactly right.

    Diagnostic only (not differentiable); useful for logging how much of the
    discrete communication content of each source the separator preserves.
    """
    predicted_indices = tokenizer.encode_indices(_split_sources(separation.float()))
    target_indices = tokenizer.encode_indices(_split_sources(targets.float()))
    return (predicted_indices == target_indices).float().mean()
