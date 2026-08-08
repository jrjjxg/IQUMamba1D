"""Reusable mixture-consistency projection layers for multi-source IQ separation.

This module implements the differentiable mixture-consistency projection
described by Scott Wisdom et al., "Differentiable Consistency Constraints for
Improved Deep Speech Enhancement" (arXiv:1811.08521), adapted from complex
STFT bins to complex IQ samples represented as `(I, Q)` channel pairs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WeightedMixtureConsistencyProjection1D(nn.Module):
    """Project separated IQ sources onto the mixture-consistent affine set.

    Given source estimates ``s_hat_k`` and the original mixture ``x``, the
    projection enforces exact consistency:

        s_hat_k' = s_hat_k + w_k * (x - sum_j s_hat_j),

    where ``w_k`` are non-negative weights that sum to one for each time step.

    - ``weight_mode="uniform"`` recovers the standard orthogonal projection
      (Wisdom et al., Eq. 7) onto the affine set ``sum_k s_hat_k' = x``.
    - ``weight_mode="energy"`` applies a weighted projection (Eq. 9) using the
      instantaneous squared IQ magnitude of each source estimate as a proxy for
      confidence/uncertainty, mirroring the paper's magnitude-squared weighting.
    """

    def __init__(
        self,
        num_sources: int,
        weight_mode: str = "energy",
        weight_power: float = 1.0,
        min_weight: float = 1e-3,
        eps: float = 1e-8,
        detach_weights: bool = False,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if weight_mode not in {"energy", "uniform"}:
            raise ValueError(
                f"Unsupported weight_mode='{weight_mode}'. Use 'energy' or 'uniform'."
            )
        if weight_power <= 0:
            raise ValueError(f"weight_power must be > 0, got {weight_power}")
        if min_weight < 0:
            raise ValueError(f"min_weight must be >= 0, got {min_weight}")

        self.num_sources = int(num_sources)
        self.weight_mode = weight_mode
        self.weight_power = float(weight_power)
        self.min_weight = float(min_weight)
        self.eps = float(eps)
        self.detach_weights = bool(detach_weights)

    def _resize_mixture(self, mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(
            mixture,
            size=target_length,
            mode="linear",
            align_corners=False,
        )

    def _compute_weights(self, est_sources: torch.Tensor) -> torch.Tensor:
        # est_sources: (B, K, 2, L)
        b, k, _, l = est_sources.shape
        if self.weight_mode == "uniform":
            return est_sources.new_full((b, k, l), 1.0 / k)

        # For IQ signals, sum(I^2 + Q^2) is the instantaneous complex magnitude
        # squared. Using it as the weighting score follows Wisdom et al.'s
        # magnitude-squared weighted mixture projection in the complex domain.
        energy = est_sources.pow(2).sum(dim=2).clamp_min(self.eps)  # (B, K, L)
        scores = energy.pow(self.weight_power)
        if self.detach_weights:
            scores = scores.detach()
        if self.min_weight > 0:
            scores = scores + self.min_weight
        denom = scores.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return scores / denom

    def forward(self, est_sources: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        """Apply exact mixture-consistency projection.

        Args:
            est_sources: (B, 2K, L)
            mixture:     (B, 2, L_mix)
        Returns:
            projected_sources: (B, 2K, L)
        """
        if est_sources.dim() != 3:
            raise ValueError(
                f"est_sources must have shape (B, 2K, L), got {tuple(est_sources.shape)}"
            )
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}"
            )
        if est_sources.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} output channels, got {est_sources.size(1)}"
            )

        target_length = est_sources.size(-1)
        mixture_resized = self._resize_mixture(mixture, target_length)

        b = est_sources.size(0)
        est = est_sources.reshape(b, self.num_sources, 2, target_length)
        residual = mixture_resized - est.sum(dim=1)  # (B, 2, L)
        weights = self._compute_weights(est)  # (B, K, L)
        est_projected = est + weights.unsqueeze(2) * residual.unsqueeze(1)
        return est_projected.reshape(b, 2 * self.num_sources, target_length)
