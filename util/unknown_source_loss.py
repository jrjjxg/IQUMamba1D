"""Mask-aware PIT objectives and metrics for the independent unknown-source mode."""

from __future__ import annotations

from itertools import permutations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_huber(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    diff = predictions.unsqueeze(2) - targets.unsqueeze(1)
    return F.smooth_l1_loss(
        diff,
        torch.zeros_like(diff),
        beta=float(beta),
        reduction="none",
    ).mean(dim=(-2, -1))


class UnknownSourcePITLoss(nn.Module):
    """Injective PIT over valid targets plus existence and inactive-slot penalties."""

    def __init__(
        self,
        max_sources: int = 3,
        huber_beta: float = 0.5,
        separation_weight: float = 1.0,
        existence_weight: float = 0.5,
        null_energy_weight: float = 0.1,
        mixture_weight: float = 0.05,
        count_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_sources = int(max_sources)
        self.huber_beta = float(huber_beta)
        self.separation_weight = float(separation_weight)
        self.existence_weight = float(existence_weight)
        self.null_energy_weight = float(null_energy_weight)
        self.mixture_weight = float(mixture_weight)
        self.count_weight = float(count_weight)

    def _best_assignments(
        self,
        pairwise: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, max_sources, _ = pairwise.shape
        target_to_prediction = torch.full(
            (batch, max_sources), -1, dtype=torch.long, device=pairwise.device
        )
        matched_prediction_mask = torch.zeros(
            batch, max_sources, dtype=torch.bool, device=pairwise.device
        )
        sample_losses: List[torch.Tensor] = []

        for batch_index in range(batch):
            valid_targets = torch.nonzero(valid_mask[batch_index] > 0.5, as_tuple=False).flatten()
            source_count = int(valid_targets.numel())
            if source_count < 1 or source_count > max_sources:
                raise ValueError(
                    f"Each sample must contain 1..{max_sources} valid sources, got {source_count}"
                )
            assignments = list(permutations(range(max_sources), source_count))
            scores = []
            for assignment in assignments:
                score = pairwise[batch_index, list(assignment), valid_targets].mean()
                scores.append(score)
            score_tensor = torch.stack(scores)
            best_index = int(score_tensor.detach().argmin().item())
            best_assignment = assignments[best_index]
            sample_losses.append(score_tensor[best_index])
            for target_index, prediction_index in zip(valid_targets.tolist(), best_assignment):
                target_to_prediction[batch_index, target_index] = prediction_index
                matched_prediction_mask[batch_index, prediction_index] = True

        return (
            torch.stack(sample_losses).mean(),
            target_to_prediction,
            matched_prediction_mask,
        )

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
        mixture: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        separation = model_output["separation"]
        existence_logits = model_output["existence_logits"]
        batch, channels, length = separation.shape
        if channels != 2 * self.max_sources:
            raise ValueError(f"Expected {2 * self.max_sources} output channels, got {channels}")
        if targets.shape != separation.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {tuple(separation.shape)} vs {tuple(targets.shape)}"
            )
        if valid_mask.shape != (batch, self.max_sources):
            raise ValueError(
                f"Expected valid_mask {(batch, self.max_sources)}, got {tuple(valid_mask.shape)}"
            )
        if existence_logits.shape != valid_mask.shape:
            raise ValueError(
                f"Existence shape mismatch: {tuple(existence_logits.shape)} vs {tuple(valid_mask.shape)}"
            )
        if mixture.shape != (batch, 2, length):
            raise ValueError(
                f"Expected mixture {(batch, 2, length)}, got {tuple(mixture.shape)}"
            )

        predictions = separation.reshape(batch, self.max_sources, 2, length)
        target_sources = targets.reshape(batch, self.max_sources, 2, length)
        pairwise = _pairwise_huber(predictions, target_sources, beta=self.huber_beta)
        separation_loss, assignments, matched_mask = self._best_assignments(pairwise, valid_mask)

        existence_targets = matched_mask.to(dtype=existence_logits.dtype)
        existence_loss = F.binary_cross_entropy_with_logits(
            existence_logits, existence_targets, reduction="mean"
        )

        null_mask = (~matched_mask).to(dtype=predictions.dtype).view(batch, self.max_sources, 1, 1)
        null_denominator = null_mask.sum().clamp_min(1.0) * predictions.size(2) * predictions.size(3)
        null_energy_loss = (predictions.pow(2) * null_mask).sum() / null_denominator

        predicted_clean_mix = predictions.sum(dim=1)
        target_clean_mix = target_sources.sum(dim=1)
        predicted_residual = mixture - predicted_clean_mix
        target_residual = mixture - target_clean_mix
        mixture_loss = F.smooth_l1_loss(
            predicted_residual,
            target_residual,
            beta=self.huber_beta,
            reduction="mean",
        )

        predicted_count = torch.sigmoid(existence_logits).sum(dim=1)
        target_count = valid_mask.sum(dim=1)
        count_loss = F.smooth_l1_loss(
            predicted_count,
            target_count,
            beta=0.5,
            reduction="mean",
        ) / float(self.max_sources)

        total = (
            self.separation_weight * separation_loss
            + self.existence_weight * existence_loss
            + self.null_energy_weight * null_energy_loss
            + self.mixture_weight * mixture_loss
            + self.count_weight * count_loss
        )
        return {
            "loss": total,
            "separation_loss": separation_loss,
            "existence_loss": existence_loss,
            "null_energy_loss": null_energy_loss,
            "mixture_loss": mixture_loss,
            "count_loss": count_loss,
            "assignments": assignments,
            "matched_prediction_mask": matched_mask,
        }


def aligned_valid_metrics(
    separation: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    assignments: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Report MSE, absolute complex correlation and complex SI-SNR after PIT."""
    batch, channels, length = separation.shape
    max_sources = channels // 2
    predictions = separation.reshape(batch, max_sources, 2, length)
    target_sources = targets.reshape(batch, max_sources, 2, length)

    mse_values = []
    corr_values = []
    si_snr_values = []
    for batch_index in range(batch):
        valid_targets = torch.nonzero(valid_mask[batch_index] > 0.5, as_tuple=False).flatten()
        for target_index in valid_targets.tolist():
            prediction_index = int(assignments[batch_index, target_index].item())
            pred = predictions[batch_index, prediction_index]
            target = target_sources[batch_index, target_index]
            mse_values.append(torch.mean((pred - target) ** 2))

            pred_complex = torch.complex(pred[0], pred[1])
            target_complex = torch.complex(target[0], target[1])
            pred_zero = pred_complex - pred_complex.mean()
            target_zero = target_complex - target_complex.mean()
            corr = torch.sum(pred_zero * torch.conj(target_zero)).abs()
            corr = corr / (
                torch.sqrt(torch.sum(pred_zero.abs() ** 2))
                * torch.sqrt(torch.sum(target_zero.abs() ** 2))
                + eps
            )
            corr_values.append(corr)

            alpha = torch.sum(pred_complex * torch.conj(target_complex)) / (
                torch.sum(target_complex.abs() ** 2) + eps
            )
            projected = alpha * target_complex
            residual = pred_complex - projected
            si_snr = 10.0 * torch.log10(
                (torch.sum(projected.abs() ** 2) + eps)
                / (torch.sum(residual.abs() ** 2) + eps)
            )
            si_snr_values.append(si_snr)

    if not mse_values:
        return {"mse": float("nan"), "correlation": float("nan"), "si_snr": float("nan")}
    return {
        "mse": float(torch.stack(mse_values).mean().item()),
        "correlation": float(torch.stack(corr_values).mean().item()),
        "si_snr": float(torch.stack(si_snr_values).mean().item()),
    }
