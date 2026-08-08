"""Confidence-adaptive Soft-PIT for low-SNR IQ source separation."""

import math
from typing import Optional

import torch
import torch.nn as nn

from util.loss import (
    _get_permutations,
    _huber_pair_per_item,
    _infer_num_sources,
    _rms_gain_loss_pair_per_item,
    _si_snr_pair_per_item,
    _split_iq_sources,
)


def normalized_softmin(losses: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Reduce `(P, B)` permutation losses without a permutation-count bias."""
    if losses.ndim != 2:
        raise ValueError(f"Expected permutation losses shaped (P, B), got {tuple(losses.shape)}")
    if temperature.ndim == 0:
        temperature = temperature.expand(losses.size(1))
    if temperature.shape != (losses.size(1),):
        raise ValueError(
            f"Expected one temperature per batch item ({losses.size(1)},), got {tuple(temperature.shape)}"
        )
    tau = temperature.to(device=losses.device, dtype=losses.dtype).clamp_min(1e-6)
    log_mean_exp = torch.logsumexp(-losses / tau.unsqueeze(0), dim=0) - math.log(losses.size(0))
    return -tau * log_mean_exp


def temperature_from_snr(
    snr: torch.Tensor,
    epoch_progress: float,
    temperature_min: float,
    temperature_max: float,
    snr_low_db: float,
    snr_high_db: float,
    anneal_power: float,
) -> torch.Tensor:
    """Map low SNR to softer assignments and anneal toward hard PIT."""
    if temperature_min <= 0 or temperature_max < temperature_min:
        raise ValueError("Require 0 < temperature_min <= temperature_max")
    if snr_high_db <= snr_low_db:
        raise ValueError("snr_high_db must be greater than snr_low_db")
    if anneal_power <= 0:
        raise ValueError("anneal_power must be positive")

    snr_weight = ((snr_high_db - snr.float()) / (snr_high_db - snr_low_db)).clamp(0.0, 1.0)
    base = temperature_min + (temperature_max - temperature_min) * snr_weight
    progress = min(max(float(epoch_progress), 0.0), 1.0)
    anneal = (1.0 - progress) ** float(anneal_power)
    return temperature_min + (base - temperature_min) * anneal


class ConfidenceAdaptiveSoftPITLoss(nn.Module):
    """SNR-aware probabilistic PIT with no inference-time model changes."""

    needs_snr = True

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        rms_lambda: float = 0.5,
        temperature_min: float = 0.05,
        temperature_max: float = 2.0,
        snr_low_db: float = -10.0,
        snr_high_db: float = 10.0,
        anneal_power: float = 2.0,
        total_epochs: int = 1,
        delta: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rms_lambda = float(rms_lambda)
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.snr_low_db = float(snr_low_db)
        self.snr_high_db = float(snr_high_db)
        self.anneal_power = float(anneal_power)
        self.total_epochs = max(1, int(total_epochs))
        self.delta = float(delta)
        self.eps = float(eps)
        self.current_epoch = 0

        # Validate scalar hyperparameters at construction time.
        temperature_from_snr(
            torch.zeros(1), 0.0, self.temperature_min, self.temperature_max,
            self.snr_low_db, self.snr_high_db, self.anneal_power,
        )

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        self.current_epoch = max(0, int(epoch))
        if total_epochs is not None:
            self.total_epochs = max(1, int(total_epochs))

    def _temperatures(self, batch_size: int, snr, device) -> torch.Tensor:
        if snr is None:
            midpoint = 0.5 * (self.snr_low_db + self.snr_high_db)
            snr_tensor = torch.full((batch_size,), midpoint, device=device)
        elif torch.is_tensor(snr):
            snr_tensor = snr.to(device=device, dtype=torch.float32).reshape(-1)
            if snr_tensor.numel() == 1:
                snr_tensor = snr_tensor.expand(batch_size)
            elif snr_tensor.numel() != batch_size:
                raise ValueError(f"Expected {batch_size} SNR values, got {snr_tensor.numel()}")
        else:
            snr_tensor = torch.full((batch_size,), float(snr), device=device)

        denominator = max(1, self.total_epochs - 1)
        progress = self.current_epoch / denominator
        return temperature_from_snr(
            snr_tensor,
            progress,
            self.temperature_min,
            self.temperature_max,
            self.snr_low_db,
            self.snr_high_db,
            self.anneal_power,
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, snr=None) -> torch.Tensor:
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[-1]
        num_sources = _infer_num_sources(outputs, targets)
        preds = _split_iq_sources(outputs, num_sources)
        tgts = _split_iq_sources(targets, num_sources)

        permutation_losses = []
        for permutation in _get_permutations(num_sources):
            total = 0.0
            for target_idx, pred_idx in enumerate(permutation):
                pred = preds[pred_idx]
                target = tgts[target_idx]
                total = total + (
                    -self.alpha * _si_snr_pair_per_item(pred, target, eps=self.eps)
                    + self.beta * _huber_pair_per_item(pred, target, delta=self.delta)
                    + self.rms_lambda * _rms_gain_loss_pair_per_item(pred, target, eps=self.eps)
                )
            permutation_losses.append(total / num_sources)

        losses = torch.stack(permutation_losses, dim=0)
        temperatures = self._temperatures(outputs.size(0), snr, outputs.device)
        return normalized_softmin(losses, temperatures).mean()
