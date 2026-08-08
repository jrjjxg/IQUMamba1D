"""Minimal, reproducible training loop for single-SOI RF Challenge models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import random
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from .metrics import official_case_ber_score_db, official_case_mse_score_db
from .models import (
    checkpoint_payload,
    extract_single_soi_output,
    save_checkpoint,
)
from .protocol import SINR_DB_VALUES, demodulate_soi


@dataclass
class TrainOptions:
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    num_workers: int = 0
    amp: bool = True
    gradient_clip_norm: float | None = 5.0
    save_every_epochs: int = 1
    loss: str = "mse"
    huber_delta: float = 1.0
    optimizer: str = "adam"
    lr_factor: float = 0.5
    lr_patience: int | None = None
    minimum_learning_rate: float = 0.0
    early_stopping_patience: int | None = None

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be positive")
        if self.loss not in {"mse", "huber"}:
            raise ValueError("loss must be 'mse' or 'huber'")
        if self.optimizer not in {"adam", "adamw"}:
            raise ValueError("optimizer must be 'adam' or 'adamw'")
        if not 0.0 < self.lr_factor < 1.0:
            raise ValueError("lr_factor must be in (0, 1)")
        if self.lr_patience is not None and self.lr_patience < 0:
            raise ValueError("lr_patience cannot be negative")
        if self.minimum_learning_rate < 0.0:
            raise ValueError("minimum_learning_rate cannot be negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    official_validation_mse_score_db: list[float | None] = field(default_factory=list)
    official_validation_mse_db_by_sinr: list[list[float] | None] = field(
        default_factory=list
    )
    validation_complex_mse: list[float | None] = field(default_factory=list)
    validation_ber: list[float | None] = field(default_factory=list)
    official_validation_ber_score_db: list[float | None] = field(default_factory=list)
    official_validation_ber_by_sinr: list[list[float] | None] = field(
        default_factory=list
    )
    # Retained only so old checkpoints/history files remain resumable. New RF
    # Challenge runs no longer compute or append SI-SNR metrics.
    validation_si_snr_paper_db: list[float] = field(default_factory=list)
    validation_si_snr_repo_db: list[float] = field(default_factory=list)
    validation_si_snr_complex_db: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)
    effective_dilations: list[list[float]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainHistory":
        """Restore history while remaining compatible with older checkpoints."""

        values = dict(payload or {})
        train_loss = list(values.get("train_loss", []))

        def padded_new_metric(name: str) -> list[Any]:
            series = list(values.get(name, []))
            if len(series) < len(train_loss):
                series = [None] * (len(train_loss) - len(series)) + series
            return series

        return cls(
            train_loss=train_loss,
            validation_loss=list(values.get("validation_loss", [])),
            official_validation_mse_score_db=list(
                values.get("official_validation_mse_score_db", [])
            ),
            official_validation_mse_db_by_sinr=list(
                values.get("official_validation_mse_db_by_sinr", [])
            ),
            validation_complex_mse=padded_new_metric("validation_complex_mse"),
            validation_ber=padded_new_metric("validation_ber"),
            official_validation_ber_score_db=padded_new_metric(
                "official_validation_ber_score_db"
            ),
            official_validation_ber_by_sinr=padded_new_metric(
                "official_validation_ber_by_sinr"
            ),
            validation_si_snr_paper_db=list(
                values.get("validation_si_snr_paper_db", [])
            ),
            validation_si_snr_repo_db=list(
                values.get("validation_si_snr_repo_db", [])
            ),
            validation_si_snr_complex_db=list(
                values.get("validation_si_snr_complex_db", [])
            ),
            epoch_seconds=list(values.get("epoch_seconds", [])),
            effective_dilations=list(values.get("effective_dilations", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "official_validation_mse_score_db": self.official_validation_mse_score_db,
            "official_validation_mse_db_by_sinr": self.official_validation_mse_db_by_sinr,
            "validation_complex_mse": self.validation_complex_mse,
            "validation_ber": self.validation_ber,
            "official_validation_ber_score_db": self.official_validation_ber_score_db,
            "official_validation_ber_by_sinr": self.official_validation_ber_by_sinr,
            "validation_si_snr_paper_db": self.validation_si_snr_paper_db,
            "validation_si_snr_repo_db": self.validation_si_snr_repo_db,
            "validation_si_snr_complex_db": self.validation_si_snr_complex_db,
            "epoch_seconds": self.epoch_seconds,
            "effective_dilations": self.effective_dilations,
        }


@dataclass(frozen=True)
class ValidationMetrics:
    """Training loss plus the public challenge's validation MSE/BER metrics."""

    loss: float
    complex_mse: float
    official_mse_score_db: float | None
    mse_db_by_sinr: list[float] | None
    ber: float | None
    official_ber_score_db: float | None
    ber_by_sinr: list[float] | None


def _loss_function(
    prediction: torch.Tensor,
    target: torch.Tensor,
    options: TrainOptions,
) -> torch.Tensor:
    # The official WaveNet baseline trains raw I/Q MSE via nn.MSELoss. It is
    # kept as the default because the public evaluator is amplitude/phase aware.
    if options.loss == "mse":
        return functional.mse_loss(prediction, target)
    return functional.huber_loss(prediction, target, delta=options.huber_delta)


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Access optional model-specific hooks through DDP/DataParallel wrappers."""

    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def _project_and_measure_dilations(model: torch.nn.Module) -> list[float]:
    """Apply the KU-TII model's bounded-dilation constraint after an update."""

    base_model = _unwrap_model(model)
    project = getattr(base_model, "project_learnable_dilations_", None)
    if callable(project):
        project()
    effective = getattr(base_model, "effective_dilations", None)
    if not callable(effective):
        return []
    values = effective().detach().float().cpu().tolist()
    return [float(value) for value in values]


def _official_validation_score(
    nominal_sinr_batches: list[np.ndarray],
    per_frame_mse_batches: list[np.ndarray],
) -> tuple[float | None, list[float] | None]:
    """Apply the official truncated dB aggregation when all 11 bins are present."""

    if not nominal_sinr_batches or not per_frame_mse_batches:
        return None, None
    nominal_sinr = np.concatenate(nominal_sinr_batches).astype(np.float32, copy=False)
    per_frame_mse = np.concatenate(per_frame_mse_batches).astype(np.float32, copy=False)
    if nominal_sinr.shape != per_frame_mse.shape:
        return None, None
    known_labels = np.any(
        np.isclose(nominal_sinr[:, None], SINR_DB_VALUES[None, :], atol=1e-4),
        axis=1,
    )
    if not np.all(known_labels):
        return None, None
    grouped_linear_mse = []
    for sinr in SINR_DB_VALUES:
        indices = np.flatnonzero(np.isclose(nominal_sinr, sinr, atol=1e-4))
        if indices.size == 0:
            return None, None
        grouped_linear_mse.append(np.mean(per_frame_mse[indices], dtype=np.float32))
    with np.errstate(divide="ignore"):
        mse_db = (10.0 * np.log10(np.asarray(grouped_linear_mse, dtype=np.float32))).astype(
            np.float32
        )
    return official_case_mse_score_db(mse_db), [float(value) for value in mse_db]


def _official_validation_ber_score(
    nominal_sinr_batches: list[np.ndarray],
    per_frame_ber_batches: list[np.ndarray],
) -> tuple[float | None, list[float] | None]:
    """Apply the official BER threshold score when all 11 bins are present."""

    if not nominal_sinr_batches or not per_frame_ber_batches:
        return None, None
    nominal_sinr = np.concatenate(nominal_sinr_batches).astype(np.float32, copy=False)
    per_frame_ber = np.concatenate(per_frame_ber_batches).astype(np.float32, copy=False)
    if nominal_sinr.shape != per_frame_ber.shape:
        return None, None
    known_labels = np.any(
        np.isclose(nominal_sinr[:, None], SINR_DB_VALUES[None, :], atol=1e-4),
        axis=1,
    )
    if not np.all(known_labels):
        return None, None
    grouped_ber = []
    for sinr in SINR_DB_VALUES:
        indices = np.flatnonzero(np.isclose(nominal_sinr, sinr, atol=1e-4))
        if indices.size == 0:
            return None, None
        grouped_ber.append(np.mean(per_frame_ber[indices], dtype=np.float32))
    ber_array = np.asarray(grouped_ber, dtype=np.float32)
    return (
        official_case_ber_score_db(SINR_DB_VALUES, ber_array),
        [float(value) for value in ber_array],
    )


@torch.no_grad()
def validate_single_soi_model_with_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    options: TrainOptions,
    soi_type: str | None = None,
) -> ValidationMetrics:
    """Compute validation loss and official MSE/BER for a fixed 11-bin set.

    Official scores require nominal SINR labels in batch element 3 and all
    eleven challenge levels. BER additionally requires transmitted bits in
    element 4 and ``soi_type`` so the protocol-matched hard demodulator can be
    applied. Complex MSE sums I/Q squared error before averaging time, matching
    the public evaluator rather than PyTorch's channel-averaged training loss.
    """

    model.eval()
    total_loss = 0.0
    total_items = 0
    total_complex_mse = 0.0
    nominal_sinr_batches: list[np.ndarray] = []
    per_frame_mse_batches: list[np.ndarray] = []
    ber_sinr_batches: list[np.ndarray] = []
    per_frame_ber_batches: list[np.ndarray] = []
    for batch in loader:
        mixture, target = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        with _autocast_context(device, options.amp):
            prediction = extract_single_soi_output(model(mixture))
            loss = _loss_function(prediction, target, options)
        total_loss += float(loss.detach().float()) * mixture.shape[0]
        total_items += int(mixture.shape[0])
        metric_prediction = prediction.detach().float()
        metric_target = target.detach().float()
        complex_error = metric_prediction - metric_target
        per_frame_mse = complex_error.square().sum(dim=1).mean(dim=-1)
        total_complex_mse += float(per_frame_mse.sum().detach().cpu())
        valid_nominal_sinr = None
        if len(batch) >= 3:
            nominal_sinr = batch[2]
            if isinstance(nominal_sinr, torch.Tensor) and nominal_sinr.ndim == 1:
                if nominal_sinr.shape[0] == mixture.shape[0]:
                    valid_nominal_sinr = (
                        nominal_sinr.detach().float().cpu().numpy().copy()
                    )
                    nominal_sinr_batches.append(valid_nominal_sinr)
                    per_frame_mse_batches.append(
                        per_frame_mse.detach().float().cpu().numpy().copy()
                    )
        if soi_type is not None and len(batch) >= 4:
            true_bits = batch[3]
            if not isinstance(true_bits, torch.Tensor) or true_bits.ndim != 2:
                raise ValueError("Validation bits must be a two-dimensional tensor")
            prediction_iq = metric_prediction.cpu().numpy()
            prediction_complex = np.ascontiguousarray(
                prediction_iq[:, 0] + 1j * prediction_iq[:, 1],
                dtype=np.complex64,
            )
            recovered_bits, _ = demodulate_soi(soi_type, prediction_complex)
            expected_bits = true_bits.detach().cpu().numpy().astype(np.uint8, copy=False)
            if recovered_bits.shape != expected_bits.shape:
                raise ValueError(
                    "Validation demodulated-bit shape mismatch: "
                    f"estimated={recovered_bits.shape}, target={expected_bits.shape}"
                )
            per_frame_ber = np.mean(
                (recovered_bits != expected_bits).astype(np.float32),
                axis=1,
                dtype=np.float32,
            )
            per_frame_ber_batches.append(per_frame_ber)
            if valid_nominal_sinr is not None:
                ber_sinr_batches.append(valid_nominal_sinr)
    if total_items == 0:
        raise RuntimeError("Validation loader produced no samples")
    official_mse_score_db, mse_db_by_sinr = _official_validation_score(
        nominal_sinr_batches,
        per_frame_mse_batches,
    )
    official_ber_score_db, ber_by_sinr = _official_validation_ber_score(
        ber_sinr_batches,
        per_frame_ber_batches,
    )
    validation_ber = (
        None
        if not per_frame_ber_batches
        else float(np.mean(np.concatenate(per_frame_ber_batches), dtype=np.float32))
    )
    return ValidationMetrics(
        loss=total_loss / total_items,
        complex_mse=total_complex_mse / total_items,
        official_mse_score_db=official_mse_score_db,
        mse_db_by_sinr=mse_db_by_sinr,
        ber=validation_ber,
        official_ber_score_db=official_ber_score_db,
        ber_by_sinr=ber_by_sinr,
    )


def _load_training_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load an IQUMamba training checkpoint trusted by the local workflow."""

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Not a resumable IQUMamba training checkpoint: {path}")
    return payload


def _validate_resume_metadata(
    checkpoint: dict[str, Any],
    expected: dict[str, Any] | None,
) -> None:
    saved = checkpoint.get("metadata")
    if not isinstance(saved, dict) or not expected:
        return
    for key in ("soi_type", "interference_type", "frame_length", "model_stage"):
        if key in saved and key in expected and saved[key] != expected[key]:
            raise ValueError(
                f"Resume checkpoint {key}={saved[key]!r} does not match "
                f"current run {key}={expected[key]!r}"
            )


def _capture_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(checkpoint: dict[str, Any]) -> None:
    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["torch_cuda_random_state_all"]]
        )


@torch.no_grad()
def validate_single_soi_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    options: TrainOptions,
) -> float:
    """Return raw validation loss, retaining the original public API."""

    return validate_single_soi_model_with_metrics(model, loader, device, options).loss


def train_single_soi_model(
    model: torch.nn.Module,
    train_dataset,
    validation_dataset,
    device: torch.device,
    output_dir: str | Path,
    options: TrainOptions,
    metadata: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
    resume_checkpoint: str | Path | None = None,
) -> TrainHistory:
    """Train one case with IQUMamba-style artifacts and resumable state."""

    options.validate()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    log = logger or logging.getLogger(__name__)
    def make_loader(dataset, shuffle: bool) -> DataLoader:
        worker_count = (
            0 if getattr(dataset, "force_single_process_loader", False) else options.num_workers
        )
        common = {
            "num_workers": worker_count,
            "pin_memory": device.type == "cuda",
            "persistent_workers": worker_count > 0,
        }
        if getattr(dataset, "prebatched", False):
            return DataLoader(dataset, batch_size=None, **common)
        return DataLoader(
            dataset,
            batch_size=options.batch_size,
            shuffle=shuffle,
            **common,
        )

    train_loader = make_loader(train_dataset, shuffle=True)
    validation_loader = make_loader(validation_dataset, shuffle=False)
    optimizer_class = torch.optim.Adam if options.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(
        model.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    scheduler = None
    if options.lr_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=options.lr_factor,
            patience=options.lr_patience,
            min_lr=options.minimum_learning_rate,
        )
    scaler_enabled = options.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        # Keep compatibility with the older PyTorch releases used by some
        # IQUMamba environments without emitting a warning on current builds.
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    history = TrainHistory()
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    start_epoch = 0

    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        restored = _load_training_checkpoint(resume_path, device)
        _validate_resume_metadata(restored, metadata)
        model.load_state_dict(restored["model_state_dict"], strict=True)
        if "optimizer_state_dict" in restored:
            optimizer.load_state_dict(restored["optimizer_state_dict"])
        if scheduler is not None and restored.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(restored["scheduler_state_dict"])
        if "scaler_state_dict" in restored:
            scaler.load_state_dict(restored["scaler_state_dict"])
        start_epoch = int(restored.get("epoch", 0))
        best_validation_loss = float(
            restored.get("best_validation_loss", float("inf"))
        )
        epochs_without_improvement = int(
            restored.get("epochs_without_improvement", 0)
        )
        history = TrainHistory.from_dict(restored.get("history"))
        _restore_random_state(restored)
        log.info(
            "Resuming training from %s at completed epoch %d/%d",
            resume_path,
            start_epoch,
            options.epochs,
        )

    if start_epoch >= options.epochs:
        log.info(
            "Training already complete at epoch %d (target=%d); skipping to test",
            start_epoch,
            options.epochs,
        )
        return history

    weights_dir = directory / "weights"
    checkpoint_dir = directory / "checkpoint"
    best_weights_path = weights_dir / "best_model_weights.pth"
    best_training_path = weights_dir / "best_training_checkpoint.pth"
    latest_training_path = weights_dir / "latest_training_checkpoint.pth"
    checkpoint_best_weights_path = checkpoint_dir / "best_model_weights.pth"
    history_path = directory / "training_history.json"

    for epoch in range(start_epoch, options.epochs):
        started = perf_counter()
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        if hasattr(validation_dataset, "set_epoch"):
            # Keep model selection deterministic across epochs.
            validation_dataset.set_epoch(0)
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            mixture = batch[0].to(device, non_blocking=True)
            target = batch[1].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, options.amp):
                prediction = extract_single_soi_output(model(mixture))
                loss = _loss_function(prediction, target, options)
            scaler.scale(loss).backward()
            if options.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            _project_and_measure_dilations(model)
            total_loss += float(loss.detach().float()) * mixture.shape[0]
            total_items += int(mixture.shape[0])

        if total_items == 0:
            raise RuntimeError("Training loader produced no samples")
        train_loss = total_loss / total_items
        validation_metrics = validate_single_soi_model_with_metrics(
            model,
            validation_loader,
            device,
            options,
            soi_type=(
                None
                if not metadata or metadata.get("soi_type") is None
                else str(metadata["soi_type"])
            ),
        )
        validation_loss = validation_metrics.loss
        duration = perf_counter() - started
        history.train_loss.append(train_loss)
        history.validation_loss.append(validation_loss)
        history.official_validation_mse_score_db.append(
            validation_metrics.official_mse_score_db
        )
        history.official_validation_mse_db_by_sinr.append(
            validation_metrics.mse_db_by_sinr
        )
        history.validation_complex_mse.append(validation_metrics.complex_mse)
        history.validation_ber.append(validation_metrics.ber)
        history.official_validation_ber_score_db.append(
            validation_metrics.official_ber_score_db
        )
        history.official_validation_ber_by_sinr.append(
            validation_metrics.ber_by_sinr
        )
        history.epoch_seconds.append(duration)
        effective_dilations = _project_and_measure_dilations(model)
        history.effective_dilations.append(effective_dilations)
        if scheduler is not None:
            scheduler.step(validation_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        dilation_summary = ""
        if effective_dilations:
            dilation_summary = (
                " dilation=[%.3f, %.3f]" % (
                    min(effective_dilations),
                    max(effective_dilations),
                )
            )
        official_mse_text = (
            "n/a"
            if validation_metrics.official_mse_score_db is None
            else f"{validation_metrics.official_mse_score_db:.5f} dB"
        )
        ber_text = (
            "n/a"
            if validation_metrics.ber is None
            else f"{validation_metrics.ber:.8f}"
        )
        official_ber_text = (
            "n/a"
            if validation_metrics.official_ber_score_db is None
            else f"{validation_metrics.official_ber_score_db:.1f} dB"
        )
        log.info(
            "Epoch [%3d/%d] - Train Loss: %8.4f, Val Loss: %8.4f, "
            "Val Complex MSE: %.6f, Official MSE: %s, "
            "Val BER: %s, Official BER Score: %s, LR: %.2e, Time: %5.1fs%s",
            epoch + 1,
            options.epochs,
            train_loss,
            validation_loss,
            validation_metrics.complex_mse,
            official_mse_text,
            ber_text,
            official_ber_text,
            learning_rate,
            duration,
            dilation_summary,
        )

        checkpoint_meta = dict(metadata or {})
        checkpoint_meta.update(
            {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_complex_mse": validation_metrics.complex_mse,
                "official_validation_mse_score_db": validation_metrics.official_mse_score_db,
                "official_validation_mse_db_by_sinr": validation_metrics.mse_db_by_sinr,
                "validation_ber": validation_metrics.ber,
                "official_validation_ber_score_db": validation_metrics.official_ber_score_db,
                "official_validation_ber_by_sinr": validation_metrics.ber_by_sinr,
            }
        )
        previous_best = best_validation_loss
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            best_validation_loss=best_validation_loss,
            metadata=checkpoint_meta,
        )
        payload.update(
            {
                "scheduler_state_dict": (
                    None if scheduler is None else scheduler.state_dict()
                ),
                "scaler_state_dict": scaler.state_dict(),
                "epochs_without_improvement": epochs_without_improvement,
                "history": history.to_dict(),
                **_capture_random_state(),
            }
        )
        if improved:
            log.info(
                "--> Validation loss improved from %.6f to %.6f "
                "(improvement: %.6f)",
                previous_best,
                validation_loss,
                previous_best - validation_loss,
            )
            save_checkpoint(best_weights_path, model.state_dict())
            log.info("Best model weights saved/updated at %s", best_weights_path)
            save_checkpoint(best_training_path, payload)
            log.info("Best training checkpoint saved/updated at %s", best_training_path)
            save_checkpoint(checkpoint_best_weights_path, model.state_dict())
            log.info(
                "Checkpoint best weights saved/updated at %s",
                checkpoint_best_weights_path,
            )
            save_checkpoint(directory / "best.pt", payload)
        save_checkpoint(latest_training_path, payload)
        log.info("Latest training checkpoint saved/updated at %s", latest_training_path)
        if (epoch + 1) % options.save_every_epochs == 0 or epoch + 1 == options.epochs:
            save_checkpoint(directory / "last.pt", payload)
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history.to_dict(), handle, indent=2)
        if (
            options.early_stopping_patience is not None
            and epochs_without_improvement >= options.early_stopping_patience
        ):
            log.info(
                "Early stopping after %d epochs without validation improvement",
                epochs_without_improvement,
            )
            break

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history.to_dict(), handle, indent=2)
    return history
