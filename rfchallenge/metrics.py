"""Official-format RF Challenge scoring and submission artifact helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .datasets import load_ground_truth
from .protocol import (
    FRAME_LENGTH,
    SINR_DB_VALUES,
    demodulate_soi,
)


OFFICIAL_MSE_SCORE_FLOOR_DB = -50.0
OFFICIAL_BER_THRESHOLD = 1.0e-2


def official_case_mse_score_db(
    mse_db: np.ndarray | Sequence[float],
    floor_db: float = OFFICIAL_MSE_SCORE_FLOOR_DB,
) -> float:
    """Return the challenge's truncated mean MSE score for one SOI/interference case.

    The public specification first computes one MSE value in dB per provided
    SINR level, clips every value below ``-50 dB``, then averages the eleven
    clipped dB values.  This intentionally differs from taking one dB value
    after averaging linear MSE across all frames.
    """

    values = np.asarray(mse_db, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("mse_db must contain at least one SINR-level value")
    if np.isnan(values).any():
        raise ValueError("mse_db must not contain NaN")
    if not np.isfinite(floor_db):
        raise ValueError("floor_db must be finite")
    return float(np.mean(np.maximum(float(floor_db), values)))


def official_case_ber_score_db(
    sinr_db: np.ndarray | Sequence[float],
    ber: np.ndarray | Sequence[float],
    threshold: float = OFFICIAL_BER_THRESHOLD,
) -> float:
    """Return the lowest target SINR whose BER reaches the official threshold.

    The public leaderboard assigns zero when no SINR bin reaches ``BER <= 1e-2``.
    More-negative values are better because they indicate successful recovery at
    a lower target SINR.
    """

    sinr_values = np.asarray(sinr_db, dtype=np.float32).reshape(-1)
    ber_values = np.asarray(ber, dtype=np.float32).reshape(-1)
    if sinr_values.size == 0 or sinr_values.shape != ber_values.shape:
        raise ValueError("sinr_db and ber must be non-empty arrays of equal shape")
    if np.isnan(sinr_values).any() or np.isnan(ber_values).any():
        raise ValueError("sinr_db and ber must not contain NaN")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    qualifying = sinr_values[ber_values <= float(threshold)]
    return 0.0 if qualifying.size == 0 else float(np.min(qualifying))


@dataclass(frozen=True)
class RFChallengeCaseMetrics:
    """Per-SINR MSE/BER values produced by the public evaluation protocol."""

    soi_type: str
    interference_type: str
    sinr_db: np.ndarray
    mse_db: np.ndarray
    ber: np.ndarray
    frame_count: np.ndarray
    raw_mse: np.ndarray

    @property
    def official_mse_score_db(self) -> float:
        """Official per-case truncated MSE scalar, not a hidden leaderboard aggregate."""

        return official_case_mse_score_db(self.mse_db)

    @property
    def official_ber_score_db(self) -> float:
        """Official per-case BER threshold SINR scalar."""

        return official_case_ber_score_db(self.sinr_db, self.ber)

    def to_dict(self) -> dict:
        return {
            "soi_type": self.soi_type,
            "interference_type": self.interference_type,
            "sinr_db": self.sinr_db.tolist(),
            "mse_db": self.mse_db.tolist(),
            "ber": self.ber.tolist(),
            "frame_count": self.frame_count.tolist(),
            "raw_mse": self.raw_mse.tolist(),
            "official_mse_score_db": self.official_mse_score_db,
            "official_ber_score_db": self.official_ber_score_db,
        }


def expected_bits_per_frame(soi_type: str, frame_length: int = FRAME_LENGTH) -> int:
    """Return the submission bit-width defined by the public starter kit."""

    normalized = str(soi_type).upper()
    if normalized == "QPSK":
        if frame_length % 16:
            raise ValueError("QPSK frame_length must be divisible by 16")
        return frame_length // 16 * 2
    if normalized == "OFDMQPSK":
        if frame_length % 80:
            raise ValueError("OFDMQPSK frame_length must be divisible by 80")
        return frame_length // 80 * 56 * 2
    raise ValueError(f"Unsupported SOI type '{soi_type}'")


def _as_complex_batch(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a complex array with shape (B, L), got {array.shape}")
    return np.ascontiguousarray(array.astype(np.complex64, copy=False))


def _ordered_sinr_groups(
    total_frames: int,
    nominal_sinr_db: np.ndarray | None,
    n_per_sinr: int | None,
) -> list[tuple[float, np.ndarray]]:
    if nominal_sinr_db is not None:
        labels = np.asarray(nominal_sinr_db, dtype=np.float32).reshape(-1)
        if labels.shape[0] != total_frames:
            raise ValueError("nominal_sinr_db and predictions must have matching frame counts")
        # Preserve official ascending SINR order, including any nonstandard labels.
        values = np.sort(np.unique(labels))
        return [(float(value), np.flatnonzero(np.isclose(labels, value))) for value in values]

    if n_per_sinr is None:
        if total_frames % len(SINR_DB_VALUES):
            raise ValueError(
                "Cannot infer official SINR groups: frame count is not divisible by 11. "
                "Pass nominal_sinr_db or n_per_sinr explicitly."
            )
        n_per_sinr = total_frames // len(SINR_DB_VALUES)
    if n_per_sinr <= 0 or total_frames != len(SINR_DB_VALUES) * int(n_per_sinr):
        raise ValueError(
            "Expected exactly 11 contiguous SINR groups; got "
            f"total_frames={total_frames}, n_per_sinr={n_per_sinr}"
        )
    return [
        (
            float(sinr),
            np.arange(index * n_per_sinr, (index + 1) * n_per_sinr, dtype=np.int64),
        )
        for index, sinr in enumerate(SINR_DB_VALUES)
    ]


def evaluate_case(
    estimated_soi: np.ndarray,
    target_soi: np.ndarray,
    target_bits: np.ndarray,
    soi_type: str,
    interference_type: str,
    nominal_sinr_db: np.ndarray | None = None,
    n_per_sinr: int | None = None,
    estimated_bits: np.ndarray | None = None,
) -> tuple[RFChallengeCaseMetrics, np.ndarray]:
    """Calculate the starter-kit MSE and BER values for a locally scored set.

    MSE is measured on the raw complex SOI estimate. No gain, phase, delay, or
    source permutation alignment is applied because the official evaluator
    does not apply one either.
    """

    prediction = _as_complex_batch(estimated_soi, "estimated_soi")
    target = _as_complex_batch(target_soi, "target_soi")
    true_bits = np.asarray(target_bits, dtype=np.uint8)
    if prediction.shape != target.shape:
        raise ValueError(
            f"estimated_soi shape {prediction.shape} does not match target_soi {target.shape}"
        )
    if true_bits.ndim != 2 or true_bits.shape[0] != prediction.shape[0]:
        raise ValueError("target_bits must have shape (B, num_bits)")

    if estimated_bits is None:
        recovered_bits, _ = demodulate_soi(soi_type, prediction)
    else:
        recovered_bits = np.asarray(estimated_bits, dtype=np.uint8)
    if recovered_bits.shape != true_bits.shape:
        raise ValueError(
            "Estimated bit shape does not match ground truth: "
            f"estimated={recovered_bits.shape}, target={true_bits.shape}"
        )
    if np.any((recovered_bits != 0) & (recovered_bits != 1)):
        raise ValueError("estimated_bits must contain only zeros and ones")
    # Keep starter-script dtypes and reduction order. Its MSE intermediates
    # remain float32, and BER is computed as a float32 bit-error count divided
    # by the bit width. Preserving this makes local score arrays byte-for-byte
    # comparable to the public evaluation script rather than merely close.
    per_frame_mse = np.mean(np.abs(prediction - target) ** 2, axis=1)
    per_frame_ber = np.sum(
        (recovered_bits != true_bits).astype(np.float32), axis=1
    ) / np.float32(true_bits.shape[1])
    groups = _ordered_sinr_groups(prediction.shape[0], nominal_sinr_db, n_per_sinr)

    sinr_values = np.asarray([group[0] for group in groups], dtype=np.float32)
    mse_values = np.asarray(
        [np.mean(per_frame_mse[group[1]]) for group in groups], dtype=np.float32
    )
    ber_values = np.asarray(
        [np.mean(per_frame_ber[group[1]]) for group in groups], dtype=np.float32
    )
    counts = np.asarray([group[1].size for group in groups], dtype=np.int64)
    with np.errstate(divide="ignore"):
        mse_db = (10.0 * np.log10(mse_values)).astype(np.float32)
    result = RFChallengeCaseMetrics(
        soi_type=str(soi_type),
        interference_type=str(interference_type),
        sinr_db=sinr_values,
        mse_db=mse_db,
        ber=ber_values,
        frame_count=counts,
        raw_mse=mse_values,
    )
    return result, recovered_bits.astype(np.uint8, copy=False)


def save_submission_artifacts(
    output_dir: str | Path,
    method_id: str,
    testset_identifier: str,
    soi_type: str,
    interference_type: str,
    estimated_soi: np.ndarray,
    estimated_bits: np.ndarray,
) -> dict[str, Path]:
    """Save arrays using the official submission filename convention."""

    prediction = _as_complex_batch(estimated_soi, "estimated_soi")
    bits = np.asarray(estimated_bits, dtype=np.uint8)
    if bits.ndim != 2 or bits.shape[0] != prediction.shape[0]:
        raise ValueError("estimated_bits must have shape (B, num_bits)")
    expected_width = expected_bits_per_frame(soi_type, prediction.shape[1])
    if bits.shape[1] != expected_width:
        raise ValueError(
            f"Expected {expected_width} bits for {soi_type}, got {bits.shape[1]}"
        )
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("estimated_bits must contain only zeros and ones")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{method_id}_{testset_identifier}"
    soi_path = directory / f"{prefix}_estimated_soi_{soi_type}_{interference_type}.npy"
    bits_path = directory / f"{prefix}_estimated_bits_{soi_type}_{interference_type}.npy"
    np.save(soi_path, prediction.astype(np.complex64, copy=False), allow_pickle=False)
    np.save(bits_path, bits, allow_pickle=False)
    return {"estimated_soi": soi_path, "estimated_bits": bits_path}


def load_submission_artifacts(
    output_dir: str | Path,
    method_id: str,
    testset_identifier: str,
    soi_type: str,
    interference_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a pair of official-format prediction arrays."""

    directory = Path(output_dir)
    prefix = f"{method_id}_{testset_identifier}"
    soi_path = directory / f"{prefix}_estimated_soi_{soi_type}_{interference_type}.npy"
    bits_path = directory / f"{prefix}_estimated_bits_{soi_type}_{interference_type}.npy"
    prediction = np.load(soi_path, allow_pickle=False)
    bits = np.load(bits_path, allow_pickle=False)
    return prediction, bits


def evaluate_ground_truth_file(
    ground_truth_path: str | Path,
    estimated_soi: np.ndarray,
    soi_type: str,
    interference_type: str,
    metadata_path: str | Path | None = None,
    estimated_bits: np.ndarray | None = None,
) -> tuple[RFChallengeCaseMetrics, np.ndarray]:
    """Score a prediction against a TestSet1Example-compatible pickle file."""

    _, target, bits = load_ground_truth(ground_truth_path)
    nominal_sinr = None
    if metadata_path is not None:
        metadata = np.load(metadata_path, allow_pickle=True)
        if metadata.ndim != 2 or metadata.shape[1] < 2:
            raise ValueError(f"Unexpected metadata shape {metadata.shape}")
        nominal_sinr = np.asarray(metadata[:, 1], dtype=np.float32)
    return evaluate_case(
        estimated_soi=estimated_soi,
        target_soi=target,
        target_bits=bits,
        soi_type=soi_type,
        interference_type=interference_type,
        nominal_sinr_db=nominal_sinr,
        estimated_bits=estimated_bits,
    )


def aggregate_case_metrics(
    results: Sequence[RFChallengeCaseMetrics],
) -> dict[str, float | int]:
    """Return macro diagnostics and the public eight-case score aggregation.

    The ``official_*`` fields follow the published leaderboard formula. They
    reproduce leaderboard arithmetic, but only hidden TestSet2 data can
    reproduce an official competition result.
    """

    if not results:
        raise ValueError("At least one case result is required")
    all_mse_db = np.concatenate([result.mse_db for result in results])
    all_ber = np.concatenate([result.ber for result in results])
    official_mse_scores = np.asarray(
        [result.official_mse_score_db for result in results], dtype=np.float64
    )
    official_ber_scores = np.asarray(
        [result.official_ber_score_db for result in results], dtype=np.float64
    )
    return {
        "case_count": len(results),
        "macro_mse_db": float(np.mean(all_mse_db)),
        "macro_truncated_mse_db": float(
            np.mean([result.official_mse_score_db for result in results])
        ),
        "macro_ber": float(np.mean(all_ber)),
        "official_mse_result_db": float(np.sum(official_mse_scores)),
        "official_mse_average_db": float(np.mean(official_mse_scores)),
        "official_ber_result_db": float(np.sum(official_ber_scores)),
        "official_ber_average_db": float(np.mean(official_ber_scores)),
    }


def save_metrics_json(
    output_path: str | Path,
    result: RFChallengeCaseMetrics,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Write a portable JSON copy of one local evaluation result."""

    payload = result.to_dict()
    if extra:
        payload.update(extra)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
