"""Audit MATLAB synthetic dataset metrics without training a model.

This script checks whether unexpectedly high early SI-SNR is already explained
by the data itself.  It reports mixture/target physical consistency and the
score of a trivial baseline that repeats the mixture as every separated source.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


EPS = 1e-8


def _as_float_scalar(value, default=None):
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return float(arr.reshape(-1)[0])


def _read_h5_scalar(handle, key: str, default=None):
    if key not in handle:
        return default
    return _as_float_scalar(handle[key][()], default)


def _dataset_base(root: Path, data_choice: str) -> Path:
    candidates = [root / data_choice, root]
    for candidate in candidates:
        if (candidate / "target").is_dir() and (candidate / "mixture").is_dir():
            return candidate
    return root / data_choice


def _snr_key(path: Path):
    match = re.search(r"_SNR=?([+-]?\d+(?:\.\d+)?)dB", path.name)
    return float(match.group(1)) if match else float("inf")


def _file_idx_key(path: Path):
    match = re.search(r"_(?:target|mixed)_(\d+)_SNR", path.name)
    return int(match.group(1)) if match else 10**9


def _discover_pairs(base: Path, data_choice: str, max_files: int | None):
    target_patterns = [
        f"*{data_choice}*Dataset_target_*_SNR*.mat",
        "*Dataset_target_*_SNR*.mat",
    ]
    target_files = []
    for pattern in target_patterns:
        target_files = sorted((base / "target").glob(pattern), key=lambda p: (_snr_key(p), _file_idx_key(p), p.name))
        if target_files:
            break
    if max_files is not None:
        target_files = target_files[: max(1, int(max_files))]

    pairs = []
    for target_path in target_files:
        mixture_name = target_path.name.replace("_target_", "_mixed_")
        mixture_path = base / "mixture" / mixture_name
        if not mixture_path.exists():
            mixture_path = base / "mixture" / mixture_name.replace("SNR=", "SNR")
        if mixture_path.exists():
            pairs.append((target_path, mixture_path))
    if not pairs:
        raise FileNotFoundError(f"No target/mixture pairs found under {base}")
    return pairs


def _load_pair(target_path: Path, mixture_path: Path):
    with h5py.File(target_path, "r") as target_file:
        target = np.asarray(target_file["ideal_frames"][:])
        target = np.transpose(target, (2, 1, 0)).astype(np.float64, copy=False)
        meta = {
            "raw_frame_length": int(_read_h5_scalar(target_file, "frame_length", target.shape[1])),
            "valid_frame_length": int(_read_h5_scalar(target_file, "valid_frame_length", target.shape[1])),
            "Fs_sps": _read_h5_scalar(target_file, "Fs_sps", None),
            "symbols_per_frame": _read_h5_scalar(target_file, "symbols_per_frame", None),
        }
        for key in ("Fs_sps_by_source", "symbols_per_frame_by_source", "bits_per_frame_by_source"):
            if key in target_file:
                meta[key] = np.asarray(target_file[key]).reshape(-1).astype(float).tolist()

    with h5py.File(mixture_path, "r") as mixture_file:
        mixture = np.asarray(mixture_file["mixed_frames"][:])
        mixture = np.transpose(mixture, (2, 1, 0)).astype(np.float64, copy=False)
    return target, mixture, meta


def _crop_frames(target: np.ndarray, mixture: np.ndarray, used_length: int | None):
    length = min(target.shape[1], mixture.shape[1])
    if used_length is not None:
        length = min(length, int(used_length))
    return target[:, :length, :], mixture[:, :length, :], length


def _iq_to_complex(x: np.ndarray) -> np.ndarray:
    return x[:, 0, :] + 1j * x[:, 1, :]


def _split_sources(target: np.ndarray, num_sources: int):
    return [target[:, :, 2 * idx : 2 * idx + 2].transpose(0, 2, 1) for idx in range(num_sources)]


def _si_snr_real_per_item(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    dot_pt = np.sum(pred * target, axis=(1, 2))
    dot_tt = np.sum(target * target, axis=(1, 2))
    alpha = dot_pt / (dot_tt + EPS)
    scaled = alpha[:, None, None] * target
    noise = pred - scaled
    return 10.0 * np.log10((np.sum(scaled * scaled, axis=(1, 2)) + EPS) / (np.sum(noise * noise, axis=(1, 2)) + EPS))


def _si_snr_complex_per_item(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_c = _iq_to_complex(pred)
    target_c = _iq_to_complex(target)
    pred_c = pred_c - pred_c.mean(axis=-1, keepdims=True)
    target_c = target_c - target_c.mean(axis=-1, keepdims=True)
    alpha = np.sum(pred_c * np.conj(target_c), axis=-1, keepdims=True) / (
        np.sum(np.abs(target_c) ** 2, axis=-1, keepdims=True) + EPS
    )
    scaled = alpha * target_c
    noise = pred_c - scaled
    return 10.0 * np.log10((np.sum(np.abs(scaled) ** 2, axis=-1) + EPS) / (np.sum(np.abs(noise) ** 2, axis=-1) + EPS))


def _pit_mean(preds: list[np.ndarray], targets: list[np.ndarray], metric_fn) -> float:
    scores = []
    for perm in itertools.permutations(range(len(targets))):
        total = None
        for target_idx, pred_idx in enumerate(perm):
            value = metric_fn(preds[pred_idx], targets[target_idx])
            total = value if total is None else total + value
        scores.append(total / len(targets))
    stacked = np.stack(scores, axis=0)
    return float(np.max(stacked, axis=0).mean())


def _abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_c = _iq_to_complex(x)
    y_c = _iq_to_complex(y)
    x_c = x_c - x_c.mean(axis=-1, keepdims=True)
    y_c = y_c - y_c.mean(axis=-1, keepdims=True)
    numerator = np.abs(np.mean(x_c * np.conj(y_c), axis=-1))
    denominator = np.sqrt(np.mean(np.abs(x_c) ** 2, axis=-1) * np.mean(np.abs(y_c) ** 2, axis=-1)) + EPS
    return float(np.mean(numerator / denominator))


def audit_dataset(data_choice: str, synthetic_root: Path, max_files: int | None = 8, used_length: int | None = 4096):
    base = _dataset_base(Path(synthetic_root), data_choice)
    pairs = _discover_pairs(base, data_choice, max_files)
    consistency_snr = []
    repeat_paper = []
    repeat_complex = []
    source_pearson = []
    source_rms_db = None
    first_meta = None
    used_lengths = []

    for target_path, mixture_path in pairs:
        target, mixture, meta = _load_pair(target_path, mixture_path)
        target, mixture, length = _crop_frames(target, mixture, used_length)
        used_lengths.append(length)
        if first_meta is None:
            first_meta = dict(meta)
            first_meta["target_path"] = str(target_path)
            first_meta["mixture_path"] = str(mixture_path)

        num_sources = target.shape[2] // 2
        targets = _split_sources(target, num_sources)
        mix = mixture.transpose(0, 2, 1)
        summed = np.zeros_like(mix)
        for source in targets:
            summed += source

        residual = mix - summed
        consistency_snr.append(
            10.0 * np.log10((np.mean(summed * summed) + EPS) / (np.mean(residual * residual) + EPS))
        )

        repeat_preds = [mix for _ in range(num_sources)]
        repeat_paper.append(_pit_mean(repeat_preds, targets, _si_snr_real_per_item))
        repeat_complex.append(_pit_mean(repeat_preds, targets, _si_snr_complex_per_item))

        if num_sources >= 2:
            source_pearson.append(_abs_pearson(targets[0], targets[1]))

        rms = []
        for source in targets:
            rms.append(float(np.sqrt(np.mean(source * source) + EPS)))
        source_rms_db = [20.0 * np.log10(value + EPS) for value in rms]

    return {
        "data_choice": data_choice,
        "base": str(base),
        "num_files": len(pairs),
        "raw_frame_length": first_meta.get("raw_frame_length") if first_meta else None,
        "valid_frame_length": first_meta.get("valid_frame_length") if first_meta else None,
        "used_frame_length": int(min(used_lengths)) if used_lengths else None,
        "first_meta": first_meta,
        "mixture_consistency_snr_db": float(np.mean(consistency_snr)),
        "repeat_mixture_pit_si_snr_paper_db": float(np.mean(repeat_paper)),
        "repeat_mixture_pit_si_snr_complex_db": float(np.mean(repeat_complex)),
        "source_abs_pearson": float(np.mean(source_pearson)) if source_pearson else None,
        "source_rms_db": source_rms_db,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit synthetic dataset metric baselines.")
    parser.add_argument("--data_choice", required=True)
    parser.add_argument("--synthetic_root", required=True)
    parser.add_argument("--max_files", type=int, default=8)
    parser.add_argument("--used_length", type=int, default=4096)
    args = parser.parse_args()

    result = audit_dataset(
        data_choice=args.data_choice,
        synthetic_root=Path(args.synthetic_root),
        max_files=args.max_files,
        used_length=args.used_length,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
