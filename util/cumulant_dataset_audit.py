"""Audit the cumulant prior against generated MATLAB IQ datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from util.cumulant_prior import complex_fourth_cumulants, cumulant_prior_loss


def _snr(path: Path) -> float:
    match = re.search(r"SNR=([+-]?\d+(?:\.\d+)?)dB", path.name)
    if not match:
        raise ValueError(f"Cannot parse SNR from {path}")
    return float(match.group(1))


def _paired_mixture(target_path: Path) -> Path:
    name = target_path.name.replace("_target_", "_mixed_")
    return target_path.parents[1] / "mixture" / name


def _load_frames(target_path: Path, frame_count: int):
    mixture_path = _paired_mixture(target_path)
    with h5py.File(target_path, "r") as handle:
        target_ds = handle["ideal_frames"]
        count = min(frame_count, target_ds.shape[2])
        indices = np.linspace(0, target_ds.shape[2] - 1, count, dtype=np.int64)
        target = np.stack([target_ds[:, :, int(index)] for index in indices], axis=0)
    with h5py.File(mixture_path, "r") as handle:
        mixture_ds = handle["mixed_frames"]
        mixture = np.stack([mixture_ds[:, :, int(index)] for index in indices], axis=0)
    return torch.from_numpy(mixture).float(), torch.from_numpy(target).float()


def _complex_sources(target: torch.Tensor) -> torch.Tensor:
    sources = target.size(1) // 2
    view = target.reshape(target.size(0), sources, 2, target.size(-1))
    return torch.complex(view[:, :, 0], view[:, :, 1])


def _cumulant_summary(iq: torch.Tensor) -> dict[str, float]:
    c40, c42 = complex_fourth_cumulants(iq)
    return {
        "c40_abs_mean": float(c40.abs().mean()),
        "c42_abs_mean": float(c42.abs().mean()),
        "c42_std": float(c42.std(unbiased=False)),
    }


def audit_file(target_path: Path, frame_count: int) -> dict:
    mixture, target = _load_frames(target_path, frame_count)
    sources = target.size(1) // 2
    source_view = target.reshape(target.size(0), sources, 2, target.size(-1))
    clean_mix = source_view.sum(dim=1)
    noise = mixture - clean_mix
    repeated_mix = mixture.unsqueeze(1).expand(-1, sources, -1, -1).reshape_as(target)
    shared_noise_sources = source_view + noise.unsqueeze(1) / sources
    noisy_proxy = shared_noise_sources.reshape_as(target)

    signal_power = clean_mix.square().mean()
    noise_power = noise.square().mean().clamp_min(1e-12)
    measured_snr = 10.0 * torch.log10(signal_power / noise_power)
    noise_complex = torch.complex(noise[:, 0], noise[:, 1]).unsqueeze(1)

    return {
        "file": target_path.name,
        "snr_label_db": _snr(target_path),
        "snr_measured_db": float(measured_snr),
        "num_sources": sources,
        "clean": _cumulant_summary(_complex_sources(target)),
        "noise": _cumulant_summary(noise_complex),
        "noisy_proxy_prior_loss": float(cumulant_prior_loss(noisy_proxy, target)),
        "repeat_mixture_prior_loss": float(cumulant_prior_loss(repeated_mix, target)),
    }


def audit_root(root: Path, frame_count: int) -> dict[str, list[dict]]:
    results = {}
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted((dataset_dir / "target").glob("*.mat"), key=_snr)
        if not files:
            continue
        snrs = sorted({_snr(path) for path in files})
        selected = []
        for snr in (snrs[0], snrs[-1]):
            candidates = [path for path in files if _snr(path) == snr]
            selected.append(candidates[0])
        results[dataset_dir.name] = [audit_file(path, frame_count) for path in selected]
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    results = audit_root(args.root, args.frames)
    payload = json.dumps(results, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
