"""Validate reference-assisted BER on saved MATLAB target streams.

The mandatory sanity check is target-vs-target BER.  A receiver configuration
must pass this check before BER from a separation model is interpreted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.metrics import reference_ber_iq_from_bits


DATASET_MODULATIONS = {
    **{f"8PSK-{suffix}": ["8PSK"] * count for suffix, count in {
        "A": 2, "B": 2, "C": 3, "D": 3, "E": 2, "F": 2,
        "G": 2, "H": 2, "I": 2, "J": 3, "K": 2, "L": 2,
    }.items()},
    "QPSK+16APSK-A": ["QPSK", "16APSK"],
    "QPSK+16APSK-B": ["QPSK", "16APSK"],
    "QAM-A": ["16QAM", "64QAM"],
    "QAM-B": ["64QAM", "64QAM"],
    "QAM-C": ["64QAM", "128QAM"],
    "QAM-D": ["16QAM", "64QAM", "128QAM"],
    "QAM-E": ["16QAM", "64QAM", "128QAM"],
}


def _vector(handle: h5py.File, key: str) -> np.ndarray:
    return np.asarray(handle[key]).reshape(-1)


def _bit_path(dataset_dir: Path, target_path: Path, source_idx: int) -> Path:
    name = target_path.name.replace("Dataset_target_", "BitData_")
    name = re.sub(r"\.mat$", f"_Source{source_idx + 1}.mat", name)
    path = dataset_dir / "bits" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing bit labels: {path}")
    return path


def audit_target_file(target_path: Path, dataset_name: str) -> dict:
    dataset_dir = target_path.parent.parent
    modulations = DATASET_MODULATIONS[dataset_name]
    with h5py.File(target_path, "r") as handle:
        raw = np.asarray(handle["ideal_frames"][:])
        sps_by_source = _vector(handle, "Fs_sps_by_source").astype(int)
        cfo_by_source = _vector(handle, "cfo_hz").astype(float)
        sample_rate_hz = float(_vector(handle, "sample_rate_mhz")[0]) * 1e6
        rrc_alpha = float(_vector(handle, "rrc_alpha")[0])
        rrc_span = int(_vector(handle, "rrc_span")[0])

    results = []
    for source_idx, modulation in enumerate(modulations):
        stream = (
            raw[2 * source_idx] + 1j * raw[2 * source_idx + 1]
        ).T.reshape(-1)
        with h5py.File(_bit_path(dataset_dir, target_path, source_idx), "r") as handle:
            bits = np.asarray(handle["file_bits"]).reshape(-1).astype(np.uint8)
        iq = torch.from_numpy(
            np.stack([stream.real, stream.imag], axis=0)[None].astype(np.float32)
        )
        bit_tensor = torch.from_numpy(bits[None])
        value = reference_ber_iq_from_bits(
            iq,
            iq,
            bit_tensor,
            modulation=modulation,
            sps=int(sps_by_source[source_idx]),
            sample_rate_hz=sample_rate_hz,
            cfo_hz=float(cfo_by_source[source_idx]),
            rrc_alpha=rrc_alpha,
            rrc_span=rrc_span,
        )
        results.append({
            "source": source_idx + 1,
            "modulation": modulation,
            "sps": int(sps_by_source[source_idx]),
            "cfo_hz": float(cfo_by_source[source_idx]),
            "target_vs_target_ber": float(value),
        })
    return {
        "dataset": dataset_name,
        "target_file": str(target_path),
        "receiver": "reference_assisted",
        "sample_rate_hz": sample_rate_hz,
        "sources": results,
        "passed": all(item["target_vs_target_ber"] <= 1e-4 for item in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_MODULATIONS))
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic",
    )
    parser.add_argument("--target-file", type=Path)
    args = parser.parse_args()

    dataset_dir = args.synthetic_root / args.dataset
    if args.target_file:
        target_path = args.target_file
    else:
        target_files = sorted((dataset_dir / "target").glob("*.mat"))
        if not target_files:
            raise FileNotFoundError(f"No target files under {dataset_dir / 'target'}")
        target_path = target_files[0]

    result = audit_target_file(target_path, args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
