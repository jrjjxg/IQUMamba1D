"""Validate reference-assisted BER on saved MATLAB target streams.

The mandatory sanity check is target-vs-target BER.  A receiver configuration
must pass this check before BER from a separation model is interpreted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.ber_receiver import evaluate_private_file


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


def audit_target_file(target_path: Path, dataset_name: str) -> dict:
    result = evaluate_private_file(target_path, dataset_name)
    result["passed"] = all(
        float(item["ber"]) <= 1e-4
        for item in result["sources"]
        if item["compared_bits"] > 0
    )
    for item in result["sources"]:
        item["target_vs_target_ber"] = item["ber"]
    return result


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
