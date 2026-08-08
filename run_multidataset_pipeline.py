"""One-command joint pretraining, per-domain fine-tuning, and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
PIPELINE_ROOT = PROJECT_ROOT / "pipeline_results"
RESULT_RE = re.compile(r"Results saved in:\s*([^\s]+)")
BEST_VAL_RE = re.compile(r"Best validation loss:\s*([-+0-9.eE]+)")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly pretrain one model, initialize a fresh fine-tuning run for "
            "each target dataset, evaluate its held-out test split, and summarize results."
        )
    )
    parser.add_argument("--pretrain_data_choices", nargs="+", required=True)
    parser.add_argument(
        "--target_data_choices",
        nargs="+",
        default=None,
        help="Fine-tuning/evaluation datasets (default: all pretraining datasets).",
    )
    parser.add_argument("--data_choice", default=None, help="Pretraining input reference dataset.")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--finetune_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_names", nargs="+", default=["S1", "S2"])
    parser.add_argument("--num_sources", type=int, default=2)
    parser.add_argument("--synthetic_root", default=None)
    parser.add_argument("--public_root", default=None)
    parser.add_argument("--split_strategy", choices=["random", "stratified_snr"], default="stratified_snr")
    parser.add_argument("--pretrain_sampling", choices=["balanced", "proportional"], default="balanced")
    parser.add_argument("--pretrain_input_size", type=int, default=4096)
    parser.add_argument(
        "--pretrain_length_policy",
        choices=["strict", "crop", "pad_crop"],
        default="strict",
    )
    parser.add_argument("--loss_fun", default="SI-SNR+Huber")
    parser.add_argument("--si_snr_huber_alpha", type=float, default=1.0)
    parser.add_argument("--si_snr_huber_beta", type=float, default=0.5)
    parser.add_argument("--si_snr_huber_delta", type=float, default=1.0)
    parser.add_argument("--eval_pit_metric", default="none")
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        help="Skip joint pretraining and use this checkpoint for all fine-tuning runs.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--pipeline_name",
        default=None,
        help="Output folder name under pipeline_results (default: timestamped stage name).",
    )
    args, main_args = parser.parse_known_args()
    return args, main_args


def validate_args(args: argparse.Namespace, main_args: list[str]) -> None:
    if args.pretrain_epochs <= 0 or args.finetune_epochs <= 0:
        raise ValueError("--pretrain_epochs and --finetune_epochs must be positive")
    if len(args.source_names) != args.num_sources:
        raise ValueError("--source_names must contain exactly --num_sources entries")
    forbidden = {
        "--mode", "--data_choice", "--pretrain_data_choices", "--target_data_choices",
        "--num_epochs", "--init_checkpoint", "--resume_checkpoint", "--weights_path",
        "--stage", "--seed", "--source_names", "--num_sources",
    }
    conflicts = [token for token in main_args if token.split("=", 1)[0] in forbidden]
    if conflicts:
        raise ValueError(f"Pipeline-controlled options cannot be forwarded: {conflicts}")


def common_main_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--stage", str(args.stage),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--split_strategy", args.split_strategy,
        "--source_names", *args.source_names,
        "--num_sources", str(args.num_sources),
        "--loss_fun", args.loss_fun,
        "--si_snr_huber_alpha", str(args.si_snr_huber_alpha),
        "--si_snr_huber_beta", str(args.si_snr_huber_beta),
        "--si_snr_huber_delta", str(args.si_snr_huber_delta),
        "--eval_pit_metric", args.eval_pit_metric,
        "--early_stop_patience", str(args.early_stop_patience),
    ]
    if args.synthetic_root:
        command.extend(["--synthetic_root", args.synthetic_root])
    if args.public_root:
        command.extend(["--public_root", args.public_root])
    return command


def display_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command)) if os.name == "nt" else " ".join(
        subprocess.list2cmdline([part]) for part in command
    )


def run_main(command: list[str], log_path: Path, dry_run: bool) -> tuple[str | None, float | None]:
    print(f"\n$ {display_command(command)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(display_command(command) + "\n", encoding="utf-8")
        return None, None

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result_folders: list[str] = []
    best_val: float | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            match = RESULT_RE.search(line)
            if match:
                result_folders.append(match.group(1).strip())
            val_match = BEST_VAL_RE.search(line)
            if val_match:
                best_val = float(val_match.group(1))
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if not result_folders:
        raise RuntimeError(f"Training finished but no results folder was reported; inspect {log_path}")
    return result_folders[-1], best_val


def find_checkpoint(results_folder: str) -> Path:
    weights = RESULTS_ROOT / results_folder / "weights"
    candidates = [weights / "best_training_checkpoint.pth", weights / "best_model_weights.pth"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No best checkpoint found under {weights}")


def safe_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def collect_metrics(dataset: str, results_folder: str, best_val: float | None) -> tuple[dict, list[dict]]:
    csv_path = RESULTS_ROOT / results_folder / "detailed_metrics_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing evaluation CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("Source") == "Overall"]
    if not rows:
        raise ValueError(f"No Overall rows found in {csv_path}")

    details = []
    for row in rows:
        details.append({
            "dataset": dataset,
            "results_folder": results_folder,
            "snr_db": safe_float(row.get("SNR")),
            "si_snr_real_db": safe_float(row.get("SI-SNR_real")),
            "si_snr_complex_db": safe_float(row.get("SI-SNR_complex")),
            "mse": safe_float(row.get("MSE")),
            "scale_aligned_mse": safe_float(row.get("ScaleAligned_MSE")),
            "correlation": safe_float(row.get("Correlation")),
        })
    summary = {
        "dataset": dataset,
        "results_folder": results_folder,
        "best_val_loss": float("nan") if best_val is None else best_val,
        "snr_points": len(details),
        "mean_si_snr_real_db": mean([row["si_snr_real_db"] for row in details]),
        "mean_si_snr_complex_db": mean([row["si_snr_complex_db"] for row in details]),
        "mean_mse": mean([row["mse"] for row in details]),
        "mean_scale_aligned_mse": mean([row["scale_aligned_mse"] for row in details]),
        "mean_correlation": mean([row["correlation"] for row in details]),
    }
    return summary, details


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args, forwarded = parse_args()
    validate_args(args, forwarded)
    targets = args.target_data_choices or list(args.pretrain_data_choices)
    reference = args.data_choice or args.pretrain_data_choices[0]
    run_name = args.pipeline_name or f"stage{args.stage}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = PIPELINE_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    base = [sys.executable, str(PROJECT_ROOT / "main.py"), *common_main_args(args), *forwarded]
    manifest = vars(args).copy()
    manifest.update({"targets": targets, "reference_data_choice": reference, "forwarded_main_args": forwarded})
    (output_dir / "pipeline_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checkpoint: Path
    pretrain_result: str | None = None
    if args.pretrained_checkpoint:
        checkpoint = Path(args.pretrained_checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    else:
        pretrain_command = [
            *base,
            "--mode", "train",
            "--data_choice", reference,
            "--pretrain_data_choices", *args.pretrain_data_choices,
            "--pretrain_sampling", args.pretrain_sampling,
            "--pretrain_input_size", str(args.pretrain_input_size),
            "--pretrain_length_policy", args.pretrain_length_policy,
            "--num_epochs", str(args.pretrain_epochs),
        ]
        pretrain_result, _ = run_main(pretrain_command, output_dir / "pretrain.log", args.dry_run)
        if args.dry_run:
            checkpoint = Path("<PRETRAIN_CHECKPOINT>")
        else:
            assert pretrain_result is not None
            checkpoint = find_checkpoint(pretrain_result)

    summaries: list[dict] = []
    details: list[dict] = []
    failures: list[dict] = []
    for dataset in targets:
        command = [
            *base,
            "--mode", "train",
            "--data_choice", dataset,
            "--init_checkpoint", str(checkpoint),
            "--num_epochs", str(args.finetune_epochs),
        ]
        try:
            results_folder, best_val = run_main(
                command, output_dir / "logs" / f"finetune_{dataset.replace('+', '_')}.log", args.dry_run
            )
            if not args.dry_run:
                assert results_folder is not None
                summary, dataset_details = collect_metrics(dataset, results_folder, best_val)
                summaries.append(summary)
                details.extend(dataset_details)
                write_csv(output_dir / "summary.csv", summaries)
                write_csv(output_dir / "per_snr_metrics.csv", details)
        except Exception as exc:
            failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            (output_dir / "failures.json").write_text(
                json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if not args.continue_on_error:
                raise

    final_manifest = {
        "pretrain_results_folder": pretrain_result,
        "pretrained_checkpoint": str(checkpoint),
        "completed_datasets": [row["dataset"] for row in summaries],
        "failures": failures,
    }
    (output_dir / "pipeline_results.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nPipeline outputs: {output_dir}")
    if summaries:
        print(f"Summary: {output_dir / 'summary.csv'}")
        print(f"Per-SNR metrics: {output_dir / 'per_snr_metrics.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
