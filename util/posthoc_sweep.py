"""Offline post-processing sweep for IQ blind separation outputs.

This diagnostic answers one concrete question:

    Does any output-side leakage cancellation / mixture-consistency step have
    oracle room to improve a trained model?

It does not train and does not change checkpoints.  It loads a trained model,
runs a validation or test split, and evaluates a grid of deterministic
post-processing parameters against the same PIT SI-SNR_complex metric used by
the project evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from itertools import permutations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "config"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader.dataloader import create_data_loaders
from models.IQUMamba1D_CyclicCorrLeakCancel import (
    apply_closed_form_leakage_cancellation,
    soft_mixture_consistency_projection,
)
from util.config import MambaConfig
from util.evaluation import extract_separation_output, reorder_outputs_by_per_sample_perm
from util.metrics import _si_snr_complex_per_item, _split_sources, pit_si_snr_complex_persample
from util.utils import Create_Mamba_model


STAGE_CONFIGS = {
    2: CONFIG_ROOT / "model_config_IQ_stage2.yaml",
    3: CONFIG_ROOT / "model_config_IQ.yaml",
    4: CONFIG_ROOT / "model_config_IQ_stage4.yaml",
    5: CONFIG_ROOT / "model_config_IQ_stage5.yaml",
    65: CONFIG_ROOT / "model_config_bimamba_stage4.yaml",
    67: CONFIG_ROOT / "model_config_IQ_stage5_320.yaml",
    68: CONFIG_ROOT / "model_config_IQ_stage4_decodermamba.yaml",
    70: CONFIG_ROOT / "model_config_IQ_stage4_rfscan_fusion.yaml",
    71: CONFIG_ROOT / "model_config_IQ_stage4_rfmamba_scan.yaml",
    72: CONFIG_ROOT / "model_config_IQ_stage4_radmamba_scan.yaml",
    73: CONFIG_ROOT / "model_config_IQ_stage4_symbol_dualpath.yaml",
    74: CONFIG_ROOT / "model_config_IQ_stage4_complex_mask_mc.yaml",
    75: CONFIG_ROOT / "model_config_IQ_stage4_noise_aware_mc.yaml",
    76: CONFIG_ROOT / "model_config_IQ_stage4_complex_adapter.yaml",
    77: CONFIG_ROOT / "model_config_IQ_stage4_cyclofresh.yaml",
    78: CONFIG_ROOT / "model_config_IQ_stage4_blind_cyclofresh.yaml",
    79: CONFIG_ROOT / "model_config_IQ_stage4_estimated_cyclofresh.yaml",
    80: CONFIG_ROOT / "model_config_IQ_stage4_cycliccorr.yaml",
    81: CONFIG_ROOT / "model_config_IQ_stage4_cycliccorr_leakcancel.yaml",
    222: CONFIG_ROOT / "model_config_IQ_stage222_evidence_moe.yaml",
    223: CONFIG_ROOT / "model_config_IQ_stage223_noise_contrastive_prior.yaml",
    224: CONFIG_ROOT / "model_config_IQ_stage224_blind_sync_factorized.yaml",
    225: CONFIG_ROOT / "model_config_IQ_stage225_gaussian_residual_prior.yaml",
    226: CONFIG_ROOT / "model_config_IQ_stage226_adaptive_multiview_prior.yaml",
    227: CONFIG_ROOT / "model_config_IQ_stage227_qam_source_prior.yaml",
    228: CONFIG_ROOT / "model_config_IQ_stage228_qam_mma_unrolled.yaml",
    229: CONFIG_ROOT / "model_config_IQ_stage229_qam_density_prior.yaml",
    230: CONFIG_ROOT / "model_config_IQ_stage230_qam_timing_prior.yaml",
    231: CONFIG_ROOT / "model_config_IQ_stage231_multiview_consistent.yaml",
    235: CONFIG_ROOT / "model_config_bimamba_cross_scale_single.yaml",
    236: CONFIG_ROOT / "model_config_bimamba_cross_scale_multi.yaml",
    237: CONFIG_ROOT / "model_config_bimamba_cross_scale_evidence.yaml",
    238: CONFIG_ROOT / "model_config_IQ_stage238_qam_turbo_unfold.yaml",
    239: CONFIG_ROOT / "model_config_bimamba_cross_scale_estimated_cyclofresh.yaml",
    240: CONFIG_ROOT / "model_config_bimamba_cross_scale_aligned.yaml",
    241: CONFIG_ROOT / "model_config_bimamba_cross_scale_multires_kv.yaml",
    242: CONFIG_ROOT / "model_config_bimamba_cross_scale_bounded_channel.yaml",
    243: CONFIG_ROOT / "model_config_bimamba_phase_equivariant_fusion.yaml",
    244: CONFIG_ROOT / "model_config_bimamba_physical_token_cross_attention.yaml",
    245: CONFIG_ROOT / "model_config_bimamba_bottleneck_self_attention.yaml",
    246: CONFIG_ROOT / "model_config_bimamba_hymba_parallel.yaml",
    247: CONFIG_ROOT / "model_config_bimamba_rf_physical_kv.yaml",
    248: CONFIG_ROOT / "model_config_bimamba_enhanced_global_cross_attention.yaml",
    249: CONFIG_ROOT / "model_config_bimamba_dual_memory_cross_attention.yaml",
    250: CONFIG_ROOT / "model_config_bimamba_hierarchical_additive_fusion.yaml",
    251: CONFIG_ROOT / "model_config_bimamba_physical_routed_enhanced_cross_attention.yaml",
    252: CONFIG_ROOT / "model_config_bimamba_unified_physical_global_kv.yaml",
    253: CONFIG_ROOT / "model_config_bimamba_physical_film_global_memory.yaml",
    254: CONFIG_ROOT / "model_config_bimamba_scale_isolated_physical_fusion.yaml",
    255: CONFIG_ROOT / "model_config_bimamba_identity_aware_physical_moe.yaml",
    256: CONFIG_ROOT / "model_config_bimamba_cross_gated_dual_memory.yaml",
    8192: CONFIG_ROOT / "model_config_IQ_stage4_8192.yaml",
    16384: CONFIG_ROOT / "model_config_IQ_stage4_16384.yaml",
    32768: CONFIG_ROOT / "model_config_IQ_stage4_32768.yaml",
}


DATA_INPUT_SIZES = {
    "2016": 128,
    "2018": 1024,
    "TorchSig": 1024,
    "debug_random": 1024,
    "QAM": 128,
    "8PSK": 2048,
    "8PSK_M": 2048,
    "8PSK_M_NS": 2048,
    "8PSK_Burst": 2048,
    "8PSK_Burst_NS": 2048,
    "8PSK_Rs": 2048,
    "8PSK_Rs_NS": 2048,
    "QPSK_16APSK": 4100,
    "QPSK_16APSK_NS": 4100,
    "16QAM_64QAM": 2048,
    "16QAM_128QAM": 2048,
    "64QAM_64QAM": 2048,
    "64QAM_128QAM": 2048,
    "16QAM_64QAM_128QAM": 2048,
    "8PSK-A": 4096,
    "8PSK-B": 4096,
    "8PSK-C": 4096,
    "8PSK-D": 4096,
    "8PSK-E": 4096,
    "8PSK-F": 4096,
    "8PSK-G": 4096,
    "8PSK-H": 4096,
    "8PSK-I": 4096,
    "8PSK-J": 4096,
    "8PSK-K": 4096,
    "8PSK-L": 4096,
    "QPSK+16APSK-A": 4096,
    "QPSK+16APSK-B": 4096,
    "QAM-A": 4096,
    "QAM-B": 4096,
    "QAM-C": 4096,
    "QAM-D": 4096,
    "QAM-E": 4096,
}


class PrintLogger:
    def info(self, message: str) -> None:
        print(message)

    def warning(self, message: str) -> None:
        print(f"[WARN] {message}")


def parse_float_grid(value, default: Iterable[float] | None = None) -> list[float]:
    if value is None:
        return list(default) if default is not None else []
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        parts = [str(part).strip() for part in value if str(part).strip()]
    out: list[float] = []
    seen: set[float] = set()
    for part in parts:
        parsed = float(part)
        if parsed not in seen:
            out.append(parsed)
            seen.add(parsed)
    return out


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_model_config_path(stage: int) -> Path:
    try:
        return STAGE_CONFIGS[int(stage)]
    except KeyError as exc:
        known = ", ".join(str(k) for k in sorted(STAGE_CONFIGS))
        raise ValueError(f"Unsupported stage {stage}. Known stages: {known}") from exc


def get_input_size(data_choice: str) -> int:
    key = str(data_choice)
    if key not in DATA_INPUT_SIZES:
        raise ValueError(f"Unsupported data_choice for posthoc sweep: {data_choice}")
    return int(DATA_INPUT_SIZES[key])


def extract_model_state_dict(loaded_obj):
    if not isinstance(loaded_obj, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(loaded_obj).__name__}")
    for key in ("best_model_state_dict", "model_state_dict", "state_dict"):
        value = loaded_obj.get(key)
        if isinstance(value, dict):
            return value, key
    return loaded_obj, "raw_state_dict"


def load_checkpoint_state(checkpoint: Path):
    try:
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(str(checkpoint), map_location="cpu")
    except Exception:
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    return extract_model_state_dict(loaded)


def maybe_strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("module.") for key in state_dict):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return state_dict


def select_loader(split: str, train_loader, val_loader, snr_loaders):
    split = str(split).lower()
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return snr_loaders
    raise ValueError("split must be one of: train, val, test")


def iter_eval_batches(selected_loader):
    if isinstance(selected_loader, dict):
        for snr in sorted(selected_loader):
            for batch in selected_loader[snr]:
                yield snr, batch
    else:
        for batch in selected_loader:
            yield None, batch


def split_batch(batch, device: torch.device):
    inputs, targets, snr, *extras = batch
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True), snr


def mean_from_sums(total: float, count: int) -> float:
    return float(total / count) if count > 0 else float("nan")


def metric_sum(outputs: torch.Tensor, targets: torch.Tensor, num_sources: int):
    vals, perms = pit_si_snr_complex_values(outputs, targets, num_sources)
    return float(vals.sum().detach().cpu().item()), int(vals.numel()), perms


def pit_si_snr_complex_values(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int,
) -> tuple[torch.Tensor, list[tuple[int, ...]]]:
    """Return per-sample best PIT SI-SNR_complex values and permutations."""
    preds = _split_sources(outputs, num_sources)
    tgts = _split_sources(targets, num_sources)
    perms = list(permutations(range(num_sources)))
    scores = []
    for perm in perms:
        total = None
        for target_idx, pred_idx in enumerate(perm):
            value = _si_snr_complex_per_item(preds[pred_idx], tgts[target_idx], zero_mean=True)
            total = value if total is None else total + value
        scores.append(total / float(num_sources))
    score_tensor = torch.stack(scores, dim=0)
    best_idx = torch.argmax(score_tensor, dim=0)
    best_vals = score_tensor.gather(0, best_idx.unsqueeze(0)).squeeze(0)
    best_perm = [perms[int(idx)] for idx in best_idx.detach().cpu().tolist()]
    return best_vals, best_perm


def postprocess_outputs(
    outputs: torch.Tensor,
    mixture: torch.Tensor,
    leak_scale: float,
    mc_scale: float,
    coeff_limit: float,
    mc_weight_mode: str,
) -> torch.Tensor:
    processed = outputs
    if abs(float(leak_scale)) > 0.0:
        processed = apply_closed_form_leakage_cancellation(
            processed,
            scale=processed.new_tensor(float(leak_scale)),
            coeff_limit=float(coeff_limit),
            center=True,
            detach_coeffs=True,
        )
    if abs(float(mc_scale)) > 0.0:
        processed = soft_mixture_consistency_projection(
            processed,
            mixture,
            mc_scale=processed.new_tensor(float(mc_scale)),
            weight_mode=mc_weight_mode,
        )
    return processed


def sweep_posthoc_batch(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
    num_sources: int,
    combos: list[tuple[float, float, float]],
    mc_weight_mode: str,
) -> tuple[dict[tuple[float, float, float], tuple[float, int]], tuple[float, int]]:
    baseline_sum, baseline_count, best_perm = metric_sum(outputs, targets, num_sources)
    reordered_outputs = reorder_outputs_by_per_sample_perm(outputs, best_perm, num_sources)

    sums: dict[tuple[float, float, float], tuple[float, int]] = {}
    for leak_scale, mc_scale, coeff_limit in combos:
        processed = postprocess_outputs(
            reordered_outputs,
            mixture,
            leak_scale=leak_scale,
            mc_scale=mc_scale,
            coeff_limit=coeff_limit,
            mc_weight_mode=mc_weight_mode,
        )
        val_sum, val_count, _ = metric_sum(processed, targets, num_sources)
        sums[(leak_scale, mc_scale, coeff_limit)] = (val_sum, val_count)
    return sums, (baseline_sum, baseline_count)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "leak_scale",
        "mc_scale",
        "coeff_limit",
        "si_snr_complex",
        "delta_vs_baseline",
        "num_items",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sweep(args) -> tuple[list[dict], float]:
    set_random_seeds(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_sources = len(args.source_names)
    input_size = args.input_size if args.input_size is not None else get_input_size(args.data_choice)

    cfg = MambaConfig(str(get_model_config_path(args.stage)), train=False)
    model = Create_Mamba_model(cfg, PrintLogger(), input_size_=input_size)
    state_dict, source_key = load_checkpoint_state(Path(args.checkpoint))
    state_dict = maybe_strip_module_prefix(state_dict)
    if args.allow_partial_load:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
            if args.verbose:
                print(f"[WARN] missing keys: {missing[:20]}")
                print(f"[WARN] unexpected keys: {unexpected[:20]}")
    else:
        model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint state '{source_key}' from {args.checkpoint}")
    model.to(device)
    model.eval()

    train_loader, val_loader, snr_loaders = create_data_loaders(
        batch_size=args.batch_size,
        data_choice=args.data_choice,
        num_sources=num_sources,
        matlab_data_root=args.synthetic_root,
        public_data_root=args.public_data_root,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        seed=args.seed,
        split_strategy=args.split_strategy,
    )
    selected_loader = select_loader(args.split, train_loader, val_loader, snr_loaders)

    leak_scales = parse_float_grid(args.leak_scales, default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    mc_scales = parse_float_grid(args.mc_scales, default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    coeff_limits = parse_float_grid(args.coeff_limits, default=[0.05, 0.1, 0.25, 0.5, 1.0])
    combos = [(l, m, c) for l in leak_scales for m in mc_scales for c in coeff_limits]

    totals = {combo: [0.0, 0] for combo in combos}
    baseline_total = 0.0
    baseline_count = 0
    num_batches = 0

    with torch.no_grad():
        for _, batch in iter_eval_batches(selected_loader):
            inputs, targets, _ = split_batch(batch, device)
            outputs = extract_separation_output(model(inputs))
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[-1]
            outputs = outputs.float()
            targets = targets.float()

            batch_sums, baseline = sweep_posthoc_batch(
                outputs=outputs,
                targets=targets,
                mixture=inputs,
                num_sources=num_sources,
                combos=combos,
                mc_weight_mode=args.mc_weight_mode,
            )
            baseline_total += baseline[0]
            baseline_count += baseline[1]
            for combo, (val_sum, val_count) in batch_sums.items():
                totals[combo][0] += val_sum
                totals[combo][1] += val_count
            num_batches += 1
            if args.max_batches is not None and num_batches >= args.max_batches:
                break

    baseline_mean = mean_from_sums(baseline_total, baseline_count)
    rows = []
    for combo, (total, count) in totals.items():
        score = mean_from_sums(total, count)
        rows.append({
            "rank": 0,
            "leak_scale": combo[0],
            "mc_scale": combo[1],
            "coeff_limit": combo[2],
            "si_snr_complex": score,
            "delta_vs_baseline": score - baseline_mean,
            "num_items": count,
        })
    rows.sort(key=lambda row: (row["si_snr_complex"] if math.isfinite(row["si_snr_complex"]) else -1e30), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    write_csv(Path(args.output_csv), rows)
    print(f"Baseline SI-SNR_complex: {baseline_mean:.6f} dB over {baseline_count} source-items")
    if rows:
        best = rows[0]
        print(
            "Best posthoc: "
            f"SI-SNR_complex={best['si_snr_complex']:.6f} dB, "
            f"delta={best['delta_vs_baseline']:+.6f} dB, "
            f"leak_scale={best['leak_scale']}, "
            f"mc_scale={best['mc_scale']}, "
            f"coeff_limit={best['coeff_limit']}"
        )
    print(f"Wrote sweep CSV: {Path(args.output_csv)}")
    return rows, baseline_mean


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline posthoc leakage/MC sweep for IQUMamba checkpoints")
    parser.add_argument("--checkpoint", required=True, help="Path to model weights or full training checkpoint")
    parser.add_argument("--stage", type=int, default=4, help="Model stage used by the checkpoint")
    parser.add_argument("--data_choice", type=str, default="8PSK-A")
    parser.add_argument("--source_names", nargs="+", default=["S1", "S2"])
    parser.add_argument("--synthetic_root", type=str, default=str(ROOT / "data" / "synthetic"))
    parser.add_argument("--public_data_root", type=str, default=None)
    parser.add_argument("--split_strategy", type=str, default="stratified_snr", choices=["random", "stratified_snr"])
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--leak_scales", default=None, help="Comma-separated leak scales, e.g. 0,0.1,0.2,0.5")
    parser.add_argument("--mc_scales", default=None, help="Comma-separated MC scales, e.g. 0,0.02,0.05,0.1")
    parser.add_argument("--coeff_limits", default=None, help="Comma-separated coefficient clipping limits")
    parser.add_argument("--mc_weight_mode", default="uniform", choices=["uniform", "energy"])
    parser.add_argument("--output_csv", default=str(ROOT / "results" / "posthoc_sweep.csv"))
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow non-strict checkpoint loading")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
