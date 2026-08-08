#!/usr/bin/env python3
"""
Dataset Diagnostic Script - 3-Point Sanity Check
=================================================
Usage (run from ANY directory):
    python util/dataset_diag.py --data_root /kaggle/input/iqumamba-data/8PSK-B
    python util/dataset_diag.py --data_root /kaggle/input/iqumamba-data/8PSK-A

Four checks performed:
    [1] Untrained (random-init) model PIT SI-SNR
    [2] Mixture-as-output PIT SI-SNR baseline  (most important sanity check)
    [3] Physical consistency: sum(sources) vs mixture SNR  (detects normalisation bug)
    [4] Signal scale statistics
"""

import argparse
import glob
import math
import os
import sys
from itertools import permutations

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

# Make sure the project root is on sys.path regardless of where the script lives.
# This file lives in   <project_root>/util/dataset_diag.py
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)   # one level up from util/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

EPS = 1e-8


def _si_snr_complex_per_item(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS):
    """SI-SNR with complex scaling (phase-rotation tolerant). pred/target: [B, 2, L]"""
    def to_c(x):
        return torch.complex(x[:, 0, :], x[:, 1, :])

    p = to_c(pred)          # [B, L]
    s = to_c(target)        # [B, L]

    dot = (p * s.conj()).sum(dim=-1)           # complex [B]
    s_energy = (s * s.conj()).sum(dim=-1).real + eps  # real [B]
    alpha = dot / s_energy                            # complex [B]
    s_target = (alpha.unsqueeze(-1) * s)              # [B, L]
    noise = p - s_target
    ratio = (s_target.abs() ** 2).sum(dim=-1) / ((noise.abs() ** 2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)            # [B]


def pit_si_snr_complex(pred: torch.Tensor, target: torch.Tensor, num_sources: int):
    """Per-sample PIT SI-SNR (complex α). Returns scalar mean (dB)."""
    C = pred.shape[1] // num_sources          # channels per source (=2 for I/Q)
    preds   = [pred  [:, k*C:(k+1)*C, :] for k in range(num_sources)]
    targets = [target[:, k*C:(k+1)*C, :] for k in range(num_sources)]

    all_perms = list(permutations(range(num_sources)))
    scores = []
    for perm in all_perms:
        total = None
        for k, p in enumerate(perm):
            v = _si_snr_complex_per_item(preds[p], targets[k])   # [B]
            total = v if total is None else total + v
        scores.append(total / num_sources)
    scores = torch.stack(scores, dim=0)       # [P, B]
    best_per_sample = scores.max(dim=0).values  # [B]
    return float(best_per_sample.mean().item())


def load_mat_file(path: str, key: str):
    """Load a single key from a .mat file (v7.3 or older)."""
    try:
        mat = sio.loadmat(path)
        return mat[key]
    except Exception:
        # Try h5py for v7.3
        import h5py
        with h5py.File(path, 'r') as f:
            arr = f[key][:]
            if arr.ndim == 3:
                arr = arr.transpose(2, 1, 0)   # h5py reverses dims
        return arr


def load_sample_batch(data_root: str, num_sources: int = 2,
                      snr_tag: str = "10dB", batch_size: int = 64,
                      target_key: str = "ideal_frames",
                      mixed_key: str = "mixed_frames"):
    """
    Load a batch of samples from the dataset.
    Returns:
        mixed:  [B, 2, L]  float32 tensor
        target: [B, 2*K, L] float32 tensor
    """
    # Find files matching the SNR tag
    mixed_files  = sorted(glob.glob(os.path.join(data_root, "mixture", f"*{snr_tag}*")))
    target_files = sorted(glob.glob(os.path.join(data_root, "target",  f"*{snr_tag}*")))

    if not mixed_files or not target_files:
        # Try without SNR filter
        mixed_files  = sorted(glob.glob(os.path.join(data_root, "mixture", "*.mat")))
        target_files = sorted(glob.glob(os.path.join(data_root, "target",  "*.mat")))

    assert mixed_files and target_files, (
        f"No .mat files found in {data_root}/mixture or {data_root}/target"
    )

    # Load from the first file
    mixed_raw  = load_mat_file(mixed_files[0],  mixed_key)   # [N, L, 2]
    target_raw = load_mat_file(target_files[0], target_key)  # [N, L, 2*K]

    N = mixed_raw.shape[0]
    B = min(batch_size, N)

    mixed_np  = mixed_raw[:B]   # [B, L, 2]
    target_np = target_raw[:B]  # [B, L, 2*K]

    # Reshape to [B, C, L]
    mixed_t  = torch.tensor(mixed_np,  dtype=torch.float32).permute(0, 2, 1)   # [B, 2, L]
    target_t = torch.tensor(target_np, dtype=torch.float32).permute(0, 2, 1)   # [B, 2*K, L]

    return mixed_t, target_t


# ─────────────────────────────────────────────────────────────────────────────
#  Check 1: Random-initialised model baseline
# ─────────────────────────────────────────────────────────────────────────────

def check1_random_model(mixed: torch.Tensor, target: torch.Tensor,
                        num_sources: int, device: str):
    """
    Evaluate two sub-checks:
      (a) Zero-output baseline — model outputs all zeros for every source.
          Normal expectation: deeply negative (< -20 dB).
          If high, the scale of mixture vs target is grossly inconsistent.
      (b) Random-init model baseline — untrained tiny IQUResUNet1D.
          Normal expectation: roughly the same or slightly above zero-output.
    """
    print("\n" + "="*60)
    print("[CHECK 1] Baseline PIT SI-SNR checks")
    print("="*60)

    B, C, L = mixed.shape

    # --- (a) Zero output ---
    zero_pred = torch.zeros(B, 2 * num_sources, L)
    val_zero = pit_si_snr_complex(zero_pred, target, num_sources)
    print(f"  (a) Zero-output   PIT SI-SNR: {val_zero:+.3f} dB", end="")
    if val_zero > -10.0:
        print("  ⚠️  Too high! Should be << -20 dB.")
    else:
        print("  ✅")

    # --- (b) Random model ---
    val_model = float('nan')
    try:
        from models.IQUResUNet1D import IQUResUNet1D
        import torch.nn as nn

        # U-Net requires input length to be divisible by 2^n_stages (16)
        L_pad = 16 * (L // 16)
        mixed_model = mixed[:, :, :L_pad].to(device)
        
        model = IQUResUNet1D(
            input_size=L_pad,
            input_channels=2,
            n_stages=4,
            features_per_stage=[16, 32, 64, 128],
            conv_op=nn.Conv1d,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[2, 2, 2, 2],
            num_classes=2 * num_sources,
            n_conv_per_stage_decoder=[2, 2, 1, 1],
            deep_supervision=False,
        ).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(mixed_model).cpu()
            
        # If we truncated, we need to match the target length for metric computation
        target_model = target[:, :, :L_pad]
        val_model = pit_si_snr_complex(pred, target_model, num_sources)
        print(f"  (b) Random model  PIT SI-SNR: {val_model:+.3f} dB", end="")
        if val_model > 2.0:
            print("  ⚠️  WARNING: >2 dB — DATA BUG LIKELY!")
        elif val_model < -5.0:
            print("  ✅")
        else:
            print("  ⚠️  Slightly elevated, check other results.")
    except ImportError as e:
        print(f"  (b) Random model  PIT SI-SNR: SKIPPED ({e})")
        print(f"      → Project root not found. Searched: {_PROJECT_ROOT}")

    return val_zero, val_model


# ─────────────────────────────────────────────────────────────────────────────
#  Check 2: Mixture-as-output baseline
# ─────────────────────────────────────────────────────────────────────────────

def check2_mixture_as_output(mixed: torch.Tensor, target: torch.Tensor,
                              num_sources: int):
    """
    Copy the mixture as every source output, then compute PIT SI-SNR.
    If this is high (>>0 dB), the data has either:
      - Very high target-mixture correlation (bad normalisation / data leakage)
      - Or the dataset is naturally very easy (asymmetric bandwidth)
    """
    print("\n" + "="*60)
    print("[CHECK 2] Mixture-as-output PIT SI-SNR baseline")
    print("="*60)

    B, _, L = mixed.shape
    # Repeat mixture for each source output
    pred = mixed.unsqueeze(1).expand(B, num_sources, 2, L).reshape(B, 2 * num_sources, L)

    val = pit_si_snr_complex(pred, target, num_sources)
    print(f"  Mixture-as-output PIT SI-SNR (complex α, per-sample): {val:.3f} dB")

    if val > 3.0:
        print("  ⚠️  WARNING: Very high! The mixture already looks 'similar' to targets.")
        print("       Possible causes:")
        print("       (a) Separate normalisation of mixture vs target (scale bug in MATLAB)")
        print("       (b) The dataset SNR is very high (targets are nearly noise-free in mixture)")
    elif val > 0.0:
        print("  ⚠️  Somewhat elevated — worth comparing against 8PSK-A to see if it's universal.")
    else:
        print("  ✅ Negative dB for mixture-as-output — separation is genuinely non-trivial.")

    return val


# ─────────────────────────────────────────────────────────────────────────────
#  Check 3: Physical consistency — sum of sources vs mixture SNR
# ─────────────────────────────────────────────────────────────────────────────

def check3_physical_consistency(data_root: str, num_sources: int = 2,
                                 snr_tag: str = "10dB", batch_size: int = 64):
    """
    Verify: mixed_baseband ≈ sum(ideal_bb_signals)

    Compute the projection scale alpha = <mix, sum> / <sum, sum>.
    If the data is correctly normalised (same scale factor for mix and target),
    then mix = sum + noise. Since noise is uncorrelated with sum, 
    <mix, sum> ≈ <sum, sum>, so |alpha| should be ≈ 1.0.
    
    If they are independently normalised, |alpha| will deviate heavily from 1.0
    especially at low SNR (where noise dominates the mixture peak).
    """
    print("\n" + "="*60)
    print("[CHECK 3] Physical consistency: Projection Scale Factor")
    print("="*60)

    # Use snr_tag to filter files properly
    mixed_files  = sorted(glob.glob(os.path.join(data_root, "mixture", f"*{snr_tag}*")))
    target_files = sorted(glob.glob(os.path.join(data_root, "target",  f"*{snr_tag}*")))

    if not mixed_files or not target_files:
        print(f"  No files found matching tag '{snr_tag}'. Skipping.")
        return float('nan'), float('nan')

    alphas = []
    observed_snrs = []

    # Sample up to 3 files matching the tag
    for mf, tf in zip(mixed_files[:3], target_files[:3]):
        try:
            snr_str = os.path.basename(mf)

            mixed_raw  = load_mat_file(mf, "mixed_frames")   # [N, L, 2]
            target_raw = load_mat_file(tf, "ideal_frames")   # [N, L, 2*K]

            N = min(batch_size, mixed_raw.shape[0])
            mixed_np  = mixed_raw[:N].astype(np.float32)    # [N, L, 2]
            target_np = target_raw[:N].astype(np.float32)   # [N, L, 2*K]

            # Mixture as complex
            mix_c = mixed_np[:, :, 0] + 1j * mixed_np[:, :, 1]  # [N, L]

            # Sum of sources
            source_sum = np.zeros_like(mix_c)
            for k in range(num_sources):
                s_re = target_np[:, :, 2*k]
                s_im = target_np[:, :, 2*k + 1]
                source_sum += s_re + 1j * s_im

            # Compute alpha = <mix, sum> / <sum, sum> for each sample
            # dot product over L
            num = np.sum(mix_c * np.conj(source_sum), axis=-1)  # [N]
            den = np.sum(np.abs(source_sum)**2, axis=-1) + 1e-12
            alpha = num / den  # [N]
            mean_alpha_mag = float(np.mean(np.abs(alpha)))
            
            # Compute empirical mixture SNR: 10log10( power(mix)/power(mix-sum) )
            err = mix_c - source_sum
            sig_p = np.mean(np.abs(mix_c)**2, axis=-1)
            err_p = np.mean(np.abs(err)**2, axis=-1)
            emp_snr = 10.0 * np.log10(sig_p / (err_p + 1e-12))
            mean_emp_snr = float(np.mean(emp_snr))

            fn = os.path.basename(mf)
            print(f"  File: {fn}")
            print(f"    |alpha| (Projection Scale):  {mean_alpha_mag:.4f}  (Ideal: ~1.0)")
            print(f"    Empirical Mix/Err SNR:       {mean_emp_snr:+.2f} dB")

            alphas.append(mean_alpha_mag)
            observed_snrs.append(mean_emp_snr)

        except Exception as e:
            print(f"  Error loading {mf}: {e}")

    if alphas:
        avg_alpha = float(np.mean(alphas))
        avg_snr = float(np.mean(observed_snrs))
        print(f"\n  Average |alpha|: {avg_alpha:.4f}")
        print(f"  Average Empirical SNR: {avg_snr:+.2f} dB")

        if avg_alpha < 0.6 or avg_alpha > 1.4:
            print("\n  ⚠️  WARNING: |alpha| deviates significantly from 1.0!")
            print("       This confirms the 'file_max_mixed / file_max_ideal' normalisation bug.")
            print("       The targets and mixture are NOT on the same physical scale.")
        else:
            print("\n  ✅ |alpha| is close to 1.0 — normalisation appears physically consistent.")

        return avg_alpha, avg_snr
    return float('nan'), float('nan')


# ─────────────────────────────────────────────────────────────────────────────
#  Check 4: Cross-dataset comparison helper
# ─────────────────────────────────────────────────────────────────────────────

def check4_scale_stats(mixed: torch.Tensor, target: torch.Tensor,
                       num_sources: int):
    """Quick RMS/Max statistics to compare datasets."""
    print("\n" + "="*60)
    print("[CHECK 4] Signal scale statistics (for cross-dataset comparison)")
    print("="*60)

    C = 2 * num_sources
    mix_rms = (mixed ** 2).mean(dim=[1,2]).sqrt().mean().item()
    mix_max = mixed.abs().max().item()

    print(f"  Mixture  RMS: {mix_rms:.6f}   Max: {mix_max:.6f}")

    for k in range(num_sources):
        src = target[:, 2*k:2*k+2, :]
        rms = (src ** 2).mean(dim=[1,2]).sqrt().mean().item()
        mx  = src.abs().max().item()
        print(f"  Source {k+1} RMS: {rms:.6f}   Max: {mx:.6f}")

    # Key ratio: each source RMS vs mixture RMS
    for k in range(num_sources):
        src_rms = (target[:, 2*k:2*k+2, :]**2).mean(dim=[1,2]).sqrt().mean().item()
        ratio = src_rms / (mix_rms + 1e-8)
        print(f"  Source {k+1} / Mixture RMS ratio: {ratio:.4f}")
        if ratio > 1.2:
            print(f"    ⚠️  Source {k+1} RMS is {ratio:.2f}x the mixture — likely independent normalisation!")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset Sanity Check Script")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the dataset (e.g. /kaggle/input/iqumamba-data/8PSK-B)")
    parser.add_argument("--num_sources", type=int, default=2)
    parser.add_argument("--snr_tag", type=str, default="10dB",
                        help="SNR substring to filter files (e.g. '10dB', '30dB')")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of frames to load for evaluation")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for model inference (cpu or cuda)")
    parser.add_argument("--no_model", action="store_true",
                        help="Skip Check 1 (model-based check)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Dataset Diagnostic: 3-Point Sanity Check")
    print("="*60)
    print(f"  Data root:   {args.data_root}")
    print(f"  Num sources: {args.num_sources}")
    print(f"  SNR tag:     {args.snr_tag}")
    print(f"  Batch size:  {args.batch_size}")

    # Load data
    print("\nLoading sample batch...")
    mixed, target = load_sample_batch(
        data_root   = args.data_root,
        num_sources = args.num_sources,
        snr_tag     = args.snr_tag,
        batch_size  = args.batch_size,
    )
    print(f"  Mixed  shape: {tuple(mixed.shape)}")
    print(f"  Target shape: {tuple(target.shape)}")

    # Run checks
    if not args.no_model:
        r1_zero, r1_model = check1_random_model(mixed, target, args.num_sources, args.device)
    else:
        r1_zero, r1_model = float('nan'), float('nan')
        print("\n[CHECK 1] Skipped (--no_model)")

    r2 = check2_mixture_as_output(mixed, target, args.num_sources)
    r3_alpha, r3_snr = check3_physical_consistency(args.data_root, args.num_sources,
                                      args.snr_tag, args.batch_size)
    check4_scale_stats(mixed, target, args.num_sources)

    # Summary
    print("\n" + "="*60)
    print("  DIAGNOSTIC SUMMARY")
    print("="*60)
    def _fmt(v, label, unit="dB"):
        return f"  {label}: {v:+.2f} {unit}" if not math.isnan(v) else f"  {label}: N/A"
    print(_fmt(r1_zero,  "[1a] Zero-output SI-SNR         ", "dB"))
    print(_fmt(r1_model, "[1b] Random model SI-SNR        ", "dB"))
    print(_fmt(r2,       "[2]  Mixture-as-output SI-SNR   ", "dB"))
    print(_fmt(r3_alpha, "[3]  Physical Consistency |alpha|", "  "))

    print()
    fail = False
    if not math.isnan(r1_zero) and r1_zero > -10.0:
        print("  🚨 FAIL [1a]: Zero-output SI-SNR > -10 dB — target scale is inflated vs mixture.")
        fail = True
    if not math.isnan(r1_model) and r1_model > 2.0:
        print("  🚨 FAIL [1b]: Random model > 2 dB — DATA BUG CONFIRMED.")
        fail = True
    if r2 > 3.0:
        print("  🚨 FAIL [2]: Mixture-copy baseline is too high — likely normalisation mismatch.")
        fail = True
    if not math.isnan(r3_alpha) and (r3_alpha < 0.6 or r3_alpha > 1.4):
        print(f"  🚨 FAIL [3]: |alpha| = {r3_alpha:.4f} (Ideal = 1.0).")
        print("       Root cause: MATLAB script used separate file_max_mixed / file_max_ideal.")
        print("       Fix: normalise both with the SAME constant (e.g. file_max_mixed).")
        fail = True
    if not fail:
        print("  ✅ Dataset appears physically consistent.")
        print("     High epoch-1 SI-SNR may be from asymmetric bandwidth, not a data bug.")
    print()


if __name__ == "__main__":
    main()
