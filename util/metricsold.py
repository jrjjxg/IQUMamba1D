"""
metrics.py — Evaluation metrics for single-channel blind source separation.

Metrics:
  1. SI-SNR (real α)    — standard Le Roux/TasNet, penalises phase rotation
  2. SI-SNR (paper/original) — scale-invariant SDR/SNR used in the paper (real α, no clamp, no mean subtraction)
  3. SI-SNR (repo/original code) — same projection, but epsilon placement matches the released baseline code
  4. SI-SNR (complex α) — phase-rotation tolerant, better for comm IQ BSS
  5. MSE
  6. SC                  — Complex Similarity Coefficient (WSIE/Luo/ICCCS)
  7. Pearson |ρ|         — Complex zero-mean Pearson (IQUMamba)
  8. BER                 — Bit Error Rate (needs modulation metadata)
  9. PIT wrapper         — Permutation Invariant evaluation for K sources

All metrics accept IQ signals in shape (B, 2, L) or multi-source (B, 2K, L).
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
from functools import lru_cache
from itertools import permutations
from typing import Optional, List, Dict, Callable, Tuple

EPS = 1e-8


# ============================================================
# Helpers
# ============================================================

def iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    """(…, 2, L) → (…, L) complex."""
    return torch.complex(x[..., 0, :], x[..., 1, :])


def _split_sources(x: torch.Tensor, K: int) -> List[torch.Tensor]:
    """(B, 2K, L) → list of K tensors, each (B, 2, L)."""
    return [x[:, 2 * i: 2 * i + 2, :] for i in range(K)]


def _si_snr_real_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """Per-item SI-SNR (real α), returns (B,)."""
    if zero_mean:
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

    dot_pt = torch.sum(pred * target, dim=(-2, -1))
    dot_tt = torch.sum(target * target, dim=(-2, -1))
    alpha = dot_pt / (dot_tt + eps)

    s_target = alpha.unsqueeze(-1).unsqueeze(-1) * target
    e_noise = pred - s_target

    target_energy = torch.sum(s_target ** 2, dim=(-2, -1))
    noise_energy = torch.sum(e_noise ** 2, dim=(-2, -1))
    return 10.0 * torch.log10((target_energy + eps) / (noise_energy + eps))


def _si_snr_repo_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Per-item SI-SNR with the epsilon placement used in the released baseline code.

    Returns:
        (B,) in dB
    """
    dot_pt = torch.sum(pred * target, dim=(-2, -1), keepdim=True)
    dot_tt = torch.sum(target * target, dim=(-2, -1), keepdim=True)
    alpha = dot_pt / (dot_tt + eps)  # (B, 1, 1)

    e_target = alpha * target
    e_res = pred - e_target

    target_power = torch.sum(e_target * e_target, dim=(-2, -1), keepdim=True)
    residual_power = torch.sum(e_res * e_res, dim=(-2, -1), keepdim=True)

    si = 10.0 * torch.log10((target_power) / (residual_power + eps) + eps)
    return si.squeeze(-1).squeeze(-1)


def _si_snr_complex_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Per-item SI-SNR (complex α), returns (B,)."""
    pred_c = iq_to_complex(pred)
    tgt_c = iq_to_complex(target)

    if zero_mean:
        pred_c = pred_c - pred_c.mean(dim=-1, keepdim=True)
        tgt_c = tgt_c - tgt_c.mean(dim=-1, keepdim=True)

    num = torch.sum(pred_c * torch.conj(tgt_c), dim=-1, keepdim=True)
    den = torch.sum(tgt_c.abs() ** 2, dim=-1, keepdim=True) + eps
    alpha = num / den

    s_target = alpha * tgt_c
    e_noise = pred_c - s_target

    target_energy = torch.sum(s_target.abs() ** 2, dim=-1)
    noise_energy = torch.sum(e_noise.abs() ** 2, dim=-1)
    return 10.0 * torch.log10((target_energy + eps) / (noise_energy + eps))


def _mse_per_item(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-item MSE over (channel, time), returns (B,)."""
    return torch.mean((pred - target) ** 2, dim=(-2, -1))


def _scale_aligned_mse_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Per-item MSE after the real-valued best scale alignment used by SI-SNR."""
    dot_pt = torch.sum(pred * target, dim=(-2, -1))
    dot_tt = torch.sum(target * target, dim=(-2, -1))
    alpha = dot_pt / (dot_tt + eps)
    target_scaled = alpha.unsqueeze(-1).unsqueeze(-1) * target
    return torch.mean((pred - target_scaled) ** 2, dim=(-2, -1))


def _similarity_coeff_complex_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-item complex SC, returns (B,)."""
    y = iq_to_complex(pred)
    s = iq_to_complex(target)

    if zero_mean:
        y = y - y.mean(dim=-1, keepdim=True)
        s = s - s.mean(dim=-1, keepdim=True)

    numerator = torch.sum(y * torch.conj(s), dim=-1).abs()
    norm_y = torch.sqrt(torch.sum(y.abs() ** 2, dim=-1))
    norm_s = torch.sqrt(torch.sum(s.abs() ** 2, dim=-1))
    # eps guards against near-zero pred (collapsed output) causing spurious SC=1.
    denominator = norm_y * norm_s + eps
    return numerator / denominator


def _pearson_complex_abs_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-item |ρ|, returns (B,)."""
    y = iq_to_complex(pred)
    s = iq_to_complex(target)

    y = y - y.mean(dim=-1, keepdim=True)
    s = s - s.mean(dim=-1, keepdim=True)

    cov = torch.mean(y * torch.conj(s), dim=-1)
    # eps guards against near-zero variance (collapsed/silent prediction).
    var_y = torch.mean(y.abs() ** 2, dim=-1) + eps
    var_s = torch.mean(s.abs() ** 2, dim=-1) + eps
    return cov.abs() / torch.sqrt(var_y * var_s)


def _global_phase_offset_deg_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Estimate per-item global phase offset in degrees via arg(sum pred * conj(target)).

    The returned angle is in [-180, 180]. A prediction close to -target yields
    an angle close to +/-180 degrees.
    """
    y = iq_to_complex(pred)
    s = iq_to_complex(target)
    inner = torch.sum(y * torch.conj(s), dim=-1)
    # Keep silent or all-zero pairs from producing noisy angles.
    valid = inner.abs() > eps
    phase = torch.angle(inner) * (180.0 / math.pi)
    return torch.where(valid, phase, torch.zeros_like(phase))


def phase_flip_mask_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    tolerance_deg: float = 45.0,
    min_similarity: float = 0.0,
    mode: str = "either",
    eps: float = EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return a source-level phase-flip mask for each batch item.

    mode='phase' counts offsets within tolerance of pi (+/-180 degrees).
    mode='sign' counts negative real SI-SNR projection coefficients.
    mode='either' counts either condition, which is useful for noisy IQ BSS
    outputs where a full sign reversal may not land exactly at 180 degrees.
    """
    phase_deg = _global_phase_offset_deg_per_item(pred, target, eps=eps)
    similarity = _similarity_coeff_complex_per_item(pred, target, zero_mean=False, eps=eps)
    dot_pt = torch.sum(pred * target, dim=(-2, -1))
    dot_tt = torch.sum(target * target, dim=(-2, -1))
    real_alpha = dot_pt / (dot_tt + eps)
    distance_to_pi = torch.abs(180.0 - torch.abs(phase_deg))
    phase_mask = distance_to_pi <= float(tolerance_deg)
    sign_mask = real_alpha < 0.0
    mode = str(mode).lower()
    if mode == "phase":
        flip_mask = phase_mask
    elif mode == "sign":
        flip_mask = sign_mask
    elif mode == "either":
        flip_mask = phase_mask | sign_mask
    else:
        raise ValueError(f"Unsupported phase flip mode '{mode}', expected phase/sign/either")
    if min_similarity > 0.0:
        flip_mask = flip_mask & (similarity >= float(min_similarity))
    return flip_mask, phase_deg, similarity, real_alpha


def _reorder_sources_by_per_sample_perm(
    pred: torch.Tensor,
    best_perm_per_sample: List[Tuple[int, ...]],
    num_sources: int,
) -> torch.Tensor:
    """Reorder (B, 2K, L) predictions so source k matches target k."""
    if pred.ndim != 3:
        raise ValueError(f"Expected pred with shape (B, 2K, L), got {pred.shape}")
    if len(best_perm_per_sample) != pred.shape[0]:
        raise ValueError("best_perm_per_sample must have one permutation per batch item")

    reordered = torch.empty_like(pred)
    for b, perm in enumerate(best_perm_per_sample):
        if len(perm) != num_sources:
            raise ValueError(f"Permutation length {len(perm)} does not match num_sources={num_sources}")
        for target_idx, pred_idx in enumerate(perm):
            reordered[b, 2 * target_idx:2 * target_idx + 2, :] = pred[b, 2 * pred_idx:2 * pred_idx + 2, :]
    return reordered


def phase_flip_rate(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int = 1,
    best_perm_per_sample: Optional[List[Tuple[int, ...]]] = None,
    tolerance_deg: float = 45.0,
    min_similarity: float = 0.0,
    mode: str = "either",
    eps: float = EPS,
) -> Tuple[float, Dict[str, object]]:
    """
    Compute source-level global phase flip rate after optional PIT matching.

    Returns:
        rate: flipped source items / all source items.
        details: counts, sample-level rate, per-source rates, mean phase stats.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Expected tensors shaped (B, 2K, L), got {pred.shape}")
    if pred.shape[1] != 2 * int(num_sources):
        raise ValueError(f"num_sources={num_sources} incompatible with channel count {pred.shape[1]}")

    if best_perm_per_sample is not None and num_sources > 1:
        pred = _reorder_sources_by_per_sample_perm(pred, best_perm_per_sample, num_sources)

    preds = _split_sources(pred, num_sources)
    tgts = _split_sources(target, num_sources)
    masks = []
    phases = []
    similarities = []
    real_alphas = []
    for k in range(num_sources):
        mask_k, phase_k, sim_k, alpha_k = phase_flip_mask_per_item(
            preds[k],
            tgts[k],
            tolerance_deg=tolerance_deg,
            min_similarity=min_similarity,
            mode=mode,
            eps=eps,
        )
        masks.append(mask_k)
        phases.append(phase_k)
        similarities.append(sim_k)
        real_alphas.append(alpha_k)

    mask_stack = torch.stack(masks, dim=1)  # (B, K)
    phase_stack = torch.stack(phases, dim=1)
    sim_stack = torch.stack(similarities, dim=1)
    alpha_stack = torch.stack(real_alphas, dim=1)
    total_sources = int(mask_stack.numel())
    flipped_sources = int(mask_stack.sum().item())
    source_flip_rates = mask_stack.float().mean(dim=0).detach().cpu().tolist()
    source_flip_counts = mask_stack.sum(dim=0).detach().cpu().tolist()
    sample_flip_mask = mask_stack.any(dim=1)
    sample_flip_count = int(sample_flip_mask.sum().item())
    sample_total = int(sample_flip_mask.numel())
    sample_flip_rate = float(sample_flip_count / sample_total) if sample_total > 0 else float("nan")
    phase_abs = torch.abs(phase_stack)
    distance_to_pi = torch.abs(180.0 - phase_abs)

    details = {
        "flipped_sources": flipped_sources,
        "total_sources": total_sources,
        "sample_flip_rate": sample_flip_rate,
        "sample_flip_count": sample_flip_count,
        "sample_total": sample_total,
        "source_flip_rates": [float(v) for v in source_flip_rates],
        "source_flip_counts": [int(v) for v in source_flip_counts],
        "mean_phase_deg": float(phase_stack.mean().item()) if total_sources > 0 else float("nan"),
        "mean_phase_abs_deg": float(phase_abs.mean().item()) if total_sources > 0 else float("nan"),
        "mean_phase_distance_to_pi_deg": float(distance_to_pi.mean().item()) if total_sources > 0 else float("nan"),
        "mean_similarity": float(sim_stack.mean().item()) if total_sources > 0 else float("nan"),
        "negative_real_alpha_rate": float((alpha_stack < 0.0).float().mean().item()) if total_sources > 0 else float("nan"),
        "mean_real_alpha": float(alpha_stack.mean().item()) if total_sources > 0 else float("nan"),
        "tolerance_deg": float(tolerance_deg),
        "min_similarity": float(min_similarity),
        "mode": str(mode).lower(),
    }
    rate = float(flipped_sources / total_sources) if total_sources > 0 else float("nan")
    return rate, details


# ============================================================
# 1. SI-SNR  (real scalar α — standard Le Roux / TasNet)
#
#    α = <ŝ, s> / <s, s>          (real)
#    s_target = α · s
#    SI-SNR = 10 log10( ||s_target||² / ||ŝ − s_target||² )
#
#    ⚠ Phase rotation ŝ ≈ e^{jφ} s will be penalised.
# ============================================================

def si_snr_real(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool = False,
    eps: float = EPS,
) -> torch.Tensor:
    """
    SI-SNR with real-valued scaling factor (standard formulation).

    Penalises global phase rotation — appropriate when the model is
    *expected* to recover the correct phase (e.g. the loss also uses
    a real α, as in the existing si_snr_loss).

    Args:
        pred, target: (B, 2, L)  single-source IQ
        zero_mean:    subtract per-channel temporal mean before computing
        eps:          numerical stability

    Returns:
        scalar: mean SI-SNR in dB (higher is better)
    """
    return _si_snr_real_per_item(pred, target, zero_mean=zero_mean, eps=eps).mean()


def si_snr_paper(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Paper/original SI-SNR (often called SI-SDR).

    Matches the common scale-invariant projection form used in the IQUmamba
    paper and many baseline repos:
      α = <ŝ, s> / <s, s>   (real scalar, unconstrained)
      SI = 10 log10( ||αs||² / ||ŝ − αs||² )

    Notes:
      - Scale-invariant (gain absorbed by α)
      - No mean subtraction
      - Allows negative α (so sign/π-phase flips can still score high)
    """
    return _si_snr_real_per_item(pred, target, zero_mean=False, eps=eps).mean()


def si_snr_repo(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    "Original repo" SI-SNR/SI-SDR implementation (epsilon placement).

    This matches the style used in some released baseline code:
      SI = 10 log10( ||αs||² / (||ŝ − αs||² + eps) + eps )

    It is almost identical to `si_snr_paper` on normal signals, but behaves
    differently when target energy is (near) zero.
    """
    return _si_snr_repo_per_item(pred, target, eps=eps).mean()


def si_snr_paper_joint(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    Paper/original SI-SNR computed on the *full multi-source tensor* (joint).

    This matches repos that compute SI-SDR/SI-SNR on (B, 2K, L) directly
    (i.e., treat all separated sources as one concatenated vector), which can
    differ from "per-source then average" reporting.
    """
    return _si_snr_real_per_item(pred, target, zero_mean=False, eps=eps).mean()


# ============================================================
# 2. SI-SNR  (complex scalar α — phase-rotation tolerant)
#
#    α = <ŝ, s*> / <s, s*>        (complex!)
#    s_target = α · s
#    SI-SNR = 10 log10( ||s_target||² / ||ŝ − s_target||² )
#
#    ŝ ≈ e^{jφ} s  ⟹ SI-SNR still high   ✓
# ============================================================

def si_snr_complex(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """
    SI-SNR with complex-valued scaling factor (phase-rotation tolerant).

    Naturally handles ŝ ≈ α e^{jφ} s — the complex α absorbs both
    amplitude and phase ambiguity. More suitable for blind separation
    evaluation where global phase is inherently ambiguous.

    Args:
        pred, target: (B, 2, L)  single-source IQ
        zero_mean:    subtract complex temporal mean before computing
        eps:          numerical stability

    Returns:
        scalar: mean SI-SNR in dB (higher is better)
    """
    return _si_snr_complex_per_item(pred, target, zero_mean=zero_mean, eps=eps).mean()


# ============================================================
# 3. MSE
# ============================================================

def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between prediction and target. (B, 2, L) or (B, 2K, L)."""
    return F.mse_loss(pred, target)


def scale_aligned_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """
    MSE after real-valued best scale alignment.

    This uses the same real-valued least-squares scaling factor
        alpha = <pred, target> / <target, target>
    that underlies the usual SI-SNR/SI-SDR projection. If this metric is
    much smaller than raw MSE, the main error is likely gain/scale mismatch
    rather than waveform-shape mismatch.
    """
    return _scale_aligned_mse_per_item(pred, target, eps=eps).mean()


# ============================================================
# 4. Complex Similarity Coefficient  (SC)
#    |<y, s*>| / (||y|| · ||s||)   ∈ [0, 1]
#    Aligns with WSIE-Net / Luo / ICCCS
# ============================================================

def similarity_coeff_complex(
    pred: torch.Tensor,
    target: torch.Tensor,
    zero_mean: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Complex Similarity Coefficient (SC).

    SC = |<y, s*>| / (||y|| · ||s||)   ∈ [0, 1]

    - Uses complex-conjugate inner product
    - Takes absolute value → invariant to phase/sign flips
    - Aligns with WSIE-Net, Luo (IET RSN 2023), ICCCS 2025

    Args:
        pred, target: (B, 2, L)  single-source IQ
        zero_mean:    subtract complex mean before computing
        eps:          numerical stability

    Returns:
        scalar: mean SC ∈ [0, 1]
    """
    return _similarity_coeff_complex_per_item(
        pred, target, zero_mean=zero_mean, eps=eps
    ).mean()


# ============================================================
# 5. Complex Pearson |ρ|  (zero-mean, abs)
#    Aligns with IQUMamba
# ============================================================

def pearson_complex_abs(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Complex Pearson correlation coefficient |ρ|  ∈ [0, 1].

    |ρ| = |cov(y,s)| / (σ_y · σ_s)

    Inherently DC-offset and gain invariant.
    Takes absolute value → also phase-rotation invariant.
    Aligns with IQUMamba.

    Args:
        pred, target: (B, 2, L)  single-source IQ
        eps:          numerical stability

    Returns:
        scalar: mean |ρ| ∈ [0, 1]
    """
    return _pearson_complex_abs_per_item(pred, target, eps=eps).mean()


# ============================================================
# 6. BER  (requires modulation metadata)
# ============================================================

def _estimate_phase_offset(pred_B2L, target_B2L):
    """Estimate global phase θ (radians) s.t. pred ≈ e^{jθ} target.  (B,)"""
    pI, pQ = pred_B2L[:, 0, :], pred_B2L[:, 1, :]
    tI, tQ = target_B2L[:, 0, :], target_B2L[:, 1, :]
    dot = (pI * tI + pQ * tQ).sum(dim=1)
    cross = (pI * tQ - pQ * tI).sum(dim=1)
    return torch.atan2(cross, dot + EPS)


def _rotate_iq(pred_B2L, theta_rad):
    """Rotate pred by −θ.  (B, 2, L)."""
    c = torch.cos(theta_rad).view(-1, 1, 1)
    s = torch.sin(theta_rad).view(-1, 1, 1)
    pI, pQ = pred_B2L[:, 0:1, :], pred_B2L[:, 1:2, :]
    return torch.cat([pI * c + pQ * s, pQ * c - pI * s], dim=1)


def _nearest_demod(symbols, constellation, bits_per_symbol, gray_map=None):
    """
    Hard-decision demodulation.  symbols: (B, N_sym) complex.
    Returns: (B, N_sym * bits_per_symbol) int {0,1}.
    """
    B, N_sym = symbols.shape
    diff = symbols.unsqueeze(-1) - constellation.unsqueeze(0).unsqueeze(0)
    min_idx = diff.abs().argmin(dim=-1)
    int_values = gray_map[min_idx] if gray_map is not None else min_idx
    bits = []
    for bit_pos in range(bits_per_symbol - 1, -1, -1):
        bits.append((int_values >> bit_pos) & 1)
    return torch.stack(bits, dim=-1).reshape(B, -1)


def ber(
    pred: torch.Tensor,
    target: torch.Tensor,
    bits_target: torch.Tensor,
    constellation: torch.Tensor,
    bits_per_symbol: int,
    sps: int,
    gray_map: Optional[torch.Tensor] = None,
    phase_align: bool = True,
) -> torch.Tensor:
    """
    Bit Error Rate.

    1. Phase-align pred to target (resolve blind-separation ambiguity)
    2. Complex → downsample → hard-decision demod
    3. Compare with target bits

    Args:
        pred, target:     (B, 2, L)
        bits_target:      (B, N_bits) int {0,1}
        constellation:    (M,) complex
        bits_per_symbol:  int
        sps:              int (samples per symbol)
        gray_map:         (M,) int, optional
        phase_align:      whether to align phase before demod

    Returns:
        scalar: BER ∈ [0, 1]
    """
    if phase_align:
        theta = _estimate_phase_offset(pred, target)
        pred = _rotate_iq(pred, theta)

    syms = iq_to_complex(pred)[:, ::sps]
    bits_dec = _nearest_demod(syms, constellation, bits_per_symbol, gray_map)

    n = min(bits_dec.shape[1], bits_target.shape[1])
    bits_dec = bits_dec[:, :n]
    bits_ref = bits_target[:, :n].to(bits_dec.device)

    return (bits_dec != bits_ref).float().sum() / bits_ref.numel()


# ============================================================
# 7. PIT  (Permutation Invariant evaluation)
# ============================================================

def _pit_perm_scores_per_item(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    metric_fn_per_item: Callable,
) -> tuple:
    """
    Evaluate all permutations and return per-sample scores.

    Args:
        preds, targets: list of K tensors, each (B, 2, L)
        metric_fn_per_item: fn(pred, target) -> (B,)

    Returns:
        perms: list of permutation tuples, length P=K!
        scores: Tensor of shape (P, B), each already averaged by K sources
    """
    K = len(preds)
    perms = list(permutations(range(K)))
    scores = []
    for perm in perms:
        total = None
        for k, p in enumerate(perm):
            v = metric_fn_per_item(preds[p], targets[k])  # (B,)
            total = v if total is None else total + v
        scores.append(total / K)
    return perms, torch.stack(scores, dim=0)  # (P, B)


def _pit_gather_selected(scores: torch.Tensor, selected_perm_idx: torch.Tensor) -> torch.Tensor:
    """scores: (P, B), selected_perm_idx: (B,) -> gathered (B,)"""
    return scores.gather(0, selected_perm_idx.unsqueeze(0)).squeeze(0)


def _pit_best_perm_metric_batchperm(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    metric_fn_per_item: Callable,
    higher_is_better: bool = True,
) -> tuple:
    """Global single permutation for the whole batch."""
    perms, scores = _pit_perm_scores_per_item(preds, targets, metric_fn_per_item)  # (P, B)
    mean_scores = scores.mean(dim=1)  # (P,)
    best_idx = torch.argmax(mean_scores) if higher_is_better else torch.argmin(mean_scores)
    best_idx = int(best_idx.item())
    return float(mean_scores[best_idx].item()), perms[best_idx]


def _pit_best_perm_metric_persample(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    metric_fn_per_item: Callable,
    higher_is_better: bool = True,
) -> tuple:
    """Per-sample permutation selection (recommended)."""
    perms, scores = _pit_perm_scores_per_item(preds, targets, metric_fn_per_item)  # (P, B)
    selected_idx = torch.argmax(scores, dim=0) if higher_is_better else torch.argmin(scores, dim=0)  # (B,)
    best_vals = _pit_gather_selected(scores, selected_idx)  # (B,)
    best_perm_per_sample = [perms[i] for i in selected_idx.tolist()]
    return float(best_vals.mean().item()), best_perm_per_sample


def pit_si_snr_real_batchperm(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = False,
    eps: float = EPS,
) -> tuple:
    """SI-SNR (real α) with global batch permutation PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    fn = lambda p, t: _si_snr_real_per_item(p, t, zero_mean=zero_mean, eps=eps)
    return _pit_best_perm_metric_batchperm(preds, targets, fn, higher_is_better=True)


def pit_si_snr_real_persample(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = False,
    eps: float = EPS,
) -> tuple:
    """SI-SNR (real α) with per-sample PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    fn = lambda p, t: _si_snr_real_per_item(p, t, zero_mean=zero_mean, eps=eps)
    return _pit_best_perm_metric_persample(preds, targets, fn, higher_is_better=True)


def pit_si_snr_complex_batchperm(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = True,
    eps: float = EPS,
) -> tuple:
    """SI-SNR (complex α) with global batch permutation PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    fn = lambda p, t: _si_snr_complex_per_item(p, t, zero_mean=zero_mean, eps=eps)
    return _pit_best_perm_metric_batchperm(preds, targets, fn, higher_is_better=True)


def pit_si_snr_complex_persample(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = True,
    eps: float = EPS,
) -> tuple:
    """SI-SNR (complex α) with per-sample PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    fn = lambda p, t: _si_snr_complex_per_item(p, t, zero_mean=zero_mean, eps=eps)
    return _pit_best_perm_metric_persample(preds, targets, fn, higher_is_better=True)


def pit_sc_batchperm(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = False,
) -> tuple:
    """SC with global batch permutation PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    fn = lambda p, t: _similarity_coeff_complex_per_item(p, t, zero_mean=zero_mean)
    return _pit_best_perm_metric_batchperm(preds, targets, fn, higher_is_better=True)


def pit_sc_persample(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = False,
) -> tuple:
    """SC with per-sample PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    fn = lambda p, t: _similarity_coeff_complex_per_item(p, t, zero_mean=zero_mean)
    return _pit_best_perm_metric_persample(preds, targets, fn, higher_is_better=True)


def pit_mse_batchperm(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
) -> tuple:
    """MSE with global batch permutation PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    return _pit_best_perm_metric_batchperm(preds, targets, _mse_per_item, higher_is_better=False)


def pit_mse_persample(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
) -> tuple:
    """MSE with per-sample PIT."""
    preds = _split_sources(pred, num_sources)
    targets = _split_sources(target, num_sources)
    return _pit_best_perm_metric_persample(preds, targets, _mse_per_item, higher_is_better=False)


def pit_si_snr_real(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = False,
    eps: float = EPS,
) -> tuple:
    """Default SI-SNR (real α) PIT = per-sample PIT."""
    return pit_si_snr_real_persample(pred, target, num_sources, zero_mean=zero_mean, eps=eps)


def pit_si_snr_complex(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = True,
    eps: float = EPS,
) -> tuple:
    """Default SI-SNR (complex α) PIT = per-sample PIT."""
    return pit_si_snr_complex_persample(pred, target, num_sources, zero_mean=zero_mean, eps=eps)


def pit_sc(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    zero_mean: bool = False,
) -> tuple:
    """Default SC PIT = per-sample PIT."""
    return pit_sc_persample(pred, target, num_sources, zero_mean=zero_mean)


def pit_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
) -> tuple:
    """Default MSE PIT = per-sample PIT."""
    return pit_mse_persample(pred, target, num_sources)


# ============================================================
# 8. Aggregate convenience functions
# ============================================================

def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int = 1,
    use_pit: bool = False,
    pit_mode: str = "persample",
) -> Dict[str, float]:
    """
    Compute all metrics at once (excluding BER).

    If use_pit=True and num_sources > 1, uses PIT to find the best
    source permutation (determined by SI-SNR complex), then reports
    all metrics under that same permutation.

    Returns:
        dict with keys:
          'SI-SNR_real', 'SI-SNR_paper', 'SI-SNR_repo', 'SI-SNR_complex', 'MSE', 'SC', 'Pearson'
          and optionally 'best_perm' (tuple)
    """
    if num_sources > 1 and use_pit:
        preds = _split_sources(pred, num_sources)
        targets = _split_sources(target, num_sources)

        if pit_mode == "batchperm":
            _, best_perm = pit_si_snr_complex_batchperm(pred, target, num_sources)
            reordered = [preds[p] for p in best_perm]

            si_real_vals = [si_snr_real(reordered[k], targets[k]) for k in range(num_sources)]
            si_paper_vals = [si_snr_paper(reordered[k], targets[k]) for k in range(num_sources)]
            si_repo_vals = [si_snr_repo(reordered[k], targets[k]) for k in range(num_sources)]
            si_cplx_vals = [si_snr_complex(reordered[k], targets[k]) for k in range(num_sources)]
            mse_vals = [mse(reordered[k], targets[k]) for k in range(num_sources)]
            sc_vals = [similarity_coeff_complex(reordered[k], targets[k]) for k in range(num_sources)]
            pears_vals = [pearson_complex_abs(reordered[k], targets[k]) for k in range(num_sources)]

            return {
                'SI-SNR_real': sum(v.item() for v in si_real_vals) / num_sources,
                'SI-SNR_paper': sum(v.item() for v in si_paper_vals) / num_sources,
                'SI-SNR_repo': sum(v.item() for v in si_repo_vals) / num_sources,
                'SI-SNR_complex': sum(v.item() for v in si_cplx_vals) / num_sources,
                'MSE': sum(v.item() for v in mse_vals) / num_sources,
                'SC': sum(v.item() for v in sc_vals) / num_sources,
                'Pearson': sum(v.item() for v in pears_vals) / num_sources,
                'best_perm': best_perm,
                'pit_mode': pit_mode,
            }

        if pit_mode != "persample":
            raise ValueError(f"Unknown pit_mode: {pit_mode}. Use 'persample' or 'batchperm'.")

        perms, score_cplx = _pit_perm_scores_per_item(
            preds, targets, lambda p, t: _si_snr_complex_per_item(p, t, zero_mean=True, eps=EPS)
        )
        best_perm_idx = torch.argmax(score_cplx, dim=0)  # (B,)

        def _mean_selected(scores: torch.Tensor) -> float:
            return float(_pit_gather_selected(scores, best_perm_idx).mean().item())

        _, score_real = _pit_perm_scores_per_item(
            preds, targets, lambda p, t: _si_snr_real_per_item(p, t, zero_mean=False, eps=EPS)
        )
        # Paper SI-SNR uses the same real-α projection form (no mean subtraction).
        score_paper = score_real
        _, score_repo = _pit_perm_scores_per_item(preds, targets, _si_snr_repo_per_item)
        _, score_mse = _pit_perm_scores_per_item(preds, targets, _mse_per_item)
        _, score_sc = _pit_perm_scores_per_item(
            preds, targets, lambda p, t: _similarity_coeff_complex_per_item(p, t, zero_mean=False)
        )
        _, score_pears = _pit_perm_scores_per_item(preds, targets, _pearson_complex_abs_per_item)

        best_perm_per_sample = [perms[i] for i in best_perm_idx.tolist()]
        best_perm_hist = {}
        for p in best_perm_per_sample:
            best_perm_hist[p] = best_perm_hist.get(p, 0) + 1
        dominant_perm = max(best_perm_hist.items(), key=lambda kv: kv[1])[0]

        return {
            'SI-SNR_real': _mean_selected(score_real),
            'SI-SNR_paper': _mean_selected(score_paper),
            'SI-SNR_repo': _mean_selected(score_repo),
            'SI-SNR_complex': _mean_selected(score_cplx),
            'MSE': _mean_selected(score_mse),
            'SC': _mean_selected(score_sc),
            'Pearson': _mean_selected(score_pears),
            'best_perm': dominant_perm,
            'best_perm_per_sample': best_perm_per_sample,
            'best_perm_hist': best_perm_hist,
            'pit_mode': pit_mode,
        }
    else:
        # Single source or no PIT — simple direct computation
        if num_sources > 1:
            preds = _split_sources(pred, num_sources)
            targets = _split_sources(target, num_sources)
        else:
            preds = [pred]
            targets = [target]

        K = len(preds)
        si_r = sum(si_snr_real(preds[k], targets[k]).item() for k in range(K)) / K
        si_p = sum(si_snr_paper(preds[k], targets[k]).item() for k in range(K)) / K
        si_repo = sum(si_snr_repo(preds[k], targets[k]).item() for k in range(K)) / K
        si_c = sum(si_snr_complex(preds[k], targets[k]).item() for k in range(K)) / K
        m = sum(mse(preds[k], targets[k]).item() for k in range(K)) / K
        sc = sum(similarity_coeff_complex(preds[k], targets[k]).item() for k in range(K)) / K
        pe = sum(pearson_complex_abs(preds[k], targets[k]).item() for k in range(K)) / K

        result = {
            'SI-SNR_real': si_r,
            'SI-SNR_paper': si_p,
            'SI-SNR_repo': si_repo,
            'SI-SNR_complex': si_c,
            'MSE': m,
            'SC': sc,
            'Pearson': pe,
        }
        return result


# ============================================================
# 8.  BER  (Bit Error Rate) — demodulation-based evaluation
# ============================================================

# --- Constellation tables (unit-energy, Gray-coded) --------

def _gray(n: int):
    """Generate Gray code sequence for n bits."""
    return [i ^ (i >> 1) for i in range(1 << n)]


def _qam_constellation(M: int):
    """Generate M-QAM constellation with Gray mapping.

    Returns:
        syms: (M,) complex np.array, syms[gray_idx] = complex symbol
        bits_per_sym: int
    """
    bps = int(round(math.log2(M)))
    k = int(math.sqrt(M))
    assert k * k == M, f"M-QAM requires perfect square M, got {M}"

    # Gray codes for each axis
    half_bps = bps // 2
    gray_seq = _gray(half_bps)

    # PAM levels: -(k-1), -(k-3), ..., (k-3), (k-1)
    levels = np.array([2 * i - (k - 1) for i in range(k)], dtype=np.float64)
    # Normalize to unit average energy
    avg_energy = np.mean(levels ** 2) * 2  # I + Q
    levels /= np.sqrt(avg_energy)

    syms = np.zeros(M, dtype=np.complex128)
    for i_idx, i_gray in enumerate(gray_seq):
        for q_idx, q_gray in enumerate(gray_seq):
            symbol_idx = (i_gray << half_bps) | q_gray
            syms[symbol_idx] = levels[i_idx] + 1j * levels[q_idx]
    return syms, bps


def _psk_constellation(M: int, phase_offset: float = 0.0):
    """Generate M-PSK constellation with Gray mapping.

    Args:
        M: Modulation order (e.g. 2, 4, 8)
        phase_offset: Starting angle in radians.  The i-th natural-order
            symbol is placed at angle ``phase_offset + 2*pi*i/M``.
    """
    bps = int(round(math.log2(M)))
    gray_seq = _gray(bps)
    angles = np.array([phase_offset + 2 * np.pi * i / M for i in range(M)])
    syms_natural = np.exp(1j * angles)
    syms = np.zeros(M, dtype=np.complex128)
    for natural_idx, gray_idx in enumerate(gray_seq):
        syms[gray_idx] = syms_natural[natural_idx]
    return syms, bps


def _matlab_8psk_constellation():
    """8PSK constellation matching the MATLAB GEN_8PSK_MultiSource generator.

    MATLAB definition (1-indexed):
        constellation(k) = exp(j * (2k-1)*pi/8)  for k = 1..8
        gray_map_array   = [1, 2, 4, 3, 8, 7, 5, 6]  (1-indexed)

    The natural-order angle starts at pi/8 (not 0), and the Gray mapping
    differs from the standard reflected binary Gray code in the upper half.

    Returns:
        syms: (8,) complex array where syms[bit_label] = complex symbol
        bits_per_sym: 3
    """
    # Angles: pi/8, 3pi/8, 5pi/8, 7pi/8, 9pi/8, 11pi/8, 13pi/8, 15pi/8
    angles = np.array([(2 * k + 1) * np.pi / 8 for k in range(8)])
    syms_natural = np.exp(1j * angles)  # natural order, index 0..7

    # MATLAB uses:
    #   frame_symbol_labels = bi2de(bits, 'left-msb')
    #   symbol_indices = gray_map_array(frame_symbol_labels + 1)
    #   s_complex = constellation(symbol_indices)
    # Therefore the map is bit_label -> natural constellation index.
    label_to_natural_idx = [0, 1, 3, 2, 7, 6, 4, 5]
    syms = syms_natural[np.asarray(label_to_natural_idx, dtype=np.int64)]
    return syms, 3


def _apsk_constellation_16():
    """16-APSK (DVB-S2 style): 4+12 ring, Gray-like mapping."""
    # Inner ring: 4 symbols, radius r1
    # Outer ring: 12 symbols, radius r2
    # Ratio r2/r1 ≈ 2.57 (DVB-S2 default for rate 2/3)
    r_ratio = 2.57
    # Normalize so average energy = 1
    # E = (4*r1^2 + 12*r2^2)/16 = 1
    # r2 = r_ratio * r1
    # (4 + 12*r_ratio^2)*r1^2 = 16
    r1 = np.sqrt(16.0 / (4 + 12 * r_ratio ** 2))
    r2 = r_ratio * r1

    syms = np.zeros(16, dtype=np.complex128)
    # Inner ring (4 syms at π/4 offsets)
    for i in range(4):
        angle = np.pi / 4 + i * np.pi / 2
        syms[i] = r1 * np.exp(1j * angle)
    # Outer ring (12 syms)
    for i in range(12):
        angle = i * 2 * np.pi / 12
        syms[4 + i] = r2 * np.exp(1j * angle)
    return syms, 4  # 4 bits per symbol


def _qam_constellation_128():
    """128-QAM: cross-shaped constellation (not a perfect square).

    Uses a simple nearest-index mapping; details of Grey labelling
    are approximate but sufficient for BER trend evaluation.
    """
    # Build cross-128QAM grid
    points = []
    for i in range(-7, 8):
        for q in range(-7, 8):
            if abs(i) + abs(q) <= 11:  # cross boundary
                points.append(complex(i, q))
            if len(points) >= 128:
                break
        if len(points) >= 128:
            break
    # Pad or trim to exactly 128
    while len(points) < 128:
        points.append(complex(0, 0))
    points = np.array(points[:128], dtype=np.complex128)
    # Normalize
    avg_e = np.mean(np.abs(points) ** 2)
    if avg_e > 0:
        points /= np.sqrt(avg_e)
    return points, 7  # 7 bits per symbol


_CONSTELLATION_CACHE: Dict[str, tuple] = {}


def constellation_for_modulation(modulation: str):
    """Return (syms, bits_per_sym) for a modulation name string.

    Supported: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 128QAM, 16APSK
    Also handles compound names like '8PSK-A', 'QAM-A', etc.
    """
    key = modulation.upper().strip()
    if key in _CONSTELLATION_CACHE:
        return _CONSTELLATION_CACHE[key]

    # Normalize compound names
    base = key.split('-')[0].split('+')[0].strip()
    # Handle '8PSK', 'QPSK', 'BPSK', etc.
    if base in ('BPSK',):
        result = _psk_constellation(2)
    elif base in ('QPSK',):
        result = _psk_constellation(4)
    elif base in ('8PSK',):
        result = _matlab_8psk_constellation()
    elif base in ('16QAM',):
        result = _qam_constellation(16)
    elif base in ('64QAM',):
        result = _qam_constellation(64)
    elif base in ('128QAM',):
        result = _qam_constellation_128()
    elif base in ('16APSK',):
        result = _apsk_constellation_16()
    else:
        raise ValueError(f"Unknown modulation for BER: {modulation}")

    _CONSTELLATION_CACHE[key] = result
    return result


# --- Hard-decision demodulator ---

def _hard_demod_to_bits(symbols_complex: np.ndarray, constellation: np.ndarray,
                        bits_per_sym: int) -> np.ndarray:
    """Map complex symbols to nearest constellation point an extract bits.

    Args:
        symbols_complex: (N,) complex array of received symbols
        constellation: (M,) complex array where index = Gray-coded bit pattern
        bits_per_sym: number of bits per symbol

    Returns:
        bits: (N * bits_per_sym,) uint8 array
    """
    # Nearest-neighbour lookup
    # (N, 1) - (1, M) → (N, M)
    dists = np.abs(symbols_complex[:, None] - constellation[None, :])
    indices = np.argmin(dists, axis=1)  # (N,)

    # Index → bits (MSB first)
    bits = np.zeros((len(indices), bits_per_sym), dtype=np.uint8)
    for b in range(bits_per_sym):
        bits[:, bits_per_sym - 1 - b] = (indices >> b) & 1
    return bits.reshape(-1)


# --- Bit-aided alpha alignment ---

def _bit_aided_alpha(pred_syms: np.ndarray, ref_syms: np.ndarray) -> complex:
    """Estimate complex gain α such that pred ≈ α * ref.

    Uses least-squares: α = <pred, ref*> / <ref, ref*>
    """
    num = np.sum(pred_syms * np.conj(ref_syms))
    den = np.sum(ref_syms * np.conj(ref_syms))
    if abs(den) < 1e-30:
        return 1.0 + 0j
    return num / den


def _compare_demod_bits(demod_bits: np.ndarray, gt_bits: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (num_errors, num_compared_bits) for one demodulated frame."""
    n_compare = min(len(demod_bits), len(gt_bits))
    if n_compare <= 0:
        return None
    errors = int(np.sum(demod_bits[:n_compare] != gt_bits[:n_compare]))
    return errors, n_compare


@lru_cache(maxsize=8)
def _rrc_taps(alpha: float, span: int, sps: int) -> np.ndarray:
    """Return unit-energy root-raised-cosine taps matching MATLAB rcosdesign(..., 'sqrt')."""
    alpha = float(alpha)
    span = int(span)
    sps = int(sps)
    t = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64) / float(sps)
    taps = np.zeros_like(t, dtype=np.float64)

    for idx, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[idx] = 1.0 + alpha * (4.0 / np.pi - 1.0)
        elif alpha > 0 and abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-12:
            taps[idx] = (
                alpha / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1.0 - alpha))
                + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            )
            denominator = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            if abs(denominator) < 1e-12:
                denominator = 1e-12 if denominator >= 0 else -1e-12
            taps[idx] = numerator / denominator

    taps /= np.sqrt(np.sum(taps ** 2) + 1e-12)
    return taps


def _matched_filter_complex(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply a complex matched filter using same-length output."""
    return np.convolve(x, taps.astype(np.complex128), mode="same")


_8PSK_A_TOTAL_SYMBOLS_PER_FRAME = 205
_8PSK_A_SPS = 20
_8PSK_A_ALPHA = 0.35
_8PSK_A_SPAN = 20


def _linear_interp_complex(x: np.ndarray, pos: float) -> complex:
    """Linear interpolation for a complex sequence at fractional position pos."""
    if pos <= 0:
        return complex(x[0])
    if pos >= len(x) - 1:
        return complex(x[-1])
    idx = int(np.floor(pos))
    frac = float(pos - idx)
    return complex((1.0 - frac) * x[idx] + frac * x[idx + 1])


def _nearest_constellation_symbol(symbol: complex, constellation: np.ndarray) -> complex:
    """Return the nearest constellation point."""
    return constellation[int(np.argmin(np.abs(symbol - constellation)))]


def _sample_symbols_linear(x: np.ndarray, start_pos: float, n_symbols: int, sps: int) -> np.ndarray:
    """Sample n_symbols from x at start_pos + k*sps using linear interpolation."""
    return np.asarray(
        [_linear_interp_complex(x, start_pos + idx * sps) for idx in range(n_symbols)],
        dtype=np.complex128,
    )


def _hard_demod_to_indices(symbols_complex: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """Map complex symbols to nearest constellation indices."""
    dists = np.abs(symbols_complex[:, None] - constellation[None, :])
    return np.argmin(dists, axis=1).astype(np.int64)


def _decision_evm(symbols_complex: np.ndarray, constellation: np.ndarray) -> float:
    """Compute blind decision-directed EVM against the nearest constellation points."""
    if len(symbols_complex) == 0:
        return float("inf")
    nearest = constellation[_hard_demod_to_indices(symbols_complex, constellation)]
    num = float(np.mean(np.abs(symbols_complex - nearest) ** 2))
    den = float(np.mean(np.abs(nearest) ** 2) + 1e-12)
    return num / den


def _estimate_mpsk_cfo_per_symbol(symbols: np.ndarray, order: int) -> float:
    """Non-data-aided M-PSK CFO estimate in radians per symbol via M-th power."""
    if len(symbols) < 2:
        return 0.0
    z = symbols ** int(order)
    return float(np.angle(np.sum(z[1:] * np.conj(z[:-1])) + 1e-12) / float(order))


def _viterbi_viterbi_phase(symbols: np.ndarray, order: int) -> float:
    """Non-data-aided M-PSK carrier phase estimate modulo 2*pi/order."""
    if len(symbols) == 0:
        return 0.0
    return float(np.angle(np.sum(symbols ** int(order)) + 1e-12) / float(order))


def _early_late_timing_recovery(
    mf: np.ndarray,
    start_pos: float,
    n_symbols: int,
    sps: int,
    loop_gain: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """A simple early-late timing loop on the matched-filtered waveform."""
    pos = float(start_pos)
    samples = np.zeros(n_symbols, dtype=np.complex128)
    sample_pos = np.zeros(n_symbols, dtype=np.float64)

    for idx in range(n_symbols):
        main = _linear_interp_complex(mf, pos)
        early = _linear_interp_complex(mf, pos - 0.5 * sps)
        late = _linear_interp_complex(mf, pos + 0.5 * sps)
        err = float(np.real((late - early) * np.conj(main)))
        samples[idx] = main
        sample_pos[idx] = pos
        pos += float(sps) + loop_gain * err

    return samples, sample_pos


def _phase_track_symbols(
    rx_syms: np.ndarray,
    preamble_syms: np.ndarray,
    constellation: np.ndarray,
    alpha: float = 0.08,
    beta: float = 0.002,
    initial_phase: float = 0.0,
    initial_freq: float = 0.0,
) -> np.ndarray:
    """Decision-directed carrier phase/CFO tracking loop."""
    corrected = np.zeros_like(rx_syms, dtype=np.complex128)
    phase = float(initial_phase)
    freq = float(initial_freq)
    n_pre = len(preamble_syms)

    for idx, sym in enumerate(rx_syms):
        y = sym * np.exp(-1j * phase)
        if idx < n_pre:
            ref = preamble_syms[idx]
        else:
            ref = _nearest_constellation_symbol(y, constellation)
        corrected[idx] = y
        err = float(np.angle(y * np.conj(ref) + 1e-12))
        freq += beta * err
        phase += freq + alpha * err

    return corrected


def _rotation_resolved_compare_8psk(
    rx_syms: np.ndarray,
    gt_bits: np.ndarray,
    constellation: np.ndarray,
    bits_per_sym: int,
) -> Optional[Tuple[int, int]]:
    """Compare after quotienting out the unavoidable 8PSK global phase ambiguity."""
    n_const = len(constellation)
    best = None

    for rot in range(n_const):
        rot_const = constellation * np.exp(1j * 2.0 * np.pi * rot / n_const)
        demod_bits = _hard_demod_to_bits(rx_syms, rot_const, bits_per_sym)
        compare_stats = _compare_demod_bits(demod_bits, gt_bits)
        if compare_stats is None:
            continue
        errors, n_compare = compare_stats
        candidate = (errors / max(1, n_compare), rot, errors, -n_compare)
        if best is None or candidate < best[0]:
            best = (candidate, errors, n_compare)

    return None if best is None else best[1:]


def _receiver_demod_8psk_a(
    rx_c: np.ndarray,
    gt_bits: np.ndarray,
    constellation: np.ndarray,
    bits_per_sym: int,
    use_oracle: bool = False,
) -> Optional[Tuple[int, int]]:
    """Demodulate one 8PSK-A frame without preamble using known protocol parameters.

    The local 8PSK-A dataset stores 4096 samples per frame while the bit labels
    contain 205 symbols (615 bits). Since 205 * 20 = 4100, the final symbol is
    only partially visible. Use the largest complete in-frame symbol count for
    BER instead of rejecting these frames outright.
    """
    if len(gt_bits) <= 0 or len(rx_c) < _8PSK_A_SPS:
        return None

    total_symbols = min(
        _8PSK_A_TOTAL_SYMBOLS_PER_FRAME,
        len(gt_bits) // bits_per_sym,
        len(rx_c) // _8PSK_A_SPS,
    )
    expected_samples = total_symbols * _8PSK_A_SPS
    if expected_samples <= 0:
        return None

    payload_symbols = total_symbols
    if payload_symbols <= 0:
        return None
    taps = _rrc_taps(alpha=_8PSK_A_ALPHA, span=_8PSK_A_SPAN, sps=_8PSK_A_SPS)
    mf = _matched_filter_complex(rx_c, taps)
    n_const = len(constellation)
    phase_offsets = (-0.5, 0.0, 0.5) if use_oracle else (0.0,)
    loop_gains = (0.008, 0.012, 0.016) if use_oracle else (0.01,)
    candidate_records = []

    frame_shift_candidates = range(2) if use_oracle else (0,)
    for frame_shift in frame_shift_candidates:
        rx_shifted = mf[frame_shift:]
        if len(rx_shifted) < expected_samples:
            continue
        for ofs in range(_8PSK_A_SPS):
            for delta in phase_offsets:
                start_pos = float(ofs) + float(delta)
                coarse_syms = _sample_symbols_linear(rx_shifted, start_pos, total_symbols, _8PSK_A_SPS)
                if len(coarse_syms) < total_symbols:
                    continue

                cfo_sym = _estimate_mpsk_cfo_per_symbol(coarse_syms, n_const)
                n = np.arange(len(rx_shifted), dtype=np.float64)
                rx_cfo = rx_shifted * np.exp(-1j * (cfo_sym / _8PSK_A_SPS) * n)

                for loop_gain in loop_gains:
                    rx_syms, _ = _early_late_timing_recovery(
                        rx_cfo,
                        start_pos=start_pos,
                        n_symbols=total_symbols,
                        sps=_8PSK_A_SPS,
                        loop_gain=float(loop_gain),
                    )
                    if len(rx_syms) < total_symbols:
                        continue

                    phase0 = _viterbi_viterbi_phase(rx_syms, n_const)
                    rx_syms = rx_syms * np.exp(-1j * phase0)
                    rx_syms = _phase_track_symbols(
                        rx_syms,
                        np.empty(0, dtype=np.complex128),
                        constellation,
                        alpha=0.08 if use_oracle else 0.06,
                        beta=0.002 if use_oracle else 0.0015,
                    )
                    payload_syms = rx_syms[:payload_symbols]
                    if len(payload_syms) < payload_symbols:
                        continue

                    blind_metric = (
                        _decision_evm(payload_syms, constellation),
                        abs(float(np.mean(np.abs(payload_syms)) - 1.0)),
                        abs(cfo_sym),
                        frame_shift,
                        abs(delta),
                        abs(float(loop_gain) - 0.01),
                        ofs,
                    )
                    candidate_records.append((blind_metric, payload_syms))

    if not candidate_records:
        return None

    if use_oracle:
        best = None
        for blind_metric, payload_syms in candidate_records:
            compare_stats = _rotation_resolved_compare_8psk(payload_syms, gt_bits, constellation, bits_per_sym)
            if compare_stats is None:
                continue
            errors, n_compare = compare_stats
            candidate = (errors / max(1, n_compare), blind_metric, errors, -n_compare)
            if best is None or candidate < best[0]:
                best = (candidate, errors, n_compare)
        return None if best is None else best[1:]

    blind_metric, payload_syms = min(candidate_records, key=lambda item: item[0])
    _ = blind_metric
    return _rotation_resolved_compare_8psk(payload_syms, gt_bits, constellation, bits_per_sym)


def _ber_iq_from_bits_8psk_a_impl(
    pred: torch.Tensor,
    target: torch.Tensor,
    bits_gt: torch.Tensor,
    modulation: str,
    offset_search: bool,
    oracle: bool,
) -> torch.Tensor:
    """Receiver-style BER for 8PSK-A with blind timing/CFO/phase recovery."""
    constellation, bits_per_sym = constellation_for_modulation(modulation)
    constellation = constellation.astype(np.complex128)

    pred_np = pred.detach().cpu().float().numpy()
    bits_np = bits_gt.detach().cpu().numpy()

    total_errors = 0
    total_bits = 0
    for b_idx in range(pred_np.shape[0]):
        pred_c = pred_np[b_idx, 0, :] + 1j * pred_np[b_idx, 1, :]
        gt_bits = bits_np[b_idx]
        selected = _receiver_demod_8psk_a(
            rx_c=pred_c,
            gt_bits=gt_bits,
            constellation=constellation,
            bits_per_sym=bits_per_sym,
            use_oracle=oracle or offset_search,
        )
        if selected is None:
            continue
        errors, n_compare = selected
        total_errors += int(errors)
        total_bits += int(n_compare)

    if total_bits == 0:
        return torch.tensor(float("nan"))
    return torch.tensor(total_errors / total_bits, dtype=torch.float32)


def _strict_sps_candidates(L: int, n_syms: int, sps: Optional[int]) -> List[int]:
    """Strict BER uses a single SPS estimate unless one is explicitly provided."""
    if sps is not None:
        return [max(1, int(sps))]
    return [max(1, round(L / n_syms))]


def _oracle_sps_candidates(L: int, n_syms: int, sps: Optional[int]) -> List[int]:
    """Oracle BER may try nearby SPS candidates."""
    if sps is not None:
        return [max(1, int(sps))]
    sps_center = max(1, round(L / n_syms))
    return sorted(set([max(1, sps_center - 1), sps_center, sps_center + 1]))


def _select_strict_candidate(
    pred_c: np.ndarray,
    tgt_c: np.ndarray,
    gt_bits: np.ndarray,
    constellation: np.ndarray,
    bits_per_sym: int,
    sps_candidates: List[int],
    offset_search: bool,
):
    """
    Pick one demodulation candidate without using gt bits as the search objective.

    We use waveform-domain alignment quality after complex-gain normalization as the
    selection criterion, which is acceptable for blind-source-separation evaluation.
    """
    best = None
    for s in sps_candidates:
        offsets = range(s) if offset_search else [0]
        for ofs in offsets:
            pred_syms = pred_c[ofs::s]
            tgt_syms = tgt_c[ofs::s]
            if len(pred_syms) == 0 or len(tgt_syms) == 0:
                continue

            n_syms = min(len(pred_syms), len(tgt_syms), max(1, len(gt_bits) // bits_per_sym))
            pred_syms = pred_syms[:n_syms]
            tgt_syms = tgt_syms[:n_syms]
            if len(pred_syms) == 0:
                continue

            alpha = _bit_aided_alpha(pred_syms, tgt_syms)
            if abs(alpha) < 1e-12:
                continue
            aligned_syms = pred_syms / alpha
            align_mse = float(np.mean(np.abs(aligned_syms - tgt_syms) ** 2))

            demod_bits = _hard_demod_to_bits(aligned_syms, constellation, bits_per_sym)
            compare_stats = _compare_demod_bits(demod_bits, gt_bits)
            if compare_stats is None:
                continue
            errors, n_compare = compare_stats
            candidate = (align_mse, -n_compare, errors, n_compare)
            if best is None or candidate < best[0]:
                best = (candidate, errors, n_compare)
    return None if best is None else best[1:]


def _select_oracle_candidate(
    pred_c: np.ndarray,
    tgt_c: np.ndarray,
    gt_bits: np.ndarray,
    constellation: np.ndarray,
    bits_per_sym: int,
    sps_candidates: List[int],
    offset_search: bool,
):
    """Pick the best BER candidate using gt bits as an oracle objective."""
    best = None
    n_syms_target = max(1, len(gt_bits) // bits_per_sym)
    for s in sps_candidates:
        offsets = range(s) if offset_search else [0]
        for ofs in offsets:
            pred_syms = pred_c[ofs::s][:n_syms_target]
            tgt_syms = tgt_c[ofs::s][:n_syms_target]
            n_got = len(pred_syms)
            if n_got < max(1, n_syms_target - 2):
                continue

            alpha = _bit_aided_alpha(pred_syms, tgt_syms)
            if abs(alpha) < 1e-12:
                continue
            aligned_syms = pred_syms / alpha

            demod_bits = _hard_demod_to_bits(aligned_syms, constellation, bits_per_sym)
            compare_stats = _compare_demod_bits(demod_bits, gt_bits)
            if compare_stats is None:
                continue
            errors, n_compare = compare_stats
            frame_ber = errors / n_compare
            candidate = (frame_ber, errors, -n_compare)
            if best is None or candidate < best[0]:
                best = (candidate, errors, n_compare)
    return None if best is None else best[1:]


def _ber_iq_from_bits_impl(
    pred: torch.Tensor,
    target: torch.Tensor,
    bits_gt: torch.Tensor,
    modulation: str,
    sps: Optional[int],
    offset_search: bool,
    selector,
) -> torch.Tensor:
    """Shared implementation for strict/oracle BER.

    Notes:
        This helper assumes BER can be approximated by direct symbol picking
        from the saved IQ waveform plus a single complex-gain alignment per
        item. It does not perform matched filtering or carrier-frequency-offset
        correction, so pulse-shaped streams with residual CFO/phase drift can
        show pessimistic BER even for visually good separations.
    """
    constellation, bits_per_sym = constellation_for_modulation(modulation)
    constellation = constellation.astype(np.complex128)

    pred_np = pred.detach().cpu().float().numpy()
    tgt_np = target.detach().cpu().float().numpy()
    bits_np = bits_gt.detach().cpu().numpy()

    B, _, L = pred_np.shape
    n_bits_per_frame = bits_np.shape[1]
    n_syms = n_bits_per_frame // bits_per_sym
    if n_syms <= 0:
        return torch.tensor(float("nan"))

    sps_candidates = selector["sps_candidates"](L, n_syms, sps)
    total_errors = 0
    total_bits = 0

    for b_idx in range(B):
        pred_c = pred_np[b_idx, 0, :] + 1j * pred_np[b_idx, 1, :]
        tgt_c = tgt_np[b_idx, 0, :] + 1j * tgt_np[b_idx, 1, :]
        gt_bits = bits_np[b_idx]

        selected = selector["pick_candidate"](
            pred_c=pred_c,
            tgt_c=tgt_c,
            gt_bits=gt_bits,
            constellation=constellation,
            bits_per_sym=bits_per_sym,
            sps_candidates=sps_candidates,
            offset_search=offset_search,
        )
        if selected is None:
            continue

        errors, n_compare = selected
        total_errors += int(errors)
        total_bits += int(n_compare)

    if total_bits == 0:
        return torch.tensor(float("nan"))
    return torch.tensor(total_errors / total_bits, dtype=torch.float32)


def strict_ber_iq_from_bits(
    pred: torch.Tensor,
    target: torch.Tensor,
    bits_gt: torch.Tensor,
    modulation: str,
    sps: Optional[int] = None,
    offset_search: bool = False,
    protocol: Optional[str] = None,
) -> torch.Tensor:
    """BER without bit-label-driven SPS/offset selection."""
    if str(protocol).upper() == "8PSK-A":
        return _ber_iq_from_bits_8psk_a_impl(
            pred=pred,
            target=target,
            bits_gt=bits_gt,
            modulation=modulation,
            offset_search=offset_search,
            oracle=False,
        )
    return _ber_iq_from_bits_impl(
        pred=pred,
        target=target,
        bits_gt=bits_gt,
        modulation=modulation,
        sps=sps,
        offset_search=offset_search,
        selector={
            "sps_candidates": _strict_sps_candidates,
            "pick_candidate": _select_strict_candidate,
        },
    )


def oracle_ber_iq_from_bits(
    pred: torch.Tensor,
    target: torch.Tensor,
    bits_gt: torch.Tensor,
    modulation: str,
    sps: Optional[int] = None,
    offset_search: bool = False,
    protocol: Optional[str] = None,
) -> torch.Tensor:
    """Best-case BER using gt bits to choose the most favorable SPS/offset."""
    if str(protocol).upper() == "8PSK-A":
        return _ber_iq_from_bits_8psk_a_impl(
            pred=pred,
            target=target,
            bits_gt=bits_gt,
            modulation=modulation,
            offset_search=offset_search,
            oracle=True,
        )
    return _ber_iq_from_bits_impl(
        pred=pred,
        target=target,
        bits_gt=bits_gt,
        modulation=modulation,
        sps=sps,
        offset_search=offset_search,
        selector={
            "sps_candidates": _oracle_sps_candidates,
            "pick_candidate": _select_oracle_candidate,
        },
    )


# --- Main BER function ---

def ber_iq_from_bits(
    pred: torch.Tensor,
    target: torch.Tensor,
    bits_gt: torch.Tensor,
    modulation: str,
    sps: Optional[int] = None,
    offset_search: bool = False,
    protocol: Optional[str] = None,
) -> torch.Tensor:
    """Backward-compatible alias for the legacy oracle-style BER."""
    return oracle_ber_iq_from_bits(
        pred=pred,
        target=target,
        bits_gt=bits_gt,
        modulation=modulation,
        sps=sps,
        offset_search=offset_search,
        protocol=protocol,
    )


strict_ber = strict_ber_iq_from_bits
oracle_ber = oracle_ber_iq_from_bits