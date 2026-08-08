import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple


_PERMUTATION_CACHE = {}


def _get_permutations(num_sources: int):
    if num_sources not in _PERMUTATION_CACHE:
        _PERMUTATION_CACHE[num_sources] = list(permutations(range(num_sources)))
    return _PERMUTATION_CACHE[num_sources]


def _si_snr_pair_per_item(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SNR for one source pair, with positive-projection constraint.

    Standard SI-SNR allows negative real alpha, meaning pred=-s and pred=+s
    give identical loss — the 'sign ambiguity' problem for IQ BSS.

    Fix: clamp alpha >= 0.  When pred and target are anti-correlated the
    projection onto target is zero, giving very low SI-SNR and a gradient
    that corrects the sign flip.

    Args:
        pred:   (B, 2, L)
        target: (B, 2, L)

    Returns:
        (B,) SI-SNR in dB
    """
    pred_flat   = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    alpha_raw = torch.sum(pred_flat * target_flat, dim=1, keepdim=True) / (
        torch.sum(target_flat * target_flat, dim=1, keepdim=True) + eps
    )
    # Clamp to non-negative: sign-flipped pred gets zero projection,
    # yielding very low SI-SNR that the optimizer is forced to correct.
    alpha = alpha_raw.clamp(min=0.0)

    target_scaled = alpha * target_flat
    noise = pred_flat - target_scaled

    si_snr = torch.sum(target_scaled ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps)
    return 10 * torch.log10(si_snr + eps)


def _mse_pair_per_item(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2, dim=(-2, -1))


def _l1_pair_per_item(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target), dim=(-2, -1))


def _huber_pair_per_item(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    error = pred - target
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta, dtype=abs_error.dtype, device=abs_error.device))
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return loss.mean(dim=(-2, -1))


def _phase_increment_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rotation-invariant phase-step error for one complex source pair.

    The phase increment is represented as a unit complex phasor instead of an
    angle, avoiding the discontinuity at +/-pi. Target transition magnitude
    down-weights near-zero samples whose phase is not reliable.
    """

    if pred.ndim != 3 or pred.size(1) != 2:
        raise ValueError(f"Expected complex pair (B, 2, L), got {tuple(pred.shape)}")
    if pred.shape[-1] < 2:
        return pred.new_zeros((pred.shape[0],))

    pred_i, pred_q = pred[:, 0], pred[:, 1]
    tgt_i, tgt_q = target[:, 0], target[:, 1]
    pred_real = pred_i[:, 1:] * pred_i[:, :-1] + pred_q[:, 1:] * pred_q[:, :-1]
    pred_imag = pred_q[:, 1:] * pred_i[:, :-1] - pred_i[:, 1:] * pred_q[:, :-1]
    tgt_real = tgt_i[:, 1:] * tgt_i[:, :-1] + tgt_q[:, 1:] * tgt_q[:, :-1]
    tgt_imag = tgt_q[:, 1:] * tgt_i[:, :-1] - tgt_i[:, 1:] * tgt_q[:, :-1]

    pred_norm = torch.sqrt(pred_real.square() + pred_imag.square() + eps)
    tgt_norm = torch.sqrt(tgt_real.square() + tgt_imag.square() + eps)
    pred_real, pred_imag = pred_real / pred_norm, pred_imag / pred_norm
    tgt_real, tgt_imag = tgt_real / tgt_norm, tgt_imag / tgt_norm

    target_weight = tgt_norm / tgt_norm.mean(dim=-1, keepdim=True).clamp_min(eps)
    target_weight = target_weight.clamp(max=4.0)
    phasor_error = (pred_real - tgt_real).square() + (pred_imag - tgt_imag).square()
    return (target_weight * phasor_error).mean(dim=-1)


def to_complex_sources(x: torch.Tensor, num_sources: int) -> torch.Tensor:
    """
    x: (B, 2*K, T)
    return: (B, K, 2, T), where dim=2 is [I, Q].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (B, 2*K, T), got {tuple(x.shape)}")
    b, c, t = x.shape
    if c != 2 * num_sources:
        raise ValueError(f"Expected {2 * num_sources} channels, got {c}")
    return x.view(b, num_sources, 2, t)


def complex_huber_per_source(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Complex Huber on |pred-target| for tensors shaped (B, K, 2, T).

    Returns:
        (B, K) loss values.
    """
    if pred.shape != target.shape or pred.ndim != 4 or pred.size(2) != 2:
        raise ValueError(f"Expected matching (B, K, 2, T), got pred={tuple(pred.shape)} target={tuple(target.shape)}")
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")
    err = pred - target
    r = torch.sqrt(err[:, :, 0].pow(2) + err[:, :, 1].pow(2) + eps)
    loss = torch.where(
        r < beta,
        0.5 * r.pow(2) / beta,
        r - 0.5 * beta,
    )
    return loss.mean(dim=-1)


def _split_iq_sources(x: torch.Tensor, num_sources: int):
    return [x[:, 2 * i : 2 * i + 2, :] for i in range(num_sources)]


def _infer_num_sources(outputs: torch.Tensor, targets: torch.Tensor, num_sources: int = None) -> int:
    if outputs.ndim != 3 or targets.ndim != 3:
        raise ValueError(f"Expected 3D tensors, got outputs={outputs.shape}, targets={targets.shape}")
    if outputs.shape != targets.shape:
        raise ValueError(f"Shape mismatch: outputs={outputs.shape}, targets={targets.shape}")
    _, channels, _ = outputs.shape
    if channels % 2 != 0:
        raise ValueError(f"Expected even channel count for IQ pairs, got {channels}")

    inferred_sources = channels // 2
    if num_sources is None:
        return inferred_sources
    if num_sources != inferred_sources:
        raise ValueError(f"num_sources={num_sources} but inferred {inferred_sources} from channels={channels}")
    return num_sources


def _pit_reduce(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pair_metric_fn,
    num_sources: int = None,
    maximize: bool = False,
) -> torch.Tensor:
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            total = total + pair_metric_fn(preds[pred_idx], tgts[target_idx])
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    if maximize:
        best_scores = all_scores.max(dim=0).values
    else:
        best_scores = all_scores.min(dim=0).values
    return best_scores.mean()


def _pit_reduce_with_index(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    pair_metric_with_index_fn,
    num_sources: int = None,
    maximize: bool = False,
) -> torch.Tensor:
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            total = total + pair_metric_with_index_fn(preds[pred_idx], tgts[target_idx], target_idx)
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    if maximize:
        best_scores = all_scores.max(dim=0).values
    else:
        best_scores = all_scores.min(dim=0).values
    return best_scores.mean()


def _iq_to_complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0, :], x[:, 1, :])


def _evm_pair_per_item(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred_c = _iq_to_complex(pred)
    tgt_c = _iq_to_complex(target)
    err_pow = torch.mean(torch.abs(pred_c - tgt_c) ** 2, dim=-1)
    ref_pow = torch.mean(torch.abs(tgt_c) ** 2, dim=-1)
    return torch.sqrt(err_pow / (ref_pow + eps) + eps)


def _gain_phase_aligned_huber_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Complex Huber after per-sample complex gain/phase alignment.

    A receiver can usually absorb one global complex gain per separated stream.
    This term therefore measures the residual complex error after the best
    least-squares scalar ``a`` aligns ``pred`` to ``target``:

        a = <target, pred> / <pred, pred>
        err = a * pred - target

    The error magnitude is normalized by target RMS, making the term EVM-like
    and less sensitive to absolute signal scale.
    """
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")

    pred_c = _iq_to_complex(pred)
    target_c = _iq_to_complex(target)

    numerator = torch.sum(target_c * torch.conj(pred_c), dim=-1, keepdim=True)
    denominator = torch.sum(torch.abs(pred_c).pow(2), dim=-1, keepdim=True).clamp_min(eps)
    gain = numerator / denominator

    aligned = gain * pred_c
    target_rms = torch.sqrt(torch.mean(torch.abs(target_c).pow(2), dim=-1, keepdim=True) + eps)
    err_mag = torch.abs(aligned - target_c) / target_rms

    loss = torch.where(
        err_mag < beta,
        0.5 * err_mag.pow(2) / beta,
        err_mag - 0.5 * beta,
    )
    return loss.mean(dim=-1)


def _qpsk_constellation():
    points = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=torch.complex64)
    return points / torch.sqrt(torch.tensor(2.0, dtype=torch.float32))


def _bpsk_constellation():
    return torch.tensor([1 + 0j, -1 + 0j], dtype=torch.complex64)


def _mpsk_constellation(m: int):
    idx = torch.arange(m, dtype=torch.float32)
    angles = 2.0 * torch.pi * idx / m
    return torch.complex(torch.cos(angles), torch.sin(angles)).to(torch.complex64)


def _mqam_constellation(m: int):
    side = int(m ** 0.5)
    if side * side != m:
        raise ValueError(f"QAM order must be perfect square, got {m}")
    axis = torch.arange(-(side - 1), side, 2, dtype=torch.float32)
    points = []
    for imag in axis:
        for real in axis:
            points.append(complex(float(real), float(imag)))
    const = torch.tensor(points, dtype=torch.complex64)
    const = const / torch.sqrt(torch.mean(torch.abs(const) ** 2))
    return const


def _normalize_mod_name(name: str) -> str:
    return name.strip().upper().replace("-", "").replace("_", "")


def _get_constellation_from_name(name: str) -> Optional[torch.Tensor]:
    norm = _normalize_mod_name(name)
    if norm in {"BPSK"}:
        return _bpsk_constellation()
    if norm in {"QPSK", "4QAM"}:
        return _qpsk_constellation()
    if norm in {"8PSK"}:
        return _mpsk_constellation(8)
    if norm in {"16QAM"}:
        return _mqam_constellation(16)
    if norm in {"64QAM"}:
        return _mqam_constellation(64)
    return None


def _constellation_distance_per_item(
    pred: torch.Tensor,
    constellation: Optional[torch.Tensor],
    symbol_stride: int = 1,
) -> torch.Tensor:
    if constellation is None:
        return torch.zeros(pred.shape[0], dtype=pred.dtype, device=pred.device)

    pred_c = _iq_to_complex(pred)
    stride = max(1, int(symbol_stride))
    pred_sym = pred_c[:, ::stride]
    const = constellation.to(device=pred.device, dtype=pred_c.dtype)
    dist = torch.abs(pred_sym.unsqueeze(-1) - const.view(1, 1, -1))
    return dist.min(dim=-1).values.mean(dim=-1)


def _build_constellation_bank(
    source_names: Optional[List[str]],
    num_sources: int,
) -> List[Optional[torch.Tensor]]:
    if not source_names:
        return [None] * num_sources
    bank = []
    for index in range(num_sources):
        if index < len(source_names):
            bank.append(_get_constellation_from_name(source_names[index]))
        else:
            bank.append(None)
    return bank


def _bandwidth_weight_vector(
    n_fft: int,
    n_freq: int,
    band_ratio: float,
    inband_weight: float,
    outband_weight: float,
    device,
    dtype,
) -> torch.Tensor:
    if n_freq == n_fft:
        freqs = torch.fft.fftfreq(n_fft, d=1.0, device=device)
    elif n_freq == (n_fft // 2 + 1):
        freqs = torch.fft.rfftfreq(n_fft, d=1.0, device=device)
    else:
        raise ValueError(f"Unexpected STFT frequency bins: got {n_freq}, expected {n_fft} or {n_fft // 2 + 1}")

    half_band = max(0.0, min(0.5, float(band_ratio) * 0.5))
    inband_mask = (torch.abs(freqs) <= half_band).to(dtype=dtype)
    weights = outband_weight + (inband_weight - outband_weight) * inband_mask
    return weights.view(1, -1, 1)


def _bw_mrstft_pair_per_item(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    n_ffts: List[int],
    band_ratio: float,
    inband_weight: float,
    outband_weight: float,
    mag_weight: float,
    complex_weight: float,
    eps: float,
) -> torch.Tensor:
    # STFT with center=True uses reflection padding, which does not support ComplexHalf.
    # Under AMP, pred/tgt may be fp16, so we compute MRSTFT terms in fp32 for stability.
    pred_work = pred.float()
    tgt_work = tgt.float()
    pred_c = _iq_to_complex(pred_work)
    tgt_c = _iq_to_complex(tgt_work)

    per_resolution = []
    for n_fft in n_ffts:
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred_work.dtype)
        pred_spec = torch.stft(
            pred_c,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
            onesided=False,
        )
        tgt_spec = torch.stft(
            tgt_c,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
            onesided=False,
        )

        weights = _bandwidth_weight_vector(
            n_fft=n_fft,
            n_freq=pred_spec.shape[1],
            band_ratio=band_ratio,
            inband_weight=inband_weight,
            outband_weight=outband_weight,
            device=pred.device,
            dtype=pred_work.dtype,
        )
        pred_mag = torch.abs(pred_spec)
        tgt_mag = torch.abs(tgt_spec)
        mag_diff = torch.abs(torch.log(pred_mag + eps) - torch.log(tgt_mag + eps))
        complex_diff = torch.abs(pred_spec - tgt_spec)

        mag_term = torch.mean(weights * mag_diff, dim=(1, 2))
        complex_term = torch.mean(weights * complex_diff, dim=(1, 2))
        per_resolution.append(mag_weight * mag_term + complex_weight * complex_term)

    return torch.stack(per_resolution, dim=0).mean(dim=0)



def si_snr_loss(outputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss for two separated signals.
    
    Args:
        outputs: (B, 4, L) - [I1, Q1, I2, Q2]
        targets: (B, 4, L) - [I1, Q1, I2, Q2]
        eps: small value to avoid division by zero
    
    Returns:
        loss: scalar tensor (mean over batch and two signals)
    """
    num_sources = _infer_num_sources(outputs, targets, None)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)
    per_source = [_si_snr_pair_per_item(pred, tgt, eps=eps) for pred, tgt in zip(preds, tgts)]
    si_snr_total = torch.stack(per_source, dim=0).mean(dim=0)  # (B,)
    return -si_snr_total.mean()


def pit_si_snr_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Permutation-Invariant SI-SNR loss (per-sample PIT).

    Args:
        outputs: (B, 2*K, L)
        targets: (B, 2*K, L)
        num_sources: K (optional, inferred from channel dim)

    Returns:
        Scalar loss: negative mean of best per-sample SI-SNR across permutations.
    """
    best_si_snr = _pit_reduce(
        outputs,
        targets,
        pair_metric_fn=lambda pred, tgt: _si_snr_pair_per_item(pred, tgt, eps=eps),
        num_sources=num_sources,
        maximize=True,
    )
    return -best_si_snr


def constellation_kde_loss(pred: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """
    Computes Constellation KDE clustering loss.
    pred: (B, 2*K, L)
    Encourages the IQ scatter plot to form tight clusters (minimize entropy).
    """
    K = pred.shape[1] // 2
    pred_c = to_complex_sources(pred, K)
    
    B, K_dim, _, L = pred_c.shape
    
    num_samples = min(128, L)
    idx = torch.randperm(L, device=pred.device)[:num_samples]
    
    loss = 0.0
    for k in range(K_dim):
        src_iq = pred_c[:, k, :, idx] # (B, 2, num_samples)
        src_iq_t = src_iq.transpose(1, 2) # (B, num_samples, 2)
        
        sq_norm = torch.sum(src_iq_t**2, dim=-1, keepdim=True)
        dist_sq = sq_norm + sq_norm.transpose(1, 2) - 2 * torch.bmm(src_iq_t, src_iq)
        
        # Information Potential (KDE proxy)
        V = torch.mean(torch.exp(-dist_sq / (2 * sigma**2)), dim=(1, 2))
        
        # Minimize -log(V)
        loss += torch.mean(-torch.log(V + 1e-8))
        
    return loss / K_dim

def pit_si_snr_constellation_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    sigma: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    si_snr = pit_si_snr_loss(outputs, targets, num_sources=num_sources, eps=eps)
    kde = constellation_kde_loss(outputs, sigma=sigma)
    return si_snr + alpha * kde
def _is_tensor_sequence(value) -> bool:
    return isinstance(value, (list, tuple)) and all(torch.is_tensor(v) for v in value)


def extract_final_and_stage_outputs(model_output) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return final separated waveform plus optional URIC stage outputs."""
    if isinstance(model_output, dict):
        final = None
        for key in ("final", "output", "separation", "sep"):
            if key in model_output:
                final = model_output[key]
                break
        stage_outputs = []
        for key in ("stage_outputs", "uric_stage_outputs"):
            if key in model_output:
                stage_outputs = model_output[key]
                break
        if final is None:
            raise ValueError("model_output dict must contain final/output/separation/sep")
        return final, list(stage_outputs)

    if (
        isinstance(model_output, tuple)
        and len(model_output) >= 2
        and torch.is_tensor(model_output[0])
        and _is_tensor_sequence(model_output[1])
    ):
        return model_output[0], list(model_output[1])

    if _is_tensor_sequence(model_output):
        outputs = list(model_output)
        return outputs[-1], outputs[:-1]

    if torch.is_tensor(model_output):
        return model_output, []

    raise TypeError(f"Unsupported model output type for URIC deep supervision: {type(model_output)}")


def _reduce_stage_losses(stage_losses: Sequence[torch.Tensor], reduction: str) -> torch.Tensor:
    if not stage_losses:
        raise ValueError("stage_losses must not be empty")
    reduction = str(reduction).lower()
    stacked = torch.stack(list(stage_losses), dim=0)
    if reduction == "sum":
        return stacked.sum()
    if reduction == "mean":
        return stacked.mean()
    raise ValueError(f"Unsupported stage reduction '{reduction}', expected 'sum' or 'mean'")


def pit_si_snr_uric_deep_supervision_loss(
    outputs,
    targets: torch.Tensor,
    stage_weight: float = 0.1,
    include_final_stage: bool = False,
    stage_reduction: str = "sum",
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    URIC deep-supervision loss:

        L = L_final + lambda * sum_k L_stage(k)

    where each term is PIT negative SI-SNR. If the model does not return
    stage outputs, this safely falls back to the final PIT-SI-SNR loss.
    """
    final_output, stage_outputs = extract_final_and_stage_outputs(outputs)
    final_loss = pit_si_snr_loss(final_output, targets, num_sources=num_sources, eps=eps)

    if stage_weight <= 0.0 or not stage_outputs:
        return final_loss

    supervised_stages = list(stage_outputs) if include_final_stage else list(stage_outputs[:-1])
    if not supervised_stages:
        return final_loss

    stage_losses = [
        pit_si_snr_loss(stage_output, targets, num_sources=num_sources, eps=eps)
        for stage_output in supervised_stages
    ]
    stage_loss = _reduce_stage_losses(stage_losses, stage_reduction)
    return final_loss + float(stage_weight) * stage_loss


def pit_si_snr_huber_uric_deep_supervision_loss(
    outputs,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    stage_weight: float = 0.1,
    include_final_stage: bool = False,
    stage_reduction: str = "sum",
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Final PIT-SI-SNR+Huber with URIC intermediate PIT-SI-SNR supervision.
    """
    final_output, stage_outputs = extract_final_and_stage_outputs(outputs)
    final_loss = pit_si_snr_huber_loss(
        final_output,
        targets,
        alpha=alpha,
        beta=beta,
        num_sources=num_sources,
        eps=eps,
        delta=delta,
    )

    if stage_weight <= 0.0 or not stage_outputs:
        return final_loss

    supervised_stages = list(stage_outputs) if include_final_stage else list(stage_outputs[:-1])
    if not supervised_stages:
        return final_loss

    stage_losses = [
        pit_si_snr_loss(stage_output, targets, num_sources=num_sources, eps=eps)
        for stage_output in supervised_stages
    ]
    stage_loss = _reduce_stage_losses(stage_losses, stage_reduction)
    return final_loss + float(stage_weight) * stage_loss


def pit_si_snr_huber_rms_uric_deep_supervision_loss(
    outputs,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    rms_lambda: float = 0.5,
    stage_weight: float = 0.1,
    include_final_stage: bool = False,
    stage_reduction: str = "sum",
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Final PIT-SI-SNR+Huber+RMS with URIC intermediate PIT-SI-SNR supervision.
    """
    final_output, stage_outputs = extract_final_and_stage_outputs(outputs)
    final_loss = pit_si_snr_huber_rms_loss(
        final_output,
        targets,
        alpha=alpha,
        beta=beta,
        rms_lambda=rms_lambda,
        num_sources=num_sources,
        eps=eps,
        delta=delta,
    )

    if stage_weight <= 0.0 or not stage_outputs:
        return final_loss

    supervised_stages = list(stage_outputs) if include_final_stage else list(stage_outputs[:-1])
    if not supervised_stages:
        return final_loss

    stage_losses = [
        pit_si_snr_loss(stage_output, targets, num_sources=num_sources, eps=eps)
        for stage_output in supervised_stages
    ]
    stage_loss = _reduce_stage_losses(stage_losses, stage_reduction)
    return final_loss + float(stage_weight) * stage_loss


def l1_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(outputs, targets)


def si_snr_huber_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    si_snr_term = si_snr_loss(outputs, targets, eps=eps)
    huber_term = F.huber_loss(outputs, targets, delta=delta)
    return alpha * si_snr_term + beta * huber_term


def pit_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
) -> torch.Tensor:
    return _pit_reduce(
        outputs,
        targets,
        pair_metric_fn=_mse_pair_per_item,
        num_sources=num_sources,
        maximize=False,
    )


def pit_l1_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
) -> torch.Tensor:
    return _pit_reduce(
        outputs,
        targets,
        pair_metric_fn=_l1_pair_per_item,
        num_sources=num_sources,
        maximize=False,
    )


def pit_huber_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    delta: float = 1.0,
) -> torch.Tensor:
    return _pit_reduce(
        outputs,
        targets,
        pair_metric_fn=lambda pred, tgt: _huber_pair_per_item(pred, tgt, delta=delta),
        num_sources=num_sources,
        maximize=False,
    )


def pit_si_snr_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pit_si_snr_term = pit_si_snr_loss(outputs, targets, num_sources=num_sources, eps=eps)
    pit_mse_term = pit_mse_loss(outputs, targets, num_sources=num_sources)
    return alpha * pit_si_snr_term + beta * pit_mse_term


def pit_demod_aware_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mse_weight: float = 1.0,
    sisnr_weight: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT loss for modulation-sensitive waveform separation.

    This is intentionally a shared-permutation objective:

        min_perm mean_k [mse_weight * MSE_k + sisnr_weight * (-SI-SNR_k)]

    The MSE term anchors absolute I/Q amplitude and sign, which is important
    for high-order QAM demodulation.  The positive-projection SI-SNR term keeps
    waveform structure pressure without giving a free pass to 180-degree flips.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            pred = preds[pred_idx]
            tgt = tgts[target_idx]
            mse_term = _mse_pair_per_item(pred, tgt)
            sisnr_term = -_si_snr_pair_per_item(pred, tgt, eps=eps)
            total = total + float(mse_weight) * mse_term + float(sisnr_weight) * sisnr_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def _pit_si_snr_huber_permutation_scores(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return combined SI-SNR+Huber costs for every source permutation.

    The returned shape is ``(num_permutations, batch)``. Keeping this helper
    shared ensures the ordinary PIT loss and the identity-anchored variant
    optimize exactly the same separation objective.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            total = total + alpha * si_term + beta * huber_term
        perm_scores.append(total / num_sources)

    return torch.stack(perm_scores, dim=0)


def pit_si_snr_huber_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    # Joint PIT on the combined objective:
    #   min_perm mean_k [alpha * (-SI-SNR_k) + beta * Huber_k]
    # This enforces one shared permutation for both terms.
    all_scores = _pit_si_snr_huber_permutation_scores(
        outputs,
        targets,
        alpha=alpha,
        beta=beta,
        num_sources=num_sources,
        eps=eps,
        delta=delta,
    )
    best_scores = all_scores.min(dim=0).values
    return best_scores.mean()


def pit_si_snr_huber_identity_anchor_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    identity_anchor_weight: float = 0.05,
    identity_anchor_margin: float = 0.2,
    identity_anchor_temperature: float = 0.5,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """PIT separation loss with a weak, vanishing fixed-identity tie-breaker.

    PIT remains the main objective. The anchor compares the fixed identity
    permutation against the best non-identity permutation and applies a smooth
    ranking penalty only when identity is not better by ``margin``:

        temperature * softplus(
            (identity_cost - best_other_cost + margin) / temperature
        )

    Unlike adding a second full fixed SI-SNR loss, this term approaches zero
    once the output slots have a stable identity, minimizing interference with
    permutation-invariant separation quality.
    """
    if identity_anchor_weight < 0.0:
        raise ValueError("identity_anchor_weight must be >= 0")
    if identity_anchor_margin < 0.0:
        raise ValueError("identity_anchor_margin must be >= 0")
    if identity_anchor_temperature <= 0.0:
        raise ValueError("identity_anchor_temperature must be > 0")

    resolved_sources = _infer_num_sources(outputs, targets, num_sources)
    all_scores = _pit_si_snr_huber_permutation_scores(
        outputs,
        targets,
        alpha=alpha,
        beta=beta,
        num_sources=resolved_sources,
        eps=eps,
        delta=delta,
    )
    pit_scores = all_scores.min(dim=0).values
    if resolved_sources <= 1 or identity_anchor_weight == 0.0:
        return pit_scores.mean()

    permutations_list = _get_permutations(resolved_sources)
    identity = tuple(range(resolved_sources))
    identity_index = permutations_list.index(identity)
    identity_scores = all_scores[identity_index]
    other_scores = torch.cat(
        (all_scores[:identity_index], all_scores[identity_index + 1 :]),
        dim=0,
    )
    best_other_scores = other_scores.min(dim=0).values
    violation = (
        identity_scores
        - best_other_scores
        + float(identity_anchor_margin)
    )
    anchor_scores = float(identity_anchor_temperature) * F.softplus(
        violation / float(identity_anchor_temperature)
    )
    return (
        pit_scores
        + float(identity_anchor_weight) * anchor_scores
    ).mean()


def pit_si_snr_huber_phase_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    phase_weight: float = 0.05,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """Shared-permutation waveform and phase-trajectory objective.

    This loss is modulation-label-free. It preserves the Stage-261 waveform
    objective while adding pressure on CFO/phase evolution, which pointwise
    Huber and scale-invariant SI-SNR do not isolate.
    """

    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)
    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            pred = preds[pred_idx]
            tgt = tgts[target_idx]
            si_term = -_si_snr_pair_per_item(pred, tgt, eps=eps)
            huber_term = _huber_pair_per_item(pred, tgt, delta=delta)
            phase_term = _phase_increment_pair_per_item(pred, tgt, eps=eps)
            total = (
                total
                + float(alpha) * si_term
                + float(beta) * huber_term
                + float(phase_weight) * phase_term
            )
        perm_scores.append(total / num_sources)
    return torch.stack(perm_scores, dim=0).min(dim=0).values.mean()


def _complex_projection_energy_ratio(
    pred: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fraction of pred complex energy explained by a least-squares reference copy."""
    pred_c = _iq_to_complex(pred)
    ref_c = _iq_to_complex(reference)

    numerator = torch.sum(pred_c * torch.conj(ref_c), dim=-1, keepdim=True)
    denominator = torch.sum(torch.abs(ref_c).pow(2), dim=-1, keepdim=True).clamp_min(eps)
    coeff = numerator / denominator

    projection = coeff * ref_c
    projection_energy = torch.mean(torch.abs(projection).pow(2), dim=-1)
    pred_energy = torch.mean(torch.abs(pred_c).pow(2), dim=-1).clamp_min(eps)
    return projection_energy / pred_energy


def _cross_talk_loss_for_perm(
    preds: Sequence[torch.Tensor],
    tgts: Sequence[torch.Tensor],
    perm: Tuple[int, ...],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize each matched prediction's projection onto non-matched targets."""
    num_sources = len(tgts)
    total = 0.0
    terms = 0
    for target_idx, pred_idx in enumerate(perm):
        pred = preds[pred_idx]
        for other_idx, other_tgt in enumerate(tgts):
            if other_idx == target_idx:
                continue
            total = total + _complex_projection_energy_ratio(pred, other_tgt, eps=eps)
            terms += 1
    if terms == 0:
        return torch.zeros(preds[0].size(0), device=preds[0].device, dtype=preds[0].dtype)
    return total / terms


def pit_si_snr_huber_xtalk_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    xtalk_lambda: float = 0.05,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT-SI-SNR + Huber with explicit source cross-talk suppression.

    For each PIT permutation, the matched prediction is also projected onto all
    non-matched clean targets.  This separates "target distortion" from
    "contains another source", which SI-SNR/Huber alone do not distinguish.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            pred = preds[pred_idx]
            tgt = tgts[target_idx]
            si_term = -_si_snr_pair_per_item(pred, tgt, eps=eps)
            huber_term = _huber_pair_per_item(pred, tgt, delta=delta)
            total = total + alpha * si_term + beta * huber_term
        xtalk_term = _cross_talk_loss_for_perm(preds, tgts, perm, eps=eps)
        perm_scores.append(total / num_sources + xtalk_lambda * xtalk_term)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def _prepare_cyclic_alphas(
    cyclic_alphas: Optional[Sequence[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cyclic_alphas is None:
        cyclic_alphas = (0.0, 0.05, 0.1, 0.15, 0.2)
    if isinstance(cyclic_alphas, torch.Tensor):
        alphas = cyclic_alphas.to(device=device, dtype=dtype)
    else:
        alphas = torch.tensor(list(cyclic_alphas), device=device, dtype=dtype)
    if alphas.numel() == 0:
        raise ValueError("cyclic_alphas must contain at least one normalized cyclic frequency")
    return alphas.flatten()


def _prepare_cyclic_lags(cyclic_lags: Optional[Sequence[int]]) -> List[int]:
    if cyclic_lags is None:
        cyclic_lags = (0, 1, 2, 4, 8)
    lags = sorted({int(lag) for lag in cyclic_lags if int(lag) >= 0})
    if not lags:
        raise ValueError("cyclic_lags must contain at least one non-negative lag")
    return lags


def _cyclic_corr_profile_iq(
    x_iq: torch.Tensor,
    y_iq: torch.Tensor,
    cyclic_alphas: Optional[Sequence[float]] = None,
    cyclic_lags: Optional[Sequence[int]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Normalized cyclic-correlation magnitude profile for two IQ streams.

    The profile samples R_xy^alpha(tau) across a small set of cyclic
    frequencies and lags:

        mean_n x[n + tau] conj(y[n]) exp(-j 2 pi alpha n)

    Returning log-magnitudes keeps the term robust to global phase and gain,
    while still preserving modulation/pulse-shape periodic structure.
    """
    if x_iq.shape != y_iq.shape or x_iq.ndim != 3 or x_iq.size(1) != 2:
        raise ValueError(f"Expected matching (B,2,T) IQ tensors, got {tuple(x_iq.shape)} and {tuple(y_iq.shape)}")

    work_dtype = torch.float64 if x_iq.dtype == torch.float64 else torch.float32
    x_work = x_iq.to(dtype=work_dtype)
    y_work = y_iq.to(dtype=work_dtype)

    x = _iq_to_complex(x_work)
    y = _iq_to_complex(y_work)
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)

    alphas = _prepare_cyclic_alphas(cyclic_alphas, x_iq.device, work_dtype)
    lags = _prepare_cyclic_lags(cyclic_lags)
    time_len = x.size(-1)
    profiles = []

    for lag in lags:
        if lag >= time_len:
            continue
        if lag == 0:
            x_lag = x
            y_base = y
        else:
            x_lag = x[:, lag:]
            y_base = y[:, : time_len - lag]

        prod = x_lag * torch.conj(y_base)
        n = torch.arange(prod.size(-1), device=x_iq.device, dtype=work_dtype)
        phase = -2.0 * math.pi * alphas[:, None] * n[None, :]
        rotator = torch.complex(torch.cos(phase), torch.sin(phase))
        cyclic_corr = torch.matmul(prod, rotator.transpose(0, 1)) / max(1, prod.size(-1))

        x_power = torch.mean(torch.abs(x_lag).pow(2), dim=-1)
        y_power = torch.mean(torch.abs(y_base).pow(2), dim=-1)
        norm = torch.sqrt(x_power * y_power).clamp_min(eps)
        profiles.append(torch.log1p(torch.abs(cyclic_corr) / norm[:, None]))

    if not profiles:
        return torch.zeros(
            x_iq.size(0),
            0,
            alphas.numel(),
            device=x_iq.device,
            dtype=work_dtype,
        )
    return torch.stack(profiles, dim=1)


def _cyclic_autocorr_profile_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    cyclic_alphas: Optional[Sequence[float]] = None,
    cyclic_lags: Optional[Sequence[int]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_profile = _cyclic_corr_profile_iq(pred, pred, cyclic_alphas, cyclic_lags, eps=eps)
    target_profile = _cyclic_corr_profile_iq(target, target, cyclic_alphas, cyclic_lags, eps=eps)
    if pred_profile.numel() == 0:
        return torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
    return (pred_profile - target_profile).pow(2).mean(dim=(-2, -1))


def _cyclic_cross_profile_penalty_per_item(
    outputs: torch.Tensor,
    num_sources: int,
    cyclic_alphas: Optional[Sequence[float]] = None,
    cyclic_lags: Optional[Sequence[int]] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    preds = _split_iq_sources(outputs, num_sources)
    terms = []
    for left_idx in range(num_sources):
        for right_idx in range(left_idx + 1, num_sources):
            cross_profile = _cyclic_corr_profile_iq(
                preds[left_idx],
                preds[right_idx],
                cyclic_alphas=cyclic_alphas,
                cyclic_lags=cyclic_lags,
                eps=eps,
            )
            if cross_profile.numel() > 0:
                terms.append(cross_profile.pow(2).mean(dim=(-2, -1)))
    if not terms:
        return torch.zeros(outputs.size(0), device=outputs.device, dtype=outputs.dtype)
    return torch.stack(terms, dim=0).mean(dim=0)


def cyclic_autocorr_profile_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    profile_lambda: float = 1.0,
    cross_lambda: float = 0.0,
    cyclic_alphas: Optional[Sequence[float]] = None,
    cyclic_lags: Optional[Sequence[int]] = None,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT-aware cyclic autocorrelation profile loss for communication signals.

    This is a soft cyclostationary prior: each separated source should preserve
    the target's normalized cyclic-correlation profile, and optionally different
    separated streams should have low cross-cyclic correlation.  It is agnostic
    to PSK/QAM/APSK labels and therefore usable across mixed modulation sets.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            total = total + _cyclic_autocorr_profile_pair_per_item(
                preds[pred_idx],
                tgts[target_idx],
                cyclic_alphas=cyclic_alphas,
                cyclic_lags=cyclic_lags,
                eps=eps,
            )
        perm_scores.append(total / num_sources)
    profile_term = torch.stack(perm_scores, dim=0).min(dim=0).values.mean()

    if float(cross_lambda) == 0.0:
        return float(profile_lambda) * profile_term
    cross_term = _cyclic_cross_profile_penalty_per_item(
        outputs,
        num_sources,
        cyclic_alphas=cyclic_alphas,
        cyclic_lags=cyclic_lags,
        eps=eps,
    ).mean()
    return float(profile_lambda) * profile_term + float(cross_lambda) * cross_term


def pit_si_snr_huber_cyclic_profile_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    cyclic_lambda: float = 0.05,
    cyclic_cross_lambda: float = 0.01,
    cyclic_alphas: Optional[Sequence[float]] = None,
    cyclic_lags: Optional[Sequence[int]] = None,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT-SI-SNR + Huber + cyclic autocorrelation profile consistency.

    Unlike constellation/CMA losses, this embeds the cyclostationary structure
    common to digitally modulated communication signals without assuming a
    specific constellation family.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            pred = preds[pred_idx]
            tgt = tgts[target_idx]
            si_term = -_si_snr_pair_per_item(pred, tgt, eps=eps)
            huber_term = _huber_pair_per_item(pred, tgt, delta=delta)
            cyclic_term = _cyclic_autocorr_profile_pair_per_item(
                pred,
                tgt,
                cyclic_alphas=cyclic_alphas,
                cyclic_lags=cyclic_lags,
                eps=eps,
            )
            total = total + alpha * si_term + beta * huber_term + cyclic_lambda * cyclic_term
        perm_scores.append(total / num_sources)

    loss = torch.stack(perm_scores, dim=0).min(dim=0).values.mean()
    if float(cyclic_cross_lambda) != 0.0:
        cross_term = _cyclic_cross_profile_penalty_per_item(
            outputs,
            num_sources,
            cyclic_alphas=cyclic_alphas,
            cyclic_lags=cyclic_lags,
            eps=eps,
        ).mean()
        loss = loss + float(cyclic_cross_lambda) * cross_term
    return loss


def _resize_mixture_to_output(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
    if mixture.size(-1) == target_length:
        return mixture
    return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)


def _source_stack_sum(outputs: torch.Tensor, num_sources: int) -> torch.Tensor:
    sources = to_complex_sources(outputs, num_sources)
    return sources.sum(dim=1)


def _residual_source_correlation_loss(
    residual_iq: torch.Tensor,
    outputs: torch.Tensor,
    num_sources: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    residual_c = _iq_to_complex(residual_iq)
    preds = _split_iq_sources(outputs, num_sources)
    terms = []
    residual_energy = torch.sum(torch.abs(residual_c).pow(2), dim=-1).clamp_min(eps)
    for pred in preds:
        pred_c = _iq_to_complex(pred)
        pred_energy = torch.sum(torch.abs(pred_c).pow(2), dim=-1).clamp_min(eps)
        corr = torch.sum(residual_c * torch.conj(pred_c), dim=-1)
        terms.append(torch.abs(corr).pow(2) / (residual_energy * pred_energy + eps))
    return torch.stack(terms, dim=0).mean(dim=0)


def _residual_whiteness_loss(
    residual_iq: torch.Tensor,
    max_lag: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    residual_c = _iq_to_complex(residual_iq)
    power = torch.mean(torch.abs(residual_c).pow(2), dim=-1).clamp_min(eps)
    lag_terms = []
    max_lag = max(0, min(int(max_lag), residual_c.size(-1) - 1))
    for lag in range(1, max_lag + 1):
        autocorr = torch.mean(residual_c[:, lag:] * torch.conj(residual_c[:, :-lag]), dim=-1)
        lag_terms.append(torch.abs(autocorr / power).pow(2))
    if not lag_terms:
        return torch.zeros(residual_iq.size(0), device=residual_iq.device, dtype=residual_iq.dtype)
    return torch.stack(lag_terms, dim=0).mean(dim=0)


def noise_residual_decorrelation_loss(
    outputs: torch.Tensor,
    mixture: torch.Tensor,
    corr_weight: float = 1.0,
    whiteness_weight: float = 0.25,
    max_lag: int = 8,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Encourage mixture residual to behave like noise, not leaked source content.

    This is intentionally softer than hard mixture consistency.  For clean
    targets and noisy mixtures, hard consistency can inject mixture noise into
    clean source estimates.  Here the residual is allowed to exist, but it is
    discouraged from correlating with separated sources or showing strong
    short-lag structure.
    """
    if outputs.ndim != 3 or mixture.ndim != 3 or mixture.size(1) != 2:
        raise ValueError(f"Expected outputs=(B,2K,L), mixture=(B,2,L), got {outputs.shape}, {mixture.shape}")
    _, channels, target_length = outputs.shape
    if channels % 2 != 0:
        raise ValueError(f"Expected even output channels, got {channels}")
    inferred_sources = channels // 2
    if num_sources is None:
        num_sources = inferred_sources
    elif num_sources != inferred_sources:
        raise ValueError(f"num_sources={num_sources} but inferred {inferred_sources}")

    mixture = _resize_mixture_to_output(mixture, target_length).to(dtype=outputs.dtype)
    source_sum = _source_stack_sum(outputs, num_sources)
    residual_iq = mixture - source_sum

    corr = _residual_source_correlation_loss(residual_iq, outputs, num_sources, eps=eps)
    white = _residual_whiteness_loss(residual_iq, max_lag=max_lag, eps=eps)
    return (float(corr_weight) * corr + float(whiteness_weight) * white).mean()


def pit_si_snr_huber_xtalk_noiseres_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    xtalk_lambda: float = 0.05,
    noiseres_lambda: float = 0.02,
    noiseres_corr_weight: float = 1.0,
    noiseres_whiteness_weight: float = 0.25,
    noiseres_max_lag: int = 8,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    base = pit_si_snr_huber_xtalk_loss(
        outputs,
        targets,
        alpha=alpha,
        beta=beta,
        xtalk_lambda=xtalk_lambda,
        num_sources=num_sources,
        eps=eps,
        delta=delta,
    )
    residual_term = noise_residual_decorrelation_loss(
        outputs,
        mixture,
        corr_weight=noiseres_corr_weight,
        whiteness_weight=noiseres_whiteness_weight,
        max_lag=noiseres_max_lag,
        num_sources=num_sources,
        eps=eps,
    )
    return base + noiseres_lambda * residual_term


def pit_si_snr_huber_rms_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    rms_lambda: float = 0.5,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT over (α·(-SI-SNR) + β·Huber + λ·RMS_gain), one shared permutation.

    Three complementary constraints:
      - SI-SNR  : waveform shape (phase / frequency), scale-invariant
      - Huber   : robust point-wise error (L2 near zero, L1 for outliers)
      - RMS     : global energy / gain anchor — fills the gap that
                  SI-SNR (scale-invariant) and Huber (per-sample, not
                  sensitive to *uniform* gain shift) both miss.

    Args:
        outputs:    (B, 2*K, L)
        targets:    (B, 2*K, L)
        alpha:      SI-SNR weight
        beta:       Huber weight
        rms_lambda: RMS gain constraint weight
        delta:      Huber delta
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts  = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term    = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            rms_term   = _rms_gain_loss_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            total = total + alpha * si_term + beta * huber_term + rms_lambda * rms_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_huber_gpahuber_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    gpahuber_lambda: float = 0.1,
    gpahuber_beta: float = 1.0,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT over SI-SNR + direct Huber + gain/phase-aligned complex Huber.

    The direct Huber keeps the waveform anchored in the original I/Q frame.
    The GPAHuber term adds a communication-style residual error after optimal
    per-source complex gain/phase correction, similar to receiver EVM.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            pred = preds[pred_idx]
            tgt = tgts[target_idx]
            si_term = -_si_snr_pair_per_item(pred, tgt, eps=eps)
            huber_term = _huber_pair_per_item(pred, tgt, delta=delta)
            gpahuber_term = _gain_phase_aligned_huber_pair_per_item(
                pred,
                tgt,
                beta=gpahuber_beta,
                eps=eps,
            )
            total = total + alpha * si_term + beta * huber_term + gpahuber_lambda * gpahuber_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_complex_huber_rms_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    rms_lambda: float = 0.5,
    num_sources: int = None,
    eps: float = 1e-8,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    """
    PIT over alpha*(-SI-SNR) + beta*ComplexHuber + rms_lambda*RMS.

    ComplexHuber is applied to the complex error magnitude
    sqrt((I_hat-I)^2 + (Q_hat-Q)^2), preserving I/Q geometry better than
    separate real-valued Huber terms.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            pred = preds[pred_idx]
            tgt = tgts[target_idx]
            si_term = -_si_snr_pair_per_item(pred, tgt, eps=eps)
            complex_huber = complex_huber_per_source(
                pred.unsqueeze(1),
                tgt.unsqueeze(1),
                beta=huber_beta,
                eps=eps,
            ).squeeze(1)
            rms_term = _rms_gain_loss_pair_per_item(pred, tgt, eps=eps)
            total = total + alpha * si_term + beta * complex_huber + rms_lambda * rms_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_huber_const_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    source_names: Optional[List[str]] = None,
    symbol_stride: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
    const_lambda: float = 0.05,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT over (α·(-SI-SNR) + β·Huber + λ·ConstDist), one shared permutation.

    Constellation constraint steers symbol points toward ideal modulation
    grid (e.g. 8PSK ring) without an explicit amplitude anchor.
    Use this when amplitude is already well-behaved but EVM is still high.

    Args:
        source_names:  list of modulation strings per source, e.g. ['8PSK', '8PSK']
        symbol_stride: down-sample to ~symbol rate before measuring const dist
        const_lambda:  constellation term weight (start small: 0.02~0.1)
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    constellation_bank = _build_constellation_bank(source_names, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts  = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term    = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            const_term = _constellation_distance_per_item(
                preds[pred_idx],
                constellation=constellation_bank[target_idx],
                symbol_stride=symbol_stride,
            )
            total = total + alpha * si_term + beta * huber_term + const_lambda * const_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_huber_rms_const_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    source_names: Optional[List[str]] = None,
    symbol_stride: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
    rms_lambda: float = 0.5,
    const_lambda: float = 0.05,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Full four-term physics-informed loss — one shared PIT permutation:
        α·(-SI-SNR) + β·Huber + λ1·RMS_gain + λ2·ConstDist

    Each term targets a distinct failure mode:
      - SI-SNR  : waveform shape/phase correlation  (scale-invariant)
      - Huber   : robust point-wise reconstruction  (L2 near 0, L1 for outliers)
      - RMS     : global energy / amplitude anchor  (fills SI-SNR's blind spot)
      - CONST   : modulation-domain star-map proximity  (lowers EVM)

    Recommended training recipe:
      Phase 1  (0~50%): PIT-SI-SNR+Huber+RMS  — coarse separation, stable
      Phase 2 (50~100%): this loss            — fine-tune phase & EVM

    Args:
        source_names:  list of modulation strings, e.g. ['8PSK', '8PSK']
        symbol_stride: down-sample factor to ~symbol rate for const. dist.
        alpha:         SI-SNR weight
        beta:          Huber weight
        rms_lambda:    RMS gain weight   (try 0.3~1.0)
        const_lambda:  constellation weight (try 0.02~0.1, start small!)
        delta:         Huber delta
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    constellation_bank = _build_constellation_bank(source_names, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts  = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term    = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            rms_term   = _rms_gain_loss_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            const_term = _constellation_distance_per_item(
                preds[pred_idx],
                constellation=constellation_bank[target_idx],
                symbol_stride=symbol_stride,
            )
            total = total + alpha * si_term + beta * huber_term + rms_lambda * rms_term + const_lambda * const_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def bw_mrstft_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    n_ffts: List[int] = None,
    band_ratio: float = 0.35,
    inband_weight: float = 2.0,
    outband_weight: float = 0.5,
    mag_weight: float = 1.0,
    complex_weight: float = 0.3,
    pit: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Bandwidth-weighted multi-resolution STFT loss for IQ sources.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    n_ffts = n_ffts if n_ffts is not None else [128, 256, 512]

    pair_metric = lambda pred, tgt: _bw_mrstft_pair_per_item(
        pred,
        tgt,
        n_ffts=n_ffts,
        band_ratio=band_ratio,
        inband_weight=inband_weight,
        outband_weight=outband_weight,
        mag_weight=mag_weight,
        complex_weight=complex_weight,
        eps=eps,
    )

    if pit:
        return _pit_reduce(
            outputs,
            targets,
            pair_metric_fn=pair_metric,
            num_sources=num_sources,
            maximize=False,
        )

    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)
    per_source = [pair_metric(pred, tgt) for pred, tgt in zip(preds, tgts)]
    return torch.stack(per_source, dim=0).mean(dim=0).mean()


def evm_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    return _pit_reduce(
        outputs,
        targets,
        pair_metric_fn=lambda pred, tgt: _evm_pair_per_item(pred, tgt, eps=eps),
        num_sources=num_sources,
        maximize=False,
    )


def evm_constellation_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    source_names: Optional[List[str]] = None,
    symbol_stride: int = 1,
    evm_weight: float = 1.0,
    const_weight: float = 0.2,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT-aware EVM + constellation proximity loss.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    constellation_bank = _build_constellation_bank(source_names, num_sources)

    def pair_metric(pred, tgt, tgt_index):
        evm_term = _evm_pair_per_item(pred, tgt, eps=eps)
        const_term = _constellation_distance_per_item(
            pred,
            constellation=constellation_bank[tgt_index],
            symbol_stride=symbol_stride,
        )
        return evm_weight * evm_term + const_weight * const_term

    return _pit_reduce_with_index(
        outputs,
        targets,
        pair_metric_with_index_fn=pair_metric,
        num_sources=num_sources,
        maximize=False,
    )


# ---------------------------------------------------------------------------
# Physics-informed loss building blocks
# ---------------------------------------------------------------------------

def _rms_gain_loss_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    RMS gain constraint: penalises when pred has a different overall energy
    (RMS level) than target.  SI-SNR is scale-invariant, so it cannot see
    pure gain errors — this term fills the gap.

    L = | RMS(pred) - RMS(target) |   (per sample in batch)

    Args:
        pred:   (B, 2, L)  separated IQ
        target: (B, 2, L)  reference IQ
    Returns:
        (B,) non-negative scalar per sample
    """
    pred_flat   = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)
    rms_pred   = torch.sqrt(torch.mean(pred_flat ** 2, dim=-1) + eps)
    rms_target = torch.sqrt(torch.mean(target_flat ** 2, dim=-1) + eps)
    return torch.abs(rms_pred - rms_target)


def _envelope_mse_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Instantaneous envelope MSE: MSE( |pred_complex|, |target_complex| ).
    Constrains the amplitude trajectory without touching phase,
    complementing SI-SNR which is phase-aware but scale-invariant.

    Args:
        pred:   (B, 2, L)
        target: (B, 2, L)
    Returns:
        (B,) MSE of envelopes
    """
    pred_env   = torch.sqrt(pred[:, 0, :] ** 2 + pred[:, 1, :] ** 2 + 1e-12)
    target_env = torch.sqrt(target[:, 0, :] ** 2 + target[:, 1, :] ** 2 + 1e-12)
    return torch.mean((pred_env - target_env) ** 2, dim=-1)


def _cma_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Constant Modulus Algorithm (CMA) penalty.
    Anchors the variance of the predicted envelope to the target envelope.
    L = MSE( |pred|^2, |target|^2 )
    """
    pred_R2 = pred[:, 0, :] ** 2 + pred[:, 1, :] ** 2
    target_R2 = target[:, 0, :] ** 2 + target[:, 1, :] ** 2
    return torch.mean((pred_R2 - target_R2) ** 2, dim=-1)


def _mf_huber_pair_per_item(
    pred: torch.Tensor,
    target: torch.Tensor,
    window: int = 5,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Matched-Filter Differentiable Demod Loss.
    Applies a moving average low-pass filter to simulate matched filtering,
    suppressing out-of-band noise before computing Huber loss.
    """
    if window <= 1:
        return _huber_pair_per_item(pred, target, delta=delta)
    
    # Pad to maintain length
    pad = window // 2
    
    pred_filtered = F.avg_pool1d(
        F.pad(pred, (pad, pad), mode='replicate'), 
        kernel_size=window, 
        stride=1
    )
    target_filtered = F.avg_pool1d(
        F.pad(target, (pad, pad), mode='replicate'), 
        kernel_size=window, 
        stride=1
    )
    
    # If window is even, we might have an extra sample
    if pred_filtered.size(-1) > pred.size(-1):
        pred_filtered = pred_filtered[..., :pred.size(-1)]
        target_filtered = target_filtered[..., :target.size(-1)]
        
    return _huber_pair_per_item(pred_filtered, target_filtered, delta=delta)


# ---------------------------------------------------------------------------
# Composite PIT-aware losses with physics priors
# ---------------------------------------------------------------------------

def pit_si_snr_rms_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    rms_lambda: float = 0.5,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT-SI-SNR + λ · RMS gain constraint.

    Solves: SI-SNR is high but MSE / amplitude drifts because SI-SNR is
    scale-invariant.  The RMS term anchors the absolute energy level.

    Total = min_perm  mean_k [ -SI-SNR_k  +  λ · |RMS(ŝ_k) - RMS(s_k)| ]
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts  = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term  = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            rms_term = _rms_gain_loss_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            total = total + si_term + rms_lambda * rms_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_constellation_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    source_names: Optional[List[str]] = None,
    symbol_stride: int = 1,
    const_lambda: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT-SI-SNR + λ · constellation proximity.

    The constellation term measures how close the separated IQ symbols are
    to the nearest ideal constellation point (e.g. 8PSK ring).  This
    embeds modulation-domain knowledge directly into the loss.

    Total = min_perm  mean_k [ -SI-SNR_k  +  λ · ConstDist_k ]
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    constellation_bank = _build_constellation_bank(source_names, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts  = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term    = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            const_term = _constellation_distance_per_item(
                preds[pred_idx],
                constellation=constellation_bank[target_idx],
                symbol_stride=symbol_stride,
            )
            total = total + si_term + const_lambda * const_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_rms_constellation_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    source_names: Optional[List[str]] = None,
    symbol_stride: int = 1,
    rms_lambda: float = 0.5,
    const_lambda: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Full physics-informed loss:
        PIT-SI-SNR  +  λ1 · RMS gain  +  λ2 · constellation proximity

    Combines data fidelity (SI-SNR), energy anchoring (RMS), and
    modulation-domain prior (constellation).

    Total = min_perm  mean_k [ -SI-SNR_k  +  λ1·RMS_k  +  λ2·Const_k ]
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    constellation_bank = _build_constellation_bank(source_names, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts  = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term    = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            rms_term   = _rms_gain_loss_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            const_term = _constellation_distance_per_item(
                preds[pred_idx],
                constellation=constellation_bank[target_idx],
                symbol_stride=symbol_stride,
            )
            total = total + si_term + rms_lambda * rms_term + const_lambda * const_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def si_snr_mse_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """Legacy non-PIT SI-SNR + MSE (kept for backward compatibility)."""
    return alpha * si_snr_loss(outputs, targets, eps=eps) + beta * F.mse_loss(outputs, targets)


def _si_snr_1d_per_item(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SI-SNR for real 1D sequences.

    Args:
        pred:   (B, L)
        target: (B, L)
    Returns:
        (B,) SI-SNR in dB
    """
    if pred.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected (B, L) tensors, got pred={pred.shape}, target={target.shape}")
    pred_zm = pred - pred.mean(dim=1, keepdim=True)
    tgt_zm = target - target.mean(dim=1, keepdim=True)
    alpha = torch.sum(pred_zm * tgt_zm, dim=1, keepdim=True) / (torch.sum(tgt_zm * tgt_zm, dim=1, keepdim=True) + eps)
    s_target = alpha * tgt_zm
    e_noise = pred_zm - s_target
    ratio = torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def _ctdcrn_pair_loss_per_item(pred_iq: torch.Tensor, tgt_iq: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    CTDCRN paper loss for one IQ pair:
      Loss = MSE_ave - SI-SNR_ave, where each is averaged over real/imag parts.

    Args:
        pred_iq: (B, 2, L)
        tgt_iq:  (B, 2, L)
    Returns:
        (B,) per-item loss
    """
    if pred_iq.ndim != 3 or tgt_iq.ndim != 3 or pred_iq.shape != tgt_iq.shape or pred_iq.size(1) != 2:
        raise ValueError(f"Expected (B,2,L) tensors, got pred={pred_iq.shape}, tgt={tgt_iq.shape}")
    pr, pi = pred_iq[:, 0, :], pred_iq[:, 1, :]
    tr, ti = tgt_iq[:, 0, :], tgt_iq[:, 1, :]

    mse_r = torch.mean((pr - tr) ** 2, dim=1)
    mse_i = torch.mean((pi - ti) ** 2, dim=1)
    mse_ave = 0.5 * (mse_r + mse_i)

    si_r = _si_snr_1d_per_item(pr, tr, eps=eps)
    si_i = _si_snr_1d_per_item(pi, ti, eps=eps)
    si_ave = 0.5 * (si_r + si_i)

    return mse_ave - si_ave


def ctdcrn_loss(outputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Non-PIT CTDCRN loss over K sources:
      mean_k mean_b [ MSE_ave(b,k) - SI-SNR_ave(b,k) ]

    Args:
        outputs: (B, 2*K, L)
        targets: (B, 2*K, L)
    """
    num_sources = _infer_num_sources(outputs, targets, None)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)
    per_source = [_ctdcrn_pair_loss_per_item(p, t, eps=eps) for p, t in zip(preds, tgts)]  # list[(B,)]
    return torch.stack(per_source, dim=0).mean(dim=0).mean()


def pit_ctdcrn_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT version of CTDCRN loss (2-source swap invariant and extends to K).
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            total = total + _ctdcrn_pair_loss_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
        perm_scores.append(total / num_sources)  # (B,)

    all_scores = torch.stack(perm_scores, dim=0)  # (P, B)
    return all_scores.min(dim=0).values.mean()


# ---------------------------------------------------------------------------
# Multi-Resolution STFT Loss  (TF-GridNet / BSRNN style)
# ---------------------------------------------------------------------------

def _mr_stft_single_channel(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    n_ffts: List[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-Resolution STFT loss for a single real-valued 1-D signal.

    For each STFT resolution:
      spectral_convergence = || |S_pred| - |S_tgt| ||_F / || |S_tgt| ||_F
      log_magnitude        = mean | log|S_pred+eps| - log|S_tgt+eps| |

    The total is the average over all resolutions of (sc + log_mag).

    Args:
        pred: (B, L) a single channel (I or Q)
        tgt:  (B, L) corresponding reference
        n_ffts: list of FFT sizes, e.g. [256, 512, 1024, 2048]
    Returns:
        (B,) loss per sample
    """
    pred_f = pred.float()
    tgt_f = tgt.float()

    per_res = []
    for n_fft in n_ffts:
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)

        pred_spec = torch.stft(
            pred_f, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True, center=True, onesided=True,
        )
        tgt_spec = torch.stft(
            tgt_f, n_fft=n_fft, hop_length=hop, win_length=n_fft,
            window=win, return_complex=True, center=True, onesided=True,
        )

        pred_mag = pred_spec.abs()
        tgt_mag = tgt_spec.abs()

        # Spectral convergence (Frobenius-norm ratio)
        sc = torch.norm(tgt_mag - pred_mag, p='fro', dim=(-2, -1)) / (
            torch.norm(tgt_mag, p='fro', dim=(-2, -1)) + eps
        )

        # Magnitude L1 (不用 log，因为 RF 信号带外接近 0，log 梯度会爆炸到 1e8)
        mag_l1 = torch.mean(
            torch.abs(pred_mag - tgt_mag),
            dim=(-2, -1),
        )

        per_res.append(sc + mag_l1)

    return torch.stack(per_res, dim=0).mean(dim=0)   # (B,)


def _mr_stft_iq_pair_per_item(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    n_ffts: List[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MR-STFT loss for one IQ source pair: average loss over I and Q channels.

    Args:
        pred: (B, 2, L)  predicted IQ
        tgt:  (B, 2, L)  reference IQ
    Returns:
        (B,) loss per sample
    """
    loss_i = _mr_stft_single_channel(pred[:, 0, :], tgt[:, 0, :], n_ffts, eps)
    loss_q = _mr_stft_single_channel(pred[:, 1, :], tgt[:, 1, :], n_ffts, eps)
    return 0.5 * (loss_i + loss_q)


def multi_resolution_stft_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    n_ffts: List[int] = None,
    pit: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-Resolution STFT Loss for IQ signal separation.

    Applies STFT at multiple FFT sizes to measure spectral distance between
    predicted and target waveforms. This captures both fine frequency structure
    (large n_fft) and transient details (small n_fft) simultaneously.

    Used by TF-GridNet, BSRNN, BS-RoFormer and virtually all SOTA speech/music
    separation systems.  Multiple papers report +2~3 dB improvement when added
    to SI-SNR loss.

    Works with ANY model — it operates on the output waveform, not on model
    internals.

    Args:
        outputs:     (B, 2*K, L)  model predictions [I1, Q1, I2, Q2, ...]
        targets:     (B, 2*K, L)  ground truth
        num_sources: K (inferred from channel dim if None)
        n_ffts:      list of FFT sizes (default: [256, 512, 1024, 2048])
        pit:         if True, use Permutation-Invariant Training
        eps:         numerical stability
    Returns:
        Scalar loss
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    if n_ffts is None:
        n_ffts = [256, 512, 1024, 2048]

    pair_fn = lambda pred, tgt: _mr_stft_iq_pair_per_item(
        pred, tgt, n_ffts=n_ffts, eps=eps
    )

    if pit:
        return _pit_reduce(
            outputs, targets,
            pair_metric_fn=pair_fn,
            num_sources=num_sources,
            maximize=False,
        )
    else:
        preds = _split_iq_sources(outputs, num_sources)
        tgts = _split_iq_sources(targets, num_sources)
        per_source = [pair_fn(p, t) for p, t in zip(preds, tgts)]
        return torch.stack(per_source, dim=0).mean(dim=0).mean()


# ---------------------------------------------------------------------------
# Mixture Consistency Loss
# ---------------------------------------------------------------------------

def mixture_consistency_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
) -> torch.Tensor:
    """
    Mixture Consistency constraint: separated sources must sum to the mixture.

    Physical law:  mixture = source_1 + source_2 + ... + source_K
    Loss:          L1( sum_of_separated_sources, mixture )

    Used by TF-GridNet (TASLP 2023) to achieve 23.5 dB SI-SDR on WSJ0-2mix.
    Can be used as a standalone loss or combined with any other loss function.

    NOTE: this loss uses output + mixture (ignores targets), so it provides an
    unsupervised constraint that does not require clean reference signals.

    Args:
        outputs:  (B, 2*K, L)  model predictions [I1, Q1, I2, Q2, ...]
        targets:  (B, 2*K, L)  ground truth (unused, kept for API compatibility)
        mixture:  (B, 2, L)    original mixed signal (I_mix, Q_mix)
    Returns:
        Scalar L1 loss
    """
    B, C, L = outputs.shape
    num_sources = C // 2

    # Sum all predicted sources: for each source k, channels are [2k, 2k+1]
    # Sum the I channels and Q channels separately
    pred_sum_I = sum(outputs[:, 2 * k, :] for k in range(num_sources))
    pred_sum_Q = sum(outputs[:, 2 * k + 1, :] for k in range(num_sources))

    mix_I = mixture[:, 0, :]
    mix_Q = mixture[:, 1, :]

    # L1 consistency
    loss_I = F.l1_loss(pred_sum_I, mix_I)
    loss_Q = F.l1_loss(pred_sum_Q, mix_Q)
    return 0.5 * (loss_I + loss_Q)


# ---------------------------------------------------------------------------
# Composite losses combining MR-STFT and Mixture Consistency
# ---------------------------------------------------------------------------

def pit_si_snr_huber_mrstft_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    mrstft_lambda: float = 0.3,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
    n_ffts: List[int] = None,
) -> torch.Tensor:
    """
    PIT-SI-SNR + Huber + Multi-Resolution STFT Loss.

    Three complementary terms evaluated synchronously under a single PIT permutation:
      - SI-SNR    : waveform shape / phase fidelity (scale-invariant)
      - Huber     : robust point-wise reconstruction
      - MR-STFT   : multi-scale spectral fidelity (phase + magnitude)
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    if n_ffts is None:
        n_ffts = [256, 512, 1024, 2048]

    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term    = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            mrstft_term = _mr_stft_iq_pair_per_item(preds[pred_idx], tgts[target_idx], n_ffts=n_ffts, eps=eps)

            total = total + alpha * si_term + beta * huber_term + mrstft_lambda * mrstft_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)  # (P, B)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_huber_mrstft_mixcons_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mixture: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    mrstft_lambda: float = 0.3,
    mixcons_lambda: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
    n_ffts: List[int] = None,
) -> torch.Tensor:
    """
    Full SOTA loss: PIT-SI-SNR + Huber + MR-STFT + Mixture Consistency.
    """
    # 1. Compute the shared PIT terms
    pit_main_loss = pit_si_snr_huber_mrstft_loss(
        outputs, targets,
        alpha=alpha, beta=beta, mrstft_lambda=mrstft_lambda,
        num_sources=num_sources, eps=eps, delta=delta, n_ffts=n_ffts,
    )

    # 2. Add Mixture Consistency (which doesn't depend on permutation target-matching)
    mixcons_term = mixture_consistency_loss(outputs, targets, mixture)

    return pit_main_loss + mixcons_lambda * mixcons_term


def pit_si_snr_huber_cma_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    cma_lambda: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT-SI-SNR + Huber + CMA Penalty.
    Adds Constant Modulus constraints.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            cma_term = _cma_pair_per_item(preds[pred_idx], tgts[target_idx])

            total = total + alpha * si_term + beta * huber_term + cma_lambda * cma_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


def pit_si_snr_huber_mf_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.5,
    mf_lambda: float = 0.5,
    mf_window: int = 5,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    PIT-SI-SNR + Huber + Matched-Filter Huber.
    Applies an average pooling filter (to approximate RRC) before computing an additional Huber loss,
    focusing on in-band symbol recovery.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    preds = _split_iq_sources(outputs, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    perm_scores = []
    for perm in _get_permutations(num_sources):
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            mf_term = _mf_huber_pair_per_item(preds[pred_idx], tgts[target_idx], window=mf_window, delta=delta)

            total = total + alpha * si_term + beta * huber_term + mf_lambda * mf_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)
    return all_scores.min(dim=0).values.mean()


# ---------------------------------------------------------------------------
# AMR (Automatic Modulation Recognition) — modulation class vocabulary
# ---------------------------------------------------------------------------

# Canonical modulation vocabulary shared across loss, evaluation, and model.
# All references to modulation class IDs must use this dict.
MOD_VOCAB = {
    'BPSK':    0,
    'QPSK':    1,
    '8PSK':    2,
    '16PSK':   3,
    '4QAM':    4,
    '16QAM':   5,
    '64QAM':   6,
    '128QAM':  7,
    '256QAM':  8,
    '16APSK':  9,
    '32APSK':  10,
}

# Inverse vocab for convenience
MOD_VOCAB_INV = {v: k for k, v in MOD_VOCAB.items()}


def get_mod_labels_from_data_choice(
    data_choice: str,
    num_sources: int,
) -> Optional[List[int]]:
    """Infer per-source modulation class IDs from data_choice.

    Uses the BER-evaluation helper _infer_modulations_from_data_choice
    (already implemented in evaluation.py) and maps names to MOD_VOCAB.

    Returns:
        list of K integer class IDs, or None if the data_choice is unknown.
    """
    # Avoid circular import — evaluation.py imports from loss.py, so we use
    # a local re-implementation of the modulation inference logic.
    dc = str(data_choice).upper().replace('_', '-')

    _SINGLE = {
        '8PSK-M': '8PSK', '8PSK-M-NS': '8PSK',
        '8PSK-BURST': '8PSK', '8PSK-BURST-NS': '8PSK',
        '8PSK-M-8192': '8PSK', '8PSK-M-16384': '8PSK', '8PSK-M-32768': '8PSK',
        '8PSK-M-8192-NS': '8PSK', '8PSK-M-16384-NS': '8PSK', '8PSK-M-32768-NS': '8PSK',
        '8PSK-RS': '8PSK', '8PSK-RS-NS': '8PSK',
        '8PSK-A': '8PSK', '8PSK-B': '8PSK', '8PSK-C': '8PSK', '8PSK-D': '8PSK',
        '8PSK-E': '8PSK', '8PSK-F': '8PSK', '8PSK-G': '8PSK',
        '8PSK-H': '8PSK', '8PSK-I': '8PSK', '8PSK-J': '8PSK',
        '8PSK-K': '8PSK', '8PSK-L': '8PSK',
    }
    if dc in _SINGLE:
        mod_names = [_SINGLE[dc]] * num_sources
    else:
        _MIXED = {
            'QPSK-16APSK': ['QPSK', '16APSK'],
            'QPSK-16APSK-NS': ['QPSK', '16APSK'],
            'QPSK+16APSK-A': ['QPSK', '16APSK'],
            'QPSK+16APSK-B': ['QPSK', '16APSK'],
            '16QAM-64QAM': ['16QAM', '64QAM'],
            '16QAM-128QAM': ['16QAM', '128QAM'],
            '64QAM-64QAM': ['64QAM', '64QAM'],
            '64QAM-128QAM': ['64QAM', '128QAM'],
            '16QAM-64QAM-128QAM': ['16QAM', '64QAM', '128QAM'],
            'QAM-A': ['16QAM', '64QAM'],
            'QAM-B': ['64QAM', '64QAM'],
            'QAM-C': ['64QAM', '128QAM'],
            'QAM-D': ['16QAM', '64QAM', '128QAM'],
            'QAM-E': ['16QAM', '64QAM', '128QAM'],
        }
        if dc in _MIXED:
            mod_names = _MIXED[dc]
            if len(mod_names) < num_sources:
                mod_names = mod_names + [mod_names[-1]] * (num_sources - len(mod_names))
            mod_names = mod_names[:num_sources]
        else:
            return None

    return [MOD_VOCAB[name] for name in mod_names]


# ---------------------------------------------------------------------------
# AMR Joint Loss — PIT-SI-SNR+Huber + CrossEntropy (AMR)
# ---------------------------------------------------------------------------

def pit_si_snr_huber_amr_loss(
    sep_output: torch.Tensor,
    targets: torch.Tensor,
    cls_logits: List[torch.Tensor],
    mod_labels: List[int],
    alpha: float = 1.0,
    beta: float = 1.0,
    cls_weight: float = 0.1,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """PIT-aware joint separation + AMR classification loss.

    The key design: PIT selects the best permutation based on the separation
    loss (α·(-SI-SNR) + β·Huber), then we evaluate CrossEntropy for the
    AMR classifier under that **same** permutation.  This guarantees the
    classification labels are always aligned with the correct model output
    even when sources are swapped by PIT.

    Args:
        sep_output: (B, 2*K, L) separated waveforms from the model
        targets:    (B, 2*K, L) ground truth waveforms
        cls_logits: list of K tensors, each (B, num_mod_classes) from AMR head
        mod_labels: list of K integers — the correct modulation class ID
                    for target source 0, 1, ..., K-1
        alpha:      SI-SNR weight
        beta:       Huber weight
        cls_weight: weight for the AMR classification loss
        delta:      Huber delta
    Returns:
        Scalar combined loss
    """
    num_sources = _infer_num_sources(sep_output, targets, num_sources)
    preds = _split_iq_sources(sep_output, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    B = sep_output.shape[0]
    device = sep_output.device

    # --- 1. PIT: find best permutation per sample based on separation loss ---
    all_perms = list(_get_permutations(num_sources))
    perm_scores = []
    for perm in all_perms:
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            total = total + alpha * si_term + beta * huber_term
        perm_scores.append(total / num_sources)  # (B,)

    all_scores = torch.stack(perm_scores, dim=0)  # (P, B)
    best_perm_indices = all_scores.argmin(dim=0)   # (B,) — best perm index per sample
    sep_loss = all_scores[best_perm_indices, torch.arange(B, device=device)].mean()

    # --- 2. AMR CrossEntropy under the best permutation ---
    # For each sample b, the best permutation maps:
    #   predicted source pred_idx → target source target_idx
    # So cls_logits[pred_idx][b] should be classified as mod_labels[target_idx].
    #
    # Build a (B, K) label tensor and a (B, K, C) logits tensor, both
    # permuted according to each sample's best perm.
    mod_label_tensor = torch.tensor(mod_labels, dtype=torch.long, device=device)  # (K,)

    # Stack cls_logits: (K, B, C)
    cls_stack = torch.stack(cls_logits, dim=0)  # (K, B, C)

    # Build per-sample reordered logits and labels
    ce_loss = torch.tensor(0.0, device=device)
    for b in range(B):
        perm = all_perms[best_perm_indices[b].item()]
        for target_idx, pred_idx in enumerate(perm):
            logit = cls_stack[pred_idx, b, :].unsqueeze(0)  # (1, C)
            label = mod_label_tensor[target_idx].unsqueeze(0)  # (1,)
            ce_loss = ce_loss + F.cross_entropy(logit, label)

    ce_loss = ce_loss / (B * num_sources)

    return sep_loss + cls_weight * ce_loss


# ---------------------------------------------------------------------------
# Soft Demodulation Joint Loss — PIT-SI-SNR+Huber + BCE (SoftDemod)
# ---------------------------------------------------------------------------

def pit_si_snr_huber_demod_loss(
    sep_output: torch.Tensor,
    targets: torch.Tensor,
    demod_outputs,
    bit_targets: tuple,
    alpha: float = 1.0,
    beta: float = 1.0,
    demod_weight: float = 0.5,
    symbol_weight: float = 0.5,
    num_sources: int = None,
    eps: float = 1e-8,
    delta: float = 1.0,
) -> torch.Tensor:
    """PIT-aware joint separation + soft demodulation loss.

    Same design as pit_si_snr_huber_amr_loss:
    1. PIT selects best permutation based on separation loss
    2. Binary Cross-Entropy for soft bits is evaluated under that same
       permutation to guarantee bit-output alignment.

    Args:
        sep_output:   (B, 2*K, L) separated waveforms
        targets:      (B, 2*K, L) ground truth waveforms
        demod_outputs: dict or list from demod head
        bit_targets:  tuple of K tensors, each (B, num_bits) ground truth bits
        alpha:        SI-SNR weight
        beta:         Huber weight
        demod_weight: weight for the demodulation BCE loss
        delta:        Huber delta
    Returns:
        Scalar combined loss
    """
    num_sources = _infer_num_sources(sep_output, targets, num_sources)
    preds = _split_iq_sources(sep_output, num_sources)
    tgts = _split_iq_sources(targets, num_sources)

    B = sep_output.shape[0]
    device = sep_output.device
    if isinstance(demod_outputs, dict):
        bit_logits = demod_outputs.get('bit_logits')
        symbol_logits = demod_outputs.get('symbol_logits')
    else:
        bit_logits = demod_outputs
        symbol_logits = None

    def _bits_to_symbol_labels(bits_1d: torch.Tensor):
        if bits_1d.numel() == 0 or (bits_1d.numel() % 3) != 0:
            return None
        bits_grouped = bits_1d.view(-1, 3).long()
        return bits_grouped[:, 0] * 4 + bits_grouped[:, 1] * 2 + bits_grouped[:, 2]

    # --- 1. PIT: find best permutation per sample based on separation loss ---
    all_perms = list(_get_permutations(num_sources))
    perm_scores = []
    for perm in all_perms:
        total = 0.0
        for target_idx, pred_idx in enumerate(perm):
            si_term = -_si_snr_pair_per_item(preds[pred_idx], tgts[target_idx], eps=eps)
            huber_term = _huber_pair_per_item(preds[pred_idx], tgts[target_idx], delta=delta)
            total = total + alpha * si_term + beta * huber_term
        perm_scores.append(total / num_sources)

    all_scores = torch.stack(perm_scores, dim=0)  # (P, B)
    best_perm_indices = all_scores.argmin(dim=0)   # (B,)
    sep_loss = all_scores[best_perm_indices, torch.arange(B, device=device)].mean()

    # --- 2. Soft demod BCE under the best permutation ---
    # bit_logits[pred_idx][b] should match bit_targets[target_idx][b]
    bce_loss = torch.tensor(0.0, device=device)
    ce_symbol_loss = torch.tensor(0.0, device=device)
    ce_symbol_terms = 0
    for b in range(B):
        perm = all_perms[best_perm_indices[b].item()]
        for target_idx, pred_idx in enumerate(perm):
            logit = bit_logits[pred_idx][b]         # (num_bits,)
            target_bits = bit_targets[target_idx][b].float().to(device)  # (num_bits,)
            bce_loss = bce_loss + F.binary_cross_entropy_with_logits(
                logit, target_bits, reduction='mean'
            )
            if symbol_logits is not None and symbol_logits[pred_idx] is not None:
                target_symbols = _bits_to_symbol_labels(bit_targets[target_idx][b].to(device))
                if target_symbols is not None:
                    pred_symbols = symbol_logits[pred_idx][b]  # (num_symbols, 8)
                    max_symbols = min(pred_symbols.shape[0], target_symbols.shape[0])
                    ce_symbol_loss = ce_symbol_loss + F.cross_entropy(
                        pred_symbols[:max_symbols],
                        target_symbols[:max_symbols],
                    )
                    ce_symbol_terms += 1

    bce_loss = bce_loss / (B * num_sources)
    if ce_symbol_terms > 0:
        ce_symbol_loss = ce_symbol_loss / ce_symbol_terms
    else:
        ce_symbol_loss = torch.tensor(0.0, device=device)

    return sep_loss + demod_weight * bce_loss + symbol_weight * ce_symbol_loss

def qam_lattice_regularizer_on_output(y_hat, axis_levels=(4, 8, 16), tau=0.03):
    """
    y_hat: [B, 4, L] = two separated sources
    weak regularizer, not main loss.
    """
    from models.IQU_QAMRDEPriorAdapter import complex_rms, make_square_qam_levels
    s1 = y_hat[:, 0:2, :]
    s2 = y_hat[:, 2:4, :]

    reg = 0.0
    for s in [s1, s2]:
        scale = complex_rms(s)
        s_norm = s / (scale + 1e-8)

        best_err = None
        for m in axis_levels:
            levels = make_square_qam_levels(m).to(s.device, s.dtype)
            sr = s_norm[:, 0, :]
            si = s_norm[:, 1, :]

            dist_r = (sr.unsqueeze(-1) - levels.view(1, 1, -1)) ** 2
            dist_i = (si.unsqueeze(-1) - levels.view(1, 1, -1)) ** 2

            wr = torch.softmax(-dist_r / tau, dim=-1)
            wi = torch.softmax(-dist_i / tau, dim=-1)

            qr = torch.sum(wr * levels.view(1, 1, -1), dim=-1)
            qi = torch.sum(wi * levels.view(1, 1, -1), dim=-1)

            err = ((qr - sr) ** 2 + (qi - si) ** 2).mean()

            if best_err is None:
                best_err = err
            else:
                best_err = torch.minimum(best_err, err)

        reg = reg + best_err

    return reg / 2.0


def _topology_source_vector(source: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Modulation-shape statistics for one separated IQ source.

    The vector is deliberately low-order: amplitude spread for APSK/QAM,
    axis occupancy for QAM-like signals, phase-step concentration for PSK-like
    signals, and kurtosis for high-order constellation density.
    """
    if source.ndim != 3 or source.size(1) != 2:
        raise ValueError(f"Expected source shape (B, 2, L), got {tuple(source.shape)}")

    rms = torch.sqrt(source.pow(2).mean(dim=(1, 2), keepdim=True) + eps)
    x = source / rms
    i = x[:, 0, :]
    q = x[:, 1, :]
    amp = torch.sqrt(i.pow(2) + q.pow(2) + eps)

    axis_abs_mean = torch.stack([i.abs().mean(dim=-1), q.abs().mean(dim=-1)], dim=-1)
    axis_abs_std = torch.stack([i.abs().std(dim=-1), q.abs().std(dim=-1)], dim=-1)
    amp_mean = amp.mean(dim=-1, keepdim=True)
    amp_std = amp.std(dim=-1, keepdim=True)
    amp_kurtosis = ((amp - amp.mean(dim=-1, keepdim=True)).pow(4).mean(dim=-1, keepdim=True)) / (
        amp.var(dim=-1, keepdim=True, unbiased=False).pow(2) + eps
    )

    if source.size(-1) > 1:
        z0_i, z0_q = i[:, :-1], q[:, :-1]
        z1_i, z1_q = i[:, 1:], q[:, 1:]
        cross_r = z1_i * z0_i + z1_q * z0_q
        cross_i = z1_q * z0_i - z1_i * z0_q
        cross_norm = torch.sqrt(cross_r.pow(2) + cross_i.pow(2) + eps)
        phase_cos = (cross_r / cross_norm).mean(dim=-1, keepdim=True)
        phase_sin = (cross_i / cross_norm).mean(dim=-1, keepdim=True)
        phase_concentration = torch.sqrt(phase_cos.pow(2) + phase_sin.pow(2) + eps)
    else:
        phase_cos = torch.zeros(source.size(0), 1, device=source.device, dtype=source.dtype)
        phase_sin = torch.zeros_like(phase_cos)
        phase_concentration = torch.zeros_like(phase_cos)

    return torch.cat(
        [
            axis_abs_mean,
            axis_abs_std,
            amp_mean,
            amp_std,
            amp_kurtosis,
            phase_cos,
            phase_sin,
            phase_concentration,
        ],
        dim=-1,
    )


def topology_profile_from_targets(
    targets: torch.Tensor,
    num_sources: int = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Return named topology statistics from target IQ sources for inspection/logging."""
    num_sources = _infer_num_sources(targets, targets, num_sources)
    target_sources = to_complex_sources(targets, num_sources)
    vectors = torch.stack(
        [_topology_source_vector(target_sources[:, idx], eps=eps) for idx in range(num_sources)],
        dim=1,
    )
    return {
        "axis_abs_mean": vectors[..., 0:2],
        "axis_abs_std": vectors[..., 2:4],
        "amp_mean": vectors[..., 4],
        "amp_std": vectors[..., 5],
        "kurtosis": vectors[..., 6],
        "phase_cos": vectors[..., 7],
        "phase_sin": vectors[..., 8],
        "phase_concentration": vectors[..., 9],
    }


def topology_stat_loss_on_output(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int = None,
    axis_weight: float = 1.0,
    amp_weight: float = 1.0,
    phase_weight: float = 0.5,
    kurtosis_weight: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PIT-aware topology-stat auxiliary loss on separated outputs.

    This is intentionally separate from constellation projection. It compares
    low-order shape statistics between predicted and target sources, allowing
    QAM/APSK structure to guide training after separation rather than forcing
    the raw mixture toward a constellation.
    """
    num_sources = _infer_num_sources(outputs, targets, num_sources)
    pred_sources = to_complex_sources(outputs, num_sources)
    target_sources = to_complex_sources(targets, num_sources)

    pred_vec = torch.stack(
        [_topology_source_vector(pred_sources[:, idx], eps=eps) for idx in range(num_sources)],
        dim=1,
    )
    target_vec = torch.stack(
        [_topology_source_vector(target_sources[:, idx], eps=eps) for idx in range(num_sources)],
        dim=1,
    )

    weights = torch.tensor(
        [
            axis_weight,
            axis_weight,
            axis_weight,
            axis_weight,
            amp_weight,
            amp_weight,
            kurtosis_weight,
            phase_weight,
            phase_weight,
            phase_weight,
        ],
        device=outputs.device,
        dtype=outputs.dtype,
    )

    perm_losses = []
    for perm in _get_permutations(num_sources):
        aligned = target_vec[:, list(perm), :]
        diff = (pred_vec - aligned).pow(2) * weights.view(1, 1, -1)
        perm_losses.append(diff.mean(dim=(1, 2)))
    return torch.stack(perm_losses, dim=0).min(dim=0).values.mean()


def separation_mechanism_loss_on_output(
    outputs: torch.Tensor,
    mixture: torch.Tensor,
    num_sources: int = None,
    mix_weight: float = 1.0,
    corr_weight: float = 1.0,
    energy_weight: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Target-free separation-mechanism auxiliary loss.

    It encourages additive mixture consistency while discouraging duplicated
    or strongly correlated source estimates. This is intentionally opposed to
    topology-stat supervision: it uses only separated outputs plus the mixture.
    """
    if outputs.ndim != 3 or outputs.size(1) % 2 != 0:
        raise ValueError(f"Expected outputs shape (B, 2*K, L), got {tuple(outputs.shape)}")
    if mixture.ndim != 3 or mixture.size(1) != 2:
        raise ValueError(f"Expected mixture shape (B, 2, L), got {tuple(mixture.shape)}")

    inferred_sources = outputs.size(1) // 2
    if num_sources is None:
        num_sources = inferred_sources
    elif int(num_sources) != inferred_sources:
        raise ValueError(f"num_sources={num_sources} but inferred {inferred_sources}")

    sources = to_complex_sources(outputs, num_sources)
    mixture = _resize_mixture_to_output(mixture, outputs.size(-1)).to(device=outputs.device, dtype=outputs.dtype)
    mix_term = F.huber_loss(sources.sum(dim=1), mixture, reduction="mean", delta=1.0)

    corr_terms = []
    for i in range(num_sources):
        si = sources[:, i] - sources[:, i].mean(dim=-1, keepdim=True)
        si_power = si.pow(2).sum(dim=1).mean(dim=-1)
        for j in range(i + 1, num_sources):
            sj = sources[:, j] - sources[:, j].mean(dim=-1, keepdim=True)
            sj_power = sj.pow(2).sum(dim=1).mean(dim=-1)
            real_cov = (si[:, 0] * sj[:, 0] + si[:, 1] * sj[:, 1]).mean(dim=-1)
            imag_cov = (si[:, 1] * sj[:, 0] - si[:, 0] * sj[:, 1]).mean(dim=-1)
            corr_terms.append((real_cov.pow(2) + imag_cov.pow(2)) / (si_power * sj_power + eps))
    corr_term = torch.stack(corr_terms, dim=0).mean() if corr_terms else outputs.new_tensor(0.0)

    source_energy = sources.pow(2).sum(dim=2).mean(dim=-1)
    energy_share = source_energy / (source_energy.sum(dim=1, keepdim=True) + eps)
    balanced_share = outputs.new_full((1,), 1.0 / float(num_sources))
    energy_term = (energy_share - balanced_share).pow(2).mean()

    return mix_weight * mix_term + corr_weight * corr_term + energy_weight * energy_term


def cross_covariance_penalty(pred_c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute statistical independence penalty (cross-covariance) between sources.
    pred_c: (B, K, 2, T) where 2 is I and Q.
    Returns: (B,) penalty scalar.
    """
    B, K, _, T = pred_c.shape
    if K != 2:
        return torch.tensor(0.0, device=pred_c.device)
    
    # Zero mean over time
    pred_mean = pred_c.mean(dim=-1, keepdim=True)
    pred_centered = pred_c - pred_mean  # (B, K, 2, T)
    
    s1_I, s1_Q = pred_centered[:, 0, 0, :], pred_centered[:, 0, 1, :]
    s2_I, s2_Q = pred_centered[:, 1, 0, :], pred_centered[:, 1, 1, :]
    
    # E[s1 * s2*] = E[(I1 + jQ1)(I2 - jQ2)] = E[I1 I2 + Q1 Q2] + j E[Q1 I2 - I1 Q2]
    real_cov = (s1_I * s2_I + s1_Q * s2_Q).mean(dim=-1)
    imag_cov = (s1_Q * s2_I - s1_I * s2_Q).mean(dim=-1)
    
    # Normalize by variances
    var1 = (s1_I**2 + s1_Q**2).mean(dim=-1)
    var2 = (s2_I**2 + s2_Q**2).mean(dim=-1)
    
    norm = torch.sqrt(var1 * var2 + eps)
    correlation2 = (real_cov**2 + imag_cov**2) / (norm**2 + eps)
    
    return correlation2


def pit_si_snr_huber_ind_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_sources: int,
    alpha: float = 0.5,
    beta: float = 1.0,
    alpha_ind: float = 0.1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    PIT SI-SNR + Huber + Statistical Independence (Decorrelation) Penalty.
    """
    # Base PIT loss
    base_loss = pit_si_snr_huber_loss(pred, target, alpha=alpha, beta=beta, num_sources=num_sources, eps=eps)
    
    # Independence penalty
    pred_c = to_complex_sources(pred, num_sources)
    ind_penalty = cross_covariance_penalty(pred_c, eps).mean()
    
    total_loss = base_loss + alpha_ind * ind_penalty
    return total_loss


def pit_si_snr_huber_mc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    num_sources: int,
    alpha: float = 0.5,
    beta: float = 1.0,
    mc_lambda: float = 0.05,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    PIT SI-SNR + Huber + Mixture Consistency Penalty.
    """
    # Base PIT loss
    base_loss = pit_si_snr_huber_loss(pred, target, alpha=alpha, beta=beta, num_sources=num_sources, eps=eps)
    
    # Mixture Consistency
    pred_c = to_complex_sources(pred, num_sources)  # [B, num_sources, 2, L]
    mixture_pred = pred_c.sum(dim=1)  # [B, 2, L]
    
    mc_loss = F.huber_loss(mixture_pred, mixture, reduction='mean', delta=1.0)
    
    total_loss = base_loss + mc_lambda * mc_loss
    return total_loss
