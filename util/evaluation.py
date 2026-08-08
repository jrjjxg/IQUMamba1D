import os
import time
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
from util.visualize import plot_signals, plot_correlation_vs_snr_enhanced
from util.metrics import (
    si_snr_real, si_snr_paper, si_snr_repo, si_snr_paper_joint, si_snr_complex, mse,
    scale_aligned_mse,
    similarity_coeff_complex, pearson_complex_abs,
    pit_si_snr_complex_persample,
    pit_si_snr_real_persample,
    pit_mse_persample,
    phase_flip_rate,
    _split_sources,
    strict_ber_iq_from_bits,
    oracle_ber_iq_from_bits,
    reference_ber_iq_from_bits,
)

EPS = 1e-12
DIAG_EMPIRICAL_SNR = True
DIAG_PHASE = True
DIAG_PHASE_ALIGN = True


def calculate_correlation(prediction, target):
    """Calculate overall correlation coefficient"""
    if prediction.ndim == 3 and prediction.shape[1] > 2 and prediction.shape[1] % 2 == 0:
        num_sources = prediction.shape[1] // 2
        corrs = calculate_correlation_per_source(prediction, target, num_sources)
        if not corrs:
            return torch.tensor(0.0, device=prediction.device)
        return torch.stack([c if isinstance(c, torch.Tensor) else torch.tensor(c, device=prediction.device) for c in corrs]).mean()
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    corr_matrix = torch.corrcoef(torch.stack([pred_flat, target_flat]))
    return corr_matrix[0, 1] if not torch.isnan(corr_matrix[0, 1]) else 0


def calculate_correlation_per_source(prediction, target, num_sources):
    """
    Calculate correlation coefficient for each source
    
    Args:
        prediction: Prediction output (B, C, L), C = num_sources * 2
        target: Target signal (B, C, L), C = num_sources * 2
        num_sources: Number of sources
    
    Returns:
        list: Correlation coefficients for each source
    """
    correlations = []
    
    for i in range(num_sources):
        # Each source contains real and imaginary parts in two channels
        real_idx = i * 2
        imag_idx = i * 2 + 1
        
        # Extract prediction and target for current source
        pred_source = prediction[:, [real_idx, imag_idx], :].flatten()
        target_source = target[:, [real_idx, imag_idx], :].flatten()
        
        # Calculate correlation coefficient
        if len(pred_source) > 0 and len(target_source) > 0:
            corr_matrix = torch.corrcoef(torch.stack([pred_source, target_source]))
            corr = corr_matrix[0, 1] if not torch.isnan(corr_matrix[0, 1]) else 0
            correlations.append(corr)
        else:
            correlations.append(torch.tensor(0.0))
    
    return correlations


def calculate_metrics_per_source(prediction, target, num_sources):
    """
    Calculate metrics for each source
    
    Args:
        prediction: Prediction output (B, C, L)
        target: Target signal (B, C, L)
        num_sources: Number of sources
    
    Returns:
        dict: Metrics dictionary for each source
    """
    source_metrics = {}
    
    for i in range(num_sources):
        # Each source contains real and imaginary parts in two channels
        real_idx = i * 2
        imag_idx = i * 2 + 1
        
        # Extract prediction and target for current source
        pred_source = prediction[:, [real_idx, imag_idx], :]  # (B, 2, L)
        target_source = target[:, [real_idx, imag_idx], :]    # (B, 2, L)
        
        # Calculate metrics
        try:
            # Calculate correlation coefficient
            corr = calculate_correlation(pred_source, target_source)

            # New metrics
            si_snr_r = si_snr_real(pred_source, target_source)
            si_snr_p = si_snr_paper(pred_source, target_source)
            si_snr_repo_v = si_snr_repo(pred_source, target_source)
            si_snr_c = si_snr_complex(pred_source, target_source)
            mse_val = mse(pred_source, target_source)
            scale_aligned_mse_val = scale_aligned_mse(pred_source, target_source)
            sc_val = similarity_coeff_complex(pred_source, target_source)
            pears = pearson_complex_abs(pred_source, target_source)
            
        except Exception as e:
            print(f"Error calculating metrics for source {i}: {e}")
            corr = torch.tensor(0.0)
            si_snr_r = torch.tensor(0.0)
            si_snr_p = torch.tensor(0.0)
            si_snr_repo_v = torch.tensor(0.0)
            si_snr_c = torch.tensor(0.0)
            mse_val = torch.tensor(0.0)
            scale_aligned_mse_val = torch.tensor(0.0)
            sc_val = torch.tensor(0.0)
            pears = torch.tensor(0.0)
        
        source_metrics[i] = {
            'Correlation': corr,
            'SI-SNR_real': si_snr_r,
            'SI-SNR_paper': si_snr_p,
            'SI-SNR_repo': si_snr_repo_v,
            'SI-SNR_complex': si_snr_c,
            'MSE': mse_val,
            'ScaleAligned_MSE': scale_aligned_mse_val,
            'SC': sc_val,
            'Pearson': pears,
        }
    
    return source_metrics


def reorder_outputs_by_per_sample_perm(outputs: torch.Tensor, best_perm_per_sample, num_sources: int) -> torch.Tensor:
    """
    Reorder prediction channels to match target source order per sample.

    Args:
        outputs: (B, 2K, L)
        best_perm_per_sample: list of length B, each a permutation tuple of length K.
                              perm[k] means target source k is matched by predicted source perm[k].
        num_sources: K

    Returns:
        Tensor (B, 2K, L) with channels reordered per sample.
    """
    reordered = torch.empty_like(outputs)
    B = outputs.shape[0]

    if not isinstance(best_perm_per_sample, list) or len(best_perm_per_sample) != B:
        raise ValueError("best_perm_per_sample must be a list with length equal to batch size")

    for b in range(B):
        perm = best_perm_per_sample[b]
        for k in range(num_sources):
            pred_idx = perm[k]
            reordered[b, 2 * k: 2 * k + 2, :] = outputs[b, 2 * pred_idx: 2 * pred_idx + 2, :]

    return reordered


def reorder_outputs_for_eval(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int,
    pit_metric: str = "si_snr_complex",
) -> Tuple[torch.Tensor, List[Tuple[int, ...]]]:
    """Apply the configured source-order alignment before metric computation."""
    pit_metric = str(pit_metric).lower()
    if pit_metric == "none":
        identity = tuple(range(num_sources))
        return outputs, [identity for _ in range(targets.shape[0])]
    if num_sources < 2:
        best_perm_per_sample = [tuple(range(num_sources)) for _ in range(targets.shape[0])]
        return outputs, best_perm_per_sample

    if pit_metric == "si_snr_complex":
        _, best_perm_per_sample = pit_si_snr_complex_persample(outputs, targets, num_sources)
    elif pit_metric == "si_snr_real":
        _, best_perm_per_sample = pit_si_snr_real_persample(outputs, targets, num_sources)
    elif pit_metric == "mse":
        _, best_perm_per_sample = pit_mse_persample(outputs, targets, num_sources)
    else:
        raise ValueError(
            f"Unsupported eval PIT metric '{pit_metric}'. "
            "Expected one of: none, si_snr_complex, si_snr_real, mse."
        )

    return reorder_outputs_by_per_sample_perm(outputs, best_perm_per_sample, num_sources), best_perm_per_sample


def get_multitask_mode(amr_mode: str = 'sep_only', demod_mode: str = 'sep_only'):
    """Resolve the forward mode for multi-task models."""
    if amr_mode in ('cls_only', 'joint'):
        return amr_mode
    if demod_mode in ('demod_only', 'joint'):
        return demod_mode
    return None


def forward_with_multitask_mode(model, inputs: torch.Tensor,
                                amr_mode: str = 'sep_only',
                                demod_mode: str = 'sep_only',
                                num_bits: int = None):
    """Run model forward with the correct explicit mode when needed."""
    mode = get_multitask_mode(amr_mode=amr_mode, demod_mode=demod_mode)
    kwargs = {}
    if demod_mode in ('demod_only', 'joint') and num_bits is not None:
        kwargs['num_bits'] = int(num_bits)
    if mode is not None:
        return model(inputs, mode=mode, **kwargs)
    return model(inputs, **kwargs)


def extract_separation_output(model_output):
    """Return the waveform separation tensor from a model output."""
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def validate_source_tensor(tensor: torch.Tensor, num_sources: int, name: str) -> torch.Tensor:
    """Validate the common separator tensor contract: [B, 2*K, L]."""
    expected_channels = 2 * int(num_sources)
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        actual = type(tensor).__name__ if not torch.is_tensor(tensor) else tuple(tensor.shape)
        raise ValueError(f"{name}: expected tensor [B, {expected_channels}, L], got {actual}")
    if tensor.size(1) != expected_channels:
        raise ValueError(
            f"{name}: expected [B, {expected_channels}, L] for {num_sources} sources, "
            f"got {tuple(tensor.shape)}"
        )
    return tensor


def extract_demod_outputs(model_output):
    """Return demod bit/symbol logits from a multitask model output."""
    if not isinstance(model_output, tuple) or len(model_output) < 2:
        return None, None
    aux = model_output[1]
    if isinstance(aux, dict):
        return aux.get('bit_logits'), aux.get('symbol_logits')
    return aux, None


def bits_to_symbol_labels(bits_1d: torch.Tensor, bits_per_symbol: int = 3):
    """Pack grouped bits into symbol class labels, e.g. 3 bits -> 0..7."""
    if bits_1d is None:
        return None
    if bits_1d.numel() == 0 or (bits_1d.numel() % bits_per_symbol) != 0:
        return None
    bits_grouped = bits_1d.view(-1, bits_per_symbol).long()
    weights = (2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=bits_grouped.device)).view(1, -1)
    return torch.sum(bits_grouped * weights, dim=1)


def split_tensor_by_channel(x: torch.Tensor, N: int):
    """
    x   : Tensor of shape (B, C, L)
    N   : Expected number of splits (must be even, and N <= C)
    return : list of N tensors, each (B, 2, L)
    """
    B, C, L = x.shape
    assert N <= C, "N must be <= number of channels C"
    
    # Split channel dimension into (N, 2) then transpose
    y = x.view(B, N, 2, L).transpose(1, 2)          # (B, 2, N, L)
    return [t.squeeze(2) for t in y.split(1, dim=2)]


def ensure_B2L(x: torch.Tensor) -> torch.Tensor:
    """Normalize mixture to shape (B, 2, L)."""
    if x.ndim != 3:
        raise ValueError(f"mixture shape must be 3D, got {x.shape}")
    if x.shape[-1] == 2:
        return x.permute(0, 2, 1).contiguous()
    if x.shape[1] == 2:
        return x.contiguous()
    raise ValueError(f"unrecognized mixture layout {x.shape}")


def ensure_BK2L(x: torch.Tensor, num_sources: int) -> torch.Tensor:
    """Normalize targets/outputs to shape (B, K, 2, L)."""
    if x.ndim != 3:
        raise ValueError(f"target/output shape must be 3D, got {x.shape}")
    if x.shape[-1] == 2 * num_sources:
        x = x.permute(0, 2, 1).contiguous()
    if x.shape[1] == 2 * num_sources:
        B, C, L = x.shape
        return x.view(B, num_sources, 2, L).contiguous()
    raise ValueError(f"unrecognized target/output layout {x.shape}, num_sources={num_sources}")


def power_B(x_B2L: torch.Tensor) -> torch.Tensor:
    """Average power per sample, returns (B,)."""
    return (x_B2L ** 2).sum(dim=1).mean(dim=1)


@torch.no_grad()
def empirical_snr_db_values(mixture_B2L: torch.Tensor, target_BK2L: torch.Tensor) -> torch.Tensor:
    """Return per-sample empirical SNR in dB."""
    sum_s = target_BK2L.sum(dim=1)  # (B, 2, L)
    noise = mixture_B2L - sum_s
    p_sig = power_B(sum_s)
    p_noi = power_B(noise) + EPS
    return 10.0 * torch.log10(p_sig / p_noi)


@torch.no_grad()
def estimate_phase_deg(pred_B2L: torch.Tensor, tgt_B2L: torch.Tensor) -> torch.Tensor:
    """Estimate global IQ rotation (degrees) of pred relative to target."""
    pI, pQ = pred_B2L[:, 0, :], pred_B2L[:, 1, :]
    tI, tQ = tgt_B2L[:, 0, :], tgt_B2L[:, 1, :]
    dot = (pI * tI + pQ * tQ).sum(dim=1)
    cross = (pI * tQ - pQ * tI).sum(dim=1)
    theta = torch.atan2(cross, dot + EPS)
    return theta * (180.0 / math.pi)


@torch.no_grad()
def rotate_by_minus_theta(pred_B2L: torch.Tensor, theta_deg: torch.Tensor) -> torch.Tensor:
    """Rotate pred by -theta (degrees)."""
    theta = theta_deg * (math.pi / 180.0)
    c = torch.cos(theta).view(-1, 1, 1)
    s = torch.sin(theta).view(-1, 1, 1)
    pI = pred_B2L[:, 0:1, :]
    pQ = pred_B2L[:, 1:2, :]
    I2 = pI * c + pQ * s
    Q2 = pQ * c - pI * s
    return torch.cat([I2, Q2], dim=1)


class RunningStats:
    """Running mean/std for 1D tensors."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: torch.Tensor):
        x = x.detach().float().view(-1)
        if x.numel() == 0:
            return
        m = x.numel()
        batch_mean = x.mean().item()
        batch_m2 = ((x - batch_mean) ** 2).sum().item()
        if self.n == 0:
            self.n = m
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.n + m
        self.mean = self.mean + delta * m / total
        self.m2 = self.m2 + batch_m2 + delta * delta * self.n * m / total
        self.n = total

    def mean_std(self):
        if self.n < 2:
            return self.mean, 0.0
        return self.mean, math.sqrt(self.m2 / (self.n - 1))


def _infer_modulations_from_data_choice(data_choice: str, num_sources: int):
    """Infer per-source modulation types from data_choice for BER calculation.

    Returns:
        list of modulation name strings (one per source), or None if unknown.
    """
    dc = str(data_choice).upper().replace('_', '-')
    _MAP = {
        # 8PSK variants (all sources are 8PSK)
        '8PSK-M': '8PSK',
        '8PSK-M-NS': '8PSK',
        '8PSK-BURST': '8PSK',
        '8PSK-BURST-NS': '8PSK',
        '8PSK-M-8192': '8PSK',
        '8PSK-M-16384': '8PSK',
        '8PSK-M-32768': '8PSK',
        '8PSK-M-8192-NS': '8PSK',
        '8PSK-M-16384-NS': '8PSK',
        '8PSK-M-32768-NS': '8PSK',
        '8PSK-RS': '8PSK',
        '8PSK-RS-NS': '8PSK',
        '8PSK-A': '8PSK',
        '8PSK-B': '8PSK',
        '8PSK-C': '8PSK',
        '8PSK-D': '8PSK',
        '8PSK-E': '8PSK',
        '8PSK-F': '8PSK',
        '8PSK-G': '8PSK',
        '8PSK-H': '8PSK',
        '8PSK-I': '8PSK',
        '8PSK-J': '8PSK',
        '8PSK-K': '8PSK',
        '8PSK-L': '8PSK',
    }
    # Single-modulation datasets
    if dc in _MAP:
        return [_MAP[dc]] * num_sources

    # Mixed-modulation datasets
    _MIXED = {
        'QPSK-16APSK': ['QPSK_MATLAB', '16APSK_MATLAB'],
        'QPSK-16APSK-NS': ['QPSK_MATLAB', '16APSK_MATLAB'],
        'QPSK+16APSK-A': ['QPSK_MATLAB', '16APSK_MATLAB'],
        'QPSK+16APSK-B': ['QPSK_MATLAB', '16APSK_MATLAB'],
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
        mods = _MIXED[dc]
        return mods[:num_sources] if len(mods) >= num_sources else mods + [mods[-1]] * (num_sources - len(mods))

    return None


def _resolve_subset_indices(dataset) -> Tuple[object, Optional[List[int]]]:
    """Resolve nested Subset objects into (base_dataset, base_indices)."""
    if not isinstance(dataset, torch.utils.data.Subset):
        return dataset, None

    base_dataset = dataset.dataset
    base_indices = list(dataset.indices)
    while isinstance(base_dataset, torch.utils.data.Subset):
        parent_indices = list(base_dataset.indices)
        base_indices = [parent_indices[i] for i in base_indices]
        base_dataset = base_dataset.dataset
    return base_dataset, base_indices


def _file_indices_for_subset(base_dataset, subset_indices: Optional[List[int]], snr: Optional[float] = None) -> List[int]:
    """Return file indices whose full frame ranges are present in the subset."""
    file_meta = getattr(base_dataset, "_file_meta", None)
    if not file_meta:
        return []

    if subset_indices is None:
        indices_set = None
    else:
        indices_set = set(int(i) for i in subset_indices)

    selected = []
    for file_idx, meta in enumerate(file_meta):
        if snr is not None and float(meta.get("snr", float("nan"))) != float(snr):
            continue
        start = int(meta["start"])
        end = int(meta["end"])
        if indices_set is not None:
            frame_indices = range(start, end)
            if not all(idx in indices_set for idx in frame_indices):
                continue
        selected.append(file_idx)
    return selected


def _build_dataset_slice_tensors(base_dataset, start: int, end: int, num_sources: int):
    """Build (inputs, targets, bits_tuple_or_none) tensors for a contiguous file slice."""
    mix_np = np.asarray(base_dataset.mixture[start:end], dtype=np.float32)      # (B, L, 2)
    inputs = torch.from_numpy(np.transpose(mix_np, (0, 2, 1))).contiguous()    # (B, 2, L)

    target_channels = []
    for source_idx in range(num_sources):
        src_np = np.asarray(base_dataset.signals[source_idx][start:end], dtype=np.float32)  # (B, L, 2)
        target_channels.append(np.transpose(src_np[:, :, 0], (0, 1)))
        target_channels.append(np.transpose(src_np[:, :, 1], (0, 1)))
    targets = torch.from_numpy(np.stack(target_channels, axis=1)).contiguous()  # (B, 2K, L)

    bits_tuple = None
    if getattr(base_dataset, "bits_per_source", None):
        bits_tuple = tuple(
            torch.from_numpy(np.asarray(base_dataset.bits_per_source[k][start:end], dtype=np.uint8)).contiguous()
            for k in range(num_sources)
        )
    return inputs, targets, bits_tuple


def _concat_source_stream(x: torch.Tensor, source_idx: int, valid_length: Optional[int] = None) -> torch.Tensor:
    """(B, 2K, L) -> (1, 2, B*valid_length) for one source."""
    src = x[:, 2 * source_idx: 2 * source_idx + 2, :]
    if valid_length is not None:
        src = src[:, :, :int(valid_length)]
    src = src.permute(1, 0, 2).contiguous()
    return src.reshape(1, 2, -1)


def compute_stream_ber_for_matlab_h5_dataset(
    model,
    base_dataset,
    signal_names,
    device,
    file_indices,
    max_files: int = 0,
    forward_batch_size: int = 64,
    ber_offset_search: bool = False,
    ber_variant: str = "strict",
    return_per_source: bool = False,
    modulations: Optional[List[str]] = None,
    num_sources: Optional[int] = None,
    protocol: Optional[str] = None,
    amr_mode: str = "sep_only",
    demod_mode: str = "sep_only",
    eval_pit_metric: str = "si_snr_complex",
):
    """Compute file-level BER.

    QPSK+16APSK is generated as a continuous file-level symbol stream and
    then sliced into 4096-sample model frames, so its file BER must preserve
    the continuous stream instead of resetting timing at every frame boundary.
    """
    if num_sources is None:
        num_sources = len(signal_names) if signal_names else getattr(base_dataset, "num_sources", 2)
    if modulations is None or len(modulations) < num_sources:
        raise ValueError("Stream BER requires known per-source modulation metadata.")

    file_meta = getattr(base_dataset, "_file_meta", None)
    if not file_meta:
        raise ValueError("Dataset does not expose _file_meta; cannot compute file-level BER.")

    if max_files and max_files > 0:
        file_indices = list(file_indices)[:int(max_files)]
    else:
        file_indices = list(file_indices)

    if not file_indices:
        raise ValueError("No file indices provided for stream BER evaluation.")

    ber_variant_key = str(ber_variant).lower()
    if ber_variant_key == "strict":
        ber_fn = strict_ber_iq_from_bits
    elif ber_variant_key == "oracle":
        ber_fn = oracle_ber_iq_from_bits
    elif ber_variant_key == "reference":
        ber_fn = reference_ber_iq_from_bits
    else:
        raise ValueError(f"Unknown ber_variant: {ber_variant}")

    per_source_values = [[] for _ in range(num_sources)]

    with torch.no_grad():
        for file_idx in file_indices:
            meta = file_meta[int(file_idx)]
            start = int(meta["start"])
            end = int(meta["end"])
            valid_frame_length = int(
                meta.get("valid_frame_length", getattr(base_dataset, "valid_frame_length", 0) or 0)
            )
            inputs_cpu, targets_cpu, bits_tuple_cpu = _build_dataset_slice_tensors(base_dataset, start, end, num_sources)
            if bits_tuple_cpu is None:
                continue

            reordered_batches = []
            target_batches = []
            for batch_start in range(0, inputs_cpu.shape[0], max(1, int(forward_batch_size))):
                batch_end = min(inputs_cpu.shape[0], batch_start + max(1, int(forward_batch_size)))
                inputs = inputs_cpu[batch_start:batch_end].to(device)
                targets = targets_cpu[batch_start:batch_end].to(device)
                batch_bits = tuple(b[batch_start:batch_end] for b in bits_tuple_cpu)
                outputs = forward_with_multitask_mode(
                    model,
                    inputs,
                    amr_mode=amr_mode,
                    demod_mode=demod_mode,
                    num_bits=int(batch_bits[0].shape[-1]) if batch_bits else None,
                )
                sep_outputs = extract_separation_output(outputs)
                if isinstance(sep_outputs, (list, tuple)):
                    sep_outputs = sep_outputs[-1]
                validate_source_tensor(sep_outputs, num_sources, 'model separation output')
                validate_source_tensor(targets, num_sources, 'targets')
                reordered, _ = reorder_outputs_for_eval(
                    sep_outputs,
                    targets,
                    num_sources,
                    pit_metric=eval_pit_metric,
                )
                if valid_frame_length > 0:
                    reordered = reordered[..., :valid_frame_length]
                    targets = targets[..., :valid_frame_length]
                reordered_batches.append(reordered.cpu())
                target_batches.append(targets.cpu())

            preds_file = torch.cat(reordered_batches, dim=0)
            tgts_file = torch.cat(target_batches, dim=0)

            for source_idx in range(num_sources):
                if ber_variant_key == "reference":
                    pred_stream = _concat_source_stream(preds_file, source_idx)
                    tgt_stream = _concat_source_stream(tgts_file, source_idx)
                    bits_stream = bits_tuple_cpu[source_idx].reshape(1, -1)
                    sps_by_source = meta.get("samples_per_symbol_by_source") or []
                    cfo_by_source = meta.get("cfo_hz_by_source") or []
                    source_sps = (
                        sps_by_source[source_idx]
                        if source_idx < len(sps_by_source)
                        else meta.get("samples_per_symbol")
                    )
                    source_cfo = (
                        cfo_by_source[source_idx]
                        if source_idx < len(cfo_by_source)
                        else 0.0
                    )
                    ber_val = ber_fn(
                        pred_stream.float(),
                        tgt_stream.float(),
                        bits_stream,
                        modulation=modulations[source_idx],
                        sps=int(source_sps),
                        sample_rate_hz=float(meta["sample_rate_hz"]),
                        cfo_hz=float(source_cfo),
                        rrc_alpha=float(meta.get("rrc_alpha", 0.35) or 0.35),
                        rrc_span=int(meta.get("rrc_span", 20) or 20),
                    )
                elif str(protocol).upper() == "8PSK-A":
                    pred_eval = preds_file[:, 2 * source_idx: 2 * source_idx + 2, :]
                    tgt_eval = tgts_file[:, 2 * source_idx: 2 * source_idx + 2, :]
                    bits_eval = bits_tuple_cpu[source_idx]
                    ber_val = ber_fn(
                        pred_eval.float(),
                        tgt_eval.float(),
                        bits_eval,
                        modulation=modulations[source_idx],
                        offset_search=bool(ber_offset_search),
                        protocol=protocol,
                    )
                else:
                    pred_stream = _concat_source_stream(preds_file, source_idx)
                    tgt_stream = _concat_source_stream(tgts_file, source_idx)
                    bits_stream = bits_tuple_cpu[source_idx].reshape(1, -1)
                    ber_val = ber_fn(
                        pred_stream.float(),
                        tgt_stream.float(),
                        bits_stream,
                        modulation=modulations[source_idx],
                        offset_search=bool(ber_offset_search),
                        protocol=protocol,
                    )
                if torch.isfinite(ber_val):
                    per_source_values[source_idx].append(float(ber_val.item()))

    src_means = [
        float(np.mean(vals)) if vals else float("nan")
        for vals in per_source_values
    ]
    finite_vals = [v for v in src_means if np.isfinite(v)]
    overall = float(np.mean(finite_vals)) if finite_vals else float("nan")
    if return_per_source:
        return overall, src_means
    return overall

def test_model(model, snr_loaders, criterion, device, logger, results_folder, 
               num_plots=1, num_points=1024, input_size=1024, data_choice='2018',
               signal_names=None, save_artifacts=True, 
               report_ber: bool = False, 
               ber_offset_search: bool = False,
               ber_mode: str = "file",
               ber_num_files: int = 2,
               ber_compute_oracle: bool = False,
               amr_mode: str = 'sep_only',
               demod_mode: str = 'sep_only',
               eval_pit_metric: str = "si_snr_complex",
               report_phase_flip: bool = False,
               phase_flip_tolerance_deg: float = 45.0,
               phase_flip_min_sc: float = 0.0,
               phase_flip_mode: str = "either"):
    """
    Enhanced test_model function that supports independent evaluation for each source
    """
    model.eval()
    
    num_source = len(signal_names) if signal_names else 2
    snr_metrics = {}
    _needs_bits = getattr(criterion, 'needs_bits', False)
    _needs_mixture = getattr(criterion, 'needs_mixture', False)
    
    # Store metrics for each source for CSV
    all_metrics_list = []
    
    # Store correlation coefficients for each source for plotting
    source_correlations = {i: [] for i in range(num_source)}
    overall_correlations = []
    snr_list = []

    # Infer modulation types for BER from data_choice
    ber_modulations = _infer_modulations_from_data_choice(data_choice, num_source) if report_ber else None
    if report_ber and ber_modulations:
        logger.info(f"[BER] Inferred modulations from data_choice='{data_choice}': {ber_modulations}")
    elif report_ber:
        logger.info(f"[BER] Could not infer modulations from data_choice='{data_choice}', BER will be NaN.")
    logger.info(f"[Eval] Source alignment mode for reporting: {eval_pit_metric}")
    dual_report_fixed_and_pit = str(eval_pit_metric).lower() == "none"
    if dual_report_fixed_and_pit:
        logger.info(
            "[Eval] Dual reporting enabled: fixed-order metrics are primary; "
            "PIT(si_snr_complex) metrics are reported alongside them."
        )
    if report_ber and not ber_compute_oracle:
        logger.info("[BER] Oracle BER is disabled by default; pass --ber_compute_oracle for the slower debug upper bound.")
    if report_phase_flip:
        logger.info(
            f"[PhaseFlip] Enabled: tolerance={phase_flip_tolerance_deg:.1f} deg from +/-180, "
            f"min_SC={phase_flip_min_sc:.3f}, mode={phase_flip_mode}"
        )
    if report_ber and str(ber_mode).lower() == "file":
        data_choice_key = str(data_choice).upper().replace("_", "-")
        if str(data_choice).upper() == "8PSK-A":
            logger.info(
                "[BER] file-mode BER for 8PSK-A uses the protocol-aware receiver "
                "frame-by-frame and then averages within each file."
            )
        elif data_choice_key in {"QPSK-16APSK", "QPSK-16APSK-NS", "QPSK+16APSK-A"}:
            logger.info(
                "[BER] file-mode BER for QPSK+16APSK preserves the continuous "
                "file-level symbol stream across 4096-sample model frames."
            )
        else:
            logger.info(
                "[BER] file-mode BER uses one constant complex-gain alignment over each "
                "concatenated file. Datasets with residual CFO/phase drift or non-symbol-"
                "synchronous framing can therefore report overly pessimistic BER; use "
                "frame mode for a more stable trend."
            )

    def _frame_level_ber_selfcheck(targets_tensor, bits_tuple):
        """Return target-vs-target BER as a sanity check for frame-level BER."""
        if bits_tuple is None or not ber_modulations:
            return None
        strict_vals = []
        oracle_vals = []
        for source_idx in range(num_source):
            if source_idx >= len(ber_modulations) or source_idx >= len(bits_tuple):
                continue
            tgt_k = targets_tensor[:, 2 * source_idx: 2 * source_idx + 2, :]
            bits_k = bits_tuple[source_idx]
            ber_strict = strict_ber_iq_from_bits(
                tgt_k.float(),
                tgt_k.float(),
                bits_k,
                modulation=ber_modulations[source_idx],
                offset_search=ber_offset_search,
                protocol=data_choice,
            )
            ber_oracle = torch.tensor(float("nan"))
            if ber_compute_oracle:
                ber_oracle = oracle_ber_iq_from_bits(
                    tgt_k.float(),
                    tgt_k.float(),
                    bits_k,
                    modulation=ber_modulations[source_idx],
                    offset_search=ber_offset_search,
                    protocol=data_choice,
                )
            if torch.isfinite(ber_strict):
                strict_vals.append(float(ber_strict.item()))
            if torch.isfinite(ber_oracle):
                oracle_vals.append(float(ber_oracle.item()))
        if not strict_vals and not oracle_vals:
            return None
        strict_mean = float(np.mean(strict_vals)) if strict_vals else float("nan")
        oracle_mean = float(np.mean(oracle_vals)) if oracle_vals else float("nan")
        return strict_mean, oracle_mean

    def _file_level_ber_selfcheck(base_dataset, file_indices):
        """Return target-vs-target BER as a sanity check for file-level BER."""
        if not file_indices or not ber_modulations:
            return None
        file_meta = getattr(base_dataset, "_file_meta", None)
        if not file_meta:
            return None
        file_idx = int(list(file_indices)[0])
        meta = file_meta[file_idx]
        start = int(meta["start"])
        end = int(meta["end"])
        valid_frame_length = int(meta.get("valid_frame_length", 0) or 0)
        _, targets_cpu, bits_tuple_cpu = _build_dataset_slice_tensors(base_dataset, start, end, num_source)
        if bits_tuple_cpu is None:
            return None

        strict_vals = []
        oracle_vals = []
        for source_idx in range(num_source):
            if source_idx >= len(ber_modulations) or source_idx >= len(bits_tuple_cpu):
                continue
            if str(data_choice).upper() == "8PSK-A":
                tgt_eval = targets_cpu[:, 2 * source_idx: 2 * source_idx + 2, :]
                if valid_frame_length > 0:
                    tgt_eval = tgt_eval[..., :valid_frame_length]
                bits_eval = bits_tuple_cpu[source_idx]
                ber_strict = strict_ber_iq_from_bits(
                    tgt_eval.float(),
                    tgt_eval.float(),
                    bits_eval,
                    modulation=ber_modulations[source_idx],
                    offset_search=ber_offset_search,
                    protocol=data_choice,
                )
                ber_oracle = torch.tensor(float("nan"))
                if ber_compute_oracle:
                    ber_oracle = oracle_ber_iq_from_bits(
                        tgt_eval.float(),
                        tgt_eval.float(),
                        bits_eval,
                        modulation=ber_modulations[source_idx],
                        offset_search=ber_offset_search,
                        protocol=data_choice,
                    )
            else:
                tgt_stream = _concat_source_stream(
                    targets_cpu,
                    source_idx,
                    valid_length=valid_frame_length if valid_frame_length > 0 else None,
                )
                bits_stream = bits_tuple_cpu[source_idx].reshape(1, -1)
                ber_strict = strict_ber_iq_from_bits(
                    tgt_stream.float(),
                    tgt_stream.float(),
                    bits_stream,
                    modulation=ber_modulations[source_idx],
                    offset_search=ber_offset_search,
                    protocol=data_choice,
                )
                ber_oracle = torch.tensor(float("nan"))
                if ber_compute_oracle:
                    ber_oracle = oracle_ber_iq_from_bits(
                        tgt_stream.float(),
                        tgt_stream.float(),
                        bits_stream,
                        modulation=ber_modulations[source_idx],
                        offset_search=ber_offset_search,
                        protocol=data_choice,
                    )
            if torch.isfinite(ber_strict):
                strict_vals.append(float(ber_strict.item()))
            if torch.isfinite(ber_oracle):
                oracle_vals.append(float(ber_oracle.item()))
        if not strict_vals and not oracle_vals:
            return None
        strict_mean = float(np.mean(strict_vals)) if strict_vals else float("nan")
        oracle_mean = float(np.mean(oracle_vals)) if oracle_vals else float("nan")
        return strict_mean, oracle_mean

    def _ber_selfcheck_failed(strict_mean, oracle_mean, threshold: float = 0.05) -> bool:
        if np.isfinite(strict_mean) and strict_mean > threshold:
            return True
        if np.isfinite(oracle_mean) and oracle_mean > threshold:
            return True
        return False
    
    with torch.no_grad():
        # Global warmup
        for _ in range(3):
            dummy_input = torch.randn(1, 2, input_size).to(device)
            _ = forward_with_multitask_mode(
                model,
                dummy_input,
                amr_mode=amr_mode,
                demod_mode=demod_mode,
                num_bits=None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    total_time = 0.0
    total_samples = 0
    filelevel_ber_fallback_logged = False

    for snr, loader in sorted(snr_loaders.items(), key=lambda x: x[0]):
        loader_dataset = getattr(loader, "dataset", None)
        loader_base_dataset, loader_base_indices = _resolve_subset_indices(loader_dataset) if loader_dataset is not None else (None, None)
        file_idxs = _file_indices_for_subset(loader_base_dataset, loader_base_indices, snr=float(snr)) if loader_base_dataset is not None else []
        test_loss = 0.0
        test_corr = 0.0
        test_si_snr_real = 0.0
        test_si_snr_paper = 0.0
        test_si_snr_repo = 0.0
        test_si_snr_complex = 0.0
        test_mse = 0.0
        test_scale_aligned_mse = 0.0
        test_sc = 0.0
        test_pearson = 0.0
        waveform_metric_names = (
            'Correlation', 'SI-SNR_real', 'SI-SNR_paper', 'SI-SNR_repo',
            'SI-SNR_complex', 'MSE', 'ScaleAligned_MSE', 'SC', 'Pearson',
        )
        pit_metric_sums = {name: 0.0 for name in waveform_metric_names}
        pit_source_metric_sums = {
            i: {name: 0.0 for name in waveform_metric_names}
            for i in range(num_source)
        }
        pit_source_metric_den = {
            i: {name: 0 for name in waveform_metric_names}
            for i in range(num_source)
        }
        test_ber_strict = 0.0
        test_ber_strict_den = 0
        test_ber_oracle = 0.0
        test_ber_oracle_den = 0
        test_demod_bit_correct = 0
        test_demod_bit_total = 0
        test_demod_symbol_correct = 0
        test_demod_symbol_total = 0
        phase_flip_source_count = 0
        phase_flip_source_total = 0
        phase_flip_sample_count = 0
        phase_flip_sample_total = 0
        phase_flip_source_counts = [0 for _ in range(num_source)]
        phase_flip_source_totals = [0 for _ in range(num_source)]
        phase_flip_phase_abs_sum = 0.0
        phase_flip_dist_pi_sum = 0.0
        phase_flip_neg_alpha_sum = 0.0
        phase_flip_diag_den = 0
        use_filelevel_ber = bool(report_ber) and str(ber_mode).lower() == "file"
        ber_metric_enabled = bool(report_ber)
        ber_selfcheck_done = False

        if ber_metric_enabled and use_filelevel_ber and loader_base_dataset is not None and file_idxs:
            selfcheck = _file_level_ber_selfcheck(loader_base_dataset, file_idxs)
            ber_selfcheck_done = selfcheck is not None
            if selfcheck is not None:
                strict_self, oracle_self = selfcheck
                if _ber_selfcheck_failed(strict_self, oracle_self):
                    logger.warning(
                        f"[BER] Disabled file-level BER at SNR={snr}: target-vs-target "
                        f"self-check is already too high (strict={strict_self:.4f}, "
                        f"oracle={oracle_self:.4f}). This usually means the saved bits "
                        "and IQ stream are not directly comparable under the current "
                        "demodulation assumptions."
                    )
                    ber_metric_enabled = False
                    use_filelevel_ber = False

        emp_snr_stats = RunningStats()
        phase_stats = [RunningStats() for _ in range(num_source)]
        
        # Cumulative metrics for each source
        per_source_metric_names = [
            'Correlation',
            'SI-SNR_real', 'SI-SNR_paper', 'SI-SNR_repo', 'SI-SNR_complex',
            'MSE', 'ScaleAligned_MSE', 'SC', 'Pearson'
        ]
        if report_ber:
            per_source_metric_names.extend(['BER_strict', 'BER_oracle'])
        if report_phase_flip:
            per_source_metric_names.append('Phase_Flip_Rate')
        source_metrics_sum = {i: {name: 0.0 for name in per_source_metric_names} for i in range(num_source)}
        source_metrics_den = {i: {name: 0 for name in per_source_metric_names} for i in range(num_source)}
        
        visualization_count = 0
        visualization_done = False
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if not isinstance(batch, (list, tuple)) or len(batch) < 3:
                    raise ValueError("Evaluation batches must contain mixture, targets and SNR")
                inputs, targets, _snr_label, *batch_extras = batch
                bits = next(
                    (
                        tuple(value)
                        for value in batch_extras
                        if isinstance(value, (tuple, list)) and len(value) == num_source
                    ),
                    None,
                )

                inputs, targets = inputs.to(device), targets.to(device)
                if bits is not None:
                    bits = tuple(b.to(device) for b in bits)
                

                start_time = time.perf_counter()
                outputs = forward_with_multitask_mode(
                    model,
                    inputs,
                    amr_mode=amr_mode,
                    demod_mode=demod_mode,
                    num_bits=int(bits[0].shape[-1]) if bits is not None else None,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                if batch_idx >= 0:
                    total_time += (end_time - start_time)
                    total_samples += inputs.size(0)

                if DIAG_EMPIRICAL_SNR or DIAG_PHASE or DIAG_PHASE_ALIGN:
                    inputs_B2L = ensure_B2L(inputs)
                    targets_BK2L = ensure_BK2L(targets, num_source)
                else:
                    inputs_B2L = None
                    targets_BK2L = None

                sep_outputs = extract_separation_output(outputs)
                if isinstance(sep_outputs, (list, tuple)):
                    sep_outputs = sep_outputs[-1]
                validate_source_tensor(sep_outputs, num_source, 'model separation output')
                validate_source_tensor(targets, num_source, 'targets')

                if DIAG_EMPIRICAL_SNR:
                    snr_vals = empirical_snr_db_values(inputs_B2L, targets_BK2L)
                    emp_snr_stats.update(snr_vals)

                if DIAG_PHASE or DIAG_PHASE_ALIGN:
                    outputs_BK2L = ensure_BK2L(sep_outputs, num_source)
                    for k in range(num_source):
                        pred_k = outputs_BK2L[:, k, :, :]
                        tgt_k = targets_BK2L[:, k, :, :]
                        theta_deg = estimate_phase_deg(pred_k, tgt_k)
                        phase_stats[k].update(theta_deg)
                    if DIAG_PHASE_ALIGN and batch_idx == 0 and num_source > 0:
                        pred0 = outputs_BK2L[:, 0, :, :]
                        tgt0 = targets_BK2L[:, 0, :, :]
                        theta0 = estimate_phase_deg(pred0, tgt0)
                        pred0_aligned = rotate_by_minus_theta(pred0, theta0)
                        corr_before = torch.nn.functional.cosine_similarity(
                            pred0.flatten(1), tgt0.flatten(1), dim=1
                        ).mean().item()
                        corr_after = torch.nn.functional.cosine_similarity(
                            pred0_aligned.flatten(1), tgt0.flatten(1), dim=1
                        ).mean().item()
                        logger.info(f'[PhaseAlign] S1 corr before={corr_before:.4f}, after={corr_after:.4f}')
                
                # Calculate overall metrics
                criterion_outputs = outputs
                # Auxiliary dictionaries are not waveform tensors. Generic
                # separation criteria must receive only the source output;
                # demodulation criteria are the exception because they
                # explicitly consume the auxiliary logits with `bits`.
                if (
                    isinstance(outputs, tuple)
                    and len(outputs) >= 2
                    and isinstance(outputs[1], dict)
                    and not _needs_bits
                ):
                    criterion_outputs = sep_outputs
                if _needs_bits and bits is not None:
                    test_loss += criterion(outputs, targets, bits=bits).item()
                elif _needs_mixture:
                    test_loss += criterion(criterion_outputs, targets, inputs).item()
                else:
                    test_loss += criterion(criterion_outputs, targets).item()

                # Use per-sample PIT alignment before computing evaluation metrics
                # to avoid penalizing valid source-order swaps.
                outputs_eval, best_perm_per_sample = reorder_outputs_for_eval(
                    sep_outputs,
                    targets,
                    num_source,
                    pit_metric=eval_pit_metric,
                )
                outputs_pit_eval = None
                if dual_report_fixed_and_pit:
                    outputs_pit_eval, _ = reorder_outputs_for_eval(
                        sep_outputs,
                        targets,
                        num_source,
                        pit_metric='si_snr_complex',
                    )

                phase_flip_source_rates = [float("nan")] * num_source
                if report_phase_flip:
                    _, phase_flip_details = phase_flip_rate(
                        outputs_eval,
                        targets,
                        num_sources=num_source,
                        tolerance_deg=phase_flip_tolerance_deg,
                        min_similarity=phase_flip_min_sc,
                        mode=phase_flip_mode,
                    )
                    batch_flipped_sources = int(phase_flip_details["flipped_sources"])
                    batch_total_sources = int(phase_flip_details["total_sources"])
                    phase_flip_source_count += batch_flipped_sources
                    phase_flip_source_total += batch_total_sources
                    phase_flip_sample_count += int(phase_flip_details["sample_flip_count"])
                    phase_flip_sample_total += int(phase_flip_details["sample_total"])
                    phase_flip_phase_abs_sum += float(phase_flip_details["mean_phase_abs_deg"]) * batch_total_sources
                    phase_flip_dist_pi_sum += float(phase_flip_details["mean_phase_distance_to_pi_deg"]) * batch_total_sources
                    phase_flip_neg_alpha_sum += float(phase_flip_details["negative_real_alpha_rate"]) * batch_total_sources
                    phase_flip_diag_den += batch_total_sources
                    source_counts = phase_flip_details["source_flip_counts"]
                    source_rates = phase_flip_details["source_flip_rates"]
                    phase_flip_source_rates = [float(v) for v in source_rates]
                    for source_idx in range(num_source):
                        phase_flip_source_counts[source_idx] += int(source_counts[source_idx])
                        phase_flip_source_totals[source_idx] += int(targets.shape[0])

                if bits is not None:
                    bit_logits, symbol_logits = extract_demod_outputs(outputs)
                    if bit_logits is not None:
                        for b, perm in enumerate(best_perm_per_sample):
                            for target_idx, pred_idx in enumerate(perm):
                                pred_bits = (bit_logits[pred_idx][b] > 0).long()
                                gt_bits = bits[target_idx][b].long()
                                test_demod_bit_correct += (pred_bits == gt_bits).sum().item()
                                test_demod_bit_total += gt_bits.numel()
                                if symbol_logits is not None and symbol_logits[pred_idx] is not None:
                                    gt_symbols = bits_to_symbol_labels(gt_bits)
                                    if gt_symbols is not None:
                                        pred_symbols = symbol_logits[pred_idx][b].argmax(dim=-1)
                                        max_symbols = min(pred_symbols.shape[0], gt_symbols.shape[0])
                                        if max_symbols > 0:
                                            test_demod_symbol_correct += (
                                                pred_symbols[:max_symbols] == gt_symbols[:max_symbols]
                                            ).sum().item()
                                            test_demod_symbol_total += max_symbols

                test_corr += calculate_correlation(outputs_eval, targets)

                # --- New metrics (per source, then average) ---
                preds_k = _split_sources(outputs_eval, num_source)
                tgts_k = _split_sources(targets, num_source)
                if ber_metric_enabled and (not use_filelevel_ber) and (not ber_selfcheck_done) and bits is not None:
                    selfcheck = _frame_level_ber_selfcheck(targets, bits)
                    ber_selfcheck_done = selfcheck is not None
                    if selfcheck is not None:
                        strict_self, oracle_self = selfcheck
                        if _ber_selfcheck_failed(strict_self, oracle_self):
                            logger.warning(
                                f"[BER] Disabled frame-level BER at SNR={snr}: target-vs-target "
                                f"self-check is already too high (strict={strict_self:.4f}, "
                                f"oracle={oracle_self:.4f}). This usually means the saved bits "
                                "and per-frame IQ slices are not directly aligned."
                            )
                            ber_metric_enabled = False
                ber_vals_per_source = {
                    'BER_strict': [None] * num_source,
                    'BER_oracle': [None] * num_source,
                }
                for k in range(num_source):
                    test_si_snr_real += si_snr_real(preds_k[k], tgts_k[k]).item()
                    test_si_snr_paper += si_snr_paper(preds_k[k], tgts_k[k]).item()
                    test_si_snr_repo += si_snr_repo(preds_k[k], tgts_k[k]).item()
                    test_si_snr_complex += si_snr_complex(preds_k[k], tgts_k[k]).item()
                    test_mse += mse(preds_k[k], tgts_k[k]).item()
                    test_scale_aligned_mse += scale_aligned_mse(preds_k[k], tgts_k[k]).item()
                    test_sc += similarity_coeff_complex(preds_k[k], tgts_k[k]).item()
                    test_pearson += pearson_complex_abs(preds_k[k], tgts_k[k]).item()

                    if ber_metric_enabled and bits is not None and ber_modulations and k < len(ber_modulations):
                        ber_k_strict = strict_ber_iq_from_bits(
                            preds_k[k].float(),
                            tgts_k[k].float(),
                            bits[k],
                            modulation=ber_modulations[k],
                            offset_search=ber_offset_search,
                            protocol=data_choice,
                        )
                        if torch.isfinite(ber_k_strict):
                            test_ber_strict += ber_k_strict.item()
                            test_ber_strict_den += 1
                            ber_vals_per_source['BER_strict'][k] = float(ber_k_strict.item())

                        if ber_compute_oracle:
                            ber_k_oracle = oracle_ber_iq_from_bits(
                                preds_k[k].float(),
                                tgts_k[k].float(),
                                bits[k],
                                modulation=ber_modulations[k],
                                offset_search=ber_offset_search,
                                protocol=data_choice,
                            )
                            if torch.isfinite(ber_k_oracle):
                                test_ber_oracle += ber_k_oracle.item()
                                test_ber_oracle_den += 1
                                ber_vals_per_source['BER_oracle'][k] = float(ber_k_oracle.item())
                
                # Calculate metrics for each source
                source_metrics_batch = calculate_metrics_per_source(outputs_eval, targets, num_source)
                
                # Accumulate metrics for each source
                for source_idx in range(num_source):
                    for metric_name in per_source_metric_names:
                        if metric_name in ber_vals_per_source:
                            v = ber_vals_per_source[metric_name][source_idx]
                            if v is not None:
                                source_metrics_sum[source_idx][metric_name] += float(v)
                                source_metrics_den[source_idx][metric_name] += 1
                            continue
                        if metric_name == 'Phase_Flip_Rate':
                            if report_phase_flip and np.isfinite(phase_flip_source_rates[source_idx]):
                                source_metrics_sum[source_idx][metric_name] += phase_flip_source_rates[source_idx]
                                source_metrics_den[source_idx][metric_name] += 1
                            continue
                        source_metrics_sum[source_idx][metric_name] += source_metrics_batch[source_idx][metric_name]
                        source_metrics_den[source_idx][metric_name] += 1

                if outputs_pit_eval is not None:
                    pit_source_metrics_batch = calculate_metrics_per_source(
                        outputs_pit_eval, targets, num_source
                    )
                    for source_idx in range(num_source):
                        for metric_name in waveform_metric_names:
                            value = pit_source_metrics_batch[source_idx][metric_name]
                            value = value.item() if torch.is_tensor(value) else float(value)
                            pit_metric_sums[metric_name] += value
                            pit_source_metric_sums[source_idx][metric_name] += value
                            pit_source_metric_den[source_idx][metric_name] += 1
                
                # Plot signal comparison
                if save_artifacts and batch_idx == 0 and num_plots > 0:
                    for sample_idx in range(min(num_plots, inputs.size(0))):
                        input_sample = inputs[sample_idx].cpu().numpy()
                        target_sample = targets[sample_idx].cpu().numpy()
                        output_sample = outputs_eval[sample_idx].cpu().numpy()
                        plot_signals(input_sample, target_sample, output_sample, 
                                   sample_idx, snr, logger, results_folder, num_points, signal_names=signal_names)

        # Calculate average metrics
        num_batches = len(loader)
        avg_loss = test_loss / num_batches
        avg_corr = test_corr / num_batches
        metric_den = num_batches * num_source
        avg_si_snr_real = test_si_snr_real / metric_den
        avg_si_snr_paper = test_si_snr_paper / metric_den
        avg_si_snr_repo = test_si_snr_repo / metric_den
        avg_si_snr_complex = test_si_snr_complex / metric_den
        avg_mse = test_mse / metric_den
        avg_scale_aligned_mse = test_scale_aligned_mse / metric_den
        avg_sc = test_sc / metric_den
        avg_pearson = test_pearson / metric_den
        pit_avg_metrics = {}
        pit_source_avg_metrics = {}
        if dual_report_fixed_and_pit:
            pit_avg_metrics = {
                name: pit_metric_sums[name] / metric_den
                for name in waveform_metric_names
            }
            for source_idx in range(num_source):
                pit_source_avg_metrics[source_idx] = {}
                for metric_name in waveform_metric_names:
                    den = pit_source_metric_den[source_idx][metric_name]
                    pit_source_avg_metrics[source_idx][metric_name] = (
                        pit_source_metric_sums[source_idx][metric_name] / den
                        if den > 0 else float('nan')
                    )
        avg_ber_strict = (test_ber_strict / test_ber_strict_den) if test_ber_strict_den > 0 else float("nan")
        avg_ber_oracle = (test_ber_oracle / test_ber_oracle_den) if test_ber_oracle_den > 0 else float("nan")
        avg_demod_bit_acc = (test_demod_bit_correct / test_demod_bit_total) if test_demod_bit_total > 0 else float("nan")
        avg_demod_symbol_acc = (test_demod_symbol_correct / test_demod_symbol_total) if test_demod_symbol_total > 0 else float("nan")
        avg_phase_flip_rate = (
            phase_flip_source_count / phase_flip_source_total
            if report_phase_flip and phase_flip_source_total > 0
            else float("nan")
        )
        avg_phase_flip_sample_rate = (
            phase_flip_sample_count / phase_flip_sample_total
            if report_phase_flip and phase_flip_sample_total > 0
            else float("nan")
        )
        avg_phase_flip_phase_abs = (
            phase_flip_phase_abs_sum / phase_flip_diag_den
            if report_phase_flip and phase_flip_diag_den > 0
            else float("nan")
        )
        avg_phase_flip_dist_pi = (
            phase_flip_dist_pi_sum / phase_flip_diag_den
            if report_phase_flip and phase_flip_diag_den > 0
            else float("nan")
        )
        avg_phase_flip_neg_alpha_rate = (
            phase_flip_neg_alpha_sum / phase_flip_diag_den
            if report_phase_flip and phase_flip_diag_den > 0
            else float("nan")
        )

        if DIAG_EMPIRICAL_SNR:
            mean_snr, std_snr = emp_snr_stats.mean_std()
            logger.info(f'[Empirical SNR] mean={mean_snr:.2f} dB, std={std_snr:.2f} dB')
        if DIAG_PHASE:
            for k in range(num_source):
                mean_theta, std_theta = phase_stats[k].mean_std()
                logger.info(f'[Phase] S{k+1}: mean={mean_theta:.2f} deg, std={std_theta:.2f} deg')
        
        # Calculate average metrics for each source
        source_avg_metrics = {} 
        for source_idx in range(num_source): 
            source_avg_metrics[source_idx] = {} 
            for metric_name in per_source_metric_names: 
                den = source_metrics_den[source_idx][metric_name]
                if den > 0:
                    source_avg_metrics[source_idx][metric_name] = source_metrics_sum[source_idx][metric_name] / den
                else:
                    source_avg_metrics[source_idx][metric_name] = float("nan")
            if report_ber:
                source_avg_metrics[source_idx]["BER"] = float(source_avg_metrics[source_idx].get("BER_strict", float("nan")))
            if report_phase_flip:
                total_k = phase_flip_source_totals[source_idx]
                source_avg_metrics[source_idx]["Phase_Flip_Rate"] = (
                    phase_flip_source_counts[source_idx] / total_k
                    if total_k > 0
                    else float("nan")
                )
        if ber_metric_enabled and ("BER_strict" in per_source_metric_names or "BER_oracle" in per_source_metric_names):
            stream_ber_fn = globals().get("compute_stream_ber_for_matlab_h5_dataset")
            if use_filelevel_ber and callable(stream_ber_fn) and loader_base_dataset is not None and file_idxs:
                try:
                    avg_ber_strict, src_means_strict = stream_ber_fn(
                        model,
                        loader_base_dataset,
                        signal_names=signal_names,
                        device=device,
                        file_indices=file_idxs,
                        max_files=int(ber_num_files),
                        forward_batch_size=int(getattr(loader, "batch_size", 64) or 64),
                        ber_offset_search=bool(ber_offset_search),
                        ber_variant="strict",
                        return_per_source=True,
                        modulations=ber_modulations,
                        num_sources=num_source,
                        protocol=data_choice,
                        amr_mode=amr_mode,
                        demod_mode=demod_mode,
                        eval_pit_metric=eval_pit_metric,
                    )
                    if ber_compute_oracle:
                        avg_ber_oracle, src_means_oracle = stream_ber_fn(
                            model,
                            loader_base_dataset,
                            signal_names=signal_names,
                            device=device,
                            file_indices=file_idxs,
                            max_files=int(ber_num_files),
                            forward_batch_size=int(getattr(loader, "batch_size", 64) or 64),
                            ber_offset_search=bool(ber_offset_search),
                            ber_variant="oracle",
                            return_per_source=True,
                            modulations=ber_modulations,
                            num_sources=num_source,
                            protocol=data_choice,
                            amr_mode=amr_mode,
                            demod_mode=demod_mode,
                            eval_pit_metric=eval_pit_metric,
                        )
                    else:
                        avg_ber_oracle = float("nan")
                        src_means_oracle = [float("nan")] * num_source
                    for source_idx in range(num_source):
                        if source_idx < len(src_means_strict):
                            source_avg_metrics[source_idx]["BER_strict"] = float(src_means_strict[source_idx])
                            source_avg_metrics[source_idx]["BER"] = float(src_means_strict[source_idx])
                        if source_idx < len(src_means_oracle):
                            source_avg_metrics[source_idx]["BER_oracle"] = float(src_means_oracle[source_idx])
                except Exception:
                    avg_ber_strict = float("nan")
                    avg_ber_oracle = float("nan")
                    for source_idx in range(num_source):
                        source_avg_metrics[source_idx]["BER_strict"] = float("nan")
                        source_avg_metrics[source_idx]["BER_oracle"] = float("nan")
                        source_avg_metrics[source_idx]["BER"] = float("nan")
            elif use_filelevel_ber and not filelevel_ber_fallback_logged:
                logger.info("[BER] File-level BER unavailable for this split; falling back to frame-level BER.")
                filelevel_ber_fallback_logged = True
            elif test_ber_strict_den == 0 and test_ber_oracle_den == 0:
                for source_idx in range(num_source):
                    source_avg_metrics[source_idx]["BER_strict"] = float("nan")
                    source_avg_metrics[source_idx]["BER_oracle"] = float("nan")
                    source_avg_metrics[source_idx]["BER"] = float("nan")
        
        # Store overall metrics
        snr_metrics[snr] = {
            'Loss': avg_loss,
            'Correlation': avg_corr,
            'SI-SNR_real': avg_si_snr_real,
            'SI-SNR_paper': avg_si_snr_paper,
            'SI-SNR_repo': avg_si_snr_repo,
            'SI-SNR_complex': avg_si_snr_complex,
            'MSE': avg_mse,
            'ScaleAligned_MSE': avg_scale_aligned_mse,
            'SC': avg_sc,
            'Pearson': avg_pearson,
            'BER': avg_ber_strict,
            'BER_strict': avg_ber_strict,
            'BER_oracle': avg_ber_oracle,
            'Demod_Bit_Acc': avg_demod_bit_acc,
            'Demod_Sym_Acc': avg_demod_symbol_acc,
            'Phase_Flip_Rate': avg_phase_flip_rate,
            'Phase_Flip_Sample_Rate': avg_phase_flip_sample_rate,
            'Phase_Flip_NegAlpha_Rate': avg_phase_flip_neg_alpha_rate,
            'Phase_Flip_MeanAbsPhase_Deg': avg_phase_flip_phase_abs,
            'Phase_Flip_MeanDistToPi_Deg': avg_phase_flip_dist_pi,
            'Source_Metrics': source_avg_metrics
        }
        if dual_report_fixed_and_pit:
            snr_metrics[snr]['PIT_Aligned_Metrics'] = {
                **pit_avg_metrics,
                'Alignment': 'si_snr_complex',
                'Source_Metrics': pit_source_avg_metrics,
            }
        
        # Store correlation coefficients for plotting
        snr_list.append(snr)
        overall_correlations.append(avg_corr.item())
        for source_idx in range(num_source):
            source_correlations[source_idx].append(source_avg_metrics[source_idx]['Correlation'].item())
        
        # Add overall metrics for CSV
        metrics_row = {
            'SNR': snr,
            'Source': 'Overall',
            'Signal_Type': 'All_Sources',
            'Loss': avg_loss,
            'Correlation': avg_corr.item(),
            'SI-SNR_real': avg_si_snr_real,
            'SI-SNR_paper': avg_si_snr_paper,
            'SI-SNR_repo': avg_si_snr_repo,
            'SI-SNR_complex': avg_si_snr_complex,
            'MSE': avg_mse,
            'ScaleAligned_MSE': avg_scale_aligned_mse,
            'SC': avg_sc,
            'Pearson': avg_pearson,
            'BER': avg_ber_strict,
            'BER_strict': avg_ber_strict,
            'BER_oracle': avg_ber_oracle,
            'Demod_Bit_Acc': avg_demod_bit_acc,
            'Demod_Sym_Acc': avg_demod_symbol_acc,
            'Phase_Flip_Rate': avg_phase_flip_rate,
            'Phase_Flip_Sample_Rate': avg_phase_flip_sample_rate,
            'Phase_Flip_NegAlpha_Rate': avg_phase_flip_neg_alpha_rate,
            'Phase_Flip_MeanAbsPhase_Deg': avg_phase_flip_phase_abs,
            'Phase_Flip_MeanDistToPi_Deg': avg_phase_flip_dist_pi,
            'Phase_Flip_Tolerance_Deg': phase_flip_tolerance_deg if report_phase_flip else float("nan"),
            'Phase_Flip_Min_SC': phase_flip_min_sc if report_phase_flip else float("nan"),
            'Phase_Flip_Mode': phase_flip_mode if report_phase_flip else "",
            'Eval_PIT_Metric': eval_pit_metric,
        }
        if dual_report_fixed_and_pit:
            metrics_row.update({
                f'PIT_{name}': pit_avg_metrics[name]
                for name in waveform_metric_names
            })
        all_metrics_list.append(metrics_row)
        
        # Add metrics for each source for CSV
        for source_idx in range(num_source):
            signal_name = signal_names[source_idx] if signal_names else f'Source_{source_idx}'
            source_row = {
                'SNR': snr,
                'Source': f'Source_{source_idx}',
                'Signal_Type': signal_name,
                'Loss': avg_loss,  # Loss is overall
                'Correlation': source_avg_metrics[source_idx]['Correlation'].item(),
                'SI-SNR_real': source_avg_metrics[source_idx]['SI-SNR_real'].item(),
                'SI-SNR_paper': source_avg_metrics[source_idx]['SI-SNR_paper'].item(),
                'SI-SNR_repo': source_avg_metrics[source_idx]['SI-SNR_repo'].item(),
                'SI-SNR_complex': source_avg_metrics[source_idx]['SI-SNR_complex'].item(),
                'MSE': source_avg_metrics[source_idx]['MSE'].item(),
                'ScaleAligned_MSE': source_avg_metrics[source_idx]['ScaleAligned_MSE'].item(),
                'SC': source_avg_metrics[source_idx]['SC'].item(),
                'Pearson': source_avg_metrics[source_idx]['Pearson'].item(),
                'BER': float(source_avg_metrics[source_idx].get('BER', float('nan'))),
                'BER_strict': float(source_avg_metrics[source_idx].get('BER_strict', float('nan'))),
                'BER_oracle': float(source_avg_metrics[source_idx].get('BER_oracle', float('nan'))),
                'Phase_Flip_Rate': float(source_avg_metrics[source_idx].get('Phase_Flip_Rate', float('nan'))),
                'Phase_Flip_Sample_Rate': float("nan"),
                'Phase_Flip_NegAlpha_Rate': float("nan"),
                'Phase_Flip_MeanAbsPhase_Deg': float("nan"),
                'Phase_Flip_MeanDistToPi_Deg': float("nan"),
                'Phase_Flip_Tolerance_Deg': phase_flip_tolerance_deg if report_phase_flip else float("nan"),
                'Phase_Flip_Min_SC': phase_flip_min_sc if report_phase_flip else float("nan"),
                'Phase_Flip_Mode': phase_flip_mode if report_phase_flip else "",
                'Eval_PIT_Metric': eval_pit_metric,
            }
            if dual_report_fixed_and_pit:
                source_row.update({
                    f'PIT_{name}': pit_source_avg_metrics[source_idx][name]
                    for name in waveform_metric_names
                })
            all_metrics_list.append(source_row)
        
        # Log results
        logger.info(f'SNR {snr}dB:')
        primary_label = 'Overall [Fixed]' if dual_report_fixed_and_pit else 'Overall'
        if dual_report_fixed_and_pit:
            logger.info(f'\tOverall - Loss (criterion): {avg_loss:.8f}')
            logger.info(f'\t{primary_label} - Correlation: {avg_corr:.8f}')
        else:
            logger.info(f'\t{primary_label} - Loss: {avg_loss:.8f}, Correlation: {avg_corr:.8f}')
        logger.info(
            f'\t{primary_label} - SI-SNR_real: {avg_si_snr_real:.4f} dB, '
            f'SI-SNR_paper: {avg_si_snr_paper:.4f} dB, '
            f'SI-SNR_repo: {avg_si_snr_repo:.4f} dB, '
            f'SI-SNR_complex: {avg_si_snr_complex:.4f} dB'
        )
        if report_ber:
            logger.info(
                f'\t{primary_label} - MSE: {avg_mse:.6f}, ScaleAligned_MSE: {avg_scale_aligned_mse:.6f}, '
                f'SC: {avg_sc:.4f}, Pearson: {avg_pearson:.4f}, '
                f'BER_strict: {avg_ber_strict:.6f}, BER_oracle: {avg_ber_oracle:.6f}'
            )
        else:
            logger.info(
                f'\t{primary_label} - MSE: {avg_mse:.6f}, ScaleAligned_MSE: {avg_scale_aligned_mse:.6f}, '
                f'SC: {avg_sc:.4f}, Pearson: {avg_pearson:.4f}'
            )
        if dual_report_fixed_and_pit:
            pit_label = 'Overall [PIT:si_snr_complex]'
            logger.info(
                f'\t{pit_label} - Correlation: {pit_avg_metrics["Correlation"]:.8f}'
            )
            logger.info(
                f'\t{pit_label} - SI-SNR_real: {pit_avg_metrics["SI-SNR_real"]:.4f} dB, '
                f'SI-SNR_paper: {pit_avg_metrics["SI-SNR_paper"]:.4f} dB, '
                f'SI-SNR_repo: {pit_avg_metrics["SI-SNR_repo"]:.4f} dB, '
                f'SI-SNR_complex: {pit_avg_metrics["SI-SNR_complex"]:.4f} dB'
            )
            logger.info(
                f'\t{pit_label} - MSE: {pit_avg_metrics["MSE"]:.6f}, '
                f'ScaleAligned_MSE: {pit_avg_metrics["ScaleAligned_MSE"]:.6f}, '
                f'SC: {pit_avg_metrics["SC"]:.4f}, '
                f'Pearson: {pit_avg_metrics["Pearson"]:.4f}'
            )
        if test_demod_bit_total > 0:
            logger.info(
                f'\tOverall - Demod Bit Acc: {100.0 * avg_demod_bit_acc:.2f}%, '
                f'Demod Sym Acc: {100.0 * avg_demod_symbol_acc:.2f}%'
            )
        if report_phase_flip:
            logger.info(
                f'\tOverall - Phase_Flip_Rate: {100.0 * avg_phase_flip_rate:.2f}% '
                f'({phase_flip_source_count}/{phase_flip_source_total} source items), '
                f'Phase_Flip_Sample_Rate: {100.0 * avg_phase_flip_sample_rate:.2f}%'
            )
            logger.info(
                f'\tOverall - PhaseFlip diagnostics: '
                f'NegAlpha_Rate={100.0 * avg_phase_flip_neg_alpha_rate:.2f}%, '
                f'MeanAbsPhase={avg_phase_flip_phase_abs:.2f} deg, '
                f'MeanDistToPi={avg_phase_flip_dist_pi:.2f} deg'
            )
        
        for source_idx in range(num_source):
            signal_name = signal_names[source_idx] if signal_names else f'Source_{source_idx}'
            metrics = source_avg_metrics[source_idx]
            source_label = f'{signal_name} [Fixed]' if dual_report_fixed_and_pit else signal_name
            logger.info(f'\t{source_label} - Correlation: {metrics["Correlation"]:.8f}')
            logger.info(
                f'\t{source_label} - SI-SNR_real: {metrics["SI-SNR_real"]:.4f} dB, '
                f'SI-SNR_paper: {metrics["SI-SNR_paper"]:.4f} dB, '
                f'SI-SNR_repo: {metrics["SI-SNR_repo"]:.4f} dB, '
                f'SI-SNR_complex: {metrics["SI-SNR_complex"]:.4f} dB'
            )
            if report_ber and "BER_strict" in metrics:
                logger.info(
                    f'\t{source_label} - MSE: {metrics["MSE"]:.6f}, '
                    f'ScaleAligned_MSE: {metrics["ScaleAligned_MSE"]:.6f}, SC: {metrics["SC"]:.4f}, '
                    f'Pearson: {metrics["Pearson"]:.4f}, BER_strict: {metrics["BER_strict"]:.6f}, '
                    f'BER_oracle: {metrics.get("BER_oracle", float("nan")):.6f}'
                )
            else:
                logger.info(
                    f'\t{source_label} - MSE: {metrics["MSE"]:.6f}, '
                    f'ScaleAligned_MSE: {metrics["ScaleAligned_MSE"]:.6f}, '
                    f'SC: {metrics["SC"]:.4f}, Pearson: {metrics["Pearson"]:.4f}'
                )
            if dual_report_fixed_and_pit:
                pit_metrics = pit_source_avg_metrics[source_idx]
                pit_source_label = f'{signal_name} [PIT:si_snr_complex]'
                logger.info(
                    f'\t{pit_source_label} - Correlation: {pit_metrics["Correlation"]:.8f}'
                )
                logger.info(
                    f'\t{pit_source_label} - SI-SNR_real: {pit_metrics["SI-SNR_real"]:.4f} dB, '
                    f'SI-SNR_paper: {pit_metrics["SI-SNR_paper"]:.4f} dB, '
                    f'SI-SNR_repo: {pit_metrics["SI-SNR_repo"]:.4f} dB, '
                    f'SI-SNR_complex: {pit_metrics["SI-SNR_complex"]:.4f} dB'
                )
                logger.info(
                    f'\t{pit_source_label} - MSE: {pit_metrics["MSE"]:.6f}, '
                    f'ScaleAligned_MSE: {pit_metrics["ScaleAligned_MSE"]:.6f}, '
                    f'SC: {pit_metrics["SC"]:.4f}, Pearson: {pit_metrics["Pearson"]:.4f}'
                )
            if report_phase_flip:
                logger.info(
                    f'\t{signal_name} - Phase_Flip_Rate: {100.0 * metrics["Phase_Flip_Rate"]:.2f}% '
                    f'({phase_flip_source_counts[source_idx]}/{phase_flip_source_totals[source_idx]} source items)'
                )
    
    if save_artifacts and results_folder:
        # Save detailed CSV file
        df = pd.DataFrame(all_metrics_list)
        project_root = Path(__file__).resolve().parents[1]
        csv_path = project_root / "results" / results_folder / "detailed_metrics_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed metrics saved to {csv_path}")
        
        # Plot enhanced correlation vs SNR
        plot_correlation_vs_snr_enhanced(snr_list, overall_correlations, source_correlations, 
                                       results_folder, signal_names)
    
    # Output average time
    if total_samples > 0:
        avg_time_per_sample = (total_time / total_samples) * 1000
        logger.info(f"Global Avg Inference Time per Sample: {avg_time_per_sample:.4f} ms")
    
    return snr_metrics
