"""Training-only low-SNR objectives for IQ source separation.

The helpers in this module do not add inference modules. They construct one
cross-SNR partner mixture and evaluate source- or receiver-domain losses for a
dynamic number of I/Q sources.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Sequence

import torch
import torch.nn.functional as F


def _split_iq_sources(x: torch.Tensor, num_sources: int) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"Expected (B, 2K, L), got {tuple(x.shape)}")
    if num_sources not in (2, 3):
        raise ValueError(f"num_sources must be 2 or 3, got {num_sources}")
    if x.size(1) != 2 * num_sources:
        raise ValueError(f"Expected {2 * num_sources} channels, got {x.size(1)}")
    return x.reshape(x.size(0), num_sources, 2, x.size(-1))


def receiver_subset_size(batch_size: int, fraction: float) -> int:
    """Return a non-empty receiver-loss subset for a non-empty batch."""
    batch_size = max(0, int(batch_size))
    if batch_size == 0:
        return 0
    fraction = min(max(float(fraction), 0.0), 1.0)
    return max(1, min(batch_size, int(math.ceil(batch_size * fraction))))


def curriculum_low_snr(
    epoch: int,
    num_epochs: int,
    start_db: float,
    middle_db: float,
    final_db: float,
    first_fraction: float = 0.2,
    second_fraction: float = 0.6,
) -> float:
    """Return the piecewise-constant low-SNR curriculum target."""
    total = max(1, int(num_epochs))
    progress = max(0.0, min(float(epoch) / total, 1.0))
    first = max(0.0, min(float(first_fraction), 1.0))
    second = max(first, min(float(second_fraction), 1.0))
    if progress < first:
        return float(start_db)
    if progress < second:
        return float(middle_db)
    return float(final_db)


def build_cross_snr_partner(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    original_snr: torch.Tensor,
    *,
    num_sources: int,
    high_snr_db: float = 10.0,
    low_snr_db: float = -10.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one complementary-SNR mixture from the same clean sources."""
    if inputs.dim() != 3 or inputs.size(1) != 2:
        raise ValueError(f"Expected input mixture (B, 2, L), got {tuple(inputs.shape)}")
    source_view = _split_iq_sources(targets, num_sources)
    clean_mix = source_view.sum(dim=1)
    if clean_mix.shape != inputs.shape:
        raise ValueError(f"Mixture/target shape mismatch: {tuple(inputs.shape)} vs {tuple(clean_mix.shape)}")

    snr = original_snr.to(device=inputs.device, dtype=torch.float32).reshape(-1)
    if snr.numel() == 1 and inputs.size(0) > 1:
        snr = snr.expand(inputs.size(0))
    if snr.numel() != inputs.size(0):
        raise ValueError(f"Expected {inputs.size(0)} SNR values, got {snr.numel()}")

    midpoint = 0.5 * (float(high_snr_db) + float(low_snr_db))
    partner_snr = torch.where(
        snr <= midpoint,
        snr.new_full(snr.shape, float(high_snr_db)),
        snr.new_full(snr.shape, float(low_snr_db)),
    )

    residual = (inputs - clean_mix).roll(shifts=1, dims=0)
    residual = residual.roll(shifts=max(1, inputs.size(-1) // 3), dims=-1)
    residual = residual - residual.mean(dim=-1, keepdim=True)
    residual_power = residual.square().mean(dim=(1, 2), keepdim=True)
    degenerate = residual_power <= float(eps)
    if bool(degenerate.any()):
        fallback = torch.randn_like(residual)
        fallback = fallback - fallback.mean(dim=-1, keepdim=True)
        residual = torch.where(degenerate, fallback, residual)
        residual_power = residual.square().mean(dim=(1, 2), keepdim=True)

    signal_power = clean_mix.square().mean(dim=(1, 2), keepdim=True).clamp_min(float(eps))
    snr_linear = torch.pow(
        signal_power.new_tensor(10.0),
        partner_snr.to(device=inputs.device, dtype=signal_power.dtype).view(-1, 1, 1) / 10.0,
    )
    target_noise_power = signal_power / snr_linear
    noise_scale = torch.sqrt(target_noise_power / residual_power.clamp_min(float(eps)))
    partner = clean_mix + residual * noise_scale
    return partner.to(dtype=inputs.dtype), partner_snr.to(dtype=original_snr.dtype)


def clean_mixture_from_targets(targets: torch.Tensor, num_sources: int) -> torch.Tensor:
    """Return the noise-free IQ mixture associated with source targets."""
    return _split_iq_sources(targets, num_sources).sum(dim=1)


def build_snr_view(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    target_snr_db: torch.Tensor | float,
    *,
    num_sources: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize an exact-SNR student view from the same clean sources."""
    clean_mix = clean_mixture_from_targets(targets, num_sources)
    if clean_mix.shape != inputs.shape:
        raise ValueError(f"Mixture/target shape mismatch: {tuple(inputs.shape)} vs {tuple(clean_mix.shape)}")
    target_snr = torch.as_tensor(
        target_snr_db, device=inputs.device, dtype=torch.float32
    ).reshape(-1)
    if target_snr.numel() == 1:
        target_snr = target_snr.expand(inputs.size(0))
    if target_snr.numel() != inputs.size(0):
        raise ValueError(f"Expected {inputs.size(0)} target SNR values, got {target_snr.numel()}")

    # Reuse measured receiver residuals while decorrelating them from their
    # original source frame, falling back to Gaussian noise when necessary.
    residual = (inputs - clean_mix).roll(shifts=1, dims=0)
    residual = residual.roll(shifts=max(1, inputs.size(-1) // 3), dims=-1)
    residual = residual - residual.mean(dim=-1, keepdim=True)
    residual_power = residual.square().mean(dim=(1, 2), keepdim=True)
    degenerate = residual_power <= float(eps)
    if bool(degenerate.any()):
        fallback = torch.randn_like(residual)
        fallback = fallback - fallback.mean(dim=-1, keepdim=True)
        residual = torch.where(degenerate, fallback, residual)
        residual_power = residual.square().mean(dim=(1, 2), keepdim=True)

    signal_power = clean_mix.square().mean(dim=(1, 2), keepdim=True).clamp_min(float(eps))
    snr_linear = torch.pow(
        signal_power.new_tensor(10.0),
        target_snr.to(dtype=signal_power.dtype).view(-1, 1, 1) / 10.0,
    )
    noise_scale = torch.sqrt(
        (signal_power / snr_linear) / residual_power.clamp_min(float(eps))
    )
    view = clean_mix + residual * noise_scale
    return view.to(dtype=inputs.dtype), target_snr


def sample_progressive_snr_range(
    batch_size: int,
    epoch: int,
    num_epochs: int,
    ranges: tuple[tuple[float, float], ...] | list,
    boundaries: tuple[float, ...] = (0.2, 0.6),
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample the article's high-to-low SNR range curriculum."""
    normalized_ranges = tuple((float(low), float(high)) for low, high in ranges)
    if not normalized_ranges:
        raise ValueError("ranges must not be empty")
    for low, high in normalized_ranges:
        if high < low:
            raise ValueError(f"invalid SNR range [{low}, {high}]")
    progress = max(0.0, min(float(epoch) / max(1, int(num_epochs)), 1.0))
    stage = 0
    for boundary in boundaries:
        if progress >= float(boundary):
            stage += 1
    stage = min(stage, len(normalized_ranges) - 1)
    low, high = normalized_ranges[stage]
    if high == low:
        return torch.full((int(batch_size),), low, device=device, dtype=torch.float32)
    return torch.empty(int(batch_size), device=device, dtype=torch.float32).uniform_(low, high)


def _pit_reorder_to_targets(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int,
) -> torch.Tensor:
    preds = _split_iq_sources(outputs, num_sources)
    refs = _split_iq_sources(targets, num_sources)
    permutations = tuple(itertools.permutations(range(num_sources)))
    candidates = torch.stack([preds[:, perm, :, :] for perm in permutations], dim=0)
    with torch.no_grad():
        errors = (candidates.detach() - refs.unsqueeze(0)).square().mean(dim=(2, 3, 4))
        best = errors.argmin(dim=0)
    batch_index = torch.arange(outputs.size(0), device=outputs.device)
    return candidates[best, batch_index]


def _shared_pit_reorder(
    reference_outputs: torch.Tensor,
    companion_outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the reference view's PIT decision for both paired views."""
    reference = _split_iq_sources(reference_outputs, num_sources)
    companion = _split_iq_sources(companion_outputs, num_sources)
    refs = _split_iq_sources(targets, num_sources)
    permutations = tuple(itertools.permutations(range(num_sources)))
    reference_candidates = torch.stack([reference[:, perm, :, :] for perm in permutations], dim=0)
    companion_candidates = torch.stack([companion[:, perm, :, :] for perm in permutations], dim=0)
    with torch.no_grad():
        errors = (reference_candidates.detach() - refs.unsqueeze(0)).square().mean(dim=(2, 3, 4))
        best = errors.argmin(dim=0)
    batch_index = torch.arange(reference_outputs.size(0), device=reference_outputs.device)
    return (
        reference_candidates[best, batch_index],
        companion_candidates[best, batch_index],
    )


def rotate_iq(x: torch.Tensor, angles_rad: torch.Tensor) -> torch.Tensor:
    """Apply one common complex-plane rotation per batch item."""
    if x.dim() != 3 or x.size(1) % 2 != 0:
        raise ValueError(f"Expected (B, 2K, L), got {tuple(x.shape)}")
    angles = angles_rad.to(device=x.device, dtype=x.dtype).reshape(-1)
    if angles.numel() == 1:
        angles = angles.expand(x.size(0))
    if angles.numel() != x.size(0):
        raise ValueError(f"Expected {x.size(0)} phase angles, got {angles.numel()}")
    view = x.reshape(x.size(0), x.size(1) // 2, 2, x.size(-1))
    cosine = angles.cos().view(-1, 1, 1)
    sine = angles.sin().view(-1, 1, 1)
    real = view[:, :, 0, :]
    imag = view[:, :, 1, :]
    rotated = torch.stack(
        [cosine * real - sine * imag, sine * real + cosine * imag],
        dim=2,
    )
    return rotated.reshape_as(x)


def phase_equivariance_consistency_loss(
    original_outputs: torch.Tensor,
    rotated_outputs: torch.Tensor,
    targets: torch.Tensor,
    angles_rad: torch.Tensor,
    *,
    num_sources: int,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize source-slot changes and non-equivariant output under IQ rotation."""
    original, rotated = _shared_pit_reorder(
        original_outputs,
        rotated_outputs,
        targets,
        num_sources,
    )
    rotated_flat = rotated.reshape(rotated.size(0), 2 * num_sources, rotated.size(-1))
    restored = rotate_iq(rotated_flat, -angles_rad).reshape_as(original)
    teacher = original.detach()
    teacher_rms = teacher.square().sum(dim=2).mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
    normalized_error = (restored - teacher) / teacher_rms.unsqueeze(2)
    return F.smooth_l1_loss(
        normalized_error,
        torch.zeros_like(normalized_error),
        beta=float(beta),
        reduction="mean",
    )


def _complex_align(pred: torch.Tensor, target: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Align `(B, ..., 2, N)` prediction to target by one complex scalar."""
    pred_c = torch.complex(pred[..., 0, :].float(), pred[..., 1, :].float())
    target_c = torch.complex(target[..., 0, :].float(), target[..., 1, :].float())
    alpha = (pred_c.conj() * target_c).sum(dim=-1, keepdim=True)
    alpha = alpha / pred_c.abs().square().sum(dim=-1, keepdim=True).clamp_min(float(eps))
    aligned = alpha * pred_c
    return aligned, target_c


def cross_snr_teacher_consistency_loss(
    original_outputs: torch.Tensor,
    partner_outputs: torch.Tensor,
    targets: torch.Tensor,
    original_snr: torch.Tensor,
    partner_snr: torch.Tensor,
    *,
    num_sources: int,
    beta: float = 0.5,
    eps: float = 1e-8,
    shared_permutation: bool = False,
    teacher_high_outputs: torch.Tensor | None = None,
    force_partner_student: bool = False,
) -> torch.Tensor:
    """Teach the lower-SNR estimate from the detached higher-SNR estimate.

    ``teacher_high_outputs`` optionally supplies a Mean-Teacher/EMA forward on
    the high-SNR view.  Without it, the historical online high-SNR branch is
    retained for backward compatibility with Stage 211.
    """
    original_is_low = (
        original_snr.to(device=original_outputs.device, dtype=torch.float32).reshape(-1)
        <= partner_snr.to(device=original_outputs.device, dtype=torch.float32).reshape(-1)
    ).view(-1, 1, 1, 1)
    original = _split_iq_sources(original_outputs, num_sources)
    partner = _split_iq_sources(partner_outputs, num_sources)
    low = partner if force_partner_student else torch.where(original_is_low, original, partner)
    if teacher_high_outputs is None:
        high = torch.where(original_is_low, partner, original)
    else:
        high = _split_iq_sources(teacher_high_outputs, num_sources)
    if shared_permutation:
        high_flat = high.reshape(high.size(0), 2 * num_sources, high.size(-1))
        low_flat = low.reshape(low.size(0), 2 * num_sources, low.size(-1))
        high, low = _shared_pit_reorder(high_flat, low_flat, targets, num_sources)
    else:
        low = _pit_reorder_to_targets(
            low.reshape(low.size(0), 2 * num_sources, low.size(-1)), targets, num_sources
        )
        high = _pit_reorder_to_targets(
            high.reshape(high.size(0), 2 * num_sources, high.size(-1)), targets, num_sources
        )
    high = high.detach()

    aligned, teacher = _complex_align(low, high, eps)
    teacher_rms = teacher.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
    error = torch.stack(
        [(aligned.real - teacher.real) / teacher_rms, (aligned.imag - teacher.imag) / teacher_rms],
        dim=-2,
    )
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=float(beta), reduction="mean")


def cross_snr_feature_consistency_loss(
    student_auxiliary: dict,
    teacher_auxiliary: dict,
    *,
    beta: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clean/high teacher to noisy student bottleneck feature distillation."""
    student = student_auxiliary.get("distillation_feature")
    teacher = teacher_auxiliary.get("distillation_feature")
    if student is None or teacher is None:
        reference = next(
            (value for value in student_auxiliary.values() if torch.is_tensor(value)),
            None,
        )
        if reference is None:
            raise ValueError("student auxiliary dictionary contains no tensors")
        return reference.new_zeros(())
    if student.shape != teacher.shape:
        raise ValueError(
            f"student/teacher feature shape mismatch: {tuple(student.shape)} vs {tuple(teacher.shape)}"
        )
    student = student.float()
    teacher = teacher.detach().float()
    student = student - student.mean(dim=-1, keepdim=True)
    teacher = teacher - teacher.mean(dim=-1, keepdim=True)
    student = student / student.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
    teacher = teacher / teacher.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
    return F.smooth_l1_loss(student, teacher, beta=float(beta), reduction="mean")


def _source_pit_indices(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    num_sources: int,
) -> torch.Tensor:
    predictions = _split_iq_sources(outputs, num_sources)
    references = _split_iq_sources(targets, num_sources)
    permutations = tuple(itertools.permutations(range(num_sources)))
    candidates = torch.stack(
        [predictions[:, permutation] for permutation in permutations], dim=0
    )
    with torch.no_grad():
        errors = (candidates.detach() - references.unsqueeze(0)).square().mean(
            dim=(2, 3, 4)
        )
        best = errors.argmin(dim=0)
        table = torch.tensor(permutations, device=outputs.device, dtype=torch.long)
        return table[best]


def _reorder_source_tensor(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2 or value.size(1) != indices.size(1):
        raise ValueError(
            f"expected source tensor [B, K, ...], got {tuple(value.shape)} for {tuple(indices.shape)}"
        )
    gather_index = indices.view(indices.size(0), indices.size(1), *([1] * (value.ndim - 2)))
    gather_index = gather_index.expand(indices.size(0), indices.size(1), *value.shape[2:])
    return torch.gather(value, dim=1, index=gather_index)


def pit_align_sync_auxiliary(
    auxiliary: dict,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_sources: int,
) -> dict:
    """Align per-source synchronization tensors to target source order.

    A separately pretrained teacher and a student can use opposite output-slot
    conventions.  Physical states must therefore follow each model's own PIT
    decision before teacher/student consistency is evaluated.  Mixture-level
    Stage-366 states are returned unchanged.
    """
    cfo = auxiliary.get("cfo_cycles_per_sample")
    if (
        not torch.is_tensor(cfo)
        or cfo.ndim < 2
        or cfo.size(1) != int(num_sources)
    ):
        return dict(auxiliary)

    indices = _source_pit_indices(outputs, targets, num_sources)
    aligned = dict(auxiliary)
    source_keys = (
        "cfo_cycles_per_sample",
        "phase_vector",
        "phase_rad",
        "timing_offset_unit",
        "sps_logits",
        "sps_probabilities",
        "sps_estimate",
        "phase_drift_rad_per_sample",
    )
    for key in source_keys:
        value = auxiliary.get(key)
        if (
            torch.is_tensor(value)
            and value.ndim >= 2
            and value.size(1) == int(num_sources)
        ):
            aligned[key] = _reorder_source_tensor(value, indices)
    return aligned


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    mask = mask.to(device=value.device, dtype=torch.bool)
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    if not bool(mask.any()):
        return None
    return value[mask].mean()


def sync_parameter_physical_supervision_loss(
    auxiliary: dict,
    metadata: dict,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_sources: int,
    cfo_weight: float = 1.0,
    phase_weight: float = 1.0,
    timing_weight: float = 1.0,
    sps_weight: float = 1.0,
    drift_weight: float = 1.0,
    cfo_scale: float = 0.25,
    drift_scale: float = 0.05,
    beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """PIT-aligned physical supervision for per-source synchronization heads."""
    source_indices = _source_pit_indices(outputs, targets, num_sources)
    losses = []

    def prediction(name: str) -> torch.Tensor:
        value = auxiliary.get(name)
        if value is None:
            raise KeyError(f"synchronization auxiliary output is missing {name!r}")
        return _reorder_source_tensor(value.float(), source_indices)

    def target(name: str) -> torch.Tensor:
        value = metadata.get(name)
        if value is None:
            raise KeyError(f"synchronization metadata is missing {name!r}")
        return value.to(device=outputs.device, dtype=torch.float32)

    if float(cfo_weight) > 0.0:
        pred = prediction("cfo_cycles_per_sample") / max(abs(float(cfo_scale)), eps)
        ref = target("cfo_cycles_per_sample") / max(abs(float(cfo_scale)), eps)
        raw = F.smooth_l1_loss(pred, ref, beta=float(beta), reduction="none")
        term = _masked_mean(raw, target("cfo_valid").bool())
        if term is not None:
            losses.append(float(cfo_weight) * term)

    if float(phase_weight) > 0.0:
        pred_vector = prediction("phase_vector")
        phase = target("phase_rad")
        ref_vector = torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
        raw = 1.0 - (pred_vector * ref_vector).sum(dim=-1).clamp(-1.0, 1.0)
        term = _masked_mean(raw, target("phase_valid").bool())
        if term is not None:
            losses.append(float(phase_weight) * term)

    if float(timing_weight) > 0.0:
        pred = prediction("timing_offset_unit")
        delay = target("timing_offset_samples")
        sps = target("samples_per_symbol").clamp_min(1.0)
        centered = torch.remainder(delay + 0.5 * sps, sps) - 0.5 * sps
        ref = (centered / (0.5 * sps).clamp_min(1.0)).clamp(-1.0, 1.0)
        raw = F.smooth_l1_loss(pred, ref, beta=float(beta), reduction="none")
        mask = target("timing_valid").bool() & target("sps_valid").bool()
        term = _masked_mean(raw, mask)
        if term is not None:
            losses.append(float(timing_weight) * term)

    if float(sps_weight) > 0.0:
        logits = prediction("sps_logits")
        candidates = auxiliary.get("sps_candidates")
        if candidates is None:
            raise KeyError("synchronization auxiliary output is missing 'sps_candidates'")
        candidates = candidates.to(device=outputs.device, dtype=torch.float32).reshape(1, 1, -1)
        labels = (target("samples_per_symbol").unsqueeze(-1) - candidates).abs().argmin(dim=-1)
        raw = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).view_as(labels)
        term = _masked_mean(raw, target("sps_valid").bool())
        if term is not None:
            losses.append(float(sps_weight) * term)

    if float(drift_weight) > 0.0:
        pred = prediction("phase_drift_rad_per_sample") / max(abs(float(drift_scale)), eps)
        ref = target("phase_drift_rad_per_sample") / max(abs(float(drift_scale)), eps)
        raw = F.smooth_l1_loss(pred, ref, beta=float(beta), reduction="none")
        term = _masked_mean(raw, target("drift_valid").bool())
        if term is not None:
            losses.append(float(drift_weight) * term)

    if not losses:
        return outputs.new_zeros(())
    return torch.stack(losses).sum()


def sync_parameter_snr_supervision_loss(
    auxiliary: dict,
    snr: torch.Tensor,
    *,
    min_db: float = -10.0,
    max_db: float = 30.0,
    beta: float = 0.1,
) -> torch.Tensor:
    """Supervise the explicit synchronization head's calibrated SNR output."""
    prediction = auxiliary.get("snr_prediction")
    if prediction is None:
        raise KeyError("synchronization auxiliary output is missing 'snr_prediction'")
    prediction = prediction.float().reshape(-1)
    target = snr.to(device=prediction.device, dtype=torch.float32).reshape(-1)
    if target.numel() == 1 and prediction.numel() > 1:
        target = target.expand_as(prediction)
    if target.numel() != prediction.numel():
        raise ValueError(
            f"SNR head produced {prediction.numel()} values for {target.numel()} targets"
        )
    span = float(max_db) - float(min_db)
    if span <= 0.0:
        raise ValueError("max_db must exceed min_db")
    prediction = (prediction - float(min_db)) / span
    target = (target - float(min_db)) / span
    return F.smooth_l1_loss(prediction, target, beta=float(beta), reduction="mean")


def sync_parameter_cross_snr_consistency_loss(
    original_auxiliary: dict,
    partner_auxiliary: dict,
    original_snr: torch.Tensor,
    partner_snr: torch.Tensor,
    *,
    teacher_high_auxiliary: dict | None = None,
    beta: float = 0.1,
    cfo_scale: float = 0.25,
    phase_drift_scale: float = 0.05,
    force_partner_student: bool = False,
) -> torch.Tensor:
    """Keep physical synchronization estimates invariant across SNR views.

    SNR itself is intentionally excluded because it changes between the two
    views.  The lower-SNR predictions receive gradients; the high-SNR online
    or EMA teacher is always stop-gradient.
    """
    original_is_low = (
        original_snr.to(dtype=torch.float32).reshape(-1)
        <= partner_snr.to(dtype=torch.float32).reshape(-1)
    )
    keys_and_scales = (
        ("cfo_cycles_per_sample", max(abs(float(cfo_scale)), 1e-8)),
        ("phase_vector", 1.0),
        ("timing_offset_unit", 1.0),
        ("sps_probabilities", 1.0),
        (
            "phase_drift_rad_per_sample",
            max(abs(float(phase_drift_scale)), 1e-8),
        ),
    )
    losses = []
    for key, scale in keys_and_scales:
        original = original_auxiliary.get(key)
        partner = partner_auxiliary.get(key)
        if original is None or partner is None:
            continue
        original = original.float()
        partner = partner.float()
        selector = original_is_low.to(device=original.device).view(
            -1, *([1] * (original.ndim - 1))
        )
        low = partner if force_partner_student else torch.where(selector, original, partner)
        if teacher_high_auxiliary is None:
            high = torch.where(selector, partner, original)
        else:
            high = teacher_high_auxiliary.get(key)
            if high is None:
                continue
            high = high.float()
        high = high.detach()
        losses.append(
            F.smooth_l1_loss(
                low / scale,
                high / scale,
                beta=float(beta),
                reduction="mean",
            )
        )
    if not losses:
        reference = next(
            (
                value
                for value in original_auxiliary.values()
                if torch.is_tensor(value)
            ),
            None,
        )
        if reference is None:
            raise ValueError("synchronization auxiliary dictionaries contain no tensors")
        return reference.new_zeros(())
    return torch.stack(losses).mean()


def root_raised_cosine_taps(
    samples_per_symbol: int,
    *,
    rolloff: float = 0.35,
    span: int = 20,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a symmetric, unit-energy root-raised-cosine FIR."""
    sps = max(1, int(samples_per_symbol))
    span = max(1, int(span))
    beta = float(rolloff)
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"rolloff must be in [0, 1], got {rolloff}")
    count = span * sps + 1
    n = torch.arange(count, device=device, dtype=torch.float64) - (count - 1) / 2.0
    t = n / float(sps)

    if beta == 0.0:
        taps = torch.sinc(t)
    else:
        numerator = torch.sin(math.pi * t * (1.0 - beta))
        numerator = numerator + 4.0 * beta * t * torch.cos(math.pi * t * (1.0 + beta))
        denominator = math.pi * t * (1.0 - (4.0 * beta * t).square())
        taps = numerator / denominator

        at_zero = t.abs() < 1e-10
        taps = torch.where(at_zero, taps.new_tensor(1.0 + beta * (4.0 / math.pi - 1.0)), taps)
        singular = (t.abs() - 1.0 / (4.0 * beta)).abs() < 1e-10
        singular_value = beta / math.sqrt(2.0) * (
            (1.0 + 2.0 / math.pi) * math.sin(math.pi / (4.0 * beta))
            + (1.0 - 2.0 / math.pi) * math.cos(math.pi / (4.0 * beta))
        )
        taps = torch.where(singular, taps.new_tensor(singular_value), taps)

    taps = taps / taps.square().sum().sqrt().clamp_min(1e-12)
    return taps.to(dtype=dtype)


def _matched_filter(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    kernel = taps.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
    filtered = F.conv1d(flat, kernel, padding=kernel.size(-1) // 2)
    return filtered.reshape(*shape)


def _constellation(name: str | None, device, dtype: torch.dtype) -> torch.Tensor | None:
    if not name:
        return None
    normalized = str(name).upper().replace("-", "").replace("_", "")
    match = re.search(r"(\d+)(PSK|QAM|APSK)", normalized)
    if normalized.startswith("BPSK"):
        order, family = 2, "PSK"
    elif normalized.startswith("QPSK"):
        order, family = 4, "PSK"
    elif match:
        order, family = int(match.group(1)), match.group(2)
    else:
        return None

    if family == "PSK":
        phase = 2.0 * math.pi * torch.arange(order, device=device, dtype=torch.float32) / order
        points = torch.polar(torch.ones_like(phase), phase)
    elif family == "APSK" and order == 16:
        inner_phase = 2.0 * math.pi * torch.arange(4, device=device, dtype=torch.float32) / 4.0
        outer_phase = 2.0 * math.pi * torch.arange(12, device=device, dtype=torch.float32) / 12.0
        points = torch.cat([torch.polar(torch.ones_like(inner_phase), inner_phase), torch.polar(2.85 * torch.ones_like(outer_phase), outer_phase)])
    elif family == "QAM":
        rows = int(math.floor(math.sqrt(order)))
        while rows > 1 and order % rows != 0:
            rows -= 1
        cols = order // rows
        real = torch.arange(-(cols - 1), cols, 2, device=device, dtype=torch.float32)
        imag = torch.arange(-(rows - 1), rows, 2, device=device, dtype=torch.float32)
        grid_i, grid_q = torch.meshgrid(real, imag, indexing="xy")
        points = torch.complex(grid_i.reshape(-1), grid_q.reshape(-1))
    else:
        return None
    points = points / points.abs().square().mean().sqrt().clamp_min(1e-8)
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    return points.to(dtype=complex_dtype)


def _nearest_constellation_distance(symbols: torch.Tensor, constellation: torch.Tensor) -> torch.Tensor:
    return (symbols.unsqueeze(-1) - constellation.view(1, 1, -1)).abs().amin(dim=-1).mean(dim=-1)


def receiver_domain_symbol_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    source_names: Sequence[str] | None,
    num_sources: int,
    sps_candidates: Sequence[int] = (10, 20),
    rrc_rolloff: float = 0.35,
    rrc_span: int = 20,
    constellation_weight: float = 0.05,
    softmin_temperature: float = 0.1,
    beta: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Matched-filter symbol EVM with target-timing and SPS soft selection."""
    preds = _pit_reorder_to_targets(outputs, targets, num_sources)
    refs = _split_iq_sources(targets, num_sources)
    candidates = tuple(sorted({max(1, int(value)) for value in sps_candidates}))
    if not candidates:
        raise ValueError("sps_candidates must not be empty")
    names = list(source_names or [])
    source_losses = []

    for source_idx in range(num_sources):
        pred_source = preds[:, source_idx]
        ref_source = refs[:, source_idx]
        constellation = _constellation(
            names[source_idx] if source_idx < len(names) else None,
            outputs.device,
            outputs.dtype,
        )
        sps_losses = []
        for sps in candidates:
            taps = root_raised_cosine_taps(
                sps,
                rolloff=rrc_rolloff,
                span=rrc_span,
                device=outputs.device,
                dtype=outputs.dtype,
            )
            pred_filtered = _matched_filter(pred_source, taps)
            ref_filtered = _matched_filter(ref_source, taps)
            offset_losses = []
            timing_scores = []
            for offset in range(sps):
                pred_symbols = pred_filtered[..., offset::sps]
                ref_symbols = ref_filtered[..., offset::sps]
                trim = min(max(0, int(rrc_span) // 2), max(0, pred_symbols.size(-1) // 4))
                if trim > 0 and pred_symbols.size(-1) > 2 * trim:
                    pred_symbols = pred_symbols[..., trim:-trim]
                    ref_symbols = ref_symbols[..., trim:-trim]
                aligned, target_c = _complex_align(pred_symbols, ref_symbols, eps)
                target_rms = target_c.abs().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
                normalized_error = torch.stack(
                    [(aligned.real - target_c.real) / target_rms, (aligned.imag - target_c.imag) / target_rms],
                    dim=-1,
                )
                evm = F.smooth_l1_loss(
                    normalized_error,
                    torch.zeros_like(normalized_error),
                    beta=float(beta),
                    reduction="none",
                ).mean(dim=(-1, -2))
                if constellation is not None and float(constellation_weight) > 0.0:
                    aligned_normalized = aligned / target_rms
                    const_loss = _nearest_constellation_distance(aligned_normalized, constellation)
                    evm = evm + float(constellation_weight) * const_loss
                offset_losses.append(evm)
                target_normalized = target_c / target_rms
                if constellation is not None:
                    timing_scores.append(_nearest_constellation_distance(target_normalized, constellation))
                else:
                    timing_scores.append(-target_c.abs().square().mean(dim=-1))

            all_offset_losses = torch.stack(offset_losses, dim=-1)
            all_timing_scores = torch.stack(timing_scores, dim=-1)
            timing_temperature = max(float(softmin_temperature), 1e-4)
            timing_weights = torch.softmax(-all_timing_scores / timing_temperature, dim=-1)
            sps_losses.append((all_offset_losses * timing_weights).sum(dim=-1))

        all_sps_losses = torch.stack(sps_losses, dim=-1)
        temperature = max(float(softmin_temperature), 1e-4)
        softmin = -temperature * (
            torch.logsumexp(-all_sps_losses / temperature, dim=-1)
            - math.log(len(sps_losses))
        )
        source_losses.append(softmin)

    return torch.stack(source_losses, dim=1).mean()
