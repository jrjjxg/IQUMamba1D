"""Fixed-slot RF transformation equivariance helpers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_iq_sources(x: torch.Tensor, num_sources: int) -> None:
    if x.ndim != 3 or x.size(1) != 2 * int(num_sources):
        raise ValueError(
            f"Expected (B, {2 * int(num_sources)}, L), got {tuple(x.shape)}"
        )


def sample_fixed_slot_rf_parameters(
    batch_size: int,
    num_sources: int,
    *,
    max_phase_degrees: float,
    max_cfo_cycles_per_sample: float,
    max_gain_db: float,
    max_shift_samples: int,
    conjugate_probability: float,
    source_mode: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample global or per-source RF transformations without data labels."""

    batch_size = int(batch_size)
    num_sources = int(num_sources)
    source_mode = str(source_mode).lower()
    if source_mode not in {"global", "per_source"}:
        raise ValueError("source_mode must be 'global' or 'per_source'")
    sampled_sources = 1 if source_mode == "global" else num_sources
    shape = (batch_size, sampled_sources)

    def symmetric(limit: float) -> torch.Tensor:
        limit = abs(float(limit))
        if limit == 0.0:
            return torch.zeros(shape, device=device, dtype=torch.float32)
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(-limit, limit)

    phase = symmetric(abs(float(max_phase_degrees)) * math.pi / 180.0)
    cfo = symmetric(max_cfo_cycles_per_sample)
    gain_db = symmetric(max_gain_db)
    max_shift = max(0, int(max_shift_samples))
    if max_shift == 0:
        shift = torch.zeros(shape, device=device, dtype=torch.long)
    else:
        shift = torch.randint(
            -max_shift, max_shift + 1, shape, device=device, dtype=torch.long
        )
    conjugate_probability = min(max(float(conjugate_probability), 0.0), 1.0)
    conjugate = torch.rand(shape, device=device) < conjugate_probability

    if source_mode == "global" and num_sources > 1:
        phase = phase.expand(-1, num_sources)
        cfo = cfo.expand(-1, num_sources)
        gain_db = gain_db.expand(-1, num_sources)
        shift = shift.expand(-1, num_sources)
        conjugate = conjugate.expand(-1, num_sources)
    return {
        "phase_rad": phase,
        "cfo_cycles_per_sample": cfo,
        "gain_db": gain_db,
        "shift_samples": shift,
        "conjugate": conjugate,
    }


def apply_fixed_slot_rf_transform(
    x: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_sources: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or invert one RF transform per fixed source slot."""

    num_sources = int(num_sources)
    _validate_iq_sources(x, num_sources)
    batch_size, _, length = x.shape
    view = x.reshape(batch_size, num_sources, 2, length)
    real = view[:, :, 0, :]
    imag = view[:, :, 1, :]

    def parameter(name: str, dtype=None) -> torch.Tensor:
        value = parameters[name].to(device=x.device)
        if tuple(value.shape) != (batch_size, num_sources):
            raise ValueError(
                f"RF parameter {name} expected {(batch_size, num_sources)}, "
                f"got {tuple(value.shape)}"
            )
        return value if dtype is None else value.to(dtype=dtype)

    phase = parameter("phase_rad", torch.float32)
    cfo = parameter("cfo_cycles_per_sample", torch.float32)
    gain_db = parameter("gain_db", torch.float32)
    shifts = parameter("shift_samples").long()
    conjugate = parameter("conjugate").bool()
    time = torch.arange(length, device=x.device, dtype=torch.float32).view(1, 1, -1)
    angle = phase.unsqueeze(-1) + 2.0 * math.pi * cfo.unsqueeze(-1) * time
    gain = torch.pow(
        torch.tensor(10.0, device=x.device, dtype=torch.float32),
        gain_db.unsqueeze(-1) / 20.0,
    )

    def variable_roll(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(length, device=x.device).view(1, 1, -1)
        indices = torch.remainder(positions - offsets.unsqueeze(-1), length).long()
        return values.gather(-1, indices)

    if inverse:
        cosine = torch.cos(angle).to(dtype=real.dtype)
        sine = torch.sin(angle).to(dtype=real.dtype)
        scale = gain.to(dtype=real.dtype).reciprocal()
        restored_real = scale * (cosine * real + sine * imag)
        restored_imag = scale * (-sine * real + cosine * imag)
        restored_imag = torch.where(
            conjugate.unsqueeze(-1), -restored_imag, restored_imag
        )
        real = variable_roll(restored_real, -shifts)
        imag = variable_roll(restored_imag, -shifts)
    else:
        real = variable_roll(real, shifts)
        imag = variable_roll(imag, shifts)
        imag = torch.where(conjugate.unsqueeze(-1), -imag, imag)
        cosine = torch.cos(angle).to(dtype=real.dtype)
        sine = torch.sin(angle).to(dtype=real.dtype)
        scale = gain.to(dtype=real.dtype)
        transformed_real = scale * (cosine * real - sine * imag)
        transformed_imag = scale * (sine * real + cosine * imag)
        real, imag = transformed_real, transformed_imag
    return torch.stack((real, imag), dim=2).reshape_as(x)


def build_fixed_slot_rf_view(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_sources: int,
    source_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a transformed mixture while preserving the measured noise residual."""

    num_sources = int(num_sources)
    _validate_iq_sources(targets, num_sources)
    if inputs.ndim != 3 or inputs.size(1) != 2:
        raise ValueError(f"Expected mixture (B, 2, L), got {tuple(inputs.shape)}")
    transformed_targets = apply_fixed_slot_rf_transform(
        targets, parameters, num_sources=num_sources
    )
    if str(source_mode).lower() == "global":
        mixture_parameters = {name: value[:, :1] for name, value in parameters.items()}
        transformed_inputs = apply_fixed_slot_rf_transform(
            inputs, mixture_parameters, num_sources=1
        )
    elif str(source_mode).lower() == "per_source":
        clean = targets.reshape(inputs.size(0), num_sources, 2, inputs.size(-1)).sum(dim=1)
        transformed_clean = transformed_targets.reshape(
            inputs.size(0), num_sources, 2, inputs.size(-1)
        ).sum(dim=1)
        transformed_inputs = transformed_clean + (inputs - clean)
    else:
        raise ValueError("source_mode must be 'global' or 'per_source'")
    return transformed_inputs, transformed_targets


def fixed_slot_rf_equivariance_consistency_loss(
    original_outputs: torch.Tensor,
    transformed_outputs: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    *,
    num_sources: int,
    beta: float = 0.5,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Compare inverse-transformed outputs in fixed slots, without PIT."""

    num_sources = int(num_sources)
    _validate_iq_sources(original_outputs, num_sources)
    _validate_iq_sources(transformed_outputs, num_sources)
    restored = apply_fixed_slot_rf_transform(
        transformed_outputs.float(),
        parameters,
        num_sources=num_sources,
        inverse=True,
    ).reshape(original_outputs.size(0), num_sources, 2, original_outputs.size(-1))
    teacher = original_outputs.detach().float().reshape_as(restored)
    rms = teacher.square().sum(dim=2).mean(dim=-1, keepdim=True).sqrt()
    rms = rms.clamp_min(float(eps)).unsqueeze(2)
    error = (restored - teacher) / rms
    return F.smooth_l1_loss(
        error,
        torch.zeros_like(error),
        beta=float(beta),
        reduction="mean",
    )


__all__ = [
    "apply_fixed_slot_rf_transform",
    "build_fixed_slot_rf_view",
    "fixed_slot_rf_equivariance_consistency_loss",
    "sample_fixed_slot_rf_parameters",
]
