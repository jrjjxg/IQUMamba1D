"""Stage-4 IQUMamba with source-wise cyclic-Wiener residual refinement.

The wrapper keeps the original IQUMamba backbone intact.  After the backbone
predicts I/Q source slots, this module builds source-wise frequency-shifted
views of the observed mixture and learns a small near-identity residual
correction for each source.  A configurable additive mixture projection can be
applied at the end.
"""

from __future__ import annotations

import math
from typing import List, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = max(1, int(kernel_size))
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _split_iq_sources(estimates: torch.Tensor, num_sources: int | None = None) -> torch.Tensor:
    if estimates.dim() != 3 or estimates.size(1) % 2 != 0:
        raise ValueError(f"Expected I/Q source tensor shaped (B, 2K, L), got {tuple(estimates.shape)}")
    inferred = estimates.size(1) // 2
    if num_sources is None:
        num_sources = inferred
    if int(num_sources) != inferred:
        raise ValueError(f"num_sources={num_sources} does not match tensor channels {inferred}")
    return estimates.view(estimates.size(0), num_sources, 2, estimates.size(-1))


def _merge_iq_sources(sources: torch.Tensor) -> torch.Tensor:
    if sources.dim() != 4 or sources.size(2) != 2:
        raise ValueError(f"Expected source tensor shaped (B, K, 2, L), got {tuple(sources.shape)}")
    return sources.flatten(1, 2)


def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
    if mixture.dim() != 3 or mixture.size(1) != 2:
        raise ValueError(f"Expected mixture tensor shaped (B, 2, L), got {tuple(mixture.shape)}")
    if mixture.size(-1) == target_length:
        return mixture
    return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)


def estimate_source_cyclic_frequencies(
    sources: torch.Tensor,
    min_freq: float = 1.0 / 128.0,
    max_freq: float = 1.0 / 4.0,
    default_freq: float = 1.0 / 32.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate one normalized cyclic frequency per source slot.

    Args:
        sources: Tensor shaped (B, K, 2, L).

    Returns:
        Tensor shaped (B, K) with frequencies in cycles/sample.
    """
    if sources.dim() != 4 or sources.size(2) != 2:
        raise ValueError(f"Expected source tensor shaped (B, K, 2, L), got {tuple(sources.shape)}")

    batch, num_sources, _, length = sources.shape
    min_freq = max(0.0, float(min_freq))
    max_freq = min(0.5, float(max_freq))
    if length < 8 or min_freq >= max_freq:
        return sources.new_full((batch, num_sources), float(default_freq))

    power = sources[:, :, 0].float().square() + sources[:, :, 1].float().square()
    centered = power - power.mean(dim=-1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=-1)
    spec_power = spectrum.abs().square()
    freqs = torch.fft.rfftfreq(length, d=1.0).to(device=sources.device)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not bool(mask.any()):
        return sources.new_full((batch, num_sources), float(default_freq))

    masked_power = spec_power[..., mask]
    masked_freqs = freqs[mask].to(device=sources.device, dtype=sources.dtype)
    peak_idx = torch.argmax(masked_power, dim=-1)
    peak_freqs = masked_freqs[peak_idx]
    peak_power = masked_power.gather(-1, peak_idx.unsqueeze(-1)).squeeze(-1)
    fallback = sources.new_full((batch, num_sources), float(default_freq))
    peak_freqs = torch.where(peak_power.to(dtype=sources.dtype) > float(eps), peak_freqs, fallback)
    return torch.nan_to_num(peak_freqs, nan=float(default_freq), posinf=float(default_freq), neginf=float(default_freq))


class CyclicWienerResidualHead1D(nn.Module):
    """Source-wise frequency-shift residual head followed by mixture projection."""

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 16,
        kernel_size: int = 9,
        min_freq: float = 1.0 / 128.0,
        max_freq: float = 1.0 / 4.0,
        default_freq: float = 1.0 / 32.0,
        num_harmonics: int = 2,
        scale_init: float = 0.01,
        projection_strength: float = 0.5,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        if self.num_sources < 1:
            raise ValueError("num_sources must be positive")
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.default_freq = float(default_freq)
        self.num_harmonics = max(1, int(num_harmonics))
        self.projection_strength = float(min(max(projection_strength, 0.0), 1.0))
        self.eps = float(eps)

        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        self.num_shift_branches = 1 + 2 * self.num_harmonics
        in_complex_channels = self.num_shift_branches + 1

        self.in_proj = ComplexTiedConv1d(
            in_complex_channels=in_complex_channels,
            out_complex_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=hidden_channels,
            out_complex_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(in_complex_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    def _branch_multipliers(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = [0.0]
        for harmonic in range(1, self.num_harmonics + 1):
            values.extend([float(harmonic), -float(harmonic)])
        return torch.tensor(values, device=device, dtype=dtype)

    def _shifted_mixture_bank(self, mixture: torch.Tensor, sources: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_sources, _, length = sources.shape
        freqs = estimate_source_cyclic_frequencies(
            sources.detach(),
            min_freq=self.min_freq,
            max_freq=self.max_freq,
            default_freq=self.default_freq,
            eps=self.eps,
        ).to(device=sources.device, dtype=torch.float32)
        multipliers = self._branch_multipliers(sources.device, torch.float32)
        alphas = (freqs.unsqueeze(-1) * multipliers.view(1, 1, -1)).clamp(min=-0.5, max=0.5)

        n = torch.arange(length, device=sources.device, dtype=torch.float32)
        phase = -2.0 * math.pi * alphas.unsqueeze(-1) * n.view(1, 1, 1, -1)
        cos = torch.cos(phase).to(dtype=sources.dtype)
        sin = torch.sin(phase).to(dtype=sources.dtype)

        real = mixture[:, 0].view(batch, 1, 1, length)
        imag = mixture[:, 1].view(batch, 1, 1, length)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos

        source_real = sources[:, :, 0].unsqueeze(2)
        source_imag = sources[:, :, 1].unsqueeze(2)
        branch_real = torch.cat([shifted_real, source_real], dim=2)
        branch_imag = torch.cat([shifted_imag, source_imag], dim=2)
        return (
            branch_real.reshape(batch * num_sources, branch_real.size(2), length),
            branch_imag.reshape(batch * num_sources, branch_imag.size(2), length),
        )

    def _residual_refine(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        sources = _split_iq_sources(estimates, self.num_sources)
        mixture = _resize_mixture(mixture, estimates.size(-1)).to(device=estimates.device, dtype=estimates.dtype)
        branch_real, branch_imag = self._shifted_mixture_bank(mixture, sources)
        hidden_real, hidden_imag = self.in_proj(branch_real, branch_imag)
        amp = torch.sqrt(branch_real.square() + branch_imag.square() + self.eps)
        gate = self.gate(amp)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)

        batch, num_sources, _, length = sources.shape
        delta = torch.stack(
            [
                delta_real.view(batch, num_sources, length),
                delta_imag.view(batch, num_sources, length),
            ],
            dim=2,
        )
        refined = sources + self.scale.to(dtype=estimates.dtype) * torch.tanh(delta)
        return _merge_iq_sources(refined)

    def _apply_mixture_projection(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if self.projection_strength <= 0.0:
            return estimates
        sources = _split_iq_sources(estimates, self.num_sources)
        mixture = _resize_mixture(mixture, estimates.size(-1)).to(device=estimates.device, dtype=estimates.dtype)
        residual = sources.sum(dim=1) - mixture
        corrected = sources - (self.projection_strength / float(self.num_sources)) * residual.unsqueeze(1)
        return _merge_iq_sources(corrected)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        refined = self._residual_refine(estimates, mixture)
        return self._apply_mixture_projection(refined, mixture)


class IQUMamba1D_CyclicWienerResidual(nn.Module):
    """IQUMamba backbone followed by source-wise cyclic-Wiener residual refinement."""

    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[nn.Conv1d],
        kernel_sizes: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        cyclic_wiener_hidden_channels: int = 16,
        cyclic_wiener_kernel_size: int = 9,
        cyclic_wiener_min_freq: float = 1.0 / 128.0,
        cyclic_wiener_max_freq: float = 1.0 / 4.0,
        cyclic_wiener_default_freq: float = 1.0 / 32.0,
        cyclic_wiener_num_harmonics: int = 2,
        cyclic_wiener_scale_init: float = 0.01,
        cyclic_wiener_projection_strength: float = 0.5,
        cyclic_wiener_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"stage 196 expects one I/Q mixture with 2 channels, got {input_channels}")
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must contain I/Q source pairs, got {num_classes}")
        self.backbone = IQUMamba1D(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.cyclic_wiener_head = CyclicWienerResidualHead1D(
            num_sources=num_classes // 2,
            hidden_channels=cyclic_wiener_hidden_channels,
            kernel_size=cyclic_wiener_kernel_size,
            min_freq=cyclic_wiener_min_freq,
            max_freq=cyclic_wiener_max_freq,
            default_freq=cyclic_wiener_default_freq,
            num_harmonics=cyclic_wiener_num_harmonics,
            scale_init=cyclic_wiener_scale_init,
            projection_strength=cyclic_wiener_projection_strength,
            zero_init=cyclic_wiener_zero_init,
        )

    def _refine(self, output: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        return self.cyclic_wiener_head(output, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = self.backbone(x)
        if isinstance(outputs, (list, tuple)):
            return [self._refine(out, x) for out in outputs]
        return self._refine(outputs, x)
