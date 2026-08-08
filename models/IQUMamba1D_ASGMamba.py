"""Adaptive Spectral Gating Mamba wrapper for stage-4 IQUMamba.

The module adapts ASGMamba's local-FFT gating idea to I/Q blind separation by
placing a lightweight spectral gate immediately before selected encoder Mamba
layers.  Patch size and stride are fixed constants, so the local FFT path stays
linear in sequence length.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUMamba1D import IQUMamba1D


class AdaptiveSpectralGating1D(nn.Module):
    """Local FFT energy gate for 1D feature maps shaped (B, C, L)."""

    def __init__(
        self,
        patch_size: int = 32,
        stride: int = 16,
        num_bands: int = 3,
        gate_hidden: int = 8,
        scale_init: float = 0.01,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.patch_size = max(2, int(patch_size))
        self.stride = max(1, int(stride))
        self.num_bands = max(1, int(num_bands))
        self.eps = float(eps)

        gate_hidden = max(1, int(gate_hidden))
        self.mlp = nn.Sequential(
            nn.Linear(self.num_bands, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        if zero_init:
            final = self.mlp[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _band_descriptors(self, x: torch.Tensor) -> torch.Tensor:
        b, c, length = x.shape
        if length < self.patch_size:
            x = F.pad(x, (0, self.patch_size - length))

        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        spectrum = torch.fft.rfft(patches.float(), dim=-1)
        power = spectrum.abs().square() + self.eps
        total = power.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        freq_bins = power.size(-1)
        band_values = []
        for band_idx in range(self.num_bands):
            start = int(round(band_idx * freq_bins / self.num_bands))
            end = int(round((band_idx + 1) * freq_bins / self.num_bands))
            start = min(max(start, 0), freq_bins - 1)
            end = min(max(end, start + 1), freq_bins)
            band_values.append(power[..., start:end].sum(dim=-1))

        desc = torch.stack(band_values, dim=-1) / total
        return desc.reshape(b, c, patches.size(-2), self.num_bands)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"AdaptiveSpectralGating1D expects (B, C, L), got {tuple(x.shape)}")

        b, c, length = x.shape
        desc = self._band_descriptors(x)
        patch_count = desc.size(-2)
        logits = self.mlp(desc.reshape(b * c * patch_count, self.num_bands)).reshape(b, c, patch_count)
        gate_map = F.interpolate(
            logits.reshape(b * c, 1, patch_count),
            size=length,
            mode="linear",
            align_corners=False,
        ).reshape(b, c, length)
        gate_map = (2.0 * torch.sigmoid(gate_map) - 1.0).to(dtype=x.dtype)
        gated = x * (1.0 + self.scale.to(dtype=x.dtype) * gate_map)
        return gated, gate_map


class ASGMambaLayer(nn.Module):
    """Wrap an existing MambaLayer with local spectral gating before it."""

    def __init__(
        self,
        original_mamba: nn.Module,
        patch_size: int = 32,
        stride: int = 16,
        num_bands: int = 3,
        gate_hidden: int = 8,
        scale_init: float = 0.01,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.original_mamba = original_mamba
        self.spectral_gate = AdaptiveSpectralGating1D(
            patch_size=patch_size,
            stride=stride,
            num_bands=num_bands,
            gate_hidden=gate_hidden,
            scale_init=scale_init,
            zero_init=zero_init,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated, _ = self.spectral_gate(x)
        return self.original_mamba(gated)


class IQUMamba1D_ASGMamba(nn.Module):
    """Stage-4 IQUMamba with ASG wrappers on selected encoder Mamba layers."""

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
        asg_patch_size: int = 32,
        asg_stride: int = 16,
        asg_num_bands: int = 3,
        asg_gate_hidden: int = 8,
        asg_scale_init: float = 0.01,
        asg_zero_init: bool = True,
        asg_apply_stages: Sequence[int] = (1, 3),
        asg_eps: float = 1e-8,
    ) -> None:
        super().__init__()
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

        apply_stages = {int(stage) for stage in asg_apply_stages}
        for stage_idx, layer in enumerate(self.backbone.encoder.mamba_layers):
            if stage_idx not in apply_stages or isinstance(layer, nn.Identity):
                continue
            self.backbone.encoder.mamba_layers[stage_idx] = ASGMambaLayer(
                original_mamba=layer,
                patch_size=asg_patch_size,
                stride=asg_stride,
                num_bands=asg_num_bands,
                gate_hidden=asg_gate_hidden,
                scale_init=asg_scale_init,
                zero_init=asg_zero_init,
                eps=asg_eps,
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.backbone(x)
