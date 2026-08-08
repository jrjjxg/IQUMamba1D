"""Complex-ConvTasNet for single-channel IQ blind source separation.

This is a fully convolutional time-domain separator adapted to complex IQ:

    IQ waveform -> learned complex analysis features -> dilated TCN separator
    -> complex masks -> shared complex synthesis decoder -> mixture consistency.

It follows the Conv-TasNet paradigm of learned analysis/synthesis transforms
and temporal convolutional mask estimation, but keeps I/Q coupling explicit
through complex-valued convolutions and complex masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.complex_dpnet import (
    ComplexFeatureEncoder,
    ComplexMaskDecoder,
    HALF_PRECISION_DTYPES,
)
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


class ComplexTemporalConvBlock(nn.Module):
    """Depthwise-separable dilated TCN block used by Complex-ConvTasNet."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be positive, got {dilation}")

        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.GroupNorm(1, int(hidden_dim)),
            nn.PReLU(),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), 1),
            nn.PReLU(),
            nn.Conv1d(
                int(hidden_dim),
                int(hidden_dim),
                kernel_size=int(kernel_size),
                padding=int(padding),
                dilation=int(dilation),
                groups=int(hidden_dim),
            ),
            nn.PReLU(),
            nn.GroupNorm(1, int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,L), got {tuple(x.shape)}")
        y = self.net(x)
        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        return x + y


class ComplexTCNSeparator(nn.Module):
    """Dilated TCN that predicts complex masks over learned complex features."""

    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        hidden_dim: int = 128,
        bottleneck_dim: int = 128,
        num_repeats: int = 3,
        blocks_per_repeat: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.0,
        mask_head_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_norm = nn.GroupNorm(1, int(in_channels))
        self.input_proj = nn.Conv1d(int(in_channels), int(bottleneck_dim), 1)

        blocks = []
        for _ in range(int(num_repeats)):
            for block_index in range(int(blocks_per_repeat)):
                blocks.append(
                    ComplexTemporalConvBlock(
                        hidden_dim=int(bottleneck_dim),
                        kernel_size=int(kernel_size),
                        dilation=2**block_index,
                        dropout=float(dropout),
                    )
                )
        self.tcn = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(int(bottleneck_dim), int(hidden_dim), 1),
            nn.PReLU(),
            nn.Conv1d(int(hidden_dim), int(mask_channels), 1),
        )
        if mask_head_zero_init:
            self._init_zero_mask_head()

    def _init_zero_mask_head(self) -> None:
        final_conv = self.output[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,L), got {tuple(x.shape)}")
        x = self.input_proj(self.input_norm(x))
        x = self.tcn(x)
        return self.output(x)


class ComplexConvTasNetSeparator1D(nn.Module):
    """Learned complex filterbank + dilated TCN separator with IQ-compatible I/O."""

    def __init__(
        self,
        n_srcs: int = 2,
        feature_channels: int = 64,
        kernel_size: int = 9,
        hidden_dim: int = 128,
        bottleneck_dim: int = 128,
        num_repeats: int = 3,
        blocks_per_repeat: int = 8,
        tcn_kernel_size: int = 3,
        dropout: float = 0.0,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        identity_init: bool = True,
        mask_head_zero_init: bool = True,
        apply_projection: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_eps: float = 1e-8,
        mc_detach_weights: bool = False,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.feature_channels = int(feature_channels)
        self.apply_projection = bool(apply_projection)
        self.encoder = ComplexFeatureEncoder(
            feature_channels=self.feature_channels,
            kernel_size=int(kernel_size),
            identity_init=bool(identity_init),
        )
        self.separator = ComplexTCNSeparator(
            in_channels=2 * self.feature_channels,
            mask_channels=2 * self.n_srcs * self.feature_channels,
            hidden_dim=int(hidden_dim),
            bottleneck_dim=int(bottleneck_dim),
            num_repeats=int(num_repeats),
            blocks_per_repeat=int(blocks_per_repeat),
            kernel_size=int(tcn_kernel_size),
            dropout=float(dropout),
            mask_head_zero_init=bool(mask_head_zero_init),
        )
        self.mask_decoder = ComplexMaskDecoder(
            n_srcs=self.n_srcs,
            feature_channels=self.feature_channels,
            kernel_size=int(kernel_size),
            mask_bound=float(mask_bound),
            mask_sum_constraint=bool(mask_sum_constraint),
            identity_init=bool(identity_init),
        )
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.n_srcs,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=float(mc_eps),
            detach_weights=mc_detach_weights,
        )

    def _features_to_real_channels(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, channels, length = features.shape
        return features.reshape(batch, 2 * channels, length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected input (B,2,L), got {tuple(x.shape)}")
        original_dtype = x.dtype
        if original_dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        features = self.encoder(x)
        separator_input = self._features_to_real_channels(features)
        mask_logits = self.separator(separator_input)
        output = self.mask_decoder(features, mask_logits)
        if self.apply_projection:
            output = self.mc_projection(output, x)
        return output.to(dtype=original_dtype)
