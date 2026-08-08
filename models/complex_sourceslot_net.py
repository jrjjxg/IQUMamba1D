"""Complex source-slot separator for IQ blind source separation.

This model changes the output paradigm from mask estimation to direct
source-slot estimation:

    IQ waveform -> complex encoder expanded into K source slots
    -> temporal convolution + source-axis attention over slots
    -> direct source-slot decoding -> mixture consistency.

The source slots are permutation-free at the architecture level; PIT loss still
decides the final source ordering during training.  The model only consumes the
observed mixture IQ waveform.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ctdcrn import ComplexConv1d
from models.complex_dpnet import HALF_PRECISION_DTYPES
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


class ComplexSourceSlotEncoder(nn.Module):
    """Complex encoder that expands one mixture into K learnable source slots."""

    def __init__(
        self,
        n_srcs: int = 2,
        slot_channels: int = 64,
        kernel_size: int = 9,
        identity_split_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.slot_channels = int(slot_channels)
        self.encoder = ComplexConv1d(
            1,
            self.n_srcs * self.slot_channels,
            kernel_size=int(kernel_size),
            padding="same",
            bias=True,
        )
        if identity_split_init:
            self._init_equal_split()

    def _init_equal_split(self) -> None:
        with torch.no_grad():
            self.encoder.conv_re.weight.zero_()
            self.encoder.conv_im.weight.zero_()
            if self.encoder.bias_re is not None:
                self.encoder.bias_re.zero_()
                self.encoder.bias_im.zero_()
            center = self.encoder.conv_re.weight.size(-1) // 2
            for source_idx in range(self.n_srcs):
                out_idx = source_idx * self.slot_channels
                self.encoder.conv_re.weight[out_idx, 0, center] = 1.0 / self.n_srcs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected (B,2,L), got {tuple(x.shape)}")
        encoded = self.encoder(x.unsqueeze(2))
        batch, _, _, length = encoded.shape
        encoded = encoded.reshape(batch, 2, self.n_srcs, self.slot_channels, length)
        return encoded.permute(0, 2, 1, 3, 4).contiguous()


class ComplexSourceSlotDecoder(nn.Module):
    """Shared complex synthesis transform for direct source-slot decoding."""

    def __init__(
        self,
        slot_channels: int = 64,
        kernel_size: int = 9,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.slot_channels = int(slot_channels)
        self.decoder = ComplexConv1d(
            self.slot_channels,
            1,
            kernel_size=int(kernel_size),
            padding="same",
            bias=True,
        )
        if identity_init:
            self._init_identity_decoder()

    def _init_identity_decoder(self) -> None:
        with torch.no_grad():
            self.decoder.conv_re.weight.zero_()
            self.decoder.conv_im.weight.zero_()
            if self.decoder.bias_re is not None:
                self.decoder.bias_re.zero_()
                self.decoder.bias_im.zero_()
            center = self.decoder.conv_re.weight.size(-1) // 2
            self.decoder.conv_re.weight[0, 0, center] = 1.0

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 5 or slots.size(2) != 2:
            raise ValueError(f"Expected (B,K,2,C,L), got {tuple(slots.shape)}")
        batch, n_srcs, _, channels, length = slots.shape
        if channels != self.slot_channels:
            raise ValueError(f"Expected {self.slot_channels} slot channels, got {channels}")

        flat = slots.reshape(batch * n_srcs, 2, channels, length)
        decoded = self.decoder(flat).squeeze(2)
        decoded = decoded.reshape(batch, n_srcs, 2, length)
        return decoded.reshape(batch, 2 * n_srcs, length)


class SourceSlotMixerBlock(nn.Module):
    """Temporal convolution plus source-axis attention over source slots."""

    def __init__(
        self,
        slot_channels: int,
        hidden_dim: int = 128,
        temporal_kernel_size: int = 5,
        dilation: int = 1,
        source_attention_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.slot_channels = int(slot_channels)
        feature_dim = 2 * self.slot_channels
        source_attention_heads = int(source_attention_heads)
        if source_attention_heads < 1:
            raise ValueError(f"source_attention_heads must be >= 1, got {source_attention_heads}")
        if feature_dim % source_attention_heads != 0:
            raise ValueError(
                f"2*slot_channels ({feature_dim}) must be divisible by "
                f"source_attention_heads ({source_attention_heads})"
            )
        padding = int(dilation) * (int(temporal_kernel_size) - 1) // 2

        self.temporal = nn.Sequential(
            nn.GroupNorm(1, feature_dim),
            nn.PReLU(),
            nn.Conv1d(feature_dim, int(hidden_dim), 1),
            nn.PReLU(),
            nn.Conv1d(
                int(hidden_dim),
                int(hidden_dim),
                kernel_size=int(temporal_kernel_size),
                padding=padding,
                dilation=int(dilation),
                groups=int(hidden_dim),
            ),
            nn.PReLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(hidden_dim), feature_dim, 1),
        )
        self.source_norm = nn.LayerNorm(feature_dim)
        self.source_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=source_attention_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.source_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _slots_to_channels(self, slots: torch.Tensor) -> torch.Tensor:
        batch, n_srcs, _, channels, length = slots.shape
        return slots.reshape(batch, n_srcs, 2 * channels, length)

    def _channels_to_slots(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_srcs, _, length = x.shape
        return x.reshape(batch, n_srcs, 2, self.slot_channels, length)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 5 or slots.size(2) != 2:
            raise ValueError(f"Expected (B,K,2,C,L), got {tuple(slots.shape)}")
        batch, n_srcs, _, channels, length = slots.shape
        if channels != self.slot_channels:
            raise ValueError(f"Expected {self.slot_channels} slot channels, got {channels}")

        x = self._slots_to_channels(slots)
        temporal_in = x.reshape(batch * n_srcs, 2 * channels, length)
        temporal_update = self.temporal(temporal_in)
        if temporal_update.size(-1) != length:
            temporal_update = F.interpolate(
                temporal_update,
                size=length,
                mode="linear",
                align_corners=False,
            )
        temporal_update = temporal_update.reshape(batch, n_srcs, 2 * channels, length)

        source_tokens = x.mean(dim=-1)
        source_tokens = self.source_norm(source_tokens)
        source_update, _ = self.source_attn(
            source_tokens,
            source_tokens,
            source_tokens,
            need_weights=False,
        )
        source_update = self.source_proj(source_update).unsqueeze(-1)

        mixed = x + self.residual_scale * (temporal_update + source_update)
        return self._channels_to_slots(mixed)


class ComplexSourceSlotSeparator1D(nn.Module):
    """Direct source-slot complex separator with IQ-compatible input/output."""

    def __init__(
        self,
        n_srcs: int = 2,
        slot_channels: int = 64,
        kernel_size: int = 9,
        hidden_dim: int = 128,
        n_layers: int = 8,
        temporal_kernel_size: int = 5,
        dilation_cycle: int = 4,
        source_attention_heads: int = 4,
        dropout: float = 0.0,
        identity_split_init: bool = True,
        slot_residual_scale_init: float = 0.0,
        apply_projection: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_eps: float = 1e-8,
        mc_detach_weights: bool = False,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.slot_channels = int(slot_channels)
        self.apply_projection = bool(apply_projection)
        self.encoder = ComplexSourceSlotEncoder(
            n_srcs=self.n_srcs,
            slot_channels=self.slot_channels,
            kernel_size=int(kernel_size),
            identity_split_init=bool(identity_split_init),
        )
        dilation_cycle = max(1, int(dilation_cycle))
        self.blocks = nn.ModuleList(
            [
                SourceSlotMixerBlock(
                    slot_channels=self.slot_channels,
                    hidden_dim=int(hidden_dim),
                    temporal_kernel_size=int(temporal_kernel_size),
                    dilation=2 ** (idx % dilation_cycle),
                    source_attention_heads=int(source_attention_heads),
                    dropout=float(dropout),
                    residual_scale_init=float(slot_residual_scale_init),
                )
                for idx in range(int(n_layers))
            ]
        )
        self.decoder = ComplexSourceSlotDecoder(
            slot_channels=self.slot_channels,
            kernel_size=int(kernel_size),
            identity_init=bool(identity_split_init),
        )
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.n_srcs,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=float(mc_eps),
            detach_weights=mc_detach_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected input (B,2,L), got {tuple(x.shape)}")
        original_dtype = x.dtype
        if original_dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        slots = self.encoder(x)
        for block in self.blocks:
            slots = block(slots)
        output = self.decoder(slots)
        if self.apply_projection:
            output = self.mc_projection(output, x)
        return output.to(dtype=original_dtype)
