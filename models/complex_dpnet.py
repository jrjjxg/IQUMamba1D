"""Complex-DPNet: learned complex encoder + dual-path separator for IQ BSS.

This is a non-IQUMamba framework-level separator.  It follows the
Conv-TasNet / DPRNN / SepFormer family:

    IQ waveform -> learned complex features -> dual-path sequence model
    -> complex feature masks -> shared complex decoder -> mixture consistency.

No modulation labels, symbol timing, cyclic-frequency estimates, or source
metadata are used.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ctdcrn import ComplexConv1d
from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class ComplexFeatureEncoder(nn.Module):
    """Learned complex convolutional analysis transform."""

    def __init__(
        self,
        feature_channels: int = 64,
        kernel_size: int = 9,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.encoder = ComplexConv1d(
            1,
            self.feature_channels,
            kernel_size=int(kernel_size),
            padding="same",
            bias=True,
        )
        if identity_init:
            self._init_identity_first_channel()

    def _init_identity_first_channel(self) -> None:
        with torch.no_grad():
            self.encoder.conv_re.weight.zero_()
            self.encoder.conv_im.weight.zero_()
            if self.encoder.bias_re is not None:
                self.encoder.bias_re.zero_()
                self.encoder.bias_im.zero_()
            center = self.encoder.conv_re.weight.size(-1) // 2
            self.encoder.conv_re.weight[0, 0, center] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected (B,2,L), got {tuple(x.shape)}")
        return self.encoder(x.unsqueeze(2))


class DualPathBlock(nn.Module):
    """Dual-path GRU block over chunk-local and inter-chunk axes."""

    def __init__(
        self,
        hidden_dim: int,
        rnn_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.intra_norm = nn.LayerNorm(hidden_dim)
        self.intra_rnn = nn.GRU(
            hidden_dim,
            rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.intra_proj = nn.Linear(2 * rnn_hidden, hidden_dim)
        self.inter_norm = nn.LayerNorm(hidden_dim)
        self.inter_rnn = nn.GRU(
            hidden_dim,
            rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.inter_proj = nn.Linear(2 * rnn_hidden, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, chunks: torch.Tensor) -> torch.Tensor:
        if chunks.ndim != 4:
            raise ValueError(f"Expected (B,N,R,H), got {tuple(chunks.shape)}")
        batch, n_chunks, chunk_size, hidden = chunks.shape

        residual = chunks
        intra = self.intra_norm(chunks).reshape(batch * n_chunks, chunk_size, hidden)
        intra, _ = self.intra_rnn(intra)
        intra = self.intra_proj(intra).reshape(batch, n_chunks, chunk_size, hidden)
        chunks = residual + self.dropout(intra)

        residual = chunks
        inter = self.inter_norm(chunks).permute(0, 2, 1, 3).reshape(batch * chunk_size, n_chunks, hidden)
        inter, _ = self.inter_rnn(inter)
        inter = self.inter_proj(inter).reshape(batch, chunk_size, n_chunks, hidden).permute(0, 2, 1, 3)
        return residual + self.dropout(inter)


class DualPathSeparator(nn.Module):
    """Overlap-chunk dual-path separator that returns full-length mask logits."""

    def __init__(
        self,
        in_channels: int,
        mask_channels: int,
        hidden_dim: int = 128,
        rnn_hidden: int = 128,
        n_layers: int = 6,
        chunk_size: int = 128,
        hop_size: int = 64,
        dropout: float = 0.0,
        mask_head_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.hop_size = int(hop_size)
        if self.chunk_size < 2:
            raise ValueError("chunk_size must be >= 2")
        if self.hop_size < 1 or self.hop_size > self.chunk_size:
            raise ValueError("hop_size must be in [1, chunk_size]")
        self.input_norm = nn.GroupNorm(1, int(in_channels))
        self.input_proj = nn.Conv1d(int(in_channels), int(hidden_dim), 1)
        self.blocks = nn.ModuleList(
            [
                DualPathBlock(
                    hidden_dim=int(hidden_dim),
                    rnn_hidden=int(rnn_hidden),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.mask_head = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(int(hidden_dim), int(mask_channels), 1),
        )
        if mask_head_zero_init:
            self._init_zero_mask_head()

    def _init_zero_mask_head(self) -> None:
        final_conv = self.mask_head[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def _pad_to_chunks(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        length = x.size(-1)
        if length <= self.chunk_size:
            pad_right = self.chunk_size - length
        else:
            remainder = (length - self.chunk_size) % self.hop_size
            pad_right = 0 if remainder == 0 else self.hop_size - remainder
        if pad_right > 0:
            x = F.pad(x, (0, pad_right))
        return x, pad_right

    def _chunk(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        x, pad_right = self._pad_to_chunks(x)
        chunks = x.unfold(dimension=2, size=self.chunk_size, step=self.hop_size)
        chunks = chunks.permute(0, 2, 3, 1).contiguous()
        return chunks, pad_right

    def _overlap_add(self, chunks: torch.Tensor, output_length: int) -> torch.Tensor:
        batch, n_chunks, chunk_size, hidden = chunks.shape
        total_length = (n_chunks - 1) * self.hop_size + chunk_size
        y = chunks.new_zeros(batch, hidden, total_length)
        denom = chunks.new_zeros(batch, 1, total_length)
        chunks = chunks.permute(0, 3, 1, 2).contiguous()
        for idx in range(n_chunks):
            start = idx * self.hop_size
            end = start + chunk_size
            y[:, :, start:end] = y[:, :, start:end] + chunks[:, :, idx, :]
            denom[:, :, start:end] = denom[:, :, start:end] + 1.0
        y = y / denom.clamp_min(1.0)
        return y[:, :, :output_length]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,L), got {tuple(x.shape)}")
        output_length = x.size(-1)
        x = self.input_proj(self.input_norm(x))
        chunks, _ = self._chunk(x)
        for block in self.blocks:
            chunks = block(chunks)
        y = self._overlap_add(chunks, output_length=output_length)
        return self.mask_head(y)


class ComplexMaskDecoder(nn.Module):
    """Apply complex masks to learned complex features and synthesize sources."""

    def __init__(
        self,
        n_srcs: int,
        feature_channels: int,
        kernel_size: int = 9,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.feature_channels = int(feature_channels)
        self.mask_bound = float(mask_bound)
        self.mask_sum_constraint = bool(mask_sum_constraint)
        self.decoder = ComplexConv1d(
            self.feature_channels,
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

    def _complex_masks(self, mask_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_logits.ndim != 3:
            raise ValueError(f"Expected (B,2*K*C,L), got {tuple(mask_logits.shape)}")
        expected = 2 * self.n_srcs * self.feature_channels
        if mask_logits.size(1) != expected:
            raise ValueError(f"Expected {expected} mask channels, got {mask_logits.size(1)}")
        batch, _, length = mask_logits.shape
        masks = mask_logits.reshape(batch, self.n_srcs, 2, self.feature_channels, length)
        if self.mask_bound > 0:
            masks = torch.tanh(masks) * self.mask_bound
        mask_real = masks[:, :, 0, :, :]
        mask_imag = masks[:, :, 1, :, :]
        if self.mask_sum_constraint:
            sum_real = mask_real.sum(dim=1, keepdim=True)
            sum_imag = mask_imag.sum(dim=1, keepdim=True)
            mask_real = mask_real - (sum_real - 1.0) / self.n_srcs
            mask_imag = mask_imag - sum_imag / self.n_srcs
        complex_masks = (mask_real, mask_imag)
        return complex_masks

    def forward(self, features: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.size(1) != 2:
            raise ValueError(f"Expected complex features (B,2,C,L), got {tuple(features.shape)}")
        batch, _, channels, length = features.shape
        if channels != self.feature_channels:
            raise ValueError(f"Expected {self.feature_channels} feature channels, got {channels}")
        if mask_logits.size(-1) != length:
            mask_logits = F.interpolate(mask_logits, size=length, mode="linear", align_corners=False)

        mask_real, mask_imag = self._complex_masks(mask_logits)
        feat_real = features[:, 0, :, :].unsqueeze(1)
        feat_imag = features[:, 1, :, :].unsqueeze(1)
        source_real = mask_real * feat_real - mask_imag * feat_imag
        source_imag = mask_real * feat_imag + mask_imag * feat_real
        source_features = torch.stack([source_real, source_imag], dim=2)
        source_features = source_features.reshape(batch * self.n_srcs, 2, self.feature_channels, length)
        decoded = self.decoder(source_features).squeeze(2)
        decoded = decoded.reshape(batch, self.n_srcs, 2, length)
        return decoded.reshape(batch, 2 * self.n_srcs, length)


class ComplexDPNetSeparator1D(nn.Module):
    """Complex learned-mask dual-path separator with IQUMamba-compatible I/O."""

    def __init__(
        self,
        n_srcs: int = 2,
        feature_channels: int = 64,
        kernel_size: int = 9,
        hidden_dim: int = 128,
        rnn_hidden: int = 128,
        n_layers: int = 6,
        chunk_size: int = 128,
        hop_size: int = 64,
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
        self.separator = DualPathSeparator(
            in_channels=2 * self.feature_channels,
            mask_channels=2 * self.n_srcs * self.feature_channels,
            hidden_dim=int(hidden_dim),
            rnn_hidden=int(rnn_hidden),
            n_layers=int(n_layers),
            chunk_size=int(chunk_size),
            hop_size=int(hop_size),
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
