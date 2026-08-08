"""RF-BandSCNet: band-split complex spectral separator for IQ BSS.

This is a non-IQUMamba framework-level baseline.  It follows the recent
separation pattern used by TF-GridNet / BSRNN-style systems:

    complex IQ STFT -> band-split tokens -> temporal and cross-band modeling
    -> complex ratio masks -> complex iSTFT -> waveform mixture consistency.

The model uses only the observed mixture.  It does not consume modulation
labels, symbol timing, cyclic-frequency metadata, or source-side information.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


def _make_band_splits(n_freq: int, n_bands: int) -> List[Tuple[int, int]]:
    if n_freq < 1:
        raise ValueError(f"n_freq must be positive, got {n_freq}")
    n_bands = max(1, min(int(n_bands), int(n_freq)))
    base = n_freq // n_bands
    rem = n_freq % n_bands
    bands = []
    start = 0
    for band in range(n_bands):
        width = base + (1 if band < rem else 0)
        bands.append((start, start + width))
        start += width
    return bands


class BandSplitSeparatorBlock(nn.Module):
    """Dual-path block over time within each band and across frequency bands."""

    def __init__(
        self,
        hidden_dim: int,
        rnn_hidden: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.time_rnn = nn.GRU(
            hidden_dim,
            rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.time_proj = nn.Linear(2 * rnn_hidden, hidden_dim)
        self.band_norm = nn.LayerNorm(hidden_dim)
        self.band_rnn = nn.GRU(
            hidden_dim,
            rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.band_proj = nn.Linear(2 * rnn_hidden, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,T,K,H), got {tuple(x.shape)}")
        batch, frames, bands, hidden = x.shape

        residual = x
        time_in = self.time_norm(x).permute(0, 2, 1, 3).reshape(batch * bands, frames, hidden)
        time_out, _ = self.time_rnn(time_in)
        time_out = self.time_proj(time_out)
        time_out = time_out.reshape(batch, bands, frames, hidden).permute(0, 2, 1, 3)
        x = residual + self.dropout(time_out)

        residual = x
        band_in = self.band_norm(x).reshape(batch * frames, bands, hidden)
        band_out, _ = self.band_rnn(band_in)
        band_out = self.band_proj(band_out)
        band_out = band_out.reshape(batch, frames, bands, hidden)
        return residual + self.dropout(band_out)


class RFBandSCNetCore(nn.Module):
    """Band-split complex mask estimator operating on complex STFT frames."""

    def __init__(
        self,
        n_srcs: int = 2,
        n_freq: int = 256,
        n_bands: int = 16,
        hidden_dim: int = 96,
        rnn_hidden: int = 96,
        n_layers: int = 6,
        dropout: float = 0.0,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_head_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.hidden_dim = int(hidden_dim)
        self.mask_bound = float(mask_bound)
        self.mask_sum_constraint = bool(mask_sum_constraint)
        self.band_splits = _make_band_splits(self.n_freq, self.n_bands)

        self.band_encoders = nn.ModuleList(
            [nn.Linear(2 * (end - start), self.hidden_dim) for start, end in self.band_splits]
        )
        self.blocks = nn.ModuleList(
            [
                BandSplitSeparatorBlock(
                    hidden_dim=self.hidden_dim,
                    rnn_hidden=int(rnn_hidden),
                    dropout=float(dropout),
                )
                for _ in range(int(n_layers))
            ]
        )
        self.band_mask_heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.n_srcs * 2 * (end - start))
                for start, end in self.band_splits
            ]
        )
        if mask_head_zero_init:
            self._init_zero_mask_heads()

    def _init_zero_mask_heads(self) -> None:
        for head in self.band_mask_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _encode_bands(self, mix_spec: torch.Tensor) -> torch.Tensor:
        feature = torch.stack([mix_spec.real, mix_spec.imag], dim=-1)
        tokens = []
        for (start, end), encoder in zip(self.band_splits, self.band_encoders):
            band = feature[:, :, start:end, :].reshape(feature.size(0), feature.size(1), -1)
            tokens.append(encoder(band))
        return torch.stack(tokens, dim=2)

    def _predict_complex_masks(self, tokens: torch.Tensor, n_frames: int, n_freq: int) -> torch.Tensor:
        mask_real = tokens.new_zeros(tokens.size(0), self.n_srcs, n_frames, n_freq)
        mask_imag = tokens.new_zeros(tokens.size(0), self.n_srcs, n_frames, n_freq)

        for band_index, ((start, end), head) in enumerate(zip(self.band_splits, self.band_mask_heads)):
            width = end - start
            logits = head(tokens[:, :, band_index, :])
            logits = logits.reshape(tokens.size(0), n_frames, self.n_srcs, 2, width)
            if self.mask_bound > 0:
                logits = torch.tanh(logits) * self.mask_bound
            mask_real[:, :, :, start:end] = logits[:, :, :, 0, :].permute(0, 2, 1, 3)
            mask_imag[:, :, :, start:end] = logits[:, :, :, 1, :].permute(0, 2, 1, 3)

        if self.mask_sum_constraint:
            sum_real = mask_real.sum(dim=1, keepdim=True)
            sum_imag = mask_imag.sum(dim=1, keepdim=True)
            mask_real = mask_real - (sum_real - 1.0) / self.n_srcs
            mask_imag = mask_imag - sum_imag / self.n_srcs

        complex_masks = torch.complex(mask_real, mask_imag)
        return complex_masks

    def forward(self, mix_spec: torch.Tensor) -> List[torch.Tensor]:
        if mix_spec.ndim != 3 or not torch.is_complex(mix_spec):
            raise ValueError(f"Expected complex (B,T,F) STFT, got {tuple(mix_spec.shape)}")
        batch, n_frames, n_freq = mix_spec.shape
        if n_freq != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequency bins, got {n_freq}")

        tokens = self._encode_bands(mix_spec)
        for block in self.blocks:
            tokens = block(tokens)

        complex_masks = self._predict_complex_masks(tokens, n_frames=n_frames, n_freq=n_freq)
        separated = complex_masks * mix_spec.unsqueeze(1)
        return [separated[:, source] for source in range(self.n_srcs)]


class RFBandSCNetSeparator1D(nn.Module):
    """Complex spectral band-split separator for input/output shaped like IQUMamba."""

    def __init__(
        self,
        n_srcs: int = 2,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        center: bool = True,
        normalize_input: bool = True,
        eps: float = 1e-8,
        n_bands: int = 16,
        hidden_dim: int = 96,
        rnn_hidden: int = 96,
        n_layers: int = 6,
        dropout: float = 0.0,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_head_zero_init: bool = True,
        apply_projection: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_detach_weights: bool = False,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.normalize_input = bool(normalize_input)
        self.eps = float(eps)
        self.apply_projection = bool(apply_projection)
        self.core = RFBandSCNetCore(
            n_srcs=self.n_srcs,
            n_freq=self.n_fft,
            n_bands=int(n_bands),
            hidden_dim=int(hidden_dim),
            rnn_hidden=int(rnn_hidden),
            n_layers=int(n_layers),
            dropout=float(dropout),
            mask_bound=float(mask_bound),
            mask_sum_constraint=bool(mask_sum_constraint),
            mask_head_zero_init=bool(mask_head_zero_init),
        )
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.n_srcs,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=self.eps,
            detach_weights=mc_detach_weights,
        )
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected input (B,2,L), got {tuple(x.shape)}")
        batch_size, _, length = x.shape
        original_dtype = x.dtype
        if original_dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        mix_complex = torch.complex(x[:, 0], x[:, 1])
        if self.normalize_input:
            scale = mix_complex.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
            mix_complex = mix_complex / scale
        else:
            scale = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype)

        window = self.window.to(device=x.device, dtype=x.dtype)
        mix_spec = torch.stft(
            mix_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            onesided=False,
            return_complex=True,
        )
        mix_spec = mix_spec.transpose(1, 2).contiguous()

        separated_specs = self.core(mix_spec)
        reconstructed = []
        for separated_spec in separated_specs:
            separated_spec = separated_spec.transpose(1, 2).contiguous()
            signal = torch.istft(
                separated_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                onesided=False,
                length=length,
                return_complex=True,
            )
            reconstructed.append(signal * scale)

        output = torch.stack(
            [torch.stack([signal.real, signal.imag], dim=1) for signal in reconstructed],
            dim=1,
        ).reshape(batch_size, 2 * self.n_srcs, length)
        output = output.to(dtype=original_dtype)
        if self.apply_projection:
            output = self.mc_projection(output, x.to(dtype=output.dtype))
        return output
