"""DualDomainBandSplit — Band-Split Mamba for the Frequency Tower.

Innovation: replace V1's flatten-everything Mamba (d_model ≈ 2000+) with a
structured Band-Split architecture inspired by BSRNN (ICASSP 2023) and
BS-RoFormer (SDC 2023 winner).

Key differences from BSRNN:
  - Uses Mamba (SSM) instead of BLSTM for both intra-band and inter-band axes
  - Integrated into a dual-domain framework with time-domain UNet encoder
  - Selective cross-attention fusion (only at deep encoder stages)
  - Per-channel sigmoid gate with conservative init

The Band-Split approach provides:
  1. Physics-aligned processing: RF signals occupy distinct frequency bands,
     so per-band modeling is a natural inductive bias.
  2. Dramatic parameter reduction: hidden_dim=128, K=8 bands vs d_model=2064.
  3. Explicit intra-band (temporal) and inter-band (spectral) separation of
     concerns, allowing each Mamba to focus on one axis.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from mamba_ssm import Mamba
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUMamba1D import (
    UpsampleLayer,
    MambaLayer,
    BasicResBlock,
    SkipConnectionProcessor,
    ChannelAttention1D,
    AdaptiveFusion1D,
    UNetResDecoder,
)

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  Helpers
# ============================================================================

def _compute_stft_output_size(input_length: int, n_fft: int, hop_length: int,
                              center: bool = True) -> Tuple[int, int]:
    n_freq = n_fft // 2 + 1
    if center:
        n_time = input_length // hop_length + 1
    else:
        n_time = (input_length - n_fft) // hop_length + 1
    return n_freq, n_time


def _compute_freq_sizes_per_stage(n_freq: int, n_stages: int) -> List[int]:
    sizes = []
    f = n_freq
    for s in range(n_stages):
        if s > 0:
            f = math.ceil(f / 2)
        sizes.append(f)
    return sizes


def _compute_band_splits(n_freq: int, n_bands: int) -> List[Tuple[int, int]]:
    """Compute non-overlapping band boundaries for n_freq bins into n_bands.

    Returns list of (start, end) tuples (end exclusive).
    Bands are as equal-sized as possible.
    """
    base_size = n_freq // n_bands
    remainder = n_freq % n_bands
    bands = []
    start = 0
    for i in range(n_bands):
        size = base_size + (1 if i < remainder else 0)
        bands.append((start, start + size))
        start += size
    return bands


# ============================================================================
#  STFT Frontend
# ============================================================================

class STFTFrontend(nn.Module):
    """Compute complex STFT, output 4 real channels (real_I, imag_I, real_Q, imag_Q)."""

    def __init__(self, n_fft: int = 256, hop_length: int = 64,
                 win_length: int = 256, center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=True)
        )

    def forward(self, x):
        B, C, L = x.shape
        parts = []
        for ch in range(C):
            s = torch.stft(
                x[:, ch, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                return_complex=True,
            )
            parts.append(s.real)
            parts.append(s.imag)
        spec = torch.stack(parts, dim=1)
        return spec


# ============================================================================
#  Band-Split Module
# ============================================================================

class BandSplitModule(nn.Module):
    """Split STFT spectrogram into sub-bands and project each to hidden_dim.

    Input:  (B, C, F, T)  — C=4 (real/imag × I/Q), F=freq bins, T=STFT frames
    Output: (B, T, K, H)  — K=num_bands, H=hidden_dim
    """

    def __init__(self, in_channels: int, n_freq: int, n_bands: int,
                 hidden_dim: int):
        super().__init__()
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim

        bands = _compute_band_splits(n_freq, n_bands)
        self.band_boundaries = bands

        # One FC layer per band: maps (C × band_width) → hidden_dim
        self.band_projections = nn.ModuleList()
        for start, end in bands:
            band_width = end - start
            self.band_projections.append(
                nn.Sequential(
                    nn.Linear(in_channels * band_width, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            )

    def forward(self, x):
        """
        Args:
            x: (B, C, F, T)
        Returns:
            (B, T, K, H)
        """
        B, C, F, T = x.shape

        # Rearrange to (B, T, C, F) for easier band slicing
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)

        band_features = []
        for k, (start, end) in enumerate(self.band_boundaries):
            # Extract band: (B, T, C, band_width)
            band = x[:, :, :, start:end]
            # Flatten channels × band_width: (B, T, C*band_width)
            band_flat = band.reshape(B, T, -1)
            # Project to hidden_dim: (B, T, H)
            band_proj = self.band_projections[k](band_flat)
            band_features.append(band_proj)

        # Stack all bands: (B, T, K, H)
        return torch.stack(band_features, dim=2)


# ============================================================================
#  Band Merge Module
# ============================================================================

class BandMergeModule(nn.Module):
    """Merge band features back to full spectrogram shape.

    Input:  (B, T, K, H)
    Output: (B, C_out, F, T)
    """

    def __init__(self, out_channels: int, n_freq: int, n_bands: int,
                 hidden_dim: int):
        super().__init__()
        self.out_channels = out_channels

        bands = _compute_band_splits(n_freq, n_bands)
        self.band_boundaries = bands

        # One FC layer per band: maps hidden_dim → (C_out × band_width)
        self.band_back_projections = nn.ModuleList()
        for start, end in bands:
            band_width = end - start
            self.band_back_projections.append(
                nn.Linear(hidden_dim, out_channels * band_width)
            )

    def forward(self, band_features, B, T, F):
        """
        Args:
            band_features: (B, T, K, H)
        Returns:
            (B, C_out, F, T)
        """
        bands_reconstructed = []
        for k, (start, end) in enumerate(self.band_boundaries):
            band_width = end - start
            # (B, T, H) → (B, T, C_out * band_width)
            out = self.band_back_projections[k](band_features[:, :, k, :])
            # → (B, T, C_out, band_width)
            out = out.reshape(B, T, self.out_channels, band_width)
            bands_reconstructed.append(out)

        # Concatenate along freq axis: (B, T, C_out, F)
        full_spec = torch.cat(bands_reconstructed, dim=3)
        # → (B, C_out, F, T)
        return full_spec.permute(0, 2, 3, 1)


# ============================================================================
#  Band-Split Mamba Block (Core Innovation)
# ============================================================================

class BandSplitMambaBlock(nn.Module):
    """One round of intra-band (temporal) + inter-band (spectral) Mamba.

    This is the core building block, applied N_layers times.

    Step 1 — Intra-band Mamba:
        For each of K bands independently, run Mamba along the T (time) axis.
        This captures temporal patterns within each frequency band.

    Step 2 — Inter-band Mamba:
        For each of T time steps independently, run Mamba along the K (band) axis.
        This captures cross-band spectral patterns.

    Both steps use residual connections and LayerNorm.
    """

    def __init__(self, hidden_dim: int, n_bands: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        # Intra-band: Mamba along time axis (shared across all bands)
        self.intra_norm = nn.LayerNorm(hidden_dim)
        self.intra_mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Inter-band: Mamba along band axis (shared across all time steps)
        self.inter_norm = nn.LayerNorm(hidden_dim)
        self.inter_mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    @autocast('cuda', enabled=False)
    def forward(self, x):
        """
        Args:
            x: (B, T, K, H)
        Returns:
            (B, T, K, H)
        """
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        B, T, K, H = x.shape

        # --- Step 1: Intra-band (temporal) Mamba ---
        # Reshape to treat each band as a separate batch item: (B*K, T, H)
        x_intra = x.permute(0, 2, 1, 3).reshape(B * K, T, H)
        x_intra = x_intra + self.intra_mamba(self.intra_norm(x_intra))
        x = x_intra.reshape(B, K, T, H).permute(0, 2, 1, 3)  # (B, T, K, H)

        # --- Step 2: Inter-band (spectral) Mamba ---
        # Reshape to treat each time step as a separate batch item: (B*T, K, H)
        x_inter = x.reshape(B * T, K, H)
        x_inter = x_inter + self.inter_mamba(self.inter_norm(x_inter))
        x = x_inter.reshape(B, T, K, H)

        return x


# ============================================================================
#  Band-Split Frequency Encoder
# ============================================================================

class BandSplitFreqEncoder(nn.Module):
    """Multi-stage frequency-domain encoder using Band-Split Mamba.

    Each stage:
      1. Optional 2D convolution for channel expansion + freq downsampling
      2. Band-Split → N_layers × BandSplitMambaBlock → Band-Merge
    """

    def __init__(self, n_stages: int, features_per_stage: List[int],
                 input_length: int,
                 freq_features_per_stage: List[int] = None,
                 n_bands: int = 8,
                 hidden_dim: int = 128,
                 n_band_mamba_layers: int = 2,
                 n_fft: int = 256, hop_length: int = 64, win_length: int = 256,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.stft = STFTFrontend(n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length)
        self.n_stages = n_stages
        self.n_fft = n_fft

        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        n_freq, _n_time = _compute_stft_output_size(input_length, n_fft, hop_length)
        freq_sizes = _compute_freq_sizes_per_stage(n_freq, n_stages)

        in_ch = 4  # STFT gives 4 channels
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            out_ch = freq_features_per_stage[s]
            freq_bins = freq_sizes[s]

            # Compute number of bands for this stage
            # Use fewer bands when freq bins are small
            stage_n_bands = min(n_bands, freq_bins)

            stage_module = BandSplitFreqStage(
                in_channels=in_ch,
                out_channels=out_ch,
                freq_bins=freq_bins,
                n_bands=stage_n_bands,
                hidden_dim=hidden_dim,
                n_mamba_layers=n_band_mamba_layers,
                stride_freq=2 if s > 0 else 1,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.stages.append(stage_module)
            in_ch = out_ch

        self.freq_features_per_stage = freq_features_per_stage
        self.freq_sizes = freq_sizes

    def forward(self, x):
        spec = self.stft(x)
        feats = []
        h = spec
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


class BandSplitFreqStage(nn.Module):
    """Single frequency encoder stage: Conv2D + BandSplit-Mamba."""

    def __init__(self, in_channels: int, out_channels: int,
                 freq_bins: int, n_bands: int, hidden_dim: int,
                 n_mamba_layers: int = 2,
                 stride_freq: int = 1,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        # 2D convolution for channel expansion and freq downsampling
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1,
                      stride=(stride_freq, 1)),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        self.skip_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=(stride_freq, 1)
        ) if (in_channels != out_channels or stride_freq != 1) else nn.Identity()

        # Band-Split Mamba processing
        self.band_split = BandSplitModule(
            in_channels=out_channels,
            n_freq=freq_bins,
            n_bands=n_bands,
            hidden_dim=hidden_dim,
        )

        self.mamba_layers = nn.ModuleList([
            BandSplitMambaBlock(
                hidden_dim=hidden_dim,
                n_bands=n_bands,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_mamba_layers)
        ])

        self.band_merge = BandMergeModule(
            out_channels=out_channels,
            n_freq=freq_bins,
            n_bands=n_bands,
            hidden_dim=hidden_dim,
        )

    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        # Conv2D + residual
        identity = self.skip_proj(x)
        h = self.conv2d(x) + identity

        B, C, F, T = h.shape

        # Band-Split → Mamba layers → Band-Merge
        band_features = self.band_split(h)     # (B, T, K, H)
        for mamba_layer in self.mamba_layers:
            band_features = mamba_layer(band_features)
        h_mamba = self.band_merge(band_features, B, T, F)  # (B, C, F, T)

        return h + h_mamba  # residual


# ============================================================================
#  Cross-Attention Fusion (per-channel gate, conservative init)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with per-channel sigmoid gate."""

    def __init__(self, time_dim: int, freq_channels: int, freq_bins: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.time_dim = time_dim

        freq_flat_dim = freq_channels * freq_bins
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_flat_dim, time_dim),
            nn.LayerNorm(time_dim),
        )

        self.cross_attn_t2f = nn.MultiheadAttention(
            embed_dim=time_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_t = nn.LayerNorm(time_dim)
        self.norm_f_for_t = nn.LayerNorm(time_dim)

        self.cross_attn_f2t = nn.MultiheadAttention(
            embed_dim=time_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_f = nn.LayerNorm(time_dim)
        self.norm_t_for_f = nn.LayerNorm(time_dim)

        self.freq_back_proj = nn.Linear(time_dim, freq_flat_dim)

        # Per-channel sigmoid gate, conservative init sigmoid(-4) ≈ 0.018
        self.gate_time = nn.Parameter(torch.full((1, 1, time_dim), -4.0))
        self.gate_freq = nn.Parameter(torch.full((1, 1, time_dim), -4.0))

    @autocast('cuda', enabled=False)
    def forward(self, x_time, x_freq):
        if x_time.dtype in HALF_PRECISION_DTYPES:
            x_time = x_time.float()
        if x_freq.dtype in HALF_PRECISION_DTYPES:
            x_freq = x_freq.float()

        B, D_t, L_t = x_time.shape
        B, C_f, F_f, T_f = x_freq.shape

        time_tokens = x_time.transpose(1, 2)
        freq_flat = x_freq.permute(0, 3, 1, 2).reshape(B, T_f, C_f * F_f)
        freq_tokens = self.freq_proj(freq_flat)

        # Time reads from freq
        time_normed = self.norm_t(time_tokens)
        freq_normed_for_t = self.norm_f_for_t(freq_tokens)
        attn_t2f, _ = self.cross_attn_t2f(
            query=time_normed, key=freq_normed_for_t, value=freq_normed_for_t,
        )
        time_fused = time_tokens + torch.sigmoid(self.gate_time) * attn_t2f

        # Freq reads from time
        freq_normed = self.norm_f(freq_tokens)
        time_normed_for_f = self.norm_t_for_f(time_tokens)
        attn_f2t, _ = self.cross_attn_f2t(
            query=freq_normed, key=time_normed_for_f, value=time_normed_for_f,
        )
        freq_fused = freq_tokens + torch.sigmoid(self.gate_freq) * attn_f2t

        x_time_fused = time_fused.transpose(1, 2)
        freq_back = self.freq_back_proj(freq_fused)
        x_freq_fused = freq_back.reshape(B, T_f, C_f, F_f).permute(0, 2, 3, 1)
        x_freq_fused = x_freq_fused + x_freq

        return x_time_fused, x_freq_fused


class IdentityFusion(nn.Module):
    """No-op fusion for shallow stages."""
    def forward(self, x_time, x_freq):
        return x_time, x_freq


# ============================================================================
#  Dual-Domain Encoder with Band-Split Frequency Tower
# ============================================================================

class DualDomainBandSplitEncoder(nn.Module):
    """Time-domain UNet encoder + Band-Split frequency tower."""

    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 # Frequency tower settings
                 n_fft: int = 256,
                 hop_length: int = 64,
                 win_length: int = 256,
                 freq_features_per_stage: List[int] = None,
                 cross_attn_heads: int = 4,
                 cross_attn_dropout: float = 0.1,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 # Band-Split specific
                 n_bands: int = 8,
                 hidden_dim: int = 128,
                 n_band_mamba_layers: int = 2,
                 fusion_start_stage: int = 2,
                 ):
        super().__init__()
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i / j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]

        # ---- Time-Domain Stem ----
        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[BasicBlockD(
                conv_op=conv_op,
                input_channels=stem_channels,
                output_channels=stem_channels,
                kernel_size=kernel_sizes[0],
                stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        # ---- Time-Domain Conv + Mamba ----
        input_ch = stem_channels
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_ch,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ),
                *[BasicBlockD(
                    conv_op=conv_op,
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )

            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    MambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s],
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_ch = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)

        # ---- Band-Split Frequency Tower ----
        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        input_length = int(input_size[0])

        self.freq_encoder = BandSplitFreqEncoder(
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            input_length=input_length,
            freq_features_per_stage=freq_features_per_stage,
            n_bands=n_bands,
            hidden_dim=hidden_dim,
            n_band_mamba_layers=n_band_mamba_layers,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # ---- Selective Cross-Attention Fusion ----
        freq_sizes = self.freq_encoder.freq_sizes
        cross_fusions = []
        for s in range(n_stages):
            if s >= fusion_start_stage:
                cross_fusions.append(
                    CrossAttentionFusion(
                        time_dim=features_per_stage[s],
                        freq_channels=freq_features_per_stage[s],
                        freq_bins=freq_sizes[s],
                        n_heads=cross_attn_heads,
                        dropout=cross_attn_dropout,
                    )
                )
            else:
                cross_fusions.append(IdentityFusion())
        self.cross_fusions = nn.ModuleList(cross_fusions)

        # Store metadata for decoder
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        freq_feats = self.freq_encoder(x)

        if self.stem is not None:
            x = self.stem(x)

        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            x, freq_feats[s] = self.cross_fusions[s](x, freq_feats[s])
            ret.append(x)

        return ret if self.return_skips else ret[-1]


# ============================================================================
#  Top-level model
# ============================================================================

class DualDomainBandSplit(nn.Module):
    """Dual-Domain Band-Split Mamba for IQ signal separation.

    Combines:
      - Time-domain UNet encoder with Mamba (proven backbone)
      - Band-Split Mamba frequency tower (BSRNN-inspired innovation)
      - Selective cross-attention fusion with per-channel gates (V4 stability)
    """

    def __init__(self,
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
                 norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
                 nonlin: Type[nn.Module] = nn.LeakyReLU,
                 nonlin_kwargs: dict = {'inplace': True},
                 deep_supervision: bool = False,
                 # Dual-domain specific
                 n_fft: int = 256,
                 hop_length: int = 64,
                 win_length: int = 256,
                 freq_features_per_stage: List[int] = None,
                 cross_attn_heads: int = 4,
                 cross_attn_dropout: float = 0.1,
                 # Band-Split specific
                 n_bands: int = 8,
                 hidden_dim: int = 128,
                 n_band_mamba_layers: int = 2,
                 fusion_start_stage: int = 2,
                 ):
        super().__init__()
        self.encoder = DualDomainBandSplitEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            freq_features_per_stage=freq_features_per_stage,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dropout=cross_attn_dropout,
            n_bands=n_bands,
            hidden_dim=hidden_dim,
            n_band_mamba_layers=n_band_mamba_layers,
            fusion_start_stage=fusion_start_stage,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
