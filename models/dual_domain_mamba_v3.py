"""DualDomainMambaV3 — Dual-Path Mamba for IQ signal separation.

Key innovations over V1/V2
==========================

1. **Dual-Path Mamba in Frequency Tower (replacing flatten/per-bin Mamba)**
   Inspired by TF-GridNet (Wang et al., 2023) and SPMamba (Li et al., 2024),
   the frequency tower processes features along TWO axes alternately:
     - *Intra-frame Mamba*: runs along the **Frequency axis** (d_model=C)
       to capture cross-subband dependencies (harmonic structure, FDM spacing).
     - *Inter-frame Mamba*: runs along the **Time axis** (d_model=C)
       to capture temporal evolution of spectral patterns.
   This keeps d_model = out_channels (small), avoids the flatten-Mamba
   parameter explosion, AND preserves the frequency-axis topology that
   flatten destroys.

2. **Positional Encoding in Cross-Attention**
   Learnable position embeddings are added to both time-domain tokens and
   frequency-domain tokens before cross-attention, giving the model an
   explicit "which time position corresponds to which STFT frame" prior.

3. **Zero-Init Gating (retained from V2)**
   The fusion gate starts at 0 → no noise injection at epoch 0.

4. **Unidirectional Cross-Attention (retained from V2)**
   Time reads from Freq; Freq tower is a read-only reference.

5. **Frequency Feature Injection into Decoder**
   Frequency tower features are projected and injected into the decoder's
   skip connections at each upsampling stage, so spectral information
   participates in waveform reconstruction — no longer "compute and discard".
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

# Re-use building blocks from the original IQUMamba1D
from models.IQUMamba1D import (
    UpsampleLayer,
    MambaLayer,
    BasicResBlock,
    SkipConnectionProcessor,
    ChannelAttention1D,
    AdaptiveFusion1D,
)

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  Helpers — compute STFT output sizes eagerly at __init__ time
# ============================================================================

def _compute_stft_output_size(input_length: int, n_fft: int, hop_length: int,
                              center: bool = True) -> Tuple[int, int]:
    """Return (n_freq_bins, n_time_frames) for the given STFT parameters."""
    n_freq = n_fft // 2 + 1
    if center:
        n_time = input_length // hop_length + 1
    else:
        n_time = (input_length - n_fft) // hop_length + 1
    return n_freq, n_time


def _compute_freq_sizes_per_stage(n_freq: int, n_stages: int) -> List[int]:
    """Compute the frequency-axis size after each stage.

    Stage 0 keeps freq unchanged (stride_freq=1); stages 1+ halve it
    via stride_freq=2 (Conv2d kernel=3, padding=1 → out = ceil(in/2)).
    """
    sizes = []
    f = n_freq
    for s in range(n_stages):
        if s > 0:
            f = math.ceil(f / 2)
        sizes.append(f)
    return sizes


# ============================================================================
#  STFT Frontend — convert raw IQ waveform to time-frequency representation
# ============================================================================

class STFTFrontend(nn.Module):
    """Compute complex STFT and return a real-valued tensor for the freq tower.

    Output channels = 4  (real_I, imag_I, real_Q, imag_Q).
    """

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
        """
        Args:
            x: (B, 2, L) — raw IQ waveform  [I channel, Q channel]
        Returns:
            spec: (B, 4, F, T_stft) — stacked real/imag for both I and Q
        """
        B, C, L = x.shape
        parts = []
        for ch in range(C):  # I and Q
            s = torch.stft(
                x[:, ch, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                return_complex=True,
            )  # (B, F, T_stft) complex
            parts.append(s.real)
            parts.append(s.imag)
        spec = torch.stack(parts, dim=1)  # (B, 4, F, T_stft)
        return spec


# ============================================================================
#  Dual-Path Mamba Block — Intra (freq-axis) + Inter (time-axis)
# ============================================================================

class DualPathMambaBlock(nn.Module):
    """Dual-path processing: Intra-frame Mamba (freq axis) + Inter-frame Mamba (time axis).

    Inspired by TF-GridNet's intra/inter BLSTM design, but uses Mamba (SSM)
    for O(L) complexity instead of O(L²).

    Unlike the original FreqConvBlock which flattens (C, F) into one giant
    dimension, here d_model = C (channels only), keeping the frequency-axis
    topology intact and params at O(C²).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        # Intra-frame: Mamba runs along F axis (captures cross-frequency patterns)
        self.norm_intra = nn.LayerNorm(d_model)
        self.mamba_intra = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )

        # Inter-frame: Mamba runs along T axis (captures temporal evolution)
        self.norm_inter = nn.LayerNorm(d_model)
        self.mamba_inter = Mamba(
            d_model=d_model, d_state=d_state,
            d_conv=d_conv, expand=expand,
        )

    @autocast('cuda', enabled=False)
    def forward(self, x):
        """
        Args:
            x: (B, C, F, T)
        Returns:
            out: (B, C, F, T)  — same shape, residually enriched
        """
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        B, C, Fr, T = x.shape

        # --- 1. Intra-frame: reshape to (B*T, F, C), Mamba along F ---
        x_intra = x.permute(0, 3, 2, 1).reshape(B * T, Fr, C)  # (B*T, F, C)
        x_intra = x_intra + self.mamba_intra(self.norm_intra(x_intra))
        x_intra = x_intra.reshape(B, T, Fr, C).permute(0, 3, 2, 1)  # (B, C, F, T)

        # --- 2. Inter-frame: reshape to (B*F, T, C), Mamba along T ---
        x_inter = x_intra.permute(0, 2, 3, 1).reshape(B * Fr, T, C)  # (B*F, T, C)
        x_inter = x_inter + self.mamba_inter(self.norm_inter(x_inter))
        x_inter = x_inter.reshape(B, Fr, T, C).permute(0, 3, 1, 2)  # (B, C, F, T)

        return x_inter


# ============================================================================
#  Frequency-Domain Encoder Block — Conv2D + Dual-Path Mamba
# ============================================================================

class FreqConvBlockV3(nn.Module):
    """Frequency-domain encoder block with Dual-Path Mamba.

    1. 2D convolution for local time-frequency feature extraction.
    2. Dual-Path Mamba for global intra-frame + inter-frame modelling.

    d_model = out_channels at all stages — no dimension explosion.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 freq_bins_out: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 stride_freq: int = 2, stride_time: int = 1):
        super().__init__()
        self.out_channels = out_channels

        # 2-D conv with possible frequency-axis downsampling
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1,
                      stride=(stride_freq, stride_time)),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
        )

        # Residual projection when channel/spatial dims change
        self.skip_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=(stride_freq, stride_time)
        ) if (in_channels != out_channels or stride_freq != 1 or stride_time != 1) else nn.Identity()

        # V3: Dual-Path Mamba (intra-freq + inter-time), d_model = out_channels
        self.dual_path_mamba = DualPathMambaBlock(
            d_model=out_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    @autocast('cuda', enabled=False)
    def forward(self, x):
        """
        Args:
            x: (B, C_in, F, T_stft)
        Returns:
            out: (B, C_out, F', T_stft')
        """
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        identity = self.skip_proj(x)
        h = self.conv2d(x) + identity  # (B, C, F', T)

        # Dual-Path Mamba processes (B, C, F', T) with residual connection
        h = h + self.dual_path_mamba(h)

        return h


class FreqDomainEncoderV3(nn.Module):
    """Multi-stage frequency-domain encoder using FreqConvBlockV3."""

    def __init__(self, n_stages: int, features_per_stage: List[int],
                 input_length: int,
                 freq_features_per_stage: List[int] = None,
                 n_fft: int = 256, hop_length: int = 64, win_length: int = 256,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.stft = STFTFrontend(n_fft=n_fft, hop_length=hop_length,
                                 win_length=win_length)
        self.n_stages = n_stages
        self.n_fft = n_fft

        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        # Pre-compute STFT output shape and per-stage freq-axis sizes
        n_freq, _n_time = _compute_stft_output_size(input_length, n_fft, hop_length)
        freq_sizes = _compute_freq_sizes_per_stage(n_freq, n_stages)

        in_ch = 4  # STFT gives 4 channels
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            out_ch = freq_features_per_stage[s]
            self.stages.append(
                FreqConvBlockV3(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    freq_bins_out=freq_sizes[s],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    stride_freq=2 if s > 0 else 1,
                    stride_time=1,
                )
            )
            in_ch = out_ch

        self.freq_features_per_stage = freq_features_per_stage
        self.freq_sizes = freq_sizes

    def forward(self, x):
        """
        Args:
            x: (B, 2, L) raw IQ waveform
        Returns:
            freq_features: list of (B, C_s, F_s, T_stft) for each stage
        """
        spec = self.stft(x)
        feats = []
        h = spec
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


# ============================================================================
#  V3 Cross-Attention Fusion — Unidirectional + Zero-Init + Positional Encoding
# ============================================================================

class CrossAttentionFusionV3(nn.Module):
    """Unidirectional cross-attention with positional encoding & zero-init gate.

    Three improvements over original CrossAttentionFusion:
    1. **Unidirectional**: Only Attn(Q=time, K=freq, V=freq).
    2. **Zero-Init Gate**: gate starts at 0 → identity at epoch 0.
    3. **Learnable Positional Encoding**: both time and freq tokens receive
       position embeddings so the model knows "time token i ≈ STFT frame j".
    """

    def __init__(self, time_dim: int, freq_channels: int, freq_bins: int,
                 n_heads: int = 4, dropout: float = 0.0,
                 max_time_len: int = 8192, max_freq_frames: int = 512):
        super().__init__()
        self.time_dim = time_dim

        # Project flattened freq feature to time_dim
        freq_flat_dim = freq_channels * freq_bins
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_flat_dim, time_dim),
            nn.LayerNorm(time_dim),
        )

        # Cross-Attention: time reads from freq
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=time_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_t = nn.LayerNorm(time_dim)
        self.norm_f = nn.LayerNorm(time_dim)

        # V3: Learnable positional encodings for alignment
        self.time_pos_enc = nn.Parameter(
            torch.randn(1, max_time_len, time_dim) * 0.02
        )
        self.freq_pos_enc = nn.Parameter(
            torch.randn(1, max_freq_frames, time_dim) * 0.02
        )

        # Zero-initialized gate
        self.gate = nn.Parameter(torch.zeros(1))

    @autocast('cuda', enabled=False)
    def forward(self, x_time, x_freq):
        """
        Args:
            x_time: (B, D_t, L_t)
            x_freq: (B, C_f, F_f, T_f)
        Returns:
            x_time_fused: (B, D_t, L_t)
            x_freq:       (B, C_f, F_f, T_f) — unchanged
        """
        if x_time.dtype in HALF_PRECISION_DTYPES:
            x_time = x_time.float()
        if x_freq.dtype in HALF_PRECISION_DTYPES:
            x_freq = x_freq.float()

        B, D_t, L_t = x_time.shape
        B, C_f, F_f, T_f = x_freq.shape

        # --- Prepare time tokens with positional encoding ---
        time_tokens = x_time.transpose(1, 2)  # (B, L_t, D_t)
        time_tokens = time_tokens + self.time_pos_enc[:, :L_t, :]

        # --- Prepare freq tokens with positional encoding ---
        freq_flat = x_freq.permute(0, 3, 1, 2).reshape(B, T_f, C_f * F_f)
        freq_tokens = self.freq_proj(freq_flat)  # (B, T_f, D_t)
        freq_tokens = freq_tokens + self.freq_pos_enc[:, :T_f, :]

        # --- Cross-Attention: time reads from freq ---
        time_normed = self.norm_t(time_tokens)
        freq_normed = self.norm_f(freq_tokens)
        attn_out, _ = self.cross_attn(
            query=time_normed,
            key=freq_normed,
            value=freq_normed,
        )

        # Zero-init gate: identity at start, gradually opens
        time_fused = time_tokens + self.gate * attn_out

        # Back to original shape
        x_time_fused = time_fused.transpose(1, 2)  # (B, D_t, L_t)

        # Freq features pass through unchanged
        return x_time_fused, x_freq


# ============================================================================
#  V3 Dual-Domain Encoder
# ============================================================================

class DualDomainMambaEncoderV3(nn.Module):
    """Time-domain encoder + Dual-Path Mamba freq tower + positional cross-attn.

    Improvements over V1/V2:
    - FreqConvBlock → FreqConvBlockV3 (Dual-Path Mamba, no flatten)
    - CrossAttentionFusion → CrossAttentionFusionV3 (with positional encoding)
    """

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
                 # Frequency-domain tower settings
                 n_fft: int = 256,
                 hop_length: int = 64,
                 win_length: int = 256,
                 freq_features_per_stage: List[int] = None,
                 cross_attn_heads: int = 4,
                 cross_attn_dropout: float = 0.0,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
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

        # ---- Time-Domain Conv + Mamba Stages ----
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

        # ---- V3: Dual-Path Mamba Frequency Tower ----
        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        input_length = int(input_size[0])

        self.freq_encoder = FreqDomainEncoderV3(
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            input_length=input_length,
            freq_features_per_stage=freq_features_per_stage,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # ---- V3: Cross-Attention with Positional Encoding ----
        freq_sizes = self.freq_encoder.freq_sizes
        self.cross_fusions = nn.ModuleList([
            CrossAttentionFusionV3(
                time_dim=features_per_stage[s],
                freq_channels=freq_features_per_stage[s],
                freq_bins=freq_sizes[s],
                n_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
                max_time_len=input_length,  # upper bound for time tokens
                max_freq_frames=input_length // hop_length + 1,
            )
            for s in range(n_stages)
        ])

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
        """
        Args:
            x: (B, 2, L) raw IQ waveform
        Returns:
            If return_skips: (time_skips, freq_feats)
            Else: time_skips[-1]
        """
        # --- Frequency tower: extract multi-scale features ---
        freq_feats = self.freq_encoder(x)

        # --- Time tower ---
        if self.stem is not None:
            x = self.stem(x)

        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)

            # V3: Unidirectional fusion with positional encoding
            x, _ = self.cross_fusions[s](x, freq_feats[s])

            ret.append(x)

        if self.return_skips:
            return ret, freq_feats
        return ret[-1]


# ============================================================================
#  V3 Decoder — UNetResDecoder with Frequency Feature Injection
# ============================================================================

class UNetResDecoderV3(nn.Module):
    """UNet decoder that also injects frequency-domain skip features.

    At each upsampling stage, the frequency tower's corresponding feature
    is average-pooled along the freq axis, projected to match the skip
    channel count, and added to the time-domain skip connection.
    """

    def __init__(self, encoder, num_classes, n_conv_per_stage,
                 deep_supervision, freq_features_per_stage, freq_sizes):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1) if isinstance(n_conv_per_stage, int) else n_conv_per_stage

        stages = []
        upsample_layers = []
        seg_layers = []
        skip_processors = []
        freq_injectors = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s][0]

            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,
                pool_op_kernel_size=stride_for_upsampling,
                mode='linear' if encoder.conv_op == nn.Conv1d else 'nearest'
            ))

            skip_processors.append(
                SkipConnectionProcessor(
                    skip_channels=input_features_skip,
                    upsampled_channels=input_features_skip,
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
            )

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    padding=encoder.conv_pad_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=True,
                ),
                *[BasicBlockD(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                ) for _ in range(n_conv_per_stage[s-1] - 1)]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1))

            # V3: Frequency feature injector for this decoder stage
            # The skip at position -(s+1) in the encoder corresponds to
            # freq_features_per_stage[-(s+1)] channels.
            freq_ch = freq_features_per_stage[-(s + 1)]
            freq_injectors.append(nn.Sequential(
                # Pool freq axis: (B, C_f, F, T) -> (B, C_f, 1, T) -> (B, C_f, T)
                nn.AdaptiveAvgPool2d((1, None)),
                nn.Flatten(1, 2),  # (B, C_f, T)
                nn.Conv1d(freq_ch, input_features_skip, kernel_size=1),
                nn.InstanceNorm1d(input_features_skip, eps=1e-5, affine=True),
                nn.LeakyReLU(inplace=True),
            ))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.skip_processors = nn.ModuleList(skip_processors)
        self.freq_injectors = nn.ModuleList(freq_injectors)

        # Learnable blending weights for freq injection (start small)
        self.freq_blend_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(n_stages_encoder - 1)
        ])

    def forward(self, skips, freq_feats):
        """
        Args:
            skips:      list of time-domain encoder features
            freq_feats: list of freq-domain encoder features
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)

            # --- V3: Inject frequency features into skip connection ---
            freq_feat = freq_feats[-(s + 2)]  # corresponding freq stage
            freq_hint = self.freq_injectors[s](freq_feat)  # (B, skip_ch, T_stft)
            # Interpolate to match time-domain skip length
            skip = skips[-(s + 2)]
            freq_hint = F.interpolate(
                freq_hint, size=skip.shape[-1], mode='linear', align_corners=False
            )
            # Blend with learned weight (zero-init → no injection at start)
            skip = skip + self.freq_blend_weights[s] * freq_hint

            processed_skip = self.skip_processors[s](skip, x)
            x = torch.cat((x, processed_skip), 1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


# ============================================================================
#  Top-level model: DualDomainMambaV3
# ============================================================================

class DualDomainMambaV3(nn.Module):
    """Dual-Path Mamba for IQ signal separation.

    Key innovations:
    1. Dual-Path Mamba in freq tower (intra-freq + inter-time, d_model=C)
    2. Positional encoding in cross-attention for time-freq alignment
    3. Zero-init gating for stable warm-up
    4. Frequency feature injection into decoder (not just encoder)
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
                 cross_attn_dropout: float = 0.0,
                 ):
        super().__init__()

        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        self.encoder = DualDomainMambaEncoderV3(
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
            # Freq tower settings
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            freq_features_per_stage=freq_features_per_stage,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dropout=cross_attn_dropout,
        )

        # V3: Decoder with frequency feature injection
        freq_sizes = self.encoder.freq_encoder.freq_sizes
        self.decoder = UNetResDecoderV3(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            freq_features_per_stage=freq_features_per_stage,
            freq_sizes=freq_sizes,
        )

    def forward(self, x):
        skips, freq_feats = self.encoder(x)
        return self.decoder(skips, freq_feats)
