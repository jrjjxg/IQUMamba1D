"""DualDomainMambaLite — Lightweight variant of DualDomainMamba.

Two independent optimizations vs. the original DualDomainMamba:

1. **FreqConvBlockLite** (freq-tower Mamba):
   d_model = out_channels  (per-bin, weights shared across freq bins)
   instead of  d_model = out_channels * freq_bins_out  (joint).
   Reshape to (B*F', T, C) so Mamba runs independently per frequency bin.
   Param reduction: O((C*F)^2) → O(C^2), roughly 800x fewer Mamba params.

2. **CrossAttentionFusionLite** (freq→time projection):
   Replaces the naive Linear(C_f * F_f → D_t) — whose size scales with
   freq_bins and is only trained on ~65 time steps — with a Conv2d-based
   global-frequency-pooling path:

       (B, C_f, F_f, T_f)
         → Conv2d(C_f, proj_dim, kernel=(F_f, 1), groups=1)  [freq collapse]
         → (B, proj_dim, 1, T_f) → squeeze → (B, T_f, proj_dim)
         → Linear(proj_dim → D_t)                           [fixed small Linear]

   proj_dim is fixed (default min(time_dim, 64)), fully decoupled from freq_bins.
   Back-projection is symmetric.  This eliminates the over-parameterized large
   matrix that received only ~65 gradient updates per forward pass.
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
    UNetResDecoder,
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
    """Compute the frequency-axis size after each FreqConvBlock stage.

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
#  Frequency-Domain Encoder Block — LIGHTWEIGHT version
# ============================================================================

class FreqConvBlockLite(nn.Module):
    """Lightweight frequency-domain encoder block.

    Key change vs. the original FreqConvBlock:

        Original:  mamba_dim = out_channels × freq_bins_out     (e.g. 2064)
        Lite:      mamba_dim = out_channels                     (e.g. 16)

    After the 2-D convolution produces (B, C, F', T), we reshape to
    (B×F', T, C) so that Mamba runs **independently per frequency bin**
    along the time axis.  Weights are shared across frequency bins.

    Cross-frequency interactions are already captured by the 2-D convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 freq_bins_out: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 stride_freq: int = 2, stride_time: int = 1):
        super().__init__()
        self.out_channels = out_channels

        # 2-D conv with possible frequency-axis downsampling (unchanged)
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

        # Residual projection when channel/spatial dims change (unchanged)
        self.skip_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=(stride_freq, stride_time)
        ) if (in_channels != out_channels or stride_freq != 1 or stride_time != 1) else nn.Identity()

        # ---------------------------------------------------------------
        #  KEY CHANGE: Mamba with d_model = out_channels (NOT × freq_bins)
        # ---------------------------------------------------------------
        self.mamba_norm = nn.LayerNorm(out_channels)
        self.mamba = Mamba(
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

        B, C, Fr, T = h.shape

        # ---------------------------------------------------------------
        #  KEY CHANGE: reshape so Mamba runs per frequency bin
        #  (B, C, F', T) → (B*F', T, C) → Mamba → (B, C, F', T)
        # ---------------------------------------------------------------
        h_for_mamba = h.permute(0, 2, 3, 1).reshape(B * Fr, T, C)  # (B*F', T, C)
        h_for_mamba = self.mamba_norm(h_for_mamba)
        h_mamba = self.mamba(h_for_mamba)                           # (B*F', T, C)
        h_mamba = h_mamba.reshape(B, Fr, T, C).permute(0, 3, 1, 2) # (B, C, F', T)

        return h + h_mamba  # residual


class FreqDomainEncoderLite(nn.Module):
    """Multi-stage frequency-domain encoder using FreqConvBlockLite."""

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
                FreqConvBlockLite(
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
#  Cross-Attention Fusion Module — LITE version (Conv2d freq-pooling)
# ============================================================================

class CrossAttentionFusionLite(nn.Module):
    """Bidirectional cross-attention with Conv2d-based frequency compression.

    Root cause of the original CrossAttentionFusion's training instability:
        freq_proj = Linear(C_f * F_f, D_t)
    This matrix has C_f*F_f rows (e.g. 16*129=2064) but is updated via only
    ~65 time steps of gradient signal — severely under-trained.

    Fix: collapse the frequency axis with a Conv2d BEFORE the Linear:
        (B, C_f, F_f, T_f)
          → permute → (B*T_f, C_f, F_f, 1)          [batch over time]
          → Conv2d(C_f, proj_dim, (F_f, 1))          [global freq pooling]
          → (B, T_f, proj_dim)                       [fixed-size representation]
          → Linear(proj_dim, D_t)                    [small, well-trained]

    proj_dim is a small fixed constant (default: min(time_dim, 64)), fully
    independent of freq_bins.  Back-projection is symmetric.
    """

    def __init__(self, time_dim: int, freq_channels: int, freq_bins: int,
                 n_heads: int = 4, dropout: float = 0.0,
                 proj_dim: int = None):
        super().__init__()
        self.time_dim = time_dim
        self.freq_channels = freq_channels
        self.freq_bins = freq_bins

        # Fixed intermediate dimension, decoupled from freq_bins
        if proj_dim is None:
            proj_dim = min(time_dim, 64)
        self.proj_dim = proj_dim

        # --- Freq → token pipeline ---
        # Step 1: collapse freq axis via a (F_f, 1) conv  [global freq pooling]
        #   input:  (B*T_f, C_f, F_f, 1)
        #   output: (B*T_f, proj_dim, 1, 1)
        # After Conv2d(kernel=(F_f,1)) the spatial size is (1,1) — InstanceNorm2d
        # requires spatial dim > 1, so we use only a pointwise activation here.
        # Normalization is handled by the LayerNorm inside freq_proj.
        self.freq_collapse = nn.Sequential(
            nn.Conv2d(freq_channels, proj_dim,
                      kernel_size=(freq_bins, 1), padding=0),
            nn.LeakyReLU(inplace=True),
        )
        # Step 2: project proj_dim → time_dim
        self.freq_proj = nn.Sequential(
            nn.Linear(proj_dim, time_dim),
            nn.LayerNorm(time_dim),
        )

        # --- freq token → back to (B, C_f, F_f, T_f) pipeline ---
        # Step 1: project time_dim → proj_dim
        self.freq_back_small = nn.Linear(time_dim, proj_dim)
        # Step 2: expand proj_dim back to (C_f * F_f) via transposed conv
        #   input:  (B*T_f, proj_dim, 1, 1)
        #   output: (B*T_f, C_f, F_f, 1)
        self.freq_expand = nn.Sequential(
            nn.ConvTranspose2d(proj_dim, freq_channels,
                               kernel_size=(freq_bins, 1), padding=0),
        )

        # --- Cross-attention heads ---
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

        # Learnable fusion gates (init to -2 → sigmoid ≈ 0.12, conservative)
        self.gate_time = nn.Parameter(torch.full((1,), -2.0))
        self.gate_freq = nn.Parameter(torch.full((1,), -2.0))

    @autocast('cuda', enabled=False)
    def forward(self, x_time, x_freq):
        """
        Args:
            x_time: (B, D_t, L_t)
            x_freq: (B, C_f, F_f, T_f)
        Returns:
            x_time_fused: (B, D_t, L_t)
            x_freq_fused: (B, C_f, F_f, T_f)
        """
        if x_time.dtype in HALF_PRECISION_DTYPES:
            x_time = x_time.float()
        if x_freq.dtype in HALF_PRECISION_DTYPES:
            x_freq = x_freq.float()

        B, D_t, L_t = x_time.shape
        B, C_f, F_f, T_f = x_freq.shape

        # --- Time tokens: (B, L_t, D_t) ---
        time_tokens = x_time.transpose(1, 2)

        # --- Freq tokens via Conv2d global-freq-pooling ---
        # (B, C_f, F_f, T_f) → (B*T_f, C_f, F_f, 1) → collapse → (B*T_f, proj_dim, 1, 1)
        freq_for_conv = x_freq.permute(0, 3, 1, 2).reshape(B * T_f, C_f, F_f, 1)
        freq_collapsed = self.freq_collapse(freq_for_conv)          # (B*T_f, proj_dim, 1, 1)
        freq_collapsed = freq_collapsed.reshape(B, T_f, self.proj_dim)  # (B, T_f, proj_dim)
        freq_tokens = self.freq_proj(freq_collapsed)                # (B, T_f, D_t)

        # --- Cross-Attention: time reads from freq ---
        time_normed = self.norm_t(time_tokens)
        freq_normed_for_t = self.norm_f_for_t(freq_tokens)
        attn_t2f, _ = self.cross_attn_t2f(
            query=time_normed,
            key=freq_normed_for_t,
            value=freq_normed_for_t,
        )
        time_fused = time_tokens + torch.sigmoid(self.gate_time) * attn_t2f

        # --- Cross-Attention: freq reads from time ---
        freq_normed = self.norm_f(freq_tokens)
        time_normed_for_f = self.norm_t_for_f(time_tokens)
        attn_f2t, _ = self.cross_attn_f2t(
            query=freq_normed,
            key=time_normed_for_f,
            value=time_normed_for_f,
        )
        freq_fused = freq_tokens + torch.sigmoid(self.gate_freq) * attn_f2t  # (B, T_f, D_t)

        # --- Back to original shapes ---
        x_time_fused = time_fused.transpose(1, 2)  # (B, D_t, L_t)

        # Back-project freq: (B, T_f, D_t) → (B, T_f, proj_dim) → conv-expand → (B, C_f, F_f, T_f)
        freq_small = self.freq_back_small(freq_fused)               # (B, T_f, proj_dim)
        freq_for_expand = freq_small.reshape(B * T_f, self.proj_dim, 1, 1)
        freq_expanded = self.freq_expand(freq_for_expand)           # (B*T_f, C_f, F_f, 1)
        x_freq_fused = freq_expanded.reshape(B, T_f, C_f, F_f).permute(0, 2, 3, 1)  # (B, C_f, F_f, T_f)
        x_freq_fused = x_freq_fused + x_freq                       # residual

        return x_time_fused, x_freq_fused


# ============================================================================
#  Dual-Domain Encoder (using FreqDomainEncoderLite + CrossAttentionFusionLite)
# ============================================================================

class DualDomainMambaEncoderLite(nn.Module):
    """Time-domain ResidualMambaEncoder + lightweight parallel freq tower
    + stage-wise cross-attention fusion."""

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

        # ---- Lightweight Frequency-Domain Tower ----
        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        input_length = int(input_size[0])

        self.freq_encoder = FreqDomainEncoderLite(
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

        # ---- Cross-Attention Fusion (Lite) at each stage ----
        freq_sizes = self.freq_encoder.freq_sizes
        self.cross_fusions = nn.ModuleList([
            CrossAttentionFusionLite(
                time_dim=features_per_stage[s],
                freq_channels=freq_features_per_stage[s],
                freq_bins=freq_sizes[s],
                n_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
                proj_dim=min(features_per_stage[s], 64),
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
            list of skip connection features (if return_skips=True)
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

            # Cross-attention fusion
            x, freq_feats[s] = self.cross_fusions[s](x, freq_feats[s])

            ret.append(x)

        return ret if self.return_skips else ret[-1]


# ============================================================================
#  Top-level model: DualDomainMambaLite
# ============================================================================

class DualDomainMambaLite(nn.Module):
    """Lightweight Dual-Domain Cross-Attention Mamba for IQ signal separation.

    Identical to DualDomainMamba except the freq-tower Mamba blocks use
    d_model=out_channels (per-bin temporal modeling) instead of
    d_model=out_channels*freq_bins (joint spectral modeling).

    This reduces total parameters from ~100M+ to ~3M while preserving the
    dual-domain cross-attention architecture.
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
        self.encoder = DualDomainMambaEncoderLite(
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
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
