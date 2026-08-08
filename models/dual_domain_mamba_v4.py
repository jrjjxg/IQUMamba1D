"""DualDomainMambaV4 — Stability-Optimised Dual-Domain Cross-Attention Mamba.

Compared to V1 (dual_domain_mamba.py), this version applies three targeted
optimisations that address the known training-instability characteristics of
Mamba (arXiv 2024) **without sacrificing** the three core strengths of V1:

Optimisation 1 — Bottleneck Mamba in Frequency Tower
    V1:  Flatten(C*F) → Mamba(d_model = C*F ≈ 2000+) — huge A-matrix, hard
         to control eigenvalues, gradient explosion/vanishing risk.
    V4:  Flatten(C*F) → Linear(C*F → bottleneck) → Mamba(bottleneck) →
         Linear(bottleneck → C*F).  Default bottleneck = 256.
         Still preserves joint spectral encoding via the Linear projections
         but reduces A-matrix size by ~8×, dramatically improving stability.

Optimisation 2 — Selective Cross-Attention (deep stages only)
    V1:  Cross-Attention at *every* encoder stage, including Stage 0 where
         time has 4096 tokens vs freq has 65 tokens (63:1 ratio) — the
         extremely peaked softmax is a major source of gradient variance.
    V4:  Cross-Attention only at stages ≥ `fusion_start_stage` (default 2).
         Shallow stages pass through with identity — features are allowed
         to mature before being fused across domains.

Optimisation 3 — Per-Channel Sigmoid Gate with Conservative Init
    V1:  Scalar sigmoid(-2) ≈ 0.12 gate — one number controls all channels.
    V4:  Per-channel gate  → (1, 1, D_t) with init sigmoid(-4) ≈ 0.018.
         Each channel independently learns its fusion strength, and the
         very conservative start gives both towers more room to stabilise.
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
    """Compute the frequency-axis size after each FreqConvBlock stage."""
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
#  Frequency-Domain Encoder Block — Bottleneck Mamba (Optimisation 1)
# ============================================================================

class FreqConvBlock(nn.Module):
    """Frequency-domain encoder block with **Bottleneck Mamba**.

    Instead of a giant Mamba(d_model = C*F ≈ 2000), we project down to
    ``bottleneck_dim`` (default 256) before Mamba and project back afterwards.
    This keeps the joint spectral encoding property of flatten-Mamba while
    dramatically reducing the A-matrix size and associated instability.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 freq_bins_out: int,
                 bottleneck_dim: int = 256,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 stride_freq: int = 2, stride_time: int = 1):
        super().__init__()

        # 2-D conv with possible frequency-axis downsampling (identical to V1)
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

        # Residual projection (identical to V1)
        self.skip_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=(stride_freq, stride_time)
        ) if (in_channels != out_channels or stride_freq != 1 or stride_time != 1) else nn.Identity()

        # --- Bottleneck Mamba (NEW in V4) ---
        flat_dim = out_channels * freq_bins_out
        # Clamp bottleneck so it never exceeds flat_dim
        actual_bottleneck = min(bottleneck_dim, flat_dim)

        self.pre_mamba = nn.Linear(flat_dim, actual_bottleneck)
        self.mamba_norm = nn.LayerNorm(actual_bottleneck)
        self.mamba = Mamba(
            d_model=actual_bottleneck,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.post_mamba = nn.Linear(actual_bottleneck, flat_dim)

    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        identity = self.skip_proj(x)
        h = self.conv2d(x) + identity

        B, C, Fr, T = h.shape
        flat_dim = C * Fr

        # (B, C, F', T) -> (B, T, C*F')
        h_flat = h.reshape(B, flat_dim, T).transpose(1, 2)

        # Bottleneck: project down → Mamba → project up
        h_bn = self.pre_mamba(h_flat)           # (B, T, bottleneck)
        h_bn = self.mamba_norm(h_bn)
        h_mamba = self.mamba(h_bn)               # (B, T, bottleneck)
        h_up = self.post_mamba(h_mamba)          # (B, T, C*F')

        h_up = h_up.transpose(1, 2).reshape(B, C, Fr, T)
        return h + h_up  # residual


class FreqDomainEncoder(nn.Module):
    """Multi-stage frequency-domain encoder (with bottleneck Mamba)."""

    def __init__(self, n_stages: int, features_per_stage: List[int],
                 input_length: int,
                 freq_features_per_stage: List[int] = None,
                 bottleneck_dim: int = 256,
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

        in_ch = 4
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            out_ch = freq_features_per_stage[s]
            self.stages.append(
                FreqConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    freq_bins_out=freq_sizes[s],
                    bottleneck_dim=bottleneck_dim,
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
        spec = self.stft(x)
        feats = []
        h = spec
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


# ============================================================================
#  Cross-Attention Fusion — Per-Channel Gate (Optimisation 3)
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention with **per-channel sigmoid gate**.

    Changes from V1:
      - gate shape:  scalar (1,)  →  per-channel (1, 1, D_t)
      - gate init:   sigmoid(-2) ≈ 0.12  →  sigmoid(-4) ≈ 0.018
    """

    def __init__(self, time_dim: int, freq_channels: int, freq_bins: int,
                 n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.time_dim = time_dim

        freq_flat_dim = freq_channels * freq_bins
        self.freq_proj = nn.Sequential(
            nn.Linear(freq_flat_dim, time_dim),
            nn.LayerNorm(time_dim),
        )

        # Time → reads from Freq
        self.cross_attn_t2f = nn.MultiheadAttention(
            embed_dim=time_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_t = nn.LayerNorm(time_dim)
        self.norm_f_for_t = nn.LayerNorm(time_dim)

        # Freq → reads from Time
        self.cross_attn_f2t = nn.MultiheadAttention(
            embed_dim=time_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_f = nn.LayerNorm(time_dim)
        self.norm_t_for_f = nn.LayerNorm(time_dim)

        # Back-project to freq space
        self.freq_back_proj = nn.Linear(time_dim, freq_flat_dim)

        # Per-channel sigmoid gate with conservative init (sigmoid(-4) ≈ 0.018)
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

        time_tokens = x_time.transpose(1, 2)  # (B, L_t, D_t)

        freq_flat = x_freq.permute(0, 3, 1, 2).reshape(B, T_f, C_f * F_f)
        freq_tokens = self.freq_proj(freq_flat)  # (B, T_f, D_t)

        # --- Cross-Attention: time reads from freq ---
        time_normed = self.norm_t(time_tokens)
        freq_normed_for_t = self.norm_f_for_t(freq_tokens)
        attn_t2f, _ = self.cross_attn_t2f(
            query=time_normed,
            key=freq_normed_for_t,
            value=freq_normed_for_t,
        )
        # Per-channel sigmoid gate (bounded [0, 1] always)
        time_fused = time_tokens + torch.sigmoid(self.gate_time) * attn_t2f

        # --- Cross-Attention: freq reads from time ---
        freq_normed = self.norm_f(freq_tokens)
        time_normed_for_f = self.norm_t_for_f(time_tokens)
        attn_f2t, _ = self.cross_attn_f2t(
            query=freq_normed,
            key=time_normed_for_f,
            value=time_normed_for_f,
        )
        freq_fused = freq_tokens + torch.sigmoid(self.gate_freq) * attn_f2t

        # --- Back to original shapes ---
        x_time_fused = time_fused.transpose(1, 2)

        freq_back = self.freq_back_proj(freq_fused)
        x_freq_fused = freq_back.reshape(B, T_f, C_f, F_f).permute(0, 2, 3, 1)
        x_freq_fused = x_freq_fused + x_freq  # residual

        return x_time_fused, x_freq_fused


# ============================================================================
#  Identity Fusion — used for shallow stages (Optimisation 2)
# ============================================================================

class IdentityFusion(nn.Module):
    """No-op fusion: returns inputs unchanged.

    Used at shallow encoder stages where features are not mature enough
    for cross-domain fusion and where the extreme sequence-length ratio
    (e.g. 4096:65) makes attention numerically fragile.
    """
    def forward(self, x_time, x_freq):
        return x_time, x_freq


# ============================================================================
#  Dual-Domain Encoder — Selective Fusion (Optimisation 2)
# ============================================================================

class DualDomainMambaEncoder(nn.Module):
    """Time-domain encoder augmented with a parallel frequency tower.

    Cross-attention fusion is applied only at stages >= ``fusion_start_stage``
    (default 2).  Earlier stages use IdentityFusion to let features mature.
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
                 # V4-specific
                 bottleneck_dim: int = 256,
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

        # ---- Frequency-Domain Tower ----
        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        input_length = int(input_size[0])

        self.freq_encoder = FreqDomainEncoder(
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            input_length=input_length,
            freq_features_per_stage=freq_features_per_stage,
            bottleneck_dim=bottleneck_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # ---- Selective Cross-Attention Fusion (Optimisation 2) ----
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
        # --- Frequency tower: extract multi-scale features in advance ---
        freq_feats = self.freq_encoder(x)

        # --- Time tower ---
        if self.stem is not None:
            x = self.stem(x)

        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)

            # Cross-attention fusion (real at deep stages, identity at shallow)
            x, freq_feats[s] = self.cross_fusions[s](x, freq_feats[s])

            ret.append(x)

        return ret if self.return_skips else ret[-1]


# ============================================================================
#  Top-level model: DualDomainMambaV4
# ============================================================================

class DualDomainMambaV4(nn.Module):
    """Stability-Optimised Dual-Domain Cross-Attention Mamba.

    Three targeted changes from V1:
      1. Bottleneck Mamba in freq tower (d_model 256 instead of 2000+).
      2. Cross-Attention only at deep stages (default stage 2+).
      3. Per-channel sigmoid gate with sigmoid(-4) ≈ 0.018 init.
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
                 # V4-specific
                 bottleneck_dim: int = 256,
                 fusion_start_stage: int = 2,
                 ):
        super().__init__()
        self.encoder = DualDomainMambaEncoder(
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
            # V4-specific
            bottleneck_dim=bottleneck_dim,
            fusion_start_stage=fusion_start_stage,
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
