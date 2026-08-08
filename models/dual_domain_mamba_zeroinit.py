"""DualDomainMambaZeroInit — V1 + Zero-Initialized Gate (controlled ablation).

This is an EXACT copy of dual_domain_mamba.py (V1) with ONE change:
  - Gate initialization:  sigmoid(-2.0) ≈ 0.12  →  zero (0.0)
  - Gate usage:           sigmoid(gate) * attn   →  gate * attn

Purpose: isolate the effect of gradient pollution prevention.
All other components (flatten Mamba, bidirectional cross-attention,
parameter counts, dimensions) are IDENTICAL to V1.

Architecture overview
=====================
Two parallel "towers" process the input signal simultaneously:

  1. **Time-Domain Tower** — the existing ResidualMambaEncoder backbone
     operating on the raw IQ waveform  (B, 2, L).
  2. **Frequency-Domain Tower** — a lightweight CNN+Mamba backbone that
     operates on the STFT magnitude/phase spectrogram (B, C_freq, F, T_stft).

At every encoder stage, a **bidirectional Cross-Attention Fusion** module
lets the two towers exchange information:

    Feat_time += CrossAttn(Q=time, K=V=freq)
    Feat_freq += CrossAttn(Q=freq, K=V=time)

The fused multi-scale skip connections from the Time Tower are then fed into
the existing UNetResDecoder for final waveform reconstruction in the time
domain — thereby preserving absolute phase accuracy.

Key design choices
------------------
* STFT parameters (n_fft, hop_length) are configurable.
* The frequency tower re-uses 1-D Mamba blocks along the STFT *time* axis,
  treating the frequency bins as channels — this keeps the model efficient.
* Cross-Attention uses grouped attention with a small number of heads.
* **All sub-modules are eagerly initialized** in __init__ so that every
  parameter is visible to the optimizer from the start and checkpoints
  can be loaded/saved with strict=True.
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
    """Return (n_freq_bins, n_time_frames) for the given STFT parameters.

    This mirrors ``torch.stft`` output dimensions exactly so that all
    downstream layers can be eagerly constructed.
    """
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

    Output channels = 4  (real_I, imag_I, real_Q, imag_Q)  so the freq tower
    can reason about both magnitude and phase of each IQ component.
    """

    def __init__(self, n_fft: int = 256, hop_length: int = 64,
                 win_length: int = 256, center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        # Hann window (registered as buffer for device tracking)
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
            # torch.stft returns (B, F, T_stft) complex tensor
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
#  Frequency-Domain Encoder Block (lightweight 2-D Conv + 1-D Mamba)
# ============================================================================

class FreqConvBlock(nn.Module):
    """A single frequency-domain encoder block.

    1. 2-D convolution(s) across (Freq, Time) to capture local spectral patterns.
    2. Reshape to (B, F*C, T_stft) and run Mamba along the *time* axis.

    The Mamba dimension ``out_channels * freq_bins_out`` is computed from
    known constants and passed in at construction time — no lazy init.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 freq_bins_out: int,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 stride_freq: int = 2, stride_time: int = 1):
        super().__init__()

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

        # 1-D Mamba along the STFT time axis — eagerly constructed
        # After conv2d: (B, out_channels, freq_bins_out, T_stft)
        # Flatten freq → channels: mamba_dim = out_channels * freq_bins_out
        mamba_dim = out_channels * freq_bins_out
        self.mamba_norm = nn.LayerNorm(mamba_dim)
        self.mamba = Mamba(
            d_model=mamba_dim,
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
        h = self.conv2d(x) + identity

        B, C, Fr, T = h.shape
        mamba_dim = C * Fr

        # (B, C, F', T) -> (B, C*F', T) -> (B, T, C*F')
        h_flat = h.reshape(B, mamba_dim, T).transpose(1, 2)
        h_flat = self.mamba_norm(h_flat)
        h_mamba = self.mamba(h_flat)  # (B, T, D)
        h_mamba = h_mamba.transpose(1, 2).reshape(B, C, Fr, T)

        return h + h_mamba  # residual


class FreqDomainEncoder(nn.Module):
    """Multi-stage frequency-domain encoder that mirrors the time-domain stages.

    At each stage, produces a feature that can be aligned with the corresponding
    time-domain feature for cross-attention.

    All layer dimensions are computed eagerly from ``input_length`` and STFT
    parameters so every parameter exists at construction time.
    """

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

        in_ch = 4  # STFT gives 4 channels (real_I, imag_I, real_Q, imag_Q)
        self.stages = nn.ModuleList()
        for s in range(n_stages):
            out_ch = freq_features_per_stage[s]
            self.stages.append(
                FreqConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    freq_bins_out=freq_sizes[s],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    stride_freq=2 if s > 0 else 1,  # downsample freq axis
                    stride_time=1,
                )
            )
            in_ch = out_ch

        self.freq_features_per_stage = freq_features_per_stage
        self.freq_sizes = freq_sizes  # exposed for CrossAttentionFusion eager init

    def forward(self, x):
        """
        Args:
            x: (B, 2, L) raw IQ waveform
        Returns:
            freq_features: list of (B, C_s, F_s, T_stft) for each stage
        """
        spec = self.stft(x)  # (B, 4, F, T_stft)
        feats = []
        h = spec
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


# ============================================================================
#  Cross-Attention Fusion Module
# ============================================================================

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between time-domain and freq-domain features.

    Given:
        x_time : (B, D_t, L_t)   — time-domain feature at some encoder stage
        x_freq : (B, C_f, F_f, T_f) — freq-domain feature at the same stage

    1. Project freq feature to (B, D_t, T_f) via Linear over flattened freq axis.
    2. Run Cross-Attention: time→freq and freq→time.
    3. Return updated (x_time_fused, x_freq_fused).
    """

    def __init__(self, time_dim: int, freq_channels: int, freq_bins: int,
                 n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.time_dim = time_dim

        # Project flattened freq feature to time_dim
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

        # Zero-initialized fusion gate (no noise injection at epoch 0)
        self.gate_time = nn.Parameter(torch.zeros(1))
        self.gate_freq = nn.Parameter(torch.zeros(1))

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

        # --- Prepare time tokens: (B, L_t, D_t) ---
        time_tokens = x_time.transpose(1, 2)  # (B, L_t, D_t)

        # --- Prepare freq tokens: flatten freq axis → (B, T_f, C_f*F_f) → project → (B, T_f, D_t) ---
        freq_flat = x_freq.permute(0, 3, 1, 2).reshape(B, T_f, C_f * F_f)  # (B, T_f, C_f*F_f)
        freq_tokens = self.freq_proj(freq_flat)  # (B, T_f, D_t)

        # --- Cross-Attention: time reads from freq ---
        time_normed = self.norm_t(time_tokens)
        freq_normed_for_t = self.norm_f_for_t(freq_tokens)
        attn_t2f, _ = self.cross_attn_t2f(
            query=time_normed,
            key=freq_normed_for_t,
            value=freq_normed_for_t,
        )
        time_fused = time_tokens + self.gate_time * attn_t2f

        # --- Cross-Attention: freq reads from time ---
        freq_normed = self.norm_f(freq_tokens)
        time_normed_for_f = self.norm_t_for_f(time_tokens)
        attn_f2t, _ = self.cross_attn_f2t(
            query=freq_normed,
            key=time_normed_for_f,
            value=time_normed_for_f,
        )
        freq_fused = freq_tokens + self.gate_freq * attn_f2t

        # --- Back to original shapes ---
        x_time_fused = time_fused.transpose(1, 2)  # (B, D_t, L_t)

        # Back-project freq tokens to original space
        freq_back = self.freq_back_proj(freq_fused)  # (B, T_f, C_f*F_f)
        x_freq_fused = freq_back.reshape(B, T_f, C_f, F_f).permute(0, 2, 3, 1)  # (B, C_f, F_f, T_f)
        x_freq_fused = x_freq_fused + x_freq  # residual

        return x_time_fused, x_freq_fused


# ============================================================================
#  Dual-Domain Time-Domain Encoder (with parallel freq tower + cross-attn)
# ============================================================================

class DualDomainMambaEncoder(nn.Module):
    """Time-domain ResidualMambaEncoder augmented with a parallel frequency
    tower and stage-wise cross-attention fusion.

    The architecture mirrors ResidualMambaEncoder but at each stage:
    1. Time feature goes through Conv+Mamba (same as original).
    2. Freq feature goes through FreqConvBlock.
    3. Cross-Attention fuses features bidirectionally.
    4. The (now frequency-informed) time feature is stored as skip.

    All sub-modules are constructed eagerly so that:
    - ``optimizer = Adam(model.parameters())`` captures every parameter.
    - ``load_state_dict(strict=True)`` works without missing/unexpected keys.
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

        # ---- Frequency-Domain Tower ----
        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        input_length = int(input_size[0])  # scalar waveform length

        self.freq_encoder = FreqDomainEncoder(
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

        # ---- Cross-Attention Fusion at each stage (eagerly initialized) ----
        freq_sizes = self.freq_encoder.freq_sizes  # list of int, known at init
        self.cross_fusions = nn.ModuleList([
            CrossAttentionFusion(
                time_dim=features_per_stage[s],
                freq_channels=freq_features_per_stage[s],
                freq_bins=freq_sizes[s],
                n_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
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
        # --- Frequency tower: extract multi-scale features in advance ---
        freq_feats = self.freq_encoder(x)  # list of (B, C_f_s, F_s, T_s)

        # --- Time tower ---
        if self.stem is not None:
            x = self.stem(x)

        ret = []
        for s in range(len(self.stages)):
            # Time-domain conv + mamba
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)

            # Cross-attention fusion with frequency features
            x, freq_feats[s] = self.cross_fusions[s](x, freq_feats[s])

            ret.append(x)

        return ret if self.return_skips else ret[-1]


# ============================================================================
#  Top-level model: DualDomainMamba
# ============================================================================

class DualDomainMambaZeroInit(nn.Module):
    """Dual-Domain Cross-Attention Mamba for IQ signal separation.

    Combines a time-domain Mamba U-Net with a parallel frequency-domain tower.
    Multi-scale cross-attention fuses the two domains at every encoder stage.
    The decoder operates purely in the time domain to preserve phase accuracy.
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
