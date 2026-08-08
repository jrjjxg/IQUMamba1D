"""DualDomainMamba2 — Mamba-2 variant of DualDomainMamba.

Structural change: all Mamba (v1) layers are replaced with Mamba2 layers that
use the Structured State Space Duality (SSD) framework.  Key API differences:

    Mamba(d_model, d_state, d_conv, expand)
    →  Mamba2(d_model, d_state, d_conv, headdim)

Mamba2 constraints:
  - d_model must be divisible by headdim.
  - d_state can be much larger (64–128) for stronger modelling at lower cost.

All other components (CrossAttentionFusion, STFTFrontend, UNetResDecoder,
SkipConnectionProcessor, etc.) are reused without modification.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from mamba_ssm import Mamba2
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

# Re-use building blocks from IQUMamba1D and the original dual_domain_mamba
from models.IQUMamba1D import (
    UpsampleLayer,
    BasicResBlock,
    SkipConnectionProcessor,
    ChannelAttention1D,
    AdaptiveFusion1D,
    UNetResDecoder,
)
from models.dual_domain_mamba import (
    STFTFrontend,
    CrossAttentionFusion,
    _compute_stft_output_size,
    _compute_freq_sizes_per_stage,
    HALF_PRECISION_DTYPES,
)


# ============================================================================
#  Mamba2Layer — drop-in replacement for MambaLayer using Mamba-2 (SSD)
# ============================================================================

def _find_valid_headdim(dim: int, preferred: int = 32) -> int:
    """Return the largest headdim <= preferred that divides dim.

    Mamba2 requires ``d_model % headdim == 0``.  When the preferred value
    does not divide dim we fall back to smaller powers of two.
    """
    for hd in [preferred, 16, 8, 4, 2, 1]:
        if dim % hd == 0:
            return hd
    return 1  # always valid


class Mamba2Layer(nn.Module):
    """Mamba-2 layer with the same external interface as the original MambaLayer.

    Parameters
    ----------
    dim : int
        Model dimension (d_model for Mamba2).  Must be divisible by headdim.
    d_state : int
        State dimension.  Mamba2 supports much larger values (default 64).
    d_conv : int
        Local convolution width inside the SSM.
    headdim : int
        Head dimension (replaces ``expand`` from Mamba-1).
    channel_token : bool
        If True, treat channels as the sequence axis instead of spatial tokens.
    """

    def __init__(self, dim, d_state=64, d_conv=4, headdim=32, channel_token=False):
        super().__init__()
        self.dim = int(dim)
        headdim = _find_valid_headdim(int(dim), headdim)
        self.norm = nn.LayerNorm(int(dim))
        self.mamba = Mamba2(
            d_model=int(dim),
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *dims)
        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *dims)
        return out

    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        return out


# ============================================================================
#  FreqConvBlock2 — uses Mamba2 instead of Mamba for the STFT time-axis scan
# ============================================================================

class FreqConvBlock2(nn.Module):
    """Frequency-domain encoder block using Mamba-2.

    Identical to FreqConvBlock except the internal SSM is Mamba2.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 freq_bins_out: int,
                 d_state: int = 64, d_conv: int = 4, headdim: int = 32,
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

        # 1-D Mamba2 along the STFT time axis
        mamba_dim = out_channels * freq_bins_out
        headdim = _find_valid_headdim(mamba_dim, headdim)
        self.mamba_norm = nn.LayerNorm(mamba_dim)
        self.mamba = Mamba2(
            d_model=mamba_dim,
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
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


# ============================================================================
#  FreqDomainEncoder2 — uses FreqConvBlock2 (Mamba2)
# ============================================================================

class FreqDomainEncoder2(nn.Module):
    """Multi-stage frequency-domain encoder using Mamba-2."""

    def __init__(self, n_stages: int, features_per_stage: List[int],
                 input_length: int,
                 freq_features_per_stage: List[int] = None,
                 n_fft: int = 256, hop_length: int = 64, win_length: int = 256,
                 d_state: int = 64, d_conv: int = 4, headdim: int = 32):
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
                FreqConvBlock2(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    freq_bins_out=freq_sizes[s],
                    d_state=d_state,
                    d_conv=d_conv,
                    headdim=headdim,
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
        spec = self.stft(x)  # (B, 4, F, T_stft)
        feats = []
        h = spec
        for stage in self.stages:
            h = stage(h)
            feats.append(h)
        return feats


# ============================================================================
#  DualDomainMamba2Encoder — Mamba2 version of DualDomainMambaEncoder
# ============================================================================

class DualDomainMamba2Encoder(nn.Module):
    """Time-domain ResNet + Mamba2 encoder augmented with a parallel frequency
    tower (also Mamba2) and stage-wise cross-attention fusion.

    Identical architecture to DualDomainMambaEncoder except all SSM layers
    use Mamba-2 (SSD framework) for improved training stability and speed.
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
                 # Mamba2 settings
                 d_state: int = 64,
                 d_conv: int = 4,
                 headdim: int = 32,
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

        # ---- Time-Domain Conv + Mamba2 Stages ----
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
                    Mamba2Layer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s],
                        d_state=d_state,
                        d_conv=d_conv,
                        headdim=headdim,
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_ch = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)

        # ---- Frequency-Domain Tower (Mamba2) ----
        if freq_features_per_stage is None:
            freq_features_per_stage = [max(16, f // 2) for f in features_per_stage]

        input_length = int(input_size[0])

        self.freq_encoder = FreqDomainEncoder2(
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            input_length=input_length,
            freq_features_per_stage=freq_features_per_stage,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            d_state=d_state,
            d_conv=d_conv,
            headdim=headdim,
        )

        # ---- Cross-Attention Fusion at each stage (unchanged) ----
        freq_sizes = self.freq_encoder.freq_sizes
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
        freq_feats = self.freq_encoder(x)

        # --- Time tower ---
        if self.stem is not None:
            x = self.stem(x)

        ret = []
        for s in range(len(self.stages)):
            # Time-domain conv + mamba2
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)

            # Cross-attention fusion with frequency features
            x, freq_feats[s] = self.cross_fusions[s](x, freq_feats[s])

            ret.append(x)

        return ret if self.return_skips else ret[-1]


# ============================================================================
#  Top-level model: DualDomainMamba2
# ============================================================================

class DualDomainMamba2(nn.Module):
    """Dual-Domain Cross-Attention Mamba-2 for IQ signal separation.

    Identical to DualDomainMamba but all SSM layers use Mamba-2 (SSD).
    Benefits over Mamba-1:
      - Higher d_state (64 vs 16) for stronger sequence modelling.
      - Tensor-Core-friendly block computation → ~2-4x faster training.
      - Improved numerical stability via structured state space duality.
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
                 # Mamba2 specific
                 d_state: int = 64,
                 headdim: int = 32,
                 ):
        super().__init__()
        self.encoder = DualDomainMamba2Encoder(
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
            # Mamba2 settings
            d_state=d_state,
            headdim=headdim,
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
