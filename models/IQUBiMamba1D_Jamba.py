"""IQUBiMamba1D_Jamba — Hybrid BiMamba + Multi-Head Attention (Jamba-style).

Inspired by the Jamba architecture (AI21, 2024) which interleaves Mamba
layers with Transformer attention layers.  The key observation:

  - **Mamba (SSM)** excels at efficient, linear-complexity long-range
    dependency modeling, but compresses context into a fixed-size hidden
    state — it may lose "precise recall" of specific past tokens.
  - **Multi-Head Attention** provides exact, pairwise token interaction
    within a window, but has O(L²) cost.

By placing attention at the **bottleneck** (where the sequence is shortest
after downsampling), we get the best of both worlds:
  - BiMamba handles long sequences at early/middle stages efficiently.
  - Attention provides precise global interaction at the bottleneck cheaply
    (sequence is already short, e.g. 256 tokens at stage 4).

Configurable via ``attn_stages`` — a list of stage indices where MHA
replaces BiMamba.  Default: last stage only (bottleneck).
"""

import numpy as np
import torch
from torch import nn
from torch.amp import autocast

from mamba_ssm import Mamba

# Re-use building blocks from existing files
from models.IQUMamba1D import (
    UNetResDecoder,
    BasicResBlock,
    SkipConnectionProcessor,
)
from models.IQUBiMamba1D import BiMambaLayer

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from typing import Union, Type, List, Tuple, Optional
from torch.nn.modules.conv import _ConvNd

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  MultiHeadAttentionLayer — drop-in replacement for BiMambaLayer
# ============================================================================

class MultiHeadAttentionLayer(nn.Module):
    """Pre-norm Multi-Head Self-Attention with the same interface as BiMambaLayer.

    Includes a two-layer FFN (MLP) block after attention, following the
    standard Transformer "Attention + FFN" pattern.  Both sub-blocks use
    pre-LayerNorm and residual connections.
    """

    def __init__(self, dim, n_heads=4, dropout=0.0, ffn_expand=4,
                 channel_token=False):
        super().__init__()
        dim = int(dim)
        self.dim = dim
        self.channel_token = channel_token

        # --- Self-Attention sub-block ---
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # --- Feed-Forward sub-block ---
        self.norm_ffn = nn.LayerNorm(dim)
        ffn_hidden = dim * ffn_expand
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, dim),
            nn.Dropout(dropout),
        )

    def _attend(self, x):
        """Self-attention + FFN with residual.  x: [B, L, D]"""
        # Attention
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # residual

        # FFN
        x = x + self.ffn(self.norm_ffn(x))  # residual
        return x

    def forward_patch_token(self, x):
        """x: [B, C, *spatial] → flatten → attend → reshape back."""
        B, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = dims.numel()
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  # [B, L, D]
        out = self._attend(x_flat)
        return out.transpose(-1, -2).reshape(B, d_model, *dims)

    def forward_channel_token(self, x):
        """x: [B, L, *feat] → flatten feat → attend → reshape back."""
        B, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)  # [B, L, D]
        out = self._attend(x_flat)
        return out.reshape(B, n_tokens, *dims)

    def forward(self, x):
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


# ============================================================================
#  ResidualJambaEncoder — BiMamba + Attention hybrid
# ============================================================================

class ResidualJambaEncoder(nn.Module):
    """U-Net encoder with Jamba-style hybrid sequence modeling.

    At each stage that would normally receive a BiMambaLayer (the alternating
    pattern from the original design), this encoder instead checks whether
    the stage index is in ``attn_stages``:
      - If yes → ``MultiHeadAttentionLayer``
      - If no  → ``BiMambaLayer`` (original bidirectional Mamba)

    Stages that were ``nn.Identity()`` in the original design remain
    ``nn.Identity()`` — the alternating skip pattern is preserved.
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
                 # ---------- Jamba-specific ----------
                 attn_stages: Optional[List[int]] = None,
                 attn_n_heads: int = 4,
                 attn_dropout: float = 0.0,
                 attn_ffn_expand: int = 4,
                 ):
        super().__init__()

        # Default: attention at last stage (bottleneck) only
        if attn_stages is None:
            attn_stages = [n_stages - 1]
        attn_stages_set = set(attn_stages)

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

        input_channels = stem_channels
        stages = []
        seq_layers = []   # BiMamba or MHA or Identity
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
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

            # Decide which sequence modeling layer to use
            use_seq_layer = bool(s % 2) ^ bool(n_stages % 2)
            if use_seq_layer:
                dim = np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s]
                if s in attn_stages_set:
                    # ★ Attention at this stage
                    seq_layers.append(
                        MultiHeadAttentionLayer(
                            dim=dim,
                            n_heads=attn_n_heads,
                            dropout=attn_dropout,
                            ffn_expand=attn_ffn_expand,
                            channel_token=do_channel_token[s],
                        )
                    )
                else:
                    # BiMamba at this stage (original behavior)
                    seq_layers.append(
                        BiMambaLayer(
                            dim=dim,
                            channel_token=do_channel_token[s],
                        )
                    )
            else:
                seq_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(seq_layers)  # keep attribute name for decoder compatibility
        self.stages = nn.ModuleList(stages)
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
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


# ============================================================================
#  IQUBiMamba1D_Jamba — top-level model class
# ============================================================================

class IQUBiMamba1D_Jamba(nn.Module):
    """Jamba-style hybrid BiMamba + Attention U-Net for IQ signal separation.

    Uses BiMamba for most encoder stages and Multi-Head Attention at
    selected stages (default: bottleneck only).  This provides efficient
    long-range modeling via SSM and precise global token interaction via
    attention where it matters most.

    Parameters
    ----------
    attn_stages : list of int, optional
        Stage indices where MHA replaces BiMamba.  Default ``None`` means
        last stage only (bottleneck).
    attn_n_heads : int
        Number of attention heads.  Default 4.
    attn_dropout : float
        Dropout rate for attention and FFN.  Default 0.0.
    attn_ffn_expand : int
        FFN expansion factor (hidden = dim * expand).  Default 4.
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
                 # ---------- Jamba-specific ----------
                 attn_stages: Optional[List[int]] = None,
                 attn_n_heads: int = 4,
                 attn_dropout: float = 0.0,
                 attn_ffn_expand: int = 4,
                 ):
        super().__init__()
        self.encoder = ResidualJambaEncoder(
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
            # Jamba-specific
            attn_stages=attn_stages,
            attn_n_heads=attn_n_heads,
            attn_dropout=attn_dropout,
            attn_ffn_expand=attn_ffn_expand,
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
