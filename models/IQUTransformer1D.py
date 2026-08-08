"""IQUTransformer1D - Pure Transformer replacement for IQUMamba1D.

This model keeps the U-Net encoder/decoder skeleton, skip refinement, and
input/output contract of IQUMamba1D, while replacing the Mamba sequence
layers with Transformer blocks.

Key design choices:
  Transformer attention is permutation-invariant, so we inject sinusoidal
  positional encodings into the flattened token sequence before attention.
  This is the main "reasonable replacement" needed when swapping Mamba for
  self-attention on ordered 1D signal features.

  Token layout can be configured as:
    - "adaptive": follow IQUMamba1D and switch to channel-token in deep stages
    - "patch": always use time/patch tokens
    - "channel": always use channel tokens
"""

import math
from typing import List, Tuple, Type, Union

import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUMamba1D import BasicResBlock, UNetResDecoder


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class TransformerLayer(nn.Module):
    """Pre-norm Transformer block with configurable positional encoding."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_expand: int = 4,
        channel_token: bool = False,
        position_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        dim = int(dim)
        self.dim = dim
        self.channel_token = channel_token
        self.position_encoding_type = str(position_encoding_type).lower()
        if self.position_encoding_type not in {"sinusoidal", "rope", "none"}:
            raise ValueError(
                f"Unsupported position_encoding_type='{position_encoding_type}'. "
                "Expected one of: sinusoidal, rope, none."
            )

        require_even_head_dim = self.position_encoding_type == "rope"
        self.n_heads = self._resolve_num_heads(dim, n_heads, require_even_head_dim=require_even_head_dim)
        self.head_dim = dim // self.n_heads

        self.norm_attn = nn.LayerNorm(dim)
        if self.position_encoding_type == "rope":
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.attn_dropout = nn.Dropout(dropout)
            self.out_dropout = nn.Dropout(dropout)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=self.n_heads,
                dropout=dropout,
                batch_first=True,
            )
        self.norm_ffn = nn.LayerNorm(dim)
        hidden_dim = dim * ffn_expand
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.pos_scale = nn.Parameter(torch.ones(1))

    @staticmethod
    def _resolve_num_heads(dim: int, requested_heads: int, require_even_head_dim: bool = False) -> int:
        requested_heads = max(1, int(requested_heads))
        for heads in range(requested_heads, 0, -1):
            if dim % heads != 0:
                continue
            head_dim = dim // heads
            if require_even_head_dim and head_dim % 2 != 0:
                continue
            if head_dim <= 0:
                continue
            return heads
        if require_even_head_dim:
            raise ValueError(
                f"Cannot find a valid number of heads for dim={dim} "
                "that yields an even head dimension required by RoPE."
            )
        return 1

    @staticmethod
    def _sinusoidal_position_encoding(length: int, dim: int, device, dtype):
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(dim, 1))
        )
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0).to(dtype=dtype)

    @staticmethod
    def _rope_frequencies(length: int, head_dim: int, device, dtype):
        half_dim = head_dim // 2
        positions = torch.arange(length, device=device, dtype=torch.float32)
        inv_freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half_dim, device=device, dtype=torch.float32)
            / max(half_dim, 1)
        )
        angles = torch.outer(positions, inv_freq)
        cos = torch.cos(angles).view(1, 1, length, half_dim).to(dtype=dtype)
        sin = torch.sin(angles).view(1, 1, length, half_dim).to(dtype=dtype)
        return cos, sin

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def _rope_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.shape
        q = self.q_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope_frequencies(length, self.head_dim, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, length, self.dim)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        return out

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm_attn(x)
        if self.position_encoding_type == "sinusoidal":
            pos = self._sinusoidal_position_encoding(
                length=x_norm.shape[1],
                dim=x_norm.shape[2],
                device=x_norm.device,
                dtype=x_norm.dtype,
            )
            x_norm = x_norm + self.pos_scale * pos
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        elif self.position_encoding_type == "rope":
            attn_out = self._rope_attention(x_norm)
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        bsz, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = int(np.prod(dims))
        x_flat = x.reshape(bsz, d_model, n_tokens).transpose(-1, -2)
        out = self._transform(x_flat)
        return out.transpose(-1, -2).reshape(bsz, d_model, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        out = self._transform(x_flat)
        return out.reshape(bsz, n_tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class ResidualTransformerEncoder(nn.Module):
    """Residual encoder mirroring IQUMamba1D with Transformer blocks."""

    def __init__(
        self,
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
        pool_type: str = "conv",
        transformer_n_heads: int = 4,
        transformer_dropout: float = 0.0,
        transformer_ffn_expand: int = 4,
        transformer_token_layout: str = "adaptive",
        transformer_pos_encoding: str = "sinusoidal",
    ):
        super().__init__()
        del pool_type

        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        token_layout = str(transformer_token_layout).lower()
        if token_layout not in {"adaptive", "patch", "channel"}:
            raise ValueError(
                f"Unsupported transformer_token_layout='{transformer_token_layout}'. "
                "Expected one of: adaptive, patch, channel."
            )

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i / j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if token_layout == "channel":
                do_channel_token[s] = True
            elif token_layout == "adaptive" and np.prod(feature_map_size) <= features_per_stage[s]:
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
            *[
                BasicBlockD(
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
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ],
        )

        input_channels = stem_channels
        stages = []
        transformer_layers = []
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
                *[
                    BasicBlockD(
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
                    )
                    for _ in range(n_blocks_per_stage[s] - 1)
                ],
            )

            if bool(s % 2) ^ bool(n_stages % 2):
                transformer_layers.append(
                    TransformerLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        n_heads=transformer_n_heads,
                        dropout=transformer_dropout,
                        ffn_expand=transformer_ffn_expand,
                        channel_token=do_channel_token[s],
                        position_encoding_type=transformer_pos_encoding,
                    )
                )
            else:
                transformer_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.transformer_layers = nn.ModuleList(transformer_layers)
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

    def forward(self, x: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.transformer_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUTransformer1D(nn.Module):
    """Transformer version of IQUMamba1D with the same public interface."""

    def __init__(
        self,
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
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        transformer_n_heads: int = 4,
        transformer_dropout: float = 0.0,
        transformer_ffn_expand: int = 4,
        transformer_token_layout: str = "adaptive",
        transformer_pos_encoding: str = "sinusoidal",
    ):
        super().__init__()
        self.encoder = ResidualTransformerEncoder(
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
            transformer_n_heads=transformer_n_heads,
            transformer_dropout=transformer_dropout,
            transformer_ffn_expand=transformer_ffn_expand,
            transformer_token_layout=transformer_token_layout,
            transformer_pos_encoding=transformer_pos_encoding,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)
