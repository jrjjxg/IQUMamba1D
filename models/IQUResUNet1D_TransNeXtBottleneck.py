from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D_InnovationBase import BaseBottleneckInnovationResUNet1D


def _valid_heads(channels: int, requested_heads: int) -> int:
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class ConvolutionalGLU1D(nn.Module):
    """1D version of TransNeXt's ConvolutionalGLU feed-forward block."""

    def __init__(self, channels: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        hidden_features = max(8, int(2 * hidden_features / 3))
        self.fc1 = nn.Linear(channels, hidden_features * 2)
        self.dwconv = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x.transpose(1, 2)).transpose(1, 2)) * v
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class TransNeXtAggregatedAttention1D(nn.Module):
    """Pooled-key 1D adaptation of TransNeXt aggregated attention + ConvGLU.

    Reference: https://github.com/DaiShiResearch/TransNeXt
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        sr_ratio: int = 4,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.sr_ratio = max(1, int(sr_ratio))

        self.norm1 = nn.LayerNorm(self.channels)
        self.q = nn.Linear(self.channels, self.channels)
        self.kv = nn.Linear(self.channels, self.channels * 2)
        self.query_embedding = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))
        temperature_init = torch.ones(1, self.num_heads, 1, 1) / 0.24
        self.temperature = nn.Parameter(torch.log(torch.exp(temperature_init) - 1.0))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.channels, self.channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = nn.LayerNorm(self.channels)
        self.mlp = ConvolutionalGLU1D(
            channels=self.channels,
            hidden_features=int(self.channels * float(mlp_ratio)),
            drop=proj_drop,
        )
        self.gamma_attn = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))
        self.gamma_mlp = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))

    def forward(self, _raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        residual = x
        tokens = self.norm1(x.transpose(1, 2))

        pooled_len = max(1, t // self.sr_ratio)
        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), pooled_len).transpose(1, 2)

        q = self.q(tokens).reshape(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(pooled).reshape(b, pooled_len, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        seq_length_scale = torch.log(torch.as_tensor(float(t), device=x.device, dtype=x.dtype)).clamp_min(1.0)
        temperature = F.softplus(self.temperature)
        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * temperature * seq_length_scale)
        attn = attn @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, t, c)
        out = self.proj(out)
        out = self.proj_drop(out).transpose(1, 2)
        x = residual + self.gamma_attn * out

        mlp_tokens = self.norm2(x.transpose(1, 2))
        mlp_out = self.mlp(mlp_tokens).transpose(1, 2)
        return x + self.gamma_mlp * mlp_out


class IQUResUNet1D_TransNeXtBottleneck(BaseBottleneckInnovationResUNet1D):
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
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        num_heads: int = 8,
        sr_ratio: int = 4,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        bottleneck = TransNeXtAggregatedAttention1D(
            channels=int(features_per_stage[-1]),
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            bottleneck_scale_init=bottleneck_scale_init,
        )
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            bottleneck=bottleneck,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
