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


class AgentAttention1D(nn.Module):
    """1D adaptation of Agent Attention's agent-token aggregate/broadcast path.

    Reference: https://github.com/LeapLabTHU/Agent-Attention
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        agent_tokens: int = 64,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.agent_tokens = int(agent_tokens)

        self.norm = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, self.channels * 3)
        self.pool = nn.AdaptiveAvgPool1d(self.agent_tokens)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.channels, self.channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dwc = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels)
        self.gamma = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))

    def forward(self, _raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t = x.shape

        tokens = self.norm(x.transpose(1, 2))
        qkv = self.qkv(tokens).reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_tokens = q.transpose(1, 2).reshape(b, t, c)
        agent_tokens = self.pool(q_tokens.transpose(1, 2)).transpose(1, 2)
        agent_len = agent_tokens.size(1)
        agent_tokens = agent_tokens.reshape(b, agent_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        agent_attn = F.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1), dim=-1)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        q_attn = F.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1), dim=-1)
        q_attn = self.attn_drop(q_attn)
        out = q_attn @ agent_v
        out = out.transpose(1, 2).reshape(b, t, c)

        local_v = v.transpose(1, 2).reshape(b, t, c).transpose(1, 2)
        out = out + self.dwc(local_v).transpose(1, 2)
        out = self.proj(out)
        out = self.proj_drop(out).transpose(1, 2)
        return residual + self.gamma * out


class IQUResUNet1D_AgentAttentionBottleneck(BaseBottleneckInnovationResUNet1D):
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
        agent_tokens: int = 64,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        bottleneck = AgentAttention1D(
            channels=int(features_per_stage[-1]),
            num_heads=num_heads,
            agent_tokens=agent_tokens,
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
