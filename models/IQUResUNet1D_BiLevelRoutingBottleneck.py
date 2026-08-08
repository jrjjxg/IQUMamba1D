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


class BiLevelRoutingAttention1D(nn.Module):
    """1D temporal-region adaptation of BiFormer's bi-level routing attention.

    Reference: https://github.com/rayleizhu/BiFormer
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        routing_segments: int = 16,
        routing_topk: int = 4,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.routing_segments = max(1, int(routing_segments))
        self.topk = max(1, int(routing_topk))

        self.norm = nn.GroupNorm(1, self.channels)
        self.qkv = nn.Conv1d(self.channels, self.channels * 3, kernel_size=1)
        self.lepe = nn.Conv1d(self.channels, self.channels, kernel_size=3, padding=1, groups=self.channels)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(self.channels, self.channels, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))

    def _pad_to_segments(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        t = x.size(-1)
        segment_count = min(self.routing_segments, max(1, t))
        segment_len = (t + segment_count - 1) // segment_count
        padded_len = segment_count * segment_len
        pad = padded_len - t
        if pad:
            x = F.pad(x, (0, pad))
        return x, segment_count, segment_len, pad

    def forward(self, _raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, original_t = x.shape
        normalized, segment_count, segment_len, pad = self._pad_to_segments(self.norm(x))
        padded_t = normalized.size(-1)

        q, k, v = self.qkv(normalized).chunk(3, dim=1)
        q_region = q.detach().reshape(b, c, segment_count, segment_len).mean(dim=-1).transpose(1, 2)
        k_region = k.detach().reshape(b, c, segment_count, segment_len).mean(dim=-1)
        region_scores = (q_region @ k_region) * (c ** -0.5)

        topk = min(self.topk, segment_count)
        _, route_index = torch.topk(region_scores, k=topk, dim=-1)
        route_index = route_index[:, None, :, :, None, None]

        q = q.reshape(b, self.num_heads, self.head_dim, segment_count, segment_len).permute(0, 1, 3, 4, 2)
        k = k.reshape(b, self.num_heads, self.head_dim, segment_count, segment_len).permute(0, 1, 3, 4, 2)
        v_heads = v.reshape(b, self.num_heads, self.head_dim, segment_count, segment_len).permute(0, 1, 3, 4, 2)

        gather_index = route_index.expand(b, self.num_heads, segment_count, topk, segment_len, self.head_dim)
        k_bank = k[:, :, None, :, :, :].expand(b, self.num_heads, segment_count, segment_count, segment_len, self.head_dim)
        v_bank = v_heads[:, :, None, :, :, :].expand_as(k_bank)
        routed_k = torch.gather(k_bank, dim=3, index=gather_index)
        routed_v = torch.gather(v_bank, dim=3, index=gather_index)

        logits = torch.einsum("bhsld,bhskmd->bhslkm", q, routed_k) * self.scale
        attn = F.softmax(logits.reshape(b, self.num_heads, segment_count, segment_len, topk * segment_len), dim=-1)
        attn = self.attn_drop(attn).reshape(b, self.num_heads, segment_count, segment_len, topk, segment_len)
        out = torch.einsum("bhslkm,bhskmd->bhsld", attn, routed_v)

        out = out.permute(0, 1, 4, 2, 3).contiguous().reshape(b, c, padded_t)
        out = out + self.lepe(v)
        out = self.proj(out)
        out = self.proj_drop(out)
        if pad:
            out = out[..., :original_t]
        return residual + self.gamma * out


class IQUResUNet1D_BiLevelRoutingBottleneck(BaseBottleneckInnovationResUNet1D):
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
        routing_segments: int = 16,
        routing_topk: int = 4,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        bottleneck = BiLevelRoutingAttention1D(
            channels=int(features_per_stage[-1]),
            num_heads=num_heads,
            routing_segments=routing_segments,
            routing_topk=routing_topk,
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
