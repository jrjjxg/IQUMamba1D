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


def linear_sample_1d(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Differentiable 1D linear sampling for DAT-style deformed K/V points."""
    b, h, d, source_t = x.shape
    _, _, query_t, points = grid.shape

    position = grid.clamp(0, source_t - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=source_t - 1)
    weight = (position - left.to(position.dtype)).unsqueeze(-1)

    x_flat = x.reshape(b * h * d, source_t)
    left_index = left[:, :, None, :, :].expand(b, h, d, query_t, points).reshape(b * h * d, query_t * points)
    right_index = right[:, :, None, :, :].expand(b, h, d, query_t, points).reshape(b * h * d, query_t * points)

    left_values = torch.gather(x_flat, dim=1, index=left_index)
    right_values = torch.gather(x_flat, dim=1, index=right_index)
    left_values = left_values.reshape(b, h, d, query_t, points).permute(0, 1, 3, 4, 2)
    right_values = right_values.reshape(b, h, d, query_t, points).permute(0, 1, 3, 4, 2)
    return left_values * (1.0 - weight) + right_values * weight


class DeformableTemporalAttention1D(nn.Module):
    """1D DAT bottleneck with learned temporal offsets and sampled K/V points.

    Reference: https://github.com/LeapLabTHU/DAT
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        deform_points: int = 8,
        offset_kernel_size: int = 5,
        offset_range: float = 8.0,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.deform_points = max(1, int(deform_points))
        self.offset_range = float(offset_range)
        self.scale = self.head_dim ** -0.5

        offset_kernel_size = max(3, int(offset_kernel_size))
        if offset_kernel_size % 2 == 0:
            offset_kernel_size += 1

        self.norm = nn.GroupNorm(1, self.channels)
        self.qkv = nn.Conv1d(self.channels, self.channels * 3, kernel_size=1)
        self.offset_net = nn.Sequential(
            nn.Conv1d(self.channels, self.channels, kernel_size=offset_kernel_size, padding=offset_kernel_size // 2, groups=self.channels),
            nn.GELU(),
            nn.Conv1d(self.channels, self.num_heads * self.deform_points, kernel_size=1),
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(self.channels, self.channels, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))

    def forward(self, _raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t = x.shape
        normalized = self.norm(x)
        q, k, v = self.qkv(normalized).chunk(3, dim=1)

        offset = torch.tanh(self.offset_net(q)).reshape(b, self.num_heads, self.deform_points, t)
        offset = offset.permute(0, 1, 3, 2) * self.offset_range
        reference = torch.arange(t, device=x.device, dtype=x.dtype).view(1, 1, t, 1)
        grid = reference + offset

        q_heads = q.reshape(b, self.num_heads, self.head_dim, t).permute(0, 1, 3, 2)
        k_heads = k.reshape(b, self.num_heads, self.head_dim, t)
        v_heads = v.reshape(b, self.num_heads, self.head_dim, t)

        sampled_k = linear_sample_1d(k_heads, grid)
        sampled_v = linear_sample_1d(v_heads, grid)
        logits = (q_heads.unsqueeze(3) * sampled_k).sum(dim=-1) * self.scale
        attn = F.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn.unsqueeze(-1) * sampled_v).sum(dim=3)
        out = out.permute(0, 1, 3, 2).contiguous().reshape(b, c, t)
        out = self.proj_drop(self.proj(out))
        return residual + self.gamma * out


class IQUResUNet1D_DeformableTemporalBottleneck(BaseBottleneckInnovationResUNet1D):
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
        deform_points: int = 8,
        offset_kernel_size: int = 5,
        offset_range: float = 8.0,
        attn_drop: float = 0.05,
        proj_drop: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        bottleneck = DeformableTemporalAttention1D(
            channels=int(features_per_stage[-1]),
            num_heads=num_heads,
            deform_points=deform_points,
            offset_kernel_size=offset_kernel_size,
            offset_range=offset_range,
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
