"""Independent Stage-235 cross-attention follow-up ablations."""

from __future__ import annotations

import math
from typing import List, Sequence, Type

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUBiMamba1D_CrossScaleAttention import (
    CompressedGlobalCrossAttention,
    IQUBiMamba1D_CrossScaleAttention,
)


class TemporallyAlignedCrossAttention(CompressedGlobalCrossAttention):
    """Cross-attention with aligned local KV windows and global summary tokens."""

    def __init__(
        self,
        *args,
        window_radius: int = 4,
        global_summary_tokens: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(window_radius) < 0:
            raise ValueError(f"window_radius must be non-negative, got {window_radius}")
        if int(global_summary_tokens) < 1:
            raise ValueError(
                f"global_summary_tokens must be positive, got {global_summary_tokens}"
            )
        self.window_radius = int(window_radius)
        self.global_summary_tokens = int(global_summary_tokens)

    def _compress_aligned_global(
        self,
        global_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        local_count = min(self.kv_tokens, int(global_feature.size(-1)))
        local_tokens = F.adaptive_avg_pool1d(global_feature, local_count)
        summary_count = min(self.global_summary_tokens, int(global_feature.size(-1)))
        summary_tokens = F.adaptive_avg_pool1d(global_feature, summary_count)
        tokens = torch.cat([local_tokens, summary_tokens], dim=-1).transpose(1, 2)
        return self.global_norm(tokens), local_count

    def _attention_mask(
        self,
        query_count: int,
        local_count: int,
        total_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_count, device=device, dtype=torch.float32)
        if query_count > 1 and local_count > 1:
            centers = torch.round(
                query_positions * float(local_count - 1) / float(query_count - 1)
            ).to(dtype=torch.long)
        else:
            centers = torch.zeros(query_count, device=device, dtype=torch.long)
        key_positions = torch.arange(local_count, device=device).view(1, -1)
        local_allowed = (key_positions - centers.view(-1, 1)).abs() <= self.window_radius
        summary_allowed = torch.ones(
            query_count,
            total_count - local_count,
            dtype=torch.bool,
            device=device,
        )
        return ~torch.cat([local_allowed, summary_allowed], dim=1)

    def forward(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query_norm(query_feature.transpose(1, 2))
        global_tokens, local_count = self._compress_aligned_global(global_feature)
        key = self.key_proj(global_tokens)
        value = self.value_proj(global_tokens)
        mask = self._attention_mask(
            query_count=int(query.size(1)),
            local_count=local_count,
            total_count=int(global_tokens.size(1)),
            device=query.device,
        )
        delta, _ = self.attention(
            query,
            key,
            value,
            attn_mask=mask,
            need_weights=False,
        )
        delta = self.output_norm(delta).transpose(1, 2)
        scale = self.residual_scale
        if gate is not None:
            scale = scale * gate.to(dtype=delta.dtype).view(-1, 1, 1)
        return query_feature + scale * delta


class MultiResolutionKVCrossAttention(nn.Module):
    """Fuse coarse and fine compressed KV banks for one query scale."""

    def __init__(
        self,
        query_channels: int,
        global_channels: int,
        coarse_kv_tokens: int = 32,
        fine_kv_tokens: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float = 0.01,
        gate_hidden: int = 32,
    ) -> None:
        super().__init__()
        if int(coarse_kv_tokens) >= int(fine_kv_tokens):
            raise ValueError("coarse_kv_tokens must be smaller than fine_kv_tokens")
        common = {
            "query_channels": int(query_channels),
            "global_channels": int(global_channels),
            "num_heads": int(num_heads),
            "dropout": float(dropout),
            "residual_scale_init": float(residual_scale_init),
        }
        self.coarse_attention = CompressedGlobalCrossAttention(
            **common,
            kv_tokens=int(coarse_kv_tokens),
        )
        self.fine_attention = CompressedGlobalCrossAttention(
            **common,
            kv_tokens=int(fine_kv_tokens),
        )
        self.route_norm = nn.LayerNorm(int(global_channels))
        self.route = nn.Sequential(
            nn.Linear(int(global_channels), int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), 2),
        )
        nn.init.zeros_(self.route[-1].weight)
        nn.init.zeros_(self.route[-1].bias)

    def forward(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coarse = self.coarse_attention(query_feature, global_feature) - query_feature
        fine = self.fine_attention(query_feature, global_feature) - query_feature
        route_input = self.route_norm(global_feature.mean(dim=-1))
        weights = torch.softmax(self.route(route_input), dim=-1)
        delta = (
            weights[:, 0].view(-1, 1, 1) * coarse
            + weights[:, 1].view(-1, 1, 1) * fine
        )
        if gate is not None:
            delta = delta * gate.to(dtype=delta.dtype).view(-1, 1, 1)
        return query_feature + delta


class BoundedChannelGatedCrossAttention(nn.Module):
    """Apply a bounded, sample-conditioned gate to each query channel."""

    def __init__(
        self,
        query_channels: int,
        global_channels: int,
        kv_tokens: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_scale: float = 0.1,
        initial_scale: float = 0.01,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        if not 0.0 < float(initial_scale) < float(max_scale):
            raise ValueError("initial_scale must be positive and smaller than max_scale")
        self.max_scale = float(max_scale)
        self.initial_scale = float(initial_scale)
        self.attention_core = CompressedGlobalCrossAttention(
            query_channels=int(query_channels),
            global_channels=int(global_channels),
            kv_tokens=int(kv_tokens),
            num_heads=int(num_heads),
            dropout=float(dropout),
            residual_scale_init=1.0,
        )
        condition_channels = int(query_channels) + int(global_channels)
        self.gate_norm = nn.LayerNorm(condition_channels)
        self.channel_gate = nn.Sequential(
            nn.Linear(condition_channels, int(gate_hidden)),
            nn.GELU(),
            nn.Linear(int(gate_hidden), int(query_channels)),
        )
        probability = self.initial_scale / self.max_scale
        initial_logit = math.log(probability / (1.0 - probability))
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.constant_(self.channel_gate[-1].bias, initial_logit)

    def gate_values(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        condition = torch.cat(
            [query_feature.mean(dim=-1), global_feature.mean(dim=-1)],
            dim=1,
        )
        logits = self.channel_gate(self.gate_norm(condition))
        return self.max_scale * torch.sigmoid(logits)

    def forward(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.attention_core(query_feature, global_feature) - query_feature
        channel_scale = self.gate_values(query_feature, global_feature).unsqueeze(-1)
        if gate is not None:
            channel_scale = channel_scale * gate.to(dtype=delta.dtype).view(-1, 1, 1)
        return query_feature + channel_scale.to(dtype=delta.dtype) * delta


class IQUBiMamba1D_AdvancedCrossScaleAttention(IQUBiMamba1D_CrossScaleAttention):
    """Stage-235 backbone with one independently selected fusion refinement."""

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
        cross_scale_variant: str = "aligned",
        cross_scale_query_stages: Sequence[int] = (2,),
        cross_scale_global_stage: int = 3,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 0.01,
        cross_scale_aligned_window_radius: int = 4,
        cross_scale_aligned_global_tokens: int = 1,
        cross_scale_coarse_kv_tokens: int = 32,
        cross_scale_fine_kv_tokens: int = 128,
        cross_scale_multires_gate_hidden: int = 32,
        cross_scale_bounded_max_scale: float = 0.1,
        cross_scale_bounded_initial_scale: float = 0.01,
        cross_scale_channel_gate_hidden: int = 64,
    ) -> None:
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
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            cross_scale_query_stages=cross_scale_query_stages,
            cross_scale_global_stage=cross_scale_global_stage,
            cross_scale_kv_tokens=cross_scale_kv_tokens,
            cross_scale_num_heads=cross_scale_num_heads,
            cross_scale_dropout=cross_scale_dropout,
            cross_scale_residual_scale_init=cross_scale_residual_scale_init,
            cross_scale_evidence_gate=False,
        )
        if len(self.cross_scale_query_stages) != 1:
            raise ValueError("Advanced Stage-235 ablations require exactly one query stage")

        self.cross_scale_variant = str(cross_scale_variant)
        stage = self.cross_scale_query_stages[0]
        query_channels = int(features_per_stage[stage])
        global_channels = int(features_per_stage[self.cross_scale_global_stage])
        common = {
            "query_channels": query_channels,
            "global_channels": global_channels,
            "num_heads": int(cross_scale_num_heads),
            "dropout": float(cross_scale_dropout),
        }

        if self.cross_scale_variant == "aligned":
            replacement = TemporallyAlignedCrossAttention(
                **common,
                kv_tokens=int(cross_scale_kv_tokens),
                residual_scale_init=float(cross_scale_residual_scale_init),
                window_radius=int(cross_scale_aligned_window_radius),
                global_summary_tokens=int(cross_scale_aligned_global_tokens),
            )
        elif self.cross_scale_variant == "multires_kv":
            replacement = MultiResolutionKVCrossAttention(
                **common,
                coarse_kv_tokens=int(cross_scale_coarse_kv_tokens),
                fine_kv_tokens=int(cross_scale_fine_kv_tokens),
                residual_scale_init=float(cross_scale_residual_scale_init),
                gate_hidden=int(cross_scale_multires_gate_hidden),
            )
        elif self.cross_scale_variant == "bounded_channel_gate":
            replacement = BoundedChannelGatedCrossAttention(
                **common,
                kv_tokens=int(cross_scale_kv_tokens),
                max_scale=float(cross_scale_bounded_max_scale),
                initial_scale=float(cross_scale_bounded_initial_scale),
                gate_hidden=int(cross_scale_channel_gate_hidden),
            )
        else:
            raise ValueError(f"Unsupported cross_scale_variant: {self.cross_scale_variant}")

        self.cross_scale_blocks[str(stage)] = replacement

