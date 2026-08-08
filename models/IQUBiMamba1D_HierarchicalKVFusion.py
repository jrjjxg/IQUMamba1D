"""Hierarchical combinations of Stages 235, 244 and 245.

Stages 248-250 keep the Stage-12 encoder and decoder unchanged.  They combine
three complementary operations before decoding:

* bottleneck self-attention strengthens the deepest semantic memory;
* compressed global K/V provides top-down context to the Stage-2 skip;
* compact RF physical K/V provides cyclic, CFO, polyphase and phase evidence.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUBiMamba1D import IQUBiMamba1D
from models.IQUBiMamba1D_CrossScaleAttention import CompressedGlobalCrossAttention
from models.IQUBiMamba1D_KVAttentionAblations import (
    BottleneckSelfAttention1D,
    CompactPhysicalCrossAttention,
    TimeDomainPhysicalTokenExtractor,
)


class DualMemoryCrossAttentionFusion(nn.Module):
    """Add the original Stage-235 and Stage-244 residual updates in parallel."""

    def __init__(
        self,
        query_channels: int,
        global_channels: int,
        global_kv_tokens: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        global_scale_init: float = 0.01,
        physical_scale_init: float = 0.01,
        physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.physical_token_extractor = TimeDomainPhysicalTokenExtractor(
            cyclic_lags=physical_cyclic_lags,
            polyphase_branches=physical_polyphase_branches,
            symbol_orders=physical_symbol_orders,
            min_cyclic_freq=physical_min_cyclic_freq,
            max_cyclic_freq=physical_max_cyclic_freq,
            cyclic_temperature=physical_cyclic_temperature,
        )
        self.global_attention = CompressedGlobalCrossAttention(
            query_channels=int(query_channels),
            global_channels=int(global_channels),
            kv_tokens=int(global_kv_tokens),
            num_heads=int(num_heads),
            dropout=float(dropout),
            residual_scale_init=float(global_scale_init),
        )
        self.physical_attention = CompactPhysicalCrossAttention(
            query_channels=int(query_channels),
            token_dim=self.physical_token_extractor.token_dim,
            token_count=self.physical_token_extractor.token_count,
            num_heads=int(num_heads),
            dropout=float(dropout),
            residual_scale_init=float(physical_scale_init),
        )

    def compute_components(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physical_tokens = self.physical_token_extractor(mixture)
        global_delta = self.global_attention.compute_delta(query_feature, global_feature)
        physical_delta = self.physical_attention.compute_delta(query_feature, physical_tokens)
        return global_delta, physical_delta, physical_tokens

    def forward(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        global_delta, physical_delta, _ = self.compute_components(
            query_feature, global_feature, mixture
        )
        global_scale = self.global_attention.residual_scale.to(
            device=global_delta.device, dtype=global_delta.dtype
        )
        physical_scale = self.physical_attention.residual_scale.to(
            device=physical_delta.device, dtype=physical_delta.dtype
        )
        return query_feature + global_scale * global_delta + physical_scale * physical_delta


class PhysicalReliabilityRouter(nn.Module):
    """Convert mixture-derived physical tokens into bounded channel gates."""

    def __init__(
        self,
        token_dim: int,
        output_channels: int,
        hidden_channels: int = 64,
        gate_init: float = 1.0,
        gate_max: float = 2.0,
    ) -> None:
        super().__init__()
        if not 0.0 < float(gate_init) < float(gate_max):
            raise ValueError("gate_init must be strictly between zero and gate_max")
        self.gate_max = float(gate_max)
        summary_dim = 4 * int(token_dim)
        self.net = nn.Sequential(
            nn.Linear(summary_dim, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), int(output_channels)),
        )
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        initial_ratio = float(gate_init) / self.gate_max
        nn.init.constant_(final.bias, math.log(initial_ratio / (1.0 - initial_ratio)))

    @staticmethod
    def summarize(tokens: torch.Tensor) -> torch.Tensor:
        # Keep raw token magnitudes: reliability and power are meaningful here.
        mean = tokens.mean(dim=1)
        std = tokens.std(dim=1, unbiased=False)
        global_tokens = tokens[:, -2:].reshape(tokens.size(0), -1)
        return torch.cat([mean, std, global_tokens], dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.net(self.summarize(tokens.float()))
        return self.gate_max * torch.sigmoid(logits)


class BoundedChannelScale(nn.Module):
    """Learnable per-channel residual strength constrained to a stable range."""

    def __init__(self, channels: int, scale_init: float = 0.1, scale_max: float = 0.5):
        super().__init__()
        if not 0.0 < float(scale_init) < float(scale_max):
            raise ValueError("scale_init must be strictly between zero and scale_max")
        self.scale_max = float(scale_max)
        ratio = float(scale_init) / self.scale_max
        raw_init = math.log(ratio / (1.0 - ratio))
        self.raw = nn.Parameter(torch.full((int(channels),), raw_init))

    def values(self) -> torch.Tensor:
        return self.scale_max * torch.sigmoid(self.raw)


class PhysicalFeatureFiLM(nn.Module):
    """Bounded FiLM parameters derived from raw physical-token summaries."""

    def __init__(self, token_dim: int, channels: int, hidden_channels: int = 64,
                 max_delta: float = 0.1) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.Linear(4 * int(token_dim), int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), 2 * int(channels)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def parameters_for(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(PhysicalReliabilityRouter.summarize(tokens.float()))
        gamma, beta = params.chunk(2, dim=-1)
        return self.max_delta * torch.tanh(gamma), self.max_delta * torch.tanh(beta)

    def forward(self, feature: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.parameters_for(tokens)
        gamma = gamma.to(device=feature.device, dtype=feature.dtype).unsqueeze(-1)
        beta = beta.to(device=feature.device, dtype=feature.dtype).unsqueeze(-1)
        return feature * (1.0 + gamma) + beta


class UnifiedGlobalPhysicalCrossAttention(nn.Module):
    """Let one attention operation select between global and physical K/V."""

    def __init__(self, query_channels: int, global_channels: int, physical_dim: int,
                 kv_tokens: int = 64, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        channels = int(query_channels)
        if channels % int(num_heads) != 0:
            raise ValueError("query_channels must be divisible by num_heads")
        self.kv_tokens = int(kv_tokens)
        self.query_norm = nn.LayerNorm(channels)
        self.global_norm = nn.LayerNorm(int(global_channels))
        self.key_proj = nn.Linear(int(global_channels), channels, bias=False)
        self.value_proj = nn.Linear(int(global_channels), channels, bias=False)
        self.physical_key_proj = nn.Linear(int(physical_dim), channels, bias=False)
        self.physical_value_proj = nn.Linear(int(physical_dim), channels, bias=False)
        self.key_type = nn.Parameter(torch.zeros(2, channels))
        self.value_type = nn.Parameter(torch.zeros(2, channels))
        self.physical_value_gate = PhysicalReliabilityRouter(
            token_dim=int(physical_dim), output_channels=channels,
            hidden_channels=max(16, channels // 2), gate_init=1.0, gate_max=2.0,
        )
        self.attention = nn.MultiheadAttention(
            channels, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.output_norm = nn.LayerNorm(channels)

    def compute_delta(self, query_feature: torch.Tensor, global_feature: torch.Tensor,
                      physical_tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(query_feature.transpose(1, 2))
        count = min(self.kv_tokens, int(global_feature.size(-1)))
        global_tokens = self.global_norm(
            F.adaptive_avg_pool1d(global_feature, count).transpose(1, 2)
        )
        global_key = self.key_proj(global_tokens) + self.key_type[0]
        global_value = self.value_proj(global_tokens) + self.value_type[0]
        physical = physical_tokens.to(dtype=query.dtype)
        physical_key = self.physical_key_proj(physical) + self.key_type[1]
        physical_value = self.physical_value_proj(physical) + self.value_type[1]
        value_gate = self.physical_value_gate(physical_tokens).to(
            device=query.device, dtype=query.dtype
        )
        physical_value = physical_value * value_gate.unsqueeze(1)
        key = torch.cat([global_key, physical_key], dim=1)
        value = torch.cat([global_value, physical_value], dim=1)
        delta, _ = self.attention(query, key, value, need_weights=False)
        return self.output_norm(delta).transpose(1, 2)


class PhysicalExpertRouter(nn.Module):
    """Route identity/global/physical/joint experts with an identity prior."""

    def __init__(self, token_dim: int, hidden_channels: int = 64,
                 prior: Sequence[float] = (0.7, 0.1, 0.1, 0.1)) -> None:
        super().__init__()
        prior_tensor = torch.tensor(tuple(float(value) for value in prior))
        if prior_tensor.numel() != 4 or bool((prior_tensor <= 0).any()):
            raise ValueError("expert prior must contain four positive values")
        prior_tensor = prior_tensor / prior_tensor.sum()
        self.net = nn.Sequential(
            nn.Linear(4 * int(token_dim), int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(prior_tensor.log())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.net(PhysicalReliabilityRouter.summarize(tokens.float()))
        return torch.softmax(logits, dim=-1)


class MixtureConditionEncoder(nn.Module):
    """Learn a compact mixture-only noise embedding and an auxiliary SNR estimate."""

    def __init__(self, hidden_channels: int = 16, embedding_dim: int = 16) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        embedding = int(embedding_dim)
        self.features = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * hidden, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(4 * hidden, embedding),
            nn.LayerNorm(embedding),
            nn.GELU(),
        )
        self.snr_head = nn.Linear(embedding, 1)

    def forward(self, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(mixture.float())
        pooled = torch.cat(
            [features.mean(dim=-1), features.std(dim=-1, unbiased=False)], dim=-1
        )
        embedding = self.embedding(pooled)
        return embedding, self.snr_head(embedding).squeeze(-1)


class IdentityAwareEvidenceRouter(nn.Module):
    """Route four residual experts using features, condition and expert agreement."""

    def __init__(
        self,
        token_dim: int,
        query_channels: int,
        global_channels: int,
        condition_dim: int,
        hidden_channels: int = 64,
        prior: Sequence[float] = (0.7, 0.1, 0.1, 0.1),
        trust_penalty_init: float = 0.1,
    ) -> None:
        super().__init__()
        prior_tensor = torch.tensor(tuple(float(value) for value in prior))
        if prior_tensor.numel() != 4 or bool((prior_tensor <= 0).any()):
            raise ValueError("expert prior must contain four positive values")
        if float(trust_penalty_init) <= 0.0:
            raise ValueError("trust_penalty_init must be positive")
        prior_tensor = prior_tensor / prior_tensor.sum()
        hidden = int(hidden_channels)
        self.physical_proj = nn.Linear(4 * int(token_dim), hidden)
        self.query_proj = nn.Linear(2 * int(query_channels), hidden)
        self.global_proj = nn.Linear(2 * int(global_channels), hidden)
        self.condition_proj = nn.Linear(int(condition_dim), hidden)
        self.net = nn.Sequential(
            nn.Linear(4 * hidden + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(int(condition_dim) + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.uncertainty_net[-1].weight)
        nn.init.zeros_(self.uncertainty_net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias.copy_(prior_tensor.log())
        raw_penalty = math.log(math.expm1(float(trust_penalty_init)))
        self.raw_trust_penalty = nn.Parameter(torch.full((2,), raw_penalty))

    @staticmethod
    def _temporal_summary(feature: torch.Tensor) -> torch.Tensor:
        feature = feature.float()
        return torch.cat(
            [feature.mean(dim=-1), feature.std(dim=-1, unbiased=False)], dim=-1
        )

    @staticmethod
    def _agreement_evidence(
        global_delta: torch.Tensor, physical_delta: torch.Tensor
    ) -> torch.Tensor:
        global_flat = global_delta.float().flatten(1)
        physical_flat = physical_delta.float().flatten(1)
        global_rms = global_flat.square().mean(dim=1).clamp_min(1e-8).sqrt()
        physical_rms = physical_flat.square().mean(dim=1).clamp_min(1e-8).sqrt()
        cosine = F.cosine_similarity(global_flat, physical_flat, dim=1, eps=1e-8)
        conflict = 0.5 * (1.0 - cosine).clamp(0.0, 2.0)
        log_ratio = torch.log(physical_rms / global_rms).clamp(-4.0, 4.0) / 4.0
        return torch.stack(
            [torch.log1p(global_rms), torch.log1p(physical_rms), log_ratio, conflict],
            dim=-1,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor,
        global_feature: torch.Tensor,
        global_delta: torch.Tensor,
        physical_delta: torch.Tensor,
        condition_embedding: torch.Tensor,
        condition_enabled: bool = True,
        evidence_context_enabled: bool = True,
        trust_penalty_enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        agreement = self._agreement_evidence(global_delta, physical_delta)
        if condition_enabled:
            condition_features = F.gelu(self.condition_proj(condition_embedding.float()))
        else:
            condition_features = condition_embedding.new_zeros(
                condition_embedding.size(0), self.condition_proj.out_features
            )
        if evidence_context_enabled:
            query_features = F.gelu(self.query_proj(self._temporal_summary(query)))
            global_features = F.gelu(
                self.global_proj(self._temporal_summary(global_feature))
            )
            routed_agreement = agreement
        else:
            query_features = query.new_zeros(
                query.size(0), self.query_proj.out_features, dtype=torch.float32
            )
            global_features = global_feature.new_zeros(
                global_feature.size(0), self.global_proj.out_features, dtype=torch.float32
            )
            routed_agreement = agreement.new_zeros(agreement.shape)
        routed = torch.cat(
            [
                F.gelu(self.physical_proj(PhysicalReliabilityRouter.summarize(tokens.float()))),
                query_features,
                global_features,
                condition_features,
                routed_agreement,
            ],
            dim=-1,
        )
        logits = self.net(routed)
        uncertainty = torch.sigmoid(
            self.uncertainty_net(torch.cat([condition_embedding.float(), agreement], dim=-1))
        ).squeeze(-1)
        if trust_penalty_enabled:
            trust_penalty = F.softplus(self.raw_trust_penalty)
            logits = logits.clone()
            logits[:, 2] = logits[:, 2] - uncertainty * trust_penalty[0]
            logits[:, 3] = logits[:, 3] - uncertainty * trust_penalty[1]
        return torch.softmax(logits, dim=-1), uncertainty, agreement


def _validate_stage_pair(model: IQUBiMamba1D, query_stage: int, global_stage: int) -> None:
    stage_count = len(model.encoder.output_channels)
    if not 0 <= int(query_stage) < stage_count:
        raise ValueError(f"Invalid fusion query stage {query_stage}")
    if not 0 <= int(global_stage) < stage_count:
        raise ValueError(f"Invalid fusion global stage {global_stage}")
    if int(query_stage) == int(global_stage):
        raise ValueError("fusion query and global stages must differ")


class IQUBiMamba1D_EnhancedGlobalCrossAttention(IQUBiMamba1D):
    """Stage 248: Stage 245 enhanced bottleneck used as Stage-235 global K/V."""

    def __init__(
        self,
        *args,
        fusion_query_stage: int = 2,
        fusion_global_stage: int = 3,
        fusion_global_kv_tokens: int = 64,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.0,
        fusion_global_scale_init: float = 0.01,
        fusion_bottleneck_num_heads: int = 4,
        fusion_bottleneck_dropout: float = 0.0,
        fusion_bottleneck_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fusion_query_stage = int(fusion_query_stage)
        self.fusion_global_stage = int(fusion_global_stage)
        _validate_stage_pair(self, self.fusion_query_stage, self.fusion_global_stage)
        query_channels = int(self.encoder.output_channels[self.fusion_query_stage])
        global_channels = int(self.encoder.output_channels[self.fusion_global_stage])
        self.bottleneck_attention = BottleneckSelfAttention1D(
            channels=global_channels,
            num_heads=int(fusion_bottleneck_num_heads),
            dropout=float(fusion_bottleneck_dropout),
            residual_scale_init=float(fusion_bottleneck_scale_init),
        )
        self.global_cross_attention = CompressedGlobalCrossAttention(
            query_channels=query_channels,
            global_channels=global_channels,
            kv_tokens=int(fusion_global_kv_tokens),
            num_heads=int(fusion_num_heads),
            dropout=float(fusion_dropout),
            residual_scale_init=float(fusion_global_scale_init),
        )

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        enhanced_global = self.bottleneck_attention(skips[self.fusion_global_stage])
        skips[self.fusion_global_stage] = enhanced_global
        skips[self.fusion_query_stage] = self.global_cross_attention(
            skips[self.fusion_query_stage], enhanced_global
        )
        return self.decoder(skips)


class IQUBiMamba1D_DualMemoryCrossAttention(IQUBiMamba1D):
    """Stage 249: directly add original Stage-235 and Stage-244 updates."""

    def __init__(
        self,
        *args,
        fusion_query_stage: int = 2,
        fusion_global_stage: int = 3,
        fusion_global_kv_tokens: int = 64,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.0,
        fusion_global_scale_init: float = 0.01,
        fusion_physical_scale_init: float = 0.01,
        physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fusion_query_stage = int(fusion_query_stage)
        self.fusion_global_stage = int(fusion_global_stage)
        _validate_stage_pair(self, self.fusion_query_stage, self.fusion_global_stage)
        self.dual_memory_fusion = DualMemoryCrossAttentionFusion(
            query_channels=int(self.encoder.output_channels[self.fusion_query_stage]),
            global_channels=int(self.encoder.output_channels[self.fusion_global_stage]),
            global_kv_tokens=int(fusion_global_kv_tokens),
            num_heads=int(fusion_num_heads),
            dropout=float(fusion_dropout),
            global_scale_init=float(fusion_global_scale_init),
            physical_scale_init=float(fusion_physical_scale_init),
            physical_cyclic_lags=physical_cyclic_lags,
            physical_polyphase_branches=int(physical_polyphase_branches),
            physical_symbol_orders=physical_symbol_orders,
            physical_min_cyclic_freq=float(physical_min_cyclic_freq),
            physical_max_cyclic_freq=float(physical_max_cyclic_freq),
            physical_cyclic_temperature=float(physical_cyclic_temperature),
        )

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        skips[self.fusion_query_stage] = self.dual_memory_fusion(
            skips[self.fusion_query_stage],
            skips[self.fusion_global_stage],
            x,
        )
        return self.decoder(skips)


class IQUBiMamba1D_HierarchicalAdditiveFusion(IQUBiMamba1D):
    """Stage 250: directly add the original Stage-235, 244 and 245 updates."""

    def __init__(
        self,
        *args,
        fusion_query_stage: int = 2,
        fusion_global_stage: int = 3,
        fusion_global_kv_tokens: int = 64,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.0,
        fusion_global_scale_init: float = 0.01,
        fusion_physical_scale_init: float = 0.01,
        fusion_bottleneck_num_heads: int = 4,
        fusion_bottleneck_dropout: float = 0.0,
        fusion_bottleneck_scale_init: float = 1.0,
        physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fusion_query_stage = int(fusion_query_stage)
        self.fusion_global_stage = int(fusion_global_stage)
        _validate_stage_pair(self, self.fusion_query_stage, self.fusion_global_stage)
        global_channels = int(self.encoder.output_channels[self.fusion_global_stage])
        self.bottleneck_attention = BottleneckSelfAttention1D(
            channels=global_channels,
            num_heads=int(fusion_bottleneck_num_heads),
            dropout=float(fusion_bottleneck_dropout),
            residual_scale_init=float(fusion_bottleneck_scale_init),
        )
        self.dual_memory_fusion = DualMemoryCrossAttentionFusion(
            query_channels=int(self.encoder.output_channels[self.fusion_query_stage]),
            global_channels=global_channels,
            global_kv_tokens=int(fusion_global_kv_tokens),
            num_heads=int(fusion_num_heads),
            dropout=float(fusion_dropout),
            global_scale_init=float(fusion_global_scale_init),
            physical_scale_init=float(fusion_physical_scale_init),
            physical_cyclic_lags=physical_cyclic_lags,
            physical_polyphase_branches=int(physical_polyphase_branches),
            physical_symbol_orders=physical_symbol_orders,
            physical_min_cyclic_freq=float(physical_min_cyclic_freq),
            physical_max_cyclic_freq=float(physical_max_cyclic_freq),
            physical_cyclic_temperature=float(physical_cyclic_temperature),
        )

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        enhanced_global = self.bottleneck_attention(skips[self.fusion_global_stage])
        skips[self.fusion_global_stage] = enhanced_global
        skips[self.fusion_query_stage] = self.dual_memory_fusion(
            skips[self.fusion_query_stage], enhanced_global, x
        )
        return self.decoder(skips)


class IQUBiMamba1D_PhysicalRoutedEnhancedCrossAttention(IQUBiMamba1D):
    """Stage 251: Stage-245 memory, Stage-235 injection and Stage-244 routing."""

    def __init__(
        self,
        *args,
        fusion_query_stage: int = 2,
        fusion_global_stage: int = 3,
        fusion_global_kv_tokens: int = 64,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.0,
        fusion_channel_scale_init: float = 0.1,
        fusion_channel_scale_max: float = 0.5,
        fusion_bottleneck_num_heads: int = 4,
        fusion_bottleneck_dropout: float = 0.0,
        fusion_bottleneck_scale_init: float = 0.1,
        fusion_router_hidden: int = 64,
        fusion_router_gate_init: float = 1.0,
        fusion_router_gate_max: float = 2.0,
        physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fusion_query_stage = int(fusion_query_stage)
        self.fusion_global_stage = int(fusion_global_stage)
        _validate_stage_pair(self, self.fusion_query_stage, self.fusion_global_stage)
        query_channels = int(self.encoder.output_channels[self.fusion_query_stage])
        global_channels = int(self.encoder.output_channels[self.fusion_global_stage])

        self.bottleneck_attention = BottleneckSelfAttention1D(
            channels=global_channels,
            num_heads=int(fusion_bottleneck_num_heads),
            dropout=float(fusion_bottleneck_dropout),
            residual_scale_init=float(fusion_bottleneck_scale_init),
        )
        self.global_cross_attention = CompressedGlobalCrossAttention(
            query_channels=query_channels,
            global_channels=global_channels,
            kv_tokens=int(fusion_global_kv_tokens),
            num_heads=int(fusion_num_heads),
            dropout=float(fusion_dropout),
            residual_scale_init=None,
        )
        self.physical_token_extractor = TimeDomainPhysicalTokenExtractor(
            cyclic_lags=physical_cyclic_lags,
            polyphase_branches=int(physical_polyphase_branches),
            symbol_orders=physical_symbol_orders,
            min_cyclic_freq=float(physical_min_cyclic_freq),
            max_cyclic_freq=float(physical_max_cyclic_freq),
            cyclic_temperature=float(physical_cyclic_temperature),
        )
        self.physical_router = PhysicalReliabilityRouter(
            token_dim=self.physical_token_extractor.token_dim,
            output_channels=query_channels,
            hidden_channels=int(fusion_router_hidden),
            gate_init=float(fusion_router_gate_init),
            gate_max=float(fusion_router_gate_max),
        )
        self.channel_scale = BoundedChannelScale(
            channels=query_channels,
            scale_init=float(fusion_channel_scale_init),
            scale_max=float(fusion_channel_scale_max),
        )

    def routed_cross_scale_update(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.physical_token_extractor(mixture)
        gate = self.physical_router(tokens).to(dtype=query_feature.dtype)
        delta = self.global_cross_attention.compute_delta(query_feature, global_feature)
        scale = self.channel_scale.values().to(device=delta.device, dtype=delta.dtype)
        update = scale.view(1, -1, 1) * gate.view(gate.size(0), -1, 1) * delta
        return query_feature + update, gate, delta

    def forward(self, x: torch.Tensor):
        skips = list(self.encoder(x))
        enhanced_global = self.bottleneck_attention(skips[self.fusion_global_stage])
        skips[self.fusion_global_stage] = enhanced_global
        skips[self.fusion_query_stage], _, _ = self.routed_cross_scale_update(
            skips[self.fusion_query_stage], enhanced_global, x
        )
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        return {
            "bottleneck_attention.residual_scale",
            "channel_scale.raw",
            "physical_router.net.2.bias",
        }

    def checkpoint_prefix_aliases(self) -> tuple[tuple[str, str], ...]:
        return ((f"cross_scale_blocks.{self.fusion_query_stage}.",
                 "global_cross_attention."),)


class _PhysicalFusionBase(IQUBiMamba1D):
    """Shared Stage-245 enhancement and Stage-244 token extraction."""

    def __init__(
        self,
        *args,
        fusion_query_stage: int = 2,
        fusion_global_stage: int = 3,
        fusion_bottleneck_num_heads: int = 4,
        fusion_bottleneck_dropout: float = 0.0,
        fusion_bottleneck_scale_init: float = 0.1,
        physical_cyclic_lags: Sequence[int] | str = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: Sequence[int] | str = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fusion_query_stage = int(fusion_query_stage)
        self.fusion_global_stage = int(fusion_global_stage)
        _validate_stage_pair(self, self.fusion_query_stage, self.fusion_global_stage)
        self.query_channels = int(self.encoder.output_channels[self.fusion_query_stage])
        self.global_channels = int(self.encoder.output_channels[self.fusion_global_stage])
        self.bottleneck_attention = BottleneckSelfAttention1D(
            channels=self.global_channels,
            num_heads=int(fusion_bottleneck_num_heads),
            dropout=float(fusion_bottleneck_dropout),
            residual_scale_init=float(fusion_bottleneck_scale_init),
        )
        self.physical_token_extractor = TimeDomainPhysicalTokenExtractor(
            cyclic_lags=physical_cyclic_lags,
            polyphase_branches=int(physical_polyphase_branches),
            symbol_orders=physical_symbol_orders,
            min_cyclic_freq=float(physical_min_cyclic_freq),
            max_cyclic_freq=float(physical_max_cyclic_freq),
            cyclic_temperature=float(physical_cyclic_temperature),
        )

    def encode_inputs(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        skips = list(self.encoder(x))
        enhanced_global = self.bottleneck_attention(skips[self.fusion_global_stage])
        skips[self.fusion_global_stage] = enhanced_global
        return skips, self.physical_token_extractor(x)

    def no_weight_decay(self) -> set[str]:
        return {"bottleneck_attention.residual_scale"}


class IQUBiMamba1D_UnifiedPhysicalGlobalKV(_PhysicalFusionBase):
    """Stage 252: one K/V bank containing enhanced global and physical tokens."""

    def __init__(self, *args, fusion_global_kv_tokens: int = 64,
                 fusion_num_heads: int = 4, fusion_dropout: float = 0.0,
                 fusion_channel_scale_init: float = 0.1,
                 fusion_channel_scale_max: float = 0.5, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unified_attention = UnifiedGlobalPhysicalCrossAttention(
            query_channels=self.query_channels,
            global_channels=self.global_channels,
            physical_dim=self.physical_token_extractor.token_dim,
            kv_tokens=int(fusion_global_kv_tokens),
            num_heads=int(fusion_num_heads),
            dropout=float(fusion_dropout),
        )
        self.channel_scale = BoundedChannelScale(
            self.query_channels, fusion_channel_scale_init, fusion_channel_scale_max
        )

    def forward(self, x: torch.Tensor):
        skips, physical_tokens = self.encode_inputs(x)
        delta = self.unified_attention.compute_delta(
            skips[self.fusion_query_stage], skips[self.fusion_global_stage], physical_tokens
        )
        scale = self.channel_scale.values().to(device=delta.device, dtype=delta.dtype)
        skips[self.fusion_query_stage] = skips[self.fusion_query_stage] + scale.view(1, -1, 1) * delta
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        return super().no_weight_decay() | {
            "channel_scale.raw", "unified_attention.physical_value_gate.net.2.bias"
        }

    def checkpoint_prefix_aliases(self) -> tuple[tuple[str, str], ...]:
        return ((f"cross_scale_blocks.{self.fusion_query_stage}.", "unified_attention."),)


class IQUBiMamba1D_PhysicalFiLMGlobalMemory(_PhysicalFusionBase):
    """Stage 253: physical evidence applies bounded FiLM to enhanced global memory."""

    def __init__(self, *args, fusion_global_kv_tokens: int = 64,
                 fusion_num_heads: int = 4, fusion_dropout: float = 0.0,
                 fusion_channel_scale_init: float = 0.1,
                 fusion_channel_scale_max: float = 0.5,
                 fusion_film_hidden: int = 64, fusion_film_max_delta: float = 0.1,
                 fusion_enable_global_film: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.physical_film = (
            PhysicalFeatureFiLM(
                self.physical_token_extractor.token_dim, self.global_channels,
                fusion_film_hidden, fusion_film_max_delta,
            )
            if bool(fusion_enable_global_film)
            else None
        )
        self.global_cross_attention = CompressedGlobalCrossAttention(
            self.query_channels, self.global_channels, fusion_global_kv_tokens,
            fusion_num_heads, fusion_dropout, residual_scale_init=None,
        )
        self.channel_scale = BoundedChannelScale(
            self.query_channels, fusion_channel_scale_init, fusion_channel_scale_max
        )

    def forward(self, x: torch.Tensor):
        skips, physical_tokens = self.encode_inputs(x)
        global_feature = self.physical_film(
            skips[self.fusion_global_stage], physical_tokens
        )
        skips[self.fusion_global_stage] = global_feature
        delta = self.global_cross_attention.compute_delta(
            skips[self.fusion_query_stage], global_feature
        )
        scale = self.channel_scale.values().to(device=delta.device, dtype=delta.dtype)
        skips[self.fusion_query_stage] = skips[self.fusion_query_stage] + scale.view(1, -1, 1) * delta
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        names = super().no_weight_decay() | {"channel_scale.raw"}
        if self.physical_film is not None:
            names.add("physical_film.net.2.bias")
        return names

    def checkpoint_prefix_aliases(self) -> tuple[tuple[str, str], ...]:
        return ((f"cross_scale_blocks.{self.fusion_query_stage}.",
                 "global_cross_attention."),)


class IQUBiMamba1D_ScaleIsolatedPhysicalFusion(IQUBiMamba1D_PhysicalFiLMGlobalMemory):
    """Stage 254: keep global fusion at Stage-2 and physical FiLM at Stage-1."""

    def __init__(self, *args, fusion_physical_stage: int = 1,
                 fusion_physical_film_hidden: int = 64,
                 fusion_physical_film_max_delta: float = 0.1, **kwargs) -> None:
        super().__init__(*args, fusion_enable_global_film=False, **kwargs)
        self.fusion_physical_stage = int(fusion_physical_stage)
        if self.fusion_physical_stage in {self.fusion_query_stage, self.fusion_global_stage}:
            raise ValueError("physical FiLM stage must differ from query and global stages")
        if not 0 <= self.fusion_physical_stage < len(self.encoder.output_channels):
            raise ValueError("invalid physical FiLM stage")
        self.physical_skip_film = PhysicalFeatureFiLM(
            self.physical_token_extractor.token_dim,
            int(self.encoder.output_channels[self.fusion_physical_stage]),
            fusion_physical_film_hidden,
            fusion_physical_film_max_delta,
        )

    def forward(self, x: torch.Tensor):
        skips, physical_tokens = self.encode_inputs(x)
        skips[self.fusion_physical_stage] = self.physical_skip_film(
            skips[self.fusion_physical_stage], physical_tokens
        )
        delta = self.global_cross_attention.compute_delta(
            skips[self.fusion_query_stage], skips[self.fusion_global_stage]
        )
        scale = self.channel_scale.values().to(device=delta.device, dtype=delta.dtype)
        skips[self.fusion_query_stage] = skips[self.fusion_query_stage] + scale.view(1, -1, 1) * delta
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        return super().no_weight_decay() | {"physical_skip_film.net.2.bias"}


class IQUBiMamba1D_IdentityAwarePhysicalMoE(_PhysicalFusionBase):
    """Stage 255: route identity, global, physical and joint residual experts."""

    def __init__(self, *args, fusion_global_kv_tokens: int = 64,
                 fusion_num_heads: int = 4, fusion_dropout: float = 0.0,
                 fusion_channel_scale_init: float = 0.1,
                 fusion_channel_scale_max: float = 0.5,
                 fusion_router_hidden: int = 64,
                 fusion_expert_prior: Sequence[float] = (0.7, 0.1, 0.1, 0.1),
                 fusion_condition_hidden: int = 16,
                 fusion_condition_embedding: int = 16,
                 fusion_trust_penalty_init: float = 0.1,
                 fusion_trust_penalty_enable: bool = False,
                 fusion_condition_routing_enable: bool = False,
                 fusion_counterfactual_enable: bool = False,
                 fusion_return_route_aux: bool = False,
                 fusion_route_candidate_probability: float = 1.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_attention = CompressedGlobalCrossAttention(
            self.query_channels, self.global_channels, fusion_global_kv_tokens,
            fusion_num_heads, fusion_dropout, residual_scale_init=None,
        )
        self.physical_attention = CompactPhysicalCrossAttention(
            self.query_channels, self.physical_token_extractor.token_dim,
            self.physical_token_extractor.token_count, fusion_num_heads,
            fusion_dropout, residual_scale_init=None,
        )
        self.joint_proj = nn.Conv1d(2 * self.query_channels, self.query_channels, 1)
        self.joint_norm = nn.LayerNorm(self.query_channels)
        self.condition_encoder = MixtureConditionEncoder(
            fusion_condition_hidden, fusion_condition_embedding
        )
        self.expert_router = IdentityAwareEvidenceRouter(
            token_dim=self.physical_token_extractor.token_dim,
            query_channels=self.query_channels,
            global_channels=self.global_channels,
            condition_dim=fusion_condition_embedding,
            hidden_channels=fusion_router_hidden,
            prior=fusion_expert_prior,
            trust_penalty_init=fusion_trust_penalty_init,
        )
        self.channel_scale = BoundedChannelScale(
            self.query_channels, fusion_channel_scale_init, fusion_channel_scale_max
        )
        self.trust_penalty_enable = bool(fusion_trust_penalty_enable)
        self.condition_routing_enable = bool(fusion_condition_routing_enable)
        self.counterfactual_enable = bool(fusion_counterfactual_enable)
        self.return_route_aux = bool(
            fusion_return_route_aux
            or self.counterfactual_enable
            or self.condition_routing_enable
        )
        probability = float(fusion_route_candidate_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("fusion_route_candidate_probability must be in [0, 1]")
        self.route_candidate_probability = probability

    def _decode_candidate(
        self, skips: list[torch.Tensor], query: torch.Tensor
    ):
        candidate_skips = list(skips)
        candidate_skips[self.fusion_query_stage] = query
        return self.decoder(candidate_skips)

    def forward(self, x: torch.Tensor):
        skips, physical_tokens = self.encode_inputs(x)
        query = skips[self.fusion_query_stage]
        global_delta = self.global_attention.compute_delta(
            query, skips[self.fusion_global_stage]
        )
        physical_delta = self.physical_attention.compute_delta(query, physical_tokens)
        joint_delta = self.joint_proj(torch.cat([global_delta, physical_delta], dim=1))
        joint_delta = self.joint_norm(joint_delta.transpose(1, 2)).transpose(1, 2)
        if self.condition_routing_enable:
            condition_embedding, snr_prediction = self.condition_encoder(x)
        else:
            condition_embedding = query.new_zeros(
                query.size(0), self.expert_router.condition_proj.in_features,
                dtype=torch.float32,
            )
            snr_prediction = None
        weights, uncertainty, agreement = self.expert_router(
            physical_tokens,
            query,
            skips[self.fusion_global_stage],
            global_delta,
            physical_delta,
            condition_embedding,
            condition_enabled=self.condition_routing_enable,
            evidence_context_enabled=self.trust_penalty_enable,
            trust_penalty_enabled=self.trust_penalty_enable,
        )
        weights = weights.to(dtype=query.dtype)
        update = (
            weights[:, 1, None, None] * global_delta
            + weights[:, 2, None, None] * physical_delta
            + weights[:, 3, None, None] * joint_delta
        )
        scale = self.channel_scale.values().to(device=query.device, dtype=query.dtype)
        scale_view = scale.view(1, -1, 1)
        skips[self.fusion_query_stage] = query + scale_view * update
        separation = self.decoder(skips)
        if not (self.training and self.return_route_aux):
            return separation

        auxiliary = {
            "route_weights": weights,
            "route_uncertainty": uncertainty,
            "route_agreement": agreement,
        }
        if snr_prediction is not None:
            auxiliary["snr_prediction"] = snr_prediction
        should_decode_candidates = (
            self.counterfactual_enable
            and (
                self.route_candidate_probability >= 1.0
                or float(torch.rand((), device=query.device).item())
                < self.route_candidate_probability
            )
        )
        if should_decode_candidates:
            with torch.no_grad():
                auxiliary["candidate_outputs"] = tuple(
                    self._decode_candidate(skips, candidate_query)
                    for candidate_query in (
                        query,
                        query + scale_view * global_delta,
                        query + scale_view * physical_delta,
                        query + scale_view * joint_delta,
                    )
                )
        return separation, auxiliary

    def routing_parameters(self):
        yield from self.expert_router.parameters()
        yield from self.condition_encoder.parameters()

    def no_weight_decay(self) -> set[str]:
        return super().no_weight_decay() | {
            "channel_scale.raw", "expert_router.net.2.bias",
            "expert_router.raw_trust_penalty",
        }

    def checkpoint_prefix_aliases(self) -> tuple[tuple[str, str], ...]:
        return (
            (f"cross_scale_blocks.{self.fusion_query_stage}.", "global_attention."),
            ("physical_cross_attention.", "physical_attention."),
        )


class IQUBiMamba1D_CrossGatedDualMemory(_PhysicalFusionBase):
    """Stage 256: global and physical residuals gate each other."""

    def __init__(self, *args, fusion_global_kv_tokens: int = 64,
                 fusion_num_heads: int = 4, fusion_dropout: float = 0.0,
                 fusion_channel_scale_init: float = 0.1,
                 fusion_channel_scale_max: float = 0.5,
                 fusion_router_hidden: int = 64, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.global_attention = CompressedGlobalCrossAttention(
            self.query_channels, self.global_channels, fusion_global_kv_tokens,
            fusion_num_heads, fusion_dropout, residual_scale_init=None,
        )
        self.physical_attention = CompactPhysicalCrossAttention(
            self.query_channels, self.physical_token_extractor.token_dim,
            self.physical_token_extractor.token_count, fusion_num_heads,
            fusion_dropout, residual_scale_init=None,
        )
        self.physical_to_global_gate = PhysicalReliabilityRouter(
            self.physical_token_extractor.token_dim, self.query_channels,
            fusion_router_hidden, gate_init=1.0, gate_max=2.0,
        )
        self.global_to_physical_gate = nn.Sequential(
            nn.Linear(2 * self.global_channels, int(fusion_router_hidden)),
            nn.GELU(),
            nn.Linear(int(fusion_router_hidden), self.query_channels),
        )
        nn.init.zeros_(self.global_to_physical_gate[-1].weight)
        nn.init.zeros_(self.global_to_physical_gate[-1].bias)
        self.global_scale = BoundedChannelScale(
            self.query_channels, fusion_channel_scale_init, fusion_channel_scale_max
        )
        self.physical_scale = BoundedChannelScale(
            self.query_channels, fusion_channel_scale_init, fusion_channel_scale_max
        )

    def forward(self, x: torch.Tensor):
        skips, physical_tokens = self.encode_inputs(x)
        query = skips[self.fusion_query_stage]
        global_feature = skips[self.fusion_global_stage]
        global_delta = self.global_attention.compute_delta(query, global_feature)
        physical_delta = self.physical_attention.compute_delta(query, physical_tokens)
        physical_gate = self.physical_to_global_gate(physical_tokens).to(dtype=query.dtype)
        pooled = torch.cat([global_feature.mean(-1), global_feature.std(-1, unbiased=False)], dim=-1)
        global_gate = 2.0 * torch.sigmoid(self.global_to_physical_gate(pooled.float()))
        global_gate = global_gate.to(device=query.device, dtype=query.dtype)
        global_scale = self.global_scale.values().to(device=query.device, dtype=query.dtype)
        physical_scale = self.physical_scale.values().to(device=query.device, dtype=query.dtype)
        skips[self.fusion_query_stage] = (
            query
            + global_scale.view(1, -1, 1) * physical_gate.unsqueeze(-1) * global_delta
            + physical_scale.view(1, -1, 1) * global_gate.unsqueeze(-1) * physical_delta
        )
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        return super().no_weight_decay() | {
            "global_scale.raw", "physical_scale.raw",
            "physical_to_global_gate.net.2.bias", "global_to_physical_gate.2.bias",
        }

    def checkpoint_prefix_aliases(self) -> tuple[tuple[str, str], ...]:
        return (
            (f"cross_scale_blocks.{self.fusion_query_stage}.", "global_attention."),
            ("physical_cross_attention.", "physical_attention."),
        )


# Compatibility for checkpoints or external code importing the former class name.
IQUBiMamba1D_HierarchicalRoutedFusion = IQUBiMamba1D_HierarchicalAdditiveFusion
