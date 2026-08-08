"""Cross-scale attention ablations built on the Stage-12 BiMamba separator.

The encoder and decoder are unchanged. Before decoding, selected higher-
resolution skip features query a fixed-size token bank compressed from the
deepest encoder feature. This keeps attention cost linear in query length.
"""

from __future__ import annotations

from typing import List, Sequence, Type

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUBiMamba1D import IQUBiMamba1D


class CompressedGlobalCrossAttention(nn.Module):
    """Inject compressed bottleneck context into one encoder scale."""

    def __init__(
        self,
        query_channels: int,
        global_channels: int,
        kv_tokens: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        residual_scale_init: float | None = 0.01,
    ) -> None:
        super().__init__()
        query_channels = int(query_channels)
        global_channels = int(global_channels)
        if query_channels % int(num_heads) != 0:
            raise ValueError(
                f"query_channels={query_channels} must be divisible by num_heads={num_heads}"
            )
        if int(kv_tokens) < 1:
            raise ValueError(f"kv_tokens must be positive, got {kv_tokens}")

        self.kv_tokens = int(kv_tokens)
        self.query_norm = nn.LayerNorm(query_channels)
        self.global_norm = nn.LayerNorm(global_channels)
        self.key_proj = nn.Linear(global_channels, query_channels, bias=False)
        self.value_proj = nn.Linear(global_channels, query_channels, bias=False)
        self.attention = nn.MultiheadAttention(
            embed_dim=query_channels,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(query_channels)
        if residual_scale_init is None:
            self.register_parameter("residual_scale", None)
        else:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _compress_global(self, global_feature: torch.Tensor) -> torch.Tensor:
        token_count = min(self.kv_tokens, int(global_feature.size(-1)))
        pooled = F.adaptive_avg_pool1d(global_feature, token_count)
        return self.global_norm(pooled.transpose(1, 2))

    def compute_delta(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_norm(query_feature.transpose(1, 2))
        global_tokens = self._compress_global(global_feature)
        key = self.key_proj(global_tokens)
        value = self.value_proj(global_tokens)
        delta, _ = self.attention(query, key, value, need_weights=False)
        return self.output_norm(delta).transpose(1, 2)

    def forward(
        self,
        query_feature: torch.Tensor,
        global_feature: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.compute_delta(query_feature, global_feature)
        scale = delta.new_tensor(1.0) if self.residual_scale is None else self.residual_scale
        if gate is not None:
            scale = scale * gate.to(dtype=delta.dtype).view(-1, 1, 1)
        return query_feature + scale * delta


class MixturePhysicalEvidenceGate(nn.Module):
    """Map scale-safe mixture statistics to one gate per fusion scale."""

    evidence_dim = 6

    def __init__(self, num_gates: int, hidden_channels: int = 32, eps: float = 1e-6) -> None:
        super().__init__()
        if int(num_gates) < 1:
            raise ValueError(f"num_gates must be positive, got {num_gates}")
        self.eps = float(eps)
        self.net = nn.Sequential(
            nn.Linear(self.evidence_dim, int(hidden_channels)),
            nn.GELU(),
            nn.Linear(int(hidden_channels), int(num_gates)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def extract_evidence(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(f"Expected mixture shaped (B, 2, L), got {tuple(mixture.shape)}")

        x = mixture.float()
        i, q = x[:, 0], x[:, 1]
        power_i = i.square().mean(dim=-1)
        power_q = q.square().mean(dim=-1)
        power = power_i + power_q
        denom = power.clamp_min(self.eps)

        log_power = torch.log(denom).clamp(-12.0, 12.0) / 12.0
        iq_balance = (power_i - power_q) / denom
        iq_corr = 2.0 * (i * q).mean(dim=-1) / denom

        if mixture.size(-1) > 1:
            i_prev, i_next = i[:, :-1], i[:, 1:]
            q_prev, q_next = q[:, :-1], q[:, 1:]
            lag_denom = (
                (i_prev.square() + q_prev.square()).mean(dim=-1)
                * (i_next.square() + q_next.square()).mean(dim=-1)
            ).clamp_min(self.eps).sqrt()
            lag_real = (i_prev * i_next + q_prev * q_next).mean(dim=-1) / lag_denom
            lag_imag = (i_prev * q_next - q_prev * i_next).mean(dim=-1) / lag_denom
        else:
            lag_real = torch.zeros_like(power)
            lag_imag = torch.zeros_like(power)

        envelope_power = i.square() + q.square()
        envelope_cv = (
            envelope_power.std(dim=-1, unbiased=False) / denom
        ).clamp(0.0, 4.0) / 4.0

        return torch.stack(
            [log_power, iq_balance, iq_corr, lag_real, lag_imag, envelope_cv],
            dim=-1,
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        # Zero-initialized logits produce gates of exactly one, matching the
        # ungated multi-scale model at initialization while allowing [0, 2].
        logits = self.net(self.extract_evidence(mixture))
        return 2.0 * torch.sigmoid(logits)


class IQUBiMamba1D_CrossScaleAttention(IQUBiMamba1D):
    """Stage-12 backbone with configurable compressed-KV cross-scale fusion."""

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
        cross_scale_query_stages: Sequence[int] = (2,),
        cross_scale_global_stage: int = 3,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 0.01,
        cross_scale_evidence_gate: bool = False,
        cross_scale_evidence_hidden: int = 32,
        cross_scale_evidence_eps: float = 1e-6,
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
        )

        self.cross_scale_global_stage = int(cross_scale_global_stage)
        self.cross_scale_query_stages = tuple(int(stage) for stage in cross_scale_query_stages)
        if not 0 <= self.cross_scale_global_stage < int(n_stages):
            raise ValueError(f"Invalid global stage {self.cross_scale_global_stage}")
        if not self.cross_scale_query_stages:
            raise ValueError("cross_scale_query_stages must not be empty")
        if len(set(self.cross_scale_query_stages)) != len(self.cross_scale_query_stages):
            raise ValueError("cross_scale_query_stages must be unique")
        for stage in self.cross_scale_query_stages:
            if not 0 <= stage < int(n_stages) or stage == self.cross_scale_global_stage:
                raise ValueError(
                    f"Query stage {stage} must be valid and differ from global stage "
                    f"{self.cross_scale_global_stage}"
                )

        global_channels = int(features_per_stage[self.cross_scale_global_stage])
        self.cross_scale_blocks = nn.ModuleDict({
            str(stage): CompressedGlobalCrossAttention(
                query_channels=int(features_per_stage[stage]),
                global_channels=global_channels,
                kv_tokens=cross_scale_kv_tokens,
                num_heads=cross_scale_num_heads,
                dropout=cross_scale_dropout,
                residual_scale_init=cross_scale_residual_scale_init,
            )
            for stage in self.cross_scale_query_stages
        })
        self.evidence_gate = (
            MixturePhysicalEvidenceGate(
                num_gates=len(self.cross_scale_query_stages),
                hidden_channels=cross_scale_evidence_hidden,
                eps=cross_scale_evidence_eps,
            )
            if bool(cross_scale_evidence_gate)
            else None
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        global_feature = skips[self.cross_scale_global_stage]
        gates = self.evidence_gate(x) if self.evidence_gate is not None else None

        enhanced_skips = list(skips)
        for gate_index, stage in enumerate(self.cross_scale_query_stages):
            gate = None if gates is None else gates[:, gate_index]
            enhanced_skips[stage] = self.cross_scale_blocks[str(stage)](
                enhanced_skips[stage],
                global_feature,
                gate=gate,
            )
        return self.decoder(enhanced_skips)
