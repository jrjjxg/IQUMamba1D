"""Unrolled residual interference-cancellation modules for IQ separation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualConvBlock1D(nn.Module):
    """Lightweight residual conv block for per-source refinement."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        padding = kernel_size // 2
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm1d(channels, eps=1e-5, affine=True)
        self.pwconv1 = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=True)
        self.pwconv2 = nn.Conv1d(channels * 2, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.pwconv2(x)
        return residual + x


class ResidualICUpdateBlock(nn.Module):
    """One learnable correction step for a single source estimate."""

    def __init__(
        self,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(6, hidden_channels, kernel_size=1, bias=True),
            nn.InstanceNorm1d(hidden_channels, eps=1e-5, affine=True),
            nn.GELU(),
        )
        self.block1 = _ResidualConvBlock1D(
            channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.block2 = _ResidualConvBlock1D(
            channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.output_proj = nn.Conv1d(hidden_channels, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        estimate: torch.Tensor,
        residual_target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([estimate, residual_target, mixture], dim=1)
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.output_proj(x)


class _DilatedGatedResidualBlock1D(nn.Module):
    """Dilated depthwise residual block with pointwise gating."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")

        padding = dilation * (kernel_size // 2)
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = nn.InstanceNorm1d(channels, eps=1e-5, affine=True)
        self.gate_proj = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=True)
        self.output_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        value, gate = self.gate_proj(x).chunk(2, dim=1)
        x = self.act(value) * torch.sigmoid(gate)
        x = self.dropout(x)
        x = self.output_proj(x)
        return residual + x


class DilatedGatedResidualICUpdateBlock(nn.Module):
    """One URIC update step using dilated gated residual refinement blocks."""

    def __init__(
        self,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        dropout: float = 0.0,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if len(dilations) < 1:
            raise ValueError("dilations must contain at least one value")

        self.input_proj = nn.Sequential(
            nn.Conv1d(6, hidden_channels, kernel_size=1, bias=True),
            nn.InstanceNorm1d(hidden_channels, eps=1e-5, affine=True),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                _DilatedGatedResidualBlock1D(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.output_proj = nn.Conv1d(hidden_channels, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        estimate: torch.Tensor,
        residual_target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([estimate, residual_target, mixture], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class CrossAttentionICUpdateBlock(nn.Module):
    """One URIC update step using estimate-to-residual cross attention."""

    def __init__(
        self,
        hidden_channels: int = 48,
        num_heads: int = 4,
        dropout: float = 0.0,
        attention_stride: int = 1,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
            )
        if attention_stride < 1:
            raise ValueError(f"attention_stride must be >= 1, got {attention_stride}")
        if ffn_multiplier < 1:
            raise ValueError(f"ffn_multiplier must be >= 1, got {ffn_multiplier}")

        self.query_proj = nn.Conv1d(2, hidden_channels, kernel_size=1, bias=True)
        self.context_proj = nn.Conv1d(4, hidden_channels, kernel_size=1, bias=True)
        self.context_pool = (
            nn.AvgPool1d(
                kernel_size=attention_stride,
                stride=attention_stride,
                ceil_mode=True,
            )
            if attention_stride > 1
            else nn.Identity()
        )
        self.query_norm = nn.LayerNorm(hidden_channels)
        self.context_norm = nn.LayerNorm(hidden_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * ffn_multiplier, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_channels * ffn_multiplier, hidden_channels, kernel_size=1, bias=True),
        )
        self.output_proj = nn.Conv1d(hidden_channels, 2, kernel_size=1, bias=True)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous()

    @staticmethod
    def _to_channels(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous()

    def forward(
        self,
        estimate: torch.Tensor,
        residual_target: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_proj(estimate)
        context = self.context_proj(torch.cat([residual_target, mixture], dim=1))
        context = self.context_pool(context)

        query_tokens = self.query_norm(self._to_tokens(query))
        context_tokens = self.context_norm(self._to_tokens(context))
        attn_out, _ = self.attn(
            query_tokens,
            context_tokens,
            context_tokens,
            need_weights=False,
        )
        x_tokens = self.attn_norm(query_tokens + attn_out)
        x = self._to_channels(x_tokens)
        x = x + self.ffn(x)
        return self.output_proj(x)


class UnrolledResidualInterferenceCancellationHead(nn.Module):
    """Iterative residual-interference-cancellation refinement head.

    Starting from separator outputs ``s^(0)``, each stage builds a per-source
    residual target assuming the other sources are interference:

        r_k^(t) = x_mix - sum_{j != k} s_j^(t)

    and refines the current estimate through a learnable update block:

        s_k^(t+1) = s_k^(t) + alpha_t * Delta_k^(t)

    The update block is shared across sources; weights can optionally be tied
    across iterations for a more algorithm-unfolding-like design.
    """

    def __init__(
        self,
        num_sources: int,
        num_steps: int = 3,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        dropout: float = 0.0,
        tied_steps: bool = True,
        step_init: float = 0.5,
        update_block_type: str = "conv",
        dilations: tuple[int, ...] = (1, 2, 4),
        num_heads: int = 4,
        attention_stride: int = 1,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if not (0.0 < step_init < 1.0):
            raise ValueError(f"step_init must be in (0, 1), got {step_init}")

        self.num_sources = int(num_sources)
        self.num_steps = int(num_steps)
        self.tied_steps = bool(tied_steps)
        self.update_block_type = str(update_block_type)

        if self.tied_steps:
            self.shared_block = self._make_update_block(
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                update_block_type=update_block_type,
                dilations=dilations,
                num_heads=num_heads,
                attention_stride=attention_stride,
                ffn_multiplier=ffn_multiplier,
            )
            self.blocks = None
        else:
            self.shared_block = None
            self.blocks = nn.ModuleList([
                self._make_update_block(
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    update_block_type=update_block_type,
                    dilations=dilations,
                    num_heads=num_heads,
                    attention_stride=attention_stride,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(self.num_steps)
            ])

        init_logit = math.log(step_init / (1.0 - step_init))
        self.step_logits = nn.Parameter(torch.full((self.num_steps,), init_logit))

    @staticmethod
    def _make_update_block(
        hidden_channels: int,
        kernel_size: int,
        dropout: float,
        update_block_type: str,
        dilations: tuple[int, ...],
        num_heads: int,
        attention_stride: int,
        ffn_multiplier: int,
    ) -> nn.Module:
        if update_block_type == "conv":
            return ResidualICUpdateBlock(
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        if update_block_type == "dilated_gated":
            return DilatedGatedResidualICUpdateBlock(
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                dilations=dilations,
            )
        if update_block_type == "cross_attention":
            return CrossAttentionICUpdateBlock(
                hidden_channels=hidden_channels,
                num_heads=num_heads,
                dropout=dropout,
                attention_stride=attention_stride,
                ffn_multiplier=ffn_multiplier,
            )
        raise ValueError(
            "update_block_type must be one of {'conv', 'dilated_gated', 'cross_attention'}, "
            f"got {update_block_type!r}"
        )

    @staticmethod
    def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(
            mixture,
            size=target_length,
            mode="linear",
            align_corners=False,
        )

    def _get_block(self, step_idx: int) -> nn.Module:
        return self.shared_block if self.tied_steps else self.blocks[step_idx]

    def forward(
        self,
        estimates: torch.Tensor,
        mixture: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor:
        if estimates.dim() != 3:
            raise ValueError(
                f"estimates must have shape (B, 2K, L), got {tuple(estimates.shape)}"
            )
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}"
            )
        if estimates.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} estimate channels, got {estimates.size(1)}"
            )

        target_length = estimates.size(-1)
        mixture = self._resize_mixture(mixture, target_length)

        b = estimates.size(0)
        est = estimates.reshape(b, self.num_sources, 2, target_length)
        mixture_expanded = mixture.unsqueeze(1)  # (B, 1, 2, L)
        stage_outputs = []

        for step_idx in range(self.num_steps):
            sum_all = est.sum(dim=1, keepdim=True)                # (B, 1, 2, L)
            interference = sum_all - est                          # (B, K, 2, L)
            residual_target = mixture_expanded - interference     # (B, K, 2, L)

            block = self._get_block(step_idx)
            est_flat = est.reshape(b * self.num_sources, 2, target_length)
            residual_flat = residual_target.reshape(b * self.num_sources, 2, target_length)
            mixture_flat = mixture_expanded.expand(-1, self.num_sources, -1, -1).reshape(
                b * self.num_sources, 2, target_length
            )

            delta = block(est_flat, residual_flat, mixture_flat)
            delta = delta.reshape(b, self.num_sources, 2, target_length)
            step_scale = torch.sigmoid(self.step_logits[step_idx]).view(1, 1, 1, 1)
            est = est + step_scale * delta
            if return_intermediate:
                stage_outputs.append(est.reshape(b, 2 * self.num_sources, target_length))

        final = est.reshape(b, 2 * self.num_sources, target_length)
        if return_intermediate:
            return final, stage_outputs
        return final
