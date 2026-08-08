import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type, Union

from models.IQUMamba1D import BasicResBlock, UpsampleLayer
from models.IQUResUNet1D import ResidualConvEncoder


def _make_norm(norm_op, channels: int, norm_op_kwargs: dict):
    if norm_op is None:
        return nn.Identity()
    return norm_op(channels, **(norm_op_kwargs or {}))


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.size(-1) == target_len:
        return x
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)


# ============================================================
# 1. Attention U-Net style skip
#    Standard decoder-guided additive attention gate
# ============================================================

class AttentionGateSkip1D(nn.Module):
    """
    Attention U-Net style decoder-guided skip gate.

    skip: [B, C_skip, L]
    dec:  [B, C_dec,  L]
    out:  [B, C_skip, L]

    This version uses a temporal scalar gate [B, 1, L], which is usually
    more stable for IQ signals than independent per-channel gates.
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 8)

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )

        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # residual form: starts close to raw skip
        self.res_scale = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))

        att = self.psi(self.skip_proj(skip) + self.dec_proj(dec))  # [B, 1, L]
        filtered = att * skip

        return skip + self.res_scale * (filtered - skip)


# ============================================================
# 2. UCTransNet / UDTransNet-lite style skip
#    Multi-scale channel fusion + decoder-guided recalibration
# ============================================================

class UCTransNetLiteSkipBridge1D(nn.Module):
    """
    UCTransNet/UDTransNet inspired 1D skip bridge.

    Core idea:
      1. Summarize all encoder skip features.
      2. Fuse multi-scale channel context.
      3. For each decoder stage, use decoder feature + local skip summary
         + global multi-scale context to produce a channel gate.

    This is channel-first, not full temporal Transformer, so it is much cheaper.
    """

    def __init__(
        self,
        skip_channels: List[int],
        dec_channels: List[int],
        context_dim: int = 128,
        hidden_ratio: float = 0.5,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        assert len(skip_channels) == len(dec_channels)
        self.skip_channels = list(skip_channels)
        self.dec_channels = list(dec_channels)
        self.num_levels = len(skip_channels)

        self.skip_to_context = nn.ModuleList([
            nn.Linear(c, context_dim) for c in skip_channels
        ])

        self.context_fuse = nn.Sequential(
            nn.Linear(context_dim * self.num_levels, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        self.gate_mlps = nn.ModuleList()
        for c_skip, c_dec in zip(skip_channels, dec_channels):
            hidden = max(int((c_skip + c_dec + context_dim) * hidden_ratio), 16)
            self.gate_mlps.append(
                nn.Sequential(
                    nn.Linear(c_skip + c_dec + context_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, c_skip),
                    nn.Sigmoid(),
                )
            )

        self.res_scale = nn.Parameter(torch.ones(1) * residual_scale_init)

    def build_context(self, encoder_skips: List[torch.Tensor]) -> torch.Tensor:
        """
        encoder_skips should be the encoder outputs excluding bottleneck,
        ordered shallow -> deep.

        Example:
            skips = encoder(x)
            context = bridge.build_context(skips[:-1])
        """
        assert len(encoder_skips) == self.num_levels

        tokens = []
        for feat, proj in zip(encoder_skips, self.skip_to_context):
            # channel descriptor: [B, C]
            desc = feat.mean(dim=-1)
            tokens.append(proj(desc))

        context = torch.cat(tokens, dim=1)
        context = self.context_fuse(context)
        return context  # [B, context_dim]

    def forward(
        self,
        skip: torch.Tensor,
        dec: torch.Tensor,
        level: int,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        level is the encoder skip index in shallow -> deep order.
        For a 4-stage encoder excluding bottleneck:
            level 0: highest resolution skip
            level 1: L/2 skip
            level 2: L/4 skip
        """
        dec = _match_length(dec, skip.size(-1))

        skip_desc = skip.mean(dim=-1)  # [B, C_skip]
        dec_desc = dec.mean(dim=-1)    # [B, C_dec]

        gate = self.gate_mlps[level](
            torch.cat([skip_desc, dec_desc, context], dim=1)
        )  # [B, C_skip]

        filtered = skip * gate.unsqueeze(-1)

        return skip + self.res_scale * (filtered - skip)


# ============================================================
# 3. DCA-lite style skip
#    Channel cross-attention + temporal cross-attention
# ============================================================

class DCALiteSkip1D(nn.Module):
    """
    DCA-inspired 1D skip processor.

    It contains:
      1. Channel cross recalibration using skip/decoder global descriptors.
      2. Temporal cross-attention: query from decoder, key/value from skip.

    To avoid O(L^2) explosion, temporal attention is done on a pooled sequence.
    """

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        attn_dim: int = 64,
        num_heads: int = 4,
        max_tokens: int = 256,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        self.skip_channels = skip_channels
        self.dec_channels = dec_channels
        self.attn_dim = attn_dim
        self.max_tokens = max_tokens

        # Channel recalibration
        hidden = max((skip_channels + dec_channels) // 2, 16)
        self.channel_gate = nn.Sequential(
            nn.Linear(skip_channels + dec_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, skip_channels),
            nn.Sigmoid(),
        )

        # Temporal cross-attention
        self.q_proj = nn.Conv1d(dec_channels, attn_dim, kernel_size=1)
        self.k_proj = nn.Conv1d(skip_channels, attn_dim, kernel_size=1)
        self.v_proj = nn.Conv1d(skip_channels, attn_dim, kernel_size=1)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.out_proj = nn.Conv1d(attn_dim, skip_channels, kernel_size=1)

        # Start temporal delta near zero for stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.res_scale = nn.Parameter(torch.ones(1) * residual_scale_init)

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, T], T <= max_tokens
        """
        L = x.size(-1)
        if L <= self.max_tokens:
            return x
        return F.adaptive_avg_pool1d(x, self.max_tokens)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        L = skip.size(-1)

        # -------- Channel cross recalibration --------
        skip_desc = skip.mean(dim=-1)
        dec_desc = dec.mean(dim=-1)

        ch_gate = self.channel_gate(
            torch.cat([skip_desc, dec_desc], dim=1)
        )  # [B, C_skip]

        skip_ch = skip * ch_gate.unsqueeze(-1)

        # -------- Temporal cross-attention --------
        skip_pool = self._pool_tokens(skip_ch)
        dec_pool = self._pool_tokens(dec)

        q = self.q_proj(dec_pool).transpose(1, 2)       # [B, T, D]
        k = self.k_proj(skip_pool).transpose(1, 2)      # [B, T, D]
        v = self.v_proj(skip_pool).transpose(1, 2)      # [B, T, D]

        attn_out, _ = self.temporal_attn(q, k, v, need_weights=False)
        attn_out = attn_out.transpose(1, 2)             # [B, D, T]

        delta = self.out_proj(attn_out)                 # [B, C_skip, T]
        delta = _match_length(delta, L)

        filtered = skip_ch + delta

        return skip + self.res_scale * (filtered - skip)


# ============================================================
# 4. Leakage-Suppressed Skip Gate (LSSG)
#    Only temporal scalar gate for interference suppression
# ============================================================

class LeakageSuppressedSkipGate1D(nn.Module):
    """
    Decoder-guided leakage-suppressed skip gate.
    
    Design principle:
      - Do not reconstruct skip features.
      - Do not channel-remix skip features.
      - Only suppress unreliable temporal positions.
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        use_vector_alpha: bool = False,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
            
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 8)
            
        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        self.gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True),
        )
        self.gate_sigmoid = nn.Sigmoid()
        
        if global_ctx_channels is not None:
            self.ctx_proj = nn.Conv1d(global_ctx_channels, 1, kernel_size=1, bias=False)
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, global_ctx: torch.Tensor = None) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        
        gate_logits = self.gate(self.skip_proj(skip) + self.dec_proj(dec))  # [B,1,L]
        
        if global_ctx is not None and hasattr(self, 'ctx_proj'):
            gate_logits = gate_logits + self.ctx_proj(global_ctx)
            
        gate = self.gate_sigmoid(gate_logits)  # [B,C,L]
        self.last_gate = gate
        
        out = skip * (1.0 - self.alpha + self.alpha * gate)
        
        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                g_mean = gate.mean().item()
                g_std = gate.std().item()
                a_val = self.alpha.mean().item()
                diff = torch.norm(out - skip) / (torch.norm(skip) + 1e-8)
                print(f"[LSSG_Temporal] alpha_mean: {a_val:.4f}, gate_mean: {g_mean:.4f}, gate_std: {g_std:.4f}, effect_ratio: {diff.item():.4f}")
                
        return out


class ChannelWiseLSSG1D(nn.Module):
    """
    Ablation C: Channel-wise Gate.
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        use_vector_alpha: bool = False,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
            
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 8)
            
        # Idea A: Use GroupNorm(1, C) (LayerNorm equivalent) to forcefully align the 
        # heavily transformed `dec` distribution with the raw `skip` distribution.
        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True)
        )
        
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True)
        )
        
        self.gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True),
        )
        self.gate_sigmoid = nn.Sigmoid()
        
        if global_ctx_channels is not None:
            self.ctx_proj = nn.Conv1d(global_ctx_channels, skip_channels, kernel_size=1, bias=False)
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, global_ctx: torch.Tensor = None) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        
        gate_logits = self.gate(self.skip_proj(skip) + self.dec_proj(dec))  # [B,C,L]
        
        if global_ctx is not None and hasattr(self, 'ctx_proj'):
            gate_logits = gate_logits + self.ctx_proj(global_ctx)
            
        gate = self.gate_sigmoid(gate_logits)
        self.last_gate = gate
        
        out = skip * (1.0 - self.alpha + self.alpha * gate)
        
        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                g_mean = gate.mean().item()
                g_std = gate.std().item()
                a_val = self.alpha.mean().item()
                diff = torch.norm(out - skip) / (torch.norm(skip) + 1e-8)
                print(f"[LSSG_Channel] alpha_mean: {a_val:.4f}, gate_mean: {g_mean:.4f}, gate_std: {g_std:.4f}, effect_ratio: {diff.item():.4f}")
                
        return out

class ChannelWiseLSSG1D_SE(nn.Module):
    """
    Stage 161: LSSG with Squeeze-and-Excitation (Global Average Pooling).
    ANTI-OVERFITTING: bottleneck //4, + Dropout1d
    """
    def __init__(self, skip_channels: int, dec_channels: int, inter_channels: int = None, norm_op: Type[nn.Module] = nn.InstanceNorm1d, norm_op_kwargs: dict = None, residual_scale_init: float = 0.1, global_ctx_channels: int = None, use_vector_alpha: bool = False):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if inter_channels is None:
            inter_channels = max(skip_channels // 4, 8) # REDUCED CAPACITY
            
        self.skip_proj = nn.Sequential(nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False), nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True))
        self.dec_proj = nn.Sequential(nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False), nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True))
        
        self.se_pool = nn.AdaptiveAvgPool1d(1)
        self.se_gate = nn.Sequential(
            nn.Linear(inter_channels, inter_channels // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels // 2, inter_channels, bias=False),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout1d(0.1)
        self.gate = nn.Sequential(nn.ReLU(inplace=True), nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True))
        self.gate_sigmoid = nn.Sigmoid()
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, global_ctx: torch.Tensor = None) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        fused = self.skip_proj(skip) + self.dec_proj(dec)
        
        # Squeeze-and-Excitation
        se_weight = self.se_gate(self.se_pool(fused).squeeze(-1)).unsqueeze(-1)
        fused = fused * se_weight
        fused = self.dropout(fused)
        
        gate_logits = self.gate(fused)
        gate = self.gate_sigmoid(gate_logits)
        return skip * (1.0 - self.alpha + self.alpha * gate)

class ChannelWiseLSSG1D_SwiGLU(nn.Module):
    """
    Stage 162: LSSG with SwiGLU / StarReLU Multiplicative Gating.
    ANTI-OVERFITTING: bottleneck //4, + Dropout1d
    """
    def __init__(self, skip_channels: int, dec_channels: int, inter_channels: int = None, norm_op: Type[nn.Module] = nn.InstanceNorm1d, norm_op_kwargs: dict = None, residual_scale_init: float = 0.1, global_ctx_channels: int = None, use_vector_alpha: bool = False):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if inter_channels is None:
            inter_channels = max(skip_channels // 4, 8) # REDUCED CAPACITY
            
        self.skip_proj = nn.Sequential(nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False), nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True))
        self.dec_proj = nn.Sequential(nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False), nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True))
        
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout1d(0.1)
        self.gate = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True)
        self.gate_sigmoid = nn.Sigmoid()
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, global_ctx: torch.Tensor = None) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        # Multiplicative SwiGLU fusion instead of addition
        fused = self.skip_proj(skip) * self.act(self.dec_proj(dec))
        fused = self.dropout(fused)
        
        gate_logits = self.gate(fused)
        gate = self.gate_sigmoid(gate_logits)
        return skip * (1.0 - self.alpha + self.alpha * gate)

class ChannelWiseLSSG1D_Depthwise(nn.Module):
    """
    Stage 154: LSSG with Local-Aware Depthwise Gating (kernel_size=5).
    """
    def __init__(self, skip_channels: int, dec_channels: int, inter_channels: int = None, norm_op: Type[nn.Module] = nn.InstanceNorm1d, norm_op_kwargs: dict = None, residual_scale_init: float = 0.1, global_ctx_channels: int = None, use_vector_alpha: bool = False):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if inter_channels is None:
            inter_channels = max(skip_channels // 4, 8)
            
        self.skip_proj = nn.Sequential(nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False), nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True))
        self.dec_proj = nn.Sequential(nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False), nn.GroupNorm(1, inter_channels, eps=1e-5, affine=True))
        
        # Depthwise local context
        self.local_dw = nn.Conv1d(inter_channels, inter_channels, kernel_size=5, padding=2, groups=inter_channels, bias=False)
        
        self.gate = nn.Sequential(nn.ReLU(inplace=True), nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True))
        self.gate_sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout1d(0.1) # Added Spatial Dropout for anti-overfitting
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, global_ctx: torch.Tensor = None) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        fused = self.skip_proj(skip) + self.dec_proj(dec)
        fused = self.local_dw(fused)
        fused = self.dropout(fused) # Apply spatial dropout here
        
        gate_logits = self.gate(fused)
        gate = self.gate_sigmoid(gate_logits)
        return skip * (1.0 - self.alpha + self.alpha * gate)


class ChannelWiseLSSG1D_MS(nn.Module):
    """
    Stage 121: Multi-Scale Context Channel-wise Gate.
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        use_vector_alpha: bool = False,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
            
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 8)
            
        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        # Multi-scale gating
        self.relu = nn.ReLU(inplace=True)
        self.gate_1x1 = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=False)
        # We don't strictly require groups to divide channels if we don't use groups, 
        # but standard depthwise needs groups=inter_channels and out=inter_channels.
        # Since we map inter_channels -> skip_channels, we just do regular conv to be safe and avoid dimension mismatches.
        self.gate_5x5 = nn.Conv1d(inter_channels, skip_channels, kernel_size=5, padding=2, bias=False)
        self.gate_9x9 = nn.Conv1d(inter_channels, skip_channels, kernel_size=9, padding=4, bias=True)
        self.gate_sigmoid = nn.Sigmoid()
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        
        feat = self.relu(self.skip_proj(skip) + self.dec_proj(dec))
        gate_feat = self.gate_1x1(feat) + self.gate_5x5(feat) + self.gate_9x9(feat)
        gate = self.gate_sigmoid(gate_feat)  # [B,C,L]
        self.last_gate = gate
        
        out = skip * (1.0 - self.alpha + self.alpha * gate)
        return out


class ChannelWiseLSSG1D_Context(nn.Module):
    """
    Stage 122: Multi-Scale + Global Context Channel-wise Gate.
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        use_vector_alpha: bool = False,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
            
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 8)
            
        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        # Multi-scale gating
        self.relu = nn.ReLU(inplace=True)
        self.gate_1x1 = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=False)
        self.gate_5x5 = nn.Conv1d(inter_channels, skip_channels, kernel_size=5, padding=2, bias=False)
        self.gate_9x9 = nn.Conv1d(inter_channels, skip_channels, kernel_size=9, padding=4, bias=False)
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True)
        if global_ctx_channels is not None:
            self.ctx_proj = nn.Conv1d(global_ctx_channels, skip_channels, kernel_size=1, bias=False)

        self.gate_sigmoid = nn.Sigmoid()
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, global_ctx: torch.Tensor = None) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))

        feat = self.relu(self.skip_proj(skip) + self.dec_proj(dec))

        gate_feat = self.gate_1x1(feat) + self.gate_5x5(feat) + self.gate_9x9(feat)
        global_ctx_feat = self.global_proj(self.global_pool(feat))
        global_ctx_feat = global_ctx_feat.expand(-1, -1, skip.size(-1))
        gate_logits = gate_feat + global_ctx_feat

        if global_ctx is not None and hasattr(self, 'ctx_proj'):
            gate_logits = gate_logits + self.ctx_proj(_match_length(global_ctx, skip.size(-1)))
            
        gate = self.gate_sigmoid(gate_logits)  # [B,C,L]
        self.last_gate = gate
        
        out = skip * (1.0 - self.alpha + self.alpha * gate)
        
        if self.training and torch.rand(1).item() < 0.001:
            with torch.no_grad():
                g_mean = gate.mean().item()
                g_std = gate.std().item()
                a_val = self.alpha.mean().item()
                diff = torch.norm(out - skip) / (torch.norm(skip) + 1e-8)
                print(f"[LSSG_Context] alpha_mean: {a_val:.4f}, gate_mean: {g_mean:.4f}, gate_std: {g_std:.4f}, effect_ratio: {diff.item():.4f}")
                
        return out


class RefinedLSSG1D(nn.Module):
    """
    Ablation D: Gate + Conv Refinement.
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        use_vector_alpha: bool = False,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
            
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 8)
            
        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        
        self.gate = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        
        self.refine = nn.Conv1d(skip_channels, skip_channels, kernel_size=3, padding=1)
        
        if use_vector_alpha:
            self.alpha = nn.Parameter(torch.ones(1, skip_channels, 1) * residual_scale_init)
        else:
            self.alpha = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        
        gate = self.gate(self.skip_proj(skip) + self.dec_proj(dec))  # [B,1,L]
        self.last_gate = gate
        
        out = skip * (1.0 - self.alpha + self.alpha * gate)
        out = self.refine(out)
        return out


# ============================================================
# 5. General Decoder
# ============================================================

class SkipEnhancedUNetResDecoder(nn.Module):
    """
    Drop-in replacement for PlainUNetResDecoder.

    Supported skip modes:
      - "attention": Attention U-Net style additive attention gate
      - "uct": UCTransNet/UDTransNet-lite multi-scale channel recalibration
      - "dca": DCA-lite channel + temporal cross-attention
      - "lssg": Leakage-Suppressed Skip Gate (Temporal scalar gate)
      - "lssg_channel": Channel-wise LSSG
      - "lssg_channel_ms": Channel-wise LSSG with Multi-Scale Gate
      - "lssg_channel_context": Channel-wise LSSG with Multi-Scale + Global Context
      - "lssg_refined": LSSG + Conv Refinement
      - "lssg_se": LSSG with SE (Global Average Pooling)
      - "lssg_dw": LSSG with Depthwise Conv Local Awareness
      - "lssg_swiglu": LSSG with SwiGLU Multiplicative Fusion
    """

    def __init__(
        self,
        encoder,
        num_classes: int,
        n_conv_per_stage,
        deep_supervision: bool,
        skip_mode: str = "attention",
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        attn_dim: int = 64,
        num_heads: int = 4,
        max_tokens: int = 256,
        gated_decoder_stages: List[int] = None,
        use_skip_mamba: bool = False,
        use_decoder_mamba: bool = False,
        decoder_mamba_block_type: str = "safe",
        mamba_residual_scale_init: float = 0.0,
        use_vector_alpha: bool = False,
    ):
        super().__init__()

        self.deep_supervision = bool(deep_supervision)
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.skip_mode = skip_mode.lower()
        self.gated_decoder_stages = gated_decoder_stages
        self.decoder_mamba_block_type = decoder_mamba_block_type
        self.use_vector_alpha = use_vector_alpha

        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        self.use_skip_mamba = use_skip_mamba
        self.use_decoder_mamba = use_decoder_mamba

        stages = []
        upsample_layers = []
        seg_layers = []
        skip_processors = []
        self.skip_mamba_blocks = nn.ModuleList() if use_skip_mamba else None
        self.decoder_mamba_blocks = nn.ModuleList() if use_decoder_mamba else None

        if use_skip_mamba or use_decoder_mamba:
            try:
                from models.IQU_EncoderBiMamba import SafeBiMambaEncoderBlock1D
            except ImportError:
                raise ImportError("use_skip_mamba or use_decoder_mamba requires SafeBiMambaEncoderBlock1D.")

        # skip channels excluding bottleneck, ordered shallow -> deep
        encoder_skip_channels = list(encoder.output_channels[:-1])

        # In this decoder design, after upsample, dec channels match skip channels
        dec_channels_for_bridge = encoder_skip_channels

        if self.skip_mode == "uct":
            self.uct_bridge = UCTransNetLiteSkipBridge1D(
                skip_channels=encoder_skip_channels,
                dec_channels=dec_channels_for_bridge,
                context_dim=128,
                residual_scale_init=residual_scale_init,
            )
        else:
            self.uct_bridge = None

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s][0]

            upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode="nearest",
                )
            )

            if self.skip_mode == "attention":
                skip_processors.append(
                    AttentionGateSkip1D(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                    )
                )
            elif self.skip_mode == "dca":
                skip_processors.append(
                    DCALiteSkip1D(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        attn_dim=attn_dim,
                        num_heads=num_heads,
                        max_tokens=max_tokens,
                        residual_scale_init=residual_scale_init,
                    )
                )
            elif self.skip_mode == "uct":
                # handled by self.uct_bridge
                skip_processors.append(nn.Identity())
            elif self.skip_mode == "lssg":
                skip_processors.append(
                    LeakageSuppressedSkipGate1D(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            elif self.skip_mode == "lssg_channel":
                skip_processors.append(
                    ChannelWiseLSSG1D(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            elif self.skip_mode == "lssg_channel_ms":
                skip_processors.append(
                    ChannelWiseLSSG1D_MS(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        global_ctx_channels=global_ctx_channels,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            elif self.skip_mode == "lssg_channel_context":
                skip_processors.append(
                    ChannelWiseLSSG1D_Context(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        global_ctx_channels=global_ctx_channels,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            elif self.skip_mode == "lssg_refined":
                skip_processors.append(
                    RefinedLSSG1D(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        use_complex_mask=getattr(encoder, "use_complex_mask", False),
                    )
                )
            elif self.skip_mode == "lssg_se":
                skip_processors.append(
                    ChannelWiseLSSG1D_SE(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            elif self.skip_mode == "lssg_swiglu":
                skip_processors.append(
                    ChannelWiseLSSG1D_SwiGLU(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            elif self.skip_mode == "lssg_dw":
                skip_processors.append(
                    ChannelWiseLSSG1D_Depthwise(
                        skip_channels=input_features_skip,
                        dec_channels=input_features_skip,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        residual_scale_init=residual_scale_init,
                        use_vector_alpha=self.use_vector_alpha,
                    )
                )
            else:
                skip_processors.append(nn.Identity())

            if self.use_skip_mamba:
                self.skip_mamba_blocks.append(
                    SafeBiMambaEncoderBlock1D(
                        channels=input_features_skip,
                        residual_scale_init=mamba_residual_scale_init,
                    )
                )

            blocks = [
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=True,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
            ]

            blocks.extend(
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=False,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
                for _ in range(n_conv_per_stage[s - 1] - 1)
            )

            if self.use_decoder_mamba:
                if self.decoder_mamba_block_type == "original":
                    from models.IQUBiMamba1D import BiMambaLayer
                    self.decoder_mamba_blocks.append(
                        BiMambaLayer(
                            dim=input_features_skip,
                            channel_token=False,
                        )
                    )
                elif self.decoder_mamba_block_type == "unidirectional":
                    from models.IQUMamba1D import MambaLayer
                    self.decoder_mamba_blocks.append(
                        MambaLayer(
                            dim=input_features_skip,
                            channel_token=False,
                        )
                    )
                else:
                    self.decoder_mamba_blocks.append(
                        SafeBiMambaEncoderBlock1D(
                            channels=input_features_skip,
                            residual_scale_init=mamba_residual_scale_init,
                        )
                    )

            stages.append(nn.Sequential(*blocks))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, kernel_size=1))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.skip_processors = nn.ModuleList(skip_processors)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: List[torch.Tensor], global_ctx: torch.Tensor = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = skips[-1]
        seg_outputs = []
        self.aux_loss = 0.0

        if self.skip_mode == "uct":
            uct_context = self.uct_bridge.build_context(skips[:-1])
        else:
            uct_context = None

        n_stages_encoder = len(skips)

        for s in range(len(self.stages)):
            s_idx = s
            x = self.upsample_layers[s_idx](x)

            skip = skips[-(s_idx + 2)]

            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)

            if self.skip_mode == "uct":
                skip_level = n_stages_encoder - s_idx - 2
                processed_skip = self.uct_bridge(
                    skip=skip,
                    dec=x,
                    level=skip_level,
                    context=uct_context,
                )
            else:
                if self.use_skip_mamba:
                    skip = self.skip_mamba_blocks[s_idx](skip)
                
                if self.gated_decoder_stages is None or s_idx in self.gated_decoder_stages:
                    try:
                        processed_skip = self.skip_processors[s_idx](skip, x, global_ctx)
                    except TypeError:
                        processed_skip = self.skip_processors[s_idx](skip, x)
                    
                    if hasattr(self.skip_processors[s_idx], 'last_gate'):
                        self.aux_loss = self.aux_loss + self.skip_processors[s_idx].last_gate.abs().mean()
                else:
                    processed_skip = skip

            x = torch.cat((x, processed_skip), dim=1)
            x = self.stages[s_idx](x)
            
            if self.use_decoder_mamba:
                x = self.decoder_mamba_blocks[s_idx](x)

            seg_outputs.append(self.seg_layers[s_idx](x))

        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQUResUNet1D_SkipEnhanced(nn.Module):
    """
    ResUNet with one of three medical-segmentation-inspired skip modules.

    skip_mode:
      - "attention": Attention U-Net style
      - "uct": UCTransNet/UDTransNet-lite style
      - "dca": DCA-lite style
      - "lssg": Leakage-Suppressed Skip Gate
      - "lssg_channel": Channel-wise LSSG
      - "lssg_channel_ms": Channel-wise LSSG with Multi-Scale Gate
      - "lssg_channel_context": Channel-wise LSSG with Multi-Scale + Global Context
      - "lssg_refined": LSSG + Conv Refinement
    """

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
        skip_mode: str = "attention",
        gated_decoder_stages: List[int] = None,
        residual_scale_init: float = 0.1,
        global_ctx_channels: int = None,
        attn_dim: int = 64,
        num_heads: int = 4,
        max_tokens: int = 256,
        use_complex_mask: bool = False,
        use_mamba_stages: List[bool] = None,
        mamba_residual_scale_init: float = 0.0,
        encoder_mamba_block_type: str = "safe",
        decoder_mamba_block_type: str = "safe",
        use_skip_mamba: bool = False,
        use_decoder_mamba: bool = False,
        use_vector_alpha: bool = False,
        **kwargs,
    ):
        super().__init__()

        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}

        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.use_complex_mask = use_complex_mask

        self.encoder = ResidualConvEncoder(
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
        )

        if use_mamba_stages is not None and any(use_mamba_stages):
            from models.IQU_EncoderBiMamba import EncoderWithBiMamba1D
            self.encoder = EncoderWithBiMamba1D(
                encoder=self.encoder,
                use_mamba_stages=use_mamba_stages,
                expand=2,
                residual_scale_init=mamba_residual_scale_init,
                block_type=encoder_mamba_block_type,
            )

        self.decoder = SkipEnhancedUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            skip_mode=skip_mode,
            gated_decoder_stages=gated_decoder_stages,
            residual_scale_init=residual_scale_init,
            global_ctx_channels=global_ctx_channels,
            attn_dim=attn_dim,
            num_heads=num_heads,
            max_tokens=max_tokens,
            use_skip_mamba=use_skip_mamba,
            use_decoder_mamba=use_decoder_mamba,
            decoder_mamba_block_type=decoder_mamba_block_type,
            mamba_residual_scale_init=mamba_residual_scale_init,
            use_vector_alpha=use_vector_alpha,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        out = self.decoder(skips)
        
        if self.use_complex_mask:
            from models.IQUResUNet1D_WLComplex import apply_complex_mask, bound_complex_mask
            if self.decoder.deep_supervision and isinstance(out, list):
                res = []
                for m in out:
                    if m.shape[-1] != x.shape[-1]:
                        m = F.interpolate(m, size=x.shape[-1], mode="linear", align_corners=False)
                    res.append(apply_complex_mask(x, bound_complex_mask(m, scale=2.0)))
                return res
            return apply_complex_mask(x, bound_complex_mask(out, scale=2.0))
            
        return out
