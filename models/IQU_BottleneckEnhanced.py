from typing import List, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D import ResidualConvEncoder
from models.IQUResUNet1D_InnovationBase import PlainUNetResDecoder
from models.IQUResUNet1D_SkipEnhanced import SkipEnhancedUNetResDecoder
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d

# ============================================================
# 1. Symbol-Rate-Aware Dilated TCN Bottleneck
# ============================================================
class SRATCNBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # Depthwise
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        # Pointwise with GLU
        self.pwconv1 = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pwconv2 = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x, gate = x.chunk(2, dim=1)
        x = x * torch.sigmoid(gate)
        x = self.pwconv2(x)
        return x + res

class SRATCNBottleneck1D(nn.Module):
    def __init__(self, channels: int, dilations: List[int] = [1, 2, 4, 5, 10, 20]):
        super().__init__()
        self.blocks = nn.ModuleList([
            SRATCNBlock1D(channels, d) for d in dilations
        ])
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x is the raw input [B, 2, L], z is the bottleneck feature [B, C, T]
        # Ignore x for this bottleneck
        for block in self.blocks:
            z = block(z)
        return z

# ============================================================
# 2. Complex ASPP-1D Bottleneck
# ============================================================
class RealASPPBottleneck1D(nn.Module):
    def __init__(self, channels: int, dilations: List[int] = [1, 2, 4, 8, 16], scale_init: float = 0.05, use_sk_routing: bool = False):
        super().__init__()
        self.channels = channels
        self.use_sk_routing = use_sk_routing
        
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d)
            )
            
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Conv1d(channels, channels, kernel_size=1)
        
        self.num_branches = len(dilations) + 1
        
        if self.use_sk_routing:
            # SK routing attention
            self.sk_pool = nn.AdaptiveAvgPool1d(1)
            sk_hidden = max(32, channels // 4)
            self.sk_attn = nn.Sequential(
                nn.Conv1d(channels, sk_hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(sk_hidden, channels * self.num_branches, kernel_size=1)
            )
        else:
            self.out_proj = nn.Conv1d(channels * self.num_branches, channels, kernel_size=1)
            self.gate = nn.Sequential(
                nn.Conv1d(channels * self.num_branches, channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(channels, channels, kernel_size=1),
                nn.Sigmoid(),
            )
        
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, T]
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(z))
            
        # Global context
        g_ctx = self.global_proj(self.global_pool(z))
        branch_outputs.append(g_ctx.expand_as(z))
        
        if self.use_sk_routing:
            # Stack branches for SK routing: [B, num_branches, C, T]
            stacked = torch.stack(branch_outputs, dim=1)
            
            # Compute fused representation for routing
            fused = sum(branch_outputs)  # [B, C, T]
            
            # SK weights
            sk_w = self.sk_attn(self.sk_pool(fused))  # [B, C * num_branches, 1]
            sk_w = sk_w.view(-1, self.num_branches, self.channels, 1)  # [B, num_branches, C, 1]
            sk_w = F.softmax(sk_w, dim=1)  # Softmax over branches
            
            # Weighted sum
            out = (stacked * sk_w).sum(dim=1)  # [B, C, T]
            
            delta = out
        else:
            cat_z = torch.cat(branch_outputs, dim=1) # [B, C * num_branches, T]
            gate = self.gate(cat_z) # [B, C, T]
            
            proj_z = self.out_proj(cat_z) # [B, C, T]
            
            delta = proj_z * gate
        return z + self.scale * delta

# ============================================================
# 3. Dual-Domain Cyclo Context Bottleneck (DCCB)
# ============================================================
class DualDomainCycloContextBottleneck1D(nn.Module):
    def __init__(self, channels: int, lags: List[int] = [0, 1, 2, 4, 8, 16], scale_init: float = 0.05, cyclo_scale_init: float = 0.05):
        super().__init__()
        self.cyclo_scale = nn.Parameter(torch.tensor(float(cyclo_scale_init)))
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        
        self.lags = lags
        num_cyclo_channels = 2 * len(lags) # Re and Im for each lag
        
        # We need to downsample from L=4096 to T=256, which is a factor of 16.
        # Strided convolutions for smoothing downsampling.
        self.cyclo_encoder = nn.Sequential(
            nn.Conv1d(num_cyclo_channels, channels // 4, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(channels // 4, affine=True),
            nn.SiLU(),
            nn.Conv1d(channels // 4, channels // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(channels // 2, affine=True),
            nn.SiLU(),
            nn.Conv1d(channels // 2, channels, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(channels, affine=True),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=5, stride=2, padding=2),
        )
        
        self.gate_cyclo = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, T]
        z_time = self.time_branch(x, z)
        
        # x: [B, 2, L]
        B, C_in, L = x.shape
        
        cyclo_features = []
        for d in self.lags:
            if d == 0:
                r_d = x * x # [B, C_in, L]
            else:
                if L <= d:
                    continue
                
                # Independent real autocorrelation for each channel
                r_d_shifted = x[:, :, d:] * x[:, :, :-d]
                r_d = F.pad(r_d_shifted, (0, d))
            
            cyclo_features.append(r_d)
            
        cyclo_in = torch.cat(cyclo_features, dim=1) # [B, C_in * len(lags), L]
        
        z_cyclo = self.cyclo_encoder(cyclo_in) # [B, C, T]
        z_cyclo = F.interpolate(z_cyclo, size=z.size(-1), mode="linear", align_corners=False)
        
        # Soft gating
        gate = self.gate_cyclo(torch.cat([z_time, z_cyclo], dim=1))
        
        return z_time + self.cyclo_scale * gate * z_cyclo



class DualDomainCycloContextBottleneck1D_DeepCyclo(nn.Module):
    """
    Stage 163: DCCB with Deep Cyclo Encoder.
    ANTI-OVERFITTING: max width channels//2, dropout in encoder.
    """
    def __init__(self, channels: int, lags: List[int] = [0, 1, 2, 4, 8, 16], scale_init: float = 0.05, cyclo_scale_init: float = 0.05):
        super().__init__()
        self.cyclo_scale = nn.Parameter(torch.tensor(float(cyclo_scale_init)))
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        self.lags = lags
        
        self.deep_proj = nn.Conv1d(2, 8, kernel_size=1, bias=False)
        num_cyclo_channels = 8 * len(lags)
        
        c_hidden = max(channels // 2, 16) # Reduced from channels
        
        self.cyclo_encoder = nn.Sequential(
            nn.Conv1d(num_cyclo_channels, c_hidden // 4, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(c_hidden // 4, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden // 4, c_hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden // 2, affine=True), nn.SiLU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(c_hidden // 2, c_hidden, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden, affine=True), nn.SiLU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(c_hidden, channels, kernel_size=5, stride=2, padding=2),
        )
        self.gate_cyclo = nn.Sequential(nn.Conv1d(channels * 2, channels, kernel_size=1), nn.SiLU(), nn.Dropout1d(0.1), nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_time = self.time_branch(x, z)
        B, _, L = x.shape
        x_deep = self.deep_proj(x)
        cyclo_features = []
        for d in self.lags:
            if d == 0:
                r_d = x_deep * x_deep
            else:
                if L <= d: continue
                r_d_shifted = x_deep[:, :, d:] * x_deep[:, :, :-d]
                r_d = F.pad(r_d_shifted, (0, d))
            cyclo_features.append(r_d)
            
        cyclo_in = torch.cat(cyclo_features, dim=1)
        z_cyclo = self.cyclo_encoder(cyclo_in)
        z_cyclo = F.interpolate(z_cyclo, size=z.size(-1), mode="linear", align_corners=False)
        gate = self.gate_cyclo(torch.cat([z_time, z_cyclo], dim=1))
        return z_time + self.cyclo_scale * gate * z_cyclo

class DualDomainCycloContextBottleneck1D_CrossAttention(nn.Module):
    """
    Stage 164: DCCB with Cross Attention.
    ANTI-OVERFITTING: dropout in attention and narrower projections.
    """
    def __init__(self, channels: int, lags: List[int] = [0, 1, 2, 4, 8, 16], scale_init: float = 0.05, cyclo_scale_init: float = 0.05):
        super().__init__()
        self.cyclo_scale = nn.Parameter(torch.tensor(float(cyclo_scale_init)))
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        self.lags = lags
        num_cyclo_channels = 2 * len(lags)
        
        c_hidden = max(channels // 2, 16)
        
        self.cyclo_encoder = nn.Sequential(
            nn.Conv1d(num_cyclo_channels, c_hidden // 4, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(c_hidden // 4, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden // 4, c_hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden // 2, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden // 2, c_hidden, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden, channels, kernel_size=5, stride=2, padding=2),
        )
        
        self.q_proj = nn.Conv1d(channels, c_hidden, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, c_hidden, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, c_hidden, kernel_size=1)
        self.attn_drop = nn.Dropout(0.1)
        self.out_proj = nn.Sequential(nn.Conv1d(c_hidden, channels, kernel_size=1), nn.Dropout1d(0.1))
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_time = self.time_branch(x, z)
        B, C_in, L = x.shape
        cyclo_features = []
        for d in self.lags:
            if d == 0:
                r_d = x * x
            else:
                if L <= d: continue
                r_d_shifted = x[:, :, d:] * x[:, :, :-d]
                r_d = F.pad(r_d_shifted, (0, d))
            cyclo_features.append(r_d)
            
        cyclo_in = torch.cat(cyclo_features, dim=1)
        z_cyclo = self.cyclo_encoder(cyclo_in)
        z_cyclo = F.interpolate(z_cyclo, size=z.size(-1), mode="linear", align_corners=False)
        
        Q = self.q_proj(z_time).transpose(1, 2)
        K = self.k_proj(z_cyclo).transpose(1, 2)
        V = self.v_proj(z_cyclo).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out_attn = torch.matmul(attn, V).transpose(1, 2)
        out_attn = self.out_proj(out_attn)
        
        return z_time + self.cyclo_scale * out_attn

class DualDomainCycloContextBottleneck1D_AdaptiveLags(nn.Module):
    """
    Stage 165: DCCB with Adaptive Lags.
    ANTI-OVERFITTING: max width channels//2, dropout in encoder and lag weights.
    """
    def __init__(self, channels: int, lags: List[int] = [0, 1, 2, 4, 8, 16], scale_init: float = 0.05, cyclo_scale_init: float = 0.05):
        super().__init__()
        self.cyclo_scale = nn.Parameter(torch.tensor(float(cyclo_scale_init)))
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        self.lags = lags
        num_cyclo_channels = 2 * len(lags)
        
        self.lag_se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout1d(0.1),
            nn.Conv1d(num_cyclo_channels, num_cyclo_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_cyclo_channels // 2, num_cyclo_channels, 1),
            nn.Sigmoid()
        )
        
        c_hidden = max(channels // 2, 16)
        self.cyclo_encoder = nn.Sequential(
            nn.Conv1d(num_cyclo_channels, c_hidden // 4, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(c_hidden // 4, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden // 4, c_hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden // 2, affine=True), nn.SiLU(),
            nn.Dropout1d(0.1),
            nn.Conv1d(c_hidden // 2, c_hidden, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden, channels, kernel_size=5, stride=2, padding=2),
        )
        self.gate_cyclo = nn.Sequential(nn.Conv1d(channels * 2, channels, kernel_size=1), nn.SiLU(), nn.Dropout1d(0.1), nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_time = self.time_branch(x, z)
        B, C_in, L = x.shape
        cyclo_features = []
        for d in self.lags:
            if d == 0:
                r_d = x * x
            else:
                if L <= d: continue
                r_d_shifted = x[:, :, d:] * x[:, :, :-d]
                r_d = F.pad(r_d_shifted, (0, d))
            cyclo_features.append(r_d)
            
        cyclo_in = torch.cat(cyclo_features, dim=1)
        lag_weights = self.lag_se(cyclo_in)
        cyclo_in = cyclo_in * lag_weights
        
        z_cyclo = self.cyclo_encoder(cyclo_in)
        z_cyclo = F.interpolate(z_cyclo, size=z.size(-1), mode="linear", align_corners=False)
        gate = self.gate_cyclo(torch.cat([z_time, z_cyclo], dim=1))
        return z_time + self.cyclo_scale * gate * z_cyclo

class DualDomainCycloContextBottleneck1D_MambaEncoder(nn.Module):
    """
    Stage 166: DCCB with Mamba Encoder.
    ANTI-OVERFITTING: max width channels//2, Mamba expand=1, Dropout.
    """
    def __init__(self, channels: int, lags: List[int] = [0, 1, 2, 4, 8, 16], scale_init: float = 0.05, cyclo_scale_init: float = 0.05):
        super().__init__()
        self.cyclo_scale = nn.Parameter(torch.tensor(float(cyclo_scale_init)))
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        self.lags = lags
        num_cyclo_channels = 2 * len(lags)
        
        c_hidden = max(channels // 2, 16)
        self.cyclo_encoder = nn.Sequential(
            nn.Conv1d(num_cyclo_channels, c_hidden // 4, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(c_hidden // 4, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden // 4, c_hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden // 2, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden // 2, c_hidden, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(c_hidden, affine=True), nn.SiLU(),
            nn.Conv1d(c_hidden, channels, kernel_size=5, stride=2, padding=2),
        )
        
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=1) # REDUCED expand
        except ImportError:
            self.mamba = None
            
        self.mamba_norm = nn.InstanceNorm1d(channels, affine=True)
        self.mamba_drop = nn.Dropout1d(0.1)
        self.gate_cyclo = nn.Sequential(nn.Conv1d(channels * 2, channels, kernel_size=1), nn.SiLU(), nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_time = self.time_branch(x, z)
        B, C_in, L = x.shape
        cyclo_features = []
        for d in self.lags:
            if d == 0:
                r_d = x * x
            else:
                if L <= d: continue
                r_d_shifted = x[:, :, d:] * x[:, :, :-d]
                r_d = F.pad(r_d_shifted, (0, d))
            cyclo_features.append(r_d)
            
        cyclo_in = torch.cat(cyclo_features, dim=1)
        z_cyclo = self.cyclo_encoder(cyclo_in)
        z_cyclo = F.interpolate(z_cyclo, size=z.size(-1), mode="linear", align_corners=False)
        
        if self.mamba is not None:
            z_mamba_in = self.mamba_norm(z_cyclo).transpose(1, 2)
            z_mamba_out = self.mamba(z_mamba_in).transpose(1, 2)
            z_cyclo = z_cyclo + self.mamba_drop(z_mamba_out)
            
        gate = self.gate_cyclo(torch.cat([z_time, z_cyclo], dim=1))
        return z_time + self.cyclo_scale * gate * z_cyclo

class DualPathMambaBottleneck1D(nn.Module):
    """
    Strategy D: Dual-Path Mamba Bottleneck
    Unfolds the 1D sequence into chunks, processes intra-chunk and inter-chunk
    using BiMamba layers to model extremely long-range dependencies efficiently.
    """
    def __init__(self, channels: int, chunk_size: int = 16, expand: int = 2):
        super().__init__()
        self.chunk_size = chunk_size
        try:
            from mamba_ssm import Mamba
        except ImportError:
            Mamba = None

        if Mamba is None:
            raise ImportError("mamba_ssm is required for DualPathMambaBottleneck1D.")

        self.intra_fwd = Mamba(d_model=channels, d_state=16, d_conv=4, expand=expand)
        self.intra_bwd = Mamba(d_model=channels, d_state=16, d_conv=4, expand=expand)
        self.intra_norm = nn.InstanceNorm1d(channels, affine=True)
        
        self.inter_fwd = Mamba(d_model=channels, d_state=16, d_conv=4, expand=expand)
        self.inter_bwd = Mamba(d_model=channels, d_state=16, d_conv=4, expand=expand)
        self.inter_norm = nn.InstanceNorm1d(channels, affine=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, L]
        B, C, L = z.shape
        pad_len = (self.chunk_size - (L % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            z_pad = F.pad(z, (0, pad_len))
        else:
            z_pad = z

        # Unfold into chunks: [B, C, K, S] where S=chunk_size
        S = self.chunk_size
        K = z_pad.shape[-1] // S
        z_unfolded = z_pad.view(B, C, K, S)

        # Intra-chunk Mamba
        # Reshape to [B*K, C, S], then transpose for Mamba -> [B*K, S, C]
        intra_in = z_unfolded.permute(0, 2, 1, 3).reshape(B * K, C, S)
        intra_in_norm = self.intra_norm(intra_in).transpose(1, 2)
        y_intra_fwd = self.intra_fwd(intra_in_norm)
        y_intra_bwd = torch.flip(self.intra_bwd(torch.flip(intra_in_norm, dims=[1])), dims=[1])
        y_intra = (y_intra_fwd + y_intra_bwd).transpose(1, 2).reshape(B, K, C, S).permute(0, 2, 1, 3)
        z_unfolded = z_unfolded + y_intra

        # Inter-chunk Mamba
        # Process across K for each S: [B*S, C, K] -> [B*S, K, C]
        inter_in = z_unfolded.permute(0, 3, 1, 2).reshape(B * S, C, K)
        inter_in_norm = self.inter_norm(inter_in).transpose(1, 2)
        y_inter_fwd = self.inter_fwd(inter_in_norm)
        y_inter_bwd = torch.flip(self.inter_bwd(torch.flip(inter_in_norm, dims=[1])), dims=[1])
        y_inter = (y_inter_fwd + y_inter_bwd).transpose(1, 2).reshape(B, S, C, K).permute(0, 2, 3, 1)
        z_unfolded = z_unfolded + y_inter

        # Fold back
        out = z_unfolded.reshape(B, C, -1)
        if pad_len > 0:
            out = out[..., :-pad_len]
            
        return out

# ============================================================
# 4. Bottleneck Enhanced Wrapper
# ============================================================
class IQUResUNet1D_BottleneckEnhanced(nn.Module):
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
        use_complex_mask: bool = False,
        bottleneck_mode: str = "sra_tcn",
        skip_mode: str = None,
        gated_decoder_stages: List[int] = None,
        use_mamba_stages: List[bool] = None,
        encoder_mamba_block_type: str = "safe",
        use_decoder_mamba: bool = False,
        decoder_mamba_block_type: str = "safe",
        mamba_residual_scale_init: float = 0.0,
        residual_scale_init: float = 0.1,
        use_sk_routing: bool = False,
        use_phase_aware_context: bool = False,
        use_bottom_up_leakage: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.use_sk_routing = use_sk_routing
        self.use_phase_aware_context = use_phase_aware_context
        self.use_bottom_up_leakage = use_bottom_up_leakage
        
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
        
        bottleneck_channels = features_per_stage[-1]
        
        if bottleneck_mode == "sra_tcn":
            self.bottleneck = SRATCNBottleneck1D(bottleneck_channels)
        elif bottleneck_mode == "caspp":
            self.bottleneck = RealASPPBottleneck1D(bottleneck_channels, use_sk_routing=self.use_sk_routing)
        elif bottleneck_mode == "dccb":
            self.bottleneck = DualDomainCycloContextBottleneck1D(bottleneck_channels)
        elif bottleneck_mode == "dual_path_mamba":
            self.bottleneck = DualPathMambaBottleneck1D(bottleneck_channels)
        elif bottleneck_mode == "dccb_deep":
            self.bottleneck = DualDomainCycloContextBottleneck1D_DeepCyclo(bottleneck_channels)
        elif bottleneck_mode == "dccb_cross_attn":
            self.bottleneck = DualDomainCycloContextBottleneck1D_CrossAttention(bottleneck_channels)
        elif bottleneck_mode == "dccb_adaptive_lags":
            self.bottleneck = DualDomainCycloContextBottleneck1D_AdaptiveLags(bottleneck_channels)
        elif bottleneck_mode == "dccb_mamba":
            self.bottleneck = DualDomainCycloContextBottleneck1D_MambaEncoder(bottleneck_channels)
        else:
            raise ValueError(f"Unknown bottleneck_mode: {bottleneck_mode}")
            
        if bottleneck_mode in ["dccb", "dccb_deep", "dccb_cross_attn", "dccb_adaptive_lags", "dccb_mamba", "caspp", "sra_tcn"]:
            self.bottleneck_adapter = nn.Sequential(
                nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=1),
                nn.GroupNorm(min(8, bottleneck_channels // 4) if bottleneck_channels >= 32 else 1, bottleneck_channels),
                nn.GELU()
            )
            self.adapter_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.bottleneck_adapter = nn.Identity()
            
        # Bottom-up leakage: send shallowest features to the bottleneck
        if self.use_bottom_up_leakage:
            self.bottom_up_proj = nn.Conv1d(features_per_stage[0], bottleneck_channels, kernel_size=1)
            self.bottom_up_scale = nn.Parameter(torch.tensor(0.05))
            
        if skip_mode is not None and skip_mode.lower() != "none":
            self.decoder = SkipEnhancedUNetResDecoder(
                encoder=self.encoder,
                num_classes=num_classes,
                n_conv_per_stage=n_conv_per_stage_decoder,
                deep_supervision=deep_supervision,
                skip_mode=skip_mode,
                gated_decoder_stages=gated_decoder_stages,
                use_decoder_mamba=use_decoder_mamba,
                decoder_mamba_block_type=decoder_mamba_block_type,
                mamba_residual_scale_init=mamba_residual_scale_init,
                residual_scale_init=residual_scale_init,
                global_ctx_channels=bottleneck_channels * 2 if self.use_phase_aware_context else bottleneck_channels,
            )
        else:
            self.decoder = PlainUNetResDecoder(
                encoder=self.encoder,
                num_classes=num_classes,
                n_conv_per_stage=n_conv_per_stage_decoder,
                deep_supervision=deep_supervision,
            )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        
        # Bottom-up leakage from the highest-resolution skip
        if self.use_bottom_up_leakage:
            z_shallow = skips[0]
            z_shallow = F.adaptive_avg_pool1d(z_shallow, skips[-1].size(-1))
            z_shallow = self.bottom_up_proj(z_shallow)
            
            # Add to bottleneck input
            skips[-1] = skips[-1] + self.bottom_up_scale * z_shallow
        
        # Apply bottleneck to the deepest feature map
        skips[-1] = self.bottleneck(x, skips[-1])
        
        # Apply adapter to align feature distribution for the Decoder/Skip Gates
        if hasattr(self, 'bottleneck_adapter') and not isinstance(self.bottleneck_adapter, nn.Identity):
            z_bn = self.bottleneck_adapter(skips[-1])
            if hasattr(self, 'adapter_scale'):
                skips[-1] = skips[-1] + self.adapter_scale * z_bn
            else:
                skips[-1] = z_bn
                
        # Extract phase-aware global context from the bottleneck feature map
        if self.use_phase_aware_context:
            g_mean = skips[-1].mean(dim=-1, keepdim=True)
            g_std = skips[-1].std(dim=-1, keepdim=True)
            global_ctx = torch.cat([g_mean, g_std], dim=1) # [B, C_bn * 2, 1]
        else:
            global_ctx = skips[-1].mean(dim=-1, keepdim=True) # [B, C_bn, 1]
            
        if isinstance(self.decoder, PlainUNetResDecoder):
            out = self.decoder(skips)
        else:
            out = self.decoder(skips, global_ctx=global_ctx)        
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
