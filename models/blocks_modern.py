import torch
import torch.nn as nn
from mamba_ssm import Mamba

class LayerScale1D(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(init_values * torch.ones(1, dim, 1))

    def forward(self, x):
        return x * self.weight


class ConvNeXt1DBlock(nn.Module):
    """
    1D adaptation of the ConvNeXt block.
    Depthwise Conv -> Norm -> Pointwise -> GELU -> Pointwise -> LayerScale -> Add
    """
    def __init__(self, channels: int, expansion: int = 4, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, groups=channels)
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.pwconv1 = nn.Conv1d(channels, channels * expansion, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(channels * expansion, channels, 1)
        self.layer_scale = LayerScale1D(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.layer_scale(x)
        return res + x


class SpatialGatingUnit1D(nn.Module):
    """
    1D adaptation of SegNeXt's Spatial Gating Unit (MSCAN).
    Cascades large-kernel dilated depthwise convolutions for multi-scale attention.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.dw5 = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dwd7 = nn.Conv1d(channels, channels, kernel_size=7, padding=6, dilation=2, groups=channels)
        self.dwd11 = nn.Conv1d(channels, channels, kernel_size=11, padding=15, dilation=3, groups=channels)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.dw5(x)
        attn = self.dwd7(attn)
        attn = self.dwd11(attn)
        attn = self.proj(attn)
        return u * attn


class MSCAN1DBlock(nn.Module):
    """
    1D Multi-Scale Convolutional Attention Block (SegNeXt style).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj_1 = nn.Conv1d(channels, channels, 1)
        self.spatial_gating_unit = SpatialGatingUnit1D(channels)
        self.proj_2 = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.proj_1(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class HybridCNNBiMambaBlock(nn.Module):
    """
    U-Mamba / LKM-UNet style block.
    Local features via ConvNeXt1DBlock, global context via BiMamba.
    Fused with zero-initialized residual scale.
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.conv_branch = ConvNeXt1DBlock(channels)
        
        self.mamba_fwd = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Zero initialization for stability
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        res = x
        
        x_conv = self.conv_branch(x)
        
        x_trans = x.transpose(1, 2) # [B, L, C]
        
        # BiMamba processing
        x_fwd = self.mamba_fwd(x_trans)
        x_trans_rev = torch.flip(x_trans, dims=[1])
        x_bwd = self.mamba_bwd(x_trans_rev)
        x_bwd = torch.flip(x_bwd, dims=[1])
        
        x_mamba = (x_fwd + x_bwd).transpose(1, 2) # [B, C, L]
        
        return res + self.alpha * (x_conv + x_mamba)
