import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from mamba_ssm import Mamba
from typing import List, Union

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


def get_norm_layer(norm_type: str, out_channels: int):
    """Factory for 1D normalization layers suited for communication signals."""
    if norm_type == "instance":
        return nn.InstanceNorm1d(out_channels, eps=1e-5, affine=True)
    elif norm_type == "group":
        # Standard GroupNorm (default groups=8)
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, out_channels)
    elif norm_type == "layer":
        # LayerNorm over (channels, length) is equivalent to GroupNorm with 1 group
        return nn.GroupNorm(1, out_channels)
    elif norm_type == "none" or norm_type is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class BambaBlock(nn.Module):
    """Bidirectional pre-norm residual Mamba block.
    
    Expected input shape: [B, C, L]
    Output shape: [B, C, L]
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion: str = "proj",
        num_layers: int = 1,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.fusion = fusion
        self.num_layers = num_layers

        if num_layers == 0:
            self.is_identity = True
            return
        else:
            self.is_identity = False

        self.norms_fwd = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        self.mambas_fwd = nn.ModuleList([
            Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        self.norms_bwd = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        self.mambas_bwd = nn.ModuleList([
            Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        if fusion == "proj":
            self.out_proj = nn.Linear(dim * 2, dim)
            nn.init.xavier_uniform_(self.out_proj.weight, gain=0.01)
            nn.init.zeros_(self.out_proj.bias)
        elif fusion == "sum":
            self.out_proj = None
        else:
            raise ValueError(f"Unknown fusion mode: {fusion}")

        self.res_scale = nn.Parameter(torch.ones(1) * residual_scale_init)

    def forward(self, x):
        if self.is_identity:
            return x

        # [B, C, L] -> [B, L, C]
        x_flat = x.transpose(1, 2).contiguous()

        # Forward scan stack (with pre-norm residual)
        yf = x_flat
        for norm, mamba in zip(self.norms_fwd, self.mambas_fwd):
            yf = yf + mamba(norm(yf))

        # Backward scan stack (with pre-norm residual)
        yb = x_flat.flip(dims=[1]).contiguous()
        for norm, mamba in zip(self.norms_bwd, self.mambas_bwd):
            yb = yb + mamba(norm(yb))
        yb = yb.flip(dims=[1]).contiguous()

        # Extract only the Mamba contribution (deltas)
        df = yf - x_flat
        db = yb - x_flat

        # Fusion
        if self.fusion == "sum":
            delta = 0.5 * (df + db)
        else:
            delta = self.out_proj(torch.cat([df, db], dim=-1))

        # Block-level residual connection with learnable scale
        out = x_flat + self.res_scale * delta
        return out.transpose(1, 2).contiguous()


class DownStage(nn.Module):
    """Downsampling stage of the U-Net.
    
    Conv1d(stride=stride) -> norm/act -> BambaBlock (optional)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion: str = "proj",
        num_layers: int = 1,
        residual_scale_init: float = 0.1,
        use_bamba: bool = True,
        norm_type: str = "instance",
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.norm = get_norm_layer(norm_type, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        if use_bamba:
            self.bamba = BambaBlock(
                dim=out_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                fusion=fusion,
                num_layers=num_layers,
                residual_scale_init=residual_scale_init,
            )
        else:
            self.bamba = nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.bamba(x)
        return x


class UpStage(nn.Module):
    """Upsampling stage of the U-Net.
    
    ConvTranspose1d -> skip concat -> 1x1 Conv -> norm/act -> BambaBlock (optional)
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion: str = "proj",
        num_layers: int = 1,
        residual_scale_init: float = 0.1,
        use_bamba: bool = True,
        norm_type: str = "instance",
    ):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.conv1x1 = nn.Conv1d(out_channels + skip_channels, out_channels, kernel_size=1, bias=True)
        self.norm = get_norm_layer(norm_type, out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        if use_bamba:
            self.bamba = BambaBlock(
                dim=out_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                fusion=fusion,
                num_layers=num_layers,
                residual_scale_init=residual_scale_init,
            )
        else:
            self.bamba = nn.Identity()
        
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Length matching in case of slight size differences
        if x.shape[-1] != skip.shape[-1]:
            if x.shape[-1] < skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            else:
                x = x[..., :skip.shape[-1]]
                
        x = torch.cat([x, skip], dim=1)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.bamba(x)
        return x


class IQUSepBambaUNet1D(nn.Module):
    """SepBambaUNet1D - A 4-stage 1D U-Net using Bidirectional Mamba blocks.
    
    Ref: SepMamba: State-space models for speaker separation using Mamba (ICASSP 2025)
    """
    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        features_per_stage: List[int] = [32, 64, 128, 256],
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fusion: str = "proj",
        num_layers: int = 1,
        residual_scale_init: float = 0.1,
        use_bamba: Union[bool, List[bool], None] = None,
        norm_type: str = "instance",
        use_complex_mask: bool = False,
        **kwargs
    ):
        super().__init__()
        self.use_complex_mask = use_complex_mask
        
        if len(features_per_stage) != 4:
            raise ValueError(f"features_per_stage must have length 4, got {len(features_per_stage)}")
            
        c0, c1, c2, c3 = features_per_stage
        
        # Handle use_bamba config per stage
        if use_bamba is None:
            use_bamba_list = [False, True, True, True]
        elif isinstance(use_bamba, bool):
            use_bamba_list = [use_bamba] * 4
        else:
            if len(use_bamba) != 4:
                raise ValueError(f"use_bamba list must have length 4, got {len(use_bamba)}")
            use_bamba_list = use_bamba
            
        # Encoder (Down stages)
        self.down0 = DownStage(
            input_channels, c0, stride=1, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[0], norm_type=norm_type
        )
        self.down1 = DownStage(
            c0, c1, stride=2, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[1], norm_type=norm_type
        )
        self.down2 = DownStage(
            c1, c2, stride=2, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[2], norm_type=norm_type
        )
        self.down3 = DownStage(
            c2, c3, stride=2, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[3], norm_type=norm_type  # Bottleneck
        )
        
        # Decoder (Up stages)
        self.up2 = UpStage(
            c3, c2, c2, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[2], norm_type=norm_type
        )
        self.up1 = UpStage(
            c2, c1, c1, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[1], norm_type=norm_type
        )
        self.up0 = UpStage(
            c1, c0, c0, d_state=d_state, d_conv=d_conv, expand=expand,
            fusion=fusion, num_layers=num_layers, residual_scale_init=residual_scale_init,
            use_bamba=use_bamba_list[0], norm_type=norm_type
        )
        
        # Output project
        self.out_conv = nn.Conv1d(c0, num_classes, kernel_size=3, padding=1, bias=True)
        
    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
            
        # Encoder
        d0 = self.down0(x)       # [B, c0, L]
        d1 = self.down1(d0)      # [B, c1, L/2]
        d2 = self.down2(d1)      # [B, c2, L/4]
        d3 = self.down3(d2)      # [B, c3, L/8] (Bottleneck)
        
        # Decoder
        u2 = self.up2(d3, d2)    # [B, c2, L/4]
        u1 = self.up1(u2, d1)    # [B, c1, L/2]
        u0 = self.up0(u1, d0)    # [B, c0, L]
        
        # Output projection
        out = self.out_conv(u0)  # [B, num_classes, L]
        
        if self.use_complex_mask:
            from models.IQUResUNet1D_WLComplex import apply_complex_mask, bound_complex_mask
            return apply_complex_mask(x, bound_complex_mask(out, scale=2.0))
            
        return out
