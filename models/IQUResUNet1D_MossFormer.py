import math
from typing import List, Type, Union
import torch
import torch.nn as nn

from models.IQUResUNet1D import ResidualConvEncoder
from models.IQUResUNet1D_InnovationBase import PlainUNetResDecoder

class MossFormerLiteBlock(nn.Module):
    """
    MossFormer-lite Bottleneck Block.
    Simultaneously models local patterns via Depthwise Conv and global dependencies via Single-Head Attention,
    fusing them with a gated interaction mechanism.
    """
    def __init__(self, d_model: int, kernel_size: int = 31):
        super().__init__()
        # Local Branch (Depthwise Conv)
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size//2, groups=d_model),
            nn.InstanceNorm1d(d_model, affine=True),
            nn.SiLU()
        )
        
        # Global Branch (Single-Head Self-Attention)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Gating Interaction
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        residual = x
        B, C, L = x.shape
        
        # 1. Local patterns
        x_local = self.local_conv(x) # [B, C, L]
        
        # 2. Global dependencies
        x_trans = x.transpose(1, 2) # [B, L, C]
        q = self.q_proj(x_trans)
        k = self.k_proj(x_trans)
        v = self.v_proj(x_trans)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        x_global = torch.matmul(attn_weights, v)
        x_global = self.attn_norm(x_global) # [B, L, C]
        
        # 3. Gated Interaction: Gate is derived from global, applied to local
        gate = torch.sigmoid(self.gate_proj(x_global)) # [B, L, C]
        x_fused = (x_local.transpose(1, 2) * gate) + x_global # [B, L, C]
        
        # 4. Output projection
        out = self.out_proj(x_fused).transpose(1, 2) # [B, C, L]
        
        return residual + out


class MossFormerBottleneck(nn.Module):
    def __init__(self, channels: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            MossFormerLiteBlock(channels) for _ in range(n_blocks)
        ])
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, T]
        for block in self.blocks:
            z = block(z)
        return z


class IQUResUNet1D_MossFormer(nn.Module):
    """
    IQ U-Net with a MossFormer-lite bottleneck for Local-Global context modeling.
    Replaces DCCB/Mamba with a specialized Speech BSS style bottleneck.
    """
    def __init__(self,
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
                 norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
                 nonlin: Type[nn.Module] = nn.LeakyReLU,
                 nonlin_kwargs: dict = {'inplace': True},
                 deep_supervision: bool = False,
                 n_mossformer_blocks: int = 2):
        super().__init__()
        
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
        
        bottleneck_channels = features_per_stage[-1]
        self.bottleneck = MossFormerBottleneck(bottleneck_channels, n_blocks=n_mossformer_blocks)
        
        self.decoder = PlainUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        # x: [B, 2, L]
        skips = self.encoder(x)
        
        # Apply bottleneck to deepest feature
        z = skips[-1]
        z = self.bottleneck(z)
        skips[-1] = z
        
        out = self.decoder(skips)
        return out
