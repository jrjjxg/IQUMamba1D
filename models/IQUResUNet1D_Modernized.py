from typing import List, Tuple, Type, Union
import torch
import torch.nn as nn

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from models.IQUResUNet1D import BasicResBlock
from models.IQU_BottleneckEnhanced import DualDomainCycloContextBottleneck1D
from models.blocks_modern import ConvNeXt1DBlock, MSCAN1DBlock, HybridCNNBiMambaBlock

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        if stride > 1 or in_channels != out_channels:
            self.down = nn.Sequential(
                nn.InstanceNorm1d(in_channels, affine=True),
                nn.Conv1d(in_channels, out_channels, kernel_size=stride if stride > 1 else 1, stride=stride)
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        return self.down(x)

class ModernizedResidualConvEncoder(nn.Module):
    """
    Encoder that allows replacing the standard ResBlocks with modern blocks
    (ConvNeXt, MSCAN, HybridMamba).
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        strides: List[int],
        n_blocks_per_stage: List[int],
        block_mode: str = "convnext", # 'convnext', 'mscan', 'hybrid'
    ):
        super().__init__()
        
        self.return_skips = True
        
        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            DownsampleLayer(input_channels, stem_channels, 1),
            *[self._build_block(block_mode, stem_channels, stage_idx=0) for _ in range(n_blocks_per_stage[0])]
        )

        input_ch = stem_channels
        stages = []
        for s in range(n_stages):
            stage_blocks = [DownsampleLayer(input_ch, features_per_stage[s], strides[s])]
            for _ in range(n_blocks_per_stage[s]):
                stage_blocks.append(self._build_block(block_mode, features_per_stage[s], stage_idx=s))
            
            stages.append(nn.Sequential(*stage_blocks))
            input_ch = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage

    def _build_block(self, mode: str, channels: int, stage_idx: int):
        if mode == "convnext":
            return ConvNeXt1DBlock(channels)
        elif mode == "mscan":
            # Only use MSCAN in L/4, L/8 (stage_idx >= 2). Otherwise use ConvNeXt.
            if stage_idx >= 2:
                return MSCAN1DBlock(channels)
            return ConvNeXt1DBlock(channels)
        elif mode == "hybrid":
            # Only use Hybrid Mamba in deep stages (stage_idx >= 3)
            if stage_idx >= 3:
                return HybridCNNBiMambaBlock(channels)
            return ConvNeXt1DBlock(channels)
        else:
            raise ValueError(f"Unknown block mode: {mode}")

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret


class ModernizedUNetResDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        strides: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        deep_supervision: bool = False,
        block_mode: str = "convnext",
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        
        n_stages = len(encoder_channels) - 1
        
        self.stages = nn.ModuleList()
        self.transpconvs = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        
        for s in range(n_stages):
            in_ch = encoder_channels[n_stages - s]
            skip_ch = encoder_channels[n_stages - s - 1]
            out_ch = encoder_channels[n_stages - s - 1]
            stride = strides[n_stages - s]
            
            # Upsample
            if stride > 1:
                tconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=stride, stride=stride)
            else:
                tconv = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            self.transpconvs.append(tconv)
            
            # Blocks after skip fusion
            stage_blocks = []
            # We concat skip and upsampled features, so in_ch = skip_ch + out_ch
            stage_blocks.append(DownsampleLayer(skip_ch + out_ch, out_ch, 1))
            for _ in range(n_conv_per_stage_decoder[s]):
                if block_mode == "convnext":
                    stage_blocks.append(ConvNeXt1DBlock(out_ch))
                else:
                    # Decoder always uses ConvNeXt for stability, even in MSCAN/Hybrid modes
                    stage_blocks.append(ConvNeXt1DBlock(out_ch))
            self.stages.append(nn.Sequential(*stage_blocks))
            
            self.seg_layers.append(nn.Conv1d(out_ch, num_classes, kernel_size=1))

    def forward(self, skips):
        x = skips[-1]
        seg_outputs = []
        
        n_stages = len(self.stages)
        for s in range(n_stages):
            x = self.transpconvs[s](x)
            skip = skips[n_stages - s - 1]
            
            x = torch.cat([x, skip], dim=1)
            x = self.stages[s](x)
            
            seg_outputs.append(self.seg_layers[s](x))
            
        if self.deep_supervision:
            return seg_outputs[::-1]
        return seg_outputs[-1]


class IQUResUNet1D_BottleneckEnhanced_Modernized(nn.Module):
    """
    Modernized U-Net architecture built upon the DCCB Bottleneck (Stage 119).
    Enhances the encoder and decoder with modern ConvNeXt, MSCAN, or Hybrid blocks.
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        deep_supervision: bool = False,
        block_mode: str = "convnext", # 'convnext', 'mscan', 'hybrid'
    ):
        super().__init__()
        
        self.encoder = ModernizedResidualConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            strides=strides,
            n_blocks_per_stage=n_conv_per_stage,
            block_mode=block_mode
        )
        
        bottleneck_channels = features_per_stage[-1]
        self.bottleneck = DualDomainCycloContextBottleneck1D(bottleneck_channels)
        
        self.decoder = ModernizedUNetResDecoder(
            encoder_channels=features_per_stage,
            strides=strides,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            block_mode=block_mode
        )

    def forward(self, x):
        # x: [B, 2, L]
        skips = self.encoder(x)
        
        z = skips[-1]
        z = self.bottleneck(x, z)
        skips[-1] = z
        
        out = self.decoder(skips)
        return out
