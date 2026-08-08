import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Type, Union

from models.IQUMamba1D import BasicBlockD, BasicResBlock, MambaLayer, SkipConnectionProcessor

class HaarDownsample1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2
        weight = torch.zeros(self.out_channels, in_channels, 2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for i in range(in_channels):
            # Low pass
            weight[i, i, 0] = inv_sqrt2
            weight[i, i, 1] = inv_sqrt2
            # High pass
            weight[in_channels + i, i, 0] = inv_sqrt2
            weight[in_channels + i, i, 1] = -inv_sqrt2
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        return F.conv1d(x, self.weight, stride=2)

class HaarUpsample1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        weight = torch.zeros(in_channels, self.out_channels, 2)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        for i in range(self.out_channels):
            weight[i, i, 0] = inv_sqrt2
            weight[i, i, 1] = inv_sqrt2
            weight[self.out_channels + i, i, 0] = inv_sqrt2
            weight[self.out_channels + i, i, 1] = -inv_sqrt2
        self.register_buffer('weight', weight)
        
    def forward(self, x):
        return F.conv_transpose1d(x, self.weight, stride=2)

class DWTResidualMambaEncoder(nn.Module):
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, n_blocks_per_stage, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, conv_bias=False):
        super().__init__()
        self.output_channels = features_per_stage
        self.strides = [[2]] * n_stages
        self.strides[0] = [1]
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = [[3]] * n_stages
        self.conv_pad_sizes = [[1]] * n_stages

        stem_channels = features_per_stage[0]
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op, input_channels=input_channels, output_channels=stem_channels,
                norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, kernel_size=3, padding=1, stride=1,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, use_1x1conv=True
            ),
            *[BasicBlockD(
                conv_op=conv_op, input_channels=stem_channels, output_channels=stem_channels,
                kernel_size=3, stride=1, conv_bias=conv_bias, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        stages = []
        mamba_layers = []
        downsamples = []
        
        in_ch = stem_channels
        for s in range(n_stages):
            if s > 0:
                downsamples.append(HaarDownsample1D(in_ch))
                in_ch = in_ch * 2 # assumed to match features_per_stage[s]
            else:
                downsamples.append(nn.Identity())
            
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    input_channels=in_ch, output_channels=features_per_stage[s],
                    kernel_size=3, padding=1, stride=1, use_1x1conv=(in_ch != features_per_stage[s]),
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                ),
                *[BasicBlockD(
                    conv_op=conv_op, input_channels=features_per_stage[s], output_channels=features_per_stage[s],
                    kernel_size=3, stride=1, conv_bias=conv_bias, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )
            stages.append(stage)
            
            mamba_layers.append(
                MambaLayer(dim=features_per_stage[s], channel_token=False)
            )
            in_ch = features_per_stage[s]

        self.downsamples = nn.ModuleList(downsamples)
        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.downsamples[s](x)
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret

class DWTUNetResDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage):
        super().__init__()
        self.encoder = encoder
        n_stages_encoder = len(encoder.output_channels)
        stages = []
        upsample_layers = []
        seg_layers = []
        skip_processors = []
        
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            
            upsample_layers.append(nn.Sequential(
                HaarUpsample1D(input_features_below),
                nn.Conv1d(input_features_below // 2, input_features_skip, 1) # match channels if needed
            ))
            
            skip_processors.append(
                SkipConnectionProcessor(
                    skip_channels=input_features_skip, upsampled_channels=input_features_skip,
                    conv_op=encoder.conv_op, norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs
                )
            )

            stages.append(nn.Sequential(
                BasicResBlock(
                    conv_op=encoder.conv_op, norm_op=encoder.norm_op, norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip, output_channels=input_features_skip,
                    kernel_size=3, padding=1, stride=1, use_1x1conv=True,
                ),
                *[BasicBlockD(
                    conv_op=encoder.conv_op, input_channels=input_features_skip, output_channels=input_features_skip,
                    kernel_size=3, stride=1, conv_bias=encoder.conv_bias, norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs, nonlin=encoder.nonlin, nonlin_kwargs=encoder.nonlin_kwargs
                ) for _ in range(n_conv_per_stage[s-1] - 1)]
            ))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.skip_processors = nn.ModuleList(skip_processors) 

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            processed_skip = self.skip_processors[s](skips[-(s+2)], x)
            x = torch.cat((x, processed_skip), 1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[-1]

class IQUMamba1D_DWT(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: List[int],
                 n_conv_per_stage: List[int],
                 num_classes: int,
                 n_conv_per_stage_decoder: List[int],
                 conv_bias: bool = True,
                 ):
        super().__init__()
        conv_op = nn.Conv1d
        norm_op = nn.InstanceNorm1d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        nonlin = nn.LeakyReLU
        nonlin_kwargs = {'inplace': True}
        
        self.encoder = DWTResidualMambaEncoder(
            input_channels=input_channels, n_stages=n_stages, features_per_stage=features_per_stage,
            conv_op=conv_op, n_blocks_per_stage=n_conv_per_stage,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, conv_bias=conv_bias
        )
        self.decoder = DWTUNetResDecoder(
            encoder=self.encoder, num_classes=num_classes, n_conv_per_stage=n_conv_per_stage_decoder
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
