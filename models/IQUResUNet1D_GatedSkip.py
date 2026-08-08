"""IQUResUNet1D_GatedSkip - ResUNet baseline with Decoder-Guided Gated Residual Skip.

This variant improves IQUResUNet1D_NoASC by adding the Decoder-Guided Gated Skip
connection on top of the pure convolutional 1D ResUNet baseline.
"""

from typing import List, Type, Union

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUMamba1D import BasicResBlock, UpsampleLayer
from models.IQUResUNet1D import ResidualConvEncoder


class DecoderGuidedGatedSkip1D(nn.Module):
    """Decoder-guided gated residual skip for 1D IQ separation.

    skip: [B, C_skip, L]
    dec:  [B, C_dec,  L]
    out:  [B, C_dec,  L]
    """
    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        gate_kernel_size: int = 3,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        padding = gate_kernel_size // 2

        # Align skip channels to decoder channels if needed
        if skip_channels != dec_channels:
            self.skip_align = nn.Conv1d(skip_channels, dec_channels, kernel_size=1)
        else:
            self.skip_align = nn.Identity()

        self.gate_net = nn.Sequential(
            nn.Conv1d(dec_channels * 2, dec_channels, kernel_size=gate_kernel_size, padding=padding),
            norm_op(dec_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv1d(dec_channels, dec_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv1d(dec_channels, dec_channels, kernel_size=3, padding=1),
            norm_op(dec_channels, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
            nn.SiLU(inplace=True),
            nn.Conv1d(dec_channels, dec_channels, kernel_size=1),
        )

        self.res_scale = nn.Parameter(torch.ones(1) * residual_scale_init)
        
        # Zero-initialize refine's last conv to strictly behave as raw skip initially
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, skip, dec):
        skip = self.skip_align(skip)

        if skip.shape[-1] != dec.shape[-1]:
            if skip.shape[-1] < dec.shape[-1]:
                skip = F.pad(skip, (0, dec.shape[-1] - skip.shape[-1]))
            else:
                skip = skip[..., :dec.shape[-1]]

        # Gate computation
        gate = self.gate_net(torch.cat([skip, dec], dim=1))

        gated_skip = gate * skip
        delta = self.refine(gated_skip)

        # Stable residual form behaving exactly as raw skip at init
        out = skip + self.res_scale * delta
        return out


class GatedSkipUNetResDecoder(nn.Module):
    """Decoder using DecoderGuidedGatedSkip1D instead of plain skip connection."""

    def __init__(
        self,
        encoder: ResidualConvEncoder,
        num_classes: int,
        n_conv_per_stage,
        deep_supervision: bool,
        residual_scale_init: float = 0.1,
        gate_kernel_size: int = 3,
    ):
        super().__init__()
        self.deep_supervision = bool(deep_supervision)
        self.encoder = encoder
        self.num_classes = int(num_classes)

        n_stages_encoder = len(encoder.output_channels)
        n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1) if isinstance(n_conv_per_stage, int) else n_conv_per_stage

        stages = []
        upsample_layers = []
        seg_layers = []
        skip_processors = []

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

            skip_processors.append(
                DecoderGuidedGatedSkip1D(
                    skip_channels=input_features_skip,
                    dec_channels=input_features_skip,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    gate_kernel_size=gate_kernel_size,
                    residual_scale_init=residual_scale_init,
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
            stages.append(nn.Sequential(*blocks))
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.skip_processors = nn.ModuleList(skip_processors)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](x)
            skip = skips[-(s + 2)]
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
            
            # Gated Skip connections
            processed_skip = self.skip_processors[s](skip, x)
            
            x = torch.cat((x, processed_skip), dim=1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQUResUNet1D_GatedSkip(nn.Module):
    """Pure convolutional ResUNet with Decoder-Guided Gated Skip Connections."""

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
        residual_scale_init: float = 0.1,
        gate_kernel_size: int = 3,
        use_complex_mask: bool = False,
        **kwargs,
    ):
        super().__init__()
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
        self.decoder = GatedSkipUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            residual_scale_init=residual_scale_init,
            gate_kernel_size=gate_kernel_size,
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
