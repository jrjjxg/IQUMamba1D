from typing import Callable, List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        pool_op_kernel_size,
        mode="nearest",
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


class BasicResBlock(nn.Module):
    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        norm_op,
        norm_op_kwargs,
        kernel_size=3,
        padding=1,
        stride=1,
        use_1x1conv=False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ):
        super().__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)

        y += x
        return self.act2(y)


class ResidualConvEncoder(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        stem_channels: int = None,
        pool_type: str = "conv",
    ):
        super().__init__()
        del input_size, pool_type

        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]

        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ],
        )

        input_channels = stem_channels
        stages = []
        for s in range(n_stages):
            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=conv_op,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        input_channels=input_channels,
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        padding=self.conv_pad_sizes[s][0],
                        stride=strides[s][0],
                        use_1x1conv=True,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=conv_op,
                            input_channels=features_per_stage[s],
                            output_channels=features_per_stage[s],
                            kernel_size=kernel_sizes[s],
                            stride=1,
                            conv_bias=conv_bias,
                            norm_op=norm_op,
                            norm_op_kwargs=norm_op_kwargs,
                            nonlin=nonlin,
                            nonlin_kwargs=nonlin_kwargs,
                        )
                        for _ in range(n_blocks_per_stage[s] - 1)
                    ],
                )
            )
            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class PlainUNetResDecoder(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage: Union[int, List[int]], deep_supervision):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_encoder = len(encoder.output_channels)
        n_conv_per_stage = (
            [n_conv_per_stage] * (n_stages_encoder - 1)
            if isinstance(n_conv_per_stage, int)
            else n_conv_per_stage
        )

        stages = []
        upsample_layers = []
        seg_layers = []

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
                    mode="linear" if encoder.conv_op == nn.Conv1d else "nearest",
                )
            )

            stages.append(
                nn.Sequential(
                    BasicResBlock(
                        conv_op=encoder.conv_op,
                        norm_op=encoder.norm_op,
                        norm_op_kwargs=encoder.norm_op_kwargs,
                        nonlin=encoder.nonlin,
                        nonlin_kwargs=encoder.nonlin_kwargs,
                        input_channels=2 * input_features_skip,
                        output_channels=input_features_skip,
                        kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                        padding=encoder.conv_pad_sizes[-(s + 1)][0],
                        stride=1,
                        use_1x1conv=True,
                    ),
                    *[
                        BasicBlockD(
                            conv_op=encoder.conv_op,
                            input_channels=input_features_skip,
                            output_channels=input_features_skip,
                            kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                            stride=1,
                            conv_bias=encoder.conv_bias,
                            norm_op=encoder.norm_op,
                            norm_op_kwargs=encoder.norm_op_kwargs,
                            nonlin=encoder.nonlin,
                            nonlin_kwargs=encoder.nonlin_kwargs,
                        )
                        for _ in range(n_conv_per_stage[s - 1] - 1)
                    ],
                )
            )
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    @staticmethod
    def _match_length(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == target.size(-1):
            return x
        return F.interpolate(x, size=target.size(-1), mode="linear", align_corners=False)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.upsample_layers[s](lres_input)
            skip = skips[-(s + 2)]
            x = self._match_length(x, skip)
            x = torch.cat((x, skip), 1)
            x = self.stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))
            lres_input = x
        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


def match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.size(-1) == target_len:
        return x
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)


def make_norm(norm_op: Type[nn.Module], channels: int, norm_op_kwargs: dict):
    if norm_op is None:
        return nn.Identity()
    return norm_op(channels, **(norm_op_kwargs or {}))


class ResearchSkipDecoder1D(nn.Module):
    """Small decoder shell that lets each stage provide its own skip processor."""

    def __init__(
        self,
        encoder,
        num_classes: int,
        n_conv_per_stage,
        deep_supervision: bool,
        processor_factory: Callable[..., nn.Module],
        gated_decoder_stages: List[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)
        self.gated_decoder_stages = gated_decoder_stages

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        encoder_skip_channels = list(encoder.output_channels[:-1])
        self.upsample_layers = nn.ModuleList()
        self.skip_processors = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(1, n_stages_encoder):
            s_idx = s - 1
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s][0]

            self.upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode="nearest",
                )
            )

            self.skip_processors.append(
                processor_factory(
                    skip_channels=input_features_skip,
                    dec_channels=input_features_skip,
                    decoder_stage=s_idx,
                    encoder_skip_channels=encoder_skip_channels,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
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
                for _ in range(n_conv_per_stage[s_idx] - 1)
            )

            self.stages.append(nn.Sequential(*blocks))
            self.seg_layers.append(encoder.conv_op(input_features_skip, num_classes, kernel_size=1))

    def forward(self, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = skips[-1]
        encoder_skips = skips[:-1]
        seg_outputs = []
        self.aux_loss = 0.0

        for s_idx, stage in enumerate(self.stages):
            x = self.upsample_layers[s_idx](x)
            skip = skips[-(s_idx + 2)]
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)

            if self.gated_decoder_stages is None or s_idx in self.gated_decoder_stages:
                processed_skip = self.skip_processors[s_idx](
                    skip,
                    x,
                    encoder_skips=encoder_skips,
                    decoder_stage=s_idx,
                )
                if hasattr(self.skip_processors[s_idx], "last_gate"):
                    self.aux_loss = self.aux_loss + self.skip_processors[s_idx].last_gate.abs().mean()
            else:
                processed_skip = skip

            x = stage(torch.cat((x, processed_skip), dim=1))
            seg_outputs.append(self.seg_layers[s_idx](x))

        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class BaseSkipInnovationResUNet1D(nn.Module):
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
        processor_factory: Callable[..., nn.Module],
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        gated_decoder_stages: List[int] = None,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

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
        self.decoder = ResearchSkipDecoder1D(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            processor_factory=processor_factory,
            gated_decoder_stages=gated_decoder_stages,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.decoder(self.encoder(x))


class BaseBottleneckInnovationResUNet1D(nn.Module):
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
        bottleneck: nn.Module,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

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
        self.bottleneck = bottleneck
        self.decoder = PlainUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        skips[-1] = self.bottleneck(x, skips[-1])
        return self.decoder(skips)
