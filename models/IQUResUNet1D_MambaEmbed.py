"""Stage-42 ResUNet variants with Mamba embedded into the convolutional backbone.

The variants in this file keep the pure ResUNet input/output contract and only
change how selected encoder stages mix long-range features:

* BottleneckMambaAdapter: minimal global Mamba residual at the deepest stage.
* LocalGlobalMambaBlock: local depthwise convolution plus global Mamba with a gate.
* TemporalChannelDualMambaGate: temporal-token and channel-token Mamba branches.

No source labels, bit streams, or sampling-prior metadata are consumed at
inference time.  These are architecture-only ablations on top of stage 42.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type, Union

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from models.IQUMamba1D import MambaLayer, UNetResDecoder
from models.IQUResUNet1D import ResidualConvEncoder


def _as_stage_set(stages: Union[None, int, List[int], Tuple[int, ...]], n_stages: int) -> set[int]:
    if stages is None:
        return {n_stages - 1}
    if isinstance(stages, int):
        stages = [stages]
    return {int(stage) for stage in stages if 0 <= int(stage) < n_stages}


def _stage_lengths(input_size: int, strides: List[int]) -> List[int]:
    length = int(input_size)
    lengths = []
    for stride in strides:
        length = math.ceil(length / int(stride))
        lengths.append(max(1, length))
    return lengths


def _zero_last_conv(module: nn.Module) -> None:
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Conv1d):
            nn.init.zeros_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
            return


class BottleneckMambaAdapter(nn.Module):
    """Lightweight temporal Mamba residual used mainly at the deepest stage."""

    def __init__(
        self,
        channels: int,
        stage_length: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scale_init: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        del stage_length
        self.mamba = MambaLayer(
            dim=int(channels),
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            channel_token=False,
        )
        self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1)
        _zero_last_conv(self.proj)
        self.adapter_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.adapter_scale * self.proj(self.mamba(x))


class LocalGlobalMambaBlock(nn.Module):
    """MambaIR-style local enhancement plus global selective-state mixing."""

    def __init__(
        self,
        channels: int,
        stage_length: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scale_init: float = 0.0,
        local_kernel_size: int = 7,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        del stage_length
        channels = int(channels)
        local_kernel_size = int(local_kernel_size)
        if local_kernel_size % 2 == 0:
            local_kernel_size += 1
        gate_hidden = max(4, int(gate_hidden))

        self.local = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=local_kernel_size, padding=local_kernel_size // 2, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.global_mamba = MambaLayer(
            dim=channels,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            channel_token=False,
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, gate_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(gate_hidden, channels, kernel_size=1),
        )
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        _zero_last_conv(self.proj)
        self.adapter_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        global_features = self.global_mamba(x)
        gate = torch.sigmoid(self.gate(x))
        mixed = gate * global_features + (1.0 - gate) * local
        return x + self.adapter_scale * self.proj(mixed)


class TemporalChannelDualMambaGate(nn.Module):
    """S2Mamba-inspired temporal/channel dual branch with learned fusion gate."""

    def __init__(
        self,
        channels: int,
        stage_length: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scale_init: float = 0.0,
        local_kernel_size: int = 5,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        channels = int(channels)
        self.stage_length = int(stage_length)
        gate_hidden = max(4, int(gate_hidden))
        local_kernel_size = int(local_kernel_size)
        if local_kernel_size % 2 == 0:
            local_kernel_size += 1

        self.temporal_mamba = MambaLayer(
            dim=channels,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            channel_token=False,
        )
        self.channel_mamba = MambaLayer(
            dim=self.stage_length,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
            channel_token=True,
        )
        self.local_anchor = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=local_kernel_size, padding=local_kernel_size // 2, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, gate_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(gate_hidden, channels, kernel_size=1),
        )
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        _zero_last_conv(self.proj)
        self.adapter_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal_mamba(x)
        if x.size(-1) == self.stage_length:
            channel = self.channel_mamba(x)
        else:
            channel = torch.zeros_like(temporal)
        gate = torch.sigmoid(self.gate(x))
        mixed = gate * temporal + (1.0 - gate) * channel + self.local_anchor(x)
        return x + self.adapter_scale * self.proj(mixed)


class IdentityMambaEmbed(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ResidualMambaEmbedEncoder(ResidualConvEncoder):
    """ResidualConvEncoder with selected post-stage Mamba embedding blocks."""

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
        adapter_cls: Type[nn.Module],
        mamba_embed_stages: Union[None, int, List[int], Tuple[int, ...]] = None,
        mamba_embed_d_state: int = 16,
        mamba_embed_d_conv: int = 4,
        mamba_embed_expand: int = 2,
        mamba_embed_scale_init: float = 0.0,
        mamba_embed_local_kernel_size: int = 7,
        mamba_embed_gate_hidden: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            **kwargs,
        )
        stage_set = _as_stage_set(mamba_embed_stages, n_stages)
        features = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else list(features_per_stage)
        stride_values = [s[0] if isinstance(s, (list, tuple)) else int(s) for s in strides]
        stage_lengths = _stage_lengths(int(input_size[0]), stride_values)

        adapters = []
        for stage_idx, channels in enumerate(features):
            if stage_idx in stage_set:
                adapters.append(
                    adapter_cls(
                        channels=int(channels),
                        stage_length=int(stage_lengths[stage_idx]),
                        d_state=int(mamba_embed_d_state),
                        d_conv=int(mamba_embed_d_conv),
                        expand=int(mamba_embed_expand),
                        scale_init=float(mamba_embed_scale_init),
                        local_kernel_size=int(mamba_embed_local_kernel_size),
                        gate_hidden=int(mamba_embed_gate_hidden),
                    )
                )
            else:
                adapters.append(IdentityMambaEmbed())
        self.mamba_embed_layers = nn.ModuleList(adapters)

    def forward(self, x: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for stage, adapter in zip(self.stages, self.mamba_embed_layers):
            x = stage(x)
            x = adapter(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class _IQUResUNet1D_MambaEmbedBase(nn.Module):
    adapter_cls: Type[nn.Module] = BottleneckMambaAdapter

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
        mamba_embed_stages: Union[None, int, List[int], Tuple[int, ...]] = None,
        mamba_embed_d_state: int = 16,
        mamba_embed_d_conv: int = 4,
        mamba_embed_expand: int = 2,
        mamba_embed_scale_init: float = 0.0,
        mamba_embed_local_kernel_size: int = 7,
        mamba_embed_gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ResidualMambaEmbedEncoder(
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
            adapter_cls=self.adapter_cls,
            mamba_embed_stages=mamba_embed_stages,
            mamba_embed_d_state=mamba_embed_d_state,
            mamba_embed_d_conv=mamba_embed_d_conv,
            mamba_embed_expand=mamba_embed_expand,
            mamba_embed_scale_init=mamba_embed_scale_init,
            mamba_embed_local_kernel_size=mamba_embed_local_kernel_size,
            mamba_embed_gate_hidden=mamba_embed_gate_hidden,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)


class IQUResUNet1D_MambaBottleneck(_IQUResUNet1D_MambaEmbedBase):
    adapter_cls = BottleneckMambaAdapter


class IQUResUNet1D_MambaLocalGlobal(_IQUResUNet1D_MambaEmbedBase):
    adapter_cls = LocalGlobalMambaBlock


class IQUResUNet1D_MambaDualGate(_IQUResUNet1D_MambaEmbedBase):
    adapter_cls = TemporalChannelDualMambaGate
