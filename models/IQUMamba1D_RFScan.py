"""Stage-4 IQUMamba variants with lightweight RF/Radar physics adapters.

Stages 70/71/72 intentionally keep the original IQUMamba ``MambaLayer`` as
the main path.  The RF/Radar-inspired part is only a small conditional adapter:

    out = original_mamba(x) + alpha * adapter(original_mamba(x), raw_iq_condition)

With alpha initialized to zero, each variant starts equivalent to IQUMamba and
can learn to use the physical condition only when it helps.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type, Union

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.modules.conv import _ConvNd

from models.IQUMamba1D import BasicResBlock, MambaLayer, UNetResDecoder


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class RawIQAmplitudePhaseConditioner(nn.Module):
    """Per-stage raw I/Q amplitude/phase conditions for the RFMamba-like path."""

    def __init__(
        self,
        input_size: int,
        features_per_stage: List[int],
        strides: List[int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.stage_lengths = _stage_lengths(input_size, strides)

        in_channels = 6  # I, Q, magnitude, cos(phase), sin(phase), phase difference
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, int(ch), kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(int(ch), int(ch), kernel_size=1),
                )
                for ch in features_per_stage
            ]
        )

    def _features(self, raw_iq: torch.Tensor) -> torch.Tensor:
        real = raw_iq[:, 0:1, :]
        imag = raw_iq[:, 1:2, :]
        mag = torch.sqrt(real.square() + imag.square() + self.eps)
        phase = torch.atan2(imag, real)
        phase_diff = torch.diff(phase, dim=-1, prepend=phase[..., :1])
        return torch.cat([real, imag, mag, torch.cos(phase), torch.sin(phase), phase_diff], dim=1)

    def forward(self, raw_iq: torch.Tensor) -> List[torch.Tensor]:
        features = self._features(raw_iq.float())
        conditions = []
        for target_len, proj in zip(self.stage_lengths, self.projections):
            x = F.interpolate(features, size=target_len, mode="linear", align_corners=False)
            conditions.append(proj(x))
        return conditions


class RawIQDopplerConditioner(nn.Module):
    """Per-stage raw complex-I/Q STFT conditions for the RadMamba-like path."""

    def __init__(
        self,
        input_size: int,
        features_per_stage: List[int],
        strides: List[int],
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int | None = None,
        freq_bins: int = 32,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = self.n_fft if win_length is None else int(win_length)
        self.freq_bins = max(1, int(freq_bins))
        self.register_buffer("window", torch.hann_window(self.win_length, periodic=True), persistent=False)
        self.stage_lengths = _stage_lengths(input_size, strides)

        in_channels = self.freq_bins * 3
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, int(ch), kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(int(ch), int(ch), kernel_size=1),
                )
                for ch in features_per_stage
            ]
        )

    def _stft_features(self, raw_iq: torch.Tensor) -> torch.Tensor:
        complex_iq = torch.complex(raw_iq[:, 0, :].float(), raw_iq[:, 1, :].float())
        spec = torch.stft(
            complex_iq,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=raw_iq.device, dtype=torch.float32),
            center=True,
            return_complex=True,
        )
        mag = torch.log1p(torch.abs(spec))
        phase = torch.angle(spec)
        time_delta = torch.diff(mag, dim=-1, prepend=mag[..., :1])
        features = torch.stack([mag, torch.cos(phase), time_delta], dim=1)
        batch, channels, freq, frames = features.shape
        features = features.reshape(batch * channels, 1, freq, frames)
        features = F.adaptive_avg_pool2d(features, (self.freq_bins, frames))
        return features.reshape(batch, channels * self.freq_bins, frames)

    def forward(self, raw_iq: torch.Tensor) -> List[torch.Tensor]:
        features = self._stft_features(raw_iq)
        conditions = []
        for target_len, proj in zip(self.stage_lengths, self.projections):
            x = F.interpolate(features, size=target_len, mode="linear", align_corners=False)
            conditions.append(proj(x))
        return conditions


def _stage_lengths(input_size: int, strides: List[int]) -> List[int]:
    length = int(input_size)
    lengths = []
    for stride in strides:
        length = math.ceil(length / int(stride))
        lengths.append(max(1, length))
    return lengths


class PhysicsFiLMAdapter(nn.Module):
    """Small conditional Conv/FiLM adapter applied after the original Mamba."""

    def __init__(self, channels: int, conv_kernel_size: int = 5) -> None:
        super().__init__()
        conv_kernel_size = int(conv_kernel_size)
        if conv_kernel_size % 2 == 0:
            conv_kernel_size += 1
        padding = conv_kernel_size // 2
        self.condition_proj = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.local = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=conv_kernel_size, padding=padding, groups=channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if condition is None:
            return torch.zeros_like(x)
        condition = condition.to(device=x.device, dtype=x.dtype)
        if condition.size(-1) != x.size(-1):
            condition = F.interpolate(condition, size=x.size(-1), mode="linear", align_corners=False)
        gamma, beta = self.condition_proj(condition).chunk(2, dim=1)
        gamma = torch.tanh(gamma)
        return self.local(x * (1.0 + gamma) + beta)


class OriginalMambaAdapterLayer(nn.Module):
    """Original IQUMamba MambaLayer plus a zero-initialized physics adapter."""

    def __init__(
        self,
        dim: int,
        channel_token: bool = False,
        conv_kernel_size: int = 5,
        residual_scale_init: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.main_mamba = MambaLayer(dim=self.dim, channel_token=self.channel_token)
        self.adapter = None if self.channel_token else PhysicsFiLMAdapter(self.dim, conv_kernel_size=conv_kernel_size)
        self.adapter_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.condition: torch.Tensor | None = None

    def set_conditioning(self, condition: torch.Tensor | None) -> None:
        self.condition = condition

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        out = self.main_mamba(x)
        if self.adapter is None:
            return out
        return out + self.adapter_scale * self.adapter(out, self.condition)


class ResidualRFScanMambaEncoder(nn.Module):
    """IQUMamba encoder skeleton using original MambaLayer + physics adapter."""

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
        scan_layer_cls: Type[OriginalMambaAdapterLayer],
        scan_kwargs: dict | None = None,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        return_skips: bool = False,
        stem_channels: int | None = None,
        pool_type: str = "conv",
        conditioner_type: str = "fusion",
        rfscan_stft_n_fft: int = 256,
        rfscan_stft_hop_length: int = 64,
        rfscan_stft_win_length: int | None = None,
        rfscan_stft_freq_bins: int = 32,
    ) -> None:
        super().__init__()
        del pool_type
        scan_kwargs = {} if scan_kwargs is None else dict(scan_kwargs)
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i / j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

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
        scan_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
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

            if bool(s % 2) ^ bool(n_stages % 2):
                dim = int(np.prod(feature_map_sizes[s])) if do_channel_token[s] else int(features_per_stage[s])
                scan_layers.append(scan_layer_cls(dim=dim, channel_token=do_channel_token[s], **scan_kwargs))
            else:
                scan_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(scan_layers)
        self.stages = nn.ModuleList(stages)
        conditioner_type = str(conditioner_type)
        stride_values = [s[0] for s in strides]
        if conditioner_type == "ap":
            self.conditioners = nn.ModuleList(
                [RawIQAmplitudePhaseConditioner(input_size[0], features_per_stage, stride_values)]
            )
        elif conditioner_type == "doppler":
            self.conditioners = nn.ModuleList(
                [
                    RawIQDopplerConditioner(
                        input_size[0],
                        features_per_stage,
                        stride_values,
                        n_fft=rfscan_stft_n_fft,
                        hop_length=rfscan_stft_hop_length,
                        win_length=rfscan_stft_win_length,
                        freq_bins=rfscan_stft_freq_bins,
                    )
                ]
            )
        elif conditioner_type == "fusion":
            self.conditioners = nn.ModuleList(
                [
                    RawIQAmplitudePhaseConditioner(input_size[0], features_per_stage, stride_values),
                    RawIQDopplerConditioner(
                        input_size[0],
                        features_per_stage,
                        stride_values,
                        n_fft=rfscan_stft_n_fft,
                        hop_length=rfscan_stft_hop_length,
                        win_length=rfscan_stft_win_length,
                        freq_bins=rfscan_stft_freq_bins,
                    ),
                ]
            )
        else:
            self.conditioners = nn.ModuleList()

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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        raw_iq = x
        conditioning_maps = None
        if len(self.conditioners) > 0:
            all_maps = [conditioner(raw_iq) for conditioner in self.conditioners]
            conditioning_maps = [
                sum(stage_maps) / float(len(stage_maps))
                for stage_maps in zip(*all_maps)
            ]

        x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            layer = self.mamba_layers[s]
            if hasattr(layer, "set_conditioning"):
                condition = None if conditioning_maps is None else conditioning_maps[s]
                layer.set_conditioning(condition)
            x = layer(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class _IQUMambaRFScanBase(nn.Module):
    scan_layer_cls: Type[OriginalMambaAdapterLayer] = OriginalMambaAdapterLayer
    conditioner_type: str = "fusion"

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
        rfscan_chunk_size: int = 256,
        rfscan_shift_size: int | None = None,
        rfscan_freq_bands: int = 16,
        rfscan_gate_hidden: int = 64,
        rfscan_conv_kernel_size: int = 5,
        rfscan_residual_scale_init: float = 0.0,
        rfscan_condition_scale_init: float = 0.0,
        rfscan_stft_n_fft: int = 256,
        rfscan_stft_hop_length: int = 64,
        rfscan_stft_win_length: int | None = None,
        rfscan_stft_freq_bins: int = 32,
    ) -> None:
        super().__init__()
        del rfscan_chunk_size, rfscan_shift_size, rfscan_freq_bands, rfscan_gate_hidden, rfscan_condition_scale_init
        scan_kwargs = {
            "conv_kernel_size": int(rfscan_conv_kernel_size),
            "residual_scale_init": float(rfscan_residual_scale_init),
        }
        self.encoder = ResidualRFScanMambaEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            scan_layer_cls=self.scan_layer_cls,
            scan_kwargs=scan_kwargs,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            conditioner_type=self.conditioner_type,
            rfscan_stft_n_fft=rfscan_stft_n_fft,
            rfscan_stft_hop_length=rfscan_stft_hop_length,
            rfscan_stft_win_length=rfscan_stft_win_length,
            rfscan_stft_freq_bins=rfscan_stft_freq_bins,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)


class IQUMamba1D_RFScanFusion(_IQUMambaRFScanBase):
    """Stage-4 IQUMamba + AP/STFT fused lightweight adapter."""

    conditioner_type = "fusion"


class IQUMamba1D_RFMambaScan(_IQUMambaRFScanBase):
    """Stage-4 IQUMamba + raw I/Q amplitude-phase lightweight adapter."""

    conditioner_type = "ap"


class IQUMamba1D_RadMambaScan(_IQUMambaRFScanBase):
    """Stage-4 IQUMamba + raw I/Q STFT time-frequency lightweight adapter."""

    conditioner_type = "doppler"
