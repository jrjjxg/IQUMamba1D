"""Symbol-aligned dual-path Mamba adapter for stage-4 IQUMamba.

This variant keeps IQUMamba's original ``MambaLayer`` as the main encoder path
and adds a small symbol-aligned dual-path adapter:

    y = original_mamba(x)
    out = y + alpha * adapter(y)

The adapter follows the dual-path separation idea: split the time axis into
overlapped symbol-length chunks, scan within each chunk, scan across chunks,
and overlap-add the refined chunks back to the original feature length.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type, Union

import numpy as np
import torch
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from torch import nn
from torch.amp import autocast
from torch.nn.modules.conv import _ConvNd

from models.IQUMamba1D import BasicResBlock, MambaLayer, UNetResDecoder


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


def _stage_downsample_factors(strides: List[List[int]]) -> List[int]:
    factors: List[int] = []
    factor = 1
    for stride in strides:
        factor *= int(stride[0])
        factors.append(max(1, factor))
    return factors


def _odd_positive(value: int, minimum: int = 1) -> int:
    value = max(int(value), int(minimum))
    return value


class SymbolAlignedDualPathMambaAdapter(nn.Module):
    """Dual-path adapter operating on symbol-aligned temporal chunks."""

    def __init__(
        self,
        channels: int,
        chunk_size: int,
        hop_size: int,
        min_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.chunk_size = _odd_positive(chunk_size, min_chunk_size)
        self.hop_size = _odd_positive(hop_size, 1)
        self.intra_mamba = MambaLayer(dim=self.channels, channel_token=False)
        self.inter_mamba = MambaLayer(dim=self.channels, channel_token=False)
        self.out_proj = nn.Conv1d(self.channels, self.channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.bias)

    def _effective_sizes(self, length: int) -> tuple[int, int]:
        chunk = min(self.chunk_size, max(1, int(length)))
        hop = min(self.hop_size, chunk)
        return chunk, max(1, hop)

    def _pad_for_chunks(self, x: torch.Tensor, chunk: int, hop: int) -> tuple[torch.Tensor, int, int]:
        length = x.size(-1)
        if length <= chunk:
            total_length = chunk
            num_chunks = 1
        else:
            num_chunks = math.ceil((length - chunk) / hop) + 1
            total_length = (num_chunks - 1) * hop + chunk
        pad_right = total_length - length
        if pad_right > 0:
            x = torch.nn.functional.pad(x, (0, pad_right))
        return x, num_chunks, pad_right

    def _chunk(self, x: torch.Tensor, chunk: int, hop: int) -> tuple[torch.Tensor, int]:
        x, num_chunks, pad_right = self._pad_for_chunks(x, chunk, hop)
        chunks = x.unfold(dimension=-1, size=chunk, step=hop)
        if chunks.size(-2) != num_chunks:
            chunks = chunks[:, :, :num_chunks, :]
        return chunks.contiguous(), pad_right

    def overlap_add(
        self,
        chunks: torch.Tensor,
        original_length: int,
        chunk: int,
        hop: int,
    ) -> torch.Tensor:
        batch, channels, num_chunks, _ = chunks.shape
        total_length = (num_chunks - 1) * hop + chunk
        out = chunks.new_zeros(batch, channels, total_length)
        weight = chunks.new_zeros(1, 1, total_length)
        for idx in range(num_chunks):
            start = idx * hop
            end = start + chunk
            out[:, :, start:end] = out[:, :, start:end] + chunks[:, :, idx, :]
            weight[:, :, start:end] = weight[:, :, start:end] + 1.0
        out = out / weight.clamp_min(1.0)
        return out[:, :, :original_length]

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        batch, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        chunk, hop = self._effective_sizes(length)
        chunks, _ = self._chunk(x, chunk=chunk, hop=hop)
        num_chunks = chunks.size(2)

        intra = chunks.permute(0, 2, 1, 3).reshape(batch * num_chunks, channels, chunk)
        intra = self.intra_mamba(intra)
        intra = intra.reshape(batch, num_chunks, channels, chunk).permute(0, 2, 1, 3).contiguous()

        inter = intra.permute(0, 3, 1, 2).reshape(batch * chunk, channels, num_chunks)
        inter = self.inter_mamba(inter)
        inter = inter.reshape(batch, chunk, channels, num_chunks).permute(0, 2, 3, 1).contiguous()

        out = self.overlap_add(inter, original_length=length, chunk=chunk, hop=hop)
        return self.out_proj(out)


class SymbolDualPathMambaLayer(nn.Module):
    """Original MambaLayer plus symbol-aligned dual-path residual adapter."""

    def __init__(
        self,
        dim: int,
        channel_token: bool = False,
        chunk_size: int = 80,
        hop_size: int = 40,
        residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.main_mamba = MambaLayer(dim=self.dim, channel_token=self.channel_token)
        self.adapter = None
        if not self.channel_token:
            self.adapter = SymbolAlignedDualPathMambaAdapter(
                channels=self.dim,
                chunk_size=chunk_size,
                hop_size=hop_size,
            )
        self.adapter_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        out = self.main_mamba(x)
        if self.adapter is None:
            return out
        return out + self.adapter_scale * self.adapter(out)


class ResidualSymbolDualPathMambaEncoder(nn.Module):
    """IQUMamba encoder with original Mamba main path and dual-path adapters."""

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
        norm_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict | None = None,
        return_skips: bool = False,
        stem_channels: int | None = None,
        pool_type: str = "conv",
        symbol_samples: int = 20,
        dual_path_chunk_symbols: int = 4,
        dual_path_hop_symbols: int = 2,
        dual_path_residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        del pool_type
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

        downsample_factors = _stage_downsample_factors(strides)
        base_chunk = int(symbol_samples) * int(dual_path_chunk_symbols)
        base_hop = int(symbol_samples) * int(dual_path_hop_symbols)

        input_channels_stage = stem_channels
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels_stage,
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
                factor = downsample_factors[s]
                chunk_size = max(4, math.ceil(base_chunk / factor))
                hop_size = max(1, math.ceil(base_hop / factor))
                mamba_layers.append(
                    SymbolDualPathMambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s],
                        chunk_size=chunk_size,
                        hop_size=hop_size,
                        residual_scale_init=dual_path_residual_scale_init,
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels_stage = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
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

    def forward(self, x: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUMamba1D_SymbolDualPath(nn.Module):
    """Stage-4 IQUMamba with symbol-aligned dual-path Mamba adapters."""

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
        symbol_samples: int = 20,
        dual_path_chunk_symbols: int = 4,
        dual_path_hop_symbols: int = 2,
        dual_path_residual_scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        self.encoder = ResidualSymbolDualPathMambaEncoder(
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
            symbol_samples=symbol_samples,
            dual_path_chunk_symbols=dual_path_chunk_symbols,
            dual_path_hop_symbols=dual_path_hop_symbols,
            dual_path_residual_scale_init=dual_path_residual_scale_init,
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
