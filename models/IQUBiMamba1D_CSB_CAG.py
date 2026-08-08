"""IQUBiMamba1D_CSB_CAG - CSB with scaled BiMamba residuals and complex-aware gates.

This variant keeps the CSB backbone unchanged and only changes temporal-token
BiMamba stages:
  - the bidirectional Mamba branch predicts a residual delta;
  - a learnable scalar alpha controls residual strength;
  - a complex-aware channel gate is computed from paired feature channels.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUBiMamba1D import BiMambaLayer
from models.IQUBiMamba1D_CSB import ComplexBottleneckBridge1D, ComplexStemBlock1D
from models.IQUMamba1D import BasicResBlock, UNetResDecoder


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class _BiScanCore(nn.Module):
    """Bidirectional Mamba transform without the residual add."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(self.dim)
        self.mamba_fwd = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(self.dim * 2, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        h_fwd = self.mamba_fwd(x_norm)
        h_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        return self.out_proj(torch.cat([h_fwd, h_bwd], dim=-1))


class ComplexAwareChannelGate(nn.Module):
    """Channel gate built from pseudo-complex feature pairs."""

    def __init__(self, dim: int, gate_hidden: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.pair_dim = (self.dim + 1) // 2
        hidden = max(4, int(gate_hidden))
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * self.pair_dim),
            nn.Linear(3 * self.pair_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.pair_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) % 2 == 1:
            x_pair = F.pad(x, (0, 1))
        else:
            x_pair = x
        real = x_pair[..., 0::2]
        imag = x_pair[..., 1::2]
        mag = torch.sqrt(real.square() + imag.square() + self.eps)
        cos_phase = real / mag
        sin_phase = imag / mag
        stats = torch.cat(
            [mag.mean(dim=1), cos_phase.mean(dim=1), sin_phase.mean(dim=1)],
            dim=-1,
        )
        gate_pair = self.gate(stats)
        gate = gate_pair.repeat_interleave(2, dim=-1)[..., : self.dim]
        return gate.unsqueeze(1)


class ComplexAwareGatedBiMambaLayer(nn.Module):
    """BiMamba residual delta modulated by alpha and complex-aware channel gates."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        alpha_init: float = 0.1,
        gate_hidden: int = 64,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.core = _BiScanCore(self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.channel_gate = ComplexAwareChannelGate(self.dim, gate_hidden=gate_hidden)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.core(x)
        gate = self.channel_gate(x)
        return x + self.alpha * gate * delta

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = dims.numel()
        x_flat = x.reshape(batch, d_model, n_tokens).transpose(-1, -2)
        out = self._fuse(x_flat)
        return out.transpose(-1, -2).reshape(batch, d_model, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("ComplexAwareGatedBiMambaLayer should not be used with channel_token=True")

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class ResidualBiMambaEncoder_CSB_CAG(nn.Module):
    """CSB encoder with scaled, complex-aware gated BiMamba temporal stages."""

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
        norm_op_kwargs: dict | None = {"eps": 1e-5, "affine": True},
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict | None = {"inplace": True},
        return_skips: bool = False,
        stem_channels: int | None = None,
        pool_type: str = "conv",
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
        cag_alpha_init: float = 0.1,
        cag_gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"CSB CAG model expects raw IQ input_channels=2, got {input_channels}")
        if pool_type != "conv":
            raise NotImplementedError("Only convolutional downsampling is supported.")

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
        self.stem = ComplexStemBlock1D(
            output_channels=stem_channels,
            hidden_complex_channels=int(complex_stem_hidden_channels),
            kernel_size=int(complex_stem_kernel_size),
        )

        input_channels = stem_channels
        stages = []
        mamba_layers = []
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
                if do_channel_token[s]:
                    mamba_layers.append(BiMambaLayer(dim=dim, channel_token=True))
                else:
                    mamba_layers.append(
                        ComplexAwareGatedBiMambaLayer(
                            dim=dim,
                            alpha_init=float(cag_alpha_init),
                            gate_hidden=int(cag_gate_hidden),
                        )
                    )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.complex_bottleneck = ComplexBottleneckBridge1D(
            channels=int(features_per_stage[-1]),
            hidden_channels=int(complex_bottleneck_hidden_channels),
            num_blocks=int(complex_bottleneck_num_blocks),
            kernel_size=int(complex_bottleneck_kernel_size),
            dilation_growth=int(complex_bottleneck_dilation_growth),
            zero_init=bool(complex_bottleneck_zero_init),
        )

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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            if s == len(self.stages) - 1:
                x = self.complex_bottleneck(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUBiMamba1D_CSB_CAG(nn.Module):
    """BiMamba CSB U-Net with scaled residual alpha and complex-aware channel gates."""

    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = nn.InstanceNorm1d,
        norm_op_kwargs: dict | None = {"eps": 1e-5, "affine": True},
        dropout_op: Union[None, Type[nn.Module]] = None,
        dropout_op_kwargs: dict | None = None,
        nonlin: Union[None, Type[torch.nn.Module]] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = {"inplace": True},
        deep_supervision: bool = False,
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
        cag_alpha_init: float = 0.1,
        cag_gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        del dropout_op, dropout_op_kwargs
        self.encoder = ResidualBiMambaEncoder_CSB_CAG(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            complex_stem_hidden_channels=complex_stem_hidden_channels,
            complex_stem_kernel_size=complex_stem_kernel_size,
            complex_bottleneck_hidden_channels=complex_bottleneck_hidden_channels,
            complex_bottleneck_num_blocks=complex_bottleneck_num_blocks,
            complex_bottleneck_kernel_size=complex_bottleneck_kernel_size,
            complex_bottleneck_dilation_growth=complex_bottleneck_dilation_growth,
            complex_bottleneck_zero_init=complex_bottleneck_zero_init,
            cag_alpha_init=cag_alpha_init,
            cag_gate_hidden=cag_gate_hidden,
        )
        self.decoder = UNetResDecoder(
            self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)
