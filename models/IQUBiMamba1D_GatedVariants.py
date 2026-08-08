"""Lightweight gated BiMamba variants for IQ source separation.

These models keep the plain IQUBiMamba1D U-Net topology and replace only the
BiMamba scan layer inside the encoder:
  - LayerScale: learn a small residual strength for each BiMamba layer.
  - LocalGlobal: fuse a cheap local depthwise-conv branch with BiMamba.
  - GLG: combine LocalGlobal fusion with LayerScale residual control.

The goal is to improve trainability and high-SNR detail retention without the
runtime/parameter cost of heavier post-decoder refinement heads.
"""

from __future__ import annotations

from typing import List, Tuple, Type, Union

import numpy as np
import torch
from mamba_ssm import Mamba
from torch import nn
from torch.amp import autocast
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from models.IQUMamba1D import BasicResBlock, UNetResDecoder


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class _BiMambaCore(nn.Module):
    """Bidirectional Mamba transform without residual addition."""

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


class LayerScaleBiMambaLayer(nn.Module):
    """BiMamba with learnable residual strength initialized below 1."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        residual_scale_init: float = 0.1,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.core = _BiMambaCore(self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.core(x)

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = dims.numel()
        x_flat = x.reshape(batch, d_model, n_tokens).transpose(-1, -2)
        out = self._fuse(x_flat)
        return out.transpose(-1, -2).reshape(batch, d_model, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        out = self._fuse(x_flat)
        return out.reshape(batch, n_tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class LocalGlobalBiMambaLayer(nn.Module):
    """Fuse local convolutional detail with global BiMamba context."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        local_kernel_size: int = 7,
        gate_hidden: int = 64,
        residual_scale_init: float = 1.0,
        use_layerscale: bool = False,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        if local_kernel_size % 2 == 0:
            raise ValueError(f"local_kernel_size should be odd, got {local_kernel_size}")
        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.use_layerscale = bool(use_layerscale)
        self.global_core = _BiMambaCore(self.dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.local_core = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=int(local_kernel_size),
                padding=int(local_kernel_size) // 2,
                groups=self.dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=True),
        )
        hidden = max(4, int(gate_hidden))
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def _local(self, x: torch.Tensor) -> torch.Tensor:
        local_in = self.local_core[0](x).transpose(1, 2)
        y = self.local_core[1](local_in)
        y = self.local_core[2](y)
        y = self.local_core[3](y)
        return y.transpose(1, 2)

    def _fuse(self, x: torch.Tensor) -> torch.Tensor:
        y_local = self._local(x)
        y_global = self.global_core(x)
        weights = torch.softmax(self.fusion_gate(x.mean(dim=1)), dim=-1).view(x.size(0), 2, 1, 1)
        branches = torch.stack([y_local, y_global], dim=1)
        delta = (branches * weights).sum(dim=1)
        if self.use_layerscale:
            return x + self.alpha * delta
        return x + delta

    def forward_patch_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = dims.numel()
        x_flat = x.reshape(batch, d_model, n_tokens).transpose(-1, -2)
        out = self._fuse(x_flat)
        return out.transpose(-1, -2).reshape(batch, d_model, *dims)

    def forward_channel_token(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)
        out = self._fuse(x_flat)
        return out.reshape(batch, n_tokens, *dims)

    @autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


class ResidualBiMambaEncoder_GatedVariants(nn.Module):
    """Residual BiMamba encoder with selectable lightweight scan variants."""

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
        variant: str = "layerscale",
        mamba_residual_scale_init: float = 0.1,
        local_kernel_size: int = 7,
        local_global_gate_hidden: int = 64,
    ) -> None:
        super().__init__()
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
        mamba_layers = []
        variant = str(variant).lower()
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
                if variant == "layerscale":
                    mamba_layers.append(
                        LayerScaleBiMambaLayer(
                            dim=dim,
                            residual_scale_init=mamba_residual_scale_init,
                            channel_token=do_channel_token[s],
                        )
                    )
                elif variant in {"localglobal", "glg"}:
                    mamba_layers.append(
                        LocalGlobalBiMambaLayer(
                            dim=dim,
                            local_kernel_size=local_kernel_size,
                            gate_hidden=local_global_gate_hidden,
                            residual_scale_init=mamba_residual_scale_init,
                            use_layerscale=variant == "glg",
                            channel_token=do_channel_token[s],
                        )
                    )
                else:
                    raise ValueError(f"Unknown gated BiMamba variant: {variant}")
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

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
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class _IQUBiMamba1D_GatedBase(nn.Module):
    """Shared top-level U-Net wrapper for gated BiMamba variants."""

    variant = "layerscale"

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
        mamba_residual_scale_init: float = 0.1,
        local_kernel_size: int = 7,
        local_global_gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = ResidualBiMambaEncoder_GatedVariants(
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
            variant=self.variant,
            mamba_residual_scale_init=mamba_residual_scale_init,
            local_kernel_size=local_kernel_size,
            local_global_gate_hidden=local_global_gate_hidden,
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


class IQUBiMamba1D_LayerScale(_IQUBiMamba1D_GatedBase):
    """Variant 2: BiMamba with learnable residual scale."""

    variant = "layerscale"


class IQUBiMamba1D_LocalGlobal(_IQUBiMamba1D_GatedBase):
    """Variant 3: gated local convolution + global BiMamba fusion."""

    variant = "localglobal"


class IQUBiMamba1D_GLG(_IQUBiMamba1D_GatedBase):
    """Variant 2+3: local-global fusion with learnable residual scale."""

    variant = "glg"
