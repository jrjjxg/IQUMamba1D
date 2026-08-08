"""IQUConvNeXt1D — ConvNeXt-style Large-Kernel variant of IQUMamba1D.

Replaces the Mamba (SSM) sequence-modeling layers with ConvNeXt-1D blocks
that use large-kernel depthwise convolutions to capture long-range temporal
context.  This is a pure-CNN architecture with NO recurrent or attention
components, making it:

  - Extremely fast on GPU (no sequential scan overhead)
  - Low memory footprint (no hidden-state accumulation)
  - Easy to train (no Mamba/CUDA kernel dependency)

The ConvNeXt block follows the "inverted bottleneck" design:
  DepthwiseLargeKernelConv → LayerNorm → PointwiseExpand → GELU → PointwiseProject

All other components (ResidualEncoder, UNetResDecoder, SkipConnection-
Processor, etc.) are inherited from IQUMamba1D.py without modification.
"""

import numpy as np
import torch
from torch import nn

# Re-use all building blocks from the original file
from models.IQUMamba1D import (
    ResidualMambaEncoder,
    UNetResDecoder,
    BasicResBlock,
    SkipConnectionProcessor,
)
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd


# ============================================================================
#  LargeKernelConvLayer — drop-in replacement for MambaLayer
# ============================================================================

class LargeKernelConvLayer(nn.Module):
    """ConvNeXt-1D style large-kernel convolution block.

    Replaces the Mamba SSM layer with a depthwise large-kernel convolution
    followed by an inverted-bottleneck FFN.  Interface is identical to the
    original ``MambaLayer`` so the encoder can use it transparently.

    Architecture (per call):
        x → DW-Conv1d(kernel=lk_kernel_size, groups=dim) → LayerNorm
          → Linear(dim→dim*expand) → GELU → Linear(dim*expand→dim)
          → residual add with input

    Parameters
    ----------
    dim : int
        Feature dimension (channel count for patch-token mode, or spatial
        size for channel-token mode).
    lk_kernel_size : int
        Kernel size for the depthwise convolution. Larger = wider receptive
        field. Default 31.
    expand : int
        Expansion factor for the pointwise FFN. Default 4.
    channel_token : bool
        Token layout flag inherited from the Mamba interface.
    """

    def __init__(self, dim, lk_kernel_size=31, expand=4, channel_token=False):
        super().__init__()
        dim = int(dim)
        self.dim = dim
        self.channel_token = channel_token

        # --- Depthwise large-kernel conv ---
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=lk_kernel_size,
            padding=lk_kernel_size // 2,
            groups=dim,  # depthwise
        )

        # --- LayerNorm (applied on last dim, so we work in [B, L, D]) ---
        self.norm = nn.LayerNorm(dim)

        # --- Inverted-bottleneck pointwise FFN ---
        ffn_hidden = dim * expand
        self.pwconv1 = nn.Linear(dim, ffn_hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(ffn_hidden, dim)

        # Learnable per-channel scale (Layer Scale, ConvNeXt trick)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def _forward_block(self, x):
        """Core ConvNeXt block.  x: [B, L, D]"""
        residual = x

        # Depthwise conv operates in [B, D, L] layout
        x_conv = x.transpose(1, 2)          # [B, D, L]
        x_conv = self.dwconv(x_conv)
        x_conv = x_conv.transpose(1, 2)     # [B, L, D]

        # Norm + pointwise FFN
        x_conv = self.norm(x_conv)
        x_conv = self.pwconv2(self.act(self.pwconv1(x_conv)))

        # Layer Scale + residual
        x_conv = self.gamma * x_conv
        return residual + x_conv

    def forward_patch_token(self, x):
        """x: [B, C, *spatial]  (C = dim, spatial collapsed to L)."""
        B, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = dims.numel()
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  # [B, L, D]
        out = self._forward_block(x_flat)
        return out.transpose(-1, -2).reshape(B, d_model, *dims)

    def forward_channel_token(self, x):
        """x: [B, L, *feat]  (feat collapsed to D)."""
        B, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)  # [B, L, D]
        out = self._forward_block(x_flat)
        return out.reshape(B, n_tokens, *dims)

    def forward(self, x):
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


# ============================================================================
#  ResidualConvNeXtEncoder — identical to ResidualMambaEncoder but uses
#  LargeKernelConvLayer instead of MambaLayer at alternating stages.
# ============================================================================

class ResidualConvNeXtEncoder(nn.Module):
    def __init__(self,
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
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 # ---------- ConvNeXt-specific ----------
                 lk_kernel_size: int = 31,
                 lk_expand: int = 4,
                 ):
        super().__init__()

        self.lk_kernel_size = lk_kernel_size
        self.lk_expand = lk_expand

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
            *[BasicBlockD(
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
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        input_channels = stem_channels
        stages = []
        convnext_layers = []
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
                *[BasicBlockD(
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
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )

            # Key difference: use LargeKernelConvLayer instead of MambaLayer
            if bool(s % 2) ^ bool(n_stages % 2):
                convnext_layers.append(
                    LargeKernelConvLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        lk_kernel_size=lk_kernel_size,
                        expand=lk_expand,
                        channel_token=do_channel_token[s],
                    )
                )
            else:
                convnext_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        # Keep attribute name `mamba_layers` for decoder compatibility
        self.mamba_layers = nn.ModuleList(convnext_layers)
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
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


# ============================================================================
#  IQUConvNeXt1D — top-level model class
# ============================================================================

class IQUConvNeXt1D(nn.Module):
    """ConvNeXt-1D U-Net for IQ signal separation.

    Identical to IQUMamba1D but uses ``LargeKernelConvLayer`` (depthwise
    large-kernel conv + FFN) instead of the Mamba SSM layer.  This is a
    pure-CNN model with no recurrent or attention components.

    Parameters
    ----------
    lk_kernel_size : int
        Kernel size for the depthwise conv in ConvNeXt blocks. Default 31.
    lk_expand : int
        FFN expansion factor in ConvNeXt blocks. Default 4.
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
                 # ---------- ConvNeXt-specific ----------
                 lk_kernel_size: int = 31,
                 lk_expand: int = 4,
                 ):
        super().__init__()
        self.encoder = ResidualConvNeXtEncoder(
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
            # ConvNeXt-specific
            lk_kernel_size=lk_kernel_size,
            lk_expand=lk_expand,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
