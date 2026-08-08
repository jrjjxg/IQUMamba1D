"""IQUBiMamba1D_LK — Large-Kernel stem variant inspired by MIT OFDM paper.

Key insight from "On Neural Architectures for Deep Learning-Based Source
Separation of Co-Channel OFDM Signals" (MIT):
  The first layer's receptive field must be large enough to capture the
  signal's structural scale (≈ sps × span for single-carrier, ≈ FFT size
  for OFDM).  A short first-layer kernel only sees locally-Gaussian samples
  and cannot expose the non-Gaussian / discrete-constellation structure
  needed for separation.

Changes compared to IQUBiMamba1D:
  1. **Stem kernel size** is controlled by a dedicated ``stem_kernel_size``
     parameter (default 33 ≈ sps=4 × span=8) rather than reusing
     ``kernel_sizes[0]``.
  2. **Stem channels** is explicitly configurable (default 128, up from 32)
     to give the first layer enough capacity to approximate an FFT-like
     transform.
  3. All later encoder stages remain unchanged (small kernel, same channel
     widths) — the MIT paper shows the first layer is the critical bottleneck.
"""

import numpy as np
import torch
from torch import nn
from torch.amp import autocast

from mamba_ssm import Mamba

# Re-use all building blocks from the original files
from models.IQUMamba1D import (
    ResidualMambaEncoder,
    UNetResDecoder,
    BasicResBlock,
    SkipConnectionProcessor,
)
from models.IQUBiMamba1D import BiMambaLayer

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd


# ============================================================================
#  ResidualBiMambaEncoder_LK — Large-Kernel stem variant
# ============================================================================

class ResidualBiMambaEncoder_LK(nn.Module):
    """ResidualBiMambaEncoder with a separate, large-kernel stem.

    The stem uses ``stem_kernel_size`` (default 33) and ``stem_channels``
    (default 128) independent of the per-stage ``kernel_sizes`` and
    ``features_per_stage``.  This follows the MIT paper's finding that the
    first layer must see a window ≥ the signal's structural scale.
    """

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
                 # ---------- MIT-inspired additions ----------
                 stem_channels: int = 128,
                 stem_kernel_size: int = 33,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
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

        # ---- Large-Kernel Stem (MIT-inspired) ----
        stem_channels = int(stem_channels)
        stem_kernel_size = int(stem_kernel_size)
        stem_ks_list = maybe_convert_scalar_to_list(conv_op, stem_kernel_size)
        stem_pad = [k // 2 for k in stem_ks_list]

        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=stem_ks_list,           # ← LARGE kernel
                padding=stem_pad[0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[BasicBlockD(
                conv_op=conv_op,
                input_channels=stem_channels,
                output_channels=stem_channels,
                kernel_size=stem_ks_list,            # ← LARGE kernel
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
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],       # normal small kernel
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
                    kernel_size=kernel_sizes[s],        # normal small kernel
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )

            # BiMambaLayer at alternating stages (same logic as original)
            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    BiMambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s]
                    )
                )
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
#  IQUBiMamba1D_LK — top-level model class
# ============================================================================

class IQUBiMamba1D_LK(nn.Module):
    """Large-Kernel BiMamba U-Net for IQ signal separation.

    Identical to IQUBiMamba1D but with MIT-inspired large-kernel stem:
    - ``stem_kernel_size``: first-layer conv kernel (default 33)
    - ``stem_channels``: first-layer width (default 128)
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
                 # ---------- MIT-inspired additions ----------
                 stem_channels: int = 128,
                 stem_kernel_size: int = 33,
                 ):
        super().__init__()
        self.encoder = ResidualBiMambaEncoder_LK(
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
            # MIT-inspired
            stem_channels=stem_channels,
            stem_kernel_size=stem_kernel_size,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
