"""IQUResUNet1D_WLComplex - ResUNet baseline with Widely-Linear stem and Complex Mask.

This model builds on top of stage 56 (IQUResUNet1D_NoASC) by integrating:
  1. A Widely-Linear (WL) input stem (Conv Wz + Conv Vz*) in the first layer.
  2. A real U-Net backbone with plain skip connections.
  3. A complex mask projection head for final IQ waveform reconstruction.
"""

from typing import List, Type, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUMamba1D import BasicResBlock
from models.IQUResUNet1D import ResidualConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD


def apply_complex_mask(x_mix: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Applies estimated complex masks to the input mixture to reconstruct signals.

    S = M * Y = (M_r + j * M_i) * (Y_r + j * Y_i)
              = (M_r * Y_r - M_i * Y_i) + j * (M_r * Y_i + M_i * Y_r)

    x_mix: [B, 2, L] (channel 0: I, channel 1: Q)
    masks: [B, 4, L] (0: M_r1, 1: M_i1, 2: M_r2, 3: M_i2)
    Returns: [B, 4, L] (S1_I, S1_Q, S2_I, S2_Q)
    """
    y_r = x_mix[:, 0:1, :]  # [B, 1, L]
    y_i = x_mix[:, 1:2, :]  # [B, 1, L]
    
    m_r1 = masks[:, 0:1, :]  # [B, 1, L]
    m_i1 = masks[:, 1:2, :]  # [B, 1, L]
    m_r2 = masks[:, 2:3, :]  # [B, 1, L]
    m_i2 = masks[:, 3:4, :]  # [B, 1, L]
    
    # Source 1 reconstruction
    s1_r = m_r1 * y_r - m_i1 * y_i
    s1_i = m_r1 * y_i + m_i1 * y_r
    
    # Source 2 reconstruction
    s2_r = m_r2 * y_r - m_i2 * y_i
    s2_i = m_r2 * y_i + m_i2 * y_r
    
    return torch.cat([s1_r, s1_i, s2_r, s2_i], dim=1)


def bound_complex_mask(masks: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    """Bounds the magnitude of estimated complex masks."""
    return torch.tanh(masks) * scale
class TrueWLComplexStem1d(nn.Module):
    """True Widely-Linear 1D Complex Convolution mapping real/imag inputs via Wz + Vz*."""

    def __init__(self, c_out: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.wr = nn.Conv1d(1, c_out, kernel_size, stride=stride, padding=padding)
        self.wi = nn.Conv1d(1, c_out, kernel_size, stride=stride, padding=padding)
        self.vr = nn.Conv1d(1, c_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.vi = nn.Conv1d(1, c_out, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 2, L]
        xr = x[:, 0:1]
        xi = x[:, 1:2]

        # Wz
        wz_r = self.wr(xr) - self.wi(xi)
        wz_i = self.wr(xi) + self.wi(xr)

        # Vz*
        vz_r = self.vr(xr) + self.vi(xi)
        vz_i = self.vi(xr) - self.vr(xi)

        yr = wz_r + vz_r
        yi = wz_i + vz_i

        # Stack real and imag channels interleaving or flat.
        # [B, 2, c_out, L] -> [B, 2 * c_out, L]
        return torch.stack([yr, yi], dim=2).flatten(1, 2)


class WidelyLinearBasicResBlock(nn.Module):
    """Residual block that applies a Widely-Linear convolution for its first operation."""

    def __init__(
        self,
        conv_op,
        input_channels: int,  # must be 2
        output_channels: int,
        norm_op,
        norm_op_kwargs,
        kernel_size=3,
        padding=1,
        stride=1,
        use_1x1conv=False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
    ):
        super().__init__()
        # Ensure output_channels is divisible by 2 for the true complex output
        c_out_half = max(1, output_channels // 2)
        
        self.stem = TrueWLComplexStem1d(c_out=c_out_half, kernel_size=kernel_size, stride=stride, padding=padding)
        self.proj = nn.Conv1d(c_out_half * 2, output_channels, kernel_size=1)
        
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding, bias=True)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = nn.Sequential(
                TrueWLComplexStem1d(c_out=c_out_half, kernel_size=1, stride=stride, padding=0),
                nn.Conv1d(c_out_half * 2, output_channels, kernel_size=1)
            )
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.stem(x)
        y = self.proj(y)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        
        if self.conv3 is not None:
            x = self.conv3(x)
            
        y = y + x
        return self.act2(y)


class WidelyLinearResidualConvEncoder(ResidualConvEncoder):
    """Residual convolutional encoder incorporating WidelyLinearBasicResBlock at the stem."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Conv1d],
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
    ):
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=return_skips,
            stem_channels=stem_channels,
            pool_type=pool_type,
        )
        
        # Override the first block of the stem with the Widely-Linear version
        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        
        stem_blocks = [
            WidelyLinearBasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=self.kernel_sizes[0][0] if isinstance(self.kernel_sizes[0], list) else self.kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0],
                stride=1,
                use_1x1conv=True,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            )
        ]
        
        for _ in range(n_blocks_per_stage[0] - 1):
            stem_blocks.append(
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=self.kernel_sizes[0][0] if isinstance(self.kernel_sizes[0], list) else self.kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
            
        self.stem = nn.Sequential(*stem_blocks)


class IQUResUNet1D_WLComplex(nn.Module):
    """Pure convolutional ResUNet incorporating a Widely-Linear stem and Complex Masking."""

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
        **kwargs,
    ):
        super().__init__()
        
        # 1. Widely-linear encoder
        self.encoder = WidelyLinearResidualConvEncoder(
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
        
        # 2. Plain U-Net decoder (No-ASC plain skip connection style)
        from models.IQUResUNet1D_InnovationBase import PlainUNetResDecoder
        self.decoder = PlainUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,  # outputs [B, 4, L] representing masks
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # x is the input mixture: [B, 2, L]
        skips = self.encoder(x)
        masks = self.decoder(skips)  # [B, 4, L] or list of [B, 4, L]
        
        if self.decoder.deep_supervision and isinstance(masks, list):
            outs = []
            for m in masks:
                if m.shape[-1] != x.shape[-1]:
                    m = F.interpolate(m, size=x.shape[-1], mode="linear", align_corners=False)
                outs.append(apply_complex_mask(x, bound_complex_mask(m, scale=2.0)))
            return outs
            
        return apply_complex_mask(x, bound_complex_mask(masks, scale=2.0))
