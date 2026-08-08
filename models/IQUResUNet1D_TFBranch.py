"""IQUResUNet1D_TFBranch - Time-Frequency Dual-Branch ResUNet baseline.

This model combines:
  1. A Time-domain U-Net branch estimating time-domain complex masks.
  2. A Frequency-domain U-Net branch estimating complex TF-domain masks.
  3. A learnable time-varying gating network to adaptively fuse both outputs.
"""

from typing import List, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC
from models.IQUResUNet1D_InnovationBase import PlainUNetResDecoder
from models.IQUResUNet1D_WLComplex import apply_complex_mask, bound_complex_mask

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class IQUResUNet1D_TFBranch(nn.Module):
    """Time-Frequency Dual-Branch ResUNet separator."""

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
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        freq_features_per_stage: List[int] = [128, 256, 384, 512],
        **kwargs,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        
        # Window for STFT/iSTFT
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

        # 1. Time-Domain U-Net branch (No-ASC style)
        self.time_unet = IQUResUNet1D_NoASC(
            input_size=input_size,
            input_channels=input_channels,  # e.g., 2
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,  # e.g., 4
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )

        # 2. Frequency-Domain U-Net branch (No-ASC style)
        # Input channels: 2 * n_fft = 512 channels (concatenated real/imag spectrogram bins)
        # Output channels: 2 sources * 2 (real/imag) * n_fft = 1024 mask components
        freq_in_channels = 2 * self.n_fft
        freq_out_channels = 2 * 2 * self.n_fft
        freq_stages = len(freq_features_per_stage)

        self.freq_unet = IQUResUNet1D_NoASC(
            input_size=128,  # dummy sequence length (frames)
            input_channels=freq_in_channels,
            n_stages=freq_stages,
            features_per_stage=freq_features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[3] * freq_stages,
            strides=[1] + [2] * (freq_stages - 1),
            n_conv_per_stage=[2] * freq_stages,
            num_classes=freq_out_channels,
            n_conv_per_stage_decoder=[2] * (freq_stages - 1),
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=False,  # Keep frequency branch simple without deep supervision
        )

        # 3. Gating Fusion Network
        # Input: time-domain waveform of time branch [B, 4, L] + frequency branch [B, 4, L] -> [B, 8, L]
        # Output: gate [B, 4, L]
        self.gate_net = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3, padding=1, bias=True),
            norm_op(16, **norm_op_kwargs) if norm_op is not None else nn.Identity(),
            nonlin(**nonlin_kwargs),
            nn.Conv1d(16, 2, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # x shape: [B, 2, L]
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        B, _, L = x.shape
        window = self.window.to(x.device)

        # ==========================================
        # 1. Time-Domain Branch
        # ==========================================
        # time_unet outputs the complex masks [B, 4, L]
        masks_time = self.time_unet(x)
        
        # If deep supervision is active, masks_time is a list of masks
        is_ds = isinstance(masks_time, list)
        
        if is_ds:
            final_masks_time = masks_time[0]
        else:
            final_masks_time = masks_time
            
        s_time = apply_complex_mask(x, bound_complex_mask(final_masks_time, scale=2.0))  # [B, 4, L]

        # ==========================================
        # 2. Frequency-Domain Branch
        # ==========================================
        # Convert mixture to complex signal
        y_complex = torch.complex(x[:, 0], x[:, 1])  # [B, L]
        
        # STFT -> [B, F, T]
        y_spec = torch.stft(
            y_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            onesided=False,
            return_complex=True,
        ).transpose(1, 2).contiguous()  # [B, T, F]

        T = y_spec.size(1)

        # Concatenate real/imag channels into [B, 2F, T]
        y_spec_real = torch.cat([y_spec.real, y_spec.imag], dim=2).permute(0, 2, 1).contiguous()

        # Freq U-Net forward pass -> [B, 4F, T]
        f_out = self.freq_unet(y_spec_real)  # [B, 4F, T]

        # Reshape to masks: [B, T, 2, 2, F] (sources, real/imag, frequency bins)
        f_out = f_out.permute(0, 2, 1).contiguous()
        f_out = f_out.reshape(B, T, 2, 2, self.n_fft)

        # Bound frequency masks
        f_out = torch.tanh(f_out) * 4.0

        mask_real = f_out[:, :, :, 0, :].permute(0, 2, 1, 3)  # [B, 2, T, F]
        mask_imag = f_out[:, :, :, 1, :].permute(0, 2, 1, 3)  # [B, 2, T, F]
        M_freq = torch.complex(mask_real, mask_imag)          # [B, 2, T, F]

        # Apply mask in frequency domain
        separated_specs = M_freq * y_spec.unsqueeze(1)         # [B, 2, T, F]

        # Reconstruct with iSTFT
        s1 = torch.istft(
            separated_specs[:, 0].transpose(1, 2).contiguous(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            onesided=False,
            length=L,
            return_complex=True,
        )
        s2 = torch.istft(
            separated_specs[:, 1].transpose(1, 2).contiguous(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            onesided=False,
            length=L,
            return_complex=True,
        )

        s_freq = torch.cat([s1.real.unsqueeze(1), s1.imag.unsqueeze(1), s2.real.unsqueeze(1), s2.imag.unsqueeze(1)], dim=1)  # [B, 4, L]

        # ==========================================
        # 3. Gating Waveform Fusion
        # ==========================================
        gate = self.gate_net(torch.cat([s_time, s_freq], dim=1))  # [B, 2, L]
        g1 = gate[:, 0:1, :]
        g2 = gate[:, 1:2, :]
        
        s1 = g1.repeat(1, 2, 1) * s_time[:, 0:2, :] + (1.0 - g1).repeat(1, 2, 1) * s_freq[:, 0:2, :]
        s2 = g2.repeat(1, 2, 1) * s_time[:, 2:4, :] + (1.0 - g2).repeat(1, 2, 1) * s_freq[:, 2:4, :]
        s_hat = torch.cat([s1, s2], dim=1)

        if is_ds:
            outs = [s_hat]
            for m in masks_time[1:]:
                if m.shape[-1] != x.shape[-1]:
                    m = F.interpolate(m, size=x.shape[-1], mode="linear", align_corners=False)
                outs.append(apply_complex_mask(x, bound_complex_mask(m, scale=2.0)))
            return outs

        return s_hat
