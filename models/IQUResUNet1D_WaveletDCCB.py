from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D_InnovationBase import BaseBottleneckInnovationResUNet1D, match_length
from models.IQU_BottleneckEnhanced import RealASPPBottleneck1D


class WaveletContextBottleneck1D(nn.Module):
    """WTConv-inspired bottleneck using low/high feature decomposition."""

    def __init__(self, channels: int, scale_init: float = 0.05, wavelet_scale: float = 0.05):
        super().__init__()
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        self.wavelet_scale = nn.Parameter(torch.tensor(float(wavelet_scale)))
        self.wavelet_encoder = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm1d(channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def low_high_decompose(self, z: torch.Tensor) -> List[torch.Tensor]:
        if z.size(-1) % 2 == 1:
            z = F.pad(z, (0, 1))
        even = z[..., 0::2]
        odd = z[..., 1::2]
        low = 0.5 * (even + odd)
        high = 0.5 * (even - odd)
        return [low, high]

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_time = self.time_branch(x, z)
        low, high = self.low_high_decompose(z)
        low = match_length(low, z.size(-1))
        high = match_length(high, z.size(-1))
        wavelet_delta = self.wavelet_encoder(torch.cat([low, high], dim=1))
        gate = self.gate(torch.cat([z_time, wavelet_delta], dim=1))
        return z_time + self.wavelet_scale * gate * wavelet_delta


class IQUResUNet1D_WaveletDCCB(BaseBottleneckInnovationResUNet1D):
    def __init__(self, *args, features_per_stage, wavelet_scale: float = 0.05, bottleneck_scale_init: float = 0.05, **kwargs):
        bottleneck = WaveletContextBottleneck1D(
            channels=features_per_stage[-1],
            scale_init=bottleneck_scale_init,
            wavelet_scale=wavelet_scale,
        )
        super().__init__(*args, features_per_stage=features_per_stage, bottleneck=bottleneck, **kwargs)
