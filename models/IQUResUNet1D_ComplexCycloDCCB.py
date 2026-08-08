from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D_InnovationBase import BaseBottleneckInnovationResUNet1D
from models.IQU_BottleneckEnhanced import RealASPPBottleneck1D


class ComplexCycloContextBottleneck1D(nn.Module):
    """Complex autocorrelation and pseudo_corr DCCB for IQ baseband signals."""

    def __init__(
        self,
        channels: int,
        lags: List[int] = None,
        scale_init: float = 0.05,
        cyclo_scale_init: float = 0.05,
    ):
        super().__init__()
        if lags is None:
            lags = [0, 1, 2, 4, 8, 16]
        self.lags = [int(lag) for lag in lags]
        self.time_branch = RealASPPBottleneck1D(channels, scale_init=scale_init)
        self.cyclo_scale = nn.Parameter(torch.tensor(float(cyclo_scale_init)))
        in_channels = 4 * len(self.lags)
        hidden = max(channels // 4, 16)
        self.cyclo_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm1d(hidden, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, channels // 2, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(channels // 2, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels // 2, channels, kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(channels, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=5, stride=2, padding=2),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def _shift_pair(self, xr: torch.Tensor, xi: torch.Tensor, lag: int):
        if lag <= 0:
            return xr, xi, xr, xi
        if xr.size(-1) <= lag:
            return xr, xi, xr, xi
        xr_now = F.pad(xr[..., lag:], (0, lag))
        xi_now = F.pad(xi[..., lag:], (0, lag))
        xr_lag = F.pad(xr[..., :-lag], (0, lag))
        xi_lag = F.pad(xi[..., :-lag], (0, lag))
        return xr_now, xi_now, xr_lag, xi_lag

    def complex_autocorr_bank(self, x: torch.Tensor) -> torch.Tensor:
        xr = x[:, 0:1, :]
        xi = x[:, 1:2, :] if x.size(1) > 1 else torch.zeros_like(xr)
        features = []
        for lag in self.lags:
            ar, ai, br, bi = self._shift_pair(xr, xi, lag)
            corr_re = ar * br + ai * bi
            corr_im = ai * br - ar * bi
            pseudo_corr_re = ar * br - ai * bi
            pseudo_corr_im = ar * bi + ai * br
            pseudo_corr = torch.cat([pseudo_corr_re, pseudo_corr_im], dim=1)
            features.extend([corr_re, corr_im, pseudo_corr[:, 0:1, :], pseudo_corr[:, 1:2, :]])
        return torch.cat(features, dim=1)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_time = self.time_branch(x, z)
        cyclo = self.complex_autocorr_bank(x)
        z_cyclo = self.cyclo_encoder(cyclo)
        if z_cyclo.size(-1) != z.size(-1):
            z_cyclo = F.interpolate(z_cyclo, size=z.size(-1), mode="linear", align_corners=False)
        gate = self.gate(torch.cat([z_time, z_cyclo], dim=1))
        return z_time + self.cyclo_scale * gate * z_cyclo


class IQUResUNet1D_ComplexCycloDCCB(BaseBottleneckInnovationResUNet1D):
    def __init__(
        self,
        *args,
        features_per_stage,
        cyclo_lags=None,
        bottleneck_scale_init: float = 0.05,
        cyclo_scale_init: float = 0.05,
        **kwargs,
    ):
        bottleneck = ComplexCycloContextBottleneck1D(
            channels=features_per_stage[-1],
            lags=cyclo_lags,
            scale_init=bottleneck_scale_init,
            cyclo_scale_init=cyclo_scale_init,
        )
        super().__init__(*args, features_per_stage=features_per_stage, bottleneck=bottleneck, **kwargs)
