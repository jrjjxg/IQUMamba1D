from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D_InnovationBase import BaseSkipInnovationResUNet1D, make_norm, match_length


class SelectiveKernelLSSGSkipGate1D(nn.Module):
    """SKNet-inspired dynamic multi-scale routing for LSSG skip calibration."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        branch_kernels: List[int] = None,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if branch_kernels is None:
            branch_kernels = [1, 5, 9, 17]
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.branch_convs = nn.ModuleList(
            [
                nn.Conv1d(inter_channels, skip_channels, kernel_size=k, padding=k // 2, bias=False)
                for k in branch_kernels
            ]
        )
        hidden = max(skip_channels // 4, 16)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(skip_channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(hidden, skip_channels * len(branch_kernels), kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, **_) -> torch.Tensor:
        dec = match_length(dec, skip.size(-1))
        fused = F.silu(self.skip_proj(skip) + self.dec_proj(dec))
        branches = [branch(fused) for branch in self.branch_convs]
        stacked = torch.stack(branches, dim=1)
        routing_seed = sum(branches)
        branch_logits = self.router(routing_seed).view(skip.size(0), len(branches), skip.size(1), 1)
        weights = F.softmax(branch_logits, dim=1)
        gate = torch.sigmoid((stacked * weights).sum(dim=1))
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class IQUResUNet1D_SKLSSG(BaseSkipInnovationResUNet1D):
    def __init__(self, *args, residual_scale_init: float = 0.1, branch_kernels=None, gated_decoder_stages=None, **kwargs):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs, **_):
            return SelectiveKernelLSSGSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                branch_kernels=branch_kernels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                residual_scale_init=residual_scale_init,
            )

        super().__init__(
            *args,
            processor_factory=processor_factory,
            gated_decoder_stages=gated_decoder_stages,
            **kwargs,
        )
