from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D_InnovationBase import BaseSkipInnovationResUNet1D, make_norm, match_length


class FocalContextLSSGSkipGate1D(nn.Module):
    """FocalNet-inspired gated aggregation of local-to-wide depthwise context."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        focal_levels: List[int] = None,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if focal_levels is None:
            focal_levels = [3, 7, 15]
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)
        self.focal_levels = [int(level) for level in focal_levels]

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.context_blocks = nn.ModuleList(
            [
                nn.Conv1d(
                    inter_channels,
                    inter_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=inter_channels,
                    bias=False,
                )
                for kernel in self.focal_levels
            ]
        )
        self.context_gate = nn.Conv1d(inter_channels, len(self.focal_levels), kernel_size=1)
        self.out_proj = nn.Conv1d(inter_channels, skip_channels, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, **_) -> torch.Tensor:
        dec = match_length(dec, skip.size(-1))
        base = F.silu(self.skip_proj(skip) + self.dec_proj(dec))
        contexts = torch.stack([block(base) for block in self.context_blocks], dim=1)
        weights = F.softmax(self.context_gate(base).unsqueeze(2), dim=1)
        focal_context = (contexts * weights).sum(dim=1)
        gate = torch.sigmoid(self.out_proj(focal_context))
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class IQUResUNet1D_FocalLSSG(BaseSkipInnovationResUNet1D):
    def __init__(self, *args, residual_scale_init: float = 0.1, focal_levels=None, gated_decoder_stages=None, **kwargs):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs, **_):
            return FocalContextLSSGSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                focal_levels=focal_levels,
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
