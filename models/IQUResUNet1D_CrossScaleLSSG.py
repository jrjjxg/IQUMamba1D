from typing import List, Type

import torch
import torch.nn as nn

from models.IQUResUNet1D_InnovationBase import (
    BaseSkipInnovationResUNet1D,
    make_norm,
    match_length,
)


class FullScaleSkipContext1D(nn.Module):
    """UNet3+/UCTransNet-inspired full-scale skip context for one decoder gate."""

    def __init__(self, skip_channels: List[int], out_channels: int, norm_op: Type[nn.Module], norm_op_kwargs: dict):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(channels, out_channels, kernel_size=1, bias=False),
                    make_norm(norm_op, out_channels, norm_op_kwargs),
                )
                for channels in skip_channels
            ]
        )

    def forward(self, encoder_skips: List[torch.Tensor], target_len: int) -> torch.Tensor:
        context = None
        for skip, proj in zip(encoder_skips, self.projections):
            item = match_length(proj(skip), target_len)
            context = item if context is None else context + item
        return context / max(len(self.projections), 1)


class CrossScaleLSSGSkipGate1D(nn.Module):
    """Decoder-guided LSSG with cross_scale_context from every encoder skip."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        encoder_skip_channels: List[int],
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
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
        self.full_scale = FullScaleSkipContext1D(
            skip_channels=encoder_skip_channels,
            out_channels=inter_channels,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
        )
        self.gate = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, encoder_skips: List[torch.Tensor] = None, **_) -> torch.Tensor:
        dec = match_length(dec, skip.size(-1))
        cross_scale_context = 0.0
        if encoder_skips is not None:
            cross_scale_context = self.full_scale(encoder_skips, skip.size(-1))

        gate_logits = self.gate(self.skip_proj(skip) + self.dec_proj(dec) + cross_scale_context)
        gate = torch.sigmoid(gate_logits)
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class IQUResUNet1D_CrossScaleLSSG(BaseSkipInnovationResUNet1D):
    def __init__(self, *args, residual_scale_init: float = 0.1, gated_decoder_stages=None, **kwargs):
        def processor_factory(skip_channels, dec_channels, encoder_skip_channels, norm_op, norm_op_kwargs, **_):
            return CrossScaleLSSGSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                encoder_skip_channels=encoder_skip_channels,
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
