from typing import List, Type

import torch
import torch.nn as nn

from models.IQUResUNet1D_InnovationBase import BaseSkipInnovationResUNet1D, make_norm, match_length


class FrequencyDescriptor1D(nn.Module):
    """FcaNet/GFNet-style selected FFT-bin descriptor for 1D IQ features."""

    def __init__(self, channels: int, out_channels: int, frequency_indices: List[int] = None):
        super().__init__()
        if frequency_indices is None:
            frequency_indices = [1, 2, 4, 8, 16, 32]
        self.frequency_indices = [int(idx) for idx in frequency_indices]
        self.proj = nn.Conv1d(channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        magnitude = spectrum.abs()
        max_bin = magnitude.size(-1) - 1
        indices = [min(max(idx, 0), max_bin) for idx in self.frequency_indices]
        index_tensor = torch.tensor(indices, device=x.device, dtype=torch.long)
        descriptor = magnitude.index_select(dim=-1, index=index_tensor).mean(dim=-1, keepdim=True)
        return self.proj(descriptor.to(dtype=x.dtype))


class FrequencyLSSGSkipGate1D(nn.Module):
    """Decoder-guided skip gate conditioned by compact Fourier descriptors."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        frequency_indices: List[int] = None,
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
        self.skip_freq = FrequencyDescriptor1D(skip_channels, inter_channels, frequency_indices)
        self.dec_freq = FrequencyDescriptor1D(dec_channels, inter_channels, frequency_indices)
        self.gate = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, skip: torch.Tensor, dec: torch.Tensor, **_) -> torch.Tensor:
        dec = match_length(dec, skip.size(-1))
        freq_context = self.skip_freq(skip).expand(-1, -1, skip.size(-1))
        freq_context = freq_context + self.dec_freq(dec).expand(-1, -1, skip.size(-1))
        gate = torch.sigmoid(self.gate(self.skip_proj(skip) + self.dec_proj(dec) + freq_context))
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class IQUResUNet1D_FreqLSSG(BaseSkipInnovationResUNet1D):
    def __init__(self, *args, residual_scale_init: float = 0.1, frequency_indices=None, gated_decoder_stages=None, **kwargs):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs, **_):
            return FrequencyLSSGSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                frequency_indices=frequency_indices,
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
