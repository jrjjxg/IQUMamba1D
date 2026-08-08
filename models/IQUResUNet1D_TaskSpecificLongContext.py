"""Task-specific long-context ResUNet variants built on the stage56 baseline.

These stages keep the plain-skip ResUNet contract and add long-context modules
where their original inductive bias is useful for IQ signals:
  - Hyena/FNO-style mixers at the bottleneck for global channel context.
  - MEGA-style moving-average gates at mid encoder stages for symbol memory.
"""

from typing import List, Sequence, Type

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUResUNet1D_InnovationBase import (
    BaseBottleneckInnovationResUNet1D,
    PlainUNetResDecoder,
    ResidualConvEncoder,
)


def _valid_stage_indices(stages: Sequence[int], n_stages: int) -> set[int]:
    return {int(stage) for stage in stages if 0 <= int(stage) < int(n_stages)}


class HyenaBottleneck1D(nn.Module):
    """Gated implicit long convolution for the ResUNet bottleneck."""

    def __init__(
        self,
        channels: int,
        filter_hidden: int = 64,
        dropout: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)
        self.in_proj = nn.Linear(self.channels, self.channels * 3)
        self.filter_mlp = nn.Sequential(
            nn.Linear(1, int(filter_hidden)),
            nn.SiLU(),
            nn.Linear(int(filter_hidden), self.channels),
        )
        self.filter_decay = nn.Parameter(torch.ones(self.channels) * 2.0)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(float(dropout))
        self.gamma = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))

    def _implicit_filter(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.linspace(0.0, 1.0, steps=int(length), device=device, dtype=dtype).unsqueeze(-1)
        filters = self.filter_mlp(pos).transpose(0, 1)
        envelope = torch.exp(-self.filter_decay.abs().to(dtype=dtype).unsqueeze(-1) * pos.squeeze(-1).unsqueeze(0))
        return filters * envelope

    def forward(self, _raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        original_dtype = x.dtype
        b, c, t = x.shape
        compute_x = x.float() if x.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else x
        tokens = self.norm(compute_x.transpose(1, 2))
        u, v, gate = self.in_proj(tokens).chunk(3, dim=-1)
        v = v.transpose(1, 2)

        v_float = v.float() if v.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else v
        filters = self._implicit_filter(t, x.device, v_float.dtype)
        fft_size = 2 * t
        v_fft = torch.fft.rfft(v_float, n=fft_size)
        filter_fft = torch.fft.rfft(filters, n=fft_size).unsqueeze(0)
        mixed = torch.fft.irfft(v_fft * filter_fft, n=fft_size)[..., :t].transpose(1, 2).to(v.dtype)

        out = mixed * F.silu(u) * torch.sigmoid(gate)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2).to(dtype=original_dtype)
        return residual + self.gamma * out


class SpectralLowRankBottleneck1D(nn.Module):
    """Capacity-controlled FNO-style bottleneck mixer for IQ spectral context."""

    def __init__(
        self,
        channels: int,
        mode_count: int = 32,
        spectral_rank: int = 4,
        dropout: float = 0.10,
        bottleneck_scale_init: float = 0.02,
    ):
        super().__init__()
        self.channels = int(channels)
        self.mode_count = max(1, int(mode_count))
        self.spectral_rank = max(1, int(spectral_rank))
        self.norm = nn.LayerNorm(self.channels)
        self.value_proj = nn.Linear(self.channels, self.channels)
        self.gate_proj = nn.Linear(self.channels, self.channels)
        self.spectral_channel_factor_real = nn.Parameter(torch.randn(self.channels, self.spectral_rank) * 0.02)
        self.spectral_channel_factor_imag = nn.Parameter(torch.randn(self.channels, self.spectral_rank) * 0.02)
        self.spectral_mode_factor_real = nn.Parameter(torch.randn(self.mode_count, self.spectral_rank) * 0.02)
        self.spectral_mode_factor_imag = nn.Parameter(torch.randn(self.mode_count, self.spectral_rank) * 0.02)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(float(dropout))
        self.gamma = nn.Parameter(torch.ones(1) * float(bottleneck_scale_init))

    def _spectral_weights(self, mode_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        channel_factor = torch.complex(
            self.spectral_channel_factor_real.to(device=device, dtype=dtype),
            self.spectral_channel_factor_imag.to(device=device, dtype=dtype),
        )
        mode_factor = torch.complex(
            self.spectral_mode_factor_real[:mode_count].to(device=device, dtype=dtype),
            self.spectral_mode_factor_imag[:mode_count].to(device=device, dtype=dtype),
        )
        return torch.einsum("cr,mr->cm", channel_factor, mode_factor) / (self.spectral_rank ** 0.5)

    def forward(self, _raw: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        residual = x
        original_dtype = x.dtype
        b, c, t = x.shape
        compute_x = x.float() if x.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else x
        tokens = self.norm(compute_x.transpose(1, 2))
        value = self.value_proj(tokens).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(tokens))

        value_float = value.float() if value.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else value
        value_fft = torch.fft.rfft(value_float, dim=-1)
        mode_count = min(self.mode_count, value_fft.size(-1))
        mixed_fft = torch.zeros_like(value_fft)
        weights = self._spectral_weights(mode_count, value_float.dtype, value_float.device)
        mixed_fft[..., :mode_count] = value_fft[..., :mode_count] * weights[None, :, :]
        mixed = torch.fft.irfft(mixed_fft, n=t, dim=-1).transpose(1, 2).to(value.dtype)

        out = self.out_proj(mixed * gate)
        out = self.drop(out).transpose(1, 2).to(dtype=original_dtype)
        return residual + self.gamma * out


class MegaMovingAverageGatedAttention1D(nn.Module):
    """MEGA-inspired moving-average gate for mid-resolution IQ features."""

    def __init__(
        self,
        channels: int,
        ema_kernel_size: int = 63,
        expansion: float = 2.0,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.ema_kernel_size = max(3, int(ema_kernel_size) | 1)
        hidden = max(self.channels, int(self.channels * float(expansion)))
        self.norm = nn.LayerNorm(self.channels)
        self.in_proj = nn.Linear(self.channels, hidden * 3)
        self.ema_kernel = nn.Parameter(torch.empty(hidden, self.ema_kernel_size))
        self.out_proj = nn.Linear(hidden, self.channels)
        self.drop = nn.Dropout(float(dropout))
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ema_kernel, mean=0.0, std=0.02)

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        kernel = F.softmax(self.ema_kernel, dim=-1).reshape(x.size(1), 1, self.ema_kernel_size)
        return F.conv1d(x, kernel, padding=self.ema_kernel_size // 2, groups=x.size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        tokens = self.norm(x.transpose(1, 2))
        u, v, gate = self.in_proj(tokens).chunk(3, dim=-1)
        v = self.moving_average(v.transpose(1, 2)).transpose(1, 2)
        out = F.silu(u) * v * torch.sigmoid(gate)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class MegaMidResidualConvEncoder(ResidualConvEncoder):
    """Stage56 encoder with MEGA blocks only at selected mid-resolution stages."""

    def __init__(
        self,
        *args,
        mega_stages: Sequence[int] = (1, 2),
        ema_kernel_size: int = 63,
        expansion: float = 2.0,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mega_stage_indices = _valid_stage_indices(mega_stages, len(self.output_channels))
        self.mega_layers = nn.ModuleList(
            MegaMovingAverageGatedAttention1D(
                channels=int(channels),
                ema_kernel_size=ema_kernel_size,
                expansion=expansion,
                dropout=dropout,
                residual_scale_init=residual_scale_init,
            )
            if stage_idx in self.mega_stage_indices
            else nn.Identity()
            for stage_idx, channels in enumerate(self.output_channels)
        )

    def forward(self, x: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            x = self.mega_layers[stage_idx](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUResUNet1D_HyenaBottleneck(BaseBottleneckInnovationResUNet1D):
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
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        hyena_filter_hidden: int = 64,
        dropout: float = 0.0,
        bottleneck_scale_init: float = 0.05,
    ):
        bottleneck = HyenaBottleneck1D(
            channels=int(features_per_stage[-1]),
            filter_hidden=hyena_filter_hidden,
            dropout=dropout,
            bottleneck_scale_init=bottleneck_scale_init,
        )
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            bottleneck=bottleneck,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )


class IQUResUNet1D_SpectralLowRankBottleneck(BaseBottleneckInnovationResUNet1D):
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
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        mode_count: int = 32,
        spectral_rank: int = 4,
        dropout: float = 0.10,
        bottleneck_scale_init: float = 0.02,
    ):
        bottleneck = SpectralLowRankBottleneck1D(
            channels=int(features_per_stage[-1]),
            mode_count=mode_count,
            spectral_rank=spectral_rank,
            dropout=dropout,
            bottleneck_scale_init=bottleneck_scale_init,
        )
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            bottleneck=bottleneck,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )


class IQUResUNet1D_MegaMidEncoder(nn.Module):
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
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        mega_stages: Sequence[int] = (1, 2),
        ema_kernel_size: int = 63,
        expansion: float = 2.0,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.encoder = MegaMidResidualConvEncoder(
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
            mega_stages=mega_stages,
            ema_kernel_size=ema_kernel_size,
            expansion=expansion,
            dropout=dropout,
            residual_scale_init=residual_scale_init,
        )
        self.decoder = PlainUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        return self.decoder(self.encoder(x))
