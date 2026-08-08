"""Compute-matched RF U-Net with shallow dilated context and BiMamba.

The ICASSP baseline U-Net is intentionally a compact reference architecture.
This model keeps its useful multi-resolution U-Net topology but makes the
high-rate I/Q feature processing more suitable for RF waveform separation:

* learned stride-2 analysis filters replace fixed max pooling;
* the first two encoder resolutions receive inexpensive, gated multi-dilation
  residual context blocks;
* decoder skip paths are conditionally gated by the current decoder feature;
* a bidirectional Mamba block operates only at the L/32 bottleneck, where
  long-range context is inexpensive.

It accepts and returns channel-first tensors, ``(batch, channels, time)``.
``num_classes`` is normally set to ``2 * num_sources`` by ``main.py``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # Keep imports usable for configuration-only tools without Mamba installed.
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - exercised in environments without Mamba.
    Mamba = None


class _ConvBlock1D(nn.Module):
    """Two same-length temporal convolutions with ReLU activations."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _LearnedDownsample1D(nn.Module):
    """A trainable anti-aliasing/downsampling replacement for max pooling."""

    def __init__(self, channels: int):
        super().__init__()
        self.downsample = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.downsample(x))


class _MultiDilatedGatedResidual1D(nn.Module):
    """Low-rank gated multi-dilation residual context at a high-rate level."""

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        dilations: Iterable[int],
    ):
        super().__init__()
        dilations = tuple(int(dilation) for dilation in dilations)
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must contain positive integers")
        if bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels must be positive")

        self.reduce = nn.Conv1d(channels, bottleneck_channels, kernel_size=1)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    2 * bottleneck_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=bottleneck_channels,
                )
                for dilation in dilations
            ]
        )
        self.project = nn.Conv1d(
            bottleneck_channels * len(dilations),
            channels,
            kernel_size=1,
        )
        # Small, non-zero residual scale keeps the base U-Net behavior stable
        # while allowing this branch to learn immediately.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        branch_outputs = []
        for branch in self.branches:
            filtered, gate = branch(reduced).chunk(2, dim=1)
            branch_outputs.append(torch.tanh(filtered) * torch.sigmoid(gate))
        delta = self.project(torch.cat(branch_outputs, dim=1))
        return x + self.residual_scale * delta


class _GroupedTemporalSkipGate(nn.Module):
    """Gate a wide encoder skip with a compact decoder-derived temporal gate."""

    def __init__(self, decoder_channels: int, skip_channels: int, groups: int = 16):
        super().__init__()
        requested_groups = max(1, min(int(groups), int(skip_channels)))
        while skip_channels % requested_groups != 0:
            requested_groups -= 1
        self.groups = requested_groups
        self.channels_per_group = skip_channels // self.groups
        self.gate = nn.Conv1d(decoder_channels, self.groups, kernel_size=1)
        # Start near an open skip connection, so the model is a stable U-Net at
        # initialization rather than suppressing information before training.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, decoder: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if decoder.shape[-1] != skip.shape[-1]:
            raise ValueError("decoder and skip lengths must match before skip gating")
        gates = torch.sigmoid(self.gate(decoder))
        gates = gates.repeat_interleave(self.channels_per_group, dim=1)
        return skip * gates


class _BiMambaBottleneck1D(nn.Module):
    """Bidirectional Mamba residual mixer used only at the L/32 bottleneck."""

    def __init__(
        self,
        channels: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        scale_init: float,
    ):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "IQResDilatedUNet with use_bottleneck_bimamba=true requires "
                "the mamba_ssm package. Install the project's Mamba dependency "
                "or set use_bottleneck_bimamba=false."
            )
        self.norm = nn.LayerNorm(channels)
        self.forward_mamba = Mamba(
            d_model=channels,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.backward_mamba = Mamba(
            d_model=channels,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.fuse = nn.Linear(2 * channels, channels)
        self.dropout = nn.Dropout(float(dropout))
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(x.transpose(1, 2))
        forward_context = self.forward_mamba(sequence)
        backward_context = self.backward_mamba(torch.flip(sequence, dims=[1]).contiguous())
        backward_context = torch.flip(backward_context, dims=[1])
        delta = self.fuse(torch.cat([forward_context, backward_context], dim=-1))
        delta = self.dropout(delta).transpose(1, 2)
        return x + self.residual_scale * delta


class IQResDilatedUNet(nn.Module):
    """RF waveform U-Net with learned analysis filters and low-rate BiMamba.

    The default ``k_neurons=64`` gives this model a compute budget near the
    public WaveNet baseline when profiled at the challenge waveform length.
    It is deliberately not a direct copy of either official baseline.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        k_neurons: int = 64,
        k_sz: int = 3,
        long_k_sz: int = 101,
        dropout_first: float = 0.10,
        dropout_rest: float = 0.10,
        shallow_dilated_channels: Sequence[int] = (128, 64),
        skip_gate_groups: int = 16,
        use_bottleneck_bimamba: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_scale_init: float = 1e-2,
    ):
        super().__init__()
        if k_neurons <= 0:
            raise ValueError("k_neurons must be positive")

        encoder_width = 8 * int(k_neurons)
        encoder_multipliers = (8, 8, 8, 8, 8)
        decoder_multipliers = (8, 8, 4, 2, 1)
        shallow_dilated_channels = tuple(
            int(channels) for channels in shallow_dilated_channels
        )

        self.input_bn = nn.BatchNorm1d(input_channels)

        self.enc_convs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        self.shallow_context = nn.ModuleList()

        in_channels = input_channels
        for index, multiplier in enumerate(encoder_multipliers):
            out_channels = int(k_neurons) * multiplier
            kernel_size = long_k_sz if index == 0 else k_sz
            self.enc_convs.append(
                _ConvBlock1D(in_channels, out_channels, kernel_size=kernel_size)
            )
            self.downsamples.append(_LearnedDownsample1D(out_channels))
            dropout_rate = dropout_first if index == 0 else dropout_rest
            self.enc_dropouts.append(nn.Dropout(p=float(dropout_rate)))
            in_channels = out_channels

        # Only the high-rate first two levels get residual dilation blocks.
        # This grows their receptive field without paying WaveNet's full-rate
        # cost at every U-Net level.
        context_specs = ((1, 2, 4), (1, 2))
        for index, dilations in enumerate(context_specs):
            if index < len(shallow_dilated_channels):
                self.shallow_context.append(
                    _MultiDilatedGatedResidual1D(
                        encoder_width,
                        shallow_dilated_channels[index],
                        dilations,
                    )
                )
            else:
                self.shallow_context.append(nn.Identity())

        self.bottleneck = _ConvBlock1D(encoder_width, encoder_width, kernel_size=k_sz)
        self.bottleneck_context: nn.Module
        if use_bottleneck_bimamba:
            self.bottleneck_context = _BiMambaBottleneck1D(
                encoder_width,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                dropout=mamba_dropout,
                scale_init=mamba_scale_init,
            )
        else:
            self.bottleneck_context = nn.Identity()

        self.dec_upconvs = nn.ModuleList()
        self.skip_gates = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()

        in_channels = encoder_width
        for multiplier in decoder_multipliers:
            out_channels = int(k_neurons) * multiplier
            self.dec_upconvs.append(
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size=k_sz,
                    stride=2,
                    padding=k_sz // 2,
                    output_padding=1,
                )
            )
            self.skip_gates.append(
                _GroupedTemporalSkipGate(
                    decoder_channels=out_channels,
                    skip_channels=encoder_width,
                    groups=skip_gate_groups,
                )
            )
            self.dec_convs.append(
                _ConvBlock1D(
                    out_channels + encoder_width,
                    out_channels,
                    kernel_size=k_sz,
                )
            )
            self.dec_dropouts.append(nn.Dropout(p=float(dropout_rest)))
            in_channels = out_channels

        self.output_conv = nn.Conv1d(int(k_neurons), num_classes, kernel_size=1)

    @staticmethod
    def _match_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Match a transposed-convolution output to the corresponding skip length."""

        current_length = x.shape[-1]
        if current_length > target_length:
            return x[..., :target_length]
        if current_length < target_length:
            return F.pad(x, (0, target_length - current_length))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)

        skips = []
        for index, (conv, downsample, dropout) in enumerate(
            zip(self.enc_convs, self.downsamples, self.enc_dropouts)
        ):
            x = conv(x)
            if index < len(self.shallow_context):
                x = self.shallow_context[index](x)
            skips.append(x)
            x = dropout(downsample(x))

        x = self.bottleneck_context(self.bottleneck(x))

        for upconv, skip_gate, dec_conv, dropout, skip in zip(
            self.dec_upconvs,
            self.skip_gates,
            self.dec_convs,
            self.dec_dropouts,
            reversed(skips),
        ):
            x = self._match_length(upconv(x), skip.shape[-1])
            gated_skip = skip_gate(x, skip)
            x = dec_conv(dropout(torch.cat([x, gated_skip], dim=1)))

        return self.output_conv(x)
