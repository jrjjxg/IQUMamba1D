"""RFDEMUCS waveform estimator for the ICASSP 2024 RF Challenge.

This is an independent, dependency-free implementation of the architecture
described by Yapar et al., "DEMUCS for Data-Driven RF Signal Denoising".  It
follows the public RFDEMUCS/Meta denoiser topology: a five-level convolutional
U-Net, a two-layer bidirectional LSTM bottleneck, and sinc resampling.  The
competition waveform path predicts one raw I/Q SOI and intentionally performs
no input standard-deviation normalization, as specified in the paper.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _sinc_interpolation_kernel(zeros: int = 56) -> torch.Tensor:
    """Build the Hann-windowed half-sample sinc kernel used by DEMUCS."""

    if int(zeros) < 1:
        raise ValueError("sinc kernel zeros must be positive")
    window = torch.hann_window(4 * int(zeros) + 1, periodic=False)
    odd_window = window[1::2]
    time = torch.linspace(-int(zeros) + 0.5, int(zeros) - 0.5, 2 * int(zeros))
    time = time * math.pi
    sinc = torch.where(time == 0, torch.ones_like(time), torch.sin(time) / time)
    return (sinc * odd_window).view(1, 1, -1)


class RFDEMUCSBLSTM(nn.Module):
    """Two-layer BLSTM followed by the published bidirectional projection."""

    def __init__(self, channels: int, layers: int = 2) -> None:
        super().__init__()
        self.channels = int(channels)
        self.layers = int(layers)
        self.lstm = nn.LSTM(
            input_size=self.channels,
            hidden_size=self.channels,
            num_layers=self.layers,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * self.channels, self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.permute(2, 0, 1)
        sequence, _ = self.lstm(sequence)
        return self.projection(sequence).permute(1, 2, 0)


def _rescale_convolutions(module: nn.Module, reference: float) -> None:
    """Apply the DEMUCS variance-preserving convolution initialization."""

    if float(reference) <= 0:
        return
    with torch.no_grad():
        for child in module.modules():
            if not isinstance(child, (nn.Conv1d, nn.ConvTranspose1d)):
                continue
            standard_deviation = child.weight.std()
            scale = torch.sqrt(standard_deviation / float(reference))
            child.weight.div_(scale)
            if child.bias is not None:
                child.bias.div_(scale)


class RFDEMUCS(nn.Module):
    """TUB RFDEMUCS time-domain SOI estimator.

    Hidden widths grow by ``growth`` at every encoder level.  For the paper's
    default setting ``hidden=64, depth=5`` the BLSTM width is 1024.  The
    QPSK+CommSignal3 configuration instead uses ``hidden=80``, ``stride=4``
    and ``resample=4``; the RF Challenge factory applies that case override.
    """

    def __init__(
        self,
        *,
        input_channels: int = 2,
        num_classes: int = 2,
        hidden: int = 64,
        depth: int = 5,
        kernel_size: int = 8,
        stride: int = 2,
        resample: int = 2,
        growth: float = 2.0,
        max_hidden: int = 10_000,
        normalize: bool = False,
        glu: bool = True,
        rescale: float = 0.1,
        normalization_floor: float = 1e-3,
        lstm_layers: int = 2,
        sinc_zeros: int = 56,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2 or int(num_classes) != 2:
            raise ValueError("paper RFDEMUCS expects one I/Q input and one I/Q SOI")
        if int(depth) < 1 or int(hidden) < 1:
            raise ValueError("RFDEMUCS depth and hidden width must be positive")
        if int(kernel_size) < int(stride) or int(stride) < 1:
            raise ValueError("expected kernel_size >= stride >= 1")
        if int(resample) not in {1, 2, 4}:
            raise ValueError("RFDEMUCS resample must be 1, 2, or 4")

        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.resample = int(resample)
        self.growth = float(growth)
        self.max_hidden = int(max_hidden)
        self.normalize = bool(normalize)
        self.normalization_floor = float(normalization_floor)
        self.paper_architecture = True
        self.uses_bidirectional_lstm = True
        self.register_buffer(
            "sinc_kernel",
            _sinc_interpolation_kernel(int(sinc_zeros)),
            persistent=False,
        )

        encoder: list[nn.Module] = []
        decoder: list[nn.Module] = []
        input_width = self.input_channels
        output_width = self.num_classes
        width = self.hidden
        channel_scale = 2 if bool(glu) else 1
        for index in range(self.depth):
            activation = nn.GLU(dim=1) if bool(glu) else nn.ReLU()
            encoder.append(
                nn.Sequential(
                    nn.Conv1d(
                        input_width,
                        width,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                    ),
                    nn.ReLU(),
                    nn.Conv1d(width, channel_scale * width, kernel_size=1),
                    activation,
                )
            )
            decode_layers: list[nn.Module] = [
                nn.Conv1d(width, channel_scale * width, kernel_size=1),
                nn.GLU(dim=1) if bool(glu) else nn.ReLU(),
                nn.ConvTranspose1d(
                    width,
                    output_width,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                ),
            ]
            if index > 0:
                decode_layers.append(nn.ReLU())
            decoder.insert(0, nn.Sequential(*decode_layers))
            output_width = width
            input_width = width
            width = min(int(self.growth * width), self.max_hidden)

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.bottleneck_channels = int(input_width)
        self.lstm = RFDEMUCSBLSTM(self.bottleneck_channels, layers=lstm_layers)
        _rescale_convolutions(self, float(rescale))

    def _upsample2(self, x: torch.Tensor) -> torch.Tensor:
        *leading, length = x.shape
        kernel = self.sinc_kernel.to(dtype=x.dtype)
        zeros = kernel.shape[-1] // 2
        interpolated = F.conv1d(
            x.reshape(-1, 1, length), kernel, padding=zeros
        )[..., 1:].reshape(*leading, length)
        return torch.stack((x, interpolated), dim=-1).reshape(*leading, -1)

    def _downsample2(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2:
            x = F.pad(x, (0, 1))
        even = x[..., ::2]
        odd = x[..., 1::2]
        *leading, length = odd.shape
        kernel = self.sinc_kernel.to(dtype=x.dtype)
        zeros = kernel.shape[-1] // 2
        filtered = F.conv1d(
            odd.reshape(-1, 1, length), kernel, padding=zeros
        )[..., :-1].reshape(*leading, length)
        return 0.5 * (even + filtered)

    def valid_length(self, length: int) -> int:
        """Return the nearest padded length exactly invertible by the U-Net."""

        valid = math.ceil(int(length) * self.resample)
        for _ in range(self.depth):
            valid = math.ceil((valid - self.kernel_size) / self.stride) + 1
            valid = max(valid, 1)
        for _ in range(self.depth):
            valid = (valid - 1) * self.stride + self.kernel_size
        return int(math.ceil(valid / self.resample))

    @property
    def total_stride(self) -> int:
        return self.stride ** self.depth // self.resample

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.ndim != 3 or mixture.shape[1] != self.input_channels:
            raise ValueError(
                "RFDEMUCS expects [B, 2, L] raw I/Q, got "
                f"{tuple(mixture.shape)}"
            )
        original_length = mixture.shape[-1]
        if self.normalize:
            mono = mixture.mean(dim=1, keepdim=True)
            scale = mono.std(dim=-1, keepdim=True)
            x = mixture / (self.normalization_floor + scale)
        else:
            scale = 1.0
            x = mixture

        x = F.pad(x, (0, self.valid_length(original_length) - original_length))
        if self.resample >= 2:
            x = self._upsample2(x)
        if self.resample == 4:
            x = self._upsample2(x)

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = self.lstm(x)
        for decode in self.decoder:
            skip = skips.pop()
            x = decode(x + skip[..., : x.shape[-1]])

        if self.resample >= 2:
            x = self._downsample2(x)
        if self.resample == 4:
            x = self._downsample2(x)
        return scale * x[..., :original_length]
