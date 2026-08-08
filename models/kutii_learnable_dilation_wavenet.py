"""KU-TII-style learnable-dilation WaveNet for the ICASSP 2024 RF Challenge.

The public challenge report attributes the winning KU-TII submission to a
WaveNet with 256 residual channels, learnable dilation, per-mixture dilation
cycle selection, and additional CommSignal2 training examples. The original
source code and the exact learnable-dilation parameterization were not
released. This module is therefore a transparent *KU-TII-style reproduction*,
not a claim of bit-for-bit champion-code recovery.

The architecture is otherwise faithful to the official PyTorch WaveNet:

* 30 gated residual blocks;
* a three-tap dilated convolution in each block;
* residual/skip width 256 by default;
* raw I/Q input and a source-slot I/Q output;
* MSE training supplied by ``rfchallenge.training``.

Each dilation is a scalar trainable parameter. A differentiable linear blend
of the adjacent integer-dilation Conv1d outputs implements fractional
dilation efficiently, while preserving the exact official fixed-dilation
operator at every integer initialization point.
"""

from __future__ import annotations

from math import sqrt
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as functional


def _kaiming_conv1d(*args, **kwargs) -> nn.Conv1d:
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class LearnableDilationConv1d(nn.Module):
    """A same-length three-tap convolution with a trainable continuous dilation.

    For ``d = floor(d) + alpha``, the output is

    ``(1-alpha) * Conv(d=floor(d)) + alpha * Conv(d=ceil(d))``.

    This interpolation is differentiable with respect to ``d`` inside each
    integer interval and avoids the memory cost of a manually sampled dynamic
    convolution on 40,960-point RF frames. At an integer dilation it produces
    precisely the same result as a regular ``Conv1d`` at that dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        initial_dilation: float,
        max_dilation: int = 1024,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if initial_dilation < 1:
            raise ValueError("initial_dilation must be >= 1")
        if max_dilation < 1:
            raise ValueError("max_dilation must be >= 1")
        if initial_dilation > max_dilation:
            raise ValueError("initial_dilation cannot exceed max_dilation")
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.dilation = nn.Parameter(torch.tensor(float(initial_dilation)))
        self.max_dilation = int(max_dilation)
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            bound = 1.0 / sqrt(in_channels * 3)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_dilation(self) -> torch.Tensor:
        """Return the bounded continuous dilation parameter."""

        return self.dilation.clamp(1.0, float(self.max_dilation))

    @torch.no_grad()
    def project_dilation_(self) -> None:
        """Keep the underlying parameter in the valid integer-conv range."""

        self.dilation.clamp_(1.0, float(self.max_dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilation = self.effective_dilation()
        lower = int(torch.floor(dilation.detach()).item())
        lower = max(1, min(lower, self.max_dilation))
        upper = min(lower + 1, self.max_dilation)
        lower_output = functional.conv1d(
            x,
            self.weight,
            self.bias,
            padding=lower,
            dilation=lower,
        )
        if upper == lower:
            return lower_output
        upper_output = functional.conv1d(
            x,
            self.weight,
            self.bias,
            padding=upper,
            dilation=upper,
        )
        alpha = (dilation - float(lower)).to(dtype=lower_output.dtype)
        return lower_output + alpha * (upper_output - lower_output)


class LearnableDilationResidualBlock(nn.Module):
    """Gated WaveNet residual block with a learnable dilation operator."""

    def __init__(
        self,
        residual_channels: int,
        initial_dilation: float,
        max_dilation: int,
    ) -> None:
        super().__init__()
        self.dilated_conv = LearnableDilationConv1d(
            residual_channels,
            2 * residual_channels,
            initial_dilation=initial_dilation,
            max_dilation=max_dilation,
        )
        self.output_projection = _kaiming_conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate_and_filter = self.dilated_conv(x)
        gate, filt = torch.chunk(gate_and_filter, 2, dim=1)
        activated = torch.sigmoid(gate) * torch.tanh(filt)
        residual_and_skip = self.output_projection(activated)
        residual, skip = torch.chunk(residual_and_skip, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class KUTIIStyleLearnableDilationWaveNet(nn.Module):
    """A learnable-dilation reconstruction of the KU-TII WaveNet.

    ``dilation_cycle_length`` is intentionally exposed per model instance so
    it can be selected independently for every SOI/interference case using a
    held-out TestSet1Example validation set, as reported by KU-TII.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        residual_channels: int = 256,
        residual_layers: int = 30,
        dilation_cycle_length: int = 10,
        max_dilation: int = 1024,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"KU-TII WaveNet expects two I/Q inputs, got {input_channels}")
        if num_classes <= 0 or num_classes % 2 != 0:
            raise ValueError(
                "KU-TII-style WaveNet requires an even I/Q output width; "
                f"got num_classes={num_classes}"
            )
        if residual_layers <= 0 or residual_channels <= 0:
            raise ValueError("residual_layers and residual_channels must be positive")
        if dilation_cycle_length <= 0:
            raise ValueError("dilation_cycle_length must be positive")
        if 2 ** (dilation_cycle_length - 1) > max_dilation:
            raise ValueError(
                "max_dilation must cover the largest initialized cycle dilation"
            )
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.num_sources = self.num_classes // 2
        self.residual_channels = int(residual_channels)
        self.residual_layers_count = int(residual_layers)
        self.dilation_cycle_length = int(dilation_cycle_length)
        self.max_dilation = int(max_dilation)
        self.input_projection = _kaiming_conv1d(
            input_channels,
            residual_channels,
            kernel_size=1,
        )
        initial_dilations = [
            float(2 ** (index % dilation_cycle_length))
            for index in range(residual_layers)
        ]
        self.residual_layers = nn.ModuleList(
            [
                LearnableDilationResidualBlock(
                    residual_channels=residual_channels,
                    initial_dilation=dilation,
                    max_dilation=max_dilation,
                )
                for dilation in initial_dilations
            ]
        )
        self.skip_projection = _kaiming_conv1d(
            residual_channels,
            residual_channels,
            kernel_size=1,
        )
        self.output_projection = _kaiming_conv1d(
            residual_channels,
            num_classes,
            kernel_size=1,
        )
        # Exactly mirrors the official WaveNet's stable zero-output start.
        nn.init.zeros_(self.output_projection.weight)

    def effective_dilations(self) -> torch.Tensor:
        """Expose the learned per-layer dilation values for logs and ablations."""

        return torch.stack(
            [block.dilated_conv.effective_dilation() for block in self.residual_layers]
        )

    @torch.no_grad()
    def project_learnable_dilations_(self) -> None:
        """Project every learned dilation after an optimizer update."""

        for block in self.residual_layers:
            block.dilated_conv.project_dilation_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(
                "Expected I/Q input with shape (B, 2, L), "
                f"got {tuple(x.shape)}"
            )
        hidden = functional.relu(self.input_projection(x))
        skip_sum: torch.Tensor | None = None
        for block in self.residual_layers:
            hidden, skip = block(hidden)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        assert skip_sum is not None
        output = skip_sum / sqrt(len(self.residual_layers))
        output = functional.relu(self.skip_projection(output))
        return self.output_projection(output)


class KUTIIDualSourceWaveNet(KUTIIStyleLearnableDilationWaveNet):
    """KU-TII-style WaveNet adapted to the ordinary multi-source BSS contract.

    The RF Challenge entry predicts one known SOI (two output channels). The
    main IQUMamba pipeline expects source slots packed as
    ``[S1_I, S1_Q, S2_I, S2_Q, ...]``. This subclass keeps one shared
    learnable-dilation trunk and changes only the final source-slot width.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 256,
        residual_layers: int = 30,
        dilation_cycle_length: int = 10,
        max_dilation: int = 1024,
    ) -> None:
        if num_classes not in (4, 6):
            raise ValueError(
                "KUTIIDualSourceWaveNet supports two or three source slots; "
                f"expected num_classes=4 or 6, got {num_classes}"
            )
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
            max_dilation=max_dilation,
        )
