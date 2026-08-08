"""ICASSP2024 Baseline WaveNet — PyTorch reimplementation for IQUMamba1D pipeline.

Faithful translation of the WaveNet from
``icassp2024rfchallenge/src/torchwavenet.py`` into the pipeline format.

Architecture summary
====================
- Input projection: Conv1d(input_channels → residual_channels, kernel=1)
- N residual blocks with exponentially increasing dilation:
    dilation = 2^(i % dilation_cycle_length)
  Each block uses gated activation (sigmoid × tanh) and produces
  a residual path + skip path (both residual_channels wide).
- Skip aggregation: sum of all skip outputs, normalized by sqrt(N)
- Output projection: Conv1d(residual_channels → num_classes, kernel=1)
  with zero-initialized weights for stable start.

Default config (matching ICASSP original):
  residual_channels = 128
  residual_layers = 30
  dilation_cycle_length = 10  (3 full cycles: max dilation = 512)

The original model has input_channels == output_channels == 2 (single
source separation). Here we parameterize output independently as
``num_classes`` so it can output multiple sources (e.g., 4 for 2 IQ sources).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


def _kaiming_conv1d(*args, **kwargs):
    """Conv1d with Kaiming normal initialization (matching ICASSP original)."""
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class WaveNetResidualBlock(nn.Module):
    """Gated residual block with dilated convolution.

    Uses the classic WaveNet gating mechanism:
        y = sigmoid(gate) * tanh(filter)
    Then splits output into residual + skip connections.
    """

    def __init__(self, residual_channels: int, dilation: int):
        super().__init__()
        self.dilated_conv = _kaiming_conv1d(
            residual_channels, 2 * residual_channels,
            kernel_size=3, padding=dilation, dilation=dilation,
        )
        self.output_projection = _kaiming_conv1d(
            residual_channels, 2 * residual_channels, kernel_size=1,
        )

    def forward(self, x):
        y = self.dilated_conv(x)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class ICASPBaselineWaveNet(nn.Module):
    """ICASSP2024 RF Challenge baseline WaveNet (PyTorch).

    Args:
        input_channels:       Number of input channels (2 for IQ).
        num_classes:           Number of output channels (2*num_sources).
        residual_channels:    Width of residual/skip paths (default 128).
        residual_layers:      Number of residual blocks (default 30).
        dilation_cycle_length: Dilation cycle (default 10 → max dilation 512).
    """

    def __init__(self,
                 input_channels: int = 2,
                 num_classes: int = 4,
                 residual_channels: int = 128,
                 residual_layers: int = 30,
                 dilation_cycle_length: int = 10):
        super().__init__()
        self.num_layers = residual_layers

        # Input projection
        self.input_projection = _kaiming_conv1d(
            input_channels, residual_channels, kernel_size=1,
        )

        # Residual stack with exponentially increasing dilation
        # Keep the official module name so the released ``weights.pt`` files
        # load strictly without a lossy/partial checkpoint path.
        self.residual_layers = nn.ModuleList([
            WaveNetResidualBlock(
                residual_channels,
                dilation=2 ** (i % dilation_cycle_length),
            )
            for i in range(residual_layers)
        ])

        # Skip aggregation → output
        self.skip_projection = _kaiming_conv1d(
            residual_channels, residual_channels, kernel_size=1,
        )
        self.output_projection = _kaiming_conv1d(
            residual_channels, num_classes, kernel_size=1,
        )
        # Zero-init output for stable training start (matching ICASSP original)
        nn.init.zeros_(self.output_projection.weight)

    @property
    def residual_blocks(self):
        """Backward-compatible alias used by the project's WaveNet variants."""

        return self.residual_layers

    def forward(self, x):
        """
        Args:
            x: (B, input_channels, L) — raw IQ waveform
        Returns:
            out: (B, num_classes, L) — separated sources
        """
        x = self.input_projection(x)
        x = F.relu(x)

        skip_sum = None
        for block in self.residual_layers:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        x = skip_sum / sqrt(self.num_layers)
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
