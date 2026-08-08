"""ICASSP2024 Baseline UNet — PyTorch reimplementation.

Faithful translation of the TensorFlow UNet from
``icassp2024rfchallenge/src/unet_model.py`` into PyTorch.

Architecture summary
====================
- 5-stage encoder with Conv1D + MaxPool + Dropout
- Bottleneck with Conv1D
- 5-stage decoder with ConvTranspose1d + skip concat + Dropout + Conv1D
- First encoder stage uses a large kernel (long_k_sz=101) for wide receptive field
- All other convolutions use kernel_size=3
- Channel widths: encoder=[8k,8k,8k,8k,8k], decoder=[8k,8k,4k,2k,1k] where k=k_neurons
- Final Conv1D(num_classes, kernel=1) for per-sample source output

This model is integrated into the IQUMamba1D pipeline as a baseline sanity
check — if the pipeline is correct, even this simple UNet should produce
reasonable separation results.

Note: The original TF model uses channel-last (B, L, C).
      This PyTorch version uses channel-first (B, C, L).
"""

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Two Conv1d + ReLU layers (same padding)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ICASPBaselineUNet(nn.Module):
    """ICASSP2024 RF Challenge baseline UNet (PyTorch).

    Args:
        input_channels: Number of input channels (2 for IQ).
        num_classes:    Number of output channels (2*num_sources for IQ separation).
        k_neurons:      Base channel multiplier (default 32, matching TF original).
        k_sz:           Small kernel size for most convolutions (default 3).
        long_k_sz:      Large kernel size for the first encoder stage (default 101).
        dropout_first:  Dropout rate for the first encoder stage (default 0.25).
        dropout_rest:   Dropout rate for other stages (default 0.5).
    """

    def __init__(self,
                 input_channels: int = 2,
                 num_classes: int = 4,
                 k_neurons: int = 32,
                 k_sz: int = 3,
                 long_k_sz: int = 101,
                 dropout_first: float = 0.25,
                 dropout_rest: float = 0.5):
        super().__init__()

        # Encoder channel multipliers (matching TF original)
        enc_mults = [8, 8, 8, 8, 8]
        dec_mults = [8, 8, 4, 2, 1]

        # Input BatchNorm (matching TF's BatchNormalization on input)
        self.input_bn = nn.BatchNorm1d(input_channels)

        # ---- Encoder ----
        self.enc_convs = nn.ModuleList()
        self.enc_dropouts = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        in_ch = input_channels
        for i, m in enumerate(enc_mults):
            out_ch = k_neurons * m
            k = long_k_sz if i == 0 else k_sz
            self.enc_convs.append(ConvBlock1D(in_ch, out_ch, kernel_size=k))
            drop_rate = dropout_first if i == 0 else dropout_rest
            self.enc_dropouts.append(nn.Dropout(p=drop_rate))
            in_ch = out_ch

        # ---- Bottleneck ----
        bottleneck_ch = k_neurons * 8
        self.bottleneck = ConvBlock1D(in_ch, bottleneck_ch, kernel_size=k_sz)

        # ---- Decoder ----
        self.dec_upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        self.dec_dropouts = nn.ModuleList()

        in_ch = bottleneck_ch
        for i, m in enumerate(dec_mults):
            out_ch = k_neurons * m
            # Transposed conv for upsampling (stride=2)
            self.dec_upconvs.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k_sz,
                                   stride=2, padding=k_sz // 2,
                                   output_padding=1)
            )
            # After concat with skip: in_channels = out_ch + enc_ch
            enc_ch = k_neurons * enc_mults[-(i + 1)]
            self.dec_convs.append(ConvBlock1D(out_ch + enc_ch, out_ch, kernel_size=k_sz))
            self.dec_dropouts.append(nn.Dropout(p=dropout_rest))
            in_ch = out_ch

        # ---- Output ----
        self.output_conv = nn.Conv1d(k_neurons * dec_mults[-1], num_classes,
                                     kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C_in, L) — raw IQ waveform, channel-first
        Returns:
            out: (B, num_classes, L) — separated sources
        """
        x = self.input_bn(x)

        # Encoder
        skips = []
        for conv, dropout in zip(self.enc_convs, self.enc_dropouts):
            x = conv(x)
            skips.append(x)
            x = self.pool(x)
            x = dropout(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for upconv, dec_conv, dropout, skip in zip(
            self.dec_upconvs, self.dec_convs, self.dec_dropouts,
            reversed(skips)
        ):
            x = upconv(x)
            # Handle possible length mismatch from pooling
            if x.shape[-1] != skip.shape[-1]:
                x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = dropout(x)
            x = dec_conv(x)

        return self.output_conv(x)
