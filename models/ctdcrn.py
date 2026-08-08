"""
CTDCRN: Complex Time-Domain Dilated Convolutional Recurrent Network

Paper:
  "Single-Channel Blind Source Separation in Wireless Communications:
   A Complex-Domain Deep Learning Approach" (IEEE WCL, 2024)

This is a practical reimplementation for the IQUMamba1D training pipeline.
The model operates directly on I/Q (real/imag) signals and keeps all internal
representations in the complex domain via paired real-valued tensors.

Interface:
  input : (B, 2, T)  where channel 0=I(real), 1=Q(imag)
  output: (B, 2*K, T) for K sources, concatenated as [I1,Q1,I2,Q2,...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


def _split_ri(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 4 or x.size(1) != 2:
        raise ValueError(f"Expected (B, 2, C, T), got {tuple(x.shape)}")
    return x[:, 0, ...], x[:, 1, ...]


def _stack_ri(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    if real.shape != imag.shape:
        raise ValueError(f"Shape mismatch: real={tuple(real.shape)} imag={tuple(imag.shape)}")
    return torch.stack([real, imag], dim=1)


class ComplexConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding: str | int = "same",
    ):
        super().__init__()
        if isinstance(padding, str) and padding.lower() != "same":
            raise ValueError(f"Unsupported padding={padding!r}. Use 'same' or an int.")

        self.conv_re = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            bias=False,
        )
        self.conv_im = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            bias=False,
        )
        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_channels))
            self.bias_im = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        yr = self.conv_re(xr) - self.conv_im(xi)
        yi = self.conv_im(xr) + self.conv_re(xi)
        if self.bias_re is not None:
            yr = yr + self.bias_re.view(1, -1, 1)
            yi = yi + self.bias_im.view(1, -1, 1)
        return _stack_ri(yr, yi)


class ComplexLayerNorm(nn.Module):
    """LayerNorm over channel dim at each time step, applied separately to real/imag."""

    def __init__(self, channels: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.ln_re = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)
        self.ln_im = nn.LayerNorm(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)  # (B, C, T)
        xr = self.ln_re(xr.transpose(1, 2)).transpose(1, 2).contiguous()
        xi = self.ln_im(xi.transpose(1, 2)).transpose(1, 2).contiguous()
        return _stack_ri(xr, xi)


class ComplexGlobalLayerNorm(nn.Module):
    """Global LN over (C, T) per sample, applied separately to real/imag."""

    def __init__(self, channels: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.eps = float(eps)
        if affine:
            self.weight_re = nn.Parameter(torch.ones(channels))
            self.bias_re = nn.Parameter(torch.zeros(channels))
            self.weight_im = nn.Parameter(torch.ones(channels))
            self.bias_im = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight_re", None)
            self.register_parameter("bias_re", None)
            self.register_parameter("weight_im", None)
            self.register_parameter("bias_im", None)

    def _norm(self, x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if weight is not None:
            y = y * weight.view(1, -1, 1) + bias.view(1, -1, 1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        yr = self._norm(xr, self.weight_re, self.bias_re)
        yi = self._norm(xi, self.weight_im, self.bias_im)
        return _stack_ri(yr, yi)


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)
        return _stack_ri(self.act(xr), self.act(xi))


class ComplexHierarchicalEncoder(nn.Module):
    """
    CHE (Fig.2):
      ComplexConv (1 -> M) -> cLN -> ComplexConv (M -> N)

    Returns:
      z: (B, 2, M, T)  (used for mask application)
      y: (B, 2, N, T)  (fed to separator)
    """

    def __init__(self, kernel_size: int, M: int, N: int, eps: float = 1e-8):
        super().__init__()
        self.conv1 = ComplexConv1d(1, M, kernel_size=kernel_size, padding="same", bias=True)
        self.cln = ComplexLayerNorm(M, eps=eps, affine=True)
        self.conv2 = ComplexConv1d(M, N, kernel_size=kernel_size, padding="same", bias=True)

    def forward(self, x_B2T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_B2T.ndim != 3 or x_B2T.size(1) != 2:
            raise ValueError(f"Expected (B, 2, T) I/Q input, got {tuple(x_B2T.shape)}")
        x = x_B2T.unsqueeze(2)  # (B, 2, 1, T)
        z = self.conv1(x)
        z = self.cln(z)
        y = self.conv2(z)
        return z, y


class ComplexDilatedConvModule(nn.Module):
    """
    CDCM (Fig.3, simplified):
      1x1 ComplexConv (C -> U) -> LeakyReLU -> cGln
      Depthwise dilated ComplexConv (U -> U) -> LeakyReLU -> cGln
      1x1 ComplexConv (U -> C)
      Residual add (complex)
    """

    def __init__(
        self,
        channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        dilation: int,
        eps: float = 1e-8,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        self.in_proj = ComplexConv1d(channels, bottleneck_channels, kernel_size=1, padding=0, bias=True)
        self.act1 = ComplexLeakyReLU(negative_slope=leaky_relu_slope)
        self.gln1 = ComplexGlobalLayerNorm(bottleneck_channels, eps=eps, affine=True)

        self.dconv = ComplexConv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=bottleneck_channels,
            padding="same",
            bias=True,
        )
        self.act2 = ComplexLeakyReLU(negative_slope=leaky_relu_slope)
        self.gln2 = ComplexGlobalLayerNorm(bottleneck_channels, eps=eps, affine=True)

        self.out_proj = ComplexConv1d(bottleneck_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.in_proj(x)
        y = self.act1(y)
        y = self.gln1(y)
        y = self.dconv(y)
        y = self.act2(y)
        y = self.gln2(y)
        y = self.out_proj(y)
        return y + res


class ComplexLSTMBlock(nn.Module):
    """CLSTM per (11)(12) using two real LSTMs (r/i) and combining outputs."""

    def __init__(self, channels: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.channels = int(channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        self.lstm_r = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )
        self.lstm_i = nn.LSTM(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            bidirectional=False,
        )
        self.proj = None
        if hidden_size != channels:
            self.proj = ComplexConv1d(hidden_size, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = _split_ri(x)  # (B, C, T)
        pr = xr.transpose(1, 2).transpose(0, 1).contiguous()  # (T, B, C)
        pi = xi.transpose(1, 2).transpose(0, 1).contiguous()

        o_rr, _ = self.lstm_r(pr)
        o_ri, _ = self.lstm_r(pi)
        o_ir, _ = self.lstm_i(pr)
        o_ii, _ = self.lstm_i(pi)

        lr = o_rr - o_ii
        li = o_ri + o_ir

        yr = lr.transpose(0, 1).transpose(1, 2).contiguous()  # (B, H, T)
        yi = li.transpose(0, 1).transpose(1, 2).contiguous()
        y = _stack_ri(yr, yi)  # (B, 2, H, T)
        if self.proj is not None:
            y = self.proj(y)
        return y


class CTDCRNLink(nn.Module):
    def __init__(
        self,
        N: int,
        M: int,
        U: int,
        S: int,
        V: int,
        L: int,
        H: int,
        eps: float = 1e-8,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        dilations = [2**i for i in range(int(V))]
        self.pre = nn.ModuleList(
            [
                ComplexDilatedConvModule(
                    channels=N,
                    bottleneck_channels=U,
                    kernel_size=S,
                    dilation=d,
                    eps=eps,
                    leaky_relu_slope=leaky_relu_slope,
                )
                for d in dilations
            ]
        )
        self.clstm = ComplexLSTMBlock(channels=N, hidden_size=H, num_layers=L)
        self.post = nn.ModuleList(
            [
                ComplexDilatedConvModule(
                    channels=N,
                    bottleneck_channels=U,
                    kernel_size=S,
                    dilation=d,
                    eps=eps,
                    leaky_relu_slope=leaky_relu_slope,
                )
                for d in dilations
            ]
        )
        self.to_mask = ComplexConv1d(N, M, kernel_size=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = y
        for m in self.pre:
            h = m(h)
        h = self.clstm(h)
        for m in self.post:
            h = m(h)
        m = self.to_mask(h)
        mr, mi = _split_ri(m)
        mr = self.sigmoid(mr)
        mi = self.sigmoid(mi)
        return _stack_ri(mr, mi)


class ComplexDecoder(nn.Module):
    def __init__(self, kernel_size: int, M: int):
        super().__init__()
        self.conv = ComplexConv1d(M, 1, kernel_size=kernel_size, padding="same", bias=True)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        y = self.conv(q)  # (B, 2, 1, T)
        return y.squeeze(2)


@dataclass
class CTDCRNConfig:
    n_srcs: int = 2
    J: int = 2
    M: int = 128
    N: int = 32
    U: int = 128
    S: int = 3
    V: int = 8
    L: int = 1
    H: int = 32
    eps: float = 1e-8
    leaky_relu_slope: float = 0.01


class CTDCRNSeparator1D(nn.Module):
    def __init__(self, cfg: CTDCRNConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ComplexHierarchicalEncoder(kernel_size=cfg.J, M=cfg.M, N=cfg.N, eps=cfg.eps)
        self.links = nn.ModuleList(
            [
                CTDCRNLink(
                    N=cfg.N,
                    M=cfg.M,
                    U=cfg.U,
                    S=cfg.S,
                    V=cfg.V,
                    L=cfg.L,
                    H=cfg.H,
                    eps=cfg.eps,
                    leaky_relu_slope=cfg.leaky_relu_slope,
                )
                for _ in range(int(cfg.n_srcs))
            ]
        )
        self.decoder = ComplexDecoder(kernel_size=cfg.J, M=cfg.M)

    def forward(self, x_B2T: torch.Tensor) -> torch.Tensor:
        z, y = self.encoder(x_B2T)
        zr, zi = _split_ri(z)
        outs: List[torch.Tensor] = []
        for link in self.links:
            mask = link(y)
            mr, mi = _split_ri(mask)
            qr = zr * mr
            qi = zi * mi
            q = _stack_ri(qr, qi)
            s_hat = self.decoder(q)  # (B, 2, T)
            outs.append(s_hat)
        return torch.cat(outs, dim=1)
