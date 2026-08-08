"""Faithful 1D adaptations of recent receptive-field operators.

The source papers define image or time-series blocks, not RF source separators.
This module preserves each paper's central operator and adapts only its spatial
axis to one-dimensional IQ features.  Operators are inserted after selected
Stage-4 encoder stages instead of being used as a single input preprocessor.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D


def _odd(value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError("kernel sizes must be positive")
    return value if value % 2 else value + 1


def _fuse_conv_bn_1d(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    std = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / std
    kernel = conv.weight * scale[:, None, None]
    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    bias = bn.bias + (conv_bias - bn.running_mean) * scale
    return kernel, bias


def _batch_conv1d(x: torch.Tensor, weight: torch.Tensor, padding: int) -> torch.Tensor:
    """Apply one dense convolution kernel per batch item."""
    batch, channels, length = x.shape
    out_channels = weight.size(1)
    y = F.conv1d(
        x.reshape(1, batch * channels, length),
        weight.reshape(batch * out_channels, channels, weight.size(-1)),
        padding=padding,
        groups=batch,
    )
    return y.reshape(batch, out_channels, y.size(-1))


def _linear_sample_grouped(
    x: torch.Tensor, positions: torch.Tensor, groups: int,
) -> torch.Tensor:
    """Sample group-local features at continuous 1D positions.

    x: [B,C,L], positions: [B,G,K,L], return: [B,G,C/G,K,L].
    """
    batch, channels, length = x.shape
    group_channels = channels // groups
    points = positions.size(2)
    lower = positions.floor()
    upper = lower + 1
    fraction = positions - lower
    lower_idx = lower.long().clamp(0, length - 1)
    upper_idx = upper.long().clamp(0, length - 1)
    values = x.reshape(batch, groups, group_channels, length)
    values = values[:, :, :, None, :].expand(-1, -1, -1, points, -1)
    lo = torch.gather(values, -1, lower_idx[:, :, None].expand(-1, -1, group_channels, -1, -1))
    hi = torch.gather(values, -1, upper_idx[:, :, None].expand(-1, -1, group_channels, -1, -1))
    lo_valid = ((lower >= 0) & (lower < length)).to(x.dtype)[:, :, None]
    hi_valid = ((upper >= 0) & (upper < length)).to(x.dtype)[:, :, None]
    return (1 - fraction[:, :, None]) * lo * lo_valid + fraction[:, :, None] * hi * hi_valid


class FrequencySelection1D(nn.Module):
    """FADC/FDConv FFT-centred progressive band decomposition."""

    def __init__(self, channels: int, cutoffs: Sequence[int] = (2, 4, 8),
                 spatial_groups: int = 1, include_low: bool = True) -> None:
        super().__init__()
        self.cutoffs = tuple(int(v) for v in cutoffs)
        self.spatial_groups = min(int(spatial_groups), int(channels))
        if channels % self.spatial_groups:
            raise ValueError("channels must be divisible by frequency-selection groups")
        self.include_low = bool(include_low)
        count = len(self.cutoffs) + int(self.include_low)
        self.band_weights = nn.ModuleList([
            nn.Conv1d(channels, self.spatial_groups, 3, padding=1,
                      groups=self.spatial_groups) for _ in range(count)
        ])
        for layer in self.band_weights:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def decompose(self, x: torch.Tensor) -> list[torch.Tensor]:
        length = x.size(-1)
        spectrum = torch.fft.fftshift(torch.fft.fft(x.float(), dim=-1, norm="ortho"), dim=-1)
        pre = x.float()
        bands = []
        center = length // 2
        for cutoff in self.cutoffs:
            half_width = max(1, int(round(length / (2 * cutoff))))
            mask = torch.zeros(length, device=x.device, dtype=spectrum.dtype)
            mask[max(0, center - half_width):min(length, center + half_width)] = 1
            low = torch.fft.ifft(
                torch.fft.ifftshift(spectrum * mask[None, None], dim=-1),
                dim=-1, norm="ortho",
            ).real
            bands.append(pre - low)
            pre = low
        if self.include_low:
            bands.append(pre)
        return [band.to(x.dtype) for band in bands]

    def forward(self, x: torch.Tensor, attention_features: torch.Tensor | None = None) -> torch.Tensor:
        attention_features = x if attention_features is None else attention_features
        batch, channels, length = x.shape
        grouped_channels = channels // self.spatial_groups
        selected = []
        for band, predictor in zip(self.decompose(x), self.band_weights):
            weight = 2.0 * torch.sigmoid(predictor(attention_features))
            selected.append((
                band.reshape(batch, self.spatial_groups, grouped_channels, length)
                * weight[:, :, None]
            ).reshape_as(x))
        return torch.stack(selected).sum(0)


class FrequencyAdaptiveDilatedConv1D(nn.Module):
    """FADC: frequency selection + continuous AdaDR + AdaKern."""

    def __init__(self, channels: int, kernel_size: int = 7,
                 dilations: Sequence[int] = (2, 4, 8), groups: int | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = _odd(kernel_size)
        self.groups = self.channels if groups is None else int(groups)
        if self.channels % self.groups:
            raise ValueError("FADC channels must be divisible by groups")
        self.cutoffs = tuple(int(v) for v in dilations)
        self.frequency_selection = FrequencySelection1D(
            self.channels, self.cutoffs, spatial_groups=min(self.groups, self.channels),
        )
        self.weight = nn.Parameter(torch.empty(self.channels, self.channels // self.groups, self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.dilation_predictor = nn.Conv1d(
            self.channels, self.groups, self.kernel_size,
            padding=self.kernel_size // 2,
        )
        self.mask_predictor = nn.Conv1d(
            self.channels, self.groups * self.kernel_size, self.kernel_size,
            padding=self.kernel_size // 2,
        )
        hidden = max(16, self.channels // 16)
        self.kernel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv1d(hidden, 2 * self.channels, 1),
        )
        nn.init.zeros_(self.dilation_predictor.weight)
        nn.init.ones_(self.dilation_predictor.bias)
        nn.init.zeros_(self.mask_predictor.weight)
        nn.init.zeros_(self.mask_predictor.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = self.frequency_selection(x)
        batch, _, length = x.shape
        dilation = self.dilation_predictor(selected).abs().clamp_min(1e-4)
        taps = torch.arange(
            -(self.kernel_size // 2), self.kernel_size // 2 + 1,
            device=x.device, dtype=x.dtype,
        )
        base = torch.arange(length, device=x.device, dtype=x.dtype)[None, None, None]
        positions = base + dilation[:, :, None] * taps[None, None, :, None]
        sampled = _linear_sample_grouped(selected, positions, self.groups)
        modulation = torch.sigmoid(self.mask_predictor(selected)).reshape(
            batch, self.groups, self.kernel_size, length,
        )
        sampled = sampled * modulation[:, :, None]

        gates = torch.sigmoid(self.kernel_attention(selected)).reshape(batch, 2, self.channels, 1)
        kernel_mean = self.weight.mean(-1, keepdim=True)
        kernel_high = self.weight - kernel_mean
        dynamic_weight = (
            gates[:, 0, :, :, None] * kernel_mean[None]
            + gates[:, 1, :, :, None] * kernel_high[None]
        )
        group_channels = self.channels // self.groups
        dynamic_weight = dynamic_weight.reshape(batch, self.groups, group_channels, group_channels, self.kernel_size)
        return torch.einsum("bgikl,bgoik->bgol", sampled, dynamic_weight).reshape(
            batch, self.channels, length,
        )


class FrequencyDynamicConv1D(nn.Module):
    """FDConv: disjoint Fourier weight experts with global and local FBM."""

    def __init__(self, channels: int, kernel_size: int = 31, bands: int = 4) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = _odd(kernel_size)
        self.bands = int(bands)
        if self.bands < 2:
            raise ValueError("FDConv requires at least two Fourier experts")
        self.weight = nn.Parameter(torch.empty(self.channels, self.channels, self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        shape = (self.channels * self.kernel_size, self.channels)
        fy = torch.fft.fftfreq(shape[0])[:, None]
        fx = torch.fft.fftfreq(shape[1])[None, :]
        radius = (fy.square() + fx.square()).sqrt().flatten()
        order = radius.argsort()
        assignment = torch.empty_like(order)
        assignment[order] = torch.arange(order.numel(), device=order.device)
        assignment = torch.div(assignment * self.bands, order.numel(), rounding_mode="floor")
        masks = F.one_hot(assignment.clamp_max(self.bands - 1), self.bands).T.reshape(self.bands, *shape)
        self.register_buffer("frequency_masks", masks.to(torch.bool), persistent=True)
        hidden = max(16, self.channels // 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.channels, hidden, 1), nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Conv1d(hidden, self.channels, 1)
        self.filter_attention = nn.Conv1d(hidden, self.channels, 1)
        self.kernel_attention = nn.Conv1d(hidden, self.bands, 1)
        self.local_modulation = nn.Conv1d(self.channels, self.channels, 3, padding=1, groups=self.channels)
        self.frequency_band_modulation = FrequencySelection1D(self.channels, (2, 4, 8))

    def expert_kernels(self) -> torch.Tensor:
        matrix = self.weight.permute(0, 2, 1).reshape(self.channels * self.kernel_size, self.channels)
        spectrum = torch.fft.fft2(matrix.float(), norm="ortho")
        experts = []
        for mask in self.frequency_masks:
            part = torch.fft.ifft2(spectrum * mask, norm="ortho").real
            experts.append(part.reshape(self.channels, self.kernel_size, self.channels).permute(0, 2, 1))
        return torch.stack(experts).to(self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.attention(x)
        channel_gate = torch.sigmoid(self.channel_attention(context))
        filter_gate = torch.sigmoid(self.filter_attention(context))
        expert_gate = torch.softmax(self.kernel_attention(context).flatten(1), dim=1)
        kernels = torch.einsum("be,eoik->boik", expert_gate, self.expert_kernels())
        kernels = kernels * filter_gate[:, :, None] * channel_gate[:, None, :, :]
        y = _batch_conv1d(x, kernels, self.kernel_size // 2)
        local = 2.0 * torch.sigmoid(self.local_modulation(x))
        return local * self.frequency_band_modulation(y, x)


_UNIREP_BRANCHES = {
    17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
    15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
    13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
    11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
    9: ([5, 5, 3, 3], [1, 2, 3, 4]),
    7: ([5, 3, 3], [1, 2, 3]),
    5: ([3, 3], [1, 2]),
}


class DilatedReparamBlock1D(nn.Module):
    """Official UniRepLKNet dilated branches, including BN fusion."""

    def __init__(self, channels: int, large_kernel: int = 17,
                 branch_kernel: int | None = None,
                 dilations: Sequence[int] | None = None) -> None:
        super().__init__()
        self.channels = int(channels)
        self.large_kernel = _odd(large_kernel)
        if branch_kernel is None and dilations is None:
            if self.large_kernel not in _UNIREP_BRANCHES:
                raise ValueError("UniRepLK kernel must be one of 5,7,9,11,13,15,17")
            kernels, rates = _UNIREP_BRANCHES[self.large_kernel]
        else:
            rates = tuple(int(v) for v in (dilations or (1, 2, 3)))
            kernels = [int(branch_kernel or 3)] * len(rates)
        self.branch_kernels = tuple(kernels)
        self.dilations = tuple(rates)
        self.large = nn.Conv1d(self.channels, self.channels, self.large_kernel,
                               padding=self.large_kernel // 2, groups=self.channels, bias=False)
        self.large_bn = nn.BatchNorm1d(self.channels)
        self.branches = nn.ModuleList()
        self.branch_bns = nn.ModuleList()
        for kernel, dilation in zip(self.branch_kernels, self.dilations):
            effective = dilation * (kernel - 1) + 1
            if effective > self.large_kernel:
                raise ValueError("dilated UniRepLK branch exceeds the large kernel")
            self.branches.append(nn.Conv1d(
                self.channels, self.channels, kernel, padding=effective // 2,
                dilation=dilation, groups=self.channels, bias=False,
            ))
            self.branch_bns.append(nn.BatchNorm1d(self.channels))
        self.reparam: nn.Conv1d | None = None

    def _expand(self, weight: torch.Tensor, dilation: int) -> torch.Tensor:
        expanded = weight.new_zeros(self.channels, 1, self.large_kernel)
        center = self.large_kernel // 2
        tap_center = weight.size(-1) // 2
        for tap in range(weight.size(-1)):
            expanded[..., center + (tap - tap_center) * dilation] = weight[..., tap]
        return expanded

    def equivalent_kernel_bias(self):
        kernel, bias = _fuse_conv_bn_1d(self.large, self.large_bn)
        for dilation, conv, bn in zip(self.dilations, self.branches, self.branch_bns):
            branch_kernel, branch_bias = _fuse_conv_bn_1d(conv, bn)
            kernel = kernel + self._expand(branch_kernel, dilation)
            bias = bias + branch_bias
        return kernel, bias

    def reparameterize(self) -> None:
        if self.reparam is not None:
            return
        kernel, bias = self.equivalent_kernel_bias()
        layer = nn.Conv1d(self.channels, self.channels, self.large_kernel,
                          padding=self.large_kernel // 2, groups=self.channels, bias=True)
        layer = layer.to(device=kernel.device, dtype=kernel.dtype)
        layer.weight.data.copy_(kernel)
        layer.bias.data.copy_(bias)
        self.reparam = layer
        del self.large, self.large_bn, self.branches, self.branch_bns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reparam is not None:
            return self.reparam(x)
        y = self.large_bn(self.large(x))
        return y + sum(bn(conv(x)) for conv, bn in zip(self.branches, self.branch_bns))


class GRN1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(p=2, dim=-1, keepdim=True)
        normalized = norm / (norm.mean(dim=1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * normalized) + self.beta


class UniRepLKNetBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 17, ffn_factor: int = 4,
                 layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = DilatedReparamBlock1D(channels, kernel_size)
        self.norm = nn.BatchNorm1d(channels)
        hidden = max(1, channels // 4)
        self.se_reduce = nn.Conv1d(channels, hidden, 1)
        self.se_expand = nn.Conv1d(hidden, channels, 1)
        ffn_channels = int(channels * ffn_factor)
        self.pwconv1 = nn.Conv1d(channels, ffn_channels, 1)
        self.grn = GRN1D(ffn_channels)
        self.pwconv2 = nn.Conv1d(ffn_channels, channels, 1, bias=False)
        self.output_bn = nn.BatchNorm1d(channels)
        self.gamma = nn.Parameter(torch.full((1, channels, 1), float(layer_scale)))

    def residual_branch(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.dwconv(x))
        se = F.adaptive_avg_pool1d(y, 1)
        y = y * torch.sigmoid(self.se_expand(F.relu(self.se_reduce(se), inplace=True)))
        y = self.output_bn(self.pwconv2(self.grn(F.gelu(self.pwconv1(y)))))
        return self.gamma * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_branch(x)


class ShiftwiseConv1D(nn.Module):
    """ShiftwiseConv implicit large kernel with rep/ghost channel split."""

    def __init__(self, channels: int, big_kernel: int = 25, small_kernel: int = 3,
                 ghost_ratio: float = 0.25, paths: int = 2) -> None:
        super().__init__()
        self.channels = int(channels)
        self.big_kernel = _odd(big_kernel)
        self.small_kernel = _odd(small_kernel)
        self.rep_channels = max(1, min(self.channels, int(round(channels * (1 - ghost_ratio)))))
        self.ghost_channels = self.channels - self.rep_channels
        self.nk = int(math.ceil(self.big_kernel / self.small_kernel))
        self.paths = max(1, int(paths))
        self.extractors = nn.ModuleList([
            nn.Conv1d(self.rep_channels, self.rep_channels * self.nk,
                      self.small_kernel, padding=self.small_kernel // 2,
                      groups=self.rep_channels, bias=False)
            for _ in range(self.paths)
        ])
        self.extract_bns = nn.ModuleList([
            nn.BatchNorm1d(self.rep_channels * self.nk) for _ in range(self.paths)
        ])
        self.fuse = nn.Conv1d(3 * self.rep_channels, self.rep_channels, 1, groups=self.rep_channels)
        self.output_bn = nn.BatchNorm1d(self.rep_channels)
        centers = torch.arange(self.nk) * self.small_kernel
        centers = centers - int(centers.float().mean().round())
        self.register_buffer("shifts", centers, persistent=True)

    @staticmethod
    def _shift(x: torch.Tensor, amount: int) -> torch.Tensor:
        if amount == 0:
            return x
        if amount > 0:
            return F.pad(x[..., :-amount], (amount, 0)) if amount < x.size(-1) else torch.zeros_like(x)
        amount = -amount
        return F.pad(x[..., amount:], (0, amount)) if amount < x.size(-1) else torch.zeros_like(x)

    def _aggregate(self, expanded: torch.Tensor) -> torch.Tensor:
        chunks = expanded.reshape(expanded.size(0), self.rep_channels, self.nk, expanded.size(-1))
        center = sum(self._shift(chunks[:, :, i], int(self.shifts[i])) for i in range(self.nk))
        left = sum(self._shift(chunks[:, :, i], int(self.shifts[i]) - self.small_kernel // 2) for i in range(self.nk))
        right = sum(self._shift(chunks[:, :, i], int(self.shifts[i]) + self.small_kernel // 2) for i in range(self.nk))
        return torch.cat((left, center, right), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep, ghost = x[:, :self.rep_channels], x[:, self.rep_channels:]
        paths = [self._aggregate(bn(conv(rep))) for conv, bn in zip(self.extractors, self.extract_bns)]
        y = self.output_bn(self.fuse(torch.stack(paths).mean(0))) + rep
        return torch.cat((y, ghost), dim=1)


class ReparamLargeKernelConv1D(nn.Module):
    def __init__(self, channels: int, large_kernel: int, small_kernel: int = 5) -> None:
        super().__init__()
        self.large_kernel = _odd(large_kernel)
        self.small_kernel = _odd(small_kernel)
        self.large = nn.Conv1d(channels, channels, self.large_kernel,
                               padding=self.large_kernel // 2, groups=channels, bias=False)
        self.large_bn = nn.BatchNorm1d(channels)
        self.small = nn.Conv1d(channels, channels, self.small_kernel,
                               padding=self.small_kernel // 2, groups=channels, bias=False)
        self.small_bn = nn.BatchNorm1d(channels)
        self.reparam: nn.Conv1d | None = None

    def equivalent_kernel_bias(self):
        kernel, bias = _fuse_conv_bn_1d(self.large, self.large_bn)
        small, small_bias = _fuse_conv_bn_1d(self.small, self.small_bn)
        pad = (self.large_kernel - self.small_kernel) // 2
        return kernel + F.pad(small, (pad, pad)), bias + small_bias

    def reparameterize(self):
        if self.reparam is not None:
            return
        kernel, bias = self.equivalent_kernel_bias()
        layer = nn.Conv1d(kernel.size(0), kernel.size(0), self.large_kernel,
                          padding=self.large_kernel // 2, groups=kernel.size(0), bias=True)
        layer = layer.to(device=kernel.device, dtype=kernel.dtype)
        layer.weight.data.copy_(kernel)
        layer.bias.data.copy_(bias)
        self.reparam = layer
        del self.large, self.large_bn, self.small, self.small_bn

    def forward(self, x):
        if self.reparam is not None:
            return self.reparam(x)
        return self.large_bn(self.large(x)) + self.small_bn(self.small(x))


class ModernTCNBlock1D(nn.Module):
    """ModernTCN block with variable-local and cross-variable grouped FFNs."""

    def __init__(self, channels: int, patch_size: int = 4, kernel_size: int = 51,
                 expansion: int = 2, nvars: int = 2, small_kernel: int = 5) -> None:
        super().__init__()
        del patch_size  # Stage-4 encoder strides already provide patch/downsampling.
        self.nvars = min(int(nvars), int(channels))
        while channels % self.nvars:
            self.nvars -= 1
        self.dmodel = channels // self.nvars
        self.dff = self.dmodel * int(expansion)
        self.dw = ReparamLargeKernelConv1D(channels, kernel_size, small_kernel)
        self.norm = nn.BatchNorm1d(self.dmodel)
        self.ffn1a = nn.Conv1d(channels, self.nvars * self.dff, 1, groups=self.nvars)
        self.ffn1b = nn.Conv1d(self.nvars * self.dff, channels, 1, groups=self.nvars)
        self.ffn2a = nn.Conv1d(channels, self.dmodel * self.nvars * int(expansion), 1, groups=self.dmodel)
        self.ffn2b = nn.Conv1d(self.dmodel * self.nvars * int(expansion), channels, 1, groups=self.dmodel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch, _, length = x.shape
        x = self.dw(x).reshape(batch * self.nvars, self.dmodel, length)
        x = self.norm(x).reshape(batch, self.nvars * self.dmodel, length)
        x = self.ffn1b(F.gelu(self.ffn1a(x)))
        x = x.reshape(batch, self.nvars, self.dmodel, length).permute(0, 2, 1, 3)
        x = x.reshape(batch, self.dmodel * self.nvars, length)
        x = self.ffn2b(F.gelu(self.ffn2a(x)))
        x = x.reshape(batch, self.dmodel, self.nvars, length).permute(0, 2, 1, 3)
        return residual + x.reshape(batch, -1, length)


class ModernTCNIQFrontEnd(nn.Module):
    """ModernTCN stem/block/head with I and Q as the two true variables."""

    def __init__(self, dmodel: int = 8, patch_size: int = 5, kernel_size: int = 51,
                 small_kernel: int = 5, expansion: int = 2,
                 residual_scale: float = 0.05) -> None:
        super().__init__()
        patch_size = _odd(patch_size)
        self.dmodel = int(dmodel)
        # As in ModernTCN, the same patch embedding is applied to every variable.
        self.patch_embed = nn.Conv1d(1, self.dmodel, patch_size, padding=patch_size // 2)
        self.patch_norm = nn.BatchNorm1d(self.dmodel)
        self.block = ModernTCNBlock1D(
            2 * self.dmodel, kernel_size=kernel_size, expansion=expansion,
            nvars=2, small_kernel=small_kernel,
        )
        self.head = nn.Conv1d(self.dmodel, 1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 2:
            raise ValueError("ModernTCN IQ front-end requires exactly I and Q channels")
        batch, _, length = x.shape
        features = self.patch_norm(self.patch_embed(x.reshape(batch * 2, 1, length)))
        features = features.reshape(batch, 2 * self.dmodel, length)
        features = self.block(features).reshape(batch * 2, self.dmodel, length)
        delta = self.head(features).reshape(batch, 2, length)
        return x + self.residual_scale * delta


class MixtureOfReceptiveFields1D(nn.Module):
    """MoRF soft/hard routing with exact per-sample kernel fusion."""

    def __init__(self, channels: int, kernels: Sequence[int] = (3, 9, 21, 41),
                 top_k: int = 2, routing: str = "soft") -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernels = tuple(_odd(v) for v in kernels)
        self.max_kernel = max(self.kernels)
        self.top_k = min(max(1, int(top_k)), len(self.kernels))
        self.routing = str(routing).lower()
        if self.routing not in {"soft", "hard"}:
            raise ValueError("MoRF routing must be soft or hard")
        self.expert_weights = nn.ParameterList([
            nn.Parameter(torch.empty(self.channels, 1, kernel)) for kernel in self.kernels
        ])
        for weight in self.expert_weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        hidden = max(16, self.channels // 4)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.channels, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv1d(hidden, len(self.kernels), 1),
        )

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.router(x).flatten(1)
        if self.routing == "hard" and self.top_k < len(self.kernels):
            keep = torch.zeros_like(logits, dtype=torch.bool)
            keep.scatter_(1, logits.topk(self.top_k, dim=1).indices, True)
            logits = logits.masked_fill(~keep, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=1)

    def padded_experts(self) -> torch.Tensor:
        return torch.stack([
            F.pad(weight, ((self.max_kernel - weight.size(-1)) // 2,) * 2)
            for weight in self.expert_weights
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = self.routing_weights(x)
        kernel = torch.einsum("be,ecik->bcik", route, self.padded_experts())
        batch, channels, length = x.shape
        y = F.conv1d(
            x.reshape(1, batch * channels, length),
            kernel.reshape(batch * channels, 1, self.max_kernel),
            padding=self.max_kernel // 2, groups=batch * channels,
        )
        return y.reshape(batch, channels, length)


class DeformableConvV4OneD(nn.Module):
    """1D DCNv4: group-specific offsets/masks without spatial softmax."""

    def __init__(self, channels: int, points: int = 5, offset_range: float = 1.0,
                 groups: int = 4, center_feature_scale: bool = True) -> None:
        super().__init__()
        self.channels = int(channels)
        self.points = _odd(points)
        self.groups = min(int(groups), self.channels)
        while self.channels % self.groups:
            self.groups -= 1
        self.offset_scale = float(offset_range)
        self.value_proj = nn.Conv1d(self.channels, self.channels, 1)
        self.offset_mask_dw = nn.Conv1d(
            self.channels, self.channels, 3, padding=1, groups=self.channels,
        )
        self.offset_mask = nn.Conv1d(self.channels, self.groups * self.points * 2, 1)
        self.output_proj = nn.Conv1d(self.channels, self.channels, 1)
        self.center_feature_scale = bool(center_feature_scale)
        if self.center_feature_scale:
            self.center_scale = nn.Conv1d(self.channels, self.groups, 1)
            nn.init.zeros_(self.center_scale.weight)
            nn.init.zeros_(self.center_scale.bias)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        base = torch.arange(-(self.points // 2), self.points // 2 + 1, dtype=torch.float)
        self.register_buffer("base_offsets", base, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length = x.shape
        value = self.value_proj(x)
        prediction = self.offset_mask(self.offset_mask_dw(x)).reshape(
            batch, self.groups, self.points, 2, length,
        )
        offsets, masks = prediction[:, :, :, 0], prediction[:, :, :, 1]
        base = torch.arange(length, device=x.device, dtype=x.dtype)[None, None, None]
        positions = base + self.base_offsets[None, None, :, None] + self.offset_scale * offsets
        sampled = _linear_sample_grouped(value, positions, self.groups)
        y = (sampled * masks[:, :, None]).sum(dim=3).reshape_as(value)
        if self.center_feature_scale:
            scale = torch.sigmoid(self.center_scale(x))
            scale = scale[:, :, None].expand(-1, -1, self.channels // self.groups, -1).reshape_as(y)
            y = y * (1 - scale) + value * scale
        return self.output_proj(y)


class Scale(nn.Module):
    def __init__(self, shape: tuple[int, ...], init: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full(shape, float(init)))

    def forward(self, x):
        return x * self.weight


class WTConv1D(nn.Module):
    """Official WTConv1d recursion using fixed Haar analysis/synthesis filters."""

    def __init__(self, channels: int, levels: int = 3, kernel_size: int = 5) -> None:
        super().__init__()
        self.channels = int(channels)
        self.levels = int(levels)
        kernel_size = _odd(kernel_size)
        scale = math.sqrt(0.5)
        dec = torch.tensor([[scale, scale], [scale, -scale]])[:, None].repeat(self.channels, 1, 1)
        rec = torch.tensor([[scale, scale], [scale, -scale]])[:, None].repeat(self.channels, 1, 1)
        self.register_buffer("dec_filter", dec, persistent=True)
        self.register_buffer("rec_filter", rec, persistent=True)
        self.base_conv = nn.Conv1d(self.channels, self.channels, kernel_size,
                                   padding=kernel_size // 2, groups=self.channels, bias=True)
        self.base_scale = Scale((1, self.channels, 1), 1.0)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(2 * self.channels, 2 * self.channels, kernel_size,
                      padding=kernel_size // 2, groups=2 * self.channels, bias=False)
            for _ in range(self.levels)
        ])
        self.wavelet_scales = nn.ModuleList([
            Scale((1, 2 * self.channels, 1), 0.1) for _ in range(self.levels)
        ])

    def _analysis(self, x):
        y = F.conv1d(x, self.dec_filter.to(x.dtype), stride=2, groups=self.channels)
        return y.reshape(x.size(0), self.channels, 2, -1)

    def _synthesis(self, x):
        flat = x.reshape(x.size(0), 2 * self.channels, x.size(-1))
        return F.conv_transpose1d(flat, self.rec_filter.to(x.dtype), stride=2, groups=self.channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lows, highs, shapes = [], [], []
        current = x
        for level in range(self.levels):
            shapes.append(current.shape)
            if current.size(-1) % 2:
                current = F.pad(current, (0, 1))
            transformed = self._analysis(current)
            current = transformed[:, :, 0]
            bands = transformed.reshape(x.size(0), 2 * self.channels, -1)
            bands = self.wavelet_scales[level](self.wavelet_convs[level](bands))
            bands = bands.reshape(x.size(0), self.channels, 2, -1)
            lows.append(bands[:, :, 0])
            highs.append(bands[:, :, 1])
        reconstructed = 0
        for level in range(self.levels - 1, -1, -1):
            low = lows[level] + reconstructed
            reconstructed = self._synthesis(torch.stack((low, highs[level]), dim=2))
            reconstructed = reconstructed[..., :shapes[level][-1]]
        return self.base_scale(self.base_conv(x)) + reconstructed


class DCLSConv1D(nn.Module):
    """Official DCLS Gaussian kernel construction followed by Conv1d."""

    def __init__(self, channels: int, taps: int = 7, max_offset: float = 24.0,
                 version: str = "gauss") -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_count = int(taps)
        self.dilated_kernel_size = _odd(int(round(2 * max_offset + 1)))
        self.version = str(version)
        if self.version not in {"gauss", "v1"}:
            raise ValueError("DCLS version must be gauss or v1")
        self.weight = nn.Parameter(torch.empty(self.channels, 1, self.kernel_count))
        self.P = nn.Parameter(torch.empty(1, self.channels, 1, self.kernel_count))
        self.SIG = nn.Parameter(torch.empty(1, self.channels, 1, self.kernel_count)) if self.version == "gauss" else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.P, 0, 0.5)
        with torch.no_grad():
            self.P.clamp_(-(self.dilated_kernel_size // 2), self.dilated_kernel_size // 2)
        if self.SIG is not None:
            nn.init.constant_(self.SIG, 0.23)

    @property
    def positions(self):
        return self.P

    def clamp_parameters(self):
        with torch.no_grad():
            self.P.clamp_(-(self.dilated_kernel_size // 2), self.dilated_kernel_size // 2)

    def constructed_kernel(self) -> torch.Tensor:
        dtype = self.weight.dtype
        grid = torch.arange(self.dilated_kernel_size, device=self.weight.device, dtype=dtype)
        positions = self.P.to(dtype) + self.dilated_kernel_size // 2
        distance = grid[:, None, None, None] - positions
        if self.version == "gauss":
            sigma = self.SIG.to(dtype).abs() + 0.27
            basis = torch.exp(-0.5 * (distance / sigma).square())
        else:
            basis = (1 - distance.abs()).relu()
        basis = basis / (basis.sum(dim=0, keepdim=True) + 1e-7)
        kernel = (basis * self.weight[None]).sum(-1)
        return kernel.permute(1, 2, 0).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.constructed_kernel().to(x.dtype)
        return F.conv1d(x, kernel, padding=self.dilated_kernel_size // 2, groups=self.channels)


class FeatureResidualAdapter(nn.Module):
    """Non-zero residual insertion; the operator receives gradients on step one."""

    def __init__(self, channels: int, operator: nn.Module, scale_init: float = 0.05) -> None:
        super().__init__()
        self.operator = operator
        self.norm = nn.GroupNorm(1, channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.norm(self.operator(x))


class ParallelFeatureDeltaAdapter(nn.Module):
    """Add only an operator's residual delta to an independent main branch."""

    def __init__(self, channels: int, operator: nn.Module,
                 scale_init: float = 0.05) -> None:
        super().__init__()
        self.operator = operator
        self.norm = nn.GroupNorm(1, channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, source: torch.Tensor, main: torch.Tensor) -> torch.Tensor:
        if source.shape != main.shape:
            raise ValueError(
                "parallel delta source and main branch must have the same shape"
            )
        residual_branch = getattr(self.operator, "residual_branch", None)
        delta = (
            residual_branch(source)
            if residual_branch is not None
            else self.operator(source) - source
        )
        return main + self.residual_scale * self.norm(delta)


class ScaleAwareParallelFusion1D(nn.Module):
    """Fuse FDConv and UniRepLK after matching their per-channel RMS scales."""

    def __init__(self, channels: int, fdconv: nn.Module, unireplk: nn.Module,
                 residual_scale_init: float = 0.05, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = int(channels)
        self.fdconv = fdconv
        self.unireplk = unireplk
        self.eps = float(eps)
        hidden = max(8, self.channels // 4)
        self.router = nn.Sequential(
            nn.Conv1d(3 * self.channels, hidden, 1),
            nn.GELU(),
            nn.Conv1d(hidden, 2 * self.channels, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.branch_scale = nn.Parameter(torch.ones(2, self.channels, 1))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.last_gates: torch.Tensor | None = None

    def _match_input_scale(self, branch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        branch_rms = branch.float().square().mean(-1, keepdim=True).add(self.eps).sqrt()
        input_rms = x.float().square().mean(-1, keepdim=True).add(self.eps).sqrt()
        return (branch.float() * (input_rms / branch_rms)).to(branch.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fd = self._match_input_scale(self.fdconv(x), x)
        uni = self._match_input_scale(self.unireplk(x), x)
        statistics = torch.cat((
            F.adaptive_avg_pool1d(x, 1),
            x.float().square().mean(-1, keepdim=True).sqrt().to(x.dtype),
            F.adaptive_avg_pool1d((fd - uni).abs(), 1),
        ), dim=1)
        gates = self.router(statistics).reshape(x.size(0), 2, self.channels, 1)
        gates = torch.softmax(gates, dim=1)
        self.last_gates = gates.detach()
        fused = (
            gates[:, 0] * self.branch_scale[0] * fd
            + gates[:, 1] * self.branch_scale[1] * uni
        )
        return x + self.residual_scale * fused


class RecentRFInputAdapter(nn.Module):
    """Compatibility adapter; new stages use FeatureResidualAdapter in the encoder."""

    def __init__(self, input_channels: int, hidden_channels: int, operator: nn.Module,
                 residual_scale_init: float = 0.05) -> None:
        super().__init__()
        groups = input_channels if hidden_channels % input_channels == 0 else 1
        self.input_projection = nn.Conv1d(input_channels, hidden_channels, 1, groups=groups)
        self.operator = operator
        self.norm = nn.GroupNorm(1, hidden_channels)
        self.output_projection = nn.Conv1d(hidden_channels, input_channels, 1, groups=groups)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm(self.operator(self.input_projection(x))))
        return x + self.residual_scale * self.output_projection(y)


def build_recent_rf_operator(kind: str, channels: int, cfg: dict) -> nn.Module:
    kind = str(kind).lower()
    if kind == "fadc":
        return FrequencyAdaptiveDilatedConv1D(
            channels, cfg.get("rf_kernel_size", 7), cfg.get("rf_cutoffs", cfg.get("rf_dilations", [2, 4, 8])),
            cfg.get("rf_groups"),
        )
    if kind == "fdconv":
        return FrequencyDynamicConv1D(channels, cfg.get("rf_kernel_size", 31), cfg.get("rf_bands", 4))
    if kind == "unireplk":
        return UniRepLKNetBlock1D(
            channels, cfg.get("rf_large_kernel", 17), cfg.get("rf_ffn_factor", 4),
            cfg.get("rf_layer_scale", 1e-6),
        )
    if kind == "shiftwise":
        return ShiftwiseConv1D(
            channels, cfg.get("rf_big_kernel", 25), cfg.get("rf_small_kernel", 3),
            cfg.get("rf_ghost_ratio", 0.25), cfg.get("rf_paths", 2),
        )
    if kind == "moderntcn":
        return ModernTCNBlock1D(
            channels, cfg.get("rf_patch_size", 4), cfg.get("rf_kernel_size", 51),
            cfg.get("rf_expansion", 2), cfg.get("rf_nvars", 2), cfg.get("rf_small_kernel", 5),
        )
    if kind == "morf":
        return MixtureOfReceptiveFields1D(
            channels, cfg.get("rf_kernels", [3, 9, 21, 41]), cfg.get("rf_top_k", 2),
            cfg.get("rf_routing", "soft"),
        )
    if kind == "dcnv4":
        return DeformableConvV4OneD(
            channels, cfg.get("rf_points", 5), cfg.get("rf_offset_scale", cfg.get("rf_offset_range", 1.0)),
            cfg.get("rf_groups", 4), cfg.get("rf_center_feature_scale", True),
        )
    if kind == "wtconv":
        return WTConv1D(channels, cfg.get("rf_levels", 3), cfg.get("rf_kernel_size", 5))
    if kind == "dcls":
        return DCLSConv1D(
            channels, cfg.get("rf_taps", 7), cfg.get("rf_max_offset", 24.0),
            cfg.get("rf_dcls_version", "gauss"),
        )
    raise ValueError(f"Unsupported recent RF operator: {kind}")


class IQUMamba1DRecentRF(IQUMamba1D):
    """Stage-4 with paper-faithful RF blocks at selected encoder stages."""

    _DEFAULT_STAGES = {
        "fadc": (0, 1), "fdconv": (0,), "unireplk": (0, 1, 2),
        "shiftwise": (0, 1, 2), "moderntcn": (), "morf": (0, 1, 2),
        "dcnv4": (0, 1), "wtconv": (0, 1, 2), "dcls": (0, 1, 2),
    }

    def __init__(self, *args, rf_module_type: str, rf_hidden_channels: int = 16,
                 rf_residual_scale_init: float = 0.05, rf_module_config: dict | None = None,
                 **kwargs) -> None:
        input_channels = int(kwargs.get("input_channels", 2))
        super().__init__(*args, **kwargs)
        config = dict(rf_module_config or {})
        self.rf_module_type = str(rf_module_type).lower()
        stages = tuple(config.get("rf_apply_stages", self._DEFAULT_STAGES[self.rf_module_type]))
        invalid = [stage for stage in stages if stage < 0 or stage >= len(self.encoder.output_channels)]
        if invalid:
            raise ValueError(f"invalid RF encoder stages: {invalid}")
        self.rf_apply_stages = stages
        self.input_rf: nn.Module = nn.Identity()
        if self.rf_module_type == "moderntcn" and bool(config.get("rf_input_iq", True)):
            if input_channels != 2:
                raise ValueError("Stage312 ModernTCN expects two input IQ channels")
            self.input_rf = ModernTCNIQFrontEnd(
                dmodel=int(config.get("rf_dmodel", max(1, int(rf_hidden_channels) // 2))),
                patch_size=int(config.get("rf_patch_size", 5)),
                kernel_size=int(config.get("rf_kernel_size", 51)),
                small_kernel=int(config.get("rf_small_kernel", 5)),
                expansion=int(config.get("rf_expansion", 2)),
                residual_scale=rf_residual_scale_init,
            )
        self.stage_rf = nn.ModuleDict({
            str(stage): FeatureResidualAdapter(
                self.encoder.output_channels[stage],
                build_recent_rf_operator(self.rf_module_type, self.encoder.output_channels[stage], config),
                rf_residual_scale_init,
            ) for stage in stages
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_rf(x)
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage, (conv_stage, mamba) in enumerate(zip(self.encoder.stages, self.encoder.mamba_layers)):
            x = mamba(conv_stage(x))
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](x)
            skips.append(x)
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        names = {f"stage_rf.{stage}.residual_scale" for stage in self.stage_rf}
        if isinstance(self.input_rf, ModernTCNIQFrontEnd):
            names.add("input_rf.residual_scale")
        for stage, adapter in self.stage_rf.items():
            if isinstance(adapter.operator, DCLSConv1D):
                names.add(f"stage_rf.{stage}.operator.P")
                names.add(f"stage_rf.{stage}.operator.SIG")
        return names


class IQUMamba1DFDConvUniRepAblation(IQUMamba1D):
    """Stage317-322 controlled FDConv/UniRepLK placement and Mamba ablations."""

    VARIANTS = {
        "serial", "hierarchical", "parallel",
        "no_mamba", "no_mamba_fdconv", "no_mamba_unireplk",
    }

    def __init__(self, *args, rf_variant: str,
                 rf_residual_scale_init: float = 0.05,
                 rf_module_config: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rf_variant = str(rf_variant).lower()
        if self.rf_variant not in self.VARIANTS:
            raise ValueError(f"Unsupported FDConv/UniRepLK ablation: {self.rf_variant}")
        config = dict(rf_module_config or {})
        self.fdconv_config = {
            "rf_kernel_size": int(config.get("fdconv_kernel_size", 31)),
            "rf_bands": int(config.get("fdconv_bands", 4)),
        }
        self.unireplk_config = {
            "rf_large_kernel": int(config.get("unireplk_large_kernel", 17)),
            "rf_ffn_factor": int(config.get("unireplk_ffn_factor", 4)),
            "rf_layer_scale": float(config.get("unireplk_layer_scale", 1e-6)),
        }
        self.stage_rf = nn.ModuleDict()
        channels = self.encoder.output_channels

        if self.rf_variant == "serial":
            self.stage_rf["0"] = nn.Sequential(
                self._adapter("fdconv", channels[0], rf_residual_scale_init),
                self._adapter("unireplk", channels[0], rf_residual_scale_init),
            )
            self.stage_rf["1"] = self._adapter("unireplk", channels[1], rf_residual_scale_init)
            self.stage_rf["2"] = self._adapter("unireplk", channels[2], rf_residual_scale_init)
        elif self.rf_variant == "hierarchical":
            self.stage_rf["0"] = self._adapter("fdconv", channels[0], rf_residual_scale_init)
            self.stage_rf["1"] = self._adapter("unireplk", channels[1], rf_residual_scale_init)
            self.stage_rf["2"] = self._adapter("unireplk", channels[2], rf_residual_scale_init)
        elif self.rf_variant == "parallel":
            self.stage_rf["0"] = ScaleAwareParallelFusion1D(
                channels[0],
                build_recent_rf_operator("fdconv", channels[0], self.fdconv_config),
                build_recent_rf_operator("unireplk", channels[0], self.unireplk_config),
                float(config.get("parallel_residual_scale_init", rf_residual_scale_init)),
                float(config.get("parallel_scale_eps", 1e-6)),
            )
            self.stage_rf["1"] = self._adapter("unireplk", channels[1], rf_residual_scale_init)
            self.stage_rf["2"] = self._adapter("unireplk", channels[2], rf_residual_scale_init)
        elif self.rf_variant == "no_mamba_fdconv":
            self.stage_rf["0"] = self._adapter("fdconv", channels[0], rf_residual_scale_init)
        elif self.rf_variant == "no_mamba_unireplk":
            for stage in (0, 1, 2):
                self.stage_rf[str(stage)] = self._adapter(
                    "unireplk", channels[stage], rf_residual_scale_init,
                )

        self.mamba_removed = self.rf_variant.startswith("no_mamba")
        if self.mamba_removed:
            self.encoder.mamba_layers = nn.ModuleList([
                nn.Identity() for _ in self.encoder.mamba_layers
            ])

    def _adapter(self, kind: str, channels: int, scale: float) -> FeatureResidualAdapter:
        config = self.fdconv_config if kind == "fdconv" else self.unireplk_config
        return FeatureResidualAdapter(
            channels, build_recent_rf_operator(kind, channels, config), scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage, (conv_stage, mamba) in enumerate(zip(
            self.encoder.stages, self.encoder.mamba_layers,
        )):
            x = mamba(conv_stage(x))
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](x)
            skips.append(x)
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        names = set()
        for module_name, module in self.stage_rf.named_modules():
            if isinstance(module, FeatureResidualAdapter):
                names.add(f"stage_rf.{module_name}.residual_scale")
            elif isinstance(module, ScaleAwareParallelFusion1D):
                names.add(f"stage_rf.{module_name}.residual_scale")
                names.add(f"stage_rf.{module_name}.branch_scale")
        return names
