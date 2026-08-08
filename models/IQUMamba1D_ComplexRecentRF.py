"""Complex-valued IQ adaptations of FDConv and UniRepLKNet."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_RecentRFModules import _UNIREP_BRANCHES, _odd


class ComplexConv1d(nn.Module):
    """Complex convolution represented by two real kernels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _odd(kernel_size)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        shape = (self.out_channels, self.in_channels // self.groups, self.kernel_size)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(self.out_channels))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    @property
    def complex_weight(self) -> torch.Tensor:
        return torch.complex(self.weight_real.float(), self.weight_imag.float())

    @property
    def complex_bias(self) -> torch.Tensor | None:
        if self.bias_real is None:
            return None
        return torch.complex(self.bias_real.float(), self.bias_imag.float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(torch.complex64)
        real = (
            F.conv1d(z.real, self.weight_real.float(), self.bias_real, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
            - F.conv1d(z.imag, self.weight_imag.float(), None, padding=self.padding,
                       dilation=self.dilation, groups=self.groups)
        )
        imag = (
            F.conv1d(z.real, self.weight_imag.float(), self.bias_imag, padding=self.padding,
                     dilation=self.dilation, groups=self.groups)
            + F.conv1d(z.imag, self.weight_real.float(), None, padding=self.padding,
                       dilation=self.dilation, groups=self.groups)
        )
        return torch.complex(real, imag)


class ComplexBatchNorm1d(nn.Module):
    """Complex-affine BN with one rotation-neutral variance per channel."""

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.channels = int(channels)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight_real = nn.Parameter(torch.ones(self.channels))
        self.weight_imag = nn.Parameter(torch.zeros(self.channels))
        self.bias_real = nn.Parameter(torch.zeros(self.channels))
        self.bias_imag = nn.Parameter(torch.zeros(self.channels))
        self.register_buffer("running_mean_real", torch.zeros(self.channels))
        self.register_buffer("running_mean_imag", torch.zeros(self.channels))
        self.register_buffer("running_var", torch.ones(self.channels))

    @property
    def complex_weight(self):
        return torch.complex(self.weight_real.float(), self.weight_imag.float())

    @property
    def complex_bias(self):
        return torch.complex(self.bias_real.float(), self.bias_imag.float())

    @property
    def running_mean(self):
        return torch.complex(self.running_mean_real, self.running_mean_imag)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(torch.complex64)
        if self.training:
            mean = z.mean(dim=(0, 2))
            centered = z - mean[None, :, None]
            variance = centered.abs().square().mean(dim=(0, 2))
            with torch.no_grad():
                self.running_mean_real.lerp_(mean.real, self.momentum)
                self.running_mean_imag.lerp_(mean.imag, self.momentum)
                self.running_var.lerp_(variance, self.momentum)
        else:
            mean = self.running_mean
            variance = self.running_var
            centered = z - mean[None, :, None]
        normalized = centered / (variance + self.eps).sqrt()[None, :, None]
        return (
            normalized * self.complex_weight[None, :, None]
            + self.complex_bias[None, :, None]
        )


def _fuse_complex_conv_bn(conv: ComplexConv1d, norm: ComplexBatchNorm1d):
    scale = norm.complex_weight / (norm.running_var + norm.eps).sqrt()
    kernel = conv.complex_weight * scale[:, None, None]
    conv_bias = conv.complex_bias
    if conv_bias is None:
        conv_bias = torch.zeros(conv.out_channels, device=kernel.device, dtype=kernel.dtype)
    bias = norm.complex_bias + (conv_bias - norm.running_mean) * scale
    return kernel, bias


def _batch_complex_conv1d(z: torch.Tensor, kernel: torch.Tensor, padding: int) -> torch.Tensor:
    batch, in_channels, length = z.shape
    out_channels = kernel.size(1)

    def apply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        y = F.conv1d(
            x.reshape(1, batch * in_channels, length),
            weight.reshape(batch * out_channels, in_channels, weight.size(-1)),
            padding=padding, groups=batch,
        )
        return y.reshape(batch, out_channels, y.size(-1))

    z = z.to(torch.complex64)
    kernel = kernel.to(torch.complex64)
    return torch.complex(
        apply(z.real, kernel.real) - apply(z.imag, kernel.imag),
        apply(z.real, kernel.imag) + apply(z.imag, kernel.real),
    )


class ComplexRMSNorm1D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.eps = float(eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        rms = z.abs().square().mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt()
        return z / rms * self.scale


def complex_gelu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    magnitude = z.abs()
    return z * (F.gelu(magnitude) / magnitude.clamp_min(eps))


class ComplexFrequencySelection1D(nn.Module):
    def __init__(self, channels: int, cutoffs=(2, 4, 8)) -> None:
        super().__init__()
        self.cutoffs = tuple(int(v) for v in cutoffs)
        self.band_weights = nn.ModuleList([
            nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
            for _ in range(len(self.cutoffs) + 1)
        ])
        for layer in self.band_weights:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def decompose(self, z: torch.Tensor) -> list[torch.Tensor]:
        length = z.size(-1)
        spectrum = torch.fft.fftshift(torch.fft.fft(z.to(torch.complex64), norm="ortho"), dim=-1)
        previous = z.to(torch.complex64)
        bands = []
        center = length // 2
        for cutoff in self.cutoffs:
            half_width = max(1, int(round(length / (2 * cutoff))))
            mask = torch.zeros(length, device=z.device)
            mask[max(0, center - half_width):min(length, center + half_width)] = 1
            low = torch.fft.ifft(
                torch.fft.ifftshift(spectrum * mask[None, None], dim=-1), norm="ortho",
            )
            bands.append(previous - low)
            previous = low
        bands.append(previous)
        return bands

    def forward(self, z: torch.Tensor, attention: torch.Tensor | None = None) -> torch.Tensor:
        attention = z.abs() if attention is None else attention
        selected = [
            band * (2 * torch.sigmoid(predictor(attention)))
            for band, predictor in zip(self.decompose(z), self.band_weights)
        ]
        return torch.stack(selected).sum(0)


class ComplexFrequencyDynamicConv1D(nn.Module):
    """Complex FDConv with real magnitude routing and complex weight experts."""

    def __init__(self, channels: int, kernel_size: int = 31, bands: int = 4) -> None:
        super().__init__()
        self.channels = int(channels)
        self.kernel_size = _odd(kernel_size)
        self.bands = int(bands)
        shape = (self.channels, self.channels, self.kernel_size)
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

        matrix_shape = (self.channels * self.kernel_size, self.channels)
        fy = torch.fft.fftfreq(matrix_shape[0])[:, None]
        fx = torch.fft.fftfreq(matrix_shape[1])[None, :]
        radius = (fy.square() + fx.square()).sqrt().flatten()
        order = radius.argsort()
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(order.numel())
        assignment = torch.div(ranks * self.bands, order.numel(), rounding_mode="floor")
        masks = F.one_hot(assignment.clamp_max(self.bands - 1), self.bands)
        self.register_buffer(
            "frequency_masks", masks.T.reshape(self.bands, *matrix_shape).to(torch.bool),
            persistent=True,
        )
        hidden = max(16, self.channels // 4)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(self.channels, hidden, 1), nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Conv1d(hidden, self.channels, 1)
        self.filter_attention = nn.Conv1d(hidden, self.channels, 1)
        self.kernel_attention = nn.Conv1d(hidden, self.bands, 1)
        self.local_modulation = nn.Conv1d(
            self.channels, self.channels, 3, padding=1, groups=self.channels,
        )
        self.band_modulation = ComplexFrequencySelection1D(self.channels)

    @property
    def complex_weight(self):
        return torch.complex(self.weight_real.float(), self.weight_imag.float())

    def expert_kernels(self) -> torch.Tensor:
        matrix = self.complex_weight.permute(0, 2, 1).reshape(
            self.channels * self.kernel_size, self.channels,
        )
        spectrum = torch.fft.fft2(matrix, norm="ortho")
        experts = []
        for mask in self.frequency_masks:
            part = torch.fft.ifft2(spectrum * mask, norm="ortho")
            experts.append(part.reshape(
                self.channels, self.kernel_size, self.channels,
            ).permute(0, 2, 1))
        return torch.stack(experts)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        magnitude = z.abs()
        context = self.context(magnitude)
        channel_gate = torch.sigmoid(self.channel_attention(context))
        filter_gate = torch.sigmoid(self.filter_attention(context))
        expert_gate = torch.softmax(self.kernel_attention(context).flatten(1), dim=1)
        kernel = torch.einsum("be,eoik->boik", expert_gate.to(torch.complex64), self.expert_kernels())
        kernel = kernel * filter_gate[:, :, None] * channel_gate[:, None, :, :]
        y = _batch_complex_conv1d(z, kernel, self.kernel_size // 2)
        local = 2 * torch.sigmoid(self.local_modulation(magnitude))
        return local * self.band_modulation(y, magnitude)


class ComplexDilatedReparamBlock1D(nn.Module):
    """Complex counterpart of the official kernel-17 UniRepLK reparam block."""

    def __init__(self, channels: int, large_kernel: int = 17) -> None:
        super().__init__()
        self.channels = int(channels)
        self.large_kernel = _odd(large_kernel)
        if self.large_kernel not in _UNIREP_BRANCHES:
            raise ValueError("Complex UniRepLK kernel must be one of 5,7,9,11,13,15,17")
        kernels, rates = _UNIREP_BRANCHES[self.large_kernel]
        self.branch_kernels = tuple(kernels)
        self.dilations = tuple(rates)
        self.large = ComplexConv1d(
            self.channels, self.channels, self.large_kernel,
            padding=self.large_kernel // 2, groups=self.channels,
        )
        self.large_bn = ComplexBatchNorm1d(self.channels)
        self.branches = nn.ModuleList()
        self.branch_bns = nn.ModuleList()
        for kernel, dilation in zip(self.branch_kernels, self.dilations):
            effective = dilation * (kernel - 1) + 1
            self.branches.append(ComplexConv1d(
                self.channels, self.channels, kernel, padding=effective // 2,
                dilation=dilation, groups=self.channels,
            ))
            self.branch_bns.append(ComplexBatchNorm1d(self.channels))
        self.reparam: ComplexConv1d | None = None

    def _expand(self, weight: torch.Tensor, dilation: int) -> torch.Tensor:
        expanded = weight.new_zeros(self.channels, 1, self.large_kernel)
        center = self.large_kernel // 2
        tap_center = weight.size(-1) // 2
        for tap in range(weight.size(-1)):
            expanded[..., center + (tap - tap_center) * dilation] = weight[..., tap]
        return expanded

    def equivalent_kernel_bias(self):
        kernel, bias = _fuse_complex_conv_bn(self.large, self.large_bn)
        for dilation, conv, bn in zip(self.dilations, self.branches, self.branch_bns):
            branch_kernel, branch_bias = _fuse_complex_conv_bn(conv, bn)
            kernel = kernel + self._expand(branch_kernel, dilation)
            bias = bias + branch_bias
        return kernel, bias

    def reparameterize(self):
        if self.reparam is not None:
            return
        kernel, bias = self.equivalent_kernel_bias()
        layer = ComplexConv1d(
            self.channels, self.channels, self.large_kernel,
            padding=self.large_kernel // 2, groups=self.channels, bias=True,
        ).to(kernel.device)
        layer.weight_real.data.copy_(kernel.real)
        layer.weight_imag.data.copy_(kernel.imag)
        layer.bias_real.data.copy_(bias.real)
        layer.bias_imag.data.copy_(bias.imag)
        self.reparam = layer
        del self.large, self.large_bn, self.branches, self.branch_bns

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.reparam is not None:
            return self.reparam(z)
        y = self.large_bn(self.large(z))
        return y + sum(bn(conv(z)) for conv, bn in zip(self.branches, self.branch_bns))


class ComplexGRN1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        response = z.abs().square().mean(-1, keepdim=True).sqrt()
        response = response / (response.mean(dim=1, keepdim=True) + 1e-6)
        return z + self.gamma * z * response


class ComplexUniRepLKNetBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 17, ffn_factor: int = 4,
                 layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.dwconv = ComplexDilatedReparamBlock1D(channels, kernel_size)
        self.norm = ComplexBatchNorm1d(channels)
        se_hidden = max(1, channels // 4)
        self.se_reduce = nn.Conv1d(channels, se_hidden, 1)
        self.se_expand = nn.Conv1d(se_hidden, channels, 1)
        ffn_channels = channels * int(ffn_factor)
        self.pwconv1 = ComplexConv1d(channels, ffn_channels, 1)
        self.grn = ComplexGRN1D(ffn_channels)
        self.pwconv2 = ComplexConv1d(ffn_channels, channels, 1)
        self.output_norm = ComplexBatchNorm1d(channels)
        self.gamma = nn.Parameter(torch.full((1, channels, 1), float(layer_scale)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.dwconv(z))
        se = F.adaptive_avg_pool1d(y.abs(), 1)
        gate = torch.sigmoid(self.se_expand(F.relu(self.se_reduce(se), inplace=True)))
        y = y * gate
        y = self.output_norm(self.pwconv2(self.grn(complex_gelu(self.pwconv1(y)))))
        return z + self.gamma * y


class ComplexRecentRFIQAdapter(nn.Module):
    def __init__(self, hidden_channels: int, operator: nn.Module,
                 residual_scale_init: float = 0.05) -> None:
        super().__init__()
        self.input_projection = ComplexConv1d(1, hidden_channels, 1)
        self.operator = operator
        self.norm = ComplexRMSNorm1D(hidden_channels)
        self.output_projection = ComplexConv1d(hidden_channels, 1, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) != 2:
            raise ValueError("Complex RF adapter requires [I,Q] input channels")
        z = torch.complex(x[:, 0].float(), x[:, 1].float())[:, None]
        delta = self.output_projection(complex_gelu(self.norm(
            self.operator(self.input_projection(z))
        )))[:, 0]
        delta_iq = torch.stack((delta.real, delta.imag), dim=1).to(x.dtype)
        return x + self.residual_scale.to(x.dtype) * delta_iq


class IQUMamba1DComplexRecentRF(IQUMamba1D):
    """Stage323/324: phase-aware complex RF front-end plus original Stage4."""

    def __init__(self, *args, complex_rf_type: str, complex_hidden_channels: int = 8,
                 complex_residual_scale_init: float = 0.05,
                 complex_rf_config: dict | None = None, **kwargs) -> None:
        input_channels = int(kwargs.get("input_channels", 2))
        if input_channels != 2:
            raise ValueError("Complex FDConv/UniRepLK stages require two IQ channels")
        super().__init__(*args, **kwargs)
        config = dict(complex_rf_config or {})
        self.complex_rf_type = str(complex_rf_type).lower()
        hidden = int(complex_hidden_channels)
        if self.complex_rf_type == "fdconv":
            operator = ComplexFrequencyDynamicConv1D(
                hidden, int(config.get("complex_kernel_size", 31)),
                int(config.get("complex_bands", 4)),
            )
        elif self.complex_rf_type == "unireplk":
            operator = ComplexUniRepLKNetBlock1D(
                hidden, int(config.get("complex_large_kernel", 17)),
                int(config.get("complex_ffn_factor", 4)),
                float(config.get("complex_layer_scale", 1e-6)),
            )
        else:
            raise ValueError(f"Unsupported complex RF type: {self.complex_rf_type}")
        self.complex_rf = ComplexRecentRFIQAdapter(
            hidden, operator, complex_residual_scale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.complex_rf(x))

    def no_weight_decay(self) -> set[str]:
        return {"complex_rf.residual_scale"}
