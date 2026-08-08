"""Compute-conscious complex WaveNet ablations based on Stage 273.

These models intentionally exclude Mamba, physical-token routing, parallel
dilation banks, and auxiliary losses.  Their only question is whether strict
complex I/Q structure is a useful inductive bias for the ordinary 20-block
WaveNet.

Channel layout inside complex sections is ``[all real, all imaginary]``.
The public output contract remains interleaved by source:
``[S1-I, S1-Q, S2-I, S2-Q, ...]``.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icassp_baseline_wavenet import WaveNetResidualBlock, _kaiming_conv1d


def _split_complex(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 3 or x.size(1) % 2:
        raise ValueError(
            "Complex features must have shape (B, 2*C, L), "
            f"got {tuple(x.shape)}"
        )
    return torch.chunk(x, 2, dim=1)


def _merge_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    return torch.cat((real, imag), dim=1)


def _interleave_source_iq(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Convert separate K-source real/imag tensors to [S1-I,S1-Q,...]."""

    if real.shape != imag.shape:
        raise ValueError("real and imaginary outputs must have identical shapes")
    batch, sources, length = real.shape
    return torch.stack((real, imag), dim=2).reshape(batch, 2 * sources, length)


class ComplexConv1d(nn.Module):
    """Bias-free strict complex convolution executed as one fused real conv.

    The constrained real kernel is assembled as

    ``[[A, -B], [B, A]]``

    for ``W = A + jB``.  This launches one convolution with the same stored
    input/output tensor widths as the real WaveNet operation.  It avoids the
    severe runtime overhead of issuing three or four smaller convolution
    kernels while retaining half as many learned convolution parameters.
    """

    def __init__(
        self,
        in_complex_channels: int,
        out_complex_channels: int,
        *,
        kernel_size: int = 1,
        dilation: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        self.in_complex_channels = int(in_complex_channels)
        self.out_complex_channels = int(out_complex_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.padding = (
            self.dilation * (self.kernel_size - 1) // 2
            if padding is None
            else int(padding)
        )
        if self.in_complex_channels < 1 or self.out_complex_channels < 1:
            raise ValueError("complex channel counts must be positive")
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be positive")

        shape = (
            self.out_complex_channels,
            self.in_complex_channels,
            self.kernel_size,
        )
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        nn.init.kaiming_normal_(self.weight_real)
        nn.init.kaiming_normal_(self.weight_imag)
        with torch.no_grad():
            # Two independent components together should have one real-layer
            # scale, rather than doubling the initial complex variance.
            self.weight_real.mul_(1.0 / math.sqrt(2.0))
            self.weight_imag.mul_(1.0 / math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _split_complex(x)  # Validate paired channel layout.
        top = torch.cat((self.weight_real, -self.weight_imag), dim=1)
        bottom = torch.cat((self.weight_imag, self.weight_real), dim=1)
        fused_weight = torch.cat((top, bottom), dim=0)
        return F.conv1d(
            x,
            fused_weight,
            stride=1,
            padding=self.padding,
            dilation=self.dilation,
        )

    def zero_weights(self) -> None:
        nn.init.zeros_(self.weight_real)
        nn.init.zeros_(self.weight_imag)


class ComplexRMSNorm1d(nn.Module):
    """Per-time complex RMS normalization with real positive channel scale."""

    def __init__(self, complex_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.complex_channels = int(complex_channels)
        self.eps = float(eps)
        self.log_scale = nn.Parameter(torch.zeros(self.complex_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        rms = (
            real.float().square().add(imag.float().square())
            .mean(dim=1, keepdim=True)
            .add(self.eps)
            .sqrt()
            .to(x.dtype)
        )
        scale = self.log_scale.exp().to(x.dtype).view(1, -1, 1)
        return _merge_complex(real / rms * scale, imag / rms * scale)


class ComplexModReLU(nn.Module):
    """Phase-preserving activation that learns only a magnitude bias."""

    def __init__(self, complex_channels: int, bias_init: float = 0.0) -> None:
        super().__init__()
        self.complex_channels = int(complex_channels)
        self.bias = nn.Parameter(
            torch.full((self.complex_channels,), float(bias_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = _split_complex(x)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        activated = F.relu(magnitude + self.bias.view(1, -1, 1))
        multiplier = activated / magnitude
        return _merge_complex(real * multiplier, imag * multiplier)


def _complex_mod_squash(
    real: torch.Tensor,
    imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cheap bounded phase-preserving activation ``z/sqrt(1+|z|^2)``."""

    multiplier = torch.rsqrt(1.0 + real.square() + imag.square())
    return real * multiplier, imag * multiplier


class ComplexWaveNetResidualBlock(nn.Module):
    """Rotation-equivariant gated residual/skip block.

    A strict complex convolution produces a complex filter and a complex gate
    feature.  The gate itself is real and depends only on gate magnitude, so a
    global phase rotation cannot change the gating decision.
    """

    def __init__(
        self,
        complex_channels: int,
        dilation: int,
        *,
        use_norm: bool = False,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.complex_channels = int(complex_channels)
        self.norm = (
            ComplexRMSNorm1d(self.complex_channels, eps=norm_eps)
            if use_norm
            else nn.Identity()
        )
        self.dilated_conv = ComplexConv1d(
            self.complex_channels,
            2 * self.complex_channels,
            kernel_size=3,
            dilation=int(dilation),
            padding=int(dilation),
        )
        self.output_projection = ComplexConv1d(
            self.complex_channels,
            2 * self.complex_channels,
            kernel_size=1,
        )
        self.gate_log_scale = nn.Parameter(torch.ones(self.complex_channels))
        self.gate_bias = nn.Parameter(torch.zeros(self.complex_channels))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_real, hidden_imag = _split_complex(
            self.dilated_conv(self.norm(x))
        )
        filter_real, gate_real = torch.chunk(hidden_real, 2, dim=1)
        filter_imag, gate_imag = torch.chunk(hidden_imag, 2, dim=1)
        filter_real, filter_imag = _complex_mod_squash(
            filter_real, filter_imag
        )
        gate_power = gate_real.square() + gate_imag.square()
        gate_logits = (
            self.gate_log_scale.view(1, -1, 1)
            * (gate_power - 1.0)
            + self.gate_bias.view(1, -1, 1)
        )
        gate = torch.sigmoid(gate_logits)
        gated = _merge_complex(gate * filter_real, gate * filter_imag)

        projected_real, projected_imag = _split_complex(
            self.output_projection(gated)
        )
        residual_real, skip_real = torch.chunk(projected_real, 2, dim=1)
        residual_imag, skip_imag = torch.chunk(projected_imag, 2, dim=1)
        residual = _merge_complex(residual_real, residual_imag)
        skip = _merge_complex(skip_real, skip_imag)
        return (x + residual) / math.sqrt(2.0), skip


class ConjugateLeakageAdapter(nn.Module):
    """Small bounded input adapter for I/Q image leakage.

    ``z' = z + c z*`` with ``|Re(c)|, |Im(c)| <= max_coefficient``.
    It starts as the exact identity and is kept outside the strict complex
    backbone so hardware correction and phase-equivariant modeling remain
    independently testable.
    """

    def __init__(self, max_coefficient: float = 0.15) -> None:
        super().__init__()
        if float(max_coefficient) <= 0:
            raise ValueError("max_coefficient must be positive")
        self.max_coefficient = float(max_coefficient)
        self.raw_real = nn.Parameter(torch.zeros(()))
        self.raw_imag = nn.Parameter(torch.zeros(()))

    def coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.max_coefficient * torch.tanh(self.raw_real),
            self.max_coefficient * torch.tanh(self.raw_imag),
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(
                "ConjugateLeakageAdapter expects (B, 2, L), "
                f"got {tuple(mixture.shape)}"
            )
        real, imag = mixture[:, 0:1], mixture[:, 1:2]
        coeff_real, coeff_imag = self.coefficients()
        delta_real = coeff_real * real + coeff_imag * imag
        delta_imag = coeff_imag * real - coeff_real * imag
        return torch.cat((real + delta_real, imag + delta_imag), dim=1)

    def coefficient_magnitude(self) -> torch.Tensor:
        real, imag = self.coefficients()
        return torch.sqrt(real.square() + imag.square())


class ICASPComplexWaveNet(nn.Module):
    """Stage-273 topology with a configurable complex prefix/full backbone."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 128,
        residual_layers: int = 20,
        dilation_cycle_length: int = 10,
        *,
        complex_layers: int = 20,
        strict_complex_output: bool = False,
        use_conjugate_adapter: bool = False,
        conjugate_adapter_max: float = 0.15,
        complex_norm_enable: bool = False,
        complex_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("ICASPComplexWaveNet expects one complex I/Q mixture")
        if int(num_classes) % 2:
            raise ValueError("num_classes must contain complete I/Q source pairs")
        if int(residual_channels) % 2:
            raise ValueError("residual_channels must be even")
        self.num_layers = int(residual_layers)
        self.complex_layers = int(complex_layers)
        self.strict_complex_output = bool(strict_complex_output)
        self.complex_channels = int(residual_channels) // 2
        if not 0 <= self.complex_layers <= self.num_layers:
            raise ValueError("complex_layers must be in [0, residual_layers]")
        if self.strict_complex_output and self.complex_layers != self.num_layers:
            raise ValueError(
                "strict_complex_output requires every residual block to be complex"
            )

        self.input_adapter = (
            ConjugateLeakageAdapter(conjugate_adapter_max)
            if use_conjugate_adapter
            else nn.Identity()
        )
        self.input_projection = ComplexConv1d(
            1, self.complex_channels, kernel_size=1
        )
        self.input_activation = ComplexModReLU(self.complex_channels)
        blocks = []
        for index in range(self.num_layers):
            dilation = 2 ** (index % int(dilation_cycle_length))
            if index < self.complex_layers:
                block = ComplexWaveNetResidualBlock(
                    self.complex_channels,
                    dilation,
                    use_norm=complex_norm_enable,
                    norm_eps=complex_norm_eps,
                )
            else:
                block = WaveNetResidualBlock(int(residual_channels), dilation)
            blocks.append(block)
        self.residual_blocks = nn.ModuleList(blocks)

        if self.strict_complex_output:
            self.skip_norm = (
                ComplexRMSNorm1d(
                    self.complex_channels, eps=complex_norm_eps
                )
                if complex_norm_enable
                else nn.Identity()
            )
            self.skip_projection = ComplexConv1d(
                self.complex_channels, self.complex_channels, kernel_size=1
            )
            self.skip_activation = ComplexModReLU(self.complex_channels)
            self.output_projection = ComplexConv1d(
                self.complex_channels, int(num_classes) // 2, kernel_size=1
            )
            self.output_projection.zero_weights()
        else:
            self.skip_norm = nn.Identity()
            self.skip_projection = _kaiming_conv1d(
                int(residual_channels), int(residual_channels), kernel_size=1
            )
            self.skip_activation = nn.ReLU()
            self.output_projection = _kaiming_conv1d(
                int(residual_channels), int(num_classes), kernel_size=1
            )
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        mixture = self.input_adapter(mixture)
        # ComplexConv1d expects [all real, all imag]. One input complex channel
        # already has the public [I,Q] layout, so no rearrangement is needed.
        x = self.input_activation(self.input_projection(mixture))
        skip_sum = None
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        x = skip_sum / math.sqrt(self.num_layers)
        x = self.skip_activation(self.skip_projection(self.skip_norm(x)))
        output = self.output_projection(x)
        if not self.strict_complex_output:
            return output
        output_real, output_imag = _split_complex(output)
        return _interleave_source_iq(output_real, output_imag)

    def diagnostics(self) -> Dict[str, torch.Tensor | float]:
        values: Dict[str, torch.Tensor | float] = {
            "complex_layer_fraction": self.complex_layers / self.num_layers,
        }
        if isinstance(self.input_adapter, ConjugateLeakageAdapter):
            values["conjugate_adapter_magnitude"] = (
                self.input_adapter.coefficient_magnitude().detach()
            )
        return values

    def no_weight_decay(self) -> set[str]:
        names = set()
        for module_name, module in self.named_modules():
            if isinstance(module, ComplexRMSNorm1d):
                names.add(f"{module_name}.log_scale")
            if isinstance(module, ComplexWaveNetResidualBlock):
                names.add(f"{module_name}.gate_log_scale")
                names.add(f"{module_name}.gate_bias")
            if isinstance(module, ComplexModReLU):
                names.add(f"{module_name}.bias")
        if isinstance(self.input_adapter, ConjugateLeakageAdapter):
            names.add("input_adapter.raw_real")
            names.add("input_adapter.raw_imag")
        return names
