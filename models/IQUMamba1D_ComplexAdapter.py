"""Stage-4 IQUMamba with lightweight complex-aware boundary adapters.

This variant keeps the original IQUMamba encoder/decoder unchanged.  The only
added operations are optional residual adapters at the raw I/Q input and final
I/Q source output.  Each adapter uses Cauchy-Riemann tied convolutions and an
amplitude gate, avoiding real/imag separate normalization or activation.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D


class ComplexTiedConv1d(nn.Module):
    """Complex 1D convolution represented by tied real-valued convolutions."""

    def __init__(
        self,
        in_complex_channels: int,
        out_complex_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if padding is None:
            padding = kernel_size // 2
        self.real = nn.Conv1d(
            in_complex_channels,
            out_complex_channels,
            kernel_size=kernel_size,
            padding=int(padding),
            dilation=int(dilation),
            groups=int(groups),
            bias=False,
        )
        self.imag = nn.Conv1d(
            in_complex_channels,
            out_complex_channels,
            kernel_size=kernel_size,
            padding=int(padding),
            dilation=int(dilation),
            groups=int(groups),
            bias=False,
        )
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_complex_channels))
            self.bias_imag = nn.Parameter(torch.zeros(out_complex_channels))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.imag(real) + self.real(imag)
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1)
        return out_real, out_imag


class ModulatedComplexResidualAdapter1D(nn.Module):
    """Residual complex adapter with a rotation-friendly amplitude gate."""

    def __init__(
        self,
        num_complex_channels: int,
        hidden_channels: int = 8,
        kernel_size: int = 5,
        scale_init: float = 0.01,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        hidden_channels = max(1, int(hidden_channels))
        self.in_proj = ComplexTiedConv1d(
            num_complex_channels,
            hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.out_proj = ComplexTiedConv1d(
            hidden_channels,
            num_complex_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(num_complex_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    @staticmethod
    def _split_pairs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.size(1) % 2 != 0:
            raise ValueError(f"Expected paired IQ tensor (B, 2C, L), got {tuple(x.shape)}")
        return x[:, 0::2, :], x[:, 1::2, :]

    @staticmethod
    def _merge_pairs(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        return torch.stack([real, imag], dim=2).flatten(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = self._split_pairs(x)
        hidden_real, hidden_imag = self.in_proj(real, imag)
        amp = torch.sqrt(real.square() + imag.square() + 1e-8)
        gate = self.gate(amp)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = self._merge_pairs(delta_real, delta_imag)
        return x + self.scale * delta


class IQUMamba1D_ComplexAdapter(nn.Module):
    """Original IQUMamba wrapped with local complex-aware input/output adapters."""

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
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        complex_adapter_hidden_channels: int = 8,
        complex_adapter_kernel_size: int = 5,
        complex_adapter_scale_init: float = 0.01,
        complex_adapter_use_input: bool = True,
        complex_adapter_use_output: bool = True,
        complex_adapter_zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels % 2 != 0:
            raise ValueError(f"input_channels must be paired I/Q channels, got {input_channels}")
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be paired I/Q source channels, got {num_classes}")
        self.backbone = IQUMamba1D(
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
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        adapter_kwargs = {
            "hidden_channels": complex_adapter_hidden_channels,
            "kernel_size": complex_adapter_kernel_size,
            "scale_init": complex_adapter_scale_init,
            "zero_init": complex_adapter_zero_init,
        }
        self.input_adapter = (
            ModulatedComplexResidualAdapter1D(input_channels // 2, **adapter_kwargs)
            if complex_adapter_use_input
            else nn.Identity()
        )
        self.output_adapter = (
            ModulatedComplexResidualAdapter1D(num_classes // 2, **adapter_kwargs)
            if complex_adapter_use_output
            else nn.Identity()
        )

    def _adapt_output(self, output: torch.Tensor) -> torch.Tensor:
        return self.output_adapter(output)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.input_adapter(x)
        outputs = self.backbone(x)
        if isinstance(outputs, (list, tuple)):
            return [self._adapt_output(out) for out in outputs]
        return self._adapt_output(outputs)
