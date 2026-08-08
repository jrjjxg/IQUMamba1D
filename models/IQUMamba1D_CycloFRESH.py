"""Cyclostationary/FRESH-inspired input adapter for stage-4 IQUMamba.

The adapter builds frequency-shifted views of the raw complex mixture, filters
them with a lightweight complex-tied FIR stack, and fuses the result back into
the original I/Q input.  The IQUMamba backbone is unchanged.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d


class CycloFRESHAdapter1D(nn.Module):
    """Residual FRESH-style adapter for paired I/Q mixture tensors."""

    def __init__(
        self,
        input_channels: int,
        sps: int = 20,
        alphas: Sequence[float] = (0.0, 1.0, -1.0, 2.0, -2.0),
        hidden_channels: int = 8,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        gate_hidden: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"CycloFRESHAdapter1D expects one complex mixture (2 channels), got {input_channels}")
        if sps <= 0:
            raise ValueError(f"sps must be positive, got {sps}")
        if len(alphas) < 1:
            raise ValueError("alphas must contain at least one cyclic frequency")

        self.sps = int(sps)
        self.num_branches = len(tuple(alphas))
        hidden_channels = max(1, int(hidden_channels))
        gate_hidden = max(1, int(gate_hidden))

        alpha_tensor = torch.tensor([float(alpha) for alpha in alphas], dtype=torch.float32)
        self.register_buffer("alphas", alpha_tensor)

        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=self.num_branches,
            out_complex_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=hidden_channels,
            out_complex_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(2 * self.num_branches, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    def _phasors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.size(-1)
        n = torch.arange(length, device=x.device, dtype=torch.float32)
        phase = -2.0 * math.pi * self.alphas.to(device=x.device, dtype=torch.float32).unsqueeze(1) * n.unsqueeze(0)
        phase = phase / float(self.sps)
        cos = torch.cos(phase).to(dtype=x.dtype).unsqueeze(0)
        sin = torch.sin(phase).to(dtype=x.dtype).unsqueeze(0)
        return cos, sin

    def _shift_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        cos, sin = self._phasors(x)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return shifted_real, shifted_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        shifted_real, shifted_imag = self._shift_branches(x)
        hidden_real, hidden_imag = self.branch_filter(shifted_real, shifted_imag)
        gate_input = torch.cat([shifted_real, shifted_imag], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        return x + self.scale * delta


class IQUMamba1D_CycloFRESH(nn.Module):
    """Stage-4 IQUMamba wrapped with a Cyclo-FRESH input adapter."""

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
        cyclofresh_sps: int = 20,
        cyclofresh_alphas: Sequence[float] = (0.0, 1.0, -1.0, 2.0, -2.0),
        cyclofresh_hidden_channels: int = 8,
        cyclofresh_kernel_size: int = 9,
        cyclofresh_scale_init: float = 0.01,
        cyclofresh_gate_hidden: int = 8,
        cyclofresh_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.cyclofresh_adapter = CycloFRESHAdapter1D(
            input_channels=input_channels,
            sps=cyclofresh_sps,
            alphas=cyclofresh_alphas,
            hidden_channels=cyclofresh_hidden_channels,
            kernel_size=cyclofresh_kernel_size,
            scale_init=cyclofresh_scale_init,
            gate_hidden=cyclofresh_gate_hidden,
            zero_init=cyclofresh_zero_init,
        )
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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.cyclofresh_adapter(x)
        return self.backbone(x)
