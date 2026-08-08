"""Blind multi-rate input adapter for stage-4 IQUMamba.

The adapter keeps the original IQUMamba backbone intact and adds a small
near-identity residual before the backbone.  It uses only the received mixture:
parallel temporal branches with different kernels/dilations are softly gated by
blind mixture statistics.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D


class BlindMultiRateInputAdapter1D(nn.Module):
    """Near-identity multi-scale input residual computed from the mixture."""

    def __init__(
        self,
        input_channels: int = 2,
        hidden_channels: int = 8,
        kernel_sizes: Sequence[int] = (5, 9, 17, 33),
        dilations: Sequence[int] = (1, 2, 4, 8),
        scale_init: float = 0.01,
        zero_init: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError(f"BlindMultiRateInputAdapter1D expects I/Q input with 2 channels, got {input_channels}")

        hidden_channels = max(1, int(hidden_channels))
        kernel_sizes = tuple(max(1, int(k)) for k in kernel_sizes)
        dilations = tuple(max(1, int(d)) for d in dilations)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("kernel_sizes and dilations must have the same length")
        if not kernel_sizes:
            raise ValueError("at least one multi-rate branch is required")

        self.eps = float(eps)
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.branches = nn.ModuleList()
        for kernel_size, dilation in zip(kernel_sizes, dilations):
            if kernel_size % 2 == 0:
                kernel_size += 1
            padding = dilation * (kernel_size // 2)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(2, hidden_channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
                    nn.SiLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
                    nn.SiLU(),
                )
            )

        self.branch_gate = nn.Sequential(
            nn.Linear(6, max(4, hidden_channels)),
            nn.SiLU(),
            nn.Linear(max(4, hidden_channels), len(self.branches)),
        )
        self.out_proj = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def _blind_stats(self, x: torch.Tensor) -> torch.Tensor:
        real = x[:, 0, :].float()
        imag = x[:, 1, :].float()
        z = torch.complex(real, imag)
        power = real.square() + imag.square()
        mean_power = power.mean(dim=-1) + self.eps
        envelope = torch.sqrt(power + self.eps)
        envelope_mean = envelope.mean(dim=-1) + self.eps

        log_rms = torch.log1p(torch.sqrt(mean_power))
        papr = torch.log1p(power.amax(dim=-1) / mean_power)
        envelope_cv = envelope.std(dim=-1, unbiased=False) / envelope_mean
        circularity = torch.abs((z * z).mean(dim=-1)) / mean_power

        if z.size(-1) > 1:
            phase_step = torch.angle(z[:, 1:] * torch.conj(z[:, :-1]))
            phase_var = phase_step.var(dim=-1, unbiased=False) / (math.pi * math.pi)
            diff_power = (x[:, :, 1:] - x[:, :, :-1]).float().square().mean(dim=(1, 2))
            diff_ratio = torch.log1p(diff_power / mean_power)
        else:
            phase_var = torch.zeros_like(log_rms)
            diff_ratio = torch.zeros_like(log_rms)

        return torch.stack(
            [log_rms, papr, envelope_cv, circularity, phase_var, diff_ratio],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"BlindMultiRateInputAdapter1D expects input shaped (B, 2, L), got {tuple(x.shape)}")

        branch_outputs = [branch(x) for branch in self.branches]
        stats = self._blind_stats(x)
        weights = torch.softmax(self.branch_gate(stats.float()), dim=-1).to(dtype=x.dtype)
        fused = None
        for idx, branch_output in enumerate(branch_outputs):
            weighted = branch_output * weights[:, idx].view(-1, 1, 1)
            fused = weighted if fused is None else fused + weighted
        delta = self.out_proj(fused)
        return x + self.scale * torch.tanh(delta)


class IQUMamba1D_BlindMultiRateInput(nn.Module):
    """Stage-4 IQUMamba with a blind multi-rate input adapter."""

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
        multirate_hidden_channels: int = 8,
        multirate_kernel_sizes: Sequence[int] = (5, 9, 17, 33),
        multirate_dilations: Sequence[int] = (1, 2, 4, 8),
        multirate_scale_init: float = 0.01,
        multirate_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.input_adapter = BlindMultiRateInputAdapter1D(
            input_channels=input_channels,
            hidden_channels=multirate_hidden_channels,
            kernel_sizes=multirate_kernel_sizes,
            dilations=multirate_dilations,
            scale_init=multirate_scale_init,
            zero_init=multirate_zero_init,
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
        x = self.input_adapter(x)
        return self.backbone(x)
