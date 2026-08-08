"""IQUBiMamba1D_CSB_Constellation - CSB BiMamba with a soft constellation prior.

This variant keeps the strong CSB-BiMamba separator unchanged and adds a
baseline-preserving refinement head after the decoder.  The head computes a
soft PSK phase reference for each estimated source and learns a zero-initialized
residual correction from:
  - the current source estimate;
  - its soft constellation reference;
  - the constellation error;
  - the mixture-consistency residual.

The prior is deliberately soft: it preserves the estimate magnitude and only
uses constellation centers as phase guidance, avoiding hard symbol decisions on
unsynchronized waveform samples.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F

from models.IQUBiMamba1D_CSB import IQUBiMamba1D_CSB


def _logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return math.log(prob / (1.0 - prob))


class SoftPSKConstellationPrior1D(nn.Module):
    """Build a magnitude-preserving soft PSK phase reference."""

    def __init__(
        self,
        constellation_order: int = 8,
        temperature: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if constellation_order < 2:
            raise ValueError(f"constellation_order must be >= 2, got {constellation_order}")
        self.constellation_order = int(constellation_order)
        self.temperature = float(temperature)
        self.eps = float(eps)

        angles = 2.0 * math.pi * torch.arange(self.constellation_order, dtype=torch.float32) / self.constellation_order
        centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("centers", centers.view(1, 1, self.constellation_order, 2, 1), persistent=False)

    def forward(self, estimates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return soft constellation reference and assignment confidence.

        Args:
            estimates: Tensor shaped (B, K, 2, T).
        """
        if estimates.ndim != 4 or estimates.size(2) != 2:
            raise ValueError(f"Expected estimates shaped (B, K, 2, T), got {tuple(estimates.shape)}")

        magnitude = torch.sqrt(estimates.square().sum(dim=2, keepdim=True) + self.eps)
        unit = estimates / magnitude.clamp_min(self.eps)
        distances = (unit.unsqueeze(2) - self.centers.to(unit.device, unit.dtype)).square().sum(dim=3)
        logits = -distances / max(self.temperature, self.eps)
        probs = torch.softmax(logits, dim=2)
        reference_unit = (probs.unsqueeze(3) * self.centers.to(unit.device, unit.dtype)).sum(dim=2)
        reference = reference_unit * magnitude
        confidence = probs.max(dim=2, keepdim=False).values
        return reference, confidence


class ConstellationGuidedRefinementHead1D(nn.Module):
    """Zero-initialized source-wise residual correction guided by PSK geometry."""

    def __init__(
        self,
        num_sources: int,
        constellation_type: str = "psk",
        constellation_order: int = 8,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        temperature: float = 0.25,
        dropout: float = 0.0,
        gate_init: float = 0.1,
        residual_scale_init: float = 1.0,
        use_mixture_residual: bool = True,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if constellation_type.lower() != "psk":
            raise ValueError("ConstellationGuidedRefinementHead1D currently supports constellation_type='psk' only")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size should be odd to preserve length, got {kernel_size}")

        self.num_sources = int(num_sources)
        self.constellation_type = constellation_type.lower()
        self.constellation_order = int(constellation_order)
        self.use_mixture_residual = bool(use_mixture_residual)
        self.eps = float(eps)
        padding = int(kernel_size) // 2
        hidden = int(hidden_channels)

        self.prior = SoftPSKConstellationPrior1D(
            constellation_order=self.constellation_order,
            temperature=temperature,
            eps=eps,
        )
        in_channels = 8 + (2 if self.use_mixture_residual else 0)
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=kernel_size, padding=padding, bias=True),
            nn.InstanceNorm1d(hidden, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding, bias=True),
            nn.InstanceNorm1d(hidden, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.delta_head = nn.Conv1d(hidden, 2, kernel_size=kernel_size, padding=padding, bias=True)
        self.gate_head = nn.Conv1d(hidden, 1, kernel_size=1, bias=True)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, _logit(gate_init))
        if zero_init:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

    def _reshape_sources(self, estimates: torch.Tensor) -> torch.Tensor:
        batch, channels, length = estimates.shape
        expected_channels = self.num_sources * 2
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} output channels, got {channels}")
        return estimates.reshape(batch, self.num_sources, 2, length)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if estimates.ndim != 3:
            raise ValueError(f"Expected estimates shaped (B, 2K, T), got {tuple(estimates.shape)}")
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(f"Expected mixture shaped (B, 2, T), got {tuple(mixture.shape)}")

        batch, _, length = estimates.shape
        source_estimates = self._reshape_sources(estimates)
        reference, confidence = self.prior(source_estimates)
        magnitude = torch.sqrt(source_estimates.square().sum(dim=2, keepdim=True) + self.eps)
        const_error = source_estimates - reference

        features = [
            source_estimates,
            reference,
            const_error,
            confidence.unsqueeze(2),
            magnitude,
        ]
        if self.use_mixture_residual:
            residual = mixture - source_estimates.sum(dim=1)
            features.append(residual.unsqueeze(1).expand(-1, self.num_sources, -1, -1))

        per_source_features = torch.cat(features, dim=2).reshape(batch * self.num_sources, -1, length)
        hidden = self.feature_encoder(per_source_features)
        gate = torch.sigmoid(self.gate_head(hidden))
        delta = self.delta_head(hidden) * gate * self.residual_scale
        delta = delta.view(batch, self.num_sources, 2, length)
        refined = source_estimates + delta
        return refined.reshape(batch, self.num_sources * 2, length)


class IQUBiMamba1D_CSB_Constellation(IQUBiMamba1D_CSB):
    """CSB-BiMamba separator followed by constellation-guided refinement."""

    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: type[nn.Conv1d],
        kernel_sizes: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        conv_bias: bool = True,
        norm_op: type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        complex_stem_hidden_channels: int = 32,
        complex_stem_kernel_size: int = 5,
        complex_bottleneck_hidden_channels: int = 128,
        complex_bottleneck_num_blocks: int = 3,
        complex_bottleneck_kernel_size: int = 5,
        complex_bottleneck_dilation_growth: int = 2,
        complex_bottleneck_zero_init: bool = True,
        constellation_type: str = "psk",
        constellation_order: int = 8,
        cgr_hidden_channels: int = 48,
        cgr_kernel_size: int = 7,
        cgr_temperature: float = 0.25,
        cgr_dropout: float = 0.0,
        cgr_gate_init: float = 0.1,
        cgr_residual_scale_init: float = 1.0,
        cgr_use_mixture_residual: bool = True,
        cgr_zero_init: bool = True,
        cgr_refine_deep_supervision: bool = False,
        cgr_apply_train: bool = True,
        cgr_apply_eval: bool = True,
    ) -> None:
        super().__init__(
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
            complex_stem_hidden_channels=complex_stem_hidden_channels,
            complex_stem_kernel_size=complex_stem_kernel_size,
            complex_bottleneck_hidden_channels=complex_bottleneck_hidden_channels,
            complex_bottleneck_num_blocks=complex_bottleneck_num_blocks,
            complex_bottleneck_kernel_size=complex_bottleneck_kernel_size,
            complex_bottleneck_dilation_growth=complex_bottleneck_dilation_growth,
            complex_bottleneck_zero_init=complex_bottleneck_zero_init,
        )
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")
        self.num_sources = num_classes // 2
        self.cgr_refine_deep_supervision = bool(cgr_refine_deep_supervision)
        self.cgr_apply_train = bool(cgr_apply_train)
        self.cgr_apply_eval = bool(cgr_apply_eval)
        self.constellation_head = ConstellationGuidedRefinementHead1D(
            num_sources=self.num_sources,
            constellation_type=constellation_type,
            constellation_order=constellation_order,
            hidden_channels=cgr_hidden_channels,
            kernel_size=cgr_kernel_size,
            temperature=cgr_temperature,
            dropout=cgr_dropout,
            gate_init=cgr_gate_init,
            residual_scale_init=cgr_residual_scale_init,
            use_mixture_residual=cgr_use_mixture_residual,
            zero_init=cgr_zero_init,
        )

    def _should_refine(self) -> bool:
        return self.cgr_apply_train if self.training else self.cgr_apply_eval

    def _refine_outputs(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        mixture: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            if self.cgr_refine_deep_supervision:
                return [self.constellation_head(out, mixture) for out in outputs]
            outputs[0] = self.constellation_head(outputs[0], mixture)
            return outputs
        return self.constellation_head(outputs, mixture)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        outputs = super().forward(x)
        if not self._should_refine():
            return outputs
        return self._refine_outputs(outputs, x)
