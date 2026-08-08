"""Stage-4 IQUMamba with IQ-pair RMS normalization and noise-aware projection.

The normalization uses one scalar power estimate for the I/Q pair, so it does
not independently distort the two axes of the complex plane.  Predictions are
returned at the original scale before the time-domain noise-aware mixture
consistency projection is applied.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_NoiseAwareMC import IQUMamba1D_NoiseAwareMC


class SharedIQPowerNorm(nn.Module):
    """Normalize a complex I/Q waveform with one shared per-example RMS."""

    def __init__(self, eps: float = 1e-6, detach_scale: bool = False) -> None:
        super().__init__()
        self.eps = float(eps)
        self.detach_scale = bool(detach_scale)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected an I/Q tensor shaped (B, 2, L), got {tuple(x.shape)}")
        # sqrt(E[I^2 + Q^2]); one value is shared by both complex axes.
        scale = x.float().square().sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
        scale = scale.clamp_min(self.eps).sqrt().to(dtype=x.dtype)
        if self.detach_scale:
            scale = scale.detach()
        return x / scale, scale


class IQUMamba1D_SharedIQNorm(IQUMamba1D):
    """Stage-4 IQUMamba with only shared-IQ input power normalization."""

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
        iq_power_norm_eps: float = 1e-6,
        iq_power_norm_detach_scale: bool = False,
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
        )
        self.iq_power_norm = SharedIQPowerNorm(
            eps=iq_power_norm_eps,
            detach_scale=iq_power_norm_detach_scale,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        normalized, scale = self.iq_power_norm(x)
        outputs = super().forward(normalized)
        if isinstance(outputs, (list, tuple)):
            return [output * scale for output in outputs]
        return outputs * scale


class IQUMamba1D_MultiViewConsistent(IQUMamba1D_NoiseAwareMC):
    """Stage-4 backbone plus low-cost complex normalization and MC projection."""

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
        iq_power_norm_eps: float = 1e-6,
        iq_power_norm_detach_scale: bool = False,
        **noise_mc_kwargs,
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
            **noise_mc_kwargs,
        )
        self.iq_power_norm = SharedIQPowerNorm(
            eps=iq_power_norm_eps,
            detach_scale=iq_power_norm_detach_scale,
        )

    @staticmethod
    def _restore_scale(
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        scale: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(outputs, (list, tuple)):
            return [output * scale for output in outputs]
        return outputs * scale

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        normalized, scale = self.iq_power_norm(x)
        # Bypass IQUMamba1D_NoiseAwareMC.forward so projection happens only
        # after restoring the physical input amplitude.
        normalized_sources = IQUMamba1D.forward(self, normalized)
        sources = self._restore_scale(normalized_sources, scale)
        return self._project_outputs(sources, x)
