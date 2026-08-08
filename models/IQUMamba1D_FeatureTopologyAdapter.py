"""Feature-domain topology adapter for IQUMamba1D.

This stage keeps the raw IQ waveform untouched. It lets the encoder features
learn small topology-aware residual corrections, which is a better fit for
high-order QAM/APSK cases where the raw mixed constellation is not reliable.
"""

from typing import List, Sequence, Type

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 1:
        return 1
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


class FeatureTopologyFiLM1D(nn.Module):
    """Near-identity feature adapter driven by local feature statistics."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 16,
        kernel_size: int = 7,
        scale_init: float = 0.01,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        channels = int(channels)
        hidden_channels = max(1, int(hidden_channels))
        kernel_size = _odd_kernel(kernel_size)
        padding = kernel_size // 2

        self.channels = channels
        self.scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Conv1d(channels + 2, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm1d(hidden_channels, affine=True),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm1d(hidden_channels, affine=True),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, channels * 2, kernel_size=1),
        )
        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _hints(self, x: torch.Tensor) -> torch.Tensor:
        energy = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        if x.size(-1) > 1:
            left = torch.nn.functional.pad(energy[..., :-1], (1, 0))
            slope = energy - left
        else:
            slope = torch.zeros_like(energy)
        return torch.cat([x, energy, slope], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError(f"Expected feature shape (B, {self.channels}, L), got {tuple(x.shape)}")
        gamma_beta = self.net(self._hints(x))
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x + self.scale.to(dtype=x.dtype) * (torch.tanh(gamma) * x + torch.tanh(beta))


class IQUMamba1D_FeatureTopologyAdapter(nn.Module):
    """IQUMamba1D with independent feature-domain topology adapters on encoder skips."""

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
        feature_topology_hidden_channels: int = 16,
        feature_topology_kernel_size: int = 7,
        feature_topology_scale_init: float = 0.01,
        feature_topology_zero_init: bool = True,
        feature_topology_apply_stages: Sequence[int] = (),
    ) -> None:
        super().__init__()
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

        if feature_topology_apply_stages:
            active = {int(stage) for stage in feature_topology_apply_stages}
        else:
            active = set(range(len(features_per_stage)))

        adapters = []
        for idx, channels in enumerate(features_per_stage):
            if idx in active:
                adapters.append(
                    FeatureTopologyFiLM1D(
                        channels=channels,
                        hidden_channels=feature_topology_hidden_channels,
                        kernel_size=feature_topology_kernel_size,
                        scale_init=feature_topology_scale_init,
                        zero_init=feature_topology_zero_init,
                    )
                )
            else:
                adapters.append(nn.Identity())
        self.feature_adapters = nn.ModuleList(adapters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.backbone.encoder(x)
        adapted_skips = [adapter(skip) for adapter, skip in zip(self.feature_adapters, skips)]
        return self.backbone.decoder(adapted_skips)
