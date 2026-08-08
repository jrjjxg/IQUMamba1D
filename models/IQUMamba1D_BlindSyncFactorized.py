"""Stage 224: low-cost blind synchronization factorization for IQUMamba."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Type

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_EvidenceRoutedMoE import PaddedStage4Backbone


def _complex_lag_correlation(z: torch.Tensor, lag: int, eps: float) -> torch.Tensor:
    if lag <= 0:
        lag = 1
    if lag >= z.size(-1):
        return torch.zeros(z.size(0), device=z.device, dtype=z.real.dtype,)
    left = z[:, lag:]
    right = z[:, :-lag]
    numerator = (left * torch.conj(right)).mean(dim=-1)
    left_power = left.abs().square().mean(dim=-1)
    right_power = right.abs().square().mean(dim=-1)
    denominator = (left_power * right_power).add(eps).sqrt()
    return numerator / denominator


class BlindSyncEvidence(nn.Module):
    """Extract global relative-CFO/delay evidence from one received IQ stream."""

    def __init__(
        self,
        lags: Iterable[int] = (1, 2, 4, 8),
        eps: float = 1e-6,
    ):
        super().__init__()
        normalized_lags = tuple(sorted({int(lag) for lag in lags if int(lag) > 0}))
        self.register_buffer("lags", torch.tensor(normalized_lags, dtype=torch.long), persistent=False)
        self.eps = float(eps)
        # rms, phase cos/sin, phase coherence, phase circular variance,
        # envelope CV, envelope roughness, and (real, imag, magnitude) per lag.
        self.num_stats = 7 + 3 * len(normalized_lags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"BlindSyncEvidence expects [B, 2, L], got {tuple(x.shape)}")
        real = x[:, 0, :].float()
        imag = x[:, 1, :].float()
        z = torch.complex(real, imag)
        envelope = z.abs()
        power = envelope.square()
        rms = power.mean(dim=-1).add(self.eps).sqrt()

        if z.size(-1) > 1:
            phase_step = z[:, 1:] * torch.conj(z[:, :-1])
            phase_denominator = (z[:, 1:].abs() * z[:, :-1].abs()).add(self.eps)
            unit_step = phase_step / phase_denominator
            phase_cos = unit_step.real.mean(dim=-1)
            phase_sin = unit_step.imag.mean(dim=-1)
            phase_coherence = unit_step.abs().mean(dim=-1)
        else:
            phase_cos = torch.ones_like(rms)
            phase_sin = torch.zeros_like(rms)
            phase_coherence = torch.zeros_like(rms)
        circular_variance = (1.0 - torch.sqrt(phase_cos.square() + phase_sin.square() + self.eps)).clamp(
            min=0.0,
        )

        envelope_mean = envelope.mean(dim=-1).add(self.eps)
        envelope_cv = envelope.std(dim=-1, unbiased=False) / envelope_mean
        if envelope.size(-1) > 1:
            envelope_roughness = envelope.diff(dim=-1).abs().mean(dim=-1) / envelope_mean
        else:
            envelope_roughness = torch.zeros_like(rms)

        stats = [
            torch.log1p(rms),
            phase_cos,
            phase_sin,
            phase_coherence,
            circular_variance,
            envelope_cv,
            envelope_roughness,
        ]
        for lag_tensor in self.lags:
            correlation = _complex_lag_correlation(z, int(lag_tensor.item()), self.eps)
            stats.extend([correlation.real, correlation.imag, correlation.abs()])
        return torch.nan_to_num(torch.stack(stats, dim=-1), nan=0.0, posinf=10.0, neginf=-10.0).to(x.dtype)


class BlindSyncFactorizedAdapter(nn.Module):
    """Factor local high-resolution IQ cues by global sync evidence."""

    local_feature_channels = 7

    def __init__(
        self,
        stat_dim: int,
        hidden: int = 12,
        kernel_size: int = 5,
        scale_init: float = 0.01,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden = max(1, int(hidden))
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.eps = float(eps)
        self.local_net = nn.Sequential(
            nn.Conv1d(self.local_feature_channels, hidden, kernel_size, padding=kernel_size // 2),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(int(stat_dim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.output = nn.Conv1d(hidden, 2, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _local_features(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.square().mean(dim=(1, 2), keepdim=True).add(self.eps).sqrt()
        normalized = x_float / rms
        delta = F.pad(normalized[..., 1:] - normalized[..., :-1], (1, 0))

        z = torch.complex(x_float[:, 0, :], x_float[:, 1, :])
        if z.size(-1) > 1:
            step = z[:, 1:] * torch.conj(z[:, :-1])
            denominator = (z[:, 1:].abs() * z[:, :-1].abs()).add(self.eps)
            unit = step / denominator
            phase_cos = F.pad(unit.real, (1, 0))
            phase_sin = F.pad(unit.imag, (1, 0))
            coherence = F.pad(unit.abs(), (1, 0))
        else:
            phase_cos = torch.ones_like(z.real)
            phase_sin = torch.zeros_like(z.real)
            coherence = torch.zeros_like(z.real)
        local = torch.cat(
            [normalized, delta, phase_cos[:, None], phase_sin[:, None], coherence[:, None]],
            dim=1,
        )
        return torch.nan_to_num(local, nan=0.0, posinf=10.0, neginf=-10.0).to(x.dtype)

    def forward(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"BlindSyncFactorizedAdapter expects [B, 2, L], got {tuple(x.shape)}")
        if stats.ndim != 2 or stats.size(0) != x.size(0):
            raise ValueError("sync statistics must have shape [B, stat_dim]")
        local = self.local_net(self._local_features(x))
        global_gate = self.global_gate(stats.float()).to(dtype=local.dtype).unsqueeze(-1)
        delta = torch.tanh(self.output(local * global_gate))
        return x + self.scale.to(dtype=x.dtype) * delta


def _rounded_backbone_length(input_size: int, strides: Sequence[int]) -> int:
    total_stride = 1
    for stride in strides:
        total_stride *= max(1, int(stride))
    return ((int(input_size) + total_stride - 1) // total_stride) * total_stride


class IQUMamba1D_BlindSyncFactorized(nn.Module):
    """Stage-4 backbone with a bounded, mixture-only sync-conditioned input residual."""

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
        norm_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        sync_hidden: int = 12,
        sync_kernel_size: int = 5,
        sync_scale_init: float = 0.01,
        sync_lags: Iterable[int] = (1, 2, 4, 8),
        sync_eps: float = 1e-6,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        if input_channels != 2:
            raise ValueError("BlindSyncFactorized requires a single complex IQ mixture with 2 channels")

        self.sync_evidence = BlindSyncEvidence(lags=sync_lags, eps=sync_eps)
        self.sync_adapter = BlindSyncFactorizedAdapter(
            stat_dim=self.sync_evidence.num_stats,
            hidden=sync_hidden,
            kernel_size=sync_kernel_size,
            scale_init=sync_scale_init,
            eps=sync_eps,
        )
        backbone_length = _rounded_backbone_length(input_size, strides)
        raw_backbone = IQUMamba1D(
            input_size=backbone_length,
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
        self.backbone = PaddedStage4Backbone(raw_backbone, backbone_length)

    def forward(self, x: torch.Tensor):
        stats = self.sync_evidence(x)
        adapted = self.sync_adapter(x, stats)
        return self.backbone(adapted)
