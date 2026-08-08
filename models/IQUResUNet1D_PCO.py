"""Phase-equivariant, correlation-gated, orthogonalized ResUNet variants.

These variants keep the stage-42 ResUNet encoder/decoder geometry and add
blind, mixture-only communication structure:

* PhaseEquivariantInputAdapter: complex-valued residual filtering that obeys
  f(exp(j theta) x) = exp(j theta) f(x).
* CorrelationSkipGate: skip conditioning from local complex autocorrelation.
* SourceOrthogonalizationHead: soft second-order output decorrelation.

No labels, bit streams, sampling metadata, or class priors are consumed.
"""

from __future__ import annotations

from typing import List, Type

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import UNetResDecoder
from models.IQUResUNet1D import ResidualConvEncoder


def _zero_conv(conv: nn.Conv1d) -> None:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class ComplexConv1d(nn.Module):
    """Complex 1D convolution implemented with tied real-valued convolutions."""

    def __init__(self, in_complex: int, out_complex: int, kernel_size: int, padding: int = 0, bias: bool = True):
        super().__init__()
        self.real = nn.Conv1d(in_complex, out_complex, kernel_size=kernel_size, padding=padding, bias=bias)
        self.imag = nn.Conv1d(in_complex, out_complex, kernel_size=kernel_size, padding=padding, bias=bias)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = self.real(real) - self.imag(imag)
        out_imag = self.real(imag) + self.imag(real)
        return out_real, out_imag


class PhaseEquivariantInputAdapter(nn.Module):
    """Residual complex adapter with magnitude-conditioned real gate."""

    def __init__(
        self,
        hidden_channels: int = 16,
        kernel_size: int = 7,
        scale_init: float = 0.01,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        hidden_channels = max(1, int(hidden_channels))
        padding = kernel_size // 2

        self.in_proj = ComplexConv1d(1, hidden_channels, kernel_size=kernel_size, padding=padding)
        self.gate = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = ComplexConv1d(hidden_channels, 1, kernel_size=1, padding=0)
        _zero_conv(self.out_proj.real)
        _zero_conv(self.out_proj.imag)
        self.adapter_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw IQ input with shape (B,2,L), got {tuple(x.shape)}")
        real = x[:, 0:1]
        imag = x[:, 1:2]
        hidden_real, hidden_imag = self.in_proj(real, imag)
        magnitude = torch.sqrt(hidden_real.pow(2) + hidden_imag.pow(2) + 1e-8)
        gate = self.gate(magnitude)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1)
        return x + self.adapter_scale * delta


class LocalCorrelationConditioner(nn.Module):
    """Compute local normalized complex autocorrelation features from mixture IQ."""

    def __init__(self, lags: List[int], window: int = 33, eps: float = 1e-6) -> None:
        super().__init__()
        clean_lags = sorted({int(lag) for lag in lags if int(lag) > 0})
        if not clean_lags:
            clean_lags = [1, 2, 4, 8]
        window = int(window)
        if window % 2 == 0:
            window += 1
        self.lags = clean_lags
        self.window = max(1, window)
        self.eps = float(eps)
        self.out_channels = 2 * len(clean_lags)

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x, kernel_size=self.window, stride=1, padding=self.window // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw IQ input with shape (B,2,L), got {tuple(x.shape)}")
        real = x[:, 0:1]
        imag = x[:, 1:2]
        power = real.pow(2) + imag.pow(2)

        features = []
        for lag in self.lags:
            lag = min(lag, x.size(-1) - 1)
            lag_real = F.pad(real[:, :, :-lag], (lag, 0))
            lag_imag = F.pad(imag[:, :, :-lag], (lag, 0))
            lag_power = lag_real.pow(2) + lag_imag.pow(2)

            corr_real = real * lag_real + imag * lag_imag
            corr_imag = imag * lag_real - real * lag_imag

            corr_real = self._smooth(corr_real)
            corr_imag = self._smooth(corr_imag)
            denom = torch.sqrt(self._smooth(power) * self._smooth(lag_power) + self.eps)
            features.extend([corr_real / denom, corr_imag / denom])
        return torch.cat(features, dim=1)


class CorrelationSkipGate(nn.Module):
    """Apply local-correlation FiLM gates to encoder skips."""

    def __init__(self, features_per_stage: List[int], cond_channels: int, scale_init: float = 0.01) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv1d(cond_channels, int(ch), kernel_size=1) for ch in features_per_stage])
        for proj in self.proj:
            _zero_conv(proj)
        self.gate_scales = nn.ParameterList(
            [nn.Parameter(torch.tensor(float(scale_init))) for _ in features_per_stage]
        )

    def forward(self, skips: List[torch.Tensor], cond: torch.Tensor) -> List[torch.Tensor]:
        gated = []
        for skip, proj, scale in zip(skips, self.proj, self.gate_scales):
            cond_l = F.interpolate(cond, size=skip.size(-1), mode="linear", align_corners=False)
            gate = torch.tanh(proj(cond_l))
            gated.append(skip * (1.0 + scale * gate))
        return gated


class SourceOrthogonalizationHead(nn.Module):
    """Soft complex whitening head for separated source slots."""

    def __init__(self, scale_init: float = 0.01, eps: float = 1e-5) -> None:
        super().__init__()
        self.orth_scale = nn.Parameter(torch.tensor(float(scale_init)))
        self.eps = float(eps)

    def _orthogonalize(self, output: torch.Tensor) -> torch.Tensor:
        if output.dim() != 3 or output.size(1) % 2 != 0:
            raise ValueError(f"Expected separated IQ output with shape (B,2K,L), got {tuple(output.shape)}")
        batch, channels, length = output.shape
        n_slots = channels // 2
        slots = output.view(batch, n_slots, 2, length)
        complex_slots = torch.complex(slots[:, :, 0], slots[:, :, 1])

        cov = torch.matmul(complex_slots, complex_slots.conj().transpose(-1, -2)) / max(1, length)
        eye = torch.eye(n_slots, device=output.device, dtype=cov.dtype).unsqueeze(0)
        eigvals, eigvecs = torch.linalg.eigh(cov + self.eps * eye)
        inv_diag = torch.diag_embed(torch.rsqrt(eigvals.clamp_min(self.eps))).to(dtype=cov.dtype)
        inv_sqrt = eigvecs @ inv_diag @ eigvecs.conj().transpose(-1, -2)
        whitened = torch.matmul(inv_sqrt, complex_slots)

        original_rms = torch.sqrt(torch.mean(torch.abs(complex_slots).pow(2), dim=-1, keepdim=True) + self.eps)
        white_rms = torch.sqrt(torch.mean(torch.abs(whitened).pow(2), dim=-1, keepdim=True) + self.eps)
        whitened = whitened * (original_rms / white_rms)

        mixed = complex_slots + self.orth_scale * (whitened - complex_slots)
        stacked = torch.stack([mixed.real, mixed.imag], dim=2).reshape(batch, channels, length)
        return stacked.to(dtype=output.dtype)

    def forward(self, output):
        if isinstance(output, (list, tuple)):
            return [self._orthogonalize(item) for item in output]
        return self._orthogonalize(output)


class _IQUResUNet1D_PCOBase(nn.Module):
    use_phase_adapter = False
    use_corr_gate = False
    use_orth_head = False

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
        pco_phase_channels: int = 16,
        pco_phase_kernel_size: int = 7,
        pco_phase_scale_init: float = 0.01,
        pco_corr_lags: List[int] = None,
        pco_corr_window: int = 33,
        pco_corr_scale_init: float = 0.01,
        pco_orth_scale_init: float = 0.01,
        pco_orth_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("PCO ResUNet expects one complex IQ mixture represented by 2 channels")

        self.phase_adapter = (
            PhaseEquivariantInputAdapter(
                hidden_channels=pco_phase_channels,
                kernel_size=pco_phase_kernel_size,
                scale_init=pco_phase_scale_init,
            )
            if self.use_phase_adapter
            else nn.Identity()
        )
        self.encoder = ResidualConvEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
        )
        if self.use_corr_gate:
            self.corr_conditioner = LocalCorrelationConditioner(
                lags=pco_corr_lags or [1, 2, 4, 8],
                window=pco_corr_window,
            )
            self.corr_gate = CorrelationSkipGate(
                features_per_stage=list(features_per_stage),
                cond_channels=self.corr_conditioner.out_channels,
                scale_init=pco_corr_scale_init,
            )
        else:
            self.corr_conditioner = None
            self.corr_gate = None

        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )
        self.orth_head = (
            SourceOrthogonalizationHead(scale_init=pco_orth_scale_init, eps=pco_orth_eps)
            if self.use_orth_head
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        raw = x
        x = self.phase_adapter(x)
        skips = self.encoder(x)
        if self.corr_gate is not None and self.corr_conditioner is not None:
            cond = self.corr_conditioner(raw)
            skips = self.corr_gate(skips, cond)
        output = self.decoder(skips)
        return self.orth_head(output)


class IQUResUNet1D_PhaseEquivariant(_IQUResUNet1D_PCOBase):
    use_phase_adapter = True


class IQUResUNet1D_CorrGate(_IQUResUNet1D_PCOBase):
    use_corr_gate = True


class IQUResUNet1D_PCO(_IQUResUNet1D_PCOBase):
    use_phase_adapter = True
    use_corr_gate = True
    use_orth_head = True
