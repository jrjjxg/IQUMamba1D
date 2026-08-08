"""Multi-resolution complex STFT mask separator for IQ BSS.

This model implements a multi-resolution spectral masking paradigm:

    several complex STFT resolutions -> per-resolution temporal mask branches
    -> complex iSTFT for each branch -> learned branch fusion
    -> mixture consistency projection.

The design is intentionally framework-level: it does not reuse IQUMamba or any
state-space layer, and it does not use metadata such as symbol timing or
modulation labels.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class TemporalMaskBlock(nn.Module):
    """Residual temporal block used inside each STFT mask branch."""

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.net = nn.Sequential(
            nn.GroupNorm(1, int(hidden_dim)),
            nn.PReLU(),
            nn.Conv1d(
                int(hidden_dim),
                int(hidden_dim),
                kernel_size=int(kernel_size),
                padding=padding,
                dilation=int(dilation),
                groups=int(hidden_dim),
            ),
            nn.PReLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(int(hidden_dim), int(hidden_dim), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.size(-1) != x.size(-1):
            y = F.interpolate(y, size=x.size(-1), mode="linear", align_corners=False)
        return x + y


class STFTMaskBranch(nn.Module):
    """Single complex STFT resolution branch."""

    def __init__(
        self,
        n_srcs: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        hidden_dim: int = 128,
        n_blocks: int = 6,
        kernel_size: int = 5,
        dilation_cycle: int = 4,
        dropout: float = 0.0,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_head_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.n_freq = self.n_fft
        self.mask_bound = float(mask_bound)
        self.mask_sum_constraint = bool(mask_sum_constraint)
        dilation_cycle = max(1, int(dilation_cycle))

        in_channels = 2 * self.n_freq
        out_channels = self.n_srcs * 2 * self.n_freq
        self.net = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv1d(in_channels, int(hidden_dim), 1),
            nn.PReLU(),
            *[
                TemporalMaskBlock(
                    hidden_dim=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=2 ** (idx % dilation_cycle),
                    dropout=float(dropout),
                )
                for idx in range(int(n_blocks))
            ],
            nn.PReLU(),
            nn.Conv1d(int(hidden_dim), out_channels, 1),
        )
        if mask_head_zero_init:
            self._init_zero_mask_head()
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    def _init_zero_mask_head(self) -> None:
        final_conv = self.net[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    def _spec_to_channels(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B,T,F) complex -> (B,2F,T)
        features = torch.cat([spec.real, spec.imag], dim=2)
        return features.permute(0, 2, 1).contiguous()

    def _predict_masks(self, logits: torch.Tensor, n_frames: int) -> torch.Tensor:
        batch = logits.size(0)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits.reshape(batch, n_frames, self.n_srcs, 2, self.n_freq)
        if self.mask_bound > 0:
            logits = torch.tanh(logits) * self.mask_bound

        mask_real = logits[:, :, :, 0, :].permute(0, 2, 1, 3)
        mask_imag = logits[:, :, :, 1, :].permute(0, 2, 1, 3)
        if self.mask_sum_constraint:
            sum_real = mask_real.sum(dim=1, keepdim=True)
            sum_imag = mask_imag.sum(dim=1, keepdim=True)
            mask_real = mask_real - (sum_real - 1.0) / self.n_srcs
            mask_imag = mask_imag - sum_imag / self.n_srcs
        return torch.complex(mask_real, mask_imag)

    def forward(
        self,
        mix_complex: torch.Tensor,
        length: int,
        center: bool = True,
    ) -> torch.Tensor:
        if mix_complex.ndim != 2 or not torch.is_complex(mix_complex):
            raise ValueError(f"Expected complex mixture (B,L), got {tuple(mix_complex.shape)}")

        real_dtype = mix_complex.real.dtype
        window = self.window.to(device=mix_complex.device, dtype=real_dtype)
        mix_spec = torch.stft(
            mix_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=bool(center),
            onesided=False,
            return_complex=True,
        ).transpose(1, 2).contiguous()

        branch_input = self._spec_to_channels(mix_spec)
        logits = self.net(branch_input)
        masks = self._predict_masks(logits, n_frames=mix_spec.size(1))
        separated_specs = masks * mix_spec.unsqueeze(1)

        reconstructed = []
        for source_idx in range(self.n_srcs):
            signal = torch.istft(
                separated_specs[:, source_idx].transpose(1, 2).contiguous(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=bool(center),
                onesided=False,
                length=length,
                return_complex=True,
            )
            reconstructed.append(signal)

        batch = mix_complex.size(0)
        return torch.stack(
            [torch.stack([signal.real, signal.imag], dim=1) for signal in reconstructed],
            dim=1,
        ).reshape(batch, 2 * self.n_srcs, length)


class MultiResolutionSTFTMaskSeparator1D(nn.Module):
    """Multi-resolution complex STFT mask separator with IQ-compatible I/O."""

    def __init__(
        self,
        n_srcs: int = 2,
        n_ffts: Sequence[int] = (128, 256, 512),
        hop_lengths: Sequence[int] | None = None,
        win_lengths: Sequence[int] | None = None,
        center: bool = True,
        normalize_input: bool = True,
        hidden_dim: int = 128,
        n_blocks: int = 6,
        kernel_size: int = 5,
        dilation_cycle: int = 4,
        dropout: float = 0.0,
        mask_bound: float = 4.0,
        mask_sum_constraint: bool = True,
        mask_head_zero_init: bool = True,
        eps: float = 1e-8,
        apply_projection: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_detach_weights: bool = False,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.center = bool(center)
        self.normalize_input = bool(normalize_input)
        self.eps = float(eps)
        self.apply_projection = bool(apply_projection)

        n_ffts = [int(v) for v in n_ffts]
        if not n_ffts:
            raise ValueError("n_ffts must contain at least one resolution")
        if hop_lengths is None:
            hop_lengths = [max(1, n_fft // 4) for n_fft in n_ffts]
        if win_lengths is None:
            win_lengths = list(n_ffts)
        hop_lengths = [int(v) for v in hop_lengths]
        win_lengths = [int(v) for v in win_lengths]
        if len(hop_lengths) != len(n_ffts) or len(win_lengths) != len(n_ffts):
            raise ValueError("n_ffts, hop_lengths, and win_lengths must have matching lengths")

        self.branches = nn.ModuleList(
            [
                STFTMaskBranch(
                    n_srcs=self.n_srcs,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    hidden_dim=int(hidden_dim),
                    n_blocks=int(n_blocks),
                    kernel_size=int(kernel_size),
                    dilation_cycle=int(dilation_cycle),
                    dropout=float(dropout),
                    mask_bound=float(mask_bound),
                    mask_sum_constraint=bool(mask_sum_constraint),
                    mask_head_zero_init=bool(mask_head_zero_init),
                )
                for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths)
            ]
        )
        self.branch_weights = nn.Parameter(torch.zeros(len(self.branches)))
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.n_srcs,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=self.eps,
            detach_weights=mc_detach_weights,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected input (B,2,L), got {tuple(x.shape)}")
        batch_size, _, length = x.shape
        original_dtype = x.dtype
        if original_dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        mix_complex = torch.complex(x[:, 0], x[:, 1])
        if self.normalize_input:
            scale = mix_complex.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
            mix_complex = mix_complex / scale
        else:
            scale = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype)

        branch_outputs: List[torch.Tensor] = []
        for branch in self.branches:
            branch_output = branch(mix_complex, length=length, center=self.center)
            branch_outputs.append(branch_output * scale.unsqueeze(1).repeat(1, 2 * self.n_srcs, 1))

        weights = torch.softmax(self.branch_weights, dim=0)
        output = sum(weight * branch_output for weight, branch_output in zip(weights, branch_outputs))
        output = output.to(dtype=original_dtype)
        if self.apply_projection:
            output = self.mc_projection(output, x.to(dtype=output.dtype))
        return output
