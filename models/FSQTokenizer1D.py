"""Finite Scalar Quantization (FSQ) tokenizer for communication waveforms.

Stage 296 support module. A small convolutional autoencoder is pretrained on
clean single-source I/Q waveforms (all modulations pooled, no labels). The
bottleneck is quantized with FSQ (Mentzer et al., 2023): each latent dim is
bounded by tanh and rounded to a small number of levels, so the implicit
codebook is the product grid of per-dim levels. There is no learned codebook,
no EMA, and no collapse mode.

After pretraining the tokenizer is frozen and used as a *learned,
modulation-agnostic communication prior*: during separator training the
predicted source waveforms are pushed (via cross entropy in the FSQ lattice,
see util/fsq_token_prior.py) to encode to the same discrete tokens as the
clean targets. This transfers the "discrete symbol structure" supervision of
the RF Transformer (arXiv:2603.09201) to a non-autoregressive separator with
zero inference-time cost: the tokenizer is only used inside the training loss.

Design notes:
- Per-sample RMS normalization inside encode/decode makes tokens invariant to
  source power (robust to SIR variation); the scale is reapplied on decode.
- Fully convolutional: any input length divisible by ``downsample`` works,
  so 4096/8192/16384/32768-point frames share one tokenizer.
- ``encode_bounded`` returns the *continuous* bounded latent (before
  rounding) and keeps gradients, which is what the CE loss needs from the
  separator side; ``encode_indices`` returns integer token indices per dim,
  which is what the CE loss needs from the target side.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "FSQ",
    "FSQTokenizer1D",
    "save_fsq_tokenizer",
    "load_fsq_tokenizer",
]


class FSQ(nn.Module):
    """Finite scalar quantizer over a per-dim level grid.

    For dim i with L_i levels, the bounded latent lives in
    [-(L_i-1)/2, (L_i-1)/2] and is rounded to the grid
    {j - (L_i-1)/2 : j = 0..L_i-1} (a half-integer grid when L_i is even).
    """

    def __init__(self, levels: Sequence[int]):
        super().__init__()
        levels = [int(l) for l in levels]
        if len(levels) == 0 or any(l < 2 for l in levels):
            raise ValueError(f"FSQ levels must all be >= 2, got {levels}")
        self.levels = levels
        half = torch.tensor([(l - 1) / 2.0 for l in levels], dtype=torch.float32)
        offset = torch.tensor(
            [0.5 if l % 2 == 0 else 0.0 for l in levels], dtype=torch.float32
        )
        self.register_buffer("half_width", half, persistent=False)
        self.register_buffer("offset", offset, persistent=False)

    @property
    def num_dims(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        size = 1
        for l in self.levels:
            size *= l
        return size

    def positions(self, dim: int, device=None, dtype=None) -> torch.Tensor:
        """Grid positions for one latent dim, shape [levels[dim]]."""
        l = self.levels[dim]
        half = (l - 1) / 2.0
        pos = torch.arange(l, device=device, dtype=dtype or torch.float32) - half
        return pos

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Map unbounded latent [B, D, T] into the level range per dim."""
        half = self.half_width.view(1, -1, 1).to(z.dtype)
        return torch.tanh(z) * half

    def quantize(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Round bounded latent to the grid.

        Args:
            y: bounded latent [B, D, T] (output of :meth:`bound`).
        Returns:
            (quantized latent with straight-through gradient, integer indices
            [B, D, T] in {0..levels[d]-1}).
        """
        offset = self.offset.view(1, -1, 1).to(y.dtype)
        half = self.half_width.view(1, -1, 1).to(y.dtype)
        q = torch.round(y - offset) + offset
        q = torch.clamp(q, min=-half, max=half)
        indices = torch.round(q + half).long()
        q = y + (q - y).detach()  # straight-through estimator
        return q, indices

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.bound(z)
        return self.quantize(y)


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of ``channels`` that is <= ``max_groups``."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _res_block(channels: int, kernel_size: int = 3) -> nn.Module:
    padding = kernel_size // 2

    class _ResBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.GroupNorm(_num_groups(channels), channels),
                nn.SiLU(),
                nn.Conv1d(channels, channels, kernel_size, padding=padding),
                nn.GroupNorm(_num_groups(channels), channels),
                nn.SiLU(),
                nn.Conv1d(channels, channels, kernel_size, padding=padding),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.body(x)

    return _ResBlock()


class FSQTokenizer1D(nn.Module):
    """Small convolutional FSQ autoencoder for [B, 2, L] I/Q waveforms."""

    def __init__(
        self,
        levels: Sequence[int] = (8, 5, 5, 5),
        base_channels: int = 32,
        rms_eps: float = 1e-6,
    ):
        super().__init__()
        self.levels = [int(l) for l in levels]
        self.base_channels = int(base_channels)
        self.rms_eps = float(rms_eps)
        self.downsample = 8  # three stride-2 stages below

        c = self.base_channels
        latent_dim = len(self.levels)

        self.encoder = nn.Sequential(
            nn.Conv1d(2, c, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(_num_groups(c), c),
            nn.SiLU(),
            nn.Conv1d(c, 2 * c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_num_groups(2 * c), 2 * c),
            nn.SiLU(),
            nn.Conv1d(2 * c, 2 * c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_num_groups(2 * c), 2 * c),
            nn.SiLU(),
            _res_block(2 * c),
            nn.Conv1d(2 * c, latent_dim, kernel_size=3, padding=1),
        )
        self.fsq = FSQ(self.levels)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 2 * c, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(2 * c), 2 * c),
            nn.SiLU(),
            _res_block(2 * c),
            nn.ConvTranspose1d(2 * c, 2 * c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(2 * c), 2 * c),
            nn.SiLU(),
            nn.ConvTranspose1d(2 * c, c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(c), c),
            nn.SiLU(),
            nn.ConvTranspose1d(c, c, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(_num_groups(c), c),
            nn.SiLU(),
            nn.Conv1d(c, 2, kernel_size=7, padding=3),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(
                f"FSQTokenizer1D expects [B, 2, L] I/Q input, got {tuple(x.shape)}"
            )
        if x.size(-1) % self.downsample != 0:
            raise ValueError(
                f"Input length {x.size(-1)} must be divisible by {self.downsample}"
            )

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.pow(2).mean(dim=(1, 2), keepdim=True) + self.rms_eps)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def encode_bounded(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous bounded latent [B, D, L/downsample]; keeps gradients."""
        self._check_input(x)
        x = x / self._rms(x)
        z = self.encoder(x)
        return self.fsq.bound(z)

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Integer token indices [B, D, L/downsample] (no gradient path)."""
        y = self.encode_bounded(x)
        _, indices = self.fsq.quantize(y)
        return indices

    def decode_from_quantized(
        self, q: torch.Tensor, rms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        recon = self.decoder(q)
        if rms is not None:
            recon = recon * rms
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Autoencode. Returns (reconstruction [B,2,L], aux dict)."""
        self._check_input(x)
        rms = self._rms(x)
        x_norm = x / rms
        z = self.encoder(x_norm)
        y = self.fsq.bound(z)
        q, indices = self.fsq.quantize(y)
        recon = self.decoder(q) * rms
        return recon, {"bounded": y, "quantized": q, "indices": indices, "rms": rms}


def save_fsq_tokenizer(path: str, model: FSQTokenizer1D, extra_meta: Optional[dict] = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "class": "FSQTokenizer1D",
            "levels": list(model.levels),
            "base_channels": model.base_channels,
            "downsample": model.downsample,
            "rms_eps": model.rms_eps,
        },
    }
    if extra_meta:
        payload["meta"].update(extra_meta)
    torch.save(payload, path)


def load_fsq_tokenizer(path: str, map_location="cpu") -> FSQTokenizer1D:
    payload = torch.load(path, map_location=map_location)
    if "meta" not in payload or "state_dict" not in payload:
        raise ValueError(
            f"{path} is not an FSQ tokenizer checkpoint (expected 'meta' and 'state_dict')"
        )
    meta = payload["meta"]
    model = FSQTokenizer1D(
        levels=meta["levels"],
        base_channels=meta.get("base_channels", 32),
        rms_eps=meta.get("rms_eps", 1e-6),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
