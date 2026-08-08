"""Complex attractor network for IQ blind source separation.

This model follows the deep-clustering / attractor separation paradigm:

    complex IQ STFT -> TF-bin embedding -> mixture-conditioned attractors
    -> source assignment masks -> complex iSTFT -> mixture consistency.

The attractors are inferred from the observed mixture only.  No modulation
labels, symbol timing, cyclic-frequency metadata, or clean-source side
information are used.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mixture_consistency_projection import WeightedMixtureConsistencyProjection1D


if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class ComplexAttractorCore(nn.Module):
    """Estimate source masks from TF-bin embedding and learned attractors."""

    def __init__(
        self,
        n_srcs: int = 2,
        embedding_dim: int = 64,
        hidden_dim: int = 96,
        rnn_hidden: int = 96,
        n_layers: int = 2,
        dropout: float = 0.0,
        attractor_temperature: float = 1.0,
        logit_scale_init: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.embedding_dim = int(embedding_dim)
        self.attractor_temperature = float(attractor_temperature)
        self.eps = float(eps)

        self.bin_encoder = nn.Sequential(
            nn.Linear(4, int(hidden_dim)),
            nn.PReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )
        self.temporal_rnns = nn.ModuleList(
            [
                nn.GRU(
                    self.embedding_dim,
                    int(rnn_hidden),
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.temporal_projs = nn.ModuleList(
            [nn.Linear(2 * int(rnn_hidden), self.embedding_dim) for _ in range(int(n_layers))]
        )
        self.freq_norm = nn.LayerNorm(self.embedding_dim)
        self.freq_rnn = nn.GRU(
            self.embedding_dim,
            int(rnn_hidden),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.freq_proj = nn.Linear(2 * int(rnn_hidden), self.embedding_dim)
        self.attractor_queries = nn.Parameter(torch.randn(self.n_srcs, self.embedding_dim) * 0.02)
        self.attractor_proj = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.n_srcs * self.embedding_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.dropout = nn.Dropout(float(dropout))

    def _tf_features(self, mix_spec: torch.Tensor) -> torch.Tensor:
        magnitude = mix_spec.abs().clamp_min(self.eps)
        logmag = torch.log1p(magnitude)
        phase = torch.atan2(mix_spec.imag, mix_spec.real)
        return torch.stack([mix_spec.real, mix_spec.imag, logmag, phase], dim=-1)

    def _embed_bins(self, mix_spec: torch.Tensor) -> torch.Tensor:
        # TF-bin embedding: (B,T,F,4) -> (B,T,F,E).
        features = self._tf_features(mix_spec)
        embedding = self.bin_encoder(features)
        batch, frames, freqs, dim = embedding.shape

        for rnn, proj in zip(self.temporal_rnns, self.temporal_projs):
            residual = embedding
            temporal_in = embedding.permute(0, 2, 1, 3).reshape(batch * freqs, frames, dim)
            temporal_out, _ = rnn(temporal_in)
            temporal_out = proj(temporal_out)
            temporal_out = temporal_out.reshape(batch, freqs, frames, dim).permute(0, 2, 1, 3)
            embedding = residual + self.dropout(temporal_out)

        residual = embedding
        freq_in = self.freq_norm(embedding).reshape(batch * frames, freqs, dim)
        freq_out, _ = self.freq_rnn(freq_in)
        freq_out = self.freq_proj(freq_out)
        freq_out = freq_out.reshape(batch, frames, freqs, dim)
        embedding = residual + self.dropout(freq_out)
        return F.normalize(embedding, dim=-1, eps=self.eps)

    def _estimate_attractors(self, embeddings: torch.Tensor, mix_spec: torch.Tensor) -> torch.Tensor:
        weights = mix_spec.abs().clamp_min(self.eps)
        weights = weights / weights.sum(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=(1, 2))
        offsets = self.attractor_proj(pooled).reshape(embeddings.size(0), self.n_srcs, self.embedding_dim)
        attractors = self.attractor_queries.unsqueeze(0) + offsets
        return F.normalize(attractors, dim=-1, eps=self.eps)

    def _source_masks(self, embeddings: torch.Tensor, attractors: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("btfe,bke->bktf", embeddings, attractors)
        logits = logits * self.logit_scale / max(self.attractor_temperature, self.eps)
        return torch.softmax(logits, dim=1)

    def forward(self, mix_spec: torch.Tensor) -> List[torch.Tensor]:
        if mix_spec.ndim != 3 or not torch.is_complex(mix_spec):
            raise ValueError(f"Expected complex (B,T,F) STFT, got {tuple(mix_spec.shape)}")

        embeddings = self._embed_bins(mix_spec)
        attractors = self._estimate_attractors(embeddings, mix_spec)
        masks = self._source_masks(embeddings, attractors)
        separated = masks.to(dtype=mix_spec.real.dtype).unsqueeze(-1) * torch.view_as_real(
            mix_spec.unsqueeze(1)
        )
        separated = torch.view_as_complex(separated.contiguous())
        return [separated[:, source] for source in range(self.n_srcs)]


class ComplexAttractorSeparator1D(nn.Module):
    """Complex STFT attractor separator with IQUMamba-compatible I/O."""

    def __init__(
        self,
        n_srcs: int = 2,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        center: bool = True,
        normalize_input: bool = True,
        embedding_dim: int = 64,
        hidden_dim: int = 96,
        rnn_hidden: int = 96,
        n_layers: int = 2,
        dropout: float = 0.0,
        attractor_temperature: float = 1.0,
        logit_scale_init: float = 0.0,
        eps: float = 1e-8,
        apply_projection: bool = True,
        mc_weight_mode: str = "uniform",
        mc_weight_power: float = 1.0,
        mc_min_weight: float = 0.0,
        mc_detach_weights: bool = False,
    ) -> None:
        super().__init__()
        self.n_srcs = int(n_srcs)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.center = bool(center)
        self.normalize_input = bool(normalize_input)
        self.eps = float(eps)
        self.apply_projection = bool(apply_projection)
        self.core = ComplexAttractorCore(
            n_srcs=self.n_srcs,
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            rnn_hidden=int(rnn_hidden),
            n_layers=int(n_layers),
            dropout=float(dropout),
            attractor_temperature=float(attractor_temperature),
            logit_scale_init=float(logit_scale_init),
            eps=self.eps,
        )
        self.mc_projection = WeightedMixtureConsistencyProjection1D(
            num_sources=self.n_srcs,
            weight_mode=mc_weight_mode,
            weight_power=mc_weight_power,
            min_weight=mc_min_weight,
            eps=self.eps,
            detach_weights=mc_detach_weights,
        )
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

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

        window = self.window.to(device=x.device, dtype=x.dtype)
        mix_spec = torch.stft(
            mix_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            onesided=False,
            return_complex=True,
        ).transpose(1, 2).contiguous()

        separated_specs = self.core(mix_spec)
        reconstructed = []
        for separated_spec in separated_specs:
            signal = torch.istft(
                separated_spec.transpose(1, 2).contiguous(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                onesided=False,
                length=length,
                return_complex=True,
            )
            reconstructed.append(signal * scale)

        output = torch.stack(
            [torch.stack([signal.real, signal.imag], dim=1) for signal in reconstructed],
            dim=1,
        ).reshape(batch_size, 2 * self.n_srcs, length)
        output = output.to(dtype=original_dtype)
        if self.apply_projection:
            output = self.mc_projection(output, x.to(dtype=output.dtype))
        return output
