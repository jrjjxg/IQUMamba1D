"""Stages 369/370: physically supervised multi-source Radio Transformer.

This module keeps the original four-level Stage-4 U-Net while replacing the
mixture-level latent synchronization coordinates of Stage 366 with per-source
parameters.  The predicted CFO, phase, fractional timing, SPS and residual
phase drift generate differentiable radio-domain views before separation and
also condition every encoder scale through bounded FiLM.
"""

from __future__ import annotations

from typing import Iterable, List, Type

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_BlindSyncFactorized import BlindSyncEvidence
from models.IQUMamba1D_SyncConditioned import SyncFiLM1D, _rounded_backbone_length


class PerSourceSynchronizationEstimator(nn.Module):
    """Predict one calibrated synchronization state for every source slot."""

    def __init__(
        self,
        evidence_dim: int,
        num_sources: int,
        *,
        hidden: int = 64,
        sps_candidates: Iterable[int] = (8, 10, 14, 16, 20, 25, 32, 40),
        snr_min_db: float = -10.0,
        snr_max_db: float = 30.0,
        max_cfo_cycles_per_sample: float = 1e-4,
        max_phase_drift_rad_per_sample: float = 1e-4,
        sps_temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_sources = int(num_sources)
        if self.num_sources < 1:
            raise ValueError("num_sources must be positive")
        hidden = max(16, int(hidden))
        candidates = tuple(sorted({max(1, int(value)) for value in sps_candidates}))
        if not candidates:
            raise ValueError("sps_candidates must not be empty")
        if float(snr_max_db) <= float(snr_min_db):
            raise ValueError("snr_max_db must exceed snr_min_db")

        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        self.max_cfo_cycles_per_sample = float(max_cfo_cycles_per_sample)
        self.max_phase_drift_rad_per_sample = float(max_phase_drift_rad_per_sample)
        self.sps_temperature = max(float(sps_temperature), 1e-4)
        self.eps = float(eps)
        self.register_buffer(
            "sps_candidates",
            torch.tensor(candidates, dtype=torch.float32),
            persistent=True,
        )

        self.raw_iq_encoder = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=hidden,
                bias=False,
            ),
            nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
        )
        self.evidence_norm = nn.LayerNorm(int(evidence_dim))
        self.trunk = nn.Sequential(
            nn.Linear(2 * hidden + int(evidence_dim), 2 * hidden),
            nn.SiLU(),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
        )
        self.source_queries = nn.Parameter(torch.empty(self.num_sources, hidden))
        nn.init.normal_(self.source_queries, mean=0.0, std=0.02)

        self.snr_head = nn.Linear(hidden, 1)
        self._slices = {
            "cfo": slice(0, 1),
            "phase": slice(1, 3),
            "timing": slice(3, 4),
            "sps": slice(4, 4 + len(candidates)),
            "drift": slice(4 + len(candidates), 5 + len(candidates)),
        }
        self.source_head = nn.Linear(hidden, 5 + len(candidates))
        nn.init.normal_(self.snr_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.snr_head.bias)
        nn.init.normal_(self.source_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.source_head.bias)
        with torch.no_grad():
            self.source_head.bias[self._slices["phase"]] = torch.tensor([1.0, 0.0])

        # One global SNR plus a physical state for each source slot.
        self.per_source_condition_dim = 5 + len(candidates)
        self.condition_dim = 1 + self.num_sources * self.per_source_condition_dim

    def forward(self, x: torch.Tensor, evidence: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(
                f"PerSourceSynchronizationEstimator expects [B, 2, L], got {tuple(x.shape)}"
            )
        local = self.raw_iq_encoder(x.float())
        local_summary = torch.cat([local.mean(dim=-1), local.amax(dim=-1)], dim=-1)
        latent = self.trunk(
            torch.cat([local_summary, self.evidence_norm(evidence.float())], dim=-1)
        )
        source_latent = latent.unsqueeze(1) + self.source_queries.unsqueeze(0)
        raw = self.source_head(source_latent)

        snr_unit = torch.sigmoid(self.snr_head(latent))
        snr_prediction = self.snr_min_db + (
            self.snr_max_db - self.snr_min_db
        ) * snr_unit
        cfo_unit = torch.tanh(raw[..., self._slices["cfo"]])
        cfo = cfo_unit * self.max_cfo_cycles_per_sample
        phase_vector = F.normalize(
            raw[..., self._slices["phase"]], dim=-1, eps=self.eps
        )
        # [-1, 1] denotes one centered fractional-symbol timing interval.
        timing_unit = torch.tanh(raw[..., self._slices["timing"]])
        sps_logits = raw[..., self._slices["sps"]]
        sps_probabilities = torch.softmax(
            sps_logits / self.sps_temperature, dim=-1
        )
        drift_unit = torch.tanh(raw[..., self._slices["drift"]])
        phase_drift = drift_unit * self.max_phase_drift_rad_per_sample

        per_source_condition = torch.cat(
            [cfo_unit, phase_vector, timing_unit, sps_probabilities, drift_unit],
            dim=-1,
        )
        condition = torch.cat(
            [2.0 * snr_unit - 1.0, per_source_condition.flatten(1)], dim=-1
        ).to(dtype=x.dtype)
        candidates_tensor = self.sps_candidates.to(
            device=x.device, dtype=sps_probabilities.dtype
        )
        sps_estimate = (sps_probabilities * candidates_tensor).sum(dim=-1)
        auxiliary = {
            "snr_prediction": snr_prediction.squeeze(-1).to(dtype=x.dtype),
            "cfo_cycles_per_sample": cfo.squeeze(-1).to(dtype=x.dtype),
            "phase_vector": phase_vector.to(dtype=x.dtype),
            "phase_rad": torch.atan2(
                phase_vector[..., 1], phase_vector[..., 0]
            ).to(dtype=x.dtype),
            "timing_offset_unit": timing_unit.squeeze(-1).to(dtype=x.dtype),
            "sps_logits": sps_logits.to(dtype=x.dtype),
            "sps_probabilities": sps_probabilities.to(dtype=x.dtype),
            "sps_estimate": sps_estimate.to(dtype=x.dtype),
            "sps_candidates": candidates_tensor.to(dtype=x.dtype),
            "phase_drift_rad_per_sample": phase_drift.squeeze(-1).to(dtype=x.dtype),
            "sync_condition": condition,
        }
        return condition, auxiliary


class MultiSourceRadioTransformer(nn.Module):
    """Create and softly fuse differentiable per-source synchronization views."""

    def __init__(self, num_sources: int, residual_scale_init: float = 0.10):
        super().__init__()
        self.num_sources = int(num_sources)
        self.view_fusion = nn.Conv1d(2 * self.num_sources, 2, kernel_size=1, bias=False)
        nn.init.zeros_(self.view_fusion.weight)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    @staticmethod
    def _fractional_shift(signal: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
        length = signal.size(-1)
        frequencies = torch.fft.fftfreq(
            length, d=1.0, device=signal.device, dtype=signal.real.dtype
        )
        phase = torch.exp(
            2j * torch.pi * delay_samples.unsqueeze(-1) * frequencies.view(1, 1, -1)
        )
        return torch.fft.ifft(torch.fft.fft(signal, dim=-1) * phase, dim=-1)

    def forward(self, x: torch.Tensor, auxiliary: dict) -> tuple[torch.Tensor, torch.Tensor]:
        real = x[:, 0].float().unsqueeze(1)
        imag = x[:, 1].float().unsqueeze(1)
        mixture = torch.complex(real, imag).expand(-1, self.num_sources, -1)
        length = x.size(-1)
        time = torch.arange(length, device=x.device, dtype=torch.float32).view(1, 1, -1)

        cfo = auxiliary["cfo_cycles_per_sample"].float().unsqueeze(-1)
        phase = auxiliary["phase_rad"].float().unsqueeze(-1)
        drift = auxiliary["phase_drift_rad_per_sample"].float().unsqueeze(-1)
        derotation = torch.exp(-1j * (2.0 * torch.pi * cfo * time + phase + drift * time))
        transformed = mixture * derotation

        timing_unit = auxiliary["timing_offset_unit"].float()
        sps = auxiliary["sps_estimate"].float()
        delay_samples = 0.5 * timing_unit * sps
        transformed = self._fractional_shift(transformed, delay_samples)

        views = torch.stack([transformed.real, transformed.imag], dim=2)
        views = views.reshape(x.size(0), 2 * self.num_sources, length).to(dtype=x.dtype)
        scale = torch.tanh(self.residual_scale).to(dtype=x.dtype)
        fused = x + scale * self.view_fusion(views)
        return fused, views


class IQUMamba1D_PhysicalSyncRTN(nn.Module):
    """Four-level Stage-4 with supervised per-source RTN and feature distillation."""

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
        sync_hidden: int = 64,
        sync_lags: Iterable[int] = (1, 2, 4, 8, 16),
        sync_sps_candidates: Iterable[int] = (8, 10, 14, 16, 20, 25, 32, 40),
        sync_snr_min_db: float = -10.0,
        sync_snr_max_db: float = 30.0,
        sync_max_cfo_cycles_per_sample: float = 1e-4,
        sync_max_phase_drift_rad_per_sample: float = 1e-4,
        sync_sps_temperature: float = 1.0,
        sync_film_max_delta: float = 0.10,
        rtn_residual_scale_init: float = 0.10,
        sync_eps: float = 1e-6,
    ):
        super().__init__()
        if int(n_stages) != 4 or len(features_per_stage) != 4:
            raise ValueError("Stages 369/370 keep the original four-level Stage-4 U-Net")
        if input_channels != 2 or int(num_classes) % 2 != 0:
            raise ValueError("PhysicalSyncRTN expects one IQ mixture and 2*K output channels")
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.num_sources = int(num_classes) // 2
        self.nominal_length = _rounded_backbone_length(input_size, strides)
        self.sync_evidence = BlindSyncEvidence(lags=sync_lags, eps=sync_eps)
        self.sync_estimator = PerSourceSynchronizationEstimator(
            self.sync_evidence.num_stats,
            self.num_sources,
            hidden=sync_hidden,
            sps_candidates=sync_sps_candidates,
            snr_min_db=sync_snr_min_db,
            snr_max_db=sync_snr_max_db,
            max_cfo_cycles_per_sample=sync_max_cfo_cycles_per_sample,
            max_phase_drift_rad_per_sample=sync_max_phase_drift_rad_per_sample,
            sps_temperature=sync_sps_temperature,
            eps=sync_eps,
        )
        self.radio_transformer = MultiSourceRadioTransformer(
            self.num_sources, residual_scale_init=rtn_residual_scale_init
        )
        self.backbone = IQUMamba1D(
            input_size=self.nominal_length,
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
        self.sync_film = nn.ModuleList(
            SyncFiLM1D(
                self.sync_estimator.condition_dim,
                channels,
                max_delta=sync_film_max_delta,
            )
            for channels in features_per_stage
        )

    def _conditioned_backbone(self, x: torch.Tensor, condition: torch.Tensor):
        encoder = self.backbone.encoder
        features = encoder.stem(x) if encoder.stem is not None else x
        skips = []
        for stage, memory, film in zip(
            encoder.stages, encoder.mamba_layers, self.sync_film
        ):
            features = film(memory(stage(features)), condition)
            skips.append(features)
        if len(skips) != 4:
            raise RuntimeError(f"PhysicalSyncRTN expected four encoder scales, got {len(skips)}")
        return self.backbone.decoder(skips), skips[-1]

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"PhysicalSyncRTN expects [B, 2, L], got {tuple(x.shape)}")
        original_length = x.size(-1)
        if original_length > self.nominal_length:
            raise ValueError(
                f"input length {original_length} exceeds configured length {self.nominal_length}"
            )

        evidence = self.sync_evidence(x)
        condition, auxiliary = self.sync_estimator(x, evidence)
        transformed, _views = self.radio_transformer(x, auxiliary)
        if original_length < self.nominal_length:
            transformed = F.pad(
                transformed,
                (0, self.nominal_length - original_length),
                mode="replicate",
            )
        separation, bottleneck = self._conditioned_backbone(transformed, condition)
        if isinstance(separation, (list, tuple)):
            separation = type(separation)(item[..., :original_length] for item in separation)
        else:
            separation = separation[..., :original_length]
        auxiliary["sync_evidence"] = evidence
        auxiliary["distillation_feature"] = bottleneck
        auxiliary["rtn_residual_scale"] = torch.tanh(
            self.radio_transformer.residual_scale
        )
        return separation, auxiliary
