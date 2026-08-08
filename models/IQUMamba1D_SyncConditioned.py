"""Stage 366: cross-SNR distilled, synchronization-conditioned Stage-4 IQUMamba.

The synchronization head exposes interpretable mixture-level estimates and
uses only those estimates to FiLM-condition every encoder scale.  It does not
hard-de-rotate the input: a mixture can contain a different CFO for every
source, so one global correction can destroy useful relative phase structure.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Type

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_BlindSyncFactorized import BlindSyncEvidence


def _rounded_backbone_length(input_size: int, strides: Sequence[int]) -> int:
    total_stride = 1
    for stride in strides:
        total_stride *= max(1, int(stride))
    return ((int(input_size) + total_stride - 1) // total_stride) * total_stride


class SynchronizationParameterEstimator(nn.Module):
    """Estimate a compact synchronization state from one received IQ mixture."""

    def __init__(
        self,
        evidence_dim: int,
        *,
        hidden: int = 48,
        sps_candidates: Iterable[int] = (8, 10, 16, 20, 32, 40),
        snr_min_db: float = -10.0,
        snr_max_db: float = 30.0,
        max_cfo_cycles_per_sample: float = 0.25,
        max_phase_drift_rad_per_sample: float = 0.05,
        sps_temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden = max(8, int(hidden))
        candidates = tuple(sorted({max(1, int(value)) for value in sps_candidates}))
        if not candidates:
            raise ValueError("sps_candidates must not be empty")
        if float(snr_max_db) <= float(snr_min_db):
            raise ValueError("snr_max_db must exceed snr_min_db")
        if float(max_cfo_cycles_per_sample) <= 0.0:
            raise ValueError("max_cfo_cycles_per_sample must be positive")
        if float(max_phase_drift_rad_per_sample) <= 0.0:
            raise ValueError("max_phase_drift_rad_per_sample must be positive")

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
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2, groups=hidden, bias=False),
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

        # SNR, CFO, phase (cos/sin), timing, SPS logits, and residual phase drift.
        self._slices = {
            "snr": slice(0, 1),
            "cfo": slice(1, 2),
            "phase": slice(2, 4),
            "timing": slice(4, 5),
            "sps": slice(5, 5 + len(candidates)),
            "drift": slice(5 + len(candidates), 6 + len(candidates)),
        }
        self.head = nn.Linear(hidden, 6 + len(candidates))
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[self._slices["phase"]] = torch.tensor([1.0, 0.0])

        # Normalized SNR, CFO, phase-vector, timing, SPS posterior and drift.
        self.condition_dim = 6 + len(candidates)

    def forward(self, x: torch.Tensor, evidence: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"SynchronizationParameterEstimator expects [B, 2, L], got {tuple(x.shape)}")
        local = self.raw_iq_encoder(x.float())
        local_summary = torch.cat(
            [local.mean(dim=-1), local.amax(dim=-1)],
            dim=-1,
        )
        latent = self.trunk(
            torch.cat([local_summary, self.evidence_norm(evidence.float())], dim=-1)
        )
        raw = self.head(latent)

        snr_unit = torch.sigmoid(raw[:, self._slices["snr"]])
        snr_prediction = self.snr_min_db + (self.snr_max_db - self.snr_min_db) * snr_unit
        cfo_unit = torch.tanh(raw[:, self._slices["cfo"]])
        cfo = cfo_unit * self.max_cfo_cycles_per_sample
        phase_vector = F.normalize(
            raw[:, self._slices["phase"]],
            dim=-1,
            eps=self.eps,
        )
        timing_unit = torch.sigmoid(raw[:, self._slices["timing"]])
        sps_logits = raw[:, self._slices["sps"]]
        sps_probabilities = torch.softmax(sps_logits / self.sps_temperature, dim=-1)
        drift_unit = torch.tanh(raw[:, self._slices["drift"]])
        phase_drift = drift_unit * self.max_phase_drift_rad_per_sample

        condition = torch.cat(
            [
                2.0 * snr_unit - 1.0,
                cfo_unit,
                phase_vector,
                2.0 * timing_unit - 1.0,
                sps_probabilities,
                drift_unit,
            ],
            dim=-1,
        ).to(dtype=x.dtype)
        candidates = self.sps_candidates.to(device=x.device, dtype=sps_probabilities.dtype)
        auxiliary = {
            "snr_prediction": snr_prediction.squeeze(-1).to(dtype=x.dtype),
            "cfo_cycles_per_sample": cfo.squeeze(-1).to(dtype=x.dtype),
            "phase_vector": phase_vector.to(dtype=x.dtype),
            "phase_rad": torch.atan2(phase_vector[:, 1], phase_vector[:, 0]).to(dtype=x.dtype),
            "timing_offset_unit": timing_unit.squeeze(-1).to(dtype=x.dtype),
            "sps_logits": sps_logits.to(dtype=x.dtype),
            "sps_probabilities": sps_probabilities.to(dtype=x.dtype),
            "sps_estimate": (sps_probabilities * candidates).sum(dim=-1).to(dtype=x.dtype),
            "phase_drift_rad_per_sample": phase_drift.squeeze(-1).to(dtype=x.dtype),
            "sync_condition": condition,
        }
        return condition, auxiliary


class SyncFiLM1D(nn.Module):
    """Zero-initialized feature-wise affine modulation for one encoder scale."""

    def __init__(self, condition_dim: int, channels: int, max_delta: float = 0.10):
        super().__init__()
        self.channels = int(channels)
        self.max_delta = float(max_delta)
        self.projection = nn.Linear(int(condition_dim), 2 * self.channels)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(condition.float()).chunk(2, dim=-1)
        gamma = self.max_delta * torch.tanh(gamma).to(dtype=features.dtype).unsqueeze(-1)
        beta = self.max_delta * torch.tanh(beta).to(dtype=features.dtype).unsqueeze(-1)
        return features * (1.0 + gamma) + beta


class IQUMamba1D_SyncConditioned(nn.Module):
    """Unchanged four-level Stage-4 backbone with explicit sync FiLM controls."""

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
        sync_hidden: int = 48,
        sync_lags: Iterable[int] = (1, 2, 4, 8, 16),
        sync_sps_candidates: Iterable[int] = (8, 10, 16, 20, 32, 40),
        sync_snr_min_db: float = -10.0,
        sync_snr_max_db: float = 30.0,
        sync_max_cfo_cycles_per_sample: float = 0.25,
        sync_max_phase_drift_rad_per_sample: float = 0.05,
        sync_sps_temperature: float = 1.0,
        sync_film_max_delta: float = 0.10,
        sync_eps: float = 1e-6,
    ):
        super().__init__()
        if int(n_stages) != 4 or len(features_per_stage) != 4:
            raise ValueError("Stage 366 deliberately keeps the original four-level Stage-4 U-Net")
        if input_channels != 2:
            raise ValueError("Stage 366 requires one complex IQ mixture with two real channels")
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.nominal_length = _rounded_backbone_length(input_size, strides)
        self.sync_evidence = BlindSyncEvidence(lags=sync_lags, eps=sync_eps)
        self.sync_estimator = SynchronizationParameterEstimator(
            self.sync_evidence.num_stats,
            hidden=sync_hidden,
            sps_candidates=sync_sps_candidates,
            snr_min_db=sync_snr_min_db,
            snr_max_db=sync_snr_max_db,
            max_cfo_cycles_per_sample=sync_max_cfo_cycles_per_sample,
            max_phase_drift_rad_per_sample=sync_max_phase_drift_rad_per_sample,
            sps_temperature=sync_sps_temperature,
            eps=sync_eps,
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
        for stage_index, (stage, memory, film) in enumerate(
            zip(encoder.stages, encoder.mamba_layers, self.sync_film)
        ):
            features = stage(features)
            features = memory(features)
            features = film(features, condition)
            skips.append(features)
        if len(skips) != 4:
            raise RuntimeError(f"Stage 366 expected four encoder scales, got {len(skips)}")
        return self.backbone.decoder(skips)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Stage 366 expects [B, 2, L], got {tuple(x.shape)}")
        original_length = x.size(-1)
        if original_length > self.nominal_length:
            raise ValueError(
                f"input length {original_length} exceeds configured length {self.nominal_length}"
            )

        evidence = self.sync_evidence(x)
        condition, auxiliary = self.sync_estimator(x, evidence)
        backbone_input = x
        if original_length < self.nominal_length:
            backbone_input = F.pad(
                backbone_input,
                (0, self.nominal_length - original_length),
                mode="replicate",
            )
        separation = self._conditioned_backbone(backbone_input, condition)
        if isinstance(separation, (list, tuple)):
            separation = type(separation)(item[..., :original_length] for item in separation)
        else:
            separation = separation[..., :original_length]
        auxiliary["sync_evidence"] = evidence
        return separation, auxiliary
