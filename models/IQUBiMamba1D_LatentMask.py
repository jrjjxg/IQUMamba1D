"""Stage365 learned complex latent-mask separators."""

from __future__ import annotations

import math

import torch
from torch import nn

from models.IQUBiMamba1D_CoreUpgrades import (
    IQUBiMamba1D_IndependentComplexStateUniRepLK,
)


class IQUBiMamba1D_ComplexLatentMask(
    IQUBiMamba1D_IndependentComplexStateUniRepLK
):
    """Stage365 with fixed-slot real or complex latent masks.

    The existing Stage365 encoder and decoder remain shared. Masks are applied
    to every encoder scale so decoder skip paths cannot bypass source routing.
    """

    def __init__(
        self,
        *args,
        latent_mask_mode: str = "real",
        latent_mask_phase_limit: float = math.pi,
        latent_mask_eps: float = 1.0e-6,
        latent_mask_residual_weight: float = 0.1,
        latent_mask_mixture_weight: float = 0.1,
        latent_mask_residual_beta: float = 0.5,
        **kwargs,
    ) -> None:
        mode = str(latent_mask_mode).lower()
        if mode not in {"real", "complex_ratio", "complex_residual", "complex_conservation"}:
            raise ValueError(
                "latent_mask_mode must be real, complex_ratio, "
                "complex_residual or complex_conservation"
            )
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("latent-mask Stage365 requires num_classes")
        num_sources = int(num_classes) // 2
        if int(num_classes) != 2 * num_sources or num_sources not in (2, 3):
            raise ValueError("latent-mask Stage365 expects two or three complex sources")
        super().__init__(*args, **kwargs)
        if bool(getattr(self.decoder, "deep_supervision", False)):
            raise ValueError("latent-mask Stage365 requires deep_supervision=false")
        # The parent Stage365 decoder normally emits all source channels. A
        # masked slot must decode to exactly one complex waveform, so keep the
        # decoder trunk shared but replace its final heads with two-channel
        # slot heads.
        self.decoder.seg_layers = nn.ModuleList(
            [
                self.encoder.conv_op(layer.in_channels, 2, 1)
                for layer in self.decoder.seg_layers
            ]
        )
        self.latent_mask_mode = mode
        self.latent_mask_num_sources = num_sources
        self.latent_mask_has_residual = mode in {"complex_residual", "complex_conservation"}
        self.latent_mask_slots = num_sources + int(self.latent_mask_has_residual)
        self.latent_mask_phase_limit = float(abs(latent_mask_phase_limit))
        self.latent_mask_eps = float(latent_mask_eps)
        self.latent_mask_residual_weight = float(max(0.0, latent_mask_residual_weight))
        self.latent_mask_mixture_weight = float(max(0.0, latent_mask_mixture_weight))
        self.latent_mask_residual_beta = float(max(1.0e-6, latent_mask_residual_beta))

        mask_heads = []
        for channels in self.encoder.output_channels:
            channels = int(channels)
            if channels % 2:
                raise ValueError("latent-mask Stage365 requires even feature channels")
            # A real mask needs one logit per real feature channel. A complex
            # mask needs one real and one imaginary logit per complex pair.
            mask_heads.append(
                nn.Conv1d(channels, self.latent_mask_slots * channels, kernel_size=1)
            )
        self.latent_mask_heads = nn.ModuleList(mask_heads)
        for head in self.latent_mask_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _make_masks(self, features: torch.Tensor, head: nn.Module) -> torch.Tensor:
        batch, channels, length = features.shape
        slots = self.latent_mask_slots
        logits = head(features).reshape(batch, slots, channels, length)
        if self.latent_mask_mode == "real":
            return torch.softmax(logits, dim=1)

        complex_channels = channels // 2
        # Complex kernels are evaluated in FP32 for AMP portability; the
        # decoder can safely consume the resulting real-valued feature pair.
        raw = logits.float().reshape(batch, slots, complex_channels, 2, length)
        raw_real = raw[:, :, :, 0, :]
        raw_imag = raw[:, :, :, 1, :]
        if self.latent_mask_mode == "complex_conservation":
            weights = torch.softmax(raw_real, dim=1)
            phase = self.latent_mask_phase_limit * torch.tanh(raw_imag)
            masks = torch.complex(weights * torch.cos(phase), weights * torch.sin(phase))
            correction = (
                torch.ones_like(masks[:, :1]) - masks.sum(dim=1, keepdim=True)
            ) / float(slots)
            return masks + correction

        return torch.complex(torch.tanh(raw_real), torch.tanh(raw_imag))

    def _apply_mask(self, features: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        if self.latent_mask_mode == "real":
            return [features * masks[:, slot] for slot in range(self.latent_mask_slots)]
        batch, channels, length = features.shape
        complex_features = torch.complex(
            features[:, 0::2, :].float(), features[:, 1::2, :].float()
        )
        masked = []
        for slot in range(self.latent_mask_slots):
            value = masks[:, slot] * complex_features
            restored = torch.stack((value.real, value.imag), dim=2)
            masked.append(restored.reshape(batch, channels, length))
        return masked

    def _encode_skips(self, x: torch.Tensor) -> list[torch.Tensor]:
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage, (conv_stage, memory) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            stage_features = conv_stage(x)
            x = memory(stage_features)
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](stage_features, x)
            skips.append(x)
        return skips

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Latent-mask Stage365 expects [B, 2, L], got {tuple(x.shape)}")
        skips = self._encode_skips(x)
        masks_per_scale = [
            self._make_masks(features, head)
            for features, head in zip(skips, self.latent_mask_heads)
        ]
        masked_skips_per_slot = [[] for _ in range(self.latent_mask_slots)]
        for features, masks in zip(skips, masks_per_scale):
            masked = self._apply_mask(features, masks)
            for slot, value in enumerate(masked):
                masked_skips_per_slot[slot].append(value)

        decoded = [self.decoder(slot_skips) for slot_skips in masked_skips_per_slot]
        if any(isinstance(value, (tuple, list)) for value in decoded):
            raise RuntimeError("latent-mask Stage365 requires deep_supervision=false")
        sources = torch.cat(decoded[: self.latent_mask_num_sources], dim=1)
        if not self.latent_mask_has_residual:
            return sources

        residual = decoded[self.latent_mask_num_sources]
        auxiliary = {
            "residual_output": residual,
            "latent_masks": masks_per_scale,
        }
        if self.latent_mask_mode == "complex_conservation":
            errors = tuple(mask.sum(dim=1) - 1.0 for mask in masks_per_scale)
            auxiliary["latent_mask_sum_error"] = errors
            auxiliary["latent_mask_sum_max_error"] = torch.stack(
                [error.abs().amax() for error in errors]
            ).amax()
        return sources, auxiliary


__all__ = ["IQUMamba1D_ComplexLatentMask"]
