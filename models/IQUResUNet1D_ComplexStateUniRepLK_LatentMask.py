"""Stage377: Stage56 + independent complex-state BiMamba + UniRepLK + mask.

This is the strict no-ASC counterpart of Stage371.  The encoder keeps the
Stage365 placement of the independent complex-state bidirectional SSMs and
parallel UniRepLK deltas, while the decoder is Stage56's plain skip decoder.
Real simplex masks are applied at every encoder scale before a single shared
decoder call for all source slots.
"""

from __future__ import annotations

import torch
from torch import nn

from models.IQUBiMamba1D_CoreUpgrades import (
    IQUBiMamba1D_IndependentComplexStateUniRepLK,
)
from models.IQUResUNet1D_NoASC import PlainSkipDecoder


class IQUResUNet1D_ComplexStateUniRepLK_LatentMask(
    IQUBiMamba1D_IndependentComplexStateUniRepLK
):
    """Stage56 decoder with the Stage365 encoder and a real simplex mask."""

    def __init__(
        self,
        *args,
        latent_mask_mode: str = "real",
        latent_mask_eps: float = 1.0e-6,
        **kwargs,
    ) -> None:
        mode = str(latent_mask_mode).lower()
        if mode != "real":
            raise ValueError(
                "Stage377 currently supports only the real simplex mask; "
                f"got latent_mask_mode={latent_mask_mode!r}"
            )
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("Stage377 requires num_classes")
        num_sources = int(num_classes) // 2
        if int(num_classes) != 2 * num_sources or num_sources not in (2, 3):
            raise ValueError("Stage377 expects two or three complex source slots")
        if bool(kwargs.get("deep_supervision", False)):
            raise ValueError("Stage377 requires deep_supervision=false")

        # The parent builds the Stage365 encoder (complex-state BiMamba plus
        # UniRepLK). Replace only its ASC decoder with Stage56's plain decoder.
        super().__init__(*args, **kwargs)
        self.decoder = PlainSkipDecoder(
            encoder=self.encoder,
            num_classes=2,
            n_conv_per_stage=kwargs["n_conv_per_stage_decoder"],
            deep_supervision=False,
        )

        self.latent_mask_mode = mode
        self.latent_mask_num_sources = num_sources
        self.latent_mask_eps = float(latent_mask_eps)
        self.latent_mask_heads = nn.ModuleList(
            [
                nn.Conv1d(
                    int(channels),
                    num_sources * int(channels),
                    kernel_size=1,
                )
                for channels in self.encoder.output_channels
            ]
        )
        for head in self.latent_mask_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _encode_skips(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the Stage365 encoder and preserve its RF branch outputs."""

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

    def _make_masks(
        self, features: torch.Tensor, head: nn.Module
    ) -> torch.Tensor:
        """Predict one source-simplex mask per real latent feature channel."""

        batch, channels, length = features.shape
        logits = head(features).reshape(
            batch, self.latent_mask_num_sources, channels, length
        )
        return torch.softmax(logits, dim=1)

    def _flatten_source_slots(
        self,
        skips: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Merge source slots into batch for one shared plain decoder call."""

        batch = skips[0].shape[0]
        slots = self.latent_mask_num_sources
        flattened = []
        for features, scale_masks in zip(skips, masks):
            masked = features.unsqueeze(1) * scale_masks
            flattened.append(masked.reshape(batch * slots, *features.shape[1:]))
        return flattened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(
                "Stage377 expects I/Q input with shape [B, 2, L], "
                f"got {tuple(x.shape)}"
            )

        skips = self._encode_skips(x)
        masks = [
            self._make_masks(features, head)
            for features, head in zip(skips, self.latent_mask_heads)
        ]
        decoded = self.decoder(self._flatten_source_slots(skips, masks))
        batch_slots, _, length = decoded.shape
        if batch_slots % self.latent_mask_num_sources != 0:
            raise RuntimeError(
                "Stage377 decoder batch is not divisible by source slots: "
                f"batch={batch_slots}, slots={self.latent_mask_num_sources}"
            )
        batch = batch_slots // self.latent_mask_num_sources
        return decoded.reshape(batch, self.latent_mask_num_sources * 2, length)


__all__ = ["IQUResUNet1D_ComplexStateUniRepLK_LatentMask"]
