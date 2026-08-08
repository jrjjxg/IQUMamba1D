"""Stage376: Stage56 with a full-resolution latent simplex mask route.

The convolutional encoder and plain skip decoder are inherited from Stage56.
At every encoder scale, a small 1x1 head predicts one real simplex mask per
source slot. The masked skip pyramids are then decoded by the same decoder
weights, matching Stage371's routing while removing Mamba and ASC entirely.
"""

from __future__ import annotations

import torch
from torch import nn

from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC


class IQUResUNet1D_NoASC_LatentMask(IQUResUNet1D_NoASC):
    """Stage56 + real latent simplex masks at all encoder scales."""

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
                "Stage376 currently reproduces Stage371's real latent mask; "
                f"got latent_mask_mode={latent_mask_mode!r}"
            )
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("Stage376 requires num_classes")
        num_sources = int(num_classes) // 2
        if int(num_classes) != 2 * num_sources or num_sources not in (2, 3):
            raise ValueError("Stage376 expects two or three complex source slots")

        super().__init__(*args, **kwargs)
        if bool(getattr(self.decoder, "deep_supervision", False)):
            raise ValueError("Stage376 requires deep_supervision=false")

        # Each source slot is decoded to one I/Q pair. The decoder trunk and
        # its plain skip path remain shared across all slots.
        self.decoder.seg_layers = nn.ModuleList(
            [
                self.encoder.conv_op(layer.in_channels, 2, kernel_size=1)
                for layer in self.decoder.seg_layers
            ]
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
        """Return the Stage56 convolutional skip pyramid."""

        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        return skips

    def _make_masks(
        self, features: torch.Tensor, head: nn.Module
    ) -> torch.Tensor:
        """Predict a source-simplex mask for one encoder scale."""

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
        """Merge source slots into the batch for one shared decoder call."""

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
                "Stage376 expects I/Q input with shape [B, 2, L], "
                f"got {tuple(x.shape)}"
            )

        skips = self._encode_skips(x)
        masks = [
            self._make_masks(features, head)
            for features, head in zip(skips, self.latent_mask_heads)
        ]
        decoded = self.decoder(self._flatten_source_slots(skips, masks))
        batch, _, length = decoded.shape
        return decoded.reshape(batch // self.latent_mask_num_sources,
                              self.latent_mask_num_sources * 2,
                              length)


__all__ = ["IQUResUNet1D_NoASC_LatentMask"]
