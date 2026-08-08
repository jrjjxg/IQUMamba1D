"""Stage 375: Stage365 with a bottleneck-only real simplex mask.

The source slots are folded into the batch dimension before the shared U-Net
decoder is called.  This keeps one decoder parameter set and one decoder
invocation while preserving the per-slot nonlinear decoding path.
"""

from __future__ import annotations

import torch
from torch import nn

from models.IQUBiMamba1D_CoreUpgrades import (
    IQUBiMamba1D_IndependentComplexStateUniRepLK,
)


class IQUBiMamba1D_BottleneckRealMask(
    IQUBiMamba1D_IndependentComplexStateUniRepLK
):
    """Stage365 plus a real simplex mask at the bottleneck only.

    The encoder skip tensors above the bottleneck remain shared.  The masked
    bottleneck is expanded over source slots, flattened into ``B * K`` and
    decoded in one call.  The project uses InstanceNorm1d, so flattening slots
    into the batch is equivalent to decoding each slot independently while
    avoiding the Python-level decoder loop.
    """

    def __init__(self, *args, **kwargs) -> None:
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("bottleneck-mask Stage365 requires num_classes")
        num_sources = int(num_classes) // 2
        if int(num_classes) != 2 * num_sources or num_sources not in (2, 3):
            raise ValueError(
                "bottleneck-mask Stage365 expects two or three complex sources"
            )
        super().__init__(*args, **kwargs)
        if bool(getattr(self.decoder, "deep_supervision", False)):
            raise ValueError("bottleneck-mask Stage365 requires deep_supervision=false")

        # A slot decoder emits one complex waveform (I/Q).  The decoder trunk
        # is unchanged; only its final prediction heads are narrowed.
        self.decoder.seg_layers = nn.ModuleList(
            [
                self.encoder.conv_op(layer.in_channels, 2, 1)
                for layer in self.decoder.seg_layers
            ]
        )
        bottleneck_channels = int(self.encoder.output_channels[-1])
        self.bottleneck_mask_num_sources = num_sources
        self.bottleneck_mask_head = nn.Conv1d(
            bottleneck_channels,
            num_sources * bottleneck_channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.bottleneck_mask_head.weight)
        nn.init.zeros_(self.bottleneck_mask_head.bias)

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

    def _make_bottleneck_masks(self, bottleneck: torch.Tensor) -> torch.Tensor:
        batch, channels, length = bottleneck.shape
        logits = self.bottleneck_mask_head(bottleneck).reshape(
            batch,
            self.bottleneck_mask_num_sources,
            channels,
            length,
        )
        return torch.softmax(logits, dim=1)

    def _flatten_source_slots(
        self,
        skips: list[torch.Tensor],
        masks: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Create decoder inputs with source slots folded into the batch."""
        batch = skips[0].shape[0]
        slots = self.bottleneck_mask_num_sources
        batched_skips: list[torch.Tensor] = []
        for index, features in enumerate(skips):
            if index == len(skips) - 1:
                # Only the bottleneck is source-routed.
                value = features.unsqueeze(1) * masks
            else:
                # Shared skip features are reused for every source slot.  The
                # reshape materializes a contiguous B*K batch for Conv1d.
                value = features.unsqueeze(1).expand(
                    -1, slots, -1, -1
                )
            batched_skips.append(
                value.reshape(batch * slots, features.shape[1], features.shape[2])
                .contiguous()
            )
        return batched_skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(
                f"bottleneck-mask Stage365 expects [B, 2, L], got {tuple(x.shape)}"
            )
        skips = self._encode_skips(x)
        masks = self._make_bottleneck_masks(skips[-1])
        decoder_skips = self._flatten_source_slots(skips, masks)
        decoded = self.decoder(decoder_skips)
        if isinstance(decoded, (tuple, list)):
            raise RuntimeError("bottleneck-mask Stage365 requires deep_supervision=false")

        batch, _, length = decoded.shape
        slots = self.bottleneck_mask_num_sources
        decoded = decoded.reshape(batch // slots, slots, 2, length)
        # Keep the existing source contract: [S1_I, S1_Q, S2_I, S2_Q, ...].
        return decoded.reshape(batch // slots, 2 * slots, length)


__all__ = ["IQUMamba1D_BottleneckRealMask"]
