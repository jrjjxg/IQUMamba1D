"""Stage382: relocate Stage377 UniRepLK adapters into mask separation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.IQUMamba1D_RecentRFModules import (
    ParallelFeatureDeltaAdapter,
    build_recent_rf_operator,
)
from models.IQUResUNet1D_ComplexStateUniRepLK_LatentMask import (
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
)


class IQUResUNet1D_ComplexStateUniRepLKSeparator(
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask
):
    """Use UniRepLK only to estimate masks, not to form encoder skips."""

    def __init__(
        self,
        *args,
        rf_apply_stages: Sequence[int] = (),
        separator_unireplk_stages: Sequence[int] = (0, 1, 2),
        rf_residual_scale_init: float = 0.05,
        rf_large_kernel: int = 17,
        rf_ffn_factor: int = 4,
        rf_layer_scale: float = 1.0e-6,
        **kwargs,
    ) -> None:
        encoder_rf_stages = tuple(int(stage) for stage in rf_apply_stages)
        if encoder_rf_stages:
            raise ValueError(
                "Stage382 requires rf_apply_stages=[] so UniRepLK is absent "
                "from the encoder path"
            )
        super().__init__(
            *args,
            rf_apply_stages=(),
            rf_residual_scale_init=rf_residual_scale_init,
            rf_large_kernel=rf_large_kernel,
            rf_ffn_factor=rf_ffn_factor,
            rf_layer_scale=rf_layer_scale,
            **kwargs,
        )

        stages = tuple(int(stage) for stage in separator_unireplk_stages)
        if len(set(stages)) != len(stages):
            raise ValueError("separator_unireplk_stages must be unique")
        for stage in stages:
            if not 0 <= stage < len(self.encoder.stages):
                raise ValueError(
                    f"separator UniRepLK stage {stage} is outside the "
                    f"{len(self.encoder.stages)}-stage encoder"
                )

        operator_config = {
            "rf_large_kernel": int(rf_large_kernel),
            "rf_ffn_factor": int(rf_ffn_factor),
            "rf_layer_scale": float(rf_layer_scale),
        }
        self.separator_unireplk_stages = stages
        self.separator_rf = nn.ModuleDict(
            {
                str(stage): ParallelFeatureDeltaAdapter(
                    int(self.encoder.output_channels[stage]),
                    build_recent_rf_operator(
                        "unireplk",
                        int(self.encoder.output_channels[stage]),
                        operator_config,
                    ),
                    float(rf_residual_scale_init),
                )
                for stage in stages
            }
        )

    def _encode_paths(
        self,
        x: torch.Tensor,
        build_separator_context: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)

        skips = []
        separator_features = []
        for stage, (conv_stage, memory) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            stage_features = conv_stage(x)
            x = memory(stage_features)
            skips.append(x)

            context = x
            if build_separator_context and str(stage) in self.separator_rf:
                context = self.separator_rf[str(stage)](stage_features, x)
            separator_features.append(context)
        return skips, separator_features

    def _encode_skips(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips, _ = self._encode_paths(x, build_separator_context=False)
        return skips

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(
                "Stage382 expects I/Q input with shape [B, 2, L], "
                f"got {tuple(x.shape)}"
            )

        skips, separator_features = self._encode_paths(
            x,
            build_separator_context=True,
        )
        masks = [
            self._make_masks(features, head)
            for features, head in zip(separator_features, self.latent_mask_heads)
        ]
        decoded = self.decoder(self._flatten_source_slots(skips, masks))
        batch_slots, _, length = decoded.shape
        slots = self.latent_mask_num_sources
        if batch_slots % slots != 0:
            raise RuntimeError(
                "Stage382 decoder batch is not divisible by source slots: "
                f"batch={batch_slots}, slots={slots}"
            )
        return decoded.reshape(batch_slots // slots, slots * 2, length)

    def no_weight_decay(self) -> set[str]:
        names = super().no_weight_decay()
        names.update(
            f"separator_rf.{stage}.residual_scale"
            for stage in self.separator_rf
        )
        return names


__all__ = ["IQUResUNet1D_ComplexStateUniRepLKSeparator"]
