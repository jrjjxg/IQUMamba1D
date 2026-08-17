"""Controlled Stage310/Stage381 UniRepLK connection ablations (391-394)."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    IQUMamba1DRecentRF,
    ParallelFeatureDeltaAdapter,
    ParallelFeatureFullAdapter,
    build_recent_rf_operator,
)
from models.IQUResUNet1D_UniRepLKBackbone import AdaptiveRealUniRepLK


class _Stage391ControlledUniRepLK(IQUMamba1DRecentRF):
    """Stage4 with controlled operator and connection choices at stages 0/2."""

    parallel_delta = False
    adaptive_routing = False
    post_delta = False
    pre_full = False

    def __init__(
        self,
        *args,
        rf_residual_scale_init: float = 0.05,
        rf_large_kernel: int = 17,
        rf_kernels: Sequence[int] = (9, 17),
        rf_ffn_factor: int = 4,
        rf_layer_scale: float = 1e-6,
        **kwargs,
    ) -> None:
        # Stage391 differs from Stage310 only by removing zero-based stage 1.
        operator_config = {
            "rf_apply_stages": (0, 2),
            "rf_large_kernel": int(rf_large_kernel),
            "rf_ffn_factor": int(rf_ffn_factor),
            "rf_layer_scale": float(rf_layer_scale),
        }
        super().__init__(
            *args,
            rf_module_type="unireplk",
            rf_residual_scale_init=float(rf_residual_scale_init),
            rf_module_config=operator_config,
            **kwargs,
        )

        if (self.parallel_delta or self.adaptive_routing
                or self.post_delta or self.pre_full):
            adapters = {}
            for stage in (0, 2):
                channels = int(self.encoder.output_channels[stage])
                if self.adaptive_routing:
                    operator = AdaptiveRealUniRepLK(
                        channels,
                        kernels=tuple(int(kernel) for kernel in rf_kernels),
                        ffn_factor=int(rf_ffn_factor),
                        layer_scale=float(rf_layer_scale),
                    )
                else:
                    operator = build_recent_rf_operator(
                        "unireplk", channels, operator_config
                    )
                if self.pre_full:
                    adapter_cls = ParallelFeatureFullAdapter
                elif self.parallel_delta or self.post_delta:
                    adapter_cls = ParallelFeatureDeltaAdapter
                else:
                    adapter_cls = FeatureResidualAdapter
                adapters[str(stage)] = adapter_cls(
                    channels, operator, float(rf_residual_scale_init)
                )
            self.stage_rf = nn.ModuleDict(adapters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.parallel_delta or self.post_delta or self.pre_full):
            # This is exactly the Stage310 connection: UniRepLK consumes the
            # Mamba output and FeatureResidualAdapter adds its complete output.
            return super().forward(x)

        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage, (conv_stage, mamba) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            stage_features = conv_stage(x)
            x = mamba(stage_features)
            if str(stage) in self.stage_rf:
                if self.post_delta:
                    # Delta-Post: use the Mamba output as both the UniRepLK
                    # source and the independent main branch.
                    x = self.stage_rf[str(stage)](x, x)
                elif self.pre_full:
                    # Full-Pre: process pre-Mamba convolutional features with
                    # the complete UniRepLK block and add them to Mamba output.
                    x = self.stage_rf[str(stage)](stage_features, x)
                else:
                    # Delta-Pre (Stage381): only the UniRepLK residual delta
                    # from pre-Mamba features is added to the Mamba output.
                    x = self.stage_rf[str(stage)](stage_features, x)
            skips.append(x)
        return self.decoder(skips)


class IQUMamba1D_Stage391(_Stage391ControlledUniRepLK):
    """Stage310 with only zero-based stage 1 UniRepLK removed."""


class IQUMamba1D_Stage392(_Stage391ControlledUniRepLK):
    """Stage391 with Stage381's parallel residual-delta connection."""

    parallel_delta = True


class IQUMamba1D_Stage393(_Stage391ControlledUniRepLK):
    """Stage391 connection with Stage389 adaptive real RF UniRepLK experts."""

    adaptive_routing = True


class IQUMamba1D_Stage394(_Stage391ControlledUniRepLK):
    """Stage392 connection with Stage389 adaptive real RF UniRepLK experts."""

    parallel_delta = True
    adaptive_routing = True


class IQUMamba1D_Stage395(_Stage391ControlledUniRepLK):
    """Controlled Delta-Post: post-Mamba input, residual delta only."""

    post_delta = True


class IQUMamba1D_Stage396(_Stage391ControlledUniRepLK):
    """Controlled Full-Pre: pre-Mamba input, complete UniRepLK output."""

    pre_full = True


class IQUMamba1D_Stage397(IQUMamba1D_Stage394):
    """Stage394 with independent complex-state BiMamba at stages 1 and 3."""

    def __init__(
        self,
        *args,
        complex_state_d_state: int = 8,
        complex_state_d_conv: int = 4,
        complex_state_expand: int = 2,
        complex_state_scan_checkpoint: bool = True,
        complex_state_scan_backend: str = "auto",
        complex_state_fusion_hidden: int = 64,
        bimamba_residual_scale_init: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        from models.IQUBiMamba1D_CoreUpgrades import (
            IndependentComplexStateBiMambaLayer,
        )

        for stage in (1, 3):
            old = self.encoder.mamba_layers[stage]
            self.encoder.mamba_layers[stage] = IndependentComplexStateBiMambaLayer(
                dim=old.dim,
                channel_token=old.channel_token,
                d_state=int(complex_state_d_state),
                d_conv=int(complex_state_d_conv),
                expand=int(complex_state_expand),
                scan_checkpoint=bool(complex_state_scan_checkpoint),
                scan_backend=str(complex_state_scan_backend),
                fusion_hidden=int(complex_state_fusion_hidden),
                residual_scale_init=float(bimamba_residual_scale_init),
            )


__all__ = [
    "IQUMamba1D_Stage391",
    "IQUMamba1D_Stage392",
    "IQUMamba1D_Stage393",
    "IQUMamba1D_Stage394",
    "IQUMamba1D_Stage395",
    "IQUMamba1D_Stage396",
    "IQUMamba1D_Stage397",
]
