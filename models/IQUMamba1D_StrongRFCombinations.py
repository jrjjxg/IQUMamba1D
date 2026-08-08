"""Stages 325-329/350: proven backbone and receptive-field combinations."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    build_recent_rf_operator,
)


class IQUMamba1DStrongRFCombination(nn.Module):
    """Compose Stages 290/79/197 with the Stage309/310 RF operators."""

    VARIANTS = {
        "complex_fdconv",
        "complex_unireplk",
        "complex_hierarchical",
        "bimamba_cyclofresh_unireplk",
        "mamba_cyclofresh_unireplk",
        "complex_cyclofresh_unireplk",
    }

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        conv_op: type[nn.Conv1d],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        n_conv_per_stage: Sequence[int],
        num_classes: int,
        n_conv_per_stage_decoder: Sequence[int],
        deep_supervision: bool = False,
        combination_variant: str,
        complex_norm_eps: float = 1e-6,
        rf_residual_scale_init: float = 0.05,
        rf_module_config: dict | None = None,
        estimated_cyclofresh_config: dict | None = None,
        **backbone_kwargs: object,
    ) -> None:
        super().__init__()
        self.combination_variant = str(combination_variant).lower()
        if self.combination_variant not in self.VARIANTS:
            raise ValueError(
                f"Unsupported strong RF combination: {self.combination_variant}"
            )
        if n_stages < 3:
            raise ValueError("Strong RF combinations require at least three encoder stages")
        if input_channels != 2:
            raise ValueError("Strong RF combinations expect one I/Q mixture (two channels)")

        use_bimamba = self.combination_variant == "bimamba_cyclofresh_unireplk"
        backbone_class: type[nn.Module]
        if use_bimamba:
            from models.IQUBiMamba1D import IQUBiMamba1D

            backbone_class = IQUBiMamba1D
        else:
            backbone_class = IQUMamba1D

        common = dict(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=list(features_per_stage),
            conv_op=conv_op,
            kernel_sizes=list(kernel_sizes),
            strides=list(strides),
            n_conv_per_stage=list(n_conv_per_stage),
            num_classes=num_classes,
            n_conv_per_stage_decoder=list(n_conv_per_stage_decoder),
            deep_supervision=deep_supervision,
        )
        for name in (
            "conv_bias", "norm_op", "norm_op_kwargs", "nonlin", "nonlin_kwargs"
        ):
            if name in backbone_kwargs:
                common[name] = backbone_kwargs[name]
        self.backbone = backbone_class(**common)

        self.uses_complex_stem = self.combination_variant.startswith("complex_")
        if self.uses_complex_stem:
            from models.IQUMamba1D_ComplexStage4 import ComplexStem1d

            self.backbone.encoder.stem = ComplexStem1d(
                int(features_per_stage[0]),
                blocks=int(n_conv_per_stage[0]),
                kernel_size=int(kernel_sizes[0]),
                norm_eps=float(complex_norm_eps),
            )

        self.uses_cyclofresh = "cyclofresh" in self.combination_variant
        self.input_adapter: nn.Module = nn.Identity()
        if self.uses_cyclofresh:
            from models.IQUMamba1D_EstimatedCycloFRESH import (
                EstimatedCycloFRESHAdapter1D,
            )

            cyclo = dict(estimated_cyclofresh_config or {})
            self.input_adapter = EstimatedCycloFRESHAdapter1D(
                input_channels=input_channels,
                min_freq=float(cyclo.get("estimated_cyclofresh_min_freq", 1 / 64)),
                max_freq=float(cyclo.get("estimated_cyclofresh_max_freq", 1 / 8)),
                default_freq=float(cyclo.get("estimated_cyclofresh_default_freq", 1 / 32)),
                momentum=float(cyclo.get("estimated_cyclofresh_momentum", 0.05)),
                hidden_channels=int(cyclo.get("estimated_cyclofresh_hidden_channels", 8)),
                kernel_size=int(cyclo.get("estimated_cyclofresh_kernel_size", 9)),
                scale_init=float(cyclo.get("estimated_cyclofresh_scale_init", 0.01)),
                gate_hidden=int(cyclo.get("estimated_cyclofresh_gate_hidden", 8)),
                zero_init=bool(cyclo.get("estimated_cyclofresh_zero_init", True)),
            )

        config = dict(rf_module_config or {})
        fdconv_config = {
            "rf_kernel_size": int(config.get("fdconv_kernel_size", 31)),
            "rf_bands": int(config.get("fdconv_bands", 4)),
        }
        unireplk_config = {
            "rf_large_kernel": int(config.get("unireplk_large_kernel", 17)),
            "rf_ffn_factor": int(config.get("unireplk_ffn_factor", 4)),
            "rf_layer_scale": float(config.get("unireplk_layer_scale", 1e-6)),
        }
        channels = self.backbone.encoder.output_channels
        placements: dict[int, str]
        if self.combination_variant == "complex_fdconv":
            placements = {0: "fdconv"}
        elif self.combination_variant == "complex_hierarchical":
            placements = {0: "fdconv", 1: "unireplk", 2: "unireplk"}
        else:
            placements = {0: "unireplk", 1: "unireplk", 2: "unireplk"}

        self.stage_rf = nn.ModuleDict()
        for stage, kind in placements.items():
            operator_config = fdconv_config if kind == "fdconv" else unireplk_config
            self.stage_rf[str(stage)] = FeatureResidualAdapter(
                int(channels[stage]),
                build_recent_rf_operator(kind, int(channels[stage]), operator_config),
                float(rf_residual_scale_init),
            )

    @property
    def encoder(self) -> nn.Module:
        return self.backbone.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.backbone.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        x = self.input_adapter(x)
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for stage, (conv_stage, mamba) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            x = mamba(conv_stage(x))
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](x)
            skips.append(x)
        return self.decoder(skips)

    def no_weight_decay(self) -> set[str]:
        names = {
            f"stage_rf.{stage}.residual_scale" for stage in self.stage_rf
        }
        if self.uses_complex_stem:
            from models.IQUMamba1D_ComplexStage4 import (
                ComplexModReLU,
                ComplexRMSNorm1d,
            )

            for module_name, module in self.named_modules():
                if isinstance(module, ComplexRMSNorm1d):
                    names.add(f"{module_name}.log_scale")
                elif isinstance(module, ComplexModReLU):
                    names.add(f"{module_name}.bias")
        return names
