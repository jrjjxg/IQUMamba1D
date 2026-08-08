"""Stages 301-305: controlled combinations of the strongest Stage-4 variants.

The classes in this module deliberately reuse the already-tested Stage 290,
295, 298, 299 and cross-scale/FRESH implementations.  Each stage changes one
thing at a time so that a failed experiment remains interpretable.
"""

from __future__ import annotations

from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUBiMamba1D import BiMambaLayer
from models.IQUBiMamba1D_CrossScaleAttention import (
    CompressedGlobalCrossAttention,
    IQUBiMamba1D_CrossScaleAttention,
)
from models.IQUBiMamba1D_EstimatedCycloFRESH import (
    IQUBiMamba1D_EstimatedCycloFRESH,
)
from models.IQUMamba1D_ComplexStage4 import (
    ComplexModReLU,
    ComplexRMSNorm1d,
    ComplexStem1d,
)
from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    IQUMamba1DComplexStateMamba,
)
from models.IQUMamba1D_EstimatedCycloFRESH import (
    EstimatedCycloFRESHAdapter1D,
)


TensorOrList = Union[torch.Tensor, List[torch.Tensor]]


def _complex_parameter_names(module: nn.Module) -> set[str]:
    """Return parameters that should not receive weight decay."""

    names: set[str] = set()
    for module_name, child in module.named_modules():
        if isinstance(child, ComplexRMSNorm1d):
            names.add(f"{module_name}.log_scale")
        if isinstance(child, ComplexModReLU):
            names.add(f"{module_name}.bias")
        if isinstance(child, ComplexStateMambaLayer):
            for parameter_name, _ in child.named_parameters():
                if parameter_name.endswith((".a_log", ".theta", ".D")):
                    names.add(f"{module_name}.{parameter_name}")
    return names


class Stage301ComplexStateCrossScale(nn.Module):
    """Stage 299 plus one bottleneck-to-C3 cross-scale attention path."""

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        complex_norm_eps: float = 1e-6,
        cross_scale_query_stages: Sequence[int] = (2,),
        cross_scale_global_stage: int = 3,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 0.01,
        **_: object,
    ) -> None:
        super().__init__()
        self.core = IQUMamba1DComplexStateMamba(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            complex_stem_enable=True,
            complex_norm_eps=complex_norm_eps,
        )
        self.cross_scale_global_stage = int(cross_scale_global_stage)
        self.cross_scale_query_stages = tuple(
            int(stage) for stage in cross_scale_query_stages
        )
        if not self.cross_scale_query_stages:
            raise ValueError("cross_scale_query_stages must not be empty")
        self.cross_scale_blocks = nn.ModuleDict(
            {
                str(stage): CompressedGlobalCrossAttention(
                    query_channels=int(features_per_stage[stage]),
                    global_channels=int(
                        features_per_stage[self.cross_scale_global_stage]
                    ),
                    kv_tokens=cross_scale_kv_tokens,
                    num_heads=cross_scale_num_heads,
                    dropout=cross_scale_dropout,
                    residual_scale_init=cross_scale_residual_scale_init,
                )
                for stage in self.cross_scale_query_stages
            }
        )

    def forward(self, x: torch.Tensor) -> TensorOrList:
        skips = self.core.backbone.encoder(x)
        global_feature = skips[self.cross_scale_global_stage]
        enhanced_skips = list(skips)
        for stage in self.cross_scale_query_stages:
            enhanced_skips[stage] = self.cross_scale_blocks[str(stage)](
                enhanced_skips[stage],
                global_feature,
            )
        return self.core.backbone.decoder(enhanced_skips)

    def no_weight_decay(self) -> set[str]:
        return _complex_parameter_names(self)


class Stage302BiFRESHComplexBottleneck(nn.Module):
    """Stage 298 with only its deepest BiMamba replaced by Stage-295 SSM."""

    def __init__(
        self,
        *,
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
        deep_supervision: bool = False,
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        complex_norm_eps: float = 1e-6,
        **fresh_kwargs: object,
    ) -> None:
        super().__init__()
        self.core = IQUBiMamba1D_EstimatedCycloFRESH(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            complex_stem_enable=True,
            complex_norm_eps=complex_norm_eps,
            **fresh_kwargs,
        )
        layers = self.core.backbone.encoder.mamba_layers
        bottleneck_index = next(
            (
                index
                for index in range(len(layers) - 1, -1, -1)
                if isinstance(layers[index], BiMambaLayer)
            ),
            None,
        )
        if bottleneck_index is None:
            raise ValueError("BiMamba backbone has no bottleneck Mamba layer")
        old_layer = layers[bottleneck_index]
        layers[bottleneck_index] = ComplexStateMambaLayer(
            dim=old_layer.dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            channel_token=old_layer.channel_token,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
        )
        self.complex_bottleneck_index = int(bottleneck_index)

    def forward(self, x: torch.Tensor) -> TensorOrList:
        return self.core(x)

    def no_weight_decay(self) -> set[str]:
        return _complex_parameter_names(self)


class Stage303ComplexStemBiMambaCrossScale(nn.Module):
    """Stage 290 stem + Stage-235 BiMamba cross-scale, without FRESH."""

    def __init__(
        self,
        *,
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
        deep_supervision: bool = False,
        complex_norm_eps: float = 1e-6,
        **cross_scale_kwargs: object,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("Stage 303 strict-complex stem expects I/Q input")
        self.core = IQUBiMamba1D_CrossScaleAttention(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            **cross_scale_kwargs,
        )
        self.core.encoder.stem = ComplexStem1d(
            int(features_per_stage[0]),
            blocks=int(n_conv_per_stage[0]),
            kernel_size=int(kernel_sizes[0]),
            norm_eps=float(complex_norm_eps),
        )

    def forward(self, x: torch.Tensor) -> TensorOrList:
        return self.core(x)

    def no_weight_decay(self) -> set[str]:
        return _complex_parameter_names(self)


class Stage304Stage298299Fusion(nn.Module):
    """Jointly trained convex output fusion of Stage 298 and Stage 299."""

    def __init__(
        self,
        *,
        stage298: nn.Module,
        stage299: nn.Module,
        fusion_logit_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.stage298 = stage298
        self.stage299 = stage299
        self.fusion_logit = nn.Parameter(
            torch.tensor(float(fusion_logit_init))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output298 = self.stage298(x)
        output299 = self.stage299(x)
        if isinstance(output298, (list, tuple)) or isinstance(
            output299, (list, tuple)
        ):
            raise RuntimeError("Stage 304 requires deep_supervision=false")
        alpha = torch.sigmoid(self.fusion_logit).to(dtype=output298.dtype)
        return alpha * output298 + (1.0 - alpha) * output299

    def no_weight_decay(self) -> set[str]:
        names = {"fusion_logit"}
        names.update(f"stage298.{name}" for name in _complex_parameter_names(self.stage298))
        names.update(f"stage299.{name}" for name in _complex_parameter_names(self.stage299))
        return names


class Stage305GatedFRESHComplexState(nn.Module):
    """Stage 299 with a conservative, learnable FRESH input residual."""

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        n_conv_per_stage: List[int],
        num_classes: int,
        n_conv_per_stage_decoder: List[int],
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        complex_norm_eps: float = 1e-6,
        fresh_gate_logit_init: float = -3.0,
        estimated_cyclofresh_min_freq: float = 1.0 / 64.0,
        estimated_cyclofresh_max_freq: float = 1.0 / 8.0,
        estimated_cyclofresh_default_freq: float = 1.0 / 32.0,
        estimated_cyclofresh_momentum: float = 0.05,
        estimated_cyclofresh_hidden_channels: int = 8,
        estimated_cyclofresh_kernel_size: int = 9,
        estimated_cyclofresh_scale_init: float = 0.01,
        estimated_cyclofresh_gate_hidden: int = 8,
        estimated_cyclofresh_zero_init: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.fresh = EstimatedCycloFRESHAdapter1D(
            input_channels=input_channels,
            min_freq=estimated_cyclofresh_min_freq,
            max_freq=estimated_cyclofresh_max_freq,
            default_freq=estimated_cyclofresh_default_freq,
            momentum=estimated_cyclofresh_momentum,
            hidden_channels=estimated_cyclofresh_hidden_channels,
            kernel_size=estimated_cyclofresh_kernel_size,
            scale_init=estimated_cyclofresh_scale_init,
            gate_hidden=estimated_cyclofresh_gate_hidden,
            zero_init=estimated_cyclofresh_zero_init,
        )
        self.core = IQUMamba1DComplexStateMamba(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            complex_stem_enable=True,
            complex_norm_eps=complex_norm_eps,
        )
        self.fresh_gate_logit = nn.Parameter(
            torch.tensor(float(fresh_gate_logit_init))
        )

    def forward(self, x: torch.Tensor) -> TensorOrList:
        fresh_x = self.fresh(x)
        gate = torch.sigmoid(self.fresh_gate_logit).to(dtype=x.dtype)
        gated_input = x + gate * (fresh_x - x)
        return self.core(gated_input)

    def no_weight_decay(self) -> set[str]:
        return {"fresh_gate_logit"} | _complex_parameter_names(self)
