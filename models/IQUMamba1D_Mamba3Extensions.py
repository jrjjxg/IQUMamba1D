"""Official Mamba-3 and full RF Stage-4 extensions (Stages 340-342)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    IQUMamba1DComplexStateMamba,
)
from models.IQUMamba1D_MemoryRFStages import (
    _replace_stage4_mamba,
    _stage4_backbone,
    _valid_headdim,
)
from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    build_recent_rf_operator,
)


def _official_mamba3_class():
    """Load Mamba-3 from either official export used across revisions."""

    top_level_error: Exception | None = None
    try:
        from mamba_ssm import Mamba3

        return Mamba3
    except (ImportError, AttributeError) as exc:
        top_level_error = exc
    try:
        from mamba_ssm.modules.mamba3 import Mamba3

        return Mamba3
    except (ImportError, AttributeError) as module_error:
        raise RuntimeError(
            "Stage 340 requires the official state-spaces/mamba `main` "
            "revision containing mamba_ssm/modules/mamba3.py and its fused "
            "Triton kernels. The installed mamba_ssm is older or incomplete. "
            f"Top-level import: {top_level_error}; module import: {module_error}"
        ) from module_error


class OfficialMamba3Layer(nn.Module):
    """Drop-in Stage-4 adapter around the public official ``Mamba3`` class."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        channel_token: bool = False,
    ) -> None:
        super().__init__()
        Mamba3 = _official_mamba3_class()

        self.dim = int(dim)
        self.channel_token = bool(channel_token)
        self.is_mimo = bool(is_mimo)
        self.norm = nn.LayerNorm(self.dim)
        self.last_scan_backend = "uninitialized"
        actual_headdim = _valid_headdim(int(expand) * self.dim, int(headdim))
        try:
            self.mamba = Mamba3(
                d_model=self.dim,
                d_state=int(d_state),
                expand=int(expand),
                headdim=actual_headdim,
                ngroups=int(ngroups),
                rope_fraction=float(rope_fraction),
                is_outproj_norm=bool(is_outproj_norm),
                is_mimo=self.is_mimo,
                mimo_rank=int(mimo_rank),
                chunk_size=int(chunk_size),
            )
        except Exception as exc:
            mode = "MIMO/TileLang" if self.is_mimo else "SISO/Triton"
            raise RuntimeError(
                f"Failed to initialize official Mamba-3 {mode} fused path: {exc}"
            ) from exc

    def _run_mamba(self, tokens: torch.Tensor) -> torch.Tensor:
        try:
            output = self.mamba(self.norm(tokens))
        except Exception as exc:
            mode = "MIMO/TileLang" if self.is_mimo else "SISO/Triton"
            raise RuntimeError(
                f"Official Mamba-3 {mode} fused forward failed. No local "
                f"fallback is enabled: {exc}"
            ) from exc
        self.last_scan_backend = (
            "official_mamba3_mimo_tilelang"
            if self.is_mimo
            else "official_mamba3_siso_triton"
        )
        return output

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        tokens = x.reshape(batch, channels, -1).transpose(1, 2)
        return self._run_mamba(tokens).transpose(1, 2).reshape(
            batch, channels, *dims
        )

    def _channel_tokens(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens = x.shape[:2]
        dims = x.shape[2:]
        return self._run_mamba(x.flatten(2)).reshape(batch, tokens, *dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the caller's autocast dtype; the official fused kernel owns its
        # mixed-precision policy.
        return self._channel_tokens(x) if self.channel_token else self._patch_tokens(x)


class IQUMamba1DOfficialMamba3(nn.Module):
    """Stage 340: replace Stage-4 sequence layers with official Mamba-3."""

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        n_conv_per_stage: Sequence[int],
        num_classes: int,
        n_conv_per_stage_decoder: Sequence[int],
        deep_supervision: bool = False,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = _stage4_backbone(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=list(features_per_stage),
            kernel_sizes=list(kernel_sizes),
            strides=list(strides),
            n_conv_per_stage=list(n_conv_per_stage),
            num_classes=num_classes,
            n_conv_per_stage_decoder=list(n_conv_per_stage_decoder),
            deep_supervision=deep_supervision,
        )
        self.replaced_layers = _replace_stage4_mamba(
            self.backbone,
            lambda dim, channel_token: OfficialMamba3Layer(
                dim,
                d_state=d_state,
                expand=expand,
                headdim=headdim,
                ngroups=ngroups,
                rope_fraction=rope_fraction,
                is_outproj_norm=is_outproj_norm,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                channel_token=channel_token,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        return self.backbone(x)

    def scan_backend_status(self) -> dict[str, str]:
        return {
            f"encoder.mamba_layers.{index}": layer.last_scan_backend
            for index, layer in enumerate(self.backbone.encoder.mamba_layers)
            if isinstance(layer, OfficialMamba3Layer)
        }

    def diagnostics(self) -> dict[str, str]:
        return {
            f"scan_backend_{name}": backend
            for name, backend in self.scan_backend_status().items()
        }

    def no_weight_decay(self) -> set[str]:
        return {
            name
            for name, parameter in self.named_parameters()
            if bool(getattr(parameter, "_no_weight_decay", False))
        }


class IQUMamba1DFullRFCombination(nn.Module):
    """Stages 342/349: complex stem + RF Mamba-3 + real UniRepLK."""

    def __init__(
        self,
        *,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        n_conv_per_stage: Sequence[int],
        num_classes: int,
        n_conv_per_stage_decoder: Sequence[int],
        mamba_d_state: int = 8,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        scan_checkpoint: bool = True,
        scan_backend: str = "auto",
        trapezoid_lambda_init: float = 0.5,
        cyclic_frequencies: Sequence[float] = (),
        cyclic_max_frequency_delta: float = 0.01,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
        complex_norm_eps: float = 1e-6,
        estimated_cyclofresh_enable: bool = True,
        estimated_cyclofresh_config: dict | None = None,
        rf_residual_scale_init: float = 0.05,
        unireplk_large_kernel: int = 17,
        unireplk_ffn_factor: int = 4,
        unireplk_layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("Stage 342 expects exactly one I/Q mixture")
        if int(n_stages) < 4:
            raise ValueError("Stage 342 requires the four-stage Stage-4 backbone")

        rf_model = IQUMamba1DComplexStateMamba(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=list(features_per_stage),
            kernel_sizes=list(kernel_sizes),
            strides=list(strides),
            n_conv_per_stage=list(n_conv_per_stage),
            num_classes=num_classes,
            n_conv_per_stage_decoder=list(n_conv_per_stage_decoder),
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            scan_checkpoint=scan_checkpoint,
            scan_backend=scan_backend,
            mamba_discretization="exponential_trapezoidal",
            trapezoid_lambda_init=trapezoid_lambda_init,
            cyclic_theta_enable=True,
            cyclic_frequencies=cyclic_frequencies,
            cyclic_max_frequency_delta=cyclic_max_frequency_delta,
            reliability_enable=True,
            reliability_hidden=reliability_hidden,
            reliability_floor=reliability_floor,
            reliability_init=reliability_init,
            complex_stem_enable=True,
            complex_norm_eps=complex_norm_eps,
        )
        self.backbone = rf_model.backbone

        self.uses_cyclofresh = bool(estimated_cyclofresh_enable)
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

        unirep_config = {
            "rf_large_kernel": int(unireplk_large_kernel),
            "rf_ffn_factor": int(unireplk_ffn_factor),
            "rf_layer_scale": float(unireplk_layer_scale),
        }
        channels = self.backbone.encoder.output_channels
        self.stage_rf = nn.ModuleDict({
            str(stage): FeatureResidualAdapter(
                int(channels[stage]),
                build_recent_rf_operator(
                    "unireplk", int(channels[stage]), unirep_config
                ),
                float(rf_residual_scale_init),
            )
            for stage in (0, 1, 2)
        })

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
        for stage, (conv_stage, memory) in enumerate(
            zip(self.encoder.stages, self.encoder.mamba_layers)
        ):
            x = memory(conv_stage(x))
            if str(stage) in self.stage_rf:
                x = self.stage_rf[str(stage)](x)
            skips.append(x)
        return self.decoder(skips)

    def effective_cyclic_frequencies(self) -> dict[str, torch.Tensor]:
        return {
            f"encoder.mamba_layers.{index}": layer.ssm.effective_cyclic_frequencies()
            for index, layer in enumerate(self.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
        }

    def scan_backend_status(self) -> dict[str, str]:
        return {
            f"encoder.mamba_layers.{index}": layer.ssm.last_scan_backend
            for index, layer in enumerate(self.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
        }

    def diagnostics(self) -> dict[str, str]:
        return {
            f"scan_backend_{name}": backend
            for name, backend in self.scan_backend_status().items()
        }

    def no_weight_decay(self) -> set[str]:
        names = {
            name
            for name, _ in self.named_parameters()
            if name.endswith((".a_log", ".theta", ".D"))
        }
        names.update(
            f"stage_rf.{stage}.residual_scale" for stage in self.stage_rf
        )
        from models.IQUMamba1D_ComplexStage4 import ComplexModReLU, ComplexRMSNorm1d

        for module_name, module in self.named_modules():
            if isinstance(module, ComplexRMSNorm1d):
                names.add(f"{module_name}.log_scale")
            elif isinstance(module, ComplexModReLU):
                names.add(f"{module_name}.bias")
        return names
