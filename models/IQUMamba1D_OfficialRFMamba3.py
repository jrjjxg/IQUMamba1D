"""Official fused Mamba-3 with RF cyclic and reliability conditioning."""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D_EstimatedCycloFRESH import (
    EstimatedCycloFRESHAdapter1D,
    estimate_cyclic_frequency_with_confidence,
)
from models.IQUMamba1D_Mamba3Extensions import _official_mamba3_class
from models.IQUMamba1D_MemoryRFStages import _stage4_backbone, _valid_headdim


class OfficialRFMamba3Core(nn.Module):
    """RF conditioning around the official SISO Mamba-3 fused kernel."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_outproj_norm: bool = False,
        chunk_size: int = 64,
        token_stride: int = 1,
        force_real_state: bool = False,
        cyclic_anchor_enable: bool = False,
        cyclic_frequencies: Sequence[float] = (),
        cyclic_max_frequency_delta: float = 0.01,
        dynamic_cyclic_enable: bool = False,
        reliability_enable: bool = False,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
        shared_confidence_enable: bool = False,
    ) -> None:
        super().__init__()
        Mamba3 = _official_mamba3_class()
        self.d_model = int(d_model)
        self.token_stride = max(1, int(token_stride))
        self.force_real_state = bool(force_real_state)
        self.cyclic_anchor_enable = bool(cyclic_anchor_enable)
        self.dynamic_cyclic_enable = bool(dynamic_cyclic_enable)
        self.reliability_enable = bool(reliability_enable)
        self.shared_confidence_enable = bool(shared_confidence_enable)
        self.cyclic_max_frequency_delta = float(cyclic_max_frequency_delta)
        if self.force_real_state and (
            self.cyclic_anchor_enable or self.dynamic_cyclic_enable
        ):
            raise ValueError(
                "force_real_state cannot be combined with cyclic rotation"
            )
        if self.cyclic_max_frequency_delta < 0.0:
            raise ValueError("cyclic_max_frequency_delta must be non-negative")

        inner = int(expand) * self.d_model
        actual_headdim = _valid_headdim(inner, int(headdim))
        self.mamba = Mamba3(
            d_model=self.d_model,
            d_state=int(d_state),
            expand=int(expand),
            headdim=actual_headdim,
            ngroups=int(ngroups),
            rope_fraction=float(rope_fraction),
            is_outproj_norm=bool(is_outproj_norm),
            is_mimo=False,
            chunk_size=int(chunk_size),
        )
        self.last_scan_backend = "uninitialized"
        self.last_reliability: torch.Tensor | None = None
        self.last_anchor_frequencies: torch.Tensor | None = None

        frequencies = torch.tensor(
            [float(value) for value in cyclic_frequencies], dtype=torch.float32
        )
        if self.cyclic_anchor_enable and not self.dynamic_cyclic_enable:
            if frequencies.numel() == 0:
                raise ValueError(
                    "Fixed cyclic anchoring requires cyclic_frequencies"
                )
        self.register_buffer("fixed_cyclic_frequencies", frequencies)

        if self.reliability_enable:
            if not 0.0 <= float(reliability_floor) < 1.0:
                raise ValueError("reliability_floor must be in [0, 1)")
            if not float(reliability_floor) < float(reliability_init) < 1.0:
                raise ValueError(
                    "reliability_init must be between reliability_floor and 1"
                )
            self.reliability_floor = float(reliability_floor)
            hidden = max(1, int(reliability_hidden))
            self.reliability_net = nn.Sequential(
                nn.Linear(3, hidden),
                nn.SiLU(),
                nn.Linear(hidden, int(self.mamba.nheads)),
            )
            normalized = (
                (float(reliability_init) - self.reliability_floor)
                / (1.0 - self.reliability_floor)
            )
            nn.init.zeros_(self.reliability_net[-1].weight)
            nn.init.constant_(
                self.reliability_net[-1].bias,
                math.log(normalized / (1.0 - normalized)),
            )
        else:
            self.reliability_floor = 1.0
            self.reliability_net = None

    @staticmethod
    def _repeat_to_length(values: torch.Tensor, length: int) -> torch.Tensor:
        repeats = math.ceil(int(length) / int(values.shape[-1]))
        repeat_shape = [1] * values.ndim
        repeat_shape[-1] = repeats
        return values.repeat(*repeat_shape)[..., :length]

    def _cyclic_pattern(
        self,
        batch: int,
        count: int,
        device: torch.device,
        cyclic_frequency: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.dynamic_cyclic_enable:
            if cyclic_frequency is None:
                raise ValueError("Dynamic cyclic anchoring requires a frequency")
            if cyclic_frequency.ndim != 1 or cyclic_frequency.shape[0] != batch:
                raise ValueError("Dynamic cyclic frequency must be shaped (batch,)")
            base = cyclic_frequency.to(device=device, dtype=torch.float32)
            multipliers = torch.tensor(
                [0.0, 1.0, -1.0, 2.0, -2.0],
                device=device,
            )
            values = base[:, None] * multipliers[None, :]
        else:
            values = self.fixed_cyclic_frequencies.to(device=device)[None, :]
        return self._repeat_to_length(values, count)

    def _condition_angles(
        self,
        raw_angles: torch.Tensor,
        cyclic_frequency: torch.Tensor | None,
        confidence: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, _, nheads, count = raw_angles.shape
        if self.force_real_state:
            self.last_anchor_frequencies = None
            return torch.zeros_like(raw_angles)
        if not self.cyclic_anchor_enable:
            self.last_anchor_frequencies = None
            return raw_angles

        input_frequencies = self._cyclic_pattern(
            batch, count, raw_angles.device, cyclic_frequency
        )
        token_frequencies = torch.remainder(
            input_frequencies * self.token_stride + 0.5, 1.0
        ) - 0.5
        self.last_anchor_frequencies = input_frequencies.detach()
        dt_reference = F.softplus(self.mamba.dt_bias.detach()).clamp_min(1e-6)
        anchor_rate = (
            2.0
            * math.pi
            * token_frequencies[:, None, None, :]
            / dt_reference[None, None, :, None]
        )
        residual_limit = (
            2.0
            * math.pi
            * self.cyclic_max_frequency_delta
            * self.token_stride
            / dt_reference[None, None, :, None]
        )
        anchored = anchor_rate + residual_limit * torch.tanh(raw_angles)
        if self.shared_confidence_enable:
            if confidence is None:
                raise ValueError("Shared cyclic conditioning requires confidence")
            weight = confidence.float().clamp(0.0, 1.0)[:, None, None, None]
            anchored = raw_angles + weight * (anchored - raw_angles)
        return anchored

    def _reliability(
        self,
        tokens: torch.Tensor,
        confidence: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, _ = tokens.shape
        if self.reliability_net is None:
            return tokens.new_ones((batch, length, int(self.mamba.nheads)))
        values = tokens.float()
        previous = F.pad(values[:, :-1], (0, 0, 1, 0))
        power = values.square().mean(dim=-1).clamp_min(1e-8)
        previous_power = previous.square().mean(dim=-1).clamp_min(1e-8)
        flux = (values - previous).square().mean(dim=-1).clamp_min(1e-8)
        correlation = (values * previous).mean(dim=-1) / torch.sqrt(
            power * previous_power
        )
        log_power = torch.log(power)
        log_flux = torch.log(flux)
        evidence = torch.stack(
            (
                log_power - log_power.mean(dim=1, keepdim=True),
                log_flux - log_flux.mean(dim=1, keepdim=True),
                correlation.clamp(-1.0, 1.0),
            ),
            dim=-1,
        )
        probability = torch.sigmoid(self.reliability_net(evidence))
        reliability = self.reliability_floor + (
            1.0 - self.reliability_floor
        ) * probability
        if self.shared_confidence_enable:
            if confidence is None:
                raise ValueError("Shared reliability conditioning requires confidence")
            weight = confidence.float().clamp(0.0, 1.0)[:, None, None]
            reliability = 1.0 - weight * (1.0 - reliability)
        return reliability

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        cyclic_frequency: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the exact official SISO kernel after RF parameter conditioning."""

        mamba = self.mamba
        batch, length, _ = tokens.shape
        projected = mamba.in_proj(tokens)
        sizes = [
            mamba.d_inner,
            mamba.d_inner,
            mamba.d_state * mamba.num_bc_heads,
            mamba.d_state * mamba.num_bc_heads,
            mamba.nheads,
            mamba.nheads,
            mamba.nheads,
            mamba.num_rope_angles,
        ]
        z, x, B, C, dd_dt, dd_A, trap, raw_angles = torch.split(
            projected, sizes, dim=-1
        )
        z = z.reshape(batch, length, mamba.nheads, mamba.headdim)
        x = x.reshape(batch, length, mamba.nheads, mamba.headdim)
        B = B.reshape(batch, length, 1, mamba.num_bc_heads, mamba.d_state)
        C = C.reshape(batch, length, 1, mamba.num_bc_heads, mamba.d_state)
        trap = trap.transpose(1, 2).contiguous()

        official_module = importlib.import_module("mamba_ssm.modules.mamba3")
        decay_rate = -official_module.heavy_tail_activation(dd_A.float())
        decay_rate = torch.clamp(decay_rate, max=-mamba.A_floor)
        dt = F.softplus(dd_dt + mamba.dt_bias)
        reliability = self._reliability(tokens, confidence)
        if self.reliability_enable:
            dt = dt * reliability
        self.last_reliability = reliability.detach()
        adt = decay_rate * dt
        DT = dt.transpose(1, 2).contiguous()
        ADT = adt.transpose(1, 2).contiguous()

        raw_angles = raw_angles[:, :, None, :].expand(
            -1, -1, mamba.nheads, -1
        ).float()
        angles = self._condition_angles(
            raw_angles, cyclic_frequency, confidence
        )
        B = mamba.B_norm(B)
        C = mamba.C_norm(C)

        try:
            y = official_module.mamba3_siso_combined(
                Q=C.squeeze(2),
                K=B.squeeze(2),
                V=x,
                ADT=ADT,
                DT=DT,
                Trap=trap,
                Q_bias=mamba.C_bias.squeeze(1),
                K_bias=mamba.B_bias.squeeze(1),
                Angles=angles,
                D=mamba.D,
                Z=z if not mamba.is_outproj_norm else None,
                chunk_size=mamba.chunk_size,
                Input_States=None,
                return_final_states=False,
                cu_seqlens=None,
            )
        except Exception as exc:
            raise RuntimeError(
                "Official RF-aware Mamba-3 fused SISO forward failed; no "
                f"fallback is enabled: {exc}"
            ) from exc
        self.last_scan_backend = "official_mamba3_siso_triton_rf_conditioned"
        y = y.reshape(batch, length, mamba.d_inner)
        if mamba.is_outproj_norm:
            y = mamba.norm(y, z.reshape(batch, length, mamba.d_inner))
        return mamba.out_proj(y.to(x.dtype))


class OfficialRFMamba3Layer(nn.Module):
    """Stage-4 feature adapter for :class:`OfficialRFMamba3Core`."""

    def __init__(self, dim: int, **core_kwargs: object) -> None:
        super().__init__()
        self.dim = int(dim)
        self.channel_token = False
        self.norm = nn.LayerNorm(self.dim)
        self.ssm = OfficialRFMamba3Core(self.dim, **core_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cyclic_frequency: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, channels = x.shape[:2]
        dims = x.shape[2:]
        tokens = x.reshape(batch, channels, -1).transpose(1, 2)
        output = self.ssm(
            self.norm(tokens),
            cyclic_frequency=cyclic_frequency,
            confidence=confidence,
        )
        return output.transpose(1, 2).reshape(batch, channels, *dims)


class IQUMamba1DOfficialRFMamba3(nn.Module):
    """Stages 343-346: controlled official fused RF-aware Mamba-3."""

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
        chunk_size: int = 64,
        force_real_state: bool = False,
        cyclic_anchor_enable: bool = False,
        cyclic_frequencies: Sequence[float] = (),
        cyclic_max_frequency_delta: float = 0.01,
        dynamic_cyclic_enable: bool = False,
        reliability_enable: bool = False,
        reliability_hidden: int = 8,
        reliability_floor: float = 0.05,
        reliability_init: float = 0.995,
        shared_conditioning_enable: bool = False,
        estimated_cyclofresh_config: dict | None = None,
    ) -> None:
        super().__init__()
        if int(input_channels) != 2:
            raise ValueError("RF-aware Stage-4 expects one I/Q mixture")
        self.shared_conditioning_enable = bool(shared_conditioning_enable)
        if self.shared_conditioning_enable and not (
            dynamic_cyclic_enable and reliability_enable
        ):
            raise ValueError(
                "Shared conditioning requires dynamic cyclic anchoring and reliability"
            )
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

        from models.IQUMamba1D import MambaLayer

        cumulative_stride = 1
        replaced = 0
        for index, layer in enumerate(self.backbone.encoder.mamba_layers):
            stride = strides[index]
            if isinstance(stride, (list, tuple)):
                stride = stride[0]
            cumulative_stride *= int(stride)
            if isinstance(layer, MambaLayer):
                # RF cycle frequencies live on the temporal axis, so these
                # layers always use temporal patch tokens.
                self.backbone.encoder.mamba_layers[index] = OfficialRFMamba3Layer(
                    int(features_per_stage[index]),
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                    rope_fraction=rope_fraction,
                    is_outproj_norm=is_outproj_norm,
                    chunk_size=chunk_size,
                    token_stride=cumulative_stride,
                    force_real_state=force_real_state,
                    cyclic_anchor_enable=cyclic_anchor_enable,
                    cyclic_frequencies=cyclic_frequencies,
                    cyclic_max_frequency_delta=cyclic_max_frequency_delta,
                    dynamic_cyclic_enable=dynamic_cyclic_enable,
                    reliability_enable=reliability_enable,
                    reliability_hidden=reliability_hidden,
                    reliability_floor=reliability_floor,
                    reliability_init=reliability_init,
                    shared_confidence_enable=self.shared_conditioning_enable,
                )
                replaced += 1
        if replaced == 0:
            raise ValueError("Stage-4 backbone exposed no MambaLayer")
        self.replaced_layers = replaced

        self.input_adapter: nn.Module = nn.Identity()
        self.estimator_config = dict(estimated_cyclofresh_config or {})
        if self.shared_conditioning_enable:
            cfg = self.estimator_config
            self.input_adapter = EstimatedCycloFRESHAdapter1D(
                input_channels=input_channels,
                min_freq=float(cfg.get("estimated_cyclofresh_min_freq", 1 / 64)),
                max_freq=float(cfg.get("estimated_cyclofresh_max_freq", 1 / 8)),
                default_freq=float(cfg.get("estimated_cyclofresh_default_freq", 1 / 32)),
                momentum=float(cfg.get("estimated_cyclofresh_momentum", 0.05)),
                hidden_channels=int(cfg.get("estimated_cyclofresh_hidden_channels", 8)),
                kernel_size=int(cfg.get("estimated_cyclofresh_kernel_size", 9)),
                scale_init=float(cfg.get("estimated_cyclofresh_scale_init", 0.01)),
                gate_hidden=int(cfg.get("estimated_cyclofresh_gate_hidden", 8)),
                zero_init=bool(cfg.get("estimated_cyclofresh_zero_init", True)),
            )
        self.last_cyclic_frequency: torch.Tensor | None = None
        self.last_cyclic_confidence: torch.Tensor | None = None

    @property
    def encoder(self) -> nn.Module:
        return self.backbone.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.backbone.decoder

    def _shared_context(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if not self.shared_conditioning_enable:
            return None, None, x
        cfg = self.estimator_config
        frequency, confidence = estimate_cyclic_frequency_with_confidence(
            x.detach(),
            min_freq=float(cfg.get("estimated_cyclofresh_min_freq", 1 / 64)),
            max_freq=float(cfg.get("estimated_cyclofresh_max_freq", 1 / 8)),
            default_freq=float(cfg.get("estimated_cyclofresh_default_freq", 1 / 32)),
        )
        self.last_cyclic_frequency = frequency.detach()
        self.last_cyclic_confidence = confidence.detach()
        adapted = self.input_adapter.forward_conditioned(
            x, frequency, confidence
        )
        return frequency, confidence, adapted

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        frequency, confidence, x = self._shared_context(x)
        if self.encoder.stem is not None:
            x = self.encoder.stem(x)
        skips = []
        for conv_stage, memory in zip(
            self.encoder.stages, self.encoder.mamba_layers
        ):
            x = conv_stage(x)
            if isinstance(memory, OfficialRFMamba3Layer):
                x = memory(
                    x,
                    cyclic_frequency=frequency,
                    confidence=confidence,
                )
            else:
                x = memory(x)
            skips.append(x)
        return self.decoder(skips)

    def scan_backend_status(self) -> dict[str, str]:
        return {
            f"encoder.mamba_layers.{index}": layer.ssm.last_scan_backend
            for index, layer in enumerate(self.encoder.mamba_layers)
            if isinstance(layer, OfficialRFMamba3Layer)
        }

    def diagnostics(self) -> dict[str, object]:
        diagnostics: dict[str, object] = self.scan_backend_status()
        if self.last_cyclic_frequency is not None:
            diagnostics["cyclic_frequency_mean"] = float(
                self.last_cyclic_frequency.float().mean().cpu()
            )
        if self.last_cyclic_confidence is not None:
            diagnostics["cyclic_confidence_mean"] = float(
                self.last_cyclic_confidence.float().mean().cpu()
            )
        return diagnostics

    def no_weight_decay(self) -> set[str]:
        names = {
            name
            for name, parameter in self.named_parameters()
            if bool(getattr(parameter, "_no_weight_decay", False))
        }
        if self.shared_conditioning_enable:
            names.add("input_adapter.scale")
        return names
