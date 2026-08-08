"""Controlled complex-valued ablations of the four-stage IQUMamba model.

The five production variants are intentionally adjacent:

* C1 / Stage 290: strict-complex stem, original real Stage-4 backbone.
* C2 / Stage 291: complex convolutional U-Net and ASC, original real Mamba,
  unconstrained real source head.
* C3 / Stage 292: C2 with a strict-complex source head.
* C4 / Stage 293: fully rotation-equivariant convolutional U-Net, ASC and
  an invariant-feature-driven complex-gain Mamba controller (full Stage-4
  width; reads magnitudes plus one-step relative-phase features, emits
  bounded complex gains), with an unconstrained real source head.
* C5 / Stage 294: C4 with a strict-complex source head.

Hidden complex tensors always use the layout ``[all real, all imaginary]``.
Public source tensors use ``[S1-I, S1-Q, S2-I, S2-Q, ...]``.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.icassp_complex_wavenet import (
    ComplexConv1d,
    ComplexModReLU,
    ComplexRMSNorm1d,
)

try:
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - exercised only in lightweight CPU envs
    Mamba = None


def split_complex(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a hidden tensor with ``[R..., I...]`` channel layout."""

    if x.ndim != 3 or x.shape[1] % 2:
        raise ValueError(
            "complex hidden tensors must have shape [B, 2*C, L]"
        )
    return torch.chunk(x, 2, dim=1)


def merge_complex(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    if real.shape != imag.shape:
        raise ValueError("real and imaginary tensors must have equal shapes")
    return torch.cat((real, imag), dim=1)


def complex_cat(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Concatenate complex features without corrupting paired channel layout."""

    pairs = [split_complex(tensor) for tensor in tensors]
    if not pairs:
        raise ValueError("complex_cat requires at least one tensor")
    return merge_complex(
        torch.cat([pair[0] for pair in pairs], dim=1),
        torch.cat([pair[1] for pair in pairs], dim=1),
    )


def hidden_to_public_sources(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[R1..RK,I1..IK]`` to public interleaved source I/Q."""

    real, imag = split_complex(x)
    return torch.stack((real, imag), dim=2).flatten(1, 2)


class ComplexResidualBlock1d(nn.Module):
    """Two-convolution phase-equivariant residual block."""

    def __init__(
        self,
        input_real_channels: int,
        output_real_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if input_real_channels % 2 or output_real_channels % 2:
            raise ValueError("complex residual widths must be even")
        if stride < 1:
            raise ValueError("stride must be positive")
        input_complex = input_real_channels // 2
        output_complex = output_real_channels // 2
        padding = kernel_size // 2

        # ComplexConv1d is stride-one by design.  Strided analysis is performed
        # identically on I/Q after convolution, which preserves phase rotation.
        self.stride = int(stride)
        self.conv1 = ComplexConv1d(
            input_complex,
            output_complex,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = ComplexRMSNorm1d(output_complex, eps=norm_eps)
        self.act1 = ComplexModReLU(output_complex)
        self.conv2 = ComplexConv1d(
            output_complex,
            output_complex,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm2 = ComplexRMSNorm1d(output_complex, eps=norm_eps)
        self.act2 = ComplexModReLU(output_complex)
        self.shortcut = (
            ComplexConv1d(input_complex, output_complex, kernel_size=1)
            if input_complex != output_complex
            else nn.Identity()
        )

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :: self.stride] if self.stride > 1 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._downsample(self.shortcut(x))
        y = self._downsample(self.conv1(x))
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        return self.act2((residual + y) / math.sqrt(2.0))


class ComplexStem1d(nn.Module):
    def __init__(
        self,
        output_real_channels: int,
        *,
        blocks: int = 2,
        kernel_size: int = 3,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("stem requires at least one block")
        layers = [
            ComplexResidualBlock1d(
                2,
                output_real_channels,
                kernel_size=kernel_size,
                norm_eps=norm_eps,
            )
        ]
        layers.extend(
            ComplexResidualBlock1d(
                output_real_channels,
                output_real_channels,
                kernel_size=kernel_size,
                norm_eps=norm_eps,
            )
            for _ in range(blocks - 1)
        )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MagnitudeChannelAttention1d(nn.Module):
    """One real attention weight shared by each complex I/Q pair."""

    def __init__(self, real_channels: int, reduction: int = 8) -> None:
        super().__init__()
        if real_channels % 2:
            raise ValueError("complex attention width must be even")
        channels = real_channels // 2
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = split_complex(x)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        pooled = F.adaptive_avg_pool1d(magnitude, 1)
        pooled = pooled + F.adaptive_max_pool1d(magnitude, 1)
        weight = torch.sigmoid(self.mlp(pooled))
        return merge_complex(real * weight, imag * weight)


class ComplexAdaptiveFusion1d(nn.Module):
    """Magnitude-conditioned, phase-equivariant skip/decoder fusion."""

    def __init__(
        self,
        skip_real_channels: int,
        upsampled_real_channels: int,
    ) -> None:
        super().__init__()
        if skip_real_channels % 2 or upsampled_real_channels % 2:
            raise ValueError("complex fusion widths must be even")
        skip_complex = skip_real_channels // 2
        upsampled_complex = upsampled_real_channels // 2
        hidden = max(1, skip_complex // 4)
        self.weight_generator = nn.Sequential(
            nn.Conv1d(upsampled_complex, hidden, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(hidden, skip_complex, 1),
            nn.Sigmoid(),
        )
        self.cross_interaction = ComplexConv1d(
            skip_complex + upsampled_complex,
            skip_complex,
            kernel_size=1,
        )
        self.norm = ComplexRMSNorm1d(skip_complex)
        self.act = ComplexModReLU(skip_complex)

    def forward(
        self,
        skip: torch.Tensor,
        upsampled: torch.Tensor,
    ) -> torch.Tensor:
        if upsampled.shape[-1] != skip.shape[-1]:
            upsampled = F.interpolate(
                upsampled,
                size=skip.shape[-1],
                mode="linear",
                align_corners=False,
            )
        up_real, up_imag = split_complex(upsampled)
        magnitude = torch.sqrt(up_real.square() + up_imag.square() + 1e-8)
        weight = self.weight_generator(F.adaptive_avg_pool1d(magnitude, 1))
        skip_real, skip_imag = split_complex(skip)
        weighted_skip = merge_complex(skip_real * weight, skip_imag * weight)
        fused = self.cross_interaction(complex_cat((weighted_skip, upsampled)))
        return self.act(self.norm(fused))


class ComplexSkipConnectionProcessor1d(nn.Module):
    """Strict-complex counterpart of the Stage-4 ASC processor."""

    def __init__(self, real_channels: int) -> None:
        super().__init__()
        if real_channels % 2:
            raise ValueError("complex skip width must be even")
        complex_channels = real_channels // 2
        self.feature_align = ComplexConv1d(
            complex_channels, complex_channels, kernel_size=1
        )
        self.align_norm = ComplexRMSNorm1d(complex_channels)
        self.align_act = ComplexModReLU(complex_channels)
        self.channel_attention = MagnitudeChannelAttention1d(real_channels)
        self.adaptive_fusion = ComplexAdaptiveFusion1d(
            real_channels, real_channels
        )
        self.refine1 = ComplexConv1d(
            complex_channels, complex_channels, kernel_size=3, padding=1
        )
        self.refine_norm1 = ComplexRMSNorm1d(complex_channels)
        self.refine_act = ComplexModReLU(complex_channels)
        self.refine2 = ComplexConv1d(
            complex_channels, complex_channels, kernel_size=1
        )
        self.refine_norm2 = ComplexRMSNorm1d(complex_channels)
        self.residual_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        skip: torch.Tensor,
        upsampled: torch.Tensor,
    ) -> torch.Tensor:
        identity = skip
        x = self.align_act(self.align_norm(self.feature_align(skip)))
        x = self.channel_attention(x)
        x = self.adaptive_fusion(x, upsampled)
        x = self.refine_act(self.refine_norm1(self.refine1(x)))
        x = self.refine_norm2(self.refine2(x))
        weight = torch.sigmoid(self.residual_logit)
        return weight * x + (1.0 - weight) * identity


class ComplexUpsample1d(nn.Module):
    def __init__(
        self,
        input_real_channels: int,
        output_real_channels: int,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.projection = ComplexConv1d(
            input_real_channels // 2,
            output_real_channels // 2,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="linear",
            align_corners=False,
        )
        return self.projection(x)


class MagnitudeControlledMamba1d(nn.Module):
    """Globally phase-equivariant Mamba controller with complex gains.

    The controller reads only rotation-invariant features: per-channel
    magnitudes plus the unit-normalized one-step conjugate products
    ``z_t * conj(z_{t-1}) / |z_t||z_{t-1}|`` (cosine/sine of the phase
    increment), so a common phase rotation cannot change its decisions.
    It runs at the full Stage-4 width and emits, per complex channel, a
    bounded complex gain: magnitude factor ``1 + max_gain_delta * tanh(a)``
    and phase rotation ``max_rotation * tanh(b)``.  Multiplying features by
    an invariant complex gain keeps the layer strictly equivariant while
    retaining amplitude *and* phase manipulation capacity.  Strict complex
    projections before and after the controller allow learned complex
    channel mixing without breaking equivariance.
    """

    def __init__(
        self,
        real_channels: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_gain_delta: float = 0.25,
        max_rotation: float = math.pi / 2,
    ) -> None:
        super().__init__()
        if real_channels % 2:
            raise ValueError("complex Mamba width must be even")
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for Stage-4 complex Mamba variants"
            )
        channels = real_channels // 2
        self.pre = ComplexConv1d(channels, channels, kernel_size=1)
        self.in_proj = nn.Conv1d(3 * channels, real_channels, kernel_size=1)
        self.norm = nn.LayerNorm(real_channels)
        self.controller = Mamba(
            d_model=real_channels,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )
        self.gain_proj = nn.Conv1d(real_channels, real_channels, kernel_size=1)
        # Zero-initialized gains start at the exact identity (gain 1,
        # rotation 0); the data-dependent modulation ramps in smoothly.
        nn.init.zeros_(self.gain_proj.weight)
        nn.init.zeros_(self.gain_proj.bias)
        self.post = ComplexConv1d(channels, channels, kernel_size=1)
        self.max_gain_delta = float(max_gain_delta)
        self.max_rotation = float(max_rotation)
        self.residual_logit = nn.Parameter(torch.zeros(()))

    def _invariant_features(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        magnitude: torch.Tensor,
    ) -> torch.Tensor:
        prev_real = F.pad(real, (1, 0), mode="replicate")[..., :-1]
        prev_imag = F.pad(imag, (1, 0), mode="replicate")[..., :-1]
        prev_magnitude = F.pad(magnitude, (1, 0), mode="replicate")[..., :-1]
        scale = 1.0 / (magnitude * prev_magnitude + 1e-8)
        cos_step = (real * prev_real + imag * prev_imag) * scale
        sin_step = (imag * prev_real - real * prev_imag) * scale
        return torch.cat((magnitude, cos_step, sin_step), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre(x)
        real, imag = split_complex(z)
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-8)
        feats = self.in_proj(self._invariant_features(real, imag, magnitude))
        control = self.controller(self.norm(feats.transpose(1, 2)))
        control = self.gain_proj(control.transpose(1, 2))
        gain_logit, rotation_logit = torch.chunk(control, 2, dim=1)
        gain = 1.0 + self.max_gain_delta * torch.tanh(gain_logit)
        rotation = self.max_rotation * torch.tanh(rotation_logit)
        gain_real = gain * torch.cos(rotation)
        gain_imag = gain * torch.sin(rotation)
        controlled = merge_complex(
            real * gain_real - imag * gain_imag,
            real * gain_imag + imag * gain_real,
        )
        residual_scale = torch.sigmoid(self.residual_logit)
        return (
            x + residual_scale * self.post(controlled)
        ) / torch.sqrt(1.0 + residual_scale.square())


class ComplexEncoder1d(nn.Module):
    def __init__(
        self,
        *,
        features_per_stage: Sequence[int],
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        blocks_per_stage: Sequence[int],
        use_equivariant_mamba: bool,
        norm_eps: float,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        mamba_max_gain_delta: float,
        mamba_max_rotation: float = math.pi / 2,
        memory_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        for sequence in (
            features_per_stage,
            kernel_sizes,
            strides,
            blocks_per_stage,
        ):
            if len(sequence) != len(features_per_stage):
                raise ValueError("all encoder stage lists must have equal length")
        if any(int(width) % 2 for width in features_per_stage):
            raise ValueError("all complex feature widths must be even")

        self.stem = ComplexStem1d(
            int(features_per_stage[0]),
            blocks=int(blocks_per_stage[0]),
            kernel_size=int(kernel_sizes[0]),
            norm_eps=norm_eps,
        )
        stages = []
        mamba_layers = []
        input_width = int(features_per_stage[0])
        n_stages = len(features_per_stage)
        for index in range(n_stages):
            output_width = int(features_per_stage[index])
            blocks = [
                ComplexResidualBlock1d(
                    input_width,
                    output_width,
                    kernel_size=int(kernel_sizes[index]),
                    stride=int(strides[index]),
                    norm_eps=norm_eps,
                )
            ]
            blocks.extend(
                ComplexResidualBlock1d(
                    output_width,
                    output_width,
                    kernel_size=int(kernel_sizes[index]),
                    norm_eps=norm_eps,
                )
                for _ in range(int(blocks_per_stage[index]) - 1)
            )
            stages.append(nn.Sequential(*blocks))
            uses_mamba = bool(index % 2) ^ bool(n_stages % 2)
            if not uses_mamba:
                mamba_layers.append(nn.Identity())
            elif memory_factory is not None:
                # A strict-complex memory implementation can be injected here
                # without ever constructing the original real Mamba layer.
                mamba_layers.append(memory_factory(output_width, index))
            elif use_equivariant_mamba:
                mamba_layers.append(
                    MagnitudeControlledMamba1d(
                        output_width,
                        d_state=mamba_d_state,
                        d_conv=mamba_d_conv,
                        expand=mamba_expand,
                        max_gain_delta=mamba_max_gain_delta,
                        max_rotation=mamba_max_rotation,
                    )
                )
            else:
                # The original Stage-4 Mamba intentionally remains an
                # unconstrained real global mixer in C2/C3.
                from models.IQUMamba1D import MambaLayer

                mamba_layers.append(MambaLayer(dim=output_width))
            input_width = output_width

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.output_channels = [int(width) for width in features_per_stage]
        self.strides = [int(stride) for stride in strides]
        self.kernel_sizes = [int(kernel) for kernel in kernel_sizes]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        skips = []
        for stage, mamba in zip(self.stages, self.mamba_layers):
            x = mamba(stage(x))
            skips.append(x)
        return skips


class ComplexDecoder1d(nn.Module):
    def __init__(
        self,
        encoder: ComplexEncoder1d,
        *,
        num_classes: int,
        blocks_per_stage: Sequence[int],
        strict_complex_output: bool,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if num_classes % 2:
            raise ValueError("num_classes must contain complete I/Q pairs")
        n_stages = len(encoder.output_channels)
        if len(blocks_per_stage) < n_stages - 1:
            raise ValueError("decoder block list is shorter than required")

        upsample = []
        skip_processors = []
        stages = []
        for index in range(1, n_stages):
            below_width = encoder.output_channels[-index]
            skip_width = encoder.output_channels[-(index + 1)]
            upsample.append(
                ComplexUpsample1d(
                    below_width,
                    skip_width,
                    encoder.strides[-index],
                )
            )
            skip_processors.append(
                ComplexSkipConnectionProcessor1d(skip_width)
            )
            layers = [
                ComplexResidualBlock1d(
                    2 * skip_width,
                    skip_width,
                    kernel_size=encoder.kernel_sizes[-(index + 1)],
                    norm_eps=norm_eps,
                )
            ]
            layers.extend(
                ComplexResidualBlock1d(
                    skip_width,
                    skip_width,
                    kernel_size=encoder.kernel_sizes[-(index + 1)],
                    norm_eps=norm_eps,
                )
                for _ in range(int(blocks_per_stage[index - 1]) - 1)
            )
            stages.append(nn.Sequential(*layers))

        self.upsample = nn.ModuleList(upsample)
        self.skip_processors = nn.ModuleList(skip_processors)
        self.stages = nn.ModuleList(stages)
        final_width = encoder.output_channels[0]
        self.strict_complex_output = bool(strict_complex_output)
        if self.strict_complex_output:
            self.output_head = ComplexConv1d(
                final_width // 2,
                num_classes // 2,
                kernel_size=1,
            )
        else:
            self.output_head = nn.Conv1d(
                final_width, num_classes, kernel_size=1
            )

    def forward(self, skips: Sequence[torch.Tensor]) -> torch.Tensor:
        x = skips[-1]
        for index, stage in enumerate(self.stages):
            upsampled = self.upsample[index](x)
            skip = self.skip_processors[index](
                skips[-(index + 2)], upsampled
            )
            x = stage(complex_cat((upsampled, skip)))
        output = self.output_head(x)
        return (
            hidden_to_public_sources(output)
            if self.strict_complex_output
            else output
        )


class IQUMamba1DComplexStage4(nn.Module):
    """C2-C5 complex Stage-4 backbone."""

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
        strict_complex_output: bool = False,
        use_equivariant_mamba: bool = False,
        complex_norm_eps: float = 1e-6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_max_gain_delta: float = 0.25,
        mamba_max_rotation: float = math.pi / 2,
        **_: object,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("complex Stage-4 expects one I/Q mixture")
        if n_stages != len(features_per_stage):
            raise ValueError("n_stages does not match features_per_stage")
        self.input_size = int(input_size)
        self.use_equivariant_mamba = bool(use_equivariant_mamba)
        self.strict_complex_output = bool(strict_complex_output)
        self.encoder = ComplexEncoder1d(
            features_per_stage=features_per_stage,
            kernel_sizes=kernel_sizes,
            strides=strides,
            blocks_per_stage=n_conv_per_stage,
            use_equivariant_mamba=use_equivariant_mamba,
            norm_eps=complex_norm_eps,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_max_gain_delta=mamba_max_gain_delta,
            mamba_max_rotation=mamba_max_rotation,
        )
        self.decoder = ComplexDecoder1d(
            self.encoder,
            num_classes=num_classes,
            blocks_per_stage=n_conv_per_stage_decoder,
            strict_complex_output=strict_complex_output,
            norm_eps=complex_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def no_weight_decay(self) -> set[str]:
        names = set()
        for module_name, module in self.named_modules():
            if isinstance(module, ComplexRMSNorm1d):
                names.add(f"{module_name}.log_scale")
            if isinstance(module, ComplexModReLU):
                names.add(f"{module_name}.bias")
            if isinstance(module, MagnitudeControlledMamba1d):
                names.add(f"{module_name}.residual_logit")
        return names


class IQUMamba1DComplexStemC1(nn.Module):
    """C1: replace only the original Stage-4 stem with a strict-complex stem."""

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
        complex_norm_eps: float = 1e-6,
        **_: object,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError("C1 expects one I/Q mixture")
        from models.IQUMamba1D import IQUMamba1D

        self.backbone = IQUMamba1D(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=list(features_per_stage),
            conv_op=nn.Conv1d,
            kernel_sizes=list(kernel_sizes),
            strides=list(strides),
            n_conv_per_stage=list(n_conv_per_stage),
            num_classes=num_classes,
            n_conv_per_stage_decoder=list(n_conv_per_stage_decoder),
            deep_supervision=False,
        )
        self.backbone.encoder.stem = ComplexStem1d(
            int(features_per_stage[0]),
            blocks=int(n_conv_per_stage[0]),
            kernel_size=int(kernel_sizes[0]),
            norm_eps=complex_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def no_weight_decay(self) -> set[str]:
        names = set()
        for module_name, module in self.named_modules():
            if isinstance(module, ComplexRMSNorm1d):
                names.add(f"{module_name}.log_scale")
            if isinstance(module, ComplexModReLU):
                names.add(f"{module_name}.bias")
        return names
