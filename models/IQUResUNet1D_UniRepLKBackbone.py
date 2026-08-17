"""Stage386: Stage381 with stride/channel blocks retained and UniRepLK core blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.IQUResUNet1D_ComplexStateUniRepLK_LatentMask import (
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
)
from models.IQUResUNet1D_NoASC import PlainSkipDecoder
from models.IQUMamba1D import UpsampleLayer
from models.IQUMamba1D_RecentRFModules import build_recent_rf_operator


def _make_unireplk(channels: int, kernel_size: int, ffn_factor: int,
                   layer_scale: float) -> nn.Module:
    """Return the native residual UniRepLK block without an outer adapter."""
    return build_recent_rf_operator(
        "unireplk", channels,
        {"rf_large_kernel": kernel_size, "rf_ffn_factor": ffn_factor,
         "rf_layer_scale": layer_scale},
    )


class UniRepLKIntegratedBlock(nn.Module):
    """One block combining an optional transition convolution and UniRepLK."""

    def __init__(self, input_channels: int, output_channels: int, stride: int,
                 norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                 rf_large_kernel: int, rf_ffn_factor: int, rf_layer_scale: float):
        super().__init__()
        self.transition = None
        if input_channels != output_channels or stride != 1:
            self.transition = _TransitionConv(
                input_channels, output_channels, stride, norm_op,
                norm_op_kwargs, nonlin, nonlin_kwargs,
            )
        self.unireplk = _make_unireplk(
            output_channels, rf_large_kernel, rf_ffn_factor, rf_layer_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.transition is not None:
            x = self.transition(x)
        return self.unireplk(x)


class _TransitionConv(nn.Sequential):
    """The minimum convolution retained for channel or resolution changes."""

    def __init__(self, input_channels: int, output_channels: int, stride: int,
                 norm_op, norm_op_kwargs, nonlin, nonlin_kwargs):
        super().__init__(
            nn.Conv1d(input_channels, output_channels, 3, stride=stride,
                      padding=1, bias=False),
            norm_op(output_channels, **norm_op_kwargs),
            nonlin(**nonlin_kwargs),
        )


class UniRepLKPlainSkipDecoder(PlainSkipDecoder):
    """Keep decoder concat/projection block; replace later same-width blocks."""

    def __init__(self, *args, rf_large_kernel=17, rf_ffn_factor=4,
                 rf_layer_scale=1.0e-6, **kwargs):
        encoder = kwargs.get("encoder", args[0] if args else None)
        n_conv = kwargs.get("n_conv_per_stage")
        super().__init__(*args, **kwargs)
        if isinstance(n_conv, int):
            n_conv = [n_conv] * len(self.stages)
        for index in range(len(self.stages)):
            channels = int(encoder.output_channels[-(index + 2)])
            blocks = [
                _TransitionConv(
                    2 * channels, channels, 1, encoder.norm_op,
                    encoder.norm_op_kwargs, encoder.nonlin,
                    encoder.nonlin_kwargs,
                )
            ]
            blocks.extend(_make_unireplk(
                    int(channels), int(rf_large_kernel), int(rf_ffn_factor),
                    float(rf_layer_scale),
                ) for _ in range(int(n_conv[index]) - 1))
            self.stages[index] = nn.Sequential(*blocks)


class IQUResUNet1D_Stage386(IQUResUNet1D_ComplexStateUniRepLK_LatentMask):
    """Stage381 with all replaceable encoder/decoder residual cores as UniRepLK."""

    def __init__(self, *args, rf_large_kernel=17, rf_ffn_factor=4,
                 rf_layer_scale=1.0e-6, **kwargs):
        # UniRepLK is the main residual-block replacement in this stage; do
        # not also construct Stage381's parallel RF adapters.
        kwargs["rf_apply_stages"] = ()
        super().__init__(*args, **kwargs)
        widths = [int(value) for value in self.encoder.output_channels]
        block_counts = [int(value) for value in kwargs["n_conv_per_stage"]]
        strides = [int(value) for value in kwargs["strides"]]

        # The input-to-feature projection is necessary; the second stem block is not.
        self.encoder.stem = nn.Sequential(
            _TransitionConv(
                int(kwargs["input_channels"]), widths[0], 1,
                self.encoder.norm_op, self.encoder.norm_op_kwargs,
                self.encoder.nonlin, self.encoder.nonlin_kwargs,
            ),
            *[_make_unireplk(widths[0], int(rf_large_kernel), int(rf_ffn_factor),
                            float(rf_layer_scale))
              for _ in range(block_counts[0] - 1)],
        )

        input_channels = widths[0]
        for stage_index in range(len(self.encoder.stages)):
            output_channels = widths[stage_index]
            needs_transition = (
                input_channels != output_channels or strides[stage_index] != 1
            )
            blocks = []
            if needs_transition:
                blocks.append(_TransitionConv(
                    input_channels, output_channels, strides[stage_index],
                    self.encoder.norm_op, self.encoder.norm_op_kwargs,
                    self.encoder.nonlin, self.encoder.nonlin_kwargs,
                ))
                unireplk_count = block_counts[stage_index] - 1
            else:
                unireplk_count = block_counts[stage_index]
            blocks.extend(
                _make_unireplk(output_channels, int(rf_large_kernel),
                              int(rf_ffn_factor), float(rf_layer_scale))
                for _ in range(unireplk_count)
            )
            self.encoder.stages[stage_index] = nn.Sequential(*blocks)
            input_channels = output_channels
        self.decoder = UniRepLKPlainSkipDecoder(
            encoder=self.encoder, num_classes=2,
            n_conv_per_stage=kwargs["n_conv_per_stage_decoder"],
            deep_supervision=False, rf_large_kernel=rf_large_kernel,
            rf_ffn_factor=rf_ffn_factor, rf_layer_scale=rf_layer_scale,
        )


__all__ = ["IQUResUNet1D_Stage386"]


class UniRepLKIntegratedDecoder(nn.Module):
    """Decoder with one integrated upsample/concat/UniRepLK block per level."""

    def __init__(self, encoder, num_classes: int, norm_op, norm_op_kwargs,
                 nonlin, nonlin_kwargs, rf_large_kernel: int,
                 rf_ffn_factor: int, rf_layer_scale: float):
        super().__init__()
        self.encoder = encoder
        self.stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.seg_layers = nn.ModuleList()
        for s in range(1, len(encoder.output_channels)):
            below = int(encoder.output_channels[-s])
            skip = int(encoder.output_channels[-(s + 1)])
            stride = int(encoder.strides[-s][0])
            self.upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op, input_channels=below,
                output_channels=skip, pool_op_kernel_size=stride,
                mode="linear" if encoder.conv_op == nn.Conv1d else "nearest",
            ))
            self.stages.append(UniRepLKIntegratedBlock(
                2 * skip, skip, 1, norm_op, norm_op_kwargs, nonlin,
                nonlin_kwargs, rf_large_kernel, rf_ffn_factor, rf_layer_scale,
            ))
            self.seg_layers.append(encoder.conv_op(skip, num_classes, 1))

    def forward(self, skips):
        x = skips[-1]
        for index, stage in enumerate(self.stages):
            x = self.upsample_layers[index](x)
            x = stage(torch.cat((x, skips[-(index + 2)]), dim=1))
        return self.seg_layers[-1](x)


class IQUResUNet1D_Stage387(IQUResUNet1D_ComplexStateUniRepLK_LatentMask):
    """Stage381 with one integrated UniRepLK block per encoder/decoder level."""

    def __init__(self, *args, rf_large_kernel=17, rf_ffn_factor=4,
                 rf_layer_scale=1.0e-6, **kwargs):
        # Avoid retaining/running Stage381's separate stage_rf branches.
        kwargs["rf_apply_stages"] = ()
        super().__init__(*args, **kwargs)
        widths = [int(value) for value in self.encoder.output_channels]
        strides = [int(value) for value in kwargs["strides"]]
        norm_op = self.encoder.norm_op
        norm_kwargs = self.encoder.norm_op_kwargs
        nonlin = self.encoder.nonlin
        nonlin_kwargs = self.encoder.nonlin_kwargs

        self.encoder.stem = UniRepLKIntegratedBlock(
            int(kwargs["input_channels"]), widths[0], 1, norm_op,
            norm_kwargs, nonlin, nonlin_kwargs, rf_large_kernel,
            rf_ffn_factor, rf_layer_scale,
        )
        input_channels = widths[0]
        for index, output_channels in enumerate(widths):
            self.encoder.stages[index] = UniRepLKIntegratedBlock(
                input_channels, output_channels, strides[index], norm_op,
                norm_kwargs, nonlin, nonlin_kwargs, rf_large_kernel,
                rf_ffn_factor, rf_layer_scale,
            )
            input_channels = output_channels

        self.decoder = UniRepLKIntegratedDecoder(
            self.encoder, num_classes=2, norm_op=norm_op,
            norm_op_kwargs=norm_kwargs, nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs, rf_large_kernel=rf_large_kernel,
            rf_ffn_factor=rf_ffn_factor, rf_layer_scale=rf_layer_scale,
        )


__all__.append("UniRepLKIntegratedBlock")
__all__.append("IQUResUNet1D_Stage387")


class _InvariantRFRouter(nn.Module):
    """Route from phase-invariant energy, envelope, and local variation."""

    def __init__(self, channels: int, experts: int, temperature: float = 1.0):
        super().__init__()
        hidden = max(8, int(channels) // 4)
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Conv1d(3 * int(channels), hidden, 1), nn.SiLU(),
            nn.Conv1d(hidden, int(experts), 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, envelope: torch.Tensor) -> torch.Tensor:
        variation = F.pad((envelope[..., 1:] - envelope[..., :-1]).abs(), (1, 0))
        statistics = torch.cat((envelope.square(), envelope, variation), dim=1)
        logits = self.net(statistics).mean(-1, keepdim=True)
        return torch.softmax(logits / max(self.temperature, 1e-4), dim=1)


class _ComplexDepthwiseConv1d(nn.Module):
    """Strict complex depthwise convolution for paired [real, imag] channels."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        shape = (int(channels), 1, int(kernel_size))
        self.channels = int(channels)
        self.padding = int(kernel_size) // 2
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        nn.init.kaiming_normal_(self.weight_real)
        nn.init.kaiming_normal_(self.weight_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x[:, :self.channels], x[:, self.channels:]
        kwargs = {"padding": self.padding, "groups": self.channels}
        yr = F.conv1d(real, self.weight_real, **kwargs) - F.conv1d(imag, self.weight_imag, **kwargs)
        yi = F.conv1d(real, self.weight_imag, **kwargs) + F.conv1d(imag, self.weight_real, **kwargs)
        return torch.cat((yr, yi), dim=1)


class _ComplexUniRepLKDelta(nn.Module):
    """Strict complex large-kernel, magnitude-gated FFN residual delta."""

    def __init__(self, channels: int, kernel_size: int, ffn_factor: int = 2,
                 layer_scale: float = 1e-6):
        super().__init__()
        from models.icassp_complex_wavenet import ComplexConv1d, ComplexRMSNorm1d
        self.channels = int(channels)
        hidden = max(1, self.channels * int(ffn_factor))
        self.norm = ComplexRMSNorm1d(self.channels)
        self.large_kernel = _ComplexDepthwiseConv1d(self.channels, kernel_size)
        self.ffn1 = ComplexConv1d(self.channels, hidden, kernel_size=1, padding=0)
        self.ffn2 = ComplexConv1d(hidden, self.channels, kernel_size=1, padding=0)
        self.gamma = nn.Parameter(torch.full((self.channels,), float(layer_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.large_kernel(self.norm(x))
        real, imag = z[:, :self.channels], z[:, self.channels:]
        gate = torch.sigmoid(torch.sqrt(real.square() + imag.square() + 1e-6))
        z = self.ffn2(self.ffn1(z))
        scale = self.gamma.view(1, -1, 1)
        return torch.cat((z[:, :self.channels] * gate * scale,
                          z[:, self.channels:] * gate * scale), dim=1)


class ComplexUniRepLKBlock(nn.Module):
    """A complete single-expert complex UniRepLK-style residual block.

    The block operates on paired channels ``[all-real, all-imag]``. Every
    learned transform in the block is complex-valued; only the outer Stage381
    feature path remains real-valued.
    """

    def __init__(self, complex_channels: int, kernel_size: int = 17,
                 ffn_factor: int = 2, layer_scale: float = 1e-2):
        super().__init__()
        self.channels = int(complex_channels)
        self.delta = _ComplexUniRepLKDelta(
            self.channels, int(kernel_size), int(ffn_factor), float(layer_scale)
        )

    def residual_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.delta(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_branch(x)


class AdaptiveComplexUniRepLK(nn.Module):
    """Complex-equivariant large-kernel experts with invariant soft routing."""

    def __init__(self, complex_channels: int, kernels=(9, 17),
                 ffn_factor: int = 2, layer_scale: float = 1e-6):
        super().__init__()
        self.channels = int(complex_channels)
        self.experts = nn.ModuleList([
            _ComplexUniRepLKDelta(self.channels, int(kernel), ffn_factor, layer_scale)
            for kernel in kernels
        ])
        self.router = _InvariantRFRouter(self.channels, len(self.experts))
        self.last_route = None

    def residual_branch(self, x: torch.Tensor) -> torch.Tensor:
        real, imag = x[:, :self.channels], x[:, self.channels:]
        envelope = torch.sqrt(real.square() + imag.square() + 1e-6)
        weights = self.router(envelope)
        deltas = torch.stack([expert(x) for expert in self.experts], dim=1)
        self.last_route = weights.detach()
        return (deltas * weights.unsqueeze(2)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_branch(x)


class AdaptiveRealUniRepLK(nn.Module):
    """Real UniRepLK experts routed only from the current feature statistics."""

    def __init__(self, channels: int, kernels=(9, 17), ffn_factor: int = 2,
                 layer_scale: float = 1e-6):
        super().__init__()
        self.experts = nn.ModuleList([
            _make_unireplk(channels, int(kernel), ffn_factor, layer_scale)
            for kernel in kernels
        ])
        self.router = _InvariantRFRouter(channels, len(self.experts))
        self.last_route = None

    def residual_branch(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.router(torch.sqrt(x.square() + 1e-6))
        deltas = torch.stack([expert.residual_branch(x) for expert in self.experts], dim=1)
        self.last_route = weights.detach()
        return (deltas * weights.unsqueeze(2)).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_branch(x)


class _ComplexParallelAdapter(nn.Module):
    def __init__(self, channels: int, operator: nn.Module, scale_init: float):
        super().__init__()
        from models.icassp_complex_wavenet import ComplexRMSNorm1d
        self.operator = operator
        self.norm = ComplexRMSNorm1d(int(channels) // 2)
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, source: torch.Tensor, main: torch.Tensor) -> torch.Tensor:
        return main + self.residual_scale * self.norm(self.operator.residual_branch(source))


class _Stage381RFBranchAblation(IQUResUNet1D_ComplexStateUniRepLK_LatentMask):
    branch_kind = ""

    def __init__(self, *args, rf_kernels=(9, 17), rf_large_kernel=17,
                 rf_ffn_factor=2, rf_layer_scale=1e-6,
                 complex_layer_scale=1e-2,
                 rf_residual_scale_init=0.05, **kwargs):
        kwargs["rf_apply_stages"] = (0, 2)
        super().__init__(*args, rf_large_kernel=rf_large_kernel,
                         rf_ffn_factor=rf_ffn_factor,
                         rf_layer_scale=rf_layer_scale,
                         rf_residual_scale_init=rf_residual_scale_init, **kwargs)
        from models.IQUMamba1D_RecentRFModules import ParallelFeatureDeltaAdapter
        replacements = {}
        for stage in (0, 2):
            channels = int(self.encoder.output_channels[stage])
            if self.branch_kind == "routed_complex":
                operator = AdaptiveComplexUniRepLK(channels // 2, rf_kernels,
                                                    rf_ffn_factor, rf_layer_scale)
                adapter = _ComplexParallelAdapter(channels, operator, rf_residual_scale_init)
            elif self.branch_kind == "routed_real":
                operator = AdaptiveRealUniRepLK(channels, rf_kernels,
                                                rf_ffn_factor, rf_layer_scale)
                adapter = ParallelFeatureDeltaAdapter(channels, operator, rf_residual_scale_init)
            elif self.branch_kind == "fixed_complex":
                operator = ComplexUniRepLKBlock(
                    channels // 2, rf_large_kernel, rf_ffn_factor,
                    complex_layer_scale,
                )
                adapter = _ComplexParallelAdapter(channels, operator, rf_residual_scale_init)
            else:
                raise ValueError(f"unsupported branch kind: {self.branch_kind}")
            replacements[str(stage)] = adapter
        self.stage_rf = nn.ModuleDict(replacements)


class IQUResUNet1D_Stage388(_Stage381RFBranchAblation):
    """Stage381 with routed receptive fields and complex-equivariant UniRepLK."""
    branch_kind = "routed_complex"


class IQUResUNet1D_Stage389(_Stage381RFBranchAblation):
    """Stage388 ablation: adaptive receptive-field routing only."""
    branch_kind = "routed_real"


class IQUResUNet1D_Stage390(_Stage381RFBranchAblation):
    """Stage388 ablation: fixed-kernel complex-equivariant UniRepLK only."""
    branch_kind = "fixed_complex"


__all__.extend(["ComplexUniRepLKBlock", "AdaptiveComplexUniRepLK", "AdaptiveRealUniRepLK",
                "IQUResUNet1D_Stage388", "IQUResUNet1D_Stage389",
                "IQUResUNet1D_Stage390"])
