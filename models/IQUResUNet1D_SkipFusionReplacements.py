"""Stage181-184 clean skip-fusion replacements for ASC-style decoder skips."""

from typing import Callable, List, Type, Union

import torch
from torch import nn
from torch.nn import functional as F

from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUResUNet1D_NoASC import BasicResBlock, ResidualConvEncoder, UpsampleLayer


def _make_norm(norm_op, channels: int, norm_op_kwargs: dict):
    if norm_op is None:
        return nn.Identity()
    return norm_op(channels, **(norm_op_kwargs or {}))


def _match_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.size(-1) == target_len:
        return x
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)


def _zero_init(module: nn.Module) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.zeros_(module.weight)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.zeros_(module.bias)


class ChannelLSSGDWSkipGate1D(nn.Module):
    """Channel-LSSG with depthwise local context, kept identity-initialized."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        depthwise_kernel_size: int = 5,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 4, 8)
        padding = depthwise_kernel_size // 2

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.depthwise = nn.Conv1d(
            inter_channels,
            inter_channels,
            kernel_size=depthwise_kernel_size,
            padding=padding,
            groups=inter_channels,
            bias=False,
        )
        self.gate = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True)
        self.gate_sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        _zero_init(self.gate)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        fused = F.silu(self.skip_proj(skip) + self.dec_proj(dec))
        gate = self.gate_sigmoid(self.gate(self.depthwise(fused)))
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class DeformableTemporalSkipGate1D(nn.Module):
    """Temporal skip alignment with learned per-position sampling offsets."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        sampling_offsets: int = 4,
        offset_kernel_size: int = 5,
        offset_range: float = 8.0,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)
        self.sampling_offsets = int(sampling_offsets)
        self.offset_range = float(offset_range)
        padding = offset_kernel_size // 2

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.offset_predictor = nn.Conv1d(
            inter_channels,
            self.sampling_offsets,
            kernel_size=offset_kernel_size,
            padding=padding,
            bias=True,
        )
        self.weight_predictor = nn.Conv1d(inter_channels, self.sampling_offsets, kernel_size=1, bias=True)
        self.gate = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        _zero_init(self.offset_predictor)
        _zero_init(self.weight_predictor)
        _zero_init(self.gate)

    @staticmethod
    def _sample_1d(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        B, _, L = x.shape
        if L <= 1:
            return x
        base = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L)
        positions = base + offsets
        norm_x = 2.0 * positions / float(L - 1) - 1.0
        norm_y = torch.zeros_like(norm_x)
        grid = torch.stack((norm_x, norm_y), dim=-1).view(B, 1, L, 2)
        sampled = F.grid_sample(
            x.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(2)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        fused = F.silu(self.skip_proj(skip) + self.dec_proj(dec))
        offsets = torch.tanh(self.offset_predictor(fused)) * self.offset_range
        weights = torch.softmax(self.weight_predictor(fused), dim=1)

        sampled = []
        for idx in range(self.sampling_offsets):
            sampled.append(self._sample_1d(skip, offsets[:, idx, :]))
        stacked = torch.stack(sampled, dim=1)
        aligned = (stacked * weights.unsqueeze(2)).sum(dim=1)

        gate = torch.sigmoid(self.gate(fused))
        self.last_gate = gate
        calibrated = aligned * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class FrequencyAwareSkipGate1D(nn.Module):
    """Frequency-conditioned skip gate using compact FFT-bin descriptors."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        freq_bins: List[int] = None,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()
        if freq_bins is None:
            freq_bins = [1, 2, 4, 8, 16, 32]
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)
        self.freq_bins = [int(item) for item in freq_bins]

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.skip_freq_proj = nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False)
        self.dec_freq_proj = nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False)
        self.gate = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        _zero_init(self.gate)

    def frequency_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        magnitude = spectrum.abs()
        max_bin = magnitude.size(-1) - 1
        bins = [min(max(item, 0), max_bin) for item in self.freq_bins]
        index = torch.tensor(bins, device=x.device, dtype=torch.long)
        descriptor = magnitude.index_select(dim=-1, index=index).mean(dim=-1, keepdim=True)
        return descriptor.to(dtype=x.dtype)

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        freq_context = self.skip_freq_proj(self.frequency_descriptor(skip))
        freq_context = freq_context + self.dec_freq_proj(self.frequency_descriptor(dec))
        freq_context = freq_context.expand(-1, -1, skip.size(-1))
        fused = F.silu(self.skip_proj(skip) + self.dec_proj(dec) + freq_context)
        gate = torch.sigmoid(self.gate(fused))
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class ComplexAwareSkipGate1D(nn.Module):
    """I/Q-pair-inspired magnitude and phase-delta gated skip fusion."""

    def __init__(
        self,
        skip_channels: int,
        dec_channels: int,
        inter_channels: int = None,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        residual_scale_init: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if skip_channels % 2 != 0:
            raise ValueError("ComplexAwareSkipGate1D requires an even skip channel count.")
        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 16)
        self.eps = float(eps)

        self.skip_proj = nn.Sequential(
            nn.Conv1d(skip_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.dec_proj = nn.Sequential(
            nn.Conv1d(dec_channels, inter_channels, kernel_size=1, bias=False),
            _make_norm(norm_op, inter_channels, norm_op_kwargs),
        )
        self.complex_gate = nn.Sequential(
            nn.Conv1d(skip_channels * 2, inter_channels, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Conv1d(inter_channels, skip_channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init)))
        _zero_init(self.complex_gate[-1])
        _zero_init(self.gate)

    def _complex_features(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        skip_i, skip_q = skip[:, 0::2, :], skip[:, 1::2, :]
        dec_i, dec_q = dec[:, 0::2, :], dec[:, 1::2, :]

        skip_magnitude = torch.sqrt(skip_i.square() + skip_q.square() + self.eps)
        dec_magnitude = torch.sqrt(dec_i.square() + dec_q.square() + self.eps)
        skip_phase = torch.atan2(skip_q, skip_i + self.eps)
        dec_phase = torch.atan2(dec_q, dec_i + self.eps)
        phase_delta = skip_phase - dec_phase

        return torch.cat(
            [
                skip_magnitude,
                dec_magnitude,
                torch.sin(phase_delta),
                torch.cos(phase_delta),
            ],
            dim=1,
        )

    def forward(self, skip: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        dec = _match_length(dec, skip.size(-1))
        fused = F.silu(self.skip_proj(skip) + self.dec_proj(dec))
        gate_logits = self.gate(fused) + self.complex_gate(self._complex_features(skip, dec))
        gate = torch.sigmoid(gate_logits)
        self.last_gate = gate
        calibrated = skip * (2.0 * gate)
        return skip + self.alpha * (calibrated - skip)


class SkipFusionUNetResDecoder(nn.Module):
    """U-Net decoder shell with one clean skip-fusion processor per stage."""

    def __init__(
        self,
        encoder,
        num_classes: int,
        n_conv_per_stage: Union[int, List[int]],
        deep_supervision: bool,
        processor_factory: Callable[..., nn.Module],
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = int(num_classes)
        self.deep_supervision = bool(deep_supervision)

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)

        self.upsample_layers = nn.ModuleList()
        self.skip_processors = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s][0]

            self.upsample_layers.append(
                UpsampleLayer(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_below,
                    output_channels=input_features_skip,
                    pool_op_kernel_size=stride_for_upsampling,
                    mode="linear" if encoder.conv_op == nn.Conv1d else "nearest",
                )
            )
            self.skip_processors.append(
                processor_factory(
                    skip_channels=input_features_skip,
                    dec_channels=input_features_skip,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                )
            )

            blocks = [
                BasicResBlock(
                    conv_op=encoder.conv_op,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    input_channels=2 * input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    padding=encoder.conv_pad_sizes[-(s + 1)][0],
                    stride=1,
                    use_1x1conv=True,
                )
            ]
            blocks.extend(
                BasicBlockD(
                    conv_op=encoder.conv_op,
                    input_channels=input_features_skip,
                    output_channels=input_features_skip,
                    kernel_size=encoder.kernel_sizes[-(s + 1)][0],
                    stride=1,
                    conv_bias=encoder.conv_bias,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                )
                for _ in range(n_conv_per_stage[s - 1] - 1)
            )
            self.stages.append(nn.Sequential(*blocks))
            self.seg_layers.append(encoder.conv_op(input_features_skip, num_classes, kernel_size=1))

    def forward(self, skips: List[torch.Tensor]) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = skips[-1]
        seg_outputs = []
        self.aux_loss = 0.0

        for s, stage in enumerate(self.stages):
            x = self.upsample_layers[s](x)
            skip = skips[-(s + 2)]
            x = _match_length(x, skip.size(-1))
            processed_skip = self.skip_processors[s](skip, x)
            if hasattr(self.skip_processors[s], "last_gate"):
                self.aux_loss = self.aux_loss + self.skip_processors[s].last_gate.abs().mean()
            x = stage(torch.cat((x, processed_skip), dim=1))
            seg_outputs.append(self.seg_layers[s](x))

        return seg_outputs[::-1] if self.deep_supervision else seg_outputs[-1]


class IQUResUNet1D_SkipFusionBase(nn.Module):
    def __init__(
        self,
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
        processor_factory: Callable[..., nn.Module],
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}

        self.encoder = ResidualConvEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
        )
        self.decoder = SkipFusionUNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            processor_factory=processor_factory,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.decoder(self.encoder(x))


class IQUResUNet1D_LSSGDWSkip(IQUResUNet1D_SkipFusionBase):
    def __init__(self, *args, residual_scale_init: float = 0.1, depthwise_kernel_size: int = 5, **kwargs):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs):
            return ChannelLSSGDWSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                depthwise_kernel_size=depthwise_kernel_size,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                residual_scale_init=residual_scale_init,
            )

        super().__init__(*args, processor_factory=processor_factory, **kwargs)


class IQUResUNet1D_DeformableTemporalSkip(IQUResUNet1D_SkipFusionBase):
    def __init__(
        self,
        *args,
        residual_scale_init: float = 0.1,
        sampling_offsets: int = 4,
        offset_kernel_size: int = 5,
        offset_range: float = 8.0,
        **kwargs,
    ):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs):
            return DeformableTemporalSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                sampling_offsets=sampling_offsets,
                offset_kernel_size=offset_kernel_size,
                offset_range=offset_range,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                residual_scale_init=residual_scale_init,
            )

        super().__init__(*args, processor_factory=processor_factory, **kwargs)


class IQUResUNet1D_FrequencyAwareSkip(IQUResUNet1D_SkipFusionBase):
    def __init__(self, *args, residual_scale_init: float = 0.1, freq_bins=None, **kwargs):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs):
            return FrequencyAwareSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                freq_bins=freq_bins,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                residual_scale_init=residual_scale_init,
            )

        super().__init__(*args, processor_factory=processor_factory, **kwargs)


class IQUResUNet1D_ComplexAwareSkip(IQUResUNet1D_SkipFusionBase):
    def __init__(self, *args, residual_scale_init: float = 0.1, complex_eps: float = 1e-6, **kwargs):
        def processor_factory(skip_channels, dec_channels, norm_op, norm_op_kwargs):
            return ComplexAwareSkipGate1D(
                skip_channels=skip_channels,
                dec_channels=dec_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                residual_scale_init=residual_scale_init,
                eps=complex_eps,
            )

        super().__init__(*args, processor_factory=processor_factory, **kwargs)
