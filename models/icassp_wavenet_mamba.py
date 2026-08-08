"""Mamba-augmented ICASSP WaveNet variants for IQ source separation.

Both models preserve the public ICASSP 2024 WaveNet backbone exactly:
the input projection, all gated dilated residual blocks, skip aggregation,
and output head are unchanged.  The only architectural change is a residual
Mamba fusion block after the normalized skip sum and before ``skip_projection``.

``ICASPBaselineWaveNetMamba`` uses one causal/unidirectional selective scan.
``ICASPBaselineWaveNetBiMamba`` uses independent forward and reversed scans,
then projects their concatenated features back to the WaveNet channel width.
The latter is intended for the offline IQUMamba separation task, not streaming
inference.

Stages 259--262 use a lower-cost multi-rate variant.  They compress a WaveNet
feature map, apply a learned stride-4 analysis filter, run Mamba at one quarter
of the original sequence length, then reconstruct a residual context feature.
Stages 259/260 apply this after a 10-block WaveNet; Stages 261/262 inject it
between two 10-block WaveNet dilation cycles.

Stages 265--267 keep the successful Stage-261 unidirectional local-context
path intact and add one independently gated enhancement at the insertion
point: a Stage-235-style bidirectional cross-scale memory, a Stage-255-style
physical-evidence MoE, or the Stage-79 estimated Cyclo-FRESH input prior.
They are architectural adaptations of those ideas to WaveNet, rather than
claims of checkpoint-compatible copies of the original U-Net BiMamba stages.

Stages 269--271 explore Mamba controls and a physically grounded reverse
context rather than Stage 262's latent pure-flip reverse scan.

Stages 276/277 make the local/global division explicit.  WaveNet is restricted
to one local dilation cycle on each side of a chunk-token Mamba.  Overlapping
chunk statistics retain temporal order, and an energy-normalized strong fusion
starts with equal local/Mamba gain instead of a near-identity residual scale.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icassp_baseline_wavenet import ICASPBaselineWaveNet

try:
    from mamba_ssm import Mamba
except ImportError:  # Keep the error at model construction, not module import.
    Mamba = None


class WaveNetMambaSkipFusion(nn.Module):
    """Pre-normalized residual Mamba fusion for a WaveNet skip tensor.

    Args:
        channels: WaveNet residual/skip channel width.
        bidirectional: If true, process the full sequence in both directions
            and project concatenated features back to ``channels``.
        scale_init: Small nonzero residual gate.  It keeps the initial model
            close to its WaveNet backbone while allowing Mamba parameters to
            receive gradients as soon as the WaveNet output head has warmed up.
    """

    def __init__(
        self,
        channels: int,
        *,
        bidirectional: bool,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        scale_init: float = 1e-2,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for the ICASSP WaveNet-Mamba variants. "
                "Install the project-compatible mamba-ssm build first."
            )

        self.bidirectional = bool(bidirectional)
        self.norm = nn.LayerNorm(channels)
        self.forward_mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        if self.bidirectional:
            self.backward_mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            projection_in_channels = 2 * channels
        else:
            self.backward_mamba = None
            projection_in_channels = channels

        # This mirrors the forward/backward concatenate -> 1x1 projection
        # pattern used by bidirectional Mamba speech-separation systems.
        self.projection = nn.Linear(projection_in_channels, channels)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse global sequence context into ``x`` with shape ``(B, C, L)``."""

        sequence = self.norm(x.transpose(1, 2))  # (B, L, C)
        forward_features = self.forward_mamba(sequence)

        if self.bidirectional:
            reverse_sequence = torch.flip(sequence, dims=[1])
            backward_features = self.backward_mamba(reverse_sequence)
            backward_features = torch.flip(backward_features, dims=[1])
            features = torch.cat((forward_features, backward_features), dim=-1)
        else:
            features = forward_features

        delta = self.projection(features).transpose(1, 2)  # (B, C, L)
        return x + self.residual_scale * self.dropout(delta)


class MambaSequenceMixer(nn.Module):
    """Mamba sequence mixer that returns a projected context delta.

    This is deliberately separate from ``WaveNetMambaSkipFusion`` so existing
    Stage-257/258 checkpoints retain their parameter layout.  The multi-rate
    variants below need only the Mamba-derived delta, not an inner residual.
    """

    def __init__(
        self,
        channels: int,
        *,
        bidirectional: bool,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for the ICASSP WaveNet-Mamba variants. "
                "Install the project-compatible mamba-ssm build first."
            )

        self.bidirectional = bool(bidirectional)
        self.norm = nn.LayerNorm(channels)
        self.forward_mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if self.bidirectional:
            self.backward_mamba = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            projection_in_channels = 2 * channels
        else:
            self.backward_mamba = None
            projection_in_channels = channels

        self.projection = nn.Linear(projection_in_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a context delta with the same shape as ``x``: ``(B, C, L)``."""

        sequence = self.norm(x.transpose(1, 2))
        forward_features = self.forward_mamba(sequence)
        if self.bidirectional:
            reverse_sequence = torch.flip(sequence, dims=[1])
            backward_features = self.backward_mamba(reverse_sequence)
            backward_features = torch.flip(backward_features, dims=[1])
            features = torch.cat((forward_features, backward_features), dim=-1)
        else:
            features = forward_features
        return self.projection(features).transpose(1, 2)


class GatedDirectionalMambaSequenceMixer(nn.Module):
    """Stage-261-preserving bidirectional Mamba correction.

    A plain bidirectional Mamba concatenates independently learned forward and
    reverse scans before projecting them together.  For the WaveNet insertion
    point this can overwrite the useful directional representation learned by
    Stage 261.  This mixer instead keeps the Stage-261 forward scan as the
    base and adds only a bounded, per-channel correction from the directional
    difference ``backward - forward``.

    At ``backward_gate_init=0`` its parameter names and forward computation
    are compatible with a unidirectional :class:`MambaSequenceMixer` checkpoint
    for all shared parameters.  A small nonzero initialization is useful for
    from-scratch runs because it gives the reverse branch immediate gradients.
    """

    def __init__(
        self,
        channels: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        backward_gate_init: float = 0.0,
    ) -> None:
        super().__init__()

        if Mamba is None:
            raise ImportError(
                "mamba_ssm is required for the ICASSP WaveNet-Mamba variants. "
                "Install the project-compatible mamba-ssm build first."
            )

        self.norm = nn.LayerNorm(channels)
        # Keep these names identical to the Stage-261 mixer so a Stage-261
        # state dict can initialize the forward path without key rewriting.
        self.forward_mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.projection = nn.Linear(channels, channels)
        self.backward_gate_raw = nn.Parameter(
            torch.full((channels,), float(backward_gate_init))
        )

    def backward_gate_values(self) -> torch.Tensor:
        """Return bounded per-channel correction strengths in ``[-1, 1]``."""

        return torch.tanh(self.backward_gate_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a Stage-261 base plus a gated reverse-scan correction."""

        sequence = self.norm(x.transpose(1, 2))
        forward_features = self.forward_mamba(sequence)
        reverse_sequence = torch.flip(sequence, dims=[1])
        backward_features = self.backward_mamba(reverse_sequence)
        backward_features = torch.flip(backward_features, dims=[1])

        gate = self.backward_gate_values().to(
            device=forward_features.device,
            dtype=forward_features.dtype,
        ).view(1, 1, -1)
        features = forward_features + gate * (backward_features - forward_features)
        return self.projection(features).transpose(1, 2)


class MultiRateMambaContext(nn.Module):
    """Learned multi-rate Mamba residual context for full-rate WaveNet features.

    A pointwise projection followed by a depthwise learned analysis filter
    reduces the sequence by ``downsample_factor``.  Mamba therefore models
    global context at a substantially lower token count while a full-rate
    residual path preserves local I/Q waveform details.
    """

    def __init__(
        self,
        channels: int,
        *,
        mamba_channels: int = 64,
        downsample_factor: int = 4,
        bidirectional: bool,
        directional_gate: bool = False,
        directional_backward_gate_init: float = 0.0,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        scale_init: float = 1e-2,
    ):
        super().__init__()

        self.downsample_factor = int(downsample_factor)
        if self.downsample_factor < 2 or self.downsample_factor % 2:
            raise ValueError(
                "downsample_factor must be an even integer >= 2 so the learned "
                "analysis/synthesis filters preserve the padded sequence length."
            )
        if int(mamba_channels) < 1:
            raise ValueError("mamba_channels must be positive")

        kernel_size = 2 * self.downsample_factor
        padding = self.downsample_factor // 2
        self.input_projection = nn.Conv1d(channels, mamba_channels, kernel_size=1)
        # Pointwise projections mix feature channels.  These depthwise learned
        # filters then perform inexpensive temporal analysis/synthesis.
        self.downsample = nn.Conv1d(
            mamba_channels,
            mamba_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=mamba_channels,
            bias=False,
        )
        if directional_gate:
            if bidirectional:
                raise ValueError(
                    "directional_gate is an asymmetric BiMamba replacement; "
                    "set bidirectional=False."
                )
            self.mamba = GatedDirectionalMambaSequenceMixer(
                mamba_channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                backward_gate_init=directional_backward_gate_init,
            )
        else:
            self.mamba = MambaSequenceMixer(
                mamba_channels,
                bidirectional=bidirectional,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        self.upsample = nn.ConvTranspose1d(
            mamba_channels,
            mamba_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=mamba_channels,
            bias=False,
        )
        self.output_projection = nn.Conv1d(mamba_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inject multi-rate Mamba context while preserving ``x`` as a residual."""

        original_length = x.shape[-1]
        right_padding = (-original_length) % self.downsample_factor

        z = self.input_projection(x)
        if right_padding:
            z = F.pad(z, (0, right_padding))
        padded_length = z.shape[-1]

        z = self.downsample(z)
        z = self.mamba(z)
        z = self.upsample(z)
        # For a padded multiple of the stride the transposed convolution is
        # exact.  Keep this fallback for unusual input lengths/configs.
        if z.shape[-1] != padded_length:
            z = F.interpolate(z, size=padded_length, mode="linear", align_corners=False)
        z = z[..., :original_length]
        delta = self.output_projection(z)
        return x + self.residual_scale * self.dropout(delta)


class ChannelRMSNorm1d(nn.Module):
    """RMS-normalize each time step across channels without mean subtraction."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        if int(channels) < 1:
            raise ValueError("channels must be positive")
        if float(eps) <= 0:
            raise ValueError("eps must be positive")
        self.weight = nn.Parameter(torch.ones(int(channels)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"ChannelRMSNorm1d expects (B, C, L), got {tuple(x.shape)}"
            )
        rms = x.float().square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
        normalized = x / rms.to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype).view(1, -1, 1)


class ChunkMambaStrongFusion(nn.Module):
    """Model ordered overlapping chunks and strongly fuse global/local features.

    The tokenizer keeps one token per overlapping chunk by concatenating the
    chunk mean and standard deviation.  Unlike Stage 269, these statistics are
    not pooled over the full frame, so Mamba retains the ordered temporal
    sequence.  Mamba tokens are linearly expanded to the original resolution
    as smooth global context; no transposed-convolution waveform reconstruction
    is used.

    Local and global features are independently RMS-normalized and fused with
    a positive per-channel gain:

        fused = (local + gain * global) / sqrt(1 + gain**2)

    ``gain`` starts at 1 by default, giving both paths equal energy and useful
    gradients from the first optimization step.
    """

    def __init__(
        self,
        channels: int,
        *,
        mamba_channels: int = 64,
        chunk_size: int = 64,
        chunk_hop: int = 32,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        fusion_gain_init: float = 1.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mamba_channels = int(mamba_channels)
        self.chunk_size = int(chunk_size)
        self.chunk_hop = int(chunk_hop)
        self.norm_eps = float(norm_eps)

        if self.channels < 1 or self.mamba_channels < 1:
            raise ValueError("channels and mamba_channels must be positive")
        if self.chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if not 1 <= self.chunk_hop <= self.chunk_size:
            raise ValueError("chunk_hop must be in [1, chunk_size]")
        if float(fusion_gain_init) <= 0:
            raise ValueError("fusion_gain_init must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

        self.token_projection = nn.Conv1d(
            2 * self.channels,
            self.mamba_channels,
            kernel_size=1,
        )
        self.mamba = MambaSequenceMixer(
            self.mamba_channels,
            bidirectional=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.context_projection = nn.Conv1d(
            self.mamba_channels,
            self.channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.local_norm = ChannelRMSNorm1d(self.channels, eps=self.norm_eps)
        self.global_norm = ChannelRMSNorm1d(self.channels, eps=self.norm_eps)

        gain_init = torch.full((self.channels,), float(fusion_gain_init))
        gain_raw = torch.log(torch.expm1(gain_init))
        self.fusion_gain_raw = nn.Parameter(gain_raw)

    def fusion_gain_values(self) -> torch.Tensor:
        """Return positive per-channel global-context gains."""

        return F.softplus(self.fusion_gain_raw)

    def _ordered_chunk_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 2*C, K)`` mean/std tokens for overlapping chunks."""

        length = int(x.shape[-1])
        if length < self.chunk_size:
            right_padding = self.chunk_size - length
        else:
            remainder = (length - self.chunk_size) % self.chunk_hop
            right_padding = (self.chunk_hop - remainder) % self.chunk_hop
        if right_padding:
            x = F.pad(x, (0, right_padding))

        chunks = x.unfold(
            dimension=-1,
            size=self.chunk_size,
            step=self.chunk_hop,
        )
        chunk_mean = chunks.mean(dim=-1)
        chunk_variance = (
            chunks.float()
            .sub(chunk_mean.float().unsqueeze(-1))
            .square()
            .mean(dim=-1)
        )
        chunk_std = chunk_variance.add(self.norm_eps).sqrt().to(dtype=x.dtype)
        return torch.cat((chunk_mean, chunk_std), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != self.channels:
            raise ValueError(
                "ChunkMambaStrongFusion expects "
                f"(B, {self.channels}, L), got {tuple(x.shape)}"
            )

        original_length = int(x.shape[-1])
        statistics = self._ordered_chunk_statistics(x)
        tokens = self.token_projection(statistics)
        global_tokens = self.mamba(tokens)
        global_context = self.context_projection(global_tokens)
        if global_context.shape[-1] == 1:
            global_context = global_context.expand(-1, -1, original_length)
        else:
            global_context = F.interpolate(
                global_context,
                size=original_length,
                mode="linear",
                align_corners=False,
            )
        global_context = self.dropout(global_context)

        local_rms = (
            x.float()
            .square()
            .mean(dim=1, keepdim=True)
            .add(self.norm_eps)
            .sqrt()
            .to(dtype=x.dtype)
        )
        local_feature = self.local_norm(x)
        global_feature = self.global_norm(global_context)
        gain = self.fusion_gain_values().to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, -1, 1)
        fused = (local_feature + gain * global_feature) / torch.sqrt(
            1.0 + gain.square()
        )
        return fused * local_rms


class ICASPBaselineWaveNetChunkMambaStrongFusion(ICASPBaselineWaveNet):
    """WaveNet local stack -> chunk Mamba strong fusion -> local stack."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 64,
        residual_layers: int = 10,
        dilation_cycle_length: int = 5,
        *,
        mamba_insert_after_block: int = 5,
        mamba_channels: int = 64,
        mamba_chunk_size: int = 64,
        mamba_chunk_hop: int = 32,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_fusion_gain_init: float = 1.0,
        mamba_fusion_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.mamba_insert_after_block = int(mamba_insert_after_block)
        if not 1 <= self.mamba_insert_after_block < self.num_layers:
            raise ValueError(
                "mamba_insert_after_block must be between 1 and residual_layers - 1"
            )
        self.chunk_mamba_fusion = ChunkMambaStrongFusion(
            residual_channels,
            mamba_channels=mamba_channels,
            chunk_size=mamba_chunk_size,
            chunk_hop=mamba_chunk_hop,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            fusion_gain_init=mamba_fusion_gain_init,
            norm_eps=mamba_fusion_norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))
        skip_sum = None
        for block_index, block in enumerate(self.residual_blocks, start=1):
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
            if block_index == self.mamba_insert_after_block:
                x = self.chunk_mamba_fusion(x)

        x = skip_sum / sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)

    def no_weight_decay(self) -> set[str]:
        return {"chunk_mamba_fusion.fusion_gain_raw"}


class MultiRateMambaGlobalMemory(nn.Module):
    """Compressed bidirectional Mamba memory for cross-scale WaveNet fusion.

    Unlike :class:`MultiRateMambaContext`, this module intentionally returns
    the low-rate token sequence instead of reconstructing a full-rate residual.
    The calling attention module can therefore use it as a compact K/V bank.
    Keeping this as a separate class leaves the parameter layout of Stages
    259--262 unchanged.
    """

    def __init__(
        self,
        channels: int,
        *,
        mamba_channels: int = 64,
        downsample_factor: int = 4,
        bidirectional: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()

        self.downsample_factor = int(downsample_factor)
        if self.downsample_factor < 2 or self.downsample_factor % 2:
            raise ValueError(
                "downsample_factor must be an even integer >= 2 so the learned "
                "analysis filter preserves a predictable token count."
            )
        if int(mamba_channels) < 1:
            raise ValueError("mamba_channels must be positive")

        kernel_size = 2 * self.downsample_factor
        padding = self.downsample_factor // 2
        self.input_projection = nn.Conv1d(channels, mamba_channels, kernel_size=1)
        self.downsample = nn.Conv1d(
            mamba_channels,
            mamba_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=mamba_channels,
            bias=False,
        )
        self.mamba = MambaSequenceMixer(
            mamba_channels,
            bidirectional=bidirectional,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return low-rate global-memory tokens shaped ``(B, C_m, ceil(L/r))``."""

        right_padding = (-x.shape[-1]) % self.downsample_factor
        z = self.input_projection(x)
        if right_padding:
            z = F.pad(z, (0, right_padding))
        z = self.downsample(z)
        return self.mamba(z)


class MultiRateMambaController(nn.Module):
    """Use low-rate Mamba context to emit sample-specific WaveNet controls.

    The controller is deliberately separate from ``MultiRateMambaContext``:
    it never adds a waveform-feature residual.  Its final projection is zero
    initialized, so every controlled WaveNet starts as the unmodified WaveNet
    and learns only the controls that lower the separation loss.
    """

    def __init__(
        self,
        channels: int,
        output_dim: int,
        *,
        mamba_channels: int = 64,
        downsample_factor: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        hidden_channels: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.downsample_factor = int(downsample_factor)
        if self.downsample_factor < 2 or self.downsample_factor % 2:
            raise ValueError("downsample_factor must be an even integer >= 2")

        kernel_size = 2 * self.downsample_factor
        padding = self.downsample_factor // 2
        self.input_projection = nn.Conv1d(channels, mamba_channels, kernel_size=1)
        self.downsample = nn.Conv1d(
            mamba_channels,
            mamba_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=mamba_channels,
            bias=False,
        )
        self.mamba = MambaSequenceMixer(
            mamba_channels,
            bidirectional=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        summary_channels = 2 * int(mamba_channels)
        self.summary_norm = nn.LayerNorm(summary_channels)
        self.controller = nn.Sequential(
            nn.Linear(summary_channels, int(hidden_channels)),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(hidden_channels), int(output_dim)),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        right_padding = (-x.shape[-1]) % self.downsample_factor
        z = self.input_projection(x)
        if right_padding:
            z = F.pad(z, (0, right_padding))
        z = self.downsample(z)
        z = self.mamba(z)
        summary = torch.cat(
            [z.mean(dim=-1), z.std(dim=-1, unbiased=False)],
            dim=1,
        )
        return self.controller(self.summary_norm(summary))


class PhaseAwareReverseMambaContext(nn.Module):
    """Physical reverse-scan context built from ``conj(flip(I/Q))``.

    The ordinary Stage-262 reverse scan flips latent real features, which does
    not preserve a complex waveform's phase evolution.  This branch instead
    reverses the raw waveform and negates Q before learned analysis, then maps
    the reverse Mamba output back to forward-time WaveNet features.
    """

    def __init__(
        self,
        feature_channels: int,
        *,
        mamba_channels: int = 64,
        downsample_factor: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.downsample_factor = int(downsample_factor)
        if self.downsample_factor < 2 or self.downsample_factor % 2:
            raise ValueError("downsample_factor must be an even integer >= 2")

        kernel_size = 2 * self.downsample_factor
        padding = self.downsample_factor // 2
        self.input_projection = nn.Conv1d(2, mamba_channels, kernel_size=1)
        self.downsample = nn.Conv1d(
            mamba_channels,
            mamba_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=mamba_channels,
            bias=False,
        )
        self.mamba = MambaSequenceMixer(
            mamba_channels,
            bidirectional=False,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.upsample = nn.ConvTranspose1d(
            mamba_channels,
            mamba_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=padding,
            groups=mamba_channels,
            bias=False,
        )
        self.output_projection = nn.Conv1d(
            mamba_channels,
            feature_channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.residual_scale = nn.Parameter(torch.tensor(float(scale_init)))

    @staticmethod
    def physical_reverse(mixture: torch.Tensor) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                "PhaseAwareReverseMambaContext expects a raw mixture shaped "
                f"(B, 2, L), got {tuple(mixture.shape)}"
            )
        reversed_iq = torch.flip(mixture, dims=[-1])
        return torch.cat([reversed_iq[:, 0:1], -reversed_iq[:, 1:2]], dim=1)

    def forward(self, feature: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        original_length = mixture.shape[-1]
        right_padding = (-original_length) % self.downsample_factor
        z = self.input_projection(self.physical_reverse(mixture))
        if right_padding:
            z = F.pad(z, (0, right_padding))
        padded_length = z.shape[-1]
        z = self.downsample(z)
        z = self.mamba(z)
        z = self.upsample(z)
        if z.shape[-1] != padded_length:
            z = F.interpolate(z, size=padded_length, mode="linear", align_corners=False)
        delta_reverse = self.output_projection(z[..., :original_length])
        # Feature channels do not have fixed I/Q axes, so only reverse time
        # here; the learned projection maps the conjugate-domain context into
        # the forward WaveNet feature basis.
        delta_forward = torch.flip(delta_reverse, dims=[-1])
        return feature + self.residual_scale * self.dropout(delta_forward)


class _ICASPBaselineWaveNetMambaControlBase(ICASPBaselineWaveNet):
    """WaveNet whose second dilation cycle is conditioned, not overwritten, by Mamba."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 64,
        residual_layers: int = 20,
        dilation_cycle_length: int = 10,
        *,
        mamba_insert_after_block: int = 10,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_controller_hidden: int = 128,
        mamba_controller_dropout: float = 0.0,
        controls_per_block: int,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.mamba_insert_after_block = int(mamba_insert_after_block)
        if not 1 <= self.mamba_insert_after_block < self.num_layers:
            raise ValueError(
                "mamba_insert_after_block must be between 1 and residual_layers - 1"
            )
        self.controlled_block_count = self.num_layers - self.mamba_insert_after_block
        self.controls_per_block = int(controls_per_block)
        self.mamba_controller = MultiRateMambaController(
            residual_channels,
            output_dim=self.controlled_block_count * self.controls_per_block,
            mamba_channels=mamba_channels,
            downsample_factor=mamba_downsample_factor,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            hidden_channels=mamba_controller_hidden,
            dropout=mamba_controller_dropout,
        )

    def _apply_block_controls(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        skip: torch.Tensor,
        controls: torch.Tensor,
        controlled_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))
        skip_sum = None
        controls = None
        for block_index, block in enumerate(self.residual_blocks, start=1):
            previous = x
            candidate, skip = block(x)
            if block_index == self.mamba_insert_after_block:
                controls = self.mamba_controller(candidate)
                x = candidate
            elif block_index > self.mamba_insert_after_block:
                if controls is None:
                    raise RuntimeError("Mamba controls were not initialized")
                x, skip = self._apply_block_controls(
                    previous,
                    candidate,
                    skip,
                    controls,
                    block_index - self.mamba_insert_after_block - 1,
                )
            else:
                x = candidate
            skip_sum = skip if skip_sum is None else skip_sum + skip

        x = skip_sum / sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)


class ICASPBaselineWaveNetMambaFiLMController(_ICASPBaselineWaveNetMambaControlBase):
    """Stage 269: Mamba emits FiLM, residual and skip controls for blocks 11--20."""

    def __init__(
        self,
        *args,
        mamba_control_gate_max_delta: float = 0.5,
        mamba_control_film_max_delta: float = 0.1,
        **kwargs,
    ) -> None:
        residual_channels = int(kwargs.get("residual_channels", 64))
        super().__init__(
            *args,
            controls_per_block=4 * residual_channels,
            **kwargs,
        )
        self.control_gate_max_delta = float(mamba_control_gate_max_delta)
        self.control_film_max_delta = float(mamba_control_film_max_delta)
        if self.control_gate_max_delta < 0 or self.control_film_max_delta < 0:
            raise ValueError("Mamba control limits must be non-negative")

    def _apply_block_controls(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        skip: torch.Tensor,
        controls: torch.Tensor,
        controlled_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _ = candidate.shape
        controls = controls.view(
            batch,
            self.controlled_block_count,
            4,
            channels,
        )
        residual_raw, skip_raw, gamma_raw, beta_raw = controls[
            :, controlled_index
        ].unbind(dim=1)
        residual_gate = 1.0 + self.control_gate_max_delta * torch.tanh(residual_raw)
        skip_gate = 1.0 + self.control_gate_max_delta * torch.tanh(skip_raw)
        gamma = self.control_film_max_delta * torch.tanh(gamma_raw)
        beta = self.control_film_max_delta * torch.tanh(beta_raw)

        # Preserve an exact ordinary WaveNet block at zero controller output.
        residual_update = sqrt(2.0) * candidate - previous
        x = (
            previous
            + residual_gate.unsqueeze(-1) * residual_update
        ) / sqrt(2.0)
        rms = previous.detach().square().mean(dim=(1, 2), keepdim=True).sqrt()
        x = x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1) * rms
        skip = skip * skip_gate.unsqueeze(-1)
        return x, skip


class ICASPBaselineWaveNetMambaDilationSkipRouter(_ICASPBaselineWaveNetMambaControlBase):
    """Stage 270: Mamba dynamically routes the second dilation-cycle paths.

    The convolutional dilations remain fixed, but each of the second-cycle
    residual/skip paths has its own sample-conditioned gain.  This is an
    efficient dynamic-dilation router: the model selects the useful receptive
    field scales without instantiating parallel convolutions at every dilation.
    """

    def __init__(
        self,
        *args,
        mamba_router_strength: float = 0.25,
        mamba_router_temperature: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, controls_per_block=2, **kwargs)
        self.router_strength = float(mamba_router_strength)
        self.router_temperature = float(mamba_router_temperature)
        if not 0.0 <= self.router_strength <= 1.0:
            raise ValueError("mamba_router_strength must be in [0, 1]")
        if self.router_temperature <= 0.0:
            raise ValueError("mamba_router_temperature must be positive")

    def _apply_block_controls(
        self,
        previous: torch.Tensor,
        candidate: torch.Tensor,
        skip: torch.Tensor,
        controls: torch.Tensor,
        controlled_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = candidate.shape[0]
        controls = controls.view(batch, self.controlled_block_count, 2)
        logits = controls / self.router_temperature

        # The late WaveNet cycle already contains one path for each dilation.
        # Normalized weights make those paths compete per frame, while the
        # multiplication by the number of blocks keeps all gains exactly one
        # at the zero-initialized controller output.
        route = torch.softmax(logits, dim=1) * self.controlled_block_count
        route = 1.0 + self.router_strength * (route - 1.0)
        residual_gate = route[:, controlled_index, 0]
        skip_gate = route[:, controlled_index, 1]
        residual_update = sqrt(2.0) * candidate - previous
        x = (
            previous
            + residual_gate.view(-1, 1, 1) * residual_update
        ) / sqrt(2.0)
        return x, skip * skip_gate.view(-1, 1, 1)


class _ICASPBaselineWaveNetMambaBase(ICASPBaselineWaveNet):
    """Shared WaveNet backbone with one post-skip Mamba fusion block."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 64,
        residual_layers: int = 30,
        dilation_cycle_length: int = 10,
        *,
        bidirectional: bool,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_scale_init: float = 1e-2,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.skip_mamba = WaveNetMambaSkipFusion(
            residual_channels,
            bidirectional=bidirectional,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            scale_init=mamba_scale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))

        skip_sum = None
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        x = skip_sum / sqrt(self.num_layers)
        x = self.skip_mamba(x)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)


class ICASPBaselineWaveNetMamba(_ICASPBaselineWaveNetMambaBase):
    """Stage 257: WaveNet plus a single forward Mamba skip fusion block."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bidirectional=False, **kwargs)


class ICASPBaselineWaveNetBiMamba(_ICASPBaselineWaveNetMambaBase):
    """Stage 258: WaveNet plus bidirectional Mamba skip fusion for offline BSS."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bidirectional=True, **kwargs)


class _ICASPBaselineWaveNetMultiRateMambaBase(ICASPBaselineWaveNet):
    """10-block WaveNet whose final skip feature receives multi-rate context."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 64,
        residual_layers: int = 10,
        dilation_cycle_length: int = 10,
        *,
        bidirectional: bool,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_scale_init: float = 1e-2,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.skip_mamba = MultiRateMambaContext(
            residual_channels,
            mamba_channels=mamba_channels,
            downsample_factor=mamba_downsample_factor,
            bidirectional=bidirectional,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            scale_init=mamba_scale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))
        skip_sum = None
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        x = skip_sum / sqrt(self.num_layers)
        x = self.skip_mamba(x)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)


class ICASPBaselineWaveNetMultiRateMamba(_ICASPBaselineWaveNetMultiRateMambaBase):
    """Stage 259: 10-block WaveNet plus unidirectional multi-rate Mamba."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bidirectional=False, **kwargs)


class ICASPBaselineWaveNetMultiRateBiMamba(_ICASPBaselineWaveNetMultiRateMambaBase):
    """Stage 260: 10-block WaveNet plus bidirectional multi-rate Mamba."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bidirectional=True, **kwargs)


class _ICASPBaselineWaveNetInterleavedMambaBase(ICASPBaselineWaveNet):
    """WaveNet with multi-rate Mamba injected between two dilation cycles."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 64,
        residual_layers: int = 20,
        dilation_cycle_length: int = 10,
        *,
        bidirectional: bool,
        mamba_insert_after_block: int = 10,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.0,
        mamba_scale_init: float = 1e-2,
        mamba_directional_gate: bool = False,
        mamba_backward_gate_init: float = 0.0,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.mamba_insert_after_block = int(mamba_insert_after_block)
        if not 1 <= self.mamba_insert_after_block < self.num_layers:
            raise ValueError(
                "mamba_insert_after_block must be between 1 and residual_layers - 1"
            )
        self.interleaved_mamba = MultiRateMambaContext(
            residual_channels,
            mamba_channels=mamba_channels,
            downsample_factor=mamba_downsample_factor,
            bidirectional=bidirectional,
            directional_gate=mamba_directional_gate,
            directional_backward_gate_init=mamba_backward_gate_init,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            scale_init=mamba_scale_init,
        )

    def _post_interleaved_context(
        self,
        feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        """Optional extension hook for Stage-261-derived variants.

        The base implementation is deliberately an identity so Stage 261/262
        retain the exact forward path and checkpoint layout they had before
        the follow-up variants were added.
        """

        del mixture
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixture = x
        x = F.relu(self.input_projection(x))
        skip_sum = None
        for block_index, block in enumerate(self.residual_blocks, start=1):
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
            if block_index == self.mamba_insert_after_block:
                x = self.interleaved_mamba(x)
                x = self._post_interleaved_context(x, mixture)

        x = skip_sum / sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)


class ICASPBaselineWaveNetInterleavedMamba(_ICASPBaselineWaveNetInterleavedMambaBase):
    """Stage 261: WaveNet(10) -> unidirectional Mamba -> WaveNet(10)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bidirectional=False, **kwargs)


class ICASPBaselineWaveNetInterleavedBiMamba(_ICASPBaselineWaveNetInterleavedMambaBase):
    """Stage 262: WaveNet(10) -> bidirectional Mamba -> WaveNet(10)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, bidirectional=True, **kwargs)


class ICASPBaselineWaveNetInterleavedGatedBiMamba(
    _ICASPBaselineWaveNetInterleavedMambaBase
):
    """Stage 268: Stage-261 forward Mamba plus a gated reverse correction."""

    def __init__(
        self,
        *args,
        mamba_backward_gate_init: float = 1e-2,
        **kwargs,
    ):
        super().__init__(
            *args,
            bidirectional=False,
            mamba_directional_gate=True,
            mamba_backward_gate_init=mamba_backward_gate_init,
            **kwargs,
        )

    def no_weight_decay(self) -> set[str]:
        return {"interleaved_mamba.mamba.backward_gate_raw"}


class ICASPBaselineWaveNetInterleavedPhaseAwareReverseMamba(
    _ICASPBaselineWaveNetInterleavedMambaBase
):
    """Stage 271: Stage 261 plus a physical complex-conjugate reverse branch.

    The main path remains the successful causal Stage-261 Mamba inserted after
    block 10.  The auxiliary path sees ``conj(flip(mixture))`` instead of a
    plain flip of hidden real features, and is therefore separate from the
    latent reverse scan used by Stage 262.
    """

    def __init__(
        self,
        *args,
        phase_reverse_mamba_channels: int | None = None,
        phase_reverse_downsample_factor: int | None = None,
        phase_reverse_d_state: int | None = None,
        phase_reverse_d_conv: int | None = None,
        phase_reverse_expand: int | None = None,
        phase_reverse_dropout: float | None = None,
        phase_reverse_scale_init: float = 1e-2,
        **kwargs,
    ) -> None:
        super().__init__(*args, bidirectional=False, **kwargs)
        channels = int(self.input_projection.out_channels)
        self.phase_reverse_context = PhaseAwareReverseMambaContext(
            channels,
            mamba_channels=int(
                phase_reverse_mamba_channels
                if phase_reverse_mamba_channels is not None
                else kwargs.get("mamba_channels", 64)
            ),
            downsample_factor=int(
                phase_reverse_downsample_factor
                if phase_reverse_downsample_factor is not None
                else kwargs.get("mamba_downsample_factor", 4)
            ),
            d_state=int(
                phase_reverse_d_state
                if phase_reverse_d_state is not None
                else kwargs.get("mamba_d_state", 16)
            ),
            d_conv=int(
                phase_reverse_d_conv
                if phase_reverse_d_conv is not None
                else kwargs.get("mamba_d_conv", 4)
            ),
            expand=int(
                phase_reverse_expand
                if phase_reverse_expand is not None
                else kwargs.get("mamba_expand", 2)
            ),
            dropout=float(
                phase_reverse_dropout
                if phase_reverse_dropout is not None
                else kwargs.get("mamba_dropout", 0.0)
            ),
            scale_init=phase_reverse_scale_init,
        )

    def _post_interleaved_context(
        self,
        feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        return self.phase_reverse_context(feature, mixture)

    def no_weight_decay(self) -> set[str]:
        return {"phase_reverse_context.residual_scale"}


class WaveNetStage235CrossScaleBiMambaContext(nn.Module):
    """Stage-235-style compact global memory for the Stage-261 insertion point.

    Stage 235 combines a BiMamba encoder with a compressed global K/V bank.
    In the WaveNet setting, the first ten residual blocks supply the local
    query map, while a new low-rate bidirectional Mamba branch supplies the
    global memory.  The original Stage-261 unidirectional residual context is
    kept outside this module and runs first.
    """

    def __init__(
        self,
        channels: int,
        *,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        # Import lazily so the original Stage-261 model has no dependency on
        # the U-Net BiMamba research variants at construction time.
        from models.IQUBiMamba1D_CrossScaleAttention import (
            CompressedGlobalCrossAttention,
        )

        self.global_memory = MultiRateMambaGlobalMemory(
            channels,
            mamba_channels=mamba_channels,
            downsample_factor=mamba_downsample_factor,
            bidirectional=True,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.cross_attention = CompressedGlobalCrossAttention(
            query_channels=channels,
            global_channels=mamba_channels,
            kv_tokens=cross_scale_kv_tokens,
            num_heads=cross_scale_num_heads,
            dropout=cross_scale_dropout,
            residual_scale_init=cross_scale_residual_scale_init,
        )

    def forward(self, query_feature: torch.Tensor) -> torch.Tensor:
        global_feature = self.global_memory(query_feature)
        return self.cross_attention(query_feature, global_feature)


class WaveNetStage255PhysicalMoEContext(nn.Module):
    """Stage-255-style identity/global/physical/joint residual routing.

    The Stage-261 feature map remains the local expert.  A compact
    bidirectional Mamba memory forms the global expert, raw I/Q-derived
    physical tokens form the physical expert, and an identity-prior router
    selects their residual combination.  This mirrors Stage 255's essential
    routing mechanism without adding its U-Net-specific candidate decoders or
    training-only curriculum.
    """

    def __init__(
        self,
        channels: int,
        *,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        fusion_global_kv_tokens: int = 64,
        fusion_num_heads: int = 4,
        fusion_dropout: float = 0.0,
        fusion_channel_scale_init: float = 0.1,
        fusion_channel_scale_max: float = 0.5,
        fusion_router_hidden: int = 64,
        fusion_expert_prior: tuple[float, float, float, float] | list[float] = (
            0.7,
            0.1,
            0.1,
            0.1,
        ),
        fusion_condition_hidden: int = 16,
        fusion_condition_embedding: int = 16,
        fusion_trust_penalty_init: float = 0.1,
        fusion_trust_penalty_enable: bool = True,
        fusion_condition_routing_enable: bool = True,
        physical_cyclic_lags: tuple[int, ...] | list[int] = (0, 1, 2, 4, 8),
        physical_polyphase_branches: int = 8,
        physical_symbol_orders: tuple[int, ...] | list[int] = (2, 4, 8),
        physical_min_cyclic_freq: float = 1.0 / 64.0,
        physical_max_cyclic_freq: float = 1.0 / 8.0,
        physical_cyclic_temperature: float = 0.25,
    ) -> None:
        super().__init__()
        from models.IQUBiMamba1D_CrossScaleAttention import (
            CompressedGlobalCrossAttention,
        )
        from models.IQUBiMamba1D_HierarchicalKVFusion import (
            BoundedChannelScale,
            IdentityAwareEvidenceRouter,
            MixtureConditionEncoder,
        )
        from models.IQUBiMamba1D_KVAttentionAblations import (
            CompactPhysicalCrossAttention,
            TimeDomainPhysicalTokenExtractor,
        )

        self.global_memory = MultiRateMambaGlobalMemory(
            channels,
            mamba_channels=mamba_channels,
            downsample_factor=mamba_downsample_factor,
            bidirectional=True,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
        self.global_attention = CompressedGlobalCrossAttention(
            query_channels=channels,
            global_channels=mamba_channels,
            kv_tokens=fusion_global_kv_tokens,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
            residual_scale_init=None,
        )
        self.physical_token_extractor = TimeDomainPhysicalTokenExtractor(
            cyclic_lags=physical_cyclic_lags,
            polyphase_branches=physical_polyphase_branches,
            symbol_orders=physical_symbol_orders,
            min_cyclic_freq=physical_min_cyclic_freq,
            max_cyclic_freq=physical_max_cyclic_freq,
            cyclic_temperature=physical_cyclic_temperature,
        )
        self.physical_attention = CompactPhysicalCrossAttention(
            query_channels=channels,
            token_dim=self.physical_token_extractor.token_dim,
            token_count=self.physical_token_extractor.token_count,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
            residual_scale_init=None,
        )
        self.joint_proj = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.joint_norm = nn.LayerNorm(channels)
        self.condition_encoder = MixtureConditionEncoder(
            fusion_condition_hidden,
            fusion_condition_embedding,
        )
        self.expert_router = IdentityAwareEvidenceRouter(
            token_dim=self.physical_token_extractor.token_dim,
            query_channels=channels,
            global_channels=mamba_channels,
            condition_dim=fusion_condition_embedding,
            hidden_channels=fusion_router_hidden,
            prior=fusion_expert_prior,
            trust_penalty_init=fusion_trust_penalty_init,
        )
        self.channel_scale = BoundedChannelScale(
            channels,
            fusion_channel_scale_init,
            fusion_channel_scale_max,
        )
        self.trust_penalty_enable = bool(fusion_trust_penalty_enable)
        self.condition_routing_enable = bool(fusion_condition_routing_enable)

    def forward(
        self,
        query_feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                "Stage-255-style WaveNet fusion expects a raw single I/Q "
                f"mixture shaped (B, 2, L), got {tuple(mixture.shape)}"
            )

        global_feature = self.global_memory(query_feature)
        physical_tokens = self.physical_token_extractor(mixture)
        global_delta = self.global_attention.compute_delta(query_feature, global_feature)
        physical_delta = self.physical_attention.compute_delta(
            query_feature,
            physical_tokens,
        )
        joint_delta = self.joint_proj(torch.cat([global_delta, physical_delta], dim=1))
        joint_delta = self.joint_norm(joint_delta.transpose(1, 2)).transpose(1, 2)

        if self.condition_routing_enable:
            condition_embedding, _ = self.condition_encoder(mixture)
        else:
            condition_embedding = query_feature.new_zeros(
                query_feature.size(0),
                self.expert_router.condition_proj.in_features,
                dtype=torch.float32,
            )

        weights, _, _ = self.expert_router(
            physical_tokens,
            query_feature,
            global_feature,
            global_delta,
            physical_delta,
            condition_embedding,
            condition_enabled=self.condition_routing_enable,
            evidence_context_enabled=self.trust_penalty_enable,
            trust_penalty_enabled=self.trust_penalty_enable,
        )
        weights = weights.to(dtype=query_feature.dtype)
        update = (
            weights[:, 1, None, None] * global_delta
            + weights[:, 2, None, None] * physical_delta
            + weights[:, 3, None, None] * joint_delta
        )
        scale = self.channel_scale.values().to(
            device=query_feature.device,
            dtype=query_feature.dtype,
        )
        return query_feature + scale.view(1, -1, 1) * update


class ICASPBaselineWaveNetInterleavedCrossScaleBiMamba(
    _ICASPBaselineWaveNetInterleavedMambaBase
):
    """Stage 265: Stage 261 plus Stage-235-style BiMamba cross-scale memory."""

    def __init__(
        self,
        *args,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 1e-2,
        **kwargs,
    ) -> None:
        super().__init__(*args, bidirectional=False, **kwargs)
        channels = int(self.input_projection.out_channels)
        self.stage235_cross_scale = WaveNetStage235CrossScaleBiMambaContext(
            channels,
            mamba_channels=int(kwargs.get("mamba_channels", 64)),
            mamba_downsample_factor=int(kwargs.get("mamba_downsample_factor", 4)),
            mamba_d_state=int(kwargs.get("mamba_d_state", 16)),
            mamba_d_conv=int(kwargs.get("mamba_d_conv", 4)),
            mamba_expand=int(kwargs.get("mamba_expand", 2)),
            cross_scale_kv_tokens=cross_scale_kv_tokens,
            cross_scale_num_heads=cross_scale_num_heads,
            cross_scale_dropout=cross_scale_dropout,
            cross_scale_residual_scale_init=cross_scale_residual_scale_init,
        )

    def _post_interleaved_context(
        self,
        feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        del mixture
        return self.stage235_cross_scale(feature)

    def no_weight_decay(self) -> set[str]:
        return {"stage235_cross_scale.cross_attention.residual_scale"}


class ICASPBaselineWaveNetInterleavedStage235Memory(ICASPBaselineWaveNet):
    """Stage 272: replace Stage-261 context with Stage-235 K/V memory.

    This is intentionally *not* a Stage-265 ablation with a disabled gate.
    Stage 265 first reconstructs a full-rate residual through the Stage-261
    unidirectional Mamba and only then adds a separate global K/V branch.  In
    this model the only Mamba computation at the WaveNet midpoint is the
    low-rate bidirectional memory used as K/V by compressed cross-attention.
    It therefore isolates whether the reconstructed local-context path of
    Stage 261 is useful relative to Stage-235-style global-memory injection.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        residual_channels: int = 64,
        residual_layers: int = 20,
        dilation_cycle_length: int = 10,
        *,
        mamba_insert_after_block: int = 10,
        mamba_channels: int = 64,
        mamba_downsample_factor: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        cross_scale_kv_tokens: int = 64,
        cross_scale_num_heads: int = 4,
        cross_scale_dropout: float = 0.0,
        cross_scale_residual_scale_init: float = 1e-2,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            residual_channels=residual_channels,
            residual_layers=residual_layers,
            dilation_cycle_length=dilation_cycle_length,
        )
        self.mamba_insert_after_block = int(mamba_insert_after_block)
        if not 1 <= self.mamba_insert_after_block < self.num_layers:
            raise ValueError(
                "mamba_insert_after_block must be between 1 and residual_layers - 1"
            )

        self.stage235_memory = WaveNetStage235CrossScaleBiMambaContext(
            residual_channels,
            mamba_channels=mamba_channels,
            mamba_downsample_factor=mamba_downsample_factor,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            cross_scale_kv_tokens=cross_scale_kv_tokens,
            cross_scale_num_heads=cross_scale_num_heads,
            cross_scale_dropout=cross_scale_dropout,
            cross_scale_residual_scale_init=cross_scale_residual_scale_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))
        skip_sum = None
        for block_index, block in enumerate(self.residual_blocks, start=1):
            x, skip = block(x)
            skip_sum = skip if skip_sum is None else skip_sum + skip
            if block_index == self.mamba_insert_after_block:
                x = self.stage235_memory(x)

        x = skip_sum / sqrt(self.num_layers)
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x)

    def no_weight_decay(self) -> set[str]:
        return {"stage235_memory.cross_attention.residual_scale"}


class ICASPBaselineWaveNetInterleavedPhysicalMoEBiMamba(
    _ICASPBaselineWaveNetInterleavedMambaBase
):
    """Stage 266: Stage 261 plus Stage-255-style physical MoE routing."""

    def __init__(self, *args, **kwargs) -> None:
        # Remove Stage-255-specific options before delegating to the unchanged
        # Stage-261 constructor, which intentionally accepts only WaveNet and
        # Mamba parameters.
        stage255_options = {
            "fusion_global_kv_tokens": kwargs.pop("fusion_global_kv_tokens", 64),
            "fusion_num_heads": kwargs.pop("fusion_num_heads", 4),
            "fusion_dropout": kwargs.pop("fusion_dropout", 0.0),
            "fusion_channel_scale_init": kwargs.pop(
                "fusion_channel_scale_init", 0.1
            ),
            "fusion_channel_scale_max": kwargs.pop("fusion_channel_scale_max", 0.5),
            "fusion_router_hidden": kwargs.pop("fusion_router_hidden", 64),
            "fusion_expert_prior": kwargs.pop(
                "fusion_expert_prior", (0.7, 0.1, 0.1, 0.1)
            ),
            "fusion_condition_hidden": kwargs.pop("fusion_condition_hidden", 16),
            "fusion_condition_embedding": kwargs.pop(
                "fusion_condition_embedding", 16
            ),
            "fusion_trust_penalty_init": kwargs.pop(
                "fusion_trust_penalty_init", 0.1
            ),
            "fusion_trust_penalty_enable": kwargs.pop(
                "fusion_trust_penalty_enable", True
            ),
            "fusion_condition_routing_enable": kwargs.pop(
                "fusion_condition_routing_enable", True
            ),
            "physical_cyclic_lags": kwargs.pop(
                "physical_cyclic_lags", (0, 1, 2, 4, 8)
            ),
            "physical_polyphase_branches": kwargs.pop(
                "physical_polyphase_branches", 8
            ),
            "physical_symbol_orders": kwargs.pop("physical_symbol_orders", (2, 4, 8)),
            "physical_min_cyclic_freq": kwargs.pop(
                "physical_min_cyclic_freq", 1.0 / 64.0
            ),
            "physical_max_cyclic_freq": kwargs.pop(
                "physical_max_cyclic_freq", 1.0 / 8.0
            ),
            "physical_cyclic_temperature": kwargs.pop(
                "physical_cyclic_temperature", 0.25
            ),
        }
        super().__init__(*args, bidirectional=False, **kwargs)
        channels = int(self.input_projection.out_channels)
        self.stage255_physical_moe = WaveNetStage255PhysicalMoEContext(
            channels,
            mamba_channels=int(kwargs.get("mamba_channels", 64)),
            mamba_downsample_factor=int(kwargs.get("mamba_downsample_factor", 4)),
            mamba_d_state=int(kwargs.get("mamba_d_state", 16)),
            mamba_d_conv=int(kwargs.get("mamba_d_conv", 4)),
            mamba_expand=int(kwargs.get("mamba_expand", 2)),
            fusion_global_kv_tokens=int(stage255_options["fusion_global_kv_tokens"]),
            fusion_num_heads=int(stage255_options["fusion_num_heads"]),
            fusion_dropout=float(stage255_options["fusion_dropout"]),
            fusion_channel_scale_init=float(
                stage255_options["fusion_channel_scale_init"]
            ),
            fusion_channel_scale_max=float(stage255_options["fusion_channel_scale_max"]),
            fusion_router_hidden=int(stage255_options["fusion_router_hidden"]),
            fusion_expert_prior=stage255_options["fusion_expert_prior"],
            fusion_condition_hidden=int(stage255_options["fusion_condition_hidden"]),
            fusion_condition_embedding=int(
                stage255_options["fusion_condition_embedding"]
            ),
            fusion_trust_penalty_init=float(
                stage255_options["fusion_trust_penalty_init"]
            ),
            fusion_trust_penalty_enable=bool(
                stage255_options["fusion_trust_penalty_enable"]
            ),
            fusion_condition_routing_enable=bool(
                stage255_options["fusion_condition_routing_enable"]
            ),
            physical_cyclic_lags=stage255_options["physical_cyclic_lags"],
            physical_polyphase_branches=int(
                stage255_options["physical_polyphase_branches"]
            ),
            physical_symbol_orders=stage255_options["physical_symbol_orders"],
            physical_min_cyclic_freq=float(
                stage255_options["physical_min_cyclic_freq"]
            ),
            physical_max_cyclic_freq=float(
                stage255_options["physical_max_cyclic_freq"]
            ),
            physical_cyclic_temperature=float(
                stage255_options["physical_cyclic_temperature"]
            ),
        )

    def _post_interleaved_context(
        self,
        feature: torch.Tensor,
        mixture: torch.Tensor,
    ) -> torch.Tensor:
        return self.stage255_physical_moe(feature, mixture)

    def no_weight_decay(self) -> set[str]:
        return {
            "stage255_physical_moe.channel_scale.raw",
            "stage255_physical_moe.expert_router.raw_trust_penalty",
        }


class ICASPBaselineWaveNetInterleavedMambaCycloFRESH(
    ICASPBaselineWaveNetInterleavedMamba
):
    """Stage 267: Stage 261 with the metadata-free Stage-79 input prior."""

    def __init__(
        self,
        *args,
        estimated_cyclofresh_min_freq: float = 1.0 / 64.0,
        estimated_cyclofresh_max_freq: float = 1.0 / 8.0,
        estimated_cyclofresh_default_freq: float = 1.0 / 32.0,
        estimated_cyclofresh_momentum: float = 0.05,
        estimated_cyclofresh_hidden_channels: int = 8,
        estimated_cyclofresh_kernel_size: int = 9,
        estimated_cyclofresh_scale_init: float = 1e-2,
        estimated_cyclofresh_gate_hidden: int = 8,
        estimated_cyclofresh_zero_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        from models.IQUMamba1D_EstimatedCycloFRESH import (
            EstimatedCycloFRESHAdapter1D,
        )

        self.estimated_cyclofresh_adapter = EstimatedCycloFRESHAdapter1D(
            input_channels=int(self.input_projection.in_channels),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(self.estimated_cyclofresh_adapter(x))

    def no_weight_decay(self) -> set[str]:
        return {"estimated_cyclofresh_adapter.scale"}
