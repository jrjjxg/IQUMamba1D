from typing import Callable, List, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from models.IQUMamba1D import BasicResBlock, UNetResDecoder


def _valid_heads(channels: int, requested_heads: int) -> int:
    requested_heads = max(1, min(int(requested_heads), int(channels)))
    for heads in range(requested_heads, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class GatedLinearAttention1D(nn.Module):
    """Non-causal gated linear attention used as a MambaLayer replacement."""

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0, residual_scale_init: float = 0.05):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.norm = nn.LayerNorm(self.channels)
        self.qkvg = nn.Linear(self.channels, self.channels * 4)
        self.proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t = x.shape
        tokens = self.norm(x.transpose(1, 2))
        q, k, v, gate = self.qkvg(tokens).chunk(4, dim=-1)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        gate = torch.sigmoid(gate)

        q = q.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        kv = torch.einsum("bhtd,bhte->bhde", k, v)
        k_sum = k.sum(dim=2)
        z = 1.0 / (torch.einsum("bhtd,bhd->bht", q, k_sum) + self.eps)
        out = torch.einsum("bhtd,bhde,bht->bhte", q, kv, z)
        out = out.transpose(1, 2).reshape(b, t, c)
        out = self.proj(out * gate)
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class MegaMovingAverageGatedAttention1D(nn.Module):
    """MEGA-inspired moving-average gated mixer for 1D IQ features."""

    def __init__(
        self,
        channels: int,
        ema_kernel_size: int = 63,
        expansion: float = 2.0,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.ema_kernel_size = max(3, int(ema_kernel_size) | 1)
        hidden = max(self.channels, int(self.channels * float(expansion)))
        self.norm = nn.LayerNorm(self.channels)
        self.in_proj = nn.Linear(self.channels, hidden * 3)
        self.ema_kernel = nn.Parameter(torch.empty(hidden, self.ema_kernel_size))
        self.out_proj = nn.Linear(hidden, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ema_kernel, mean=0.0, std=0.02)

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        kernel = F.softmax(self.ema_kernel, dim=-1).reshape(x.size(1), 1, self.ema_kernel_size)
        return F.conv1d(x, kernel, padding=self.ema_kernel_size // 2, groups=x.size(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        tokens = self.norm(x.transpose(1, 2))
        u, v, gate = self.in_proj(tokens).chunk(3, dim=-1)
        v = self.moving_average(v.transpose(1, 2)).transpose(1, 2)
        out = F.silu(u) * v * torch.sigmoid(gate)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class HyenaLongConv1D(nn.Module):
    """Hyena-style gated implicit long convolution for 1D encoder stages."""

    def __init__(
        self,
        channels: int,
        filter_hidden: int = 64,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)
        self.in_proj = nn.Linear(self.channels, self.channels * 3)
        self.filter_mlp = nn.Sequential(
            nn.Linear(1, int(filter_hidden)),
            nn.SiLU(),
            nn.Linear(int(filter_hidden), self.channels),
        )
        self.filter_decay = nn.Parameter(torch.ones(self.channels) * 2.0)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def _implicit_filter(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.linspace(0.0, 1.0, steps=length, device=device, dtype=dtype).unsqueeze(-1)
        filters = self.filter_mlp(pos).transpose(0, 1)
        envelope = torch.exp(-self.filter_decay.abs().to(dtype=dtype).unsqueeze(-1) * pos.squeeze(-1).unsqueeze(0))
        return filters * envelope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        original_dtype = x.dtype
        b, c, t = x.shape
        compute_x = x.float() if x.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else x
        tokens = self.norm(compute_x.transpose(1, 2))
        u, v, gates = self.in_proj(tokens).chunk(3, dim=-1)
        v = v.transpose(1, 2)

        filters = self._implicit_filter(t, x.device, v.dtype)
        fft_size = 2 * t
        v_fft = torch.fft.rfft(v, n=fft_size)
        filter_fft = torch.fft.rfft(filters, n=fft_size).unsqueeze(0)
        y = torch.fft.irfft(v_fft * filter_fft, n=fft_size)[..., :t].transpose(1, 2)

        out = y * torch.sigmoid(gates) * F.silu(u)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2).to(dtype=original_dtype)
        return residual + self.gamma * out


class Retention1D(nn.Module):
    """RetNet-inspired bidirectional diagonal retention for 1D IQ features."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        retention_kernel_size: int = 128,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.retention_kernel_size = max(3, int(retention_kernel_size))
        self.norm = nn.LayerNorm(self.channels)
        self.qkvg = nn.Linear(self.channels, self.channels * 4)
        self.logit_decay = nn.Parameter(torch.linspace(1.0, 3.0, steps=self.num_heads))
        self.proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def retention_kernel(self, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        kernel_len = min(self.retention_kernel_size, int(length))
        steps = torch.arange(kernel_len, dtype=dtype, device=device)
        decay = torch.sigmoid(self.logit_decay).to(dtype=dtype).clamp(0.50, 0.995)
        kernel = decay[:, None] ** steps[None, :]
        return kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _directional_retention(self, kv: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b, h, d, t = kv.shape
        kernel_len = kernel.size(-1)
        weight = kernel[:, None, :].expand(h, d, kernel_len).reshape(h * d, 1, kernel_len)
        kv = kv.reshape(b, h * d, t)
        kv = F.pad(kv, (kernel_len - 1, 0))
        return F.conv1d(kv, weight, groups=h * d).reshape(b, h, d, t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t = x.shape
        tokens = self.norm(x.transpose(1, 2))
        q, k, v, gate = self.qkvg(tokens).chunk(4, dim=-1)
        q = q.reshape(b, t, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        k = torch.tanh(k.reshape(b, t, self.num_heads, self.head_dim).permute(0, 2, 3, 1))
        v = v.reshape(b, t, self.num_heads, self.head_dim).permute(0, 2, 3, 1)

        kernel = self.retention_kernel(t, x.dtype, x.device)
        kv = k * v
        forward_state = self._directional_retention(kv, kernel)
        backward_state = torch.flip(self._directional_retention(torch.flip(kv, dims=[-1]), kernel), dims=[-1])
        retained = 0.5 * (forward_state + backward_state)

        out = (q * retained).permute(0, 3, 1, 2).reshape(b, t, c)
        out = self.proj(out * torch.sigmoid(gate))
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class GriffinGatedLinearRecurrence1D(nn.Module):
    """Griffin/Hawk-style gated linear recurrence with local convolution."""

    def __init__(
        self,
        channels: int,
        recurrence_kernel_size: int = 128,
        local_kernel_size: int = 5,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.recurrence_kernel_size = max(3, int(recurrence_kernel_size))
        local_kernel_size = max(3, int(local_kernel_size) | 1)
        self.norm = nn.LayerNorm(self.channels)
        self.in_proj = nn.Linear(self.channels, self.channels * 3)
        self.local_conv = nn.Conv1d(self.channels, self.channels, kernel_size=local_kernel_size, padding=local_kernel_size // 2, groups=self.channels)
        self.logit_decay = nn.Parameter(torch.linspace(1.0, 3.0, steps=self.channels))
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def recurrence_kernel(self, length: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        kernel_len = min(self.recurrence_kernel_size, int(length))
        steps = torch.arange(kernel_len, dtype=dtype, device=device)
        decay = torch.sigmoid(self.logit_decay).to(dtype=dtype).clamp(0.50, 0.995)
        kernel = decay[:, None] ** steps[None, :]
        return kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def recurrent_mix(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        kernel_len = kernel.size(-1)
        weight = kernel.reshape(c, 1, kernel_len)
        forward = F.conv1d(F.pad(x, (kernel_len - 1, 0)), weight, groups=c)
        backward = torch.flip(
            F.conv1d(F.pad(torch.flip(x, dims=[-1]), (kernel_len - 1, 0)), weight, groups=c),
            dims=[-1],
        )
        return 0.5 * (forward + backward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t = x.shape
        tokens = self.norm(x.transpose(1, 2))
        u, v, gate = self.in_proj(tokens).chunk(3, dim=-1)
        v = self.local_conv(v.transpose(1, 2))
        kernel = self.recurrence_kernel(t, x.dtype, x.device)
        mixed = self.recurrent_mix(v, kernel).transpose(1, 2)
        out = F.silu(u) * mixed * torch.sigmoid(gate)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class xLSTM1D(nn.Module):
    """Small sLSTM-style bidirectional gated memory block for 1D features."""

    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
        forget_bias: float = 1.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.norm = nn.LayerNorm(self.channels)
        self.in_proj = nn.Linear(self.channels, self.channels * 4)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.forget_bias = float(forget_bias)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def _scan(self, gates: torch.Tensor) -> torch.Tensor:
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)

        cell_state = torch.zeros_like(candidate[:, 0])
        outputs = []
        for t_idx in range(gates.size(1)):
            cell_state = forget_gate[:, t_idx] * cell_state + input_gate[:, t_idx] * candidate[:, t_idx]
            outputs.append(output_gate[:, t_idx] * torch.tanh(cell_state))
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        tokens = self.norm(x.transpose(1, 2))
        gates = self.in_proj(tokens)
        forward = self._scan(gates)
        backward = torch.flip(self._scan(torch.flip(gates, dims=[1])), dims=[1])
        out = 0.5 * (forward + backward)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class SpectralMixer1D(nn.Module):
    """FNO-style low-frequency spectral mixer for 1D IQ encoder features."""

    def __init__(
        self,
        channels: int,
        mode_count: int = 128,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.mode_count = max(1, int(mode_count))
        self.norm = nn.LayerNorm(self.channels)
        self.value_proj = nn.Linear(self.channels, self.channels)
        self.gate_proj = nn.Linear(self.channels, self.channels)
        self.spectral_weight_real = nn.Parameter(torch.randn(self.channels, self.mode_count) * 0.02)
        self.spectral_weight_imag = nn.Parameter(torch.randn(self.channels, self.mode_count) * 0.02)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        original_dtype = x.dtype
        b, c, t = x.shape
        compute_x = x.float() if x.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else x
        tokens = self.norm(compute_x.transpose(1, 2))
        value = self.value_proj(tokens).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(tokens))

        value_fft = torch.fft.rfft(value, dim=-1)
        mode_count = min(self.mode_count, value_fft.size(-1))
        mixed_fft = torch.zeros_like(value_fft)
        weights = torch.complex(
            self.spectral_weight_real[:, :mode_count].to(dtype=value.dtype),
            self.spectral_weight_imag[:, :mode_count].to(dtype=value.dtype),
        )
        mixed_fft[..., :mode_count] = value_fft[..., :mode_count] * weights[None, :, :]
        mixed = torch.fft.irfft(mixed_fft, n=t, dim=-1).transpose(1, 2)

        out = self.out_proj(mixed * gate)
        out = self.drop(out).transpose(1, 2).to(dtype=original_dtype)
        return residual + self.gamma * out


class LowRankSpectralMixer1D(nn.Module):
    """Capacity-controlled FNO-style mixer with low-rank complex spectral weights."""

    def __init__(
        self,
        channels: int,
        mode_count: int = 32,
        spectral_rank: int = 4,
        dropout: float = 0.10,
        residual_scale_init: float = 0.02,
    ):
        super().__init__()
        self.channels = int(channels)
        self.mode_count = max(1, int(mode_count))
        self.spectral_rank = max(1, int(spectral_rank))
        self.norm = nn.LayerNorm(self.channels)
        self.value_proj = nn.Linear(self.channels, self.channels)
        self.gate_proj = nn.Linear(self.channels, self.channels)
        self.spectral_channel_factor_real = nn.Parameter(torch.randn(self.channels, self.spectral_rank) * 0.02)
        self.spectral_channel_factor_imag = nn.Parameter(torch.randn(self.channels, self.spectral_rank) * 0.02)
        self.spectral_mode_factor_real = nn.Parameter(torch.randn(self.mode_count, self.spectral_rank) * 0.02)
        self.spectral_mode_factor_imag = nn.Parameter(torch.randn(self.mode_count, self.spectral_rank) * 0.02)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))

    def _spectral_weights(self, mode_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        channel_factor = torch.complex(
            self.spectral_channel_factor_real.to(device=device, dtype=dtype),
            self.spectral_channel_factor_imag.to(device=device, dtype=dtype),
        )
        mode_factor = torch.complex(
            self.spectral_mode_factor_real[:mode_count].to(device=device, dtype=dtype),
            self.spectral_mode_factor_imag[:mode_count].to(device=device, dtype=dtype),
        )
        return torch.einsum("cr,mr->cm", channel_factor, mode_factor) / (self.spectral_rank ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        original_dtype = x.dtype
        b, c, t = x.shape
        compute_x = x.float() if x.dtype in (torch.float16, getattr(torch, "bfloat16", torch.float16)) else x
        tokens = self.norm(compute_x.transpose(1, 2))
        value = self.value_proj(tokens).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(tokens))

        value_fft = torch.fft.rfft(value, dim=-1)
        mode_count = min(self.mode_count, value_fft.size(-1))
        mixed_fft = torch.zeros_like(value_fft)
        weights = self._spectral_weights(mode_count, value.dtype, value.device)
        mixed_fft[..., :mode_count] = value_fft[..., :mode_count] * weights[None, :, :]
        mixed = torch.fft.irfft(mixed_fft, n=t, dim=-1).transpose(1, 2)

        out = self.out_proj(mixed * gate)
        out = self.drop(out).transpose(1, 2).to(dtype=original_dtype)
        return residual + self.gamma * out


class DeltaLinearAttention1D(nn.Module):
    """Kimi/DeltaNet-style delta-rule linear attention with fixed memory."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_scale_init: float = 0.05,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = _valid_heads(self.channels, num_heads)
        self.head_dim = self.channels // self.num_heads
        self.norm = nn.LayerNorm(self.channels)
        self.qkvg = nn.Linear(self.channels, self.channels * 4)
        self.decay_proj = nn.Linear(self.channels, self.num_heads)
        self.out_proj = nn.Linear(self.channels, self.channels)
        self.drop = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(1) * float(residual_scale_init))
        self.eps = 1e-6

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(x) + 1.0
        return x / x.sum(dim=-1, keepdim=True).clamp_min(self.eps)

    def _scan(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, delta_gate: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
        b, h, t, d = q.shape
        memory = torch.zeros(b, h, d, d, dtype=q.dtype, device=q.device)
        outputs = []
        for t_idx in range(t):
            k_t = k[:, :, t_idx]
            v_t = v[:, :, t_idx]
            q_t = q[:, :, t_idx]
            read_k = torch.einsum("bhde,bhd->bhe", memory, k_t)
            delta = v_t - read_k
            memory = decay[:, :, t_idx, None, None] * memory + delta_gate[:, :, t_idx, None, None] * torch.einsum("bhd,bhe->bhde", k_t, delta)
            outputs.append(torch.einsum("bhde,bhd->bhe", memory, q_t))
        return torch.stack(outputs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        b, c, t = x.shape
        tokens = self.norm(x.transpose(1, 2))
        q, k, v, gate = self.qkvg(tokens).chunk(4, dim=-1)
        decay = torch.sigmoid(self.decay_proj(tokens)).transpose(1, 2).clamp(0.50, 0.995)
        delta_gate = torch.sigmoid(gate).reshape(b, t, self.num_heads, self.head_dim).mean(dim=-1).transpose(1, 2)

        q = self._feature_map(q).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self._feature_map(k).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        forward = self._scan(q, k, v, delta_gate, decay)
        backward = torch.flip(
            self._scan(
                torch.flip(q, dims=[2]),
                torch.flip(k, dims=[2]),
                torch.flip(v, dims=[2]),
                torch.flip(delta_gate, dims=[2]),
                torch.flip(decay, dims=[2]),
            ),
            dims=[2],
        )
        out = 0.5 * (forward + backward)
        out = out.transpose(1, 2).reshape(b, t, c)
        out = self.out_proj(out)
        out = self.drop(out).transpose(1, 2)
        return residual + self.gamma * out


class ReplacementMambaEncoder(nn.Module):
    """Stage4-compatible encoder that swaps MambaLayer for a provided block."""

    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        replacement_factory: Callable[[int], nn.Module],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        stem_channels: int = None,
        pool_type: str = "conv",
    ):
        super().__init__()
        del pool_type
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]
        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]

        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[
                BasicBlockD(
                    conv_op=conv_op,
                    input_channels=stem_channels,
                    output_channels=stem_channels,
                    kernel_size=kernel_sizes[0],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
                for _ in range(n_blocks_per_stage[0] - 1)
            ],
        )

        input_channels = stem_channels
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ),
                *[
                    BasicBlockD(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        conv_bias=conv_bias,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                    for _ in range(n_blocks_per_stage[s] - 1)
                ],
            )
            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(replacement_factory(int(features_per_stage[s])))
            else:
                mamba_layers.append(nn.Identity())
            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x: torch.Tensor):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


class IQUMamba1D_MambaReplacementBase(nn.Module):
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
        replacement_factory: Callable[[int], nn.Module],
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

        self.encoder = ReplacementMambaEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            replacement_factory=replacement_factory,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        skips = self.encoder(x)
        return self.decoder(skips)
