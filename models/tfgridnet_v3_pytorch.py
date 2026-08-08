import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


class LayerNormalization(nn.Module):
    def __init__(self, input_dim: int, dim: int = 1, total_dim: int = 4, eps: float = 1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if index != self.dim else input_dim for index in range(total_dim)]
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        self.eps = eps

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim - 1 < self.dim:
            raise ValueError(f"Expect x to have {self.dim + 1} dimensions, but got {x.ndim}")
        if x.dtype in HALF_PRECISION_DTYPES:
            dtype = x.dtype
            x = x.float()
        else:
            dtype = None
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype=dtype) if dtype else x_hat


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension: Sequence[int], eps: float = 1e-5):
        super().__init__()
        if len(input_dimension) != 2:
            raise ValueError(f"input_dimension must have length 2, got {input_dimension}")
        n_head, emb_dim = input_dimension
        param_size = [1, n_head, emb_dim, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=n_head, init=0.25)
        self.eps = eps
        self.n_head = n_head
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
        batch, _, n_frames, n_freqs = x.shape
        x = x.view([batch, self.n_head, self.emb_dim, n_frames, n_freqs])
        x = self.act(x)
        stat_dim = (2,)
        mu_ = x.mean(dim=stat_dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        x = ((x - mu_) / std_) * self.gamma + self.beta
        return x


class GridNetV3Block(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        hidden_channels: int,
        n_head: int = 4,
        qk_output_channel: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True)
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        if emb_dim % n_head != 0:
            raise ValueError(f"emb_dim ({emb_dim}) must be divisible by n_head ({n_head})")
        qk_dim = qk_output_channel
        v_dim = emb_dim // n_head

        self.attn_conv_q = nn.Conv2d(emb_dim, n_head * qk_dim, 1)
        self.attn_norm_q = AllHeadPReLULayerNormalization4DC((n_head, qk_dim), eps=eps)
        self.attn_conv_k = nn.Conv2d(emb_dim, n_head * qk_dim, 1)
        self.attn_norm_k = AllHeadPReLULayerNormalization4DC((n_head, qk_dim), eps=eps)
        self.attn_conv_v = nn.Conv2d(emb_dim, n_head * v_dim, 1)
        self.attn_norm_v = AllHeadPReLULayerNormalization4DC((n_head, v_dim), eps=eps)
        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1),
            nn.PReLU(),
            LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, old_t, old_q = x.shape
        overlap = self.emb_ks - self.emb_hs
        padded_t = math.ceil((old_t + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        padded_q = math.ceil((old_q + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks

        x = x.permute(0, 2, 3, 1)
        x = F.pad(x, (0, 0, overlap, padded_q - old_q - overlap, overlap, padded_t - old_t - overlap))

        input_ = x
        intra_rnn = self.intra_norm(input_)
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([batch_size * padded_t, -1, self.emb_ks * channels])
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([batch_size, padded_t, padded_q, channels])
        else:
            intra_rnn = intra_rnn.view([batch_size * padded_t, padded_q, channels])
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([batch_size, padded_t, channels, padded_q]).transpose(-2, -1)
        intra_rnn = intra_rnn + input_
        intra_rnn = intra_rnn.transpose(1, 2)

        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([batch_size * padded_q, -1, self.emb_ks * channels])
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([batch_size, padded_q, padded_t, channels])
        else:
            inter_rnn = inter_rnn.view([batch_size * padded_q, padded_t, channels])
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([batch_size, padded_q, channels, padded_t]).transpose(-2, -1)
        inter_rnn = inter_rnn + input_

        inter_rnn = inter_rnn.permute(0, 3, 2, 1)
        inter_rnn = inter_rnn[..., overlap : overlap + old_t, overlap : overlap + old_q]
        residual = inter_rnn

        q = self.attn_norm_q(self.attn_conv_q(residual))
        k = self.attn_norm_k(self.attn_conv_k(residual))
        v = self.attn_norm_v(self.attn_conv_v(residual))
        q = q.view(-1, *q.shape[2:])
        k = k.view(-1, *k.shape[2:])
        v = v.view(-1, *v.shape[2:])

        q = q.transpose(1, 2).flatten(start_dim=2)
        k = k.transpose(2, 3).contiguous().view([batch_size * self.n_head, -1, old_t])
        v = v.transpose(1, 2)
        old_v_shape = v.shape
        v = v.flatten(start_dim=2)
        emb_dim = q.shape[-1]

        attn_mat = torch.matmul(q, k) / (emb_dim**0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        v = torch.matmul(attn_mat, v)

        v = v.reshape(old_v_shape).transpose(1, 2)
        emb_dim = v.shape[1]
        batch = v.contiguous().view([batch_size, self.n_head * emb_dim, old_t, old_q])
        batch = self.attn_concat_proj(batch)
        return batch + residual


class TFGridNetV3Core(nn.Module):
    def __init__(
        self,
        n_srcs: int = 2,
        n_imics: int = 1,
        n_layers: int = 6,
        lstm_hidden_units: int = 192,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 4,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        eps: float = 1.0e-5,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        if self.n_imics != 1:
            raise ValueError(f"Only single-channel mixture is supported. Got n_imics={self.n_imics}")

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.blocks = nn.ModuleList(
            [
                GridNetV3Block(
                    emb_dim=emb_dim,
                    emb_ks=emb_ks,
                    emb_hs=emb_hs,
                    hidden_channels=lstm_hidden_units,
                    n_head=attn_n_head,
                    qk_output_channel=attn_qk_output_channel,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    def forward(self, input_spectrum: torch.Tensor) -> List[torch.Tensor]:
        input_is_complex = torch.is_complex(input_spectrum)
        if input_is_complex:
            feature = torch.stack([input_spectrum.real, input_spectrum.imag], dim=1)
        else:
            if input_spectrum.size(-1) != 2:
                raise ValueError(
                    f"Expected real-imag axis in last dim with size 2, got shape={tuple(input_spectrum.shape)}"
                )
            feature = input_spectrum.moveaxis(-1, 1)

        if feature.ndim != 4:
            raise ValueError(f"Expected 4D feature tensor, got shape={tuple(feature.shape)}")

        n_batch, _, n_frames, n_freqs = feature.shape
        batch = self.conv(feature)
        for block in self.blocks:
            batch = block(batch)
        batch = self.deconv(batch)
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs])

        if input_is_complex:
            batch = batch.float()
            batch_complex = torch.complex(batch[:, :, 0], batch[:, :, 1])
            return [batch_complex[:, src] for src in range(self.n_srcs)]

        batch = batch.permute(0, 1, 3, 4, 2).contiguous()
        return [batch[:, src] for src in range(self.n_srcs)]


class TFGridNetV3Separator1D(nn.Module):
    """Pure-PyTorch TF-GridNet wrapper for IQ time-domain separation.

    Input:
        x: [B, 2, L] where channels are [I_mix, Q_mix]
    Output:
        y: [B, 2*n_srcs, L] where channels are [I1, Q1, I2, Q2, ...]
    """

    def __init__(
        self,
        n_srcs: int = 2,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        center: bool = True,
        normalize_input: bool = True,
        eps: float = 1e-8,
        n_layers: int = 6,
        lstm_hidden_units: int = 192,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 4,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.normalize_input = normalize_input
        self.eps = eps

        self.separator = TFGridNetV3Core(
            n_srcs=n_srcs,
            n_imics=1,
            n_layers=n_layers,
            lstm_hidden_units=lstm_hidden_units,
            attn_n_head=attn_n_head,
            attn_qk_output_channel=attn_qk_output_channel,
            emb_dim=emb_dim,
            emb_ks=emb_ks,
            emb_hs=emb_hs,
            eps=eps,
        )
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected input [B, 2, L], got shape={tuple(x.shape)}")

        batch_size, _, length = x.shape
        original_dtype = x.dtype
        if original_dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        mix_complex = torch.complex(x[:, 0], x[:, 1])
        if self.normalize_input:
            std = mix_complex.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
            mix_complex = mix_complex / std
        else:
            std = torch.ones((batch_size, 1), device=x.device, dtype=x.real.dtype)

        window = self.window.to(device=x.device, dtype=x.real.dtype)
        mix_spec = torch.stft(
            mix_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            return_complex=True,
        )
        mix_spec = mix_spec.transpose(1, 2).contiguous()

        separated_specs = self.separator(mix_spec)
        reconstructed = []
        for separated_spec in separated_specs:
            separated_spec = separated_spec.transpose(1, 2).contiguous()
            signal = torch.istft(
                separated_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                length=length,
                return_complex=True,
            )
            signal = signal * std
            reconstructed.append(signal)

        output = torch.stack(
            [torch.stack([signal.real, signal.imag], dim=1) for signal in reconstructed],
            dim=1,
        )
        output = output.reshape(batch_size, self.n_srcs * 2, length)
        return output.to(dtype=original_dtype)
