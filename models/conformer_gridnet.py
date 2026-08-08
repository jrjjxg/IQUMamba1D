"""Conformer-GridNet — TF-GridNet with BLSTM replaced by Conformer blocks.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", Interspeech 2020.

The macaron-style Conformer block (FFN → MHSA → DepthwiseConv → FFN) replaces
the intra-frame and inter-frame BLSTMs in each GridNetV3Block.

Pure PyTorch implementation — NO external dependencies beyond PyTorch.

Input:  [B, 2, L]  (I_mix, Q_mix)
Output: [B, 2*n_srcs, L]  (I1, Q1, I2, Q2, ...)
"""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn import init
from torch.nn.parameter import Parameter

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  Reused normalizations from TFGridNet (same as spmamba_gridnet.py)
# ============================================================================

class LayerNormalization(nn.Module):
    def __init__(self, input_dim: int, dim: int = 1, total_dim: int = 4, eps: float = 1e-5):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1] * total_dim
        param_size[self.dim] = input_dim
        self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        dim = self.dim if self.dim >= 0 else self.dim + x.dim()
        mu_ = x.mean(dim=dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=dim, unbiased=False, keepdim=True) + self.eps)
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat.to(dtype)


class AllHeadPReLULayerNormalization4DC(nn.Module):
    def __init__(self, input_dimension: Sequence[int], eps: float = 1e-5):
        super().__init__()
        if len(input_dimension) != 2:
            raise ValueError(f"Expected 2D, got {len(input_dimension)}D")
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

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
        dtype = x.dtype
        x = x.float()
        batch, _, n_frames, n_freqs = x.shape
        x = x.view([batch, self.n_head, self.emb_dim, n_frames, n_freqs])
        x = self.act(x)
        mu_ = x.mean(dim=2, keepdim=True)
        std_ = torch.sqrt(x.var(dim=2, unbiased=False, keepdim=True) + self.eps)
        x = ((x - mu_) / std_) * self.gamma + self.beta
        return x.to(dtype)


# ============================================================================
#  Conformer building blocks (Gulati et al., 2020)
# ============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerFeedForward(nn.Module):
    """Macaron-style Feed-Forward Module with half-step residual.

    FFN(x) = Linear(Swish(Linear(LayerNorm(x)))) * dropout
    Output: x + 0.5 * FFN(x)
    """

    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        d_ff = d_model * expansion_factor
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.ffn(self.norm(x))


class ConformerConvModule(nn.Module):
    """Convolution Module with GLU gating and depthwise conv.

    Conv(x) = Pointwise → GLU → DepthwiseConv → BatchNorm → Swish → Pointwise → Dropout
    """

    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Linear(d_model, d_model * 2)
        self.glu = nn.GLU(dim=-1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = Swish()
        self.pointwise2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L, D]"""
        residual = x
        x = self.norm(x)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = x.transpose(1, 2)  # [B, D, L] for Conv1d
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # [B, L, D]
        x = self.pointwise2(x)
        x = self.dropout(x)
        return residual + x


class ConformerMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with relative positional encoding.

    Simplified version: no positional encoding (sufficient for short sequences).
    """

    def __init__(self, d_model, n_head=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, L, D]"""
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        q = self.w_q(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_head, self.d_k).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.w_o(out)
        out = self.dropout(out)
        return residual + out


class ConformerBlock(nn.Module):
    """Full Conformer Block (Macaron structure):

        x → FFN(½) → MHSA → ConvModule → FFN(½) → LayerNorm → out
    """

    def __init__(self, d_model, n_head=4, conv_kernel_size=31,
                 ff_expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, ff_expansion_factor, dropout)
        self.mhsa = ConformerMultiHeadAttention(d_model, n_head, dropout)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = ConformerFeedForward(d_model, ff_expansion_factor, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [B, L, D]"""
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv_module(x)
        x = self.ffn2(x)
        return self.final_norm(x)


# ============================================================================
#  Conformer-based sequence module replacing BLSTM
# ============================================================================

class ConformerSeqModule(nn.Module):
    """Drop-in replacement for nn.LSTM(bidirectional=True).

    Input:  [B, L, D_in]
    Output: [B, L, 2*hidden]  (to match BLSTM output dimension)
    """

    def __init__(self, d_input, d_hidden, n_head=4, conv_kernel_size=31,
                 ff_expansion_factor=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_hidden)
        self.conformer = ConformerBlock(
            d_model=d_hidden,
            n_head=n_head,
            conv_kernel_size=conv_kernel_size,
            ff_expansion_factor=ff_expansion_factor,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(d_hidden, d_hidden * 2)

    def forward(self, x):
        """x: [B, L, D_in] → [B, L, 2*d_hidden]"""
        x = self.input_proj(x)
        x = self.conformer(x)
        return self.output_proj(x)


# ============================================================================
#  ConformerGrid Block — like GridNetV3Block but with ConformerSeqModule
# ============================================================================

class ConformerGridBlock(nn.Module):
    """TF-GridNet block with Conformer replacing BLSTM.

    Structure:
      1. Intra-frame processing (frequency axis) — Conformer
      2. Inter-frame processing (time axis) — Conformer
      3. Cross-frame self-attention — unchanged
    """

    def __init__(
        self,
        emb_dim: int,
        emb_ks: int,
        emb_hs: int,
        hidden_channels: int,
        n_head: int = 4,
        qk_output_channel: int = 4,
        eps: float = 1e-5,
        conformer_n_head: int = 4,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        in_channels = emb_dim * emb_ks
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        # ── Intra-frame (frequency axis): Conformer replaces BLSTM ──
        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_conformer = ConformerSeqModule(
            d_input=in_channels, d_hidden=hidden_channels,
            n_head=conformer_n_head, conv_kernel_size=conv_kernel_size,
            ff_expansion_factor=ff_expansion_factor, dropout=dropout
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        # ── Inter-frame (time axis): Conformer replaces BLSTM ──
        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_conformer = ConformerSeqModule(
            d_input=in_channels, d_hidden=hidden_channels,
            n_head=conformer_n_head, conv_kernel_size=conv_kernel_size,
            ff_expansion_factor=ff_expansion_factor, dropout=dropout
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs)

        # ── Self-Attention (unchanged from GridNetV3Block) ──
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

    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        batch_size, channels, old_t, old_q = x.shape
        overlap = self.emb_ks - self.emb_hs
        padded_t = math.ceil((old_t + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        padded_q = math.ceil((old_q + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks

        x = x.permute(0, 2, 3, 1)
        x = F.pad(x, (0, 0, overlap, padded_q - old_q - overlap, overlap, padded_t - old_t - overlap))

        # ── 1. Intra-frame (frequency axis) ──
        input_ = x
        intra = self.intra_norm(input_)
        if self.emb_ks == self.emb_hs:
            intra = intra.view([batch_size * padded_t, -1, self.emb_ks * channels])
            intra = self.intra_conformer(intra)
            intra = self.intra_linear(intra)
            intra = intra.view([batch_size, padded_t, padded_q, channels])
        else:
            intra = intra.view([batch_size * padded_t, padded_q, channels])
            intra = intra.transpose(1, 2)
            intra = F.unfold(intra[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            intra = intra.transpose(1, 2)
            intra = self.intra_conformer(intra)
            intra = intra.transpose(1, 2)
            intra = self.intra_linear(intra)
            intra = intra.view([batch_size, padded_t, channels, padded_q]).transpose(-2, -1)
        intra = intra + input_
        intra = intra.transpose(1, 2)

        # ── 2. Inter-frame (time axis) ──
        input_ = intra
        inter = self.inter_norm(input_)
        if self.emb_ks == self.emb_hs:
            inter = inter.view([batch_size * padded_q, -1, self.emb_ks * channels])
            inter = self.inter_conformer(inter)
            inter = self.inter_linear(inter)
            inter = inter.view([batch_size, padded_q, padded_t, channels])
        else:
            inter = inter.view([batch_size * padded_q, padded_t, channels])
            inter = inter.transpose(1, 2)
            inter = F.unfold(inter[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            inter = inter.transpose(1, 2)
            inter = self.inter_conformer(inter)
            inter = inter.transpose(1, 2)
            inter = self.inter_linear(inter)
            inter = inter.view([batch_size, padded_q, channels, padded_t]).transpose(-2, -1)
        inter = inter + input_

        inter = inter.permute(0, 3, 2, 1)
        inter = inter[..., overlap: overlap + old_t, overlap: overlap + old_q]
        residual = inter

        # ── 3. Self-Attention (unchanged) ──
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

        attn_mat = torch.matmul(q, k) / (emb_dim ** 0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        v = torch.matmul(attn_mat, v)

        v = v.reshape(old_v_shape).transpose(1, 2)
        emb_dim = v.shape[1]
        batch = v.contiguous().view([batch_size, self.n_head * emb_dim, old_t, old_q])
        batch = self.attn_concat_proj(batch)
        return batch + residual


# ============================================================================
#  Conformer-GridNet Core & Separator
# ============================================================================

class ConformerGridNetCore(nn.Module):
    def __init__(
        self,
        n_srcs: int = 2,
        n_imics: int = 1,
        n_layers: int = 3,
        hidden_channels: int = 128,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 4,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        eps: float = 1.0e-5,
        conformer_n_head: int = 4,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )

        self.blocks = nn.ModuleList([
            ConformerGridBlock(
                emb_dim=emb_dim,
                emb_ks=emb_ks,
                emb_hs=emb_hs,
                hidden_channels=hidden_channels,
                n_head=attn_n_head,
                qk_output_channel=attn_qk_output_channel,
                eps=eps,
                conformer_n_head=conformer_n_head,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=padding)

    def forward(self, input_spectrum: torch.Tensor):
        batch_size = input_spectrum.shape[0]
        mix_spec_ri = torch.stack([input_spectrum.real, input_spectrum.imag], dim=1)

        batch = self.conv(mix_spec_ri)
        for block in self.blocks:
            batch = block(batch)
        batch = self.deconv(batch)

        batch = batch.view([batch_size, self.n_srcs, 2, *input_spectrum.shape[1:]])
        return torch.complex(batch[:, :, 0], batch[:, :, 1])


class ConformerGridNetSeparator1D(nn.Module):
    """Conformer-GridNet wrapper for IQ time-domain separation.

    Interface matches TFGridNetV3Separator1D exactly.

    Input:  [B, 2, L]
    Output: [B, 2*n_srcs, L]
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
        n_layers: int = 3,
        hidden_channels: int = 128,
        attn_n_head: int = 4,
        attn_qk_output_channel: int = 4,
        emb_dim: int = 48,
        emb_ks: int = 4,
        emb_hs: int = 1,
        conformer_n_head: int = 4,
        conv_kernel_size: int = 31,
        ff_expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.normalize_input = normalize_input
        self.eps = eps

        self.separator = ConformerGridNetCore(
            n_srcs=n_srcs,
            n_layers=n_layers,
            hidden_channels=hidden_channels,
            attn_n_head=attn_n_head,
            attn_qk_output_channel=attn_qk_output_channel,
            emb_dim=emb_dim,
            emb_ks=emb_ks,
            emb_hs=emb_hs,
            eps=eps,
            conformer_n_head=conformer_n_head,
            conv_kernel_size=conv_kernel_size,
            ff_expansion_factor=ff_expansion_factor,
            dropout=dropout,
        )
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != 2:
            raise ValueError(f"Expected [B, 2, L], got {tuple(x.shape)}")

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
            onesided=False,
            return_complex=True,
        )
        mix_spec = mix_spec.transpose(1, 2).contiguous()

        sep_specs = self.separator(mix_spec)

        reconstructed = []
        for src_idx in range(self.n_srcs):
            sep_spec = sep_specs[:, src_idx].to(torch.complex64)
            sep_spec = sep_spec.transpose(1, 2).contiguous()
            signal = torch.istft(
                sep_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                center=self.center,
                onesided=False,
                length=length,
                return_complex=True,
            )
            signal = signal * std
            reconstructed.append(signal)

        output = torch.stack(
            [torch.stack([sig.real, sig.imag], dim=1) for sig in reconstructed],
            dim=1,
        )
        output = output.reshape(batch_size, self.n_srcs * 2, length)
        return output.to(dtype=original_dtype)
