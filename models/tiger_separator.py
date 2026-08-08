"""TIGER (Time-frequency Interleaved Gain Extraction and Reconstruction)
adapted for IQ signal separation in the IQUMamba pipeline.

Original paper: Xu et al., "TIGER: Time-frequency Interleaved Gain Extraction
and Reconstruction for Efficient Speech Separation", ICLR 2025.

This file is self-contained – all required layers (activations, normalizations,
sub-modules) are inlined so there are NO external look2hear dependencies.

Input:  [B, 2, L]  (I_mix, Q_mix)
Output: [B, 2*n_srcs, L]  (I1, Q1, I2, Q2, ...)
"""

import math
import inspect
from collections.abc import Iterable
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  Inlined activations & normalizations (from look2hear.layers)
# ============================================================================

_ACTIVATION_MAP = {
    "linear": nn.Identity,
    "relu": nn.ReLU,
    "prelu": nn.PReLU,
    "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def _get_activation(identifier):
    if identifier is None:
        return nn.Identity()
    if callable(identifier) and not isinstance(identifier, str):
        return identifier
    key = identifier.lower() if isinstance(identifier, str) else str(identifier)
    cls = _ACTIVATION_MAP.get(key)
    if cls is None:
        raise ValueError(f"Unknown activation: {identifier}")
    return cls()


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension: Iterable, eps: float = 1e-5):
        super().__init__()
        if isinstance(input_dimension, int):
            input_dimension = (input_dimension, 1)
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.dim = (1, 3) if param_size[-1] > 1 else (1,)
        self.gamma = nn.Parameter(torch.ones(*param_size))
        self.beta = nn.Parameter(torch.zeros(*param_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mu_ = x.mean(dim=self.dim, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.dim, unbiased=False, keepdim=True) + self.eps)
        return ((x - mu_) / std_) * self.gamma + self.beta


_NORM_MAP = {
    "layernormalization4d": LayerNormalization4D,
    "gln": lambda ch: nn.GroupNorm(1, ch, eps=1e-8),
}


def _get_norm(identifier, *args, **kwargs):
    if identifier is None:
        return nn.Identity()
    key = identifier.lower() if isinstance(identifier, str) else str(identifier)
    cls = _NORM_MAP.get(key)
    if cls is None:
        return nn.GroupNorm(1, args[0], eps=1e-8)
    return cls(*args, **kwargs)


# ============================================================================
#  Building blocks (from TIGER tiger.py, made self-contained)
# ============================================================================

def GlobLN(nOut):
    return nn.GroupNorm(1, nOut, eps=1e-8)


class ConvNormAct(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ConvNorm(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, x):
        return self.norm(self.conv(x))


class ATTConvActNorm(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=-1, stride=1, groups=1,
                 dilation=1, padding=None, norm_type=None, act_type=None,
                 n_freqs=-1, xavier_init=False, bias=True, is2d=False):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        if padding is None:
            padding = 0 if stride > 1 else "same"

        if kernel_size > 0:
            conv = nn.Conv2d if is2d else nn.Conv1d
            self.conv = conv(in_chan, out_chan, kernel_size, stride=stride,
                             padding=padding, dilation=dilation, groups=groups, bias=bias)
            if xavier_init:
                nn.init.xavier_uniform_(self.conv.weight)
        else:
            self.conv = nn.Identity()

        self.act = _get_activation(act_type)

        if norm_type and norm_type.lower() == "layernormalization4d":
            self.norm = LayerNormalization4D((out_chan, n_freqs))
        else:
            self.norm = nn.GroupNorm(1, out_chan, eps=1e-8) if out_chan > 0 else nn.Identity()

    def forward(self, x):
        return self.norm(self.act(self.conv(x)))


class DilatedConvNorm(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, x):
        return self.norm(self.conv(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size)
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InjectionMultiSum(nn.Module):
    def __init__(self, inp, oup, kernel=1):
        super().__init__()
        groups = inp if inp == oup else 1
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")
        return local_feat * sig_act + global_feat


class UConvBlock(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1))
        for _ in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5, stride=2, groups=in_channels, d=1))

        self.loc_glo_fus = nn.ModuleList([InjectionMultiSum(in_channels, in_channels) for _ in range(upsampling_depth)])
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)
        self.last_layer = nn.ModuleList([InjectionMultiSum(in_channels, in_channels, 5) for _ in range(self.depth - 1)])

    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            output.append(self.spp_dw[k](output[-1]))

        global_f = torch.zeros(output[-1].shape, requires_grad=True, device=output1.device)
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool1d(fea, output[-1].shape[-1])
        global_f = self.globalatt(global_f)

        x_fused = []
        for idx in range(self.depth):
            x_fused.append(self.loc_glo_fus[idx](output[idx], global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)

        return self.res_conv(expanded) + residual


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_chan, n_freqs, n_head=4, hid_chan=4, act_type="prelu",
                 norm_type="LayerNormalization4D", dim=3):
        super().__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.dim = dim
        assert in_chan % n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()
        for _ in range(n_head):
            self.Queries.append(
                ATTConvActNorm(in_chan, hid_chan, 1, act_type=act_type,
                               norm_type=norm_type, n_freqs=n_freqs, is2d=True))
            self.Keys.append(
                ATTConvActNorm(in_chan, hid_chan, 1, act_type=act_type,
                               norm_type=norm_type, n_freqs=n_freqs, is2d=True))
            self.Values.append(
                ATTConvActNorm(in_chan, in_chan // n_head, 1, act_type=act_type,
                               norm_type=norm_type, n_freqs=n_freqs, is2d=True))

        self.attn_concat_proj = ATTConvActNorm(
            in_chan, in_chan, 1, act_type=act_type,
            norm_type=norm_type, n_freqs=n_freqs, is2d=True)

    def forward(self, x):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        batch_size, _, time, freq = x.size()
        residual = x

        Q = torch.cat([q(x) for q in self.Queries], dim=0)
        K = torch.cat([k(x) for k in self.Keys], dim=0)
        V = torch.cat([v(x) for v in self.Values], dim=0)

        Q = Q.transpose(1, 2).flatten(start_dim=2)
        K = K.transpose(1, 2).flatten(start_dim=2)
        V = V.transpose(1, 2)
        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim ** 0.5)
        attn_mat = F.softmax(attn_mat, dim=2)
        V = torch.matmul(attn_mat, V)

        V = V.reshape(old_shape).transpose(1, 2)
        emb_dim = V.shape[1]
        x = V.view([self.n_head, batch_size, emb_dim, time, freq])
        x = x.transpose(0, 1).contiguous()
        x = x.view([batch_size, self.n_head * emb_dim, time, freq])
        x = self.attn_concat_proj(x)
        x = x + residual

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        return x


class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, nband=8,
                 upsampling_depth=3, n_head=4, att_hid_chan=4,
                 kernel_size=8, stride=1, _iter=4):
        super().__init__()
        self.nband = nband
        self.freq_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan,
                                     act_type="prelu", norm_type="LayerNormalization4D", dim=4),
            LayerNormalization4D((out_channels, 1)),
        ])
        self.frame_path = nn.ModuleList([
            UConvBlock(out_channels, in_channels, upsampling_depth),
            MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan,
                                     act_type="prelu", norm_type="LayerNormalization4D", dim=4),
            LayerNormalization4D((out_channels, 1)),
        ])
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        B, nband, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self._freq_time_process(x, B, nband, N, T)
            else:
                x = self._freq_time_process(self.concat_block(mixture + x), B, nband, N, T)
        return x.permute(0, 2, 1, 3).contiguous()

    def _freq_time_process(self, x, B, nband, N, T):
        residual_1 = x.clone()
        x = x.permute(0, 3, 1, 2).contiguous()
        freq_fea = self.freq_path[0](x.view(B * T, N, nband))
        freq_fea = freq_fea.view(B, T, N, nband).permute(0, 2, 1, 3).contiguous()
        freq_fea = self.freq_path[1](freq_fea)
        freq_fea = self.freq_path[2](freq_fea)
        freq_fea = freq_fea.permute(0, 1, 3, 2).contiguous()
        x = freq_fea + residual_1

        residual_2 = x.clone()
        x2 = x.permute(0, 2, 1, 3).contiguous()
        frame_fea = self.frame_path[0](x2.view(B * nband, N, T))
        frame_fea = frame_fea.view(B, nband, N, T).permute(0, 2, 1, 3).contiguous()
        frame_fea = self.frame_path[1](frame_fea)
        frame_fea = self.frame_path[2](frame_fea)
        x = frame_fea + residual_2
        return x


class TIGERCore(nn.Module):
    """Core TIGER separator adapted for RF/IQ signals.

    Unlike the original TIGER which uses auditory band-split (25 Hz, 100 Hz, …),
    this version uses *uniform* frequency band splitting suitable for
    baseband IQ signals where all bands are equally important.
    """

    def __init__(
        self,
        n_srcs: int = 2,
        n_fft: int = 256,
        out_channels: int = 128,
        in_channels: int = 512,
        num_blocks: int = 16,
        upsampling_depth: int = 4,
        att_n_head: int = 4,
        att_hid_chan: int = 4,
        nband: int = 8,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.enc_dim = n_fft
        self.feature_dim = out_channels
        self.nband = nband
        self.eps = eps

        # Compute uniform band widths
        base_bw = self.enc_dim // nband
        remainder = self.enc_dim - base_bw * nband
        self.band_width = [base_bw + (1 if i < remainder else 0) for i in range(nband)]

        # Per-band bottleneck: real+imag → feature_dim
        self.BN = nn.ModuleList()
        for bw in self.band_width:
            self.BN.append(nn.Sequential(
                nn.GroupNorm(1, bw * 2, eps),
                nn.Conv1d(bw * 2, self.feature_dim, 1),
            ))

        # Separator core
        self.separator = Recurrent(
            self.feature_dim, in_channels, self.nband,
            upsampling_depth, att_n_head, att_hid_chan,
            kernel_size=8, stride=1, _iter=num_blocks,
        )

        # Per-band mask estimation
        self.mask = nn.ModuleList()
        for bw in self.band_width:
            self.mask.append(nn.Sequential(
                nn.PReLU(),
                nn.Conv1d(self.feature_dim, bw * 4 * n_srcs, 1, groups=n_srcs),
            ))

    def forward(self, spec: torch.Tensor):
        """
        Args:
            spec: complex spectrogram [B, F, T]
        Returns:
            separated complex spectrograms [B, n_srcs, F, T]
        """
        B = spec.shape[0]

        # Stack real & imag and split into sub-bands
        spec_RI = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, T]
        subband_spec_RI = []
        subband_spec = []
        band_idx = 0
        for bw in self.band_width:
            subband_spec_RI.append(spec_RI[:, :, band_idx:band_idx + bw].contiguous())
            subband_spec.append(spec[:, band_idx:band_idx + bw])
            band_idx += bw

        # Bottleneck per band
        subband_feature = []
        for i, bw in enumerate(self.band_width):
            feat = self.BN[i](subband_spec_RI[i].view(B, bw * 2, -1))
            subband_feature.append(feat)
        subband_feature = torch.stack(subband_feature, dim=1)  # [B, nband, feat_dim, T]

        # Separator
        sep_output = self.separator(subband_feature)  # [B, nband, feat_dim, T]

        # Mask estimation and application
        sep_subband_spec = []
        for i, bw in enumerate(self.band_width):
            this_output = self.mask[i](sep_output[:, i]).view(B, 2, 2, self.n_srcs, bw, -1)
            this_mask = this_output[:, 0] * torch.sigmoid(this_output[:, 1])  # [B, 2, K, BW, T]
            this_mask_real = this_mask[:, 0]
            this_mask_imag = this_mask[:, 1]
            # Force mask sum to 1
            this_mask_real = this_mask_real - (this_mask_real.sum(1, keepdim=True) - 1) / self.n_srcs
            this_mask_imag = this_mask_imag - this_mask_imag.sum(1, keepdim=True) / self.n_srcs
            est_real = (subband_spec[i].real.unsqueeze(1) * this_mask_real
                        - subband_spec[i].imag.unsqueeze(1) * this_mask_imag)
            est_imag = (subband_spec[i].real.unsqueeze(1) * this_mask_imag
                        + subband_spec[i].imag.unsqueeze(1) * this_mask_real)
            sep_subband_spec.append(torch.complex(est_real, est_imag))

        return torch.cat(sep_subband_spec, dim=2)  # [B, n_srcs, F, T]


class TIGERSeparator1D(nn.Module):
    """TIGER wrapper for IQ time-domain separation, matching
    TFGridNetV3Separator1D interface.

    Input:  x: [B, 2, L]   where channels are [I_mix, Q_mix]
    Output: y: [B, 2*n_srcs, L]  where channels are [I1, Q1, I2, Q2, ...]
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
        # TIGER-specific
        out_channels: int = 128,
        in_channels: int = 512,
        num_blocks: int = 16,
        upsampling_depth: int = 4,
        att_n_head: int = 4,
        att_hid_chan: int = 4,
        nband: int = 8,
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.normalize_input = normalize_input
        self.eps = eps

        self.separator = TIGERCore(
            n_srcs=n_srcs,
            n_fft=n_fft,
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            att_n_head=att_n_head,
            att_hid_chan=att_hid_chan,
            nband=nband,
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

        # Build complex IQ signal
        mix_complex = torch.complex(x[:, 0], x[:, 1])

        # Optional power normalization
        if self.normalize_input:
            std = mix_complex.abs().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
            mix_complex = mix_complex / std
        else:
            std = torch.ones((batch_size, 1), device=x.device, dtype=x.real.dtype)

        # STFT
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
        )  # [B, F, T]  (F = n_fft for complex input)

        # Separate
        sep_specs = self.separator(mix_spec)  # [B, n_srcs, F, T]

        # iSTFT per source
        reconstructed = []
        for src_idx in range(self.n_srcs):
            sep_spec = sep_specs[:, src_idx]  # [B, F, T]
            # Force float32 for iSTFT to avoid ComplexHalf issues
            sep_spec = sep_spec.to(torch.complex64)
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

        # Stack: [I1, Q1, I2, Q2, ...]
        output = torch.stack(
            [torch.stack([sig.real, sig.imag], dim=1) for sig in reconstructed],
            dim=1,
        )
        output = output.reshape(batch_size, self.n_srcs * 2, length)
        return output.to(dtype=original_dtype)
