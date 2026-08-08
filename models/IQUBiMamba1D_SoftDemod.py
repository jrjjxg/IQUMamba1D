"""IQUBiMamba1D_SoftDemod — Joint Blind Source Separation + Differentiable
Soft Demodulation network.

Architecture overview
=====================
    mixed IQ signal  (B, 2, L)
          │
          ▼
    ┌─────────────────────────────┐
    │  Separation Backbone        │
    │  (ResidualBiMambaEncoder    │
    │   + UNetResDecoder)         │
    │  — identical to IQUBiMamba1D│
    └─────────┬───────────────────┘
              │  separated sources: (B, 2*K, L)
              │
     ┌────────┴────────┐
     │  split per src  │  →  K tensors of (B, 2, L)
     └────────┬────────┘
              │  (optionally .detach() to block grad)
              ▼
    ┌─────────────────────────────┐
    │  Shared Soft Demod Head     │  ← Conv1D + BiGRU + Adaptive Pool + FC
    │  applied to each source     │
    └─────────┬───────────────────┘
              │
              ▼
    soft bit logits: K tensors of (B, num_bits)

Training modes
==============
- ``mode='sep_only'``   : forward returns separation output only (standard BSS).
- ``mode='demod_only'`` : encoder/decoder frozen, only demod head trains.
- ``mode='joint'``      : end-to-end, both outputs returned.
- ``detach_demod=True`` : demod head receives *detached* waveforms so its
                          gradients do NOT flow back into the separator.

Loss design
===========
The companion ``pit_si_snr_huber_demod_loss`` (in util/loss.py) uses **PIT to
select the best permutation based on the separation loss**, then evaluates
Binary Cross-Entropy for the soft bits under that *same* permutation.
This guarantees bit-output alignment even when sources are swapped by PIT.

Why BiGRU for demod?
====================
Communication signals exhibit strong inter-symbol interference (ISI) and
cyclo-stationary structure.  A bidirectional GRU captures both forward and
backward temporal dependencies, which is essential for matched-filtering
and soft bit estimation.  It is lightweight, well-proven, and avoids adding
CUDA kernel requirements beyond what the Mamba backbone already needs.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Type

from models.IQUBiMamba1D import (
    ResidualBiMambaEncoder,
    UNetResDecoder,
)


class SoftDemodHead(nn.Module):
    """Differentiable soft demodulation head.

    Maps a separated IQ signal → soft bit probabilities.
    Architecture: Conv1D (local feature extraction) → BiGRU (ISI modeling)
                  → AdaptiveAvgPool1d (temporal alignment) → FC (bit output).

    The head is **shared** across all separated sources, which keeps
    parameter count low and encourages modulation-agnostic features.

    Args:
        num_bits:    default number of output bits (can be overridden in forward)
        hidden:      intermediate Conv1D channel width
        rnn_hidden:  BiGRU hidden dim (output is 2×rnn_hidden due to bidirectional)
        dropout:     dropout before the output layer
    """

    def __init__(self, num_bits: int = 615, bits_per_symbol: int = 3,
                 hidden: int = 64, rnn_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.num_bits = num_bits
        self.bits_per_symbol = bits_per_symbol

        # Local feature extraction from raw IQ
        self.conv = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=15, padding=7),
            nn.GELU(),
            nn.InstanceNorm1d(hidden),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.InstanceNorm1d(hidden),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )

        # Bidirectional GRU for inter-symbol dependency modeling
        self.rnn = nn.GRU(
            input_size=hidden,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
        )

        # Output: pointwise Conv1d → 1 logit per time position
        self.num_symbol_classes = 1 << bits_per_symbol
        self.dropout = nn.Dropout(dropout)
        self.bit_symbol_output = nn.Linear(rnn_hidden * 2, bits_per_symbol)
        self.symbol_class_output = nn.Linear(rnn_hidden * 2, self.num_symbol_classes)
        self.bit_output = nn.Conv1d(rnn_hidden * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, num_bits: int = None):
        """
        Args:
            x:        (B, 2, L) — one separated IQ source
            num_bits: override for output length (default: self.num_bits)
        Returns:
            (B, num_bits) — raw logits (apply sigmoid for probabilities)
        """
        nb = num_bits if num_bits is not None else self.num_bits

        feat = self.conv(x)                               # (B, hidden, L)
        feat_t = feat.transpose(1, 2)                     # (B, L, hidden)
        rnn_out, _ = self.rnn(feat_t)                     # (B, L, 2*rnn_hidden)
        rnn_out = rnn_out.transpose(1, 2)                 # (B, 2*rnn_hidden, L)

        if nb % self.bits_per_symbol == 0:
            num_symbols = nb // self.bits_per_symbol
            pooled = F.adaptive_avg_pool1d(rnn_out, num_symbols)  # (B, 2H, S)
            pooled = pooled.transpose(1, 2)                       # (B, S, 2H)
            pooled = self.dropout(pooled)
            bit_logits = self.bit_symbol_output(pooled)           # (B, S, bps)
            symbol_logits = self.symbol_class_output(pooled)      # (B, S, M)
            return bit_logits.reshape(bit_logits.shape[0], -1), symbol_logits

        pooled = F.adaptive_avg_pool1d(rnn_out, nb)       # (B, 2*rnn_hidden, nb)
        pooled = self.dropout(pooled)
        bit_logits = self.bit_output(pooled).squeeze(1)   # (B, nb)
        return bit_logits, None


class IQUBiMamba1D_SoftDemod(nn.Module):
    """Bidirectional Mamba U-Net + Soft Demodulation head for joint
    signal separation and differentiable bit estimation.

    The separation backbone is identical to IQUBiMamba1D (stage 12).
    A shared SoftDemodHead is applied to each separated source.
    """

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
        # Demod head parameters
        num_bits: int = 615,
        demod_bits_per_symbol: int = 3,
        demod_hidden: int = 64,
        demod_rnn_hidden: int = 64,
        demod_dropout: float = 0.2,
        detach_demod: bool = False,
        # Standard params
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {'inplace': True},
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.encoder = ResidualBiMambaEncoder(
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
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )

        self.num_sources = num_classes // 2
        self.detach_demod = detach_demod

        # Shared soft demodulation head
        self.demod_head = SoftDemodHead(
            num_bits=num_bits,
            bits_per_symbol=demod_bits_per_symbol,
            hidden=demod_hidden,
            rnn_hidden=demod_rnn_hidden,
            dropout=demod_dropout,
        )

    def forward(self, x: torch.Tensor, mode: str = 'sep_only',
                num_bits: int = None) -> torch.Tensor:
        """
        Args:
            x:        (B, 2, L) mixed IQ input
            mode:     'sep_only' | 'demod_only' | 'joint'
            num_bits: override output bits length (None = use default)
        Returns:
            mode='sep_only'  → (B, 2*K, L) separated waveforms
            mode='demod_only'→ ((B, 2*K, L), [K × (B, num_bits)])  separator detached
            mode='joint'     → ((B, 2*K, L), [K × (B, num_bits)])
        """
        skips = self.encoder(x)
        sep_out = self.decoder(skips)

        if mode == 'sep_only':
            return sep_out

        K = self.num_sources
        bit_logits = []
        symbol_logits = []
        for k in range(K):
            src = sep_out[:, 2 * k : 2 * k + 2, :]
            if self.detach_demod or mode == 'demod_only':
                src = src.detach()
            bit_logit_k, symbol_logit_k = self.demod_head(src, num_bits=num_bits)
            bit_logits.append(bit_logit_k)
            symbol_logits.append(symbol_logit_k)

        if mode == 'demod_only':
            return sep_out.detach(), {
                'bit_logits': bit_logits,
                'symbol_logits': symbol_logits,
            }

        # mode == 'joint'
        return sep_out, {
            'bit_logits': bit_logits,
            'symbol_logits': symbol_logits,
        }

    # ------------------------------------------------------------------
    # Convenience methods for staged training
    # ------------------------------------------------------------------
    def freeze_separator(self):
        """Freeze encoder + decoder (only demod head trains)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def unfreeze_separator(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        for p in self.decoder.parameters():
            p.requires_grad = True

    def freeze_demod(self):
        """Freeze demod head (only separator trains)."""
        for p in self.demod_head.parameters():
            p.requires_grad = False

    def unfreeze_demod(self):
        for p in self.demod_head.parameters():
            p.requires_grad = True

    def load_separator_weights(self, state_dict_or_path, strict: bool = True):
        """Load pretrained IQUBiMamba1D weights into encoder + decoder."""
        from pathlib import Path
        if isinstance(state_dict_or_path, (str, Path)):
            ckpt = torch.load(str(state_dict_or_path), map_location='cpu')
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = state_dict_or_path

        sep_dict = {}
        for key, val in state_dict.items():
            if key.startswith('encoder.') or key.startswith('decoder.'):
                sep_dict[key] = val
        missing, unexpected = self.load_state_dict(sep_dict, strict=False)
        loaded = len(sep_dict) - len(unexpected)
        print(f"[SoftDemod] Loaded {loaded} separator keys "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")


class ReceiverAwareSymbolAdapter(nn.Module):
    """Convert separated waveform into symbol-rate tokens with light receiver priors.

    The adapter keeps the pipeline end-to-end trainable, while injecting
    information a plain classification head usually misses:
      - waveform RMS normalization
      - coarse symbol-rate downsampling
      - per-symbol normalized IQ / magnitude
      - local symbol-to-symbol deltas
    """

    def __init__(
        self,
        token_hidden: int,
        adapter_hidden: int,
        approx_symbol_span: int,
        dropout: float,
    ):
        super().__init__()
        span = max(3, int(approx_symbol_span))
        kernel = max(5, span * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1
        stride = max(1, span // 2)

        self.eps = 1e-6
        self.stem = nn.Sequential(
            nn.Conv1d(2, adapter_hidden, kernel_size=15, padding=7),
            nn.GELU(),
            nn.InstanceNorm1d(adapter_hidden),
            nn.Conv1d(adapter_hidden, adapter_hidden, kernel_size=kernel, padding=kernel // 2, groups=adapter_hidden),
            nn.GELU(),
            nn.Conv1d(adapter_hidden, token_hidden, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm1d(token_hidden),
        )
        self.coarse_downsample = nn.Sequential(
            nn.Conv1d(token_hidden, token_hidden, kernel_size=kernel, stride=stride, padding=kernel // 2),
            nn.GELU(),
            nn.InstanceNorm1d(token_hidden),
        )
        self.token_fusion = nn.Sequential(
            nn.Conv1d(token_hidden * 2 + 5, token_hidden, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm1d(token_hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, num_symbols: int) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + self.eps)
        x_norm = x / rms

        feat = self.stem(x_norm)
        coarse = self.coarse_downsample(feat)

        pooled_avg = F.adaptive_avg_pool1d(coarse, num_symbols)
        pooled_max = F.adaptive_max_pool1d(coarse, num_symbols)
        pooled_iq = F.adaptive_avg_pool1d(x_norm, num_symbols)

        amp = torch.sqrt(torch.sum(pooled_iq ** 2, dim=1, keepdim=True) + self.eps)
        unit_iq = pooled_iq / amp

        delta = unit_iq[:, :, 1:] - unit_iq[:, :, :-1]
        delta = F.pad(delta, (1, 0))

        tokens = torch.cat([pooled_avg, pooled_max, unit_iq, amp, delta], dim=1)
        tokens = self.token_fusion(tokens)
        return tokens.transpose(1, 2)


class PrototypeSymbolClassifier(nn.Module):
    """Prototype classifier encourages constellation-aware symbol embeddings."""

    def __init__(self, in_dim: int, num_classes: int, logit_scale: float = 12.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, in_dim)
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)).log())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.proj(x), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * torch.matmul(x, prototypes.t())


class ReceiverAwareSoftDemodHead(nn.Module):
    """Receiver-aware demapper head inspired by neural demapper literature.

    Compared with the baseline head, this version first builds symbol-rate
    tokens and only then predicts symbol classes / soft bits.
    """

    def __init__(
        self,
        input_size: int,
        num_bits: int = 615,
        bits_per_symbol: int = 3,
        adapter_hidden: int = 96,
        token_hidden: int = 128,
        rnn_hidden: int = 96,
        context_layers: int = 2,
        dropout: float = 0.2,
        symbol_logit_scale: float = 12.0,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.bits_per_symbol = bits_per_symbol
        self.num_symbol_classes = 1 << bits_per_symbol
        default_symbols = max(1, num_bits // bits_per_symbol)
        approx_symbol_span = max(1, int(round(float(input_size) / float(default_symbols))))

        self.adapter = ReceiverAwareSymbolAdapter(
            token_hidden=token_hidden,
            adapter_hidden=adapter_hidden,
            approx_symbol_span=approx_symbol_span,
            dropout=dropout,
        )
        self.context = nn.GRU(
            input_size=token_hidden,
            hidden_size=rnn_hidden,
            num_layers=context_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if context_layers > 1 else 0.0,
        )
        ctx_dim = rnn_hidden * 2
        self.context_norm = nn.LayerNorm(ctx_dim)
        self.bit_head = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_dim, bits_per_symbol),
        )
        self.symbol_head = PrototypeSymbolClassifier(
            in_dim=ctx_dim,
            num_classes=self.num_symbol_classes,
            logit_scale=symbol_logit_scale,
        )
        self.fallback_bit_head = nn.Sequential(
            nn.Conv1d(2, adapter_hidden, kernel_size=15, padding=7),
            nn.GELU(),
            nn.InstanceNorm1d(adapter_hidden),
            nn.Conv1d(adapter_hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, num_bits: int = None):
        nb = num_bits if num_bits is not None else self.num_bits

        if nb % self.bits_per_symbol != 0:
            bit_logits = self.fallback_bit_head(F.adaptive_avg_pool1d(x, nb)).squeeze(1)
            return bit_logits, None

        num_symbols = max(1, nb // self.bits_per_symbol)
        tokens = self.adapter(x, num_symbols=num_symbols)
        context, _ = self.context(tokens)
        context = self.context_norm(context)

        bit_logits = self.bit_head(context).reshape(context.shape[0], -1)
        symbol_logits = self.symbol_head(context)
        return bit_logits[:, :nb], symbol_logits


class IQUBiMamba1D_SoftDemodV2(IQUBiMamba1D_SoftDemod):
    """Receiver-aware SoftDemod variant.

    Keeps the original BiMamba separator and swaps in a stronger demapper head
    that operates on symbol-rate tokens instead of directly pooling the whole
    waveform into bit positions.
    """

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
        num_bits: int = 615,
        demod_bits_per_symbol: int = 3,
        demod_hidden: int = 64,
        demod_rnn_hidden: int = 96,
        demod_dropout: float = 0.2,
        detach_demod: bool = False,
        demod_adapter_hidden: int = 96,
        demod_symbol_hidden: int = 128,
        demod_context_layers: int = 2,
        demod_symbol_logit_scale: float = 12.0,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {'inplace': True},
        deep_supervision: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            num_bits=num_bits,
            demod_bits_per_symbol=demod_bits_per_symbol,
            demod_hidden=demod_hidden,
            demod_rnn_hidden=demod_rnn_hidden,
            demod_dropout=demod_dropout,
            detach_demod=detach_demod,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.demod_head = ReceiverAwareSoftDemodHead(
            input_size=input_size,
            num_bits=num_bits,
            bits_per_symbol=demod_bits_per_symbol,
            adapter_hidden=demod_adapter_hidden,
            token_hidden=demod_symbol_hidden,
            rnn_hidden=demod_rnn_hidden,
            context_layers=demod_context_layers,
            dropout=demod_dropout,
            symbol_logit_scale=demod_symbol_logit_scale,
        )


class OffsetPhaseAwareSymbolAdapter(nn.Module):
    """Stronger receiver-aware adapter with timing-hypothesis fusion and phase correction."""

    def __init__(
        self,
        token_hidden: int,
        adapter_hidden: int,
        approx_symbol_span: int,
        num_offset_hypotheses: int,
        dropout: float,
    ):
        super().__init__()
        span = max(3, int(approx_symbol_span))
        kernel = max(5, span * 2 + 1)
        if kernel % 2 == 0:
            kernel += 1

        self.eps = 1e-6
        self.approx_symbol_span = span
        self.offsets = sorted(set(
            int(round(i * span / max(1, num_offset_hypotheses)))
            for i in range(max(1, num_offset_hypotheses))
        ))

        self.stem = nn.Sequential(
            nn.Conv1d(2, adapter_hidden, kernel_size=15, padding=7),
            nn.GELU(),
            nn.InstanceNorm1d(adapter_hidden),
            nn.Conv1d(adapter_hidden, adapter_hidden, kernel_size=kernel, padding=kernel // 2, groups=adapter_hidden),
            nn.GELU(),
            nn.Conv1d(adapter_hidden, token_hidden, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm1d(token_hidden),
        )
        self.score_head = nn.Sequential(
            nn.Conv1d(token_hidden + 3, adapter_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(adapter_hidden, 1, kernel_size=1),
        )
        self.phase_head = nn.Sequential(
            nn.Conv1d(token_hidden * 2 + 5, adapter_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(adapter_hidden, 1, kernel_size=1),
        )
        self.token_fusion = nn.Sequential(
            nn.Conv1d(token_hidden * 2 + 9, token_hidden, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm1d(token_hidden),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _shift_with_pad(x: torch.Tensor, offset: int) -> torch.Tensor:
        if offset <= 0:
            return x
        return F.pad(x[:, :, offset:], (0, offset))

    def forward(self, x: torch.Tensor, num_symbols: int) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + self.eps)
        x_norm = x / rms
        feat = self.stem(x_norm)

        cand_avg_list = []
        cand_max_list = []
        cand_unit_iq_list = []
        cand_amp_list = []
        cand_scores = []
        offset_values = []

        for offset in self.offsets:
            shifted_feat = self._shift_with_pad(feat, offset)
            shifted_iq = self._shift_with_pad(x_norm, offset)

            cand_avg = F.adaptive_avg_pool1d(shifted_feat, num_symbols)
            cand_max = F.adaptive_max_pool1d(shifted_feat, num_symbols)
            cand_iq = F.adaptive_avg_pool1d(shifted_iq, num_symbols)
            cand_amp = torch.sqrt(torch.sum(cand_iq ** 2, dim=1, keepdim=True) + self.eps)
            cand_unit_iq = cand_iq / cand_amp

            score_in = torch.cat([cand_avg, cand_unit_iq, cand_amp], dim=1)
            cand_score = self.score_head(score_in)

            cand_avg_list.append(cand_avg)
            cand_max_list.append(cand_max)
            cand_unit_iq_list.append(cand_unit_iq)
            cand_amp_list.append(cand_amp)
            cand_scores.append(cand_score)
            offset_values.append(float(offset) / float(max(1, self.approx_symbol_span)))

        score_stack = torch.stack(cand_scores, dim=1).squeeze(2)  # (B, O, S)
        weights = torch.softmax(score_stack, dim=1)

        def _weighted_sum(candidates):
            stacked = torch.stack(candidates, dim=1)  # (B, O, C, S)
            return torch.sum(stacked * weights.unsqueeze(2), dim=1)

        pooled_avg = _weighted_sum(cand_avg_list)
        pooled_max = _weighted_sum(cand_max_list)
        pooled_unit_iq = _weighted_sum(cand_unit_iq_list)
        pooled_amp = _weighted_sum(cand_amp_list)

        offset_tensor = x.new_tensor(offset_values).view(1, -1, 1)
        offset_mean = torch.sum(weights * offset_tensor, dim=1, keepdim=True)
        offset_conf = weights.max(dim=1, keepdim=True).values

        phase_in = torch.cat([pooled_avg, pooled_max, pooled_unit_iq, pooled_amp, offset_mean, offset_conf], dim=1)
        phase = torch.tanh(self.phase_head(phase_in)) * torch.pi
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        i_comp = pooled_unit_iq[:, 0:1, :]
        q_comp = pooled_unit_iq[:, 1:2, :]
        i_rot = i_comp * cos_phase + q_comp * sin_phase
        q_rot = q_comp * cos_phase - i_comp * sin_phase
        rotated_iq = torch.cat([i_rot, q_rot], dim=1)

        delta = rotated_iq[:, :, 1:] - rotated_iq[:, :, :-1]
        delta = F.pad(delta, (1, 0))
        phase_feat = torch.cat([cos_phase, sin_phase], dim=1)

        tokens = torch.cat(
            [pooled_avg, pooled_max, rotated_iq, pooled_amp, delta, offset_mean, offset_conf, phase_feat],
            dim=1,
        )
        tokens = self.token_fusion(tokens)
        return tokens.transpose(1, 2)


class ReceiverAwareSoftDemodHeadV3(nn.Module):
    """Stronger demapper head with timing-hypothesis fusion and phase-aware context."""

    def __init__(
        self,
        input_size: int,
        num_bits: int = 615,
        bits_per_symbol: int = 3,
        adapter_hidden: int = 128,
        token_hidden: int = 160,
        rnn_hidden: int = 128,
        context_layers: int = 2,
        num_offset_hypotheses: int = 4,
        attn_heads: int = 4,
        dropout: float = 0.2,
        symbol_logit_scale: float = 14.0,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.bits_per_symbol = bits_per_symbol
        self.num_symbol_classes = 1 << bits_per_symbol
        default_symbols = max(1, num_bits // bits_per_symbol)
        approx_symbol_span = max(1, int(round(float(input_size) / float(default_symbols))))

        self.adapter = OffsetPhaseAwareSymbolAdapter(
            token_hidden=token_hidden,
            adapter_hidden=adapter_hidden,
            approx_symbol_span=approx_symbol_span,
            num_offset_hypotheses=num_offset_hypotheses,
            dropout=dropout,
        )
        self.pre_context = nn.Sequential(
            nn.Conv1d(token_hidden, token_hidden, kernel_size=5, padding=2, groups=max(1, token_hidden // 16)),
            nn.GELU(),
            nn.Conv1d(token_hidden, token_hidden, kernel_size=1),
            nn.GELU(),
            nn.InstanceNorm1d(token_hidden),
        )
        self.context = nn.GRU(
            input_size=token_hidden,
            hidden_size=rnn_hidden,
            num_layers=context_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if context_layers > 1 else 0.0,
        )
        ctx_dim = rnn_hidden * 2
        self.self_attn = nn.MultiheadAttention(
            embed_dim=ctx_dim,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(ctx_dim)
        self.ffn_norm = nn.LayerNorm(ctx_dim)
        self.ffn = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_dim * 2, ctx_dim),
        )
        self.bit_head = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ctx_dim, bits_per_symbol),
        )
        self.symbol_head = PrototypeSymbolClassifier(
            in_dim=ctx_dim,
            num_classes=self.num_symbol_classes,
            logit_scale=symbol_logit_scale,
        )
        self.fallback_bit_head = nn.Sequential(
            nn.Conv1d(2, adapter_hidden, kernel_size=15, padding=7),
            nn.GELU(),
            nn.InstanceNorm1d(adapter_hidden),
            nn.Conv1d(adapter_hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, num_bits: int = None):
        nb = num_bits if num_bits is not None else self.num_bits
        if nb % self.bits_per_symbol != 0:
            bit_logits = self.fallback_bit_head(F.adaptive_avg_pool1d(x, nb)).squeeze(1)
            return bit_logits, None

        num_symbols = max(1, nb // self.bits_per_symbol)
        tokens = self.adapter(x, num_symbols=num_symbols)              # (B, S, H)
        tokens = self.pre_context(tokens.transpose(1, 2)).transpose(1, 2)
        context, _ = self.context(tokens)
        attn_out, _ = self.self_attn(context, context, context, need_weights=False)
        context = self.attn_norm(context + attn_out)
        context = self.ffn_norm(context + self.ffn(context))

        bit_logits = self.bit_head(context).reshape(context.shape[0], -1)
        symbol_logits = self.symbol_head(context)
        return bit_logits[:, :nb], symbol_logits


class IQUBiMamba1D_SoftDemodV3(IQUBiMamba1D_SoftDemod):
    """Stronger receiver-structured SoftDemod variant.

    This version adds:
      - multi-offset symbol timing hypotheses
      - explicit phase correction inside the adapter
      - symbol-level self-attention after recurrent context modeling
    """

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
        num_bits: int = 615,
        demod_bits_per_symbol: int = 3,
        demod_hidden: int = 64,
        demod_rnn_hidden: int = 128,
        demod_dropout: float = 0.2,
        detach_demod: bool = False,
        demod_adapter_hidden: int = 128,
        demod_symbol_hidden: int = 160,
        demod_context_layers: int = 2,
        demod_symbol_logit_scale: float = 14.0,
        demod_timing_offsets: int = 4,
        demod_attn_heads: int = 4,
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {'inplace': True},
        deep_supervision: bool = False,
    ):
        super().__init__(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            num_bits=num_bits,
            demod_bits_per_symbol=demod_bits_per_symbol,
            demod_hidden=demod_hidden,
            demod_rnn_hidden=demod_rnn_hidden,
            demod_dropout=demod_dropout,
            detach_demod=detach_demod,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.demod_head = ReceiverAwareSoftDemodHeadV3(
            input_size=input_size,
            num_bits=num_bits,
            bits_per_symbol=demod_bits_per_symbol,
            adapter_hidden=demod_adapter_hidden,
            token_hidden=demod_symbol_hidden,
            rnn_hidden=demod_rnn_hidden,
            context_layers=demod_context_layers,
            num_offset_hypotheses=demod_timing_offsets,
            attn_heads=demod_attn_heads,
            dropout=demod_dropout,
            symbol_logit_scale=demod_symbol_logit_scale,
        )
