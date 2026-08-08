"""IQUBiMamba1D_AMR — Joint Blind Source Separation + Automatic Modulation
Recognition (AMR) network.

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
    │  Shared AMR Classifier Head │  ← lightweight Conv1D + BiMamba + FC
    │  applied to each source     │
    └─────────┬───────────────────┘
              │
              ▼
    modulation logits: K tensors of (B, num_mod_classes)

Training modes
==============
- ``mode='sep_only'``  : forward returns separation output only (standard BSS).
- ``mode='cls_only'``  : encoder/decoder frozen, only classifier trains.
- ``mode='joint'``     : end-to-end, both outputs returned.
- ``detach_cls=True``  : classifier receives *detached* waveforms so its
                         gradients do NOT flow back into the separator
                         (safe baseline / ablation).

Loss design
===========
The companion ``pit_separation_amr_loss`` (in util/loss.py) uses **PIT to
select the best permutation based on the separation loss**, then evaluates
the classification CrossEntropy under that *same* permutation.  This
guarantees label-output alignment even when sources share a modulation type.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.amp import autocast

from mamba_ssm import Mamba

# Re-use all building blocks from existing models
from models.IQUBiMamba1D import (
    BiMambaLayer,
    ResidualBiMambaEncoder,
)
from models.IQUMamba1D import (
    UNetResDecoder,
    BasicResBlock,
    SkipConnectionProcessor,
)
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list

from typing import Union, Type, List, Tuple, Optional
from torch.nn.modules.conv import _ConvNd

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  AMRClassifierHead — lightweight modulation classifier
# ============================================================================

class AMRClassifierHead(nn.Module):
    """Lightweight classifier for automatic modulation recognition.

    Takes a single separated IQ source ``(B, 2, L)`` and outputs class logits
    ``(B, num_mod_classes)``.

    Architecture:
        1. Multi-scale Conv1D feature extractor (3 layers, residual)
        2. BiMamba layer for global temporal context
        3. Adaptive average pooling → fixed-length vector
        4. Two FC layers with dropout → logits
    """

    def __init__(
        self,
        num_mod_classes: int = 11,
        input_channels: int = 2,
        hidden_channels: int = 64,
        mamba_dim: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        fc_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_mod_classes = num_mod_classes

        # ---------- Feature extractor ----------
        self.feature_extractor = nn.Sequential(
            # Block 1: input (B, 2, L) → (B, hidden, L)
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),

            # Block 2: (B, hidden, L) → (B, hidden, L)
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),

            # Block 3: deeper features
            nn.Conv1d(hidden_channels, mamba_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(mamba_dim),
            nn.GELU(),
        )

        # ---------- BiMamba for global sequence context ----------
        self.norm = nn.LayerNorm(mamba_dim)
        self.mamba_fwd = Mamba(
            d_model=mamba_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=mamba_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_proj = nn.Linear(mamba_dim * 2, mamba_dim)

        # ---------- Classifier head ----------
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(mamba_dim, mamba_dim),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(mamba_dim, num_mod_classes),
        )

    @autocast('cuda', enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L) — single IQ source (I and Q channels)
        Returns:
            logits: (B, num_mod_classes)
        """
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()

        # Conv feature extraction: (B, 2, L) → (B, mamba_dim, L)
        h = self.feature_extractor(x)

        # BiMamba: (B, mamba_dim, L) → (B, L, mamba_dim) for sequence model
        h_seq = h.transpose(1, 2)  # (B, L, D)
        h_norm = self.norm(h_seq)
        h_fwd = self.mamba_fwd(h_norm)
        h_bwd = self.mamba_bwd(h_norm.flip(dims=[1])).flip(dims=[1])
        h_bi = self.mamba_proj(torch.cat([h_fwd, h_bwd], dim=-1))  # (B, L, D)
        h_bi = h_bi + h_seq  # residual

        # Pool → classify
        h_bi = h_bi.transpose(1, 2)  # (B, D, L)
        h_pooled = self.pool(h_bi).squeeze(-1)  # (B, D)
        logits = self.classifier(h_pooled)  # (B, num_mod_classes)

        return logits


# ============================================================================
#  IQUBiMamba1D_AMR — top-level joint model
# ============================================================================

class IQUBiMamba1D_AMR(nn.Module):
    """Joint Blind Source Separation + Automatic Modulation Recognition.

    Wraps a full IQUBiMamba1D separation backbone and attaches a shared
    AMR classifier head to each separated source.

    Args:
        input_size:        signal length (e.g. 4096)
        input_channels:    IQ channels (always 2)
        n_stages:          UNet encoder depth
        features_per_stage: channel widths per encoder stage
        conv_op:           1-D conv class
        kernel_sizes:      per-stage kernel sizes
        strides:           per-stage strides
        n_conv_per_stage:  residual blocks per encoder stage
        num_classes:       separation output channels (2*K for K sources)
        n_conv_per_stage_decoder: residual blocks per decoder stage
        num_mod_classes:   number of modulation types to classify
        cls_hidden:        AMR classifier hidden channel width
        cls_mamba_dim:     AMR classifier BiMamba dimension
        cls_dropout:       AMR classifier FC dropout
        detach_cls:        if True, stop gradients from classifier to separator
        deep_supervision:  UNet deep supervision flag
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
        # ----- AMR-specific -----
        num_mod_classes: int = 11,
        cls_hidden: int = 64,
        cls_mamba_dim: int = 64,
        cls_dropout: float = 0.3,
        detach_cls: bool = False,
        # ----- standard -----
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}

        self.num_classes = num_classes          # 2*K (separation output channels)
        self.num_sources = num_classes // 2     # K
        self.num_mod_classes = num_mod_classes
        self.detach_cls = detach_cls

        # ---- Separation backbone (identical to IQUBiMamba1D) ----
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

        # ---- AMR Classifier Head (shared across all sources) ----
        self.amr_head = AMRClassifierHead(
            num_mod_classes=num_mod_classes,
            input_channels=input_channels,    # 2 (I, Q)
            hidden_channels=cls_hidden,
            mamba_dim=cls_mamba_dim,
            fc_dropout=cls_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'joint',
    ):
        """
        Args:
            x:    (B, 2, L) mixed IQ signal
            mode: 'sep_only'  → returns only separation output
                  'cls_only'  → returns only classification logits
                                (encoder/decoder run but gradients blocked)
                  'joint'     → returns (sep_output, cls_logits_list)

        Returns:
            Depending on mode:
            - sep_only:  (B, 2*K, L)  separated waveforms
            - cls_only:  list of K tensors, each (B, num_mod_classes)
            - joint:     tuple(sep_output, cls_logits_list)
                         sep_output: (B, 2*K, L)
                         cls_logits_list: list of K tensors (B, num_mod_classes)
        """
        # ---- 1. Separation ----
        skips = self.encoder(x)
        sep_output = self.decoder(skips)  # (B, 2*K, L)

        if mode == 'sep_only':
            return sep_output

        # ---- 2. Split into per-source IQ tensors ----
        K = self.num_sources
        # sep_output is (B, 2*K, L), split into K sources each (B, 2, L)
        sources = []
        for k in range(K):
            src_k = sep_output[:, 2*k : 2*k+2, :]  # (B, 2, L)
            if self.detach_cls or mode == 'cls_only':
                src_k = src_k.detach()
            sources.append(src_k)

        # ---- 3. Classify each source ----
        cls_logits = [self.amr_head(src) for src in sources]  # K × (B, C)

        if mode == 'cls_only':
            return cls_logits

        # mode == 'joint'
        return sep_output, cls_logits

    # ------------------------------------------------------------------
    # Convenience methods for staged training
    # ------------------------------------------------------------------
    def freeze_separator(self):
        """Freeze encoder + decoder (for cls_only training stage)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def unfreeze_separator(self):
        """Unfreeze encoder + decoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        """Freeze AMR head (for sep_only training stage)."""
        for param in self.amr_head.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        """Unfreeze AMR head."""
        for param in self.amr_head.parameters():
            param.requires_grad = True

    def load_separator_weights(self, state_dict_or_path, strict: bool = True):
        """Load pretrained IQUBiMamba1D weights into encoder + decoder.

        This allows bootstrapping from a pretrained separation-only model.
        Keys are matched by prefix ('encoder.' and 'decoder.').
        """
        from pathlib import Path
        if isinstance(state_dict_or_path, (str, Path)):
            ckpt = torch.load(str(state_dict_or_path), map_location='cpu')
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt
        else:
            state_dict = state_dict_or_path

        # Filter only encoder/decoder keys
        sep_keys = {k: v for k, v in state_dict.items()
                    if k.startswith('encoder.') or k.startswith('decoder.')}
        missing, unexpected = self.load_state_dict(sep_keys, strict=False)
        
        # Report
        loaded = len(sep_keys) - len(unexpected)
        print(f"[AMR] Loaded {loaded} separator params "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        return missing, unexpected
