"""
NES2Net: Number Estimation and Signal Separation Network
for Single-Channel Blind Signal Separation.

Reimplemented from:
  "NES2Net: A Number Estimation and Signal Separation Network
   for Single-Channel Blind Signal Separation"

Key ideas:
  1. NEM (Number Estimation Module): 1D-CNN classifier -> predicts # of sources m.
  2. SSM (Signal Separation Module): a shared 1D U-Net is applied *iteratively*
     (m-1 times), each pass extracting one source from the current residual.
     The final residual is taken as the last source.
  3. Loss: Greedy cosine-similarity matching + MSE per extraction step.

Integration note:
  In the IQUMamba1D pipeline the model receives (B, C_in, L) and returns
  (B, C_out, L) where C_out = num_sources * C_in (e.g. 4 for 2 IQ sources).
  NES2Net keeps the *same* interface when `num_sources` is fixed at train time,
  so it drops in as a replacement model without any changes to the training loop.
  The NEM branch is trained separately or jointly (see below).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock1D(nn.Module):
    """Conv1d -> BN -> ReLU  (repeated twice as in NES2Net Fig.)"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownSampleBlock(nn.Module):
    """Strided convolution for downsampling (stride=2, kernel=3)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# NEM - Number Estimation Module
# ---------------------------------------------------------------------------

class NEM(nn.Module):
    """
    1D-CNN classifier that predicts the number of sources.

    Architecture (paper Table I, repeated x5):
        ConvBlock(k=3) -> DownSample(stride=2) -> ... -> Flatten -> Linear -> M

    Channel progression: in_channels -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048
    (5 down-samples with channel doubling each time).

    We use AdaptiveAvgPool1d(1) to stay input-length-agnostic.
    """

    def __init__(
        self,
        input_channels: int = 2,
        max_sources: int = 5,
        base_channels: int = 64,
        num_blocks: int = 5,
        use_global_pool: bool = True,
    ):
        super().__init__()
        self.max_sources = max_sources

        layers = []
        in_ch = input_channels
        ch = base_channels
        for i in range(num_blocks):
            out_ch = ch * (2 ** i)
            layers.append(ConvBlock1D(in_ch, out_ch))
            layers.append(DownSampleBlock(out_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.last_channels = in_ch

        self.use_global_pool = use_global_pool
        if use_global_pool:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(in_ch, max_sources)
        else:
            self.pool = None
            self.classifier = None  # built lazily

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L) mixture waveform
        Returns:
            logits: (B, max_sources) classification logits
        """
        feat = self.features(x)                      # (B, C, L')
        if self.use_global_pool:
            feat = self.pool(feat).squeeze(-1)        # (B, C)
        else:
            feat = feat.flatten(1)                    # (B, C*L')
            if self.classifier is None:
                self.classifier = nn.Linear(feat.shape[1], self.max_sources).to(feat.device)
        return self.classifier(feat)                  # (B, M)


# ---------------------------------------------------------------------------
# 1D U-Net for SSM
# ---------------------------------------------------------------------------

class UNetEncoderBlock(nn.Module):
    """Two conv layers followed by stride-2 downsampling."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feat = self.conv(x)     # skip features (before downsampling)
        down = self.pool(feat)  # downsampled output
        return feat, down


class UNetDecoderBlock(nn.Module):
    """Upsample + concatenate skip + two conv layers."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch + skip_ch, out_ch, kernel_size, padding=pad),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential size mismatch from non-power-of-2 lengths
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetBottleneck(nn.Module):
    def __init__(self, ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size, padding=pad),
            nn.InstanceNorm1d(ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ch, ch, kernel_size, padding=pad),
            nn.InstanceNorm1d(ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SeparationUNet1D(nn.Module):
    """
    Lightweight 1D U-Net used as the separation backbone in SSM.

    Each forward pass takes a *residual* waveform (B, C_in, L) and returns
    a single estimated source (B, C_in, L).

    Example with features=[32, 64, 128, 256]:
        Encoder:  C_in->32 |32->64 |64->128 |128->256
        Bottleneck: 256->256
        Decoder:  256+128->128 |128+64->64 |64+32->32
        Output:   32->C_in (1x1 conv)
    """

    def __init__(
        self,
        input_channels: int = 2,
        features: List[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256]
        self.depth = len(features)
        self.features = features

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = input_channels
        for f in features:
            self.encoders.append(UNetEncoderBlock(in_ch, f, kernel_size))
            in_ch = f

        # Bottleneck
        self.bottleneck = UNetBottleneck(features[-1], kernel_size)

        # Decoder: goes from deepest to shallowest
        # decoder[0]: features[-1] up, skip from features[-2]  -> features[-2]
        # decoder[1]: features[-2] up, skip from features[-3]  -> features[-3]
        # ...
        # decoder[-1]: features[1] up, skip from features[0]   -> features[0]
        self.decoders = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            self.decoders.append(
                UNetDecoderBlock(features[i], features[i - 1], features[i - 1], kernel_size)
            )

        # Output projection back to input_channels
        self.out_conv = nn.Conv1d(features[0], input_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L) current residual signal
        Returns:
            estimated source: (B, C_in, L)
        """
        # Encoder: collect skip features (before pooling) at each level
        skips = []
        h = x
        for enc in self.encoders:
            feat, h = enc(h)  # feat = pre-pool, h = post-pool
            skips.append(feat)
        # skips = [feat_0(32ch), feat_1(64ch), feat_2(128ch), feat_3(256ch)]
        # h is the bottleneck input (256ch, smallest spatial dim)

        # Bottleneck
        h = self.bottleneck(h)  # (B, 256, L/16)

        # Decoder: upsample and fuse with corresponding encoder skip
        # decoder[0] uses skip from skips[-2] (second-deepest encoder = 128ch)
        # decoder[1] uses skip from skips[-3] (= 64ch)
        # decoder[2] uses skip from skips[-4] (= 32ch)
        for i, dec in enumerate(self.decoders):
            skip = skips[self.depth - 2 - i]  # -2, -3, ... counting from end
            h = dec(h, skip)

        # Ensure output matches input length
        if h.shape[-1] != x.shape[-1]:
            h = F.interpolate(h, size=x.shape[-1], mode='linear', align_corners=False)

        return self.out_conv(h)


# ---------------------------------------------------------------------------
# SSM - Signal Separation Module (iterative)
# ---------------------------------------------------------------------------

class SSM(nn.Module):
    """
    Iterative signal separation using a shared U-Net.

    For m sources:
      residual = mixture
      for j in 1 .. m-1:
          s_hat_j = UNet(residual)
          residual = residual - s_hat_j
      s_hat_m = residual

    The same UNet is reused across iterations (weight sharing).
    """

    def __init__(
        self,
        input_channels: int = 2,
        unet_features: List[int] = None,
        unet_kernel_size: int = 3,
    ):
        super().__init__()
        self.unet = SeparationUNet1D(
            input_channels=input_channels,
            features=unet_features,
            kernel_size=unet_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_sources: int = 2,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, C_in, L) mixture
            num_sources:  number of sources to extract

        Returns:
            sources: (B, num_sources * C_in, L) - channels interleaved per source
        """
        B, C, L = x.shape
        estimated = []
        residual = x

        for j in range(num_sources - 1):
            s_hat = self.unet(residual)
            estimated.append(s_hat)
            residual = residual - s_hat

        # Last source is the residual
        estimated.append(residual)

        # Stack: list of (B, C_in, L) -> (B, num_sources * C_in, L)
        out = torch.cat(estimated, dim=1)
        return out


# ---------------------------------------------------------------------------
# NES2Net (complete model)
# ---------------------------------------------------------------------------

class NES2Net(nn.Module):
    """
    NES2Net: NEM + SSM

    Training modes:
      - 'separation' (default): Only SSM is used; num_sources comes from the
        data config (= pipeline's existing fixed-source setup).  NEM is not
        exercised and no NEM loss is computed.  This mode is **drop-in
        compatible** with IQUMamba1D's training loop.
      - 'joint': Both NEM and SSM are used; NEM predicts num_sources and SSM
        uses that prediction.  Requires a separate training procedure.
      - 'nem_only': Only NEM forward pass for classification pre-training.

    In 'separation' mode the interface is identical to IQUMamba1D:
        forward(x)  ->  (B, num_classes, L)
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 4,
        # NEM
        max_sources: int = 5,
        nem_base_channels: int = 64,
        nem_num_blocks: int = 5,
        # SSM / U-Net
        unet_features: List[int] = None,
        unet_kernel_size: int = 3,
        # Mode
        mode: str = 'separation',
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_sources = num_classes // input_channels
        self.mode = mode
        self.max_sources = max_sources

        # NEM (always built, may not be used at train time)
        self.nem = NEM(
            input_channels=input_channels,
            max_sources=max_sources,
            base_channels=nem_base_channels,
            num_blocks=nem_num_blocks,
        )

        # SSM
        self.ssm = SSM(
            input_channels=input_channels,
            unet_features=unet_features,
            unet_kernel_size=unet_kernel_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_sources: int = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L) mixture signal
            num_sources: override source count (None -> use self.num_sources)

        Returns:
            If mode == 'separation' or 'joint':
                (B, num_sources * C_in, L) separated sources
            If mode == 'nem_only':
                (B, max_sources) NEM logits
        """
        if self.mode == 'nem_only':
            return self.nem(x)

        n = num_sources if num_sources is not None else self.num_sources

        if self.mode == 'joint':
            # Use NEM to predict number of sources
            nem_logits = self.nem(x)                    # (B, M)
            predicted_n = nem_logits.argmax(dim=-1) + 1  # 1-indexed
            # During training we still use ground-truth n for SSM;
            # switch to predicted at inference time
            if not self.training:
                n = int(predicted_n[0].item())

        return self.ssm(x, num_sources=n)

    def get_nem_parameters(self):
        """Return NEM parameters for separate optimizer."""
        return self.nem.parameters()

    def get_ssm_parameters(self):
        """Return SSM parameters for separate optimizer."""
        return self.ssm.parameters()


# ---------------------------------------------------------------------------
# NES2Net greedy-matching loss (paper Eq. 3-5)
# ---------------------------------------------------------------------------

class NES2NetGreedyLoss(nn.Module):
    """
    Greedy cosine-similarity matching + MSE loss for iterative separation.

    For each extraction step j:
      1. Compute cosine similarity between s_hat_j and all *unmatched* targets.
      2. Select the best-matching target (greedy).
      3. Accumulate MSE(s_hat_j, matched_target).

    This loss wraps around the standard PIT pipeline - when used as a
    drop-in criterion it receives the *stacked* model output and computes
    the greedy assignment internally.
    """

    def __init__(self, num_sources: int = 2, input_channels: int = 2):
        super().__init__()
        self.num_sources = num_sources
        self.input_channels = input_channels

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            outputs: (B, num_sources * C_in, L) - model output
            targets: (B, num_sources * C_in, L) - ground truth
        Returns:
            scalar loss
        """
        B, _, L = outputs.shape
        C = self.input_channels
        K = self.num_sources

        # Reshape to (B, K, C, L)
        pred = outputs.view(B, K, C, L)
        tgt = targets.view(B, K, C, L)

        # Flatten spatial dims for cosine similarity: (B, K, C*L)
        pred_flat = pred.reshape(B, K, -1)
        tgt_flat = tgt.reshape(B, K, -1)

        total_loss = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

        for b in range(B):
            available = list(range(K))
            sample_loss = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

            for j in range(K):
                if len(available) == 1:
                    k_star = available[0]
                else:
                    cos_sims = []
                    for k in available:
                        sim = F.cosine_similarity(
                            pred_flat[b, j].unsqueeze(0),
                            tgt_flat[b, k].unsqueeze(0),
                            dim=-1,
                        )
                        cos_sims.append(sim)
                    cos_sims = torch.stack(cos_sims)
                    best_idx = cos_sims.argmax().item()
                    k_star = available[best_idx]

                sample_loss = sample_loss + F.mse_loss(pred[b, j], tgt[b, k_star])
                available.remove(k_star)

            total_loss = total_loss + sample_loss / K

        return total_loss / B


# ---------------------------------------------------------------------------
# Vectorised greedy loss (much faster - recommended for training)
# ---------------------------------------------------------------------------

class NES2NetGreedyLossVectorized(nn.Module):
    """
    Vectorised implementation of the greedy cosine-similarity matching + MSE
    loss.  Processes the full batch in parallel using a greedy mask.
    """

    def __init__(self, num_sources: int = 2, input_channels: int = 2):
        super().__init__()
        self.num_sources = num_sources
        self.input_channels = input_channels

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, _, L = outputs.shape
        C = self.input_channels
        K = self.num_sources

        # (B, K, C, L) -> (B, K, C*L)
        pred = outputs.view(B, K, C, L).reshape(B, K, -1)   # (B, K, D)
        tgt = targets.view(B, K, C, L).reshape(B, K, -1)    # (B, K, D)

        # Full cosine similarity matrix: (B, K_pred, K_tgt)
        pred_norm = F.normalize(pred, dim=-1)
        tgt_norm = F.normalize(tgt, dim=-1)
        sim_matrix = torch.bmm(pred_norm, tgt_norm.transpose(1, 2))  # (B, K, K)

        # Greedy matching: iterate over prediction slots
        mask_used = torch.zeros(B, K, dtype=torch.bool, device=outputs.device)
        assignment = torch.zeros(B, K, dtype=torch.long, device=outputs.device)

        for j in range(K):
            sim_j = sim_matrix[:, j, :].clone()              # (B, K_tgt)
            sim_j[mask_used] = -float('inf')
            best_k = sim_j.argmax(dim=-1)                    # (B,)
            assignment[:, j] = best_k
            mask_used.scatter_(1, best_k.unsqueeze(1), True)

        # Gather matched targets  (B, K, D)
        idx = assignment.unsqueeze(-1).expand(-1, -1, C * L)  # (B, K, D)
        tgt_matched = tgt.gather(1, idx)                      # (B, K, D)

        # MSE per source, averaged
        mse = (pred - tgt_matched).pow(2).mean(dim=-1)        # (B, K)
        return mse.mean()
