import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type, Union
import random


# ============================================================
# Helper functions
# ============================================================

def _num_groups(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


def make_norm(channels: int):
    return nn.GroupNorm(_num_groups(channels), channels)


def rms(x: torch.Tensor, eps: float = 1e-8):
    return torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + eps)


def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [B, 2, L] = ar, ai
    b: [B, 2, L] = br, bi
    return a * b
    """
    ar, ai = a[:, 0:1], a[:, 1:2]
    br, bi = b[:, 0:1], b[:, 1:2]

    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return torch.cat([real, imag], dim=1)


def iq_hints(x: torch.Tensor) -> torch.Tensor:
    """
    Construct lightweight communication hints from IQ.

    x: [B, 2, L]
    return: [B, 6, L]
        I
        Q
        amplitude
        normalized I
        normalized Q
        phase increment proxy
    """
    i = x[:, 0:1, :]
    q = x[:, 1:2, :]

    amp = torch.sqrt(i ** 2 + q ** 2 + 1e-8)
    i_norm = i / (amp + 1e-6)
    q_norm = q / (amp + 1e-6)

    # phase increment proxy:
    # z[n] * conj(z[n-1])
    i_prev = F.pad(i[..., :-1], (1, 0))
    q_prev = F.pad(q[..., :-1], (1, 0))

    cross_r = i * i_prev + q * q_prev
    cross_i = q * i_prev - i * q_prev
    phase_inc = cross_i / (torch.sqrt(cross_r ** 2 + cross_i ** 2 + 1e-8))

    return torch.cat([i, q, amp, i_norm, q_norm, phase_inc], dim=1)


def init_small_conv(conv: nn.Conv1d, std: float = 1e-2):
    nn.init.normal_(conv.weight, mean=0.0, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


# ============================================================
# Basic blocks
# ============================================================

class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=True),
            make_norm(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            make_norm(hidden),
            nn.SiLU(inplace=True),

            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=5,
                padding=2 * dilation,
                dilation=dilation,
                groups=hidden,
                bias=True,
            ),
            make_norm(hidden),
            nn.SiLU(inplace=True),

            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
        )

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        return x + self.scale * self.net(x)


# ============================================================
# Branch 1: local waveform correction
# ============================================================

class LocalIQResidualBranch(nn.Module):
    """
    Local convolutional residual branch.
    Good at correcting local IQ waveform distortion.
    """
    def __init__(self, channels: int = 32):
        super().__init__()
        self.in_proj = ConvGNAct(6, channels, kernel_size=7)

        self.blocks = nn.Sequential(
            ResidualTCNBlock(channels, dilation=1),
            ResidualTCNBlock(channels, dilation=1),
        )

        self.out = nn.Conv1d(channels, 2, kernel_size=3, padding=1, bias=True)
        init_small_conv(self.out, std=1e-2)

    def forward(self, hints):
        h = self.in_proj(hints)
        h = self.blocks(h)
        return self.out(h)


# ============================================================
# Branch 2: dilated temporal correction
# ============================================================

class DilatedIQResidualBranch(nn.Module):
    """
    Multi-dilation branch.
    Good at slow phase drift, CFO-like trends, and longer context.
    """
    def __init__(self, channels: int = 32, dilations: List[int] = [1, 2, 4, 8, 16]):
        super().__init__()
        self.in_proj = ConvGNAct(6, channels, kernel_size=3)

        self.blocks = nn.Sequential(*[
            ResidualTCNBlock(channels, dilation=d) for d in dilations
        ])

        self.out = nn.Conv1d(channels, 2, kernel_size=3, padding=1, bias=True)
        init_small_conv(self.out, std=1e-2)

    def forward(self, hints):
        h = self.in_proj(hints)
        h = self.blocks(h)
        return self.out(h)


# ============================================================
# Branch 3: complex mask residual
# ============================================================

class ComplexMaskResidualBranch(nn.Module):
    """
    Predicts a bounded complex residual mask M and returns delta = M * x.
    This is phase-reference preserving.
    """
    def __init__(self, channels: int = 32, mask_scale: float = 0.5):
        super().__init__()
        self.mask_scale = mask_scale

        self.net = nn.Sequential(
            ConvGNAct(6, channels, kernel_size=5),
            ResidualTCNBlock(channels, dilation=1),
            ResidualTCNBlock(channels, dilation=2),
            nn.Conv1d(channels, 2, kernel_size=3, padding=1, bias=True),
        )

        init_small_conv(self.net[-1], std=1e-2)

    def forward(self, x, hints):
        mask = torch.tanh(self.net(hints)) * self.mask_scale
        return complex_mul(mask, x)


# ============================================================
# Branch 4: amplitude/phase-gated residual
# ============================================================

class AmpPhaseGatedResidualBranch(nn.Module):
    """
    Uses amplitude and phase-increment hints to generate a temporal gate and residual.
    Good for bursty or locally unreliable regions.
    """
    def __init__(self, channels: int = 32):
        super().__init__()

        self.feat = nn.Sequential(
            ConvGNAct(6, channels, kernel_size=7),
            ResidualTCNBlock(channels, dilation=2),
            ResidualTCNBlock(channels, dilation=4),
        )

        self.delta = nn.Conv1d(channels, 2, kernel_size=3, padding=1, bias=True)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        init_small_conv(self.delta, std=1e-2)

    def forward(self, hints):
        h = self.feat(hints)
        return self.delta(h) * self.gate(h)


# ============================================================
# Strong adapter
# ============================================================

class StrongResidualPriorAdapter1D(nn.Module):
    """
    Strong front-end residual prior adapter.

    Input:
        x: [B, 2, L]

    Output:
        x_prior = x + scale * temporal_gate * weighted_delta

    Key design:
        - no identity expert in router
        - no hard physical routing
        - no detached evidence
        - all branches learn end-to-end
        - residual normalized to input RMS
    """

    def __init__(
        self,
        channels: int = 32,
        max_scale: float = 0.5,
        scale_init: float = 0.0,
        router_hidden: int = 64,
        temperature: float = 1.0,
        normalize_delta: bool = True,
    ):
        super().__init__()

        self.temperature = temperature
        self.normalize_delta = normalize_delta

        self.branch_local = LocalIQResidualBranch(channels=channels)
        self.branch_dilated = DilatedIQResidualBranch(channels=channels)
        self.branch_mask = ComplexMaskResidualBranch(channels=channels, mask_scale=0.5)
        self.branch_gate = AmpPhaseGatedResidualBranch(channels=channels)

        self.num_branches = 4

        # Shared feature extractor for routing and temporal gate
        self.shared = nn.Sequential(
            ConvGNAct(6, channels, kernel_size=7),
            ResidualTCNBlock(channels, dilation=1),
            ResidualTCNBlock(channels, dilation=4),
        )

        self.router = nn.Sequential(
            nn.Linear(channels * 2 + 4, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, self.num_branches),
        )

        self.temporal_gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=True),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.max_scale = max_scale
        self.raw_scale = nn.Parameter(torch.tensor(float(scale_init)))

    def get_scale(self):
        return self.max_scale * torch.sigmoid(self.raw_scale)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        hints = iq_hints(x)

        # Branch deltas
        d_local = self.branch_local(hints)
        d_dilated = self.branch_dilated(hints)
        d_mask = self.branch_mask(x, hints)
        d_gate = self.branch_gate(hints)

        deltas = torch.stack(
            [d_local, d_dilated, d_mask, d_gate],
            dim=1,
        )  # [B, 4, 2, L]

        # Shared features
        feat = self.shared(hints)  # [B, C, L]

        feat_mean = feat.mean(dim=-1)
        feat_std = feat.std(dim=-1)

        # Simple physical summary, differentiable but only used for routing
        amp = hints[:, 2:3, :]
        phase_inc = hints[:, 5:6, :]

        amp_mean = amp.mean(dim=-1)
        amp_std = amp.std(dim=-1)
        phase_mean = phase_inc.mean(dim=-1)
        phase_std = phase_inc.std(dim=-1)

        route_desc = torch.cat(
            [feat_mean, feat_std, amp_mean, amp_std, phase_mean, phase_std],
            dim=1,
        )

        logits = self.router(route_desc) / self.temperature
        weights = torch.softmax(logits, dim=-1)  # [B, 4]

        delta = torch.sum(
            weights[:, :, None, None] * deltas,
            dim=1,
        )  # [B, 2, L]

        # Normalize residual strength relative to input RMS.
        # This makes adapter strong enough to matter but controlled.
        if self.normalize_delta:
            x_rms = rms(x)
            d_rms = rms(delta)
            delta = delta * (x_rms / (d_rms + 1e-8))

        t_gate = self.temporal_gate(feat)  # [B, 1, L]
        scale = self.get_scale()

        x_prior = x + scale * t_gate * delta
        
        # Periodically log the statistics so we don't blind run
        if self.training and random.random() < 0.01:
            try:
                res_ratio = rms(x_prior - x).mean().item() / (rms(x).mean().item() + 1e-8)
                print(f"\\n[Strong Prior Adapter]")
                print(f"  Router Weights (mean): {weights.mean(dim=0).detach().cpu().numpy()}")
                print(f"  Scale (item):          {scale.item():.4f}")
                print(f"  Temporal Gate (mean):  {t_gate.mean().item():.4f}")
                print(f"  Residual Ratio:        {res_ratio:.4f}")
            except:
                pass

        if return_aux:
            with torch.no_grad():
                residual_ratio = rms(x_prior - x) / (rms(x) + 1e-8)
            aux = {
                "weights": weights,
                "scale": scale.detach(),
                "temporal_gate_mean": t_gate.mean().detach(),
                "residual_ratio": residual_ratio.mean().detach(),
            }
            return x_prior, aux

        return x_prior


# ============================================================
# Wrapper: direct placement before backbone
# ============================================================

class IQUResUNet1D_StrongPrior(nn.Module):
    """
    Strong adapter + IQUMamba1D backbone.

    Same placement as your MoEPrior version:
        x -> adapter -> backbone
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
        deep_supervision: bool = False,
        adapter_channels: int = 32,
        adapter_max_scale: float = 0.5,
        adapter_scale_init: float = 0.0,
        adapter_temperature: float = 1.0,
        return_adapter_aux: bool = False,
        **kwargs,
    ):
        super().__init__()

        from models.IQUMamba1D import IQUMamba1D

        self.return_adapter_aux = return_adapter_aux

        self.adapter = StrongResidualPriorAdapter1D(
            channels=adapter_channels,
            max_scale=adapter_max_scale,
            scale_init=adapter_scale_init,
            temperature=adapter_temperature,
            normalize_delta=True,
        )

        self.backbone = IQUMamba1D(
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
            conv_bias=True,
            norm_op=nn.InstanceNorm1d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        if self.return_adapter_aux:
            x_prior, aux = self.adapter(x, return_aux=True)
            out = self.backbone(x_prior)
            return out, aux

        x_prior = self.adapter(x)
        return self.backbone(x_prior)
