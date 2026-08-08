import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Helper functions
# ============================================================

def complex_rotate_2ch(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate complex 2-channel IQ by theta.

    x:     [B, 2, L]
    theta: scalar tensor or [B]
    """
    xr = x[:, 0:1, :]
    xi = x[:, 1:2, :]

    c = torch.cos(theta)
    s = torch.sin(theta)

    while c.dim() < xr.dim():
        c = c.view(-1, 1, 1) if theta.dim() > 0 else c.view(1, 1, 1)
        s = s.view(-1, 1, 1) if theta.dim() > 0 else s.view(1, 1, 1)

    yr = xr * c - xi * s
    yi = xr * s + xi * c
    return torch.cat([yr, yi], dim=1)


def complex_rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [B, 2, L]
    return: [B, 1, 1]
    """
    power = x[:, 0:1, :] ** 2 + x[:, 1:2, :] ** 2
    return torch.sqrt(power.mean(dim=-1, keepdim=True) + eps)


def make_square_qam_levels(num_axis_levels: int) -> torch.Tensor:
    """
    For square QAM-like constellation:
        num_axis_levels=4  -> 16QAM axis levels
        num_axis_levels=8  -> 64QAM axis levels
        num_axis_levels=16 -> 256QAM-like axis levels

    Normalize average complex symbol energy to 1.
    """
    levels = torch.arange(
        -(num_axis_levels - 1),
        num_axis_levels,
        2,
        dtype=torch.float32,
    )
    avg_complex_power = 2.0 * torch.mean(levels ** 2)
    levels = levels / torch.sqrt(avg_complex_power)
    return levels


def make_qam_radii(levels: torch.Tensor) -> torch.Tensor:
    """
    Build unique radii from square QAM axis levels.
    """
    ii, qq = torch.meshgrid(levels, levels, indexing="ij")
    radii = torch.sqrt(ii.reshape(-1) ** 2 + qq.reshape(-1) ** 2)
    radii = torch.unique(torch.round(radii * 1e6) / 1e6)
    radii, _ = torch.sort(radii)
    return radii


# ============================================================
# Soft QAM projection
# ============================================================

class SoftQAMLatticeProjector(nn.Module):
    """
    Soft projection to QAM I/Q axis levels and QAM radius set.

    This is not hard slicing.
    It gives differentiable residual hints.
    """
    def __init__(
        self,
        axis_levels: int,
        tau_axis: float = 0.03,
        tau_radius: float = 0.03,
    ):
        super().__init__()

        levels = make_square_qam_levels(axis_levels)
        radii = make_qam_radii(levels)

        self.register_buffer("levels", levels)
        self.register_buffer("radii", radii)

        self.tau_axis = tau_axis
        self.tau_radius = tau_radius

    def soft_axis_project(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, 2, L]
        return:
            z_axis: [B, 2, L]
            err_axis: [B]
        """
        zr = z[:, 0, :]  # [B, L]
        zi = z[:, 1, :]

        levels = self.levels.to(dtype=z.dtype, device=z.device)

        dist_r = (zr.unsqueeze(-1) - levels.view(1, 1, -1)) ** 2
        dist_i = (zi.unsqueeze(-1) - levels.view(1, 1, -1)) ** 2

        wr = torch.softmax(-dist_r / self.tau_axis, dim=-1)
        wi = torch.softmax(-dist_i / self.tau_axis, dim=-1)

        qr = torch.sum(wr * levels.view(1, 1, -1), dim=-1)
        qi = torch.sum(wi * levels.view(1, 1, -1), dim=-1)

        z_axis = torch.stack([qr, qi], dim=1)

        err = ((z_axis - z) ** 2).mean(dim=(1, 2))
        return z_axis, err

    def soft_radius_project(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Radius-directed projection:
            preserve angle, softly snap radius to known QAM radii.

        z: [B, 2, L]
        return:
            z_radius: [B, 2, L]
            err_radius: [B]
        """
        zr = z[:, 0, :]
        zi = z[:, 1, :]

        r = torch.sqrt(zr ** 2 + zi ** 2 + 1e-8)  # [B, L]
        radii = self.radii.to(dtype=z.dtype, device=z.device)

        dist = (r.unsqueeze(-1) - radii.view(1, 1, -1)) ** 2
        w = torch.softmax(-dist / self.tau_radius, dim=-1)
        rq = torch.sum(w * radii.view(1, 1, -1), dim=-1)

        scale = rq / (r + 1e-6)

        z_radius = torch.stack([zr * scale, zi * scale], dim=1)
        err = ((z_radius - z) ** 2).mean(dim=(1, 2))
        return z_radius, err

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z: [B, 2, L], normalized complex signal

        return:
            delta: [B, 2, L]
            err:   [B]
        """
        z_axis, err_axis = self.soft_axis_project(z)
        z_radius, err_radius = self.soft_radius_project(z)

        # MMA-like axis prior is usually more informative for square QAM.
        # RDE-like radius prior is a secondary stabilizer.
        delta_axis = z_axis - z
        delta_radius = z_radius - z

        delta = 0.7 * delta_axis + 0.3 * delta_radius
        err = 0.7 * err_axis + 0.3 * err_radius

        return delta, err


# ============================================================
# QAM Lattice-Radius Prior Adapter
# ============================================================

class QAMLatticeRadiusPriorAdapter1D(nn.Module):
    """
    QAM-specific front-end prior adapter.

    Design:
      - phase-bank soft alignment
      - QAM axis-level projection, MMA-like
      - QAM radius projection, RDE-like
      - soft branch routing by projection consistency
      - temporal gate to avoid forcing non-symbol samples to constellation points
      - residual-only update

    Input:
        x: [B, 2, L]

    Output:
        x_prior: [B, 2, L]
    """
    def __init__(
        self,
        axis_level_bank: Tuple[int, ...] = (4, 8, 16),
        phase_bank: Tuple[float, ...] = (
            -math.pi / 8,
            -math.pi / 16,
            0.0,
            math.pi / 16,
            math.pi / 8,
            math.pi / 4,
        ),
        tau_axis: float = 0.03,
        tau_radius: float = 0.03,
        route_temperature: float = 0.05,
        max_scale: float = 0.35,
        scale_init: float = -1.0,
        gate_channels: int = 16,
    ):
        super().__init__()

        self.axis_level_bank = tuple(axis_level_bank)
        self.phase_bank = tuple(float(p) for p in phase_bank)
        self.route_temperature = route_temperature

        self.projectors = nn.ModuleList([
            SoftQAMLatticeProjector(
                axis_levels=m,
                tau_axis=tau_axis,
                tau_radius=tau_radius,
            )
            for m in self.axis_level_bank
        ])

        self.register_buffer(
            "phase_tensor",
            torch.tensor(self.phase_bank, dtype=torch.float32),
        )

        # Temporal gate: decides where constellation prior is reliable.
        # It is intentionally light.
        self.gate_net = nn.Sequential(
            nn.Conv1d(6, gate_channels, kernel_size=7, padding=3),
            nn.InstanceNorm1d(gate_channels, affine=True),
            nn.SiLU(),
            nn.Conv1d(gate_channels, gate_channels, kernel_size=5, padding=2),
            nn.InstanceNorm1d(gate_channels, affine=True),
            nn.SiLU(),
            nn.Conv1d(gate_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.max_scale = float(max_scale)
        self.raw_scale = nn.Parameter(torch.tensor(float(scale_init)))

        # Learnable branch bias lets the model prefer 16QAM/64QAM/etc.
        num_branches = len(self.axis_level_bank) * len(self.phase_bank)
        self.branch_bias = nn.Parameter(torch.zeros(num_branches))

    def get_scale(self) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.raw_scale)

    def make_hints(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm: [B, 2, L]
        return: [B, 6, L]
        """
        i = x_norm[:, 0:1, :]
        q = x_norm[:, 1:2, :]

        amp = torch.sqrt(i ** 2 + q ** 2 + 1e-8)
        i_unit = i / (amp + 1e-6)
        q_unit = q / (amp + 1e-6)

        i_prev = F.pad(i[..., :-1], (1, 0))
        q_prev = F.pad(q[..., :-1], (1, 0))

        cross_r = i * i_prev + q * q_prev
        cross_i = q * i_prev - i * q_prev
        phase_inc = cross_i / torch.sqrt(cross_r ** 2 + cross_i ** 2 + 1e-8)

        return torch.cat([i, q, amp, i_unit, q_unit, phase_inc], dim=1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        x: [B, 2, L]
        """
        B, C, L = x.shape
        assert C == 2, "QAMLatticeRadiusPriorAdapter1D expects [B, 2, L] IQ input."

        # Normalize because QAM levels are normalized to average symbol energy 1.
        x_scale = complex_rms(x)
        x_norm = x / (x_scale + 1e-8)

        all_deltas = []
        all_errs = []

        phase_tensor = self.phase_tensor.to(device=x.device, dtype=x.dtype)

        branch_idx = 0
        for theta in phase_tensor:
            # rotate into candidate QAM axis
            z_rot = complex_rotate_2ch(x_norm, theta)

            for projector in self.projectors:
                delta_rot, err = projector(z_rot)

                # rotate residual back
                delta_back = complex_rotate_2ch(delta_rot, -theta)

                all_deltas.append(delta_back)
                all_errs.append(err)

                branch_idx += 1

        deltas = torch.stack(all_deltas, dim=1)  # [B, N, 2, L]
        errs = torch.stack(all_errs, dim=1)      # [B, N]

        # Lower projection error => more likely this QAM order/phase explains the signal.
        logits = -errs / self.route_temperature + self.branch_bias.view(1, -1)
        weights = torch.softmax(logits, dim=1)   # [B, N]

        delta_norm = torch.sum(
            weights[:, :, None, None] * deltas,
            dim=1,
        )  # [B, 2, L]

        # Temporal gate prevents forcing every oversampled waveform sample to QAM lattice.
        hints = self.make_hints(x_norm)
        gate = self.gate_net(hints)  # [B, 1, L]

        scale = self.get_scale()

        # Rescale delta back to original amplitude.
        delta = delta_norm * x_scale

        x_prior = x + scale * gate * delta

        if return_aux:
            with torch.no_grad():
                residual_ratio = torch.sqrt(((x_prior - x) ** 2).mean()) / (
                    torch.sqrt((x ** 2).mean()) + 1e-8
                )
                aux = {
                    "scale": scale.detach(),
                    "gate_mean": gate.mean().detach(),
                    "branch_weights_mean": weights.mean(dim=0).detach(),
                    "projection_error_mean": errs.mean().detach(),
                    "residual_ratio": residual_ratio.detach(),
                }
            return x_prior, aux

        return x_prior

class IQUResUNet1D_QAMPrior(nn.Module):
    """
    QAM prior adapter + backbone.

    Placement:
        x -> QAMLatticeRadiusPriorAdapter1D -> backbone
    """
    def __init__(
        self,
        input_size: int,
        input_channels: int,
        n_stages: int,
        features_per_stage,
        conv_op,
        kernel_sizes,
        strides,
        n_conv_per_stage,
        num_classes: int,
        n_conv_per_stage_decoder,
        deep_supervision: bool = False,
        qam_axis_level_bank=(4, 8, 16),
        qam_max_scale: float = 0.35,
        qam_scale_init: float = -1.0,
        return_adapter_aux: bool = False,
        **kwargs,
    ):
        super().__init__()

        from models.IQUMamba1D import IQUMamba1D

        self.return_adapter_aux = return_adapter_aux

        self.adapter = QAMLatticeRadiusPriorAdapter1D(
            axis_level_bank=qam_axis_level_bank,
            max_scale=qam_max_scale,
            scale_init=qam_scale_init,
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

    def forward(self, x):
        if self.return_adapter_aux:
            x_prior, aux = self.adapter(x, return_aux=True)
            out = self.backbone(x_prior)
            return out, aux

        x_prior = self.adapter(x)
        return self.backbone(x_prior)
