"""Stage 223: Stage-4 IQUMamba with a training-only noise prior projector."""

from __future__ import annotations

from typing import List, Type

import torch
from torch import nn
from torch.nn import functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_EvidenceRoutedMoE import PaddedStage4Backbone


def _rounded_backbone_length(input_size: int, strides) -> int:
    total_stride = 1
    for stride in strides:
        total_stride *= max(1, int(stride))
    return ((int(input_size) + total_stride - 1) // total_stride) * total_stride


class NoisePatchProjector(nn.Module):
    """Small shared IQ patch encoder used only by the auxiliary loss."""

    def __init__(self, hidden: int = 12, embedding: int = 16):
        super().__init__()
        hidden = max(1, int(hidden))
        embedding = max(2, int(embedding))
        self.network = nn.Sequential(
            nn.Conv1d(4, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden, embedding)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 3 or patches.size(1) != 2:
            raise ValueError(f"NoisePatchProjector expects [N, 2, S], got {tuple(patches.shape)}")
        patches = patches.float()
        eps = 1e-8
        scale = patches.square().mean(dim=(1, 2), keepdim=True).add(eps).sqrt()
        normalized = patches / scale
        delta = F.pad(normalized[..., 1:] - normalized[..., :-1], (1, 0))
        features = torch.cat([normalized, delta], dim=1)
        hidden = self.network(features)
        pooled = hidden.mean(dim=-1)
        return self.output(pooled)


class IQUMamba1D_NoiseContrastivePrior(nn.Module):
    """Keep Stage-4 inference unchanged while exposing a training projector."""

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
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = nn.InstanceNorm1d,
        norm_op_kwargs: dict | None = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict | None = None,
        deep_supervision: bool = False,
        noise_prior_hidden: int = 12,
        noise_prior_embedding: int = 16,
        noise_prior_patch_size: int = 64,
        noise_prior_patch_stride: int = 32,
    ):
        super().__init__()
        if norm_op_kwargs is None:
            norm_op_kwargs = {"eps": 1e-5, "affine": True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {"inplace": True}
        self.noise_prior_patch_size = max(1, int(noise_prior_patch_size))
        self.noise_prior_patch_stride = max(1, int(noise_prior_patch_stride))

        backbone_length = _rounded_backbone_length(input_size, strides)
        raw_backbone = IQUMamba1D(
            input_size=backbone_length,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.backbone = PaddedStage4Backbone(raw_backbone, backbone_length)
        self.noise_prior_projector = NoisePatchProjector(
            hidden=noise_prior_hidden,
            embedding=noise_prior_embedding,
        )

    def project_patches(
        self,
        value: torch.Tensor,
        patch_size: int | None = None,
        patch_stride: int | None = None,
    ) -> torch.Tensor:
        """Project [B, 2, L] local patches to [B, P, embedding]."""
        if value.ndim != 3 or value.size(1) != 2:
            raise ValueError(f"project_patches expects [B, 2, L], got {tuple(value.shape)}")
        patch_size = self.noise_prior_patch_size if patch_size is None else max(1, int(patch_size))
        patch_stride = self.noise_prior_patch_stride if patch_stride is None else max(1, int(patch_stride))
        if value.size(-1) < patch_size:
            value = F.pad(value, (0, patch_size - value.size(-1)), mode="replicate")
        patches = value.unfold(-1, patch_size, patch_stride).permute(0, 2, 1, 3).contiguous()
        projected = self.noise_prior_projector(patches.reshape(-1, 2, patch_size))
        return projected.reshape(value.size(0), patches.size(1), -1)

    def forward(self, x: torch.Tensor):
        # The projector is intentionally absent from this path.  The training
        # criterion calls it only when gradients are enabled.
        return self.backbone(x)
