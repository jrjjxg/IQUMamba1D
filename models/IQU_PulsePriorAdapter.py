import math
from typing import List, Type, Union, Tuple

import torch
import torch.nn as nn

from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC

def generate_rrc_filter(length: int, rolloff: float, sps: int = 4) -> torch.Tensor:
    """Generate Root-Raised Cosine filter weights.
    length: filter length, must be odd.
    sps: samples per symbol (pseudo-parameter for generation)
    """
    if length % 2 == 0:
        length += 1
    t = torch.arange(-(length // 2), (length // 2) + 1, dtype=torch.float32) / sps
    t = t + 1e-8 # avoid division by zero
    numerator = torch.sin(math.pi * t * (1 - rolloff)) + 4 * rolloff * t * torch.cos(math.pi * t * (1 + rolloff))
    denominator = math.pi * t * (1 - (4 * rolloff * t)**2)
    h = numerator / denominator
    if rolloff != 0:
        singularity_idx = torch.abs(torch.abs(t) - 1 / (4 * rolloff)) < 1e-6
        if singularity_idx.any():
            val = (rolloff / math.sqrt(2)) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * rolloff)) + (1 - 2 / math.pi) * math.cos(math.pi / (4 * rolloff)))
            h[singularity_idx] = val
    h = h / torch.sqrt(torch.sum(h**2))
    return h

class PulsePriorAdapter1D(nn.Module):
    def __init__(
        self,
        rolloffs: List[float] = [0.2, 0.35, 0.5],
        rrc_kernel_size: int = 31,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
    ):
        super().__init__()
        self.num_filters = len(rolloffs)
        
        # Parallel convolutions initialized with RRC, requires_grad=True allows residual learning
        self.conv_real = nn.Conv1d(1, self.num_filters, rrc_kernel_size, padding=rrc_kernel_size//2, bias=True)
        self.conv_imag = nn.Conv1d(1, self.num_filters, rrc_kernel_size, padding=rrc_kernel_size//2, bias=True)
        
        with torch.no_grad():
            for i, r in enumerate(rolloffs):
                h = generate_rrc_filter(rrc_kernel_size, r).view(1, 1, -1)
                self.conv_real.weight[i:i+1] = h
                self.conv_imag.weight[i:i+1] = h
            nn.init.zeros_(self.conv_real.bias)
            nn.init.zeros_(self.conv_imag.bias)

        # Gate and Fusion
        # Input features (2) + Output features (2 * num_filters) -> Gate weights
        total_channels = 2 + 2 * self.num_filters
        self.gate = nn.Sequential(
            nn.Conv1d(total_channels, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, 2 * self.num_filters, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.out_proj = nn.Conv1d(2 * self.num_filters, 2, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        
        # Identity initialization
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L]
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        
        r_out = self.conv_real(real) # [B, num_filters, L]
        i_out = self.conv_imag(imag)
        
        out_features = torch.cat([r_out, i_out], dim=1) # [B, 2*num_filters, L]
        
        gate_input = torch.cat([x, out_features], dim=1) # [B, 2 + 2*num_filters, L]
        gate = self.gate(gate_input) # [B, 2*num_filters, L]
        
        gated_features = out_features * gate
        delta = self.out_proj(gated_features) # [B, 2, L]
        
        return x + self.scale * delta

class IQUResUNet1D_PulsePrior(nn.Module):
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
        use_complex_mask: bool = False,
        rolloffs: List[float] = [0.2, 0.35, 0.5],
        rrc_kernel_size: int = 31,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.adapter = PulsePriorAdapter1D(
            rolloffs=rolloffs,
            rrc_kernel_size=rrc_kernel_size,
            gate_hidden=gate_hidden,
            scale_init=scale_init,
        )
        self.backbone = IQUResUNet1D_NoASC(
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
            deep_supervision=deep_supervision,
            use_complex_mask=use_complex_mask,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x_prior = self.adapter(x)
        return self.backbone(x_prior)
