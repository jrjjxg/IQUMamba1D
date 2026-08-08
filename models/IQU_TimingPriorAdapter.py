import math
from typing import List, Type, Union

import torch
import torch.nn as nn

from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC

def fft_fractional_delay(x: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """
    Apply differentiable fractional delay using FFT phase shift.
    x: [B, C, L] 
    taus: [B, M] fractional delays
    returns: [B, M, C, L] shifted signals
    """
    B, C, L = x.shape
    M = taus.shape[1]
    
    # We do real FFT. If x is [B, C, L] real components, we just apply delay to each channel independently.
    X = torch.fft.rfft(x, dim=-1) # [B, C, L//2+1]
    
    freqs = torch.fft.rfftfreq(L, d=1.0).to(x.device) # [L//2+1]
    
    # phase: -2pi * f * tau
    # freqs: [1, 1, 1, L//2+1]
    # taus: [B, 1, M, 1]
    phase = -2.0 * math.pi * freqs.view(1, 1, 1, -1) * taus.view(B, 1, M, 1) # [B, 1, M, L//2+1]
    
    shift_phasor = torch.exp(1j * phase) # [B, 1, M, L//2+1]
    
    # Apply phasor to X: X is [B, C, 1, L//2+1]
    X_shifted = X.unsqueeze(2) * shift_phasor # [B, C, M, L//2+1]
    
    x_shifted = torch.fft.irfft(X_shifted, n=L, dim=-1) # [B, C, M, L]
    
    # Transpose to [B, M, C, L]
    return x_shifted.transpose(1, 2)

class TimingPriorAdapter1D(nn.Module):
    def __init__(
        self,
        num_hypotheses: int = 4,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
    ):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        
        # Timing Head: estimates M fractional delays from input
        self.timing_head = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=9, padding=4, stride=2),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_hypotheses),
            nn.Tanh() # Restrict tau between -1 and 1
        )
        
        # Gate and Fusion
        # Input features (2) + Shifted features (2 * M)
        total_channels = 2 + 2 * num_hypotheses
        self.gate = nn.Sequential(
            nn.Conv1d(total_channels, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, 2 * num_hypotheses, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.out_proj = nn.Conv1d(2 * num_hypotheses, 2, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L]
        B, C, L = x.shape
        
        taus = self.timing_head(x) # [B, M]
        
        shifted_x = fft_fractional_delay(x, taus) # [B, M, 2, L]
        shifted_features = shifted_x.reshape(B, self.num_hypotheses * 2, L) # [B, 2M, L]
        
        gate_input = torch.cat([x, shifted_features], dim=1) # [B, 2 + 2M, L]
        gate = self.gate(gate_input) # [B, 2M, L]
        
        gated_features = shifted_features * gate
        delta = self.out_proj(gated_features) # [B, 2, L]
        
        return x + self.scale * delta

class IQUResUNet1D_TimingPrior(nn.Module):
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
        num_hypotheses: int = 4,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.adapter = TimingPriorAdapter1D(
            num_hypotheses=num_hypotheses,
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
