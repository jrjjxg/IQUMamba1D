import torch
import torch.nn as nn
from typing import List

from models.IQUMamba1D import IQUMamba1D

class Sine(nn.Module):
    """
    Sine activation function for Implicit Neural Representations (SIREN).
    f(x) = sin(omega_0 * x)
    """
    def __init__(self, omega_0=1.0, inplace=False):
        super().__init__()
        self.omega_0 = omega_0
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.omega_0).sin_()
        else:
            return torch.sin(self.omega_0 * x)

def sine_init(m):
    """
    SIREN initialization scheme.
    """
    with torch.no_grad():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            num_input = m.weight.size(1)
            # Ensure spatial dimensions are multiplied if it's a conv layer
            if m.weight.ndim > 2:
                for i in range(2, m.weight.ndim):
                    num_input *= m.weight.size(i)
            # Standard SIREN bounds: sqrt(6 / fan_in)
            # We scale by omega_0=1 for hidden layers. 
            # First layer uses a different omega_0 but we assume omega_0=1 for simplicity across hidden layers.
            c = torch.sqrt(torch.tensor(6.0 / num_input))
            m.weight.uniform_(-c, c)
            if m.bias is not None:
                m.bias.fill_(0)

class IQUMamba1D_SIREN(nn.Module):
    def __init__(self,
                 input_size: int,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: List[int],
                 n_conv_per_stage: List[int],
                 num_classes: int,
                 n_conv_per_stage_decoder: List[int],
                 conv_bias: bool = True,
                 ):
        super().__init__()
        # Wrap IQUMamba1D but replace the nonlin with Sine
        self.model = IQUMamba1D(
            input_size=input_size,
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv1d,
            kernel_sizes=[3] * n_stages,
            strides=[1] + [2] * (n_stages - 1),
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=nn.InstanceNorm1d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=Sine,
            nonlin_kwargs={'omega_0': 1.0, 'inplace': False},
            deep_supervision=False,
        )
        
        # Apply SIREN initialization to the CNN parts
        self.model.apply(sine_init)

    def forward(self, x):
        return self.model(x)
