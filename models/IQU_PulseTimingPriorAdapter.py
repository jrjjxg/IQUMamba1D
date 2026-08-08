from typing import List, Type, Union

import torch
import torch.nn as nn

from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC
from models.IQU_PulsePriorAdapter import PulsePriorAdapter1D
from models.IQU_TimingPriorAdapter import TimingPriorAdapter1D

class IQUResUNet1D_PulseTimingPrior(nn.Module):
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
        num_hypotheses: int = 4,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        
        self.pulse_adapter = PulsePriorAdapter1D(
            rolloffs=rolloffs,
            rrc_kernel_size=rrc_kernel_size,
            gate_hidden=gate_hidden,
            scale_init=scale_init,
        )
        
        self.timing_adapter = TimingPriorAdapter1D(
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
        x = self.pulse_adapter(x)
        x = self.timing_adapter(x)
        return self.backbone(x)
