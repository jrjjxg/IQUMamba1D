import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Type
from models.IQUBiMamba1D import IQUBiMamba1D
from models.IQU_DeepUnfoldedEq import PGDEqualizationUnfoldedHead

class PhysicalChannelConsistencyHead(nn.Module):
    r"""
    Physical Channel Consistency Head embedding Multipath Fading and Carrier Frequency Offset (CFO).
    
    1. Estimates CFO (f_k) and Multipath Filter (h_k) from the coarse separated sources.
    2. Simulates the physical channel: X_hat = sum_k (s_k * h_k) * exp(j * 2pi * f_k * t)
    3. Computes the mixture residual: R = X_mix - X_hat
    4. Computes the gradient w.r.t sources and applies an update:
       s_k^{new} = s_k + \eta \left[ (R * exp(-j * 2pi * f_k * t)) * h_k^*(-t) \right]
    """
    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 32,
        param_kernel_size: int = 7,
        multipath_taps: int = 5,
        max_cfo_hz: float = 1e3,
        sampling_rate: float = 1e6, # assumed default
        step_size_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.multipath_taps = int(multipath_taps)
        if self.multipath_taps % 2 == 0:
            raise ValueError("multipath_taps must be odd")
        
        self.max_cfo_norm = max_cfo_hz / sampling_rate
        
        # Parameter Estimator
        padding = param_kernel_size // 2
        self.param_head = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=param_kernel_size, padding=padding, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            # Output: 1 for CFO, 2 * multipath_taps for complex FIR
            nn.Conv1d(hidden_channels, 1 + 2 * self.multipath_taps, kernel_size=1, bias=True),
        )
        # Init close to identity (zero CFO, delta function for multipath)
        nn.init.zeros_(self.param_head[-1].weight)
        nn.init.zeros_(self.param_head[-1].bias)
        
        # We'll override the center tap bias to 1.0 so that the initial filter is a Dirac delta
        center_idx = 1 + 2 * (self.multipath_taps // 2)
        with torch.no_grad():
            self.param_head[-1].bias[center_idx] = 1.0 # Real part of center tap = 1.0
            
        self.step_size_logit = nn.Parameter(torch.tensor(self._logit(step_size_init)))

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(value / (1.0 - value))
        
    @property
    def step_size(self):
        return torch.sigmoid(self.step_size_logit)
        
    @staticmethod
    def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)

    def _apply_cfo(self, x: torch.Tensor, f_cfo: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L), f_cfo: (B, 1), t: (L)
        # phase = 2 * pi * f_cfo * t
        phase = 2.0 * math.pi * f_cfo.unsqueeze(2) * t.view(1, 1, -1) # (B, 1, L)
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        
        i = x[:, 0, :]
        q = x[:, 1, :]
        
        out_i = i * cos_p.squeeze(1) - q * sin_p.squeeze(1)
        out_q = i * sin_p.squeeze(1) + q * cos_p.squeeze(1)
        return torch.stack((out_i, out_q), dim=1)
        
    def _complex_conv1d(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L) - standard signal
        # h: (B, 2, T) - filter for each batch item separately!
        # Because h is batch-dependent, we use grouped convolution
        b, _, l = x.shape
        taps = h.shape[2]
        
        # x reshape to (1, B*2, L)
        x_g = x.view(1, b * 2, l)
        
        # h: (B, 2, taps) -> we need to form a weight matrix for grouped conv
        # Weight shape for grouped conv: (out_channels, in_channels/groups, kernel_size)
        # We treat each batch item as a group. Groups = B.
        # out_channels = B * 2. in_channels = B * 2.
        # Wait, complex convolution: 
        # Out_R = In_R * h_R - In_I * h_I
        # Out_I = In_R * h_I + In_I * h_R
        
        # Let's do it manually via unfolding or standard padding since T is small (e.g. 5)
        # We can pad x
        pad = taps // 2
        x_padded = F.pad(x, (pad, pad)) # (B, 2, L + 2*pad)
        
        out = torch.zeros_like(x)
        for tau in range(taps):
            x_shifted = x_padded[:, :, tau:tau+l]
            h_tau_r = h[:, 0, tau].view(b, 1)
            h_tau_i = h[:, 1, tau].view(b, 1)
            
            x_r = x_shifted[:, 0, :]
            x_i = x_shifted[:, 1, :]
            
            out_r = x_r * h_tau_r - x_i * h_tau_i
            out_i = x_r * h_tau_i + x_i * h_tau_r
            
            out[:, 0, :] += out_r
            out[:, 1, :] += out_i
            
        return out

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        b, _, target_length = estimates.shape
        mixture = self._resize_mixture(mixture, target_length)
        z = estimates.reshape(b, self.num_sources, 2, target_length)
        
        device = z.device
        t = torch.arange(target_length, device=device, dtype=z.dtype)
        
        new_z = torch.zeros_like(z)
        x_hat_total = torch.zeros_like(mixture)
        
        cfo_list = []
        h_list = []
        
        # Forward Physics
        for k in range(self.num_sources):
            s_k = z[:, k, :, :] # (B, 2, L)
            
            # Predict parameters from S_k
            params = self.param_head(s_k).mean(dim=-1) # (B, 1 + 2T)
            
            f_cfo = torch.tanh(params[:, 0:1]) * self.max_cfo_norm # (B, 1)
            h_r = params[:, 1:1+self.multipath_taps] # (B, T)
            h_i = params[:, 1+self.multipath_taps:] # (B, T)
            h_k = torch.stack((h_r, h_i), dim=1) # (B, 2, T)
            
            cfo_list.append(f_cfo)
            h_list.append(h_k)
            
            # Conv(s_k, h_k)
            filtered = self._complex_conv1d(s_k, h_k)
            
            # CFO Modulation
            x_hat_k = self._apply_cfo(filtered, f_cfo, t)
            x_hat_total += x_hat_k
            
        # Residual
        R = mixture - x_hat_total # (B, 2, L)
        
        # Backward Refinement
        eta = self.step_size
        for k in range(self.num_sources):
            f_cfo = cfo_list[k]
            h_k = h_list[k]
            
            # 1. De-rotate R by -f_cfo
            R_derotated = self._apply_cfo(R, -f_cfo, t)
            
            # 2. Matched filter (conjugate and time-reversed h_k)
            # Time reverse
            h_k_rev = torch.flip(h_k, dims=[2])
            # Conjugate
            h_k_matched = torch.stack((h_k_rev[:, 0, :], -h_k_rev[:, 1, :]), dim=1)
            
            # 3. Apply matched filter
            grad_k = self._complex_conv1d(R_derotated, h_k_matched)
            
            # 4. Update
            s_k = z[:, k, :, :]
            new_z[:, k, :, :] = s_k + eta * grad_k
            
        return new_z.reshape(b, 2 * self.num_sources, target_length)


class IQUBiMamba1D_PhysicalChannel(IQUBiMamba1D):
    """BiMamba separator followed by Physical Channel Consistency head."""

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
        norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {'inplace': True},
        deep_supervision: bool = False,
        phys_multipath_taps: int = 5,
        phys_max_cfo_hz: float = 1e3,
    ) -> None:
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
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")
        
        self.num_sources = num_classes // 2
        self.phys_head = PhysicalChannelConsistencyHead(
            num_sources=self.num_sources,
            multipath_taps=phys_multipath_taps,
            max_cfo_hz=phys_max_cfo_hz,
        )

    def _refine_outputs(self, outputs, mixture: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            outputs[-1] = self.phys_head(outputs[-1], mixture)
            return outputs
        return self.phys_head(outputs, mixture)

    def forward(self, x):
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)


class IQUBiMamba1D_PhysicalChannel_PGDEQ(IQUBiMamba1D):
    """Combines Physical Channel Estimation with PGD Unfolded CMA/MMA Equalization."""

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
        norm_op_kwargs: dict = {'eps': 1e-5, 'affine': True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {'inplace': True},
        deep_supervision: bool = False,
        phys_multipath_taps: int = 5,
        phys_max_cfo_hz: float = 1e3,
        pgd_eq_num_steps: int = 3,
    ) -> None:
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
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        if num_classes % 2 != 0:
            raise ValueError(f"num_classes must be even for I/Q source pairs, got {num_classes}")
        
        self.num_sources = num_classes // 2
        
        # 1. First, compensate channel impairments
        self.phys_head = PhysicalChannelConsistencyHead(
            num_sources=self.num_sources,
            multipath_taps=phys_multipath_taps,
            max_cfo_hz=phys_max_cfo_hz,
        )
        
        # 2. Then, enforce constant/multi-modulus properties
        self.pgd_eq_head = PGDEqualizationUnfoldedHead(
            num_sources=self.num_sources,
            num_steps=pgd_eq_num_steps,
        )

    def _refine_outputs(self, outputs, mixture: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            z_phys = self.phys_head(outputs[-1], mixture)
            # PGD EQ head requires a mixture representation too. We can pass the original mixture.
            z_eq = self.pgd_eq_head(z_phys, mixture)
            outputs[-1] = z_eq
            return outputs
            
        z_phys = self.phys_head(outputs, mixture)
        z_eq = self.pgd_eq_head(z_phys, mixture)
        return z_eq

    def forward(self, x):
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)
