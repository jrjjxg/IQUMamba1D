import math
from typing import List, Type, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUResUNet1D_NoASC import IQUResUNet1D_NoASC
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d

def generate_rrc_filter(length: int, rolloff: float, sps: int = 4) -> torch.Tensor:
    """Generate Root-Raised Cosine filter weights.
    length: filter length, must be odd.
    sps: samples per symbol (pseudo-parameter for generation)
    """
    if length % 2 == 0:
        length += 1
    t = torch.arange(-(length // 2), (length // 2) + 1, dtype=torch.float32) / sps
    
    # Avoid division by zero
    t = t + 1e-8
    
    # RRC formula
    numerator = torch.sin(math.pi * t * (1 - rolloff)) + 4 * rolloff * t * torch.cos(math.pi * t * (1 + rolloff))
    denominator = math.pi * t * (1 - (4 * rolloff * t)**2)
    h = numerator / denominator
    
    # Handle singularity at t = +/- 1/(4*rolloff)
    if rolloff != 0:
        singularity_idx = torch.abs(torch.abs(t) - 1 / (4 * rolloff)) < 1e-6
        if singularity_idx.any():
            val = (rolloff / math.sqrt(2)) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * rolloff)) + (1 - 2 / math.pi) * math.cos(math.pi / (4 * rolloff)))
            h[singularity_idx] = val
            
    # Normalize
    h = h / torch.sqrt(torch.sum(h**2))
    return h

class MultiLagCyclicEstimator(nn.Module):
    def __init__(self, lags: List[int] = [0, 1, 2, 4, 8, 16], min_freq: float = 1/64, max_freq: float = 1/8, top_k: int = 3):
        super().__init__()
        self.lags = lags
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 2, L]
        B, C, L = x.shape
        device = x.device
        
        # We need a complex tensor to do conj mult
        x_c = torch.complex(x[:, 0, :], x[:, 1, :]) # [B, L]
        
        combined_power = None
        
        for d in self.lags:
            if d == 0:
                r_d = x_c * torch.conj(x_c)
            else:
                if L <= d:
                    continue
                r_d = x_c[:, d:] * torch.conj(x_c[:, :-d])
                # zero pad to L
                r_d = F.pad(r_d, (0, d))
            
            # center
            r_d = r_d - r_d.mean(dim=-1, keepdim=True)
            
            # magnitude spectrum
            spec = torch.fft.fft(r_d, dim=-1).abs()[:, :L//2 + 1] # [B, L//2 + 1]
            if combined_power is None:
                combined_power = spec
            else:
                combined_power = combined_power + spec
                
        if combined_power is None:
            # Fallback
            alphas = torch.ones(B, self.top_k, device=device) * self.min_freq
            conf = torch.zeros(B, self.top_k, device=device)
            return alphas, conf
            
        freqs = torch.fft.rfftfreq(L, d=1.0).to(device)
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        
        masked_power = combined_power[:, mask] # [B, num_valid_bins]
        masked_freqs = freqs[mask]
        
        if masked_power.shape[1] == 0:
            alphas = torch.ones(B, self.top_k, device=device) * self.min_freq
            conf = torch.zeros(B, self.top_k, device=device)
            return alphas, conf
            
        # Get top K
        topk_vals, topk_idx = torch.topk(masked_power, k=min(self.top_k, masked_power.shape[1]), dim=-1)
        
        # Pad if we don't have enough bins
        if topk_vals.shape[1] < self.top_k:
            pad_size = self.top_k - topk_vals.shape[1]
            topk_vals = F.pad(topk_vals, (0, pad_size))
            topk_idx = F.pad(topk_idx, (0, pad_size))
            
        alphas = masked_freqs[topk_idx] # [B, K]
        
        # Confidence score (peak power vs average power in the mask)
        avg_power = masked_power.mean(dim=-1, keepdim=True) + 1e-8
        conf = (topk_vals / avg_power) # Relative peak sharpness
        # Normalize confidence to somewhat 0-1 range using tanh
        conf = torch.tanh(conf / 10.0) 
        
        return alphas, conf

class MultiBranchFRESH(nn.Module):
    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k
        self.num_branches = 1 + 4 * top_k # 0, +/- alpha, +/- 2alpha for each K
        
    def _phasors(self, x: torch.Tensor, alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 2, L]
        # alphas: [B, K]
        B, C, L = x.shape
        device = x.device
        
        # freqs: [B, 1+4K]
        freq_list = [torch.zeros(B, 1, device=device)]
        for i in range(self.top_k):
            a = alphas[:, i:i+1]
            freq_list.extend([a, -a, 2*a, -2*a])
            
        freqs = torch.cat(freq_list, dim=1) # [B, 1+4K]
        
        n = torch.arange(L, device=device, dtype=torch.float32) # [L]
        phase = -2.0 * math.pi * freqs.unsqueeze(2) * n.unsqueeze(0).unsqueeze(0) # [B, 1+4K, L]
        
        cos = torch.cos(phase).to(dtype=x.dtype)
        sin = torch.sin(phase).to(dtype=x.dtype)
        return cos, sin
        
    def forward(self, x: torch.Tensor, alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 2, L]
        real = x[:, 0:1, :] # [B, 1, L]
        imag = x[:, 1:2, :]
        
        cos, sin = self._phasors(x, alphas)
        
        shifted_real = real * cos - imag * sin # [B, 1+4K, L]
        shifted_imag = real * sin + imag * cos
        
        return shifted_real, shifted_imag

class RRCBank(nn.Module):
    def __init__(self, rolloffs: List[float] = [0.2, 0.35, 0.5], kernel_size: int = 31):
        super().__init__()
        self.num_filters = len(rolloffs)
        self.conv_real = nn.Conv1d(1, self.num_filters, kernel_size, padding=kernel_size//2, bias=False)
        self.conv_imag = nn.Conv1d(1, self.num_filters, kernel_size, padding=kernel_size//2, bias=False)
        
        # Initialize with RRC
        with torch.no_grad():
            for i, r in enumerate(rolloffs):
                h = generate_rrc_filter(kernel_size, r).view(1, 1, -1)
                self.conv_real.weight[i:i+1] = h
                self.conv_imag.weight[i:i+1] = h
                
        # We also need to map from (B, 1, L) to (B, num_filters, L)
        # We process real and imag separately to maintain them as 2 channels.
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        
        r_out = self.conv_real(real) # [B, num_filters, L]
        i_out = self.conv_imag(imag)
        
        return r_out, i_out

class UniversalPriorAdapter1D(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        min_freq: float = 1/64,
        max_freq: float = 1/8,
        top_k: int = 3,
        rolloffs: List[float] = [0.2, 0.35, 0.5],
        rrc_kernel_size: int = 31,
        fresh_kernel_size: int = 9,
        hidden_channels: int = 16,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
    ):
        super().__init__()
        
        self.top_k = top_k
        self.estimator = MultiLagCyclicEstimator(min_freq=min_freq, max_freq=max_freq, top_k=top_k)
        self.fresh = MultiBranchFRESH(top_k=top_k)
        self.rrc = RRCBank(rolloffs=rolloffs, kernel_size=rrc_kernel_size)
        
        num_fresh_branches = 1 + 4 * top_k
        num_rrc_branches = len(rolloffs)
        
        total_branches = num_fresh_branches + num_rrc_branches
        
        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=total_branches,
            out_complex_channels=hidden_channels,
            kernel_size=fresh_kernel_size,
            bias=True,
        )
        
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=hidden_channels,
            out_complex_channels=1, # output 1 complex channel (2 real channels)
            kernel_size=1,
            bias=True,
        )
        
        # Router
        # Takes input features + confidence scores -> generates a gate
        self.conf_proj = nn.Linear(top_k, gate_hidden)
        self.gate = nn.Sequential(
            nn.Conv1d(2 * total_branches + gate_hidden, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        
        # Zero init output projection to start as pure identity
        nn.init.zeros_(self.out_proj.real.weight)
        nn.init.zeros_(self.out_proj.imag.weight)
        if self.out_proj.bias_real is not None:
            nn.init.zeros_(self.out_proj.bias_real)
            nn.init.zeros_(self.out_proj.bias_imag)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, L]
        B, _, L = x.shape
        
        alphas, conf = self.estimator(x) # alphas: [B, K], conf: [B, K]
        
        fresh_real, fresh_imag = self.fresh(x, alphas) # [B, 13, L]
        rrc_real, rrc_imag = self.rrc(x) # [B, 3, L]
        
        all_real = torch.cat([fresh_real, rrc_real], dim=1) # [B, 16, L]
        all_imag = torch.cat([fresh_imag, rrc_imag], dim=1)
        
        hidden_real, hidden_imag = self.branch_filter(all_real, all_imag) # [B, hidden, L]
        
        conf_feat = self.conf_proj(conf).unsqueeze(-1).expand(-1, -1, L) # [B, gate_hidden, L]
        
        gate_input = torch.cat([all_real, all_imag, conf_feat], dim=1)
        gate = self.gate(gate_input) # [B, hidden, L]
        
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        delta = torch.cat([delta_real, delta_imag], dim=1) # [B, 2, L]
        
        return x + self.scale * delta

class IQUResUNet1D_UniversalPrior(nn.Module):
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
        # Universal Prior kwargs
        min_freq: float = 1/64,
        max_freq: float = 1/8,
        top_k: int = 3,
        rolloffs: List[float] = [0.2, 0.35, 0.5],
        rrc_kernel_size: int = 31,
        fresh_kernel_size: int = 9,
        hidden_channels: int = 16,
        gate_hidden: int = 16,
        scale_init: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        
        self.adapter = UniversalPriorAdapter1D(
            input_channels=input_channels,
            min_freq=min_freq,
            max_freq=max_freq,
            top_k=top_k,
            rolloffs=rolloffs,
            rrc_kernel_size=rrc_kernel_size,
            fresh_kernel_size=fresh_kernel_size,
            hidden_channels=hidden_channels,
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
