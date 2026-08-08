import math
from typing import List, Type, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d

# ==============================================================================
# Helper Modules
# ==============================================================================

class MultiBranchFRESH(nn.Module):
    def __init__(self, top_k: int = 1):
        super().__init__()
        self.top_k = top_k
        self.num_branches = 1 + 4 * top_k
        
    def _phasors(self, x: torch.Tensor, alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, L = x.shape
        device = x.device
        freq_list = [torch.zeros(B, 1, device=device)]
        for i in range(self.top_k):
            a = alphas[:, i:i+1]
            freq_list.extend([a, -a, 2*a, -2*a])
        freqs = torch.cat(freq_list, dim=1) # [B, 1+4K]
        n = torch.arange(L, device=device, dtype=torch.float32)
        phase = -2.0 * math.pi * freqs.unsqueeze(2) * n.unsqueeze(0).unsqueeze(0)
        cos = torch.cos(phase).to(dtype=x.dtype)
        sin = torch.sin(phase).to(dtype=x.dtype)
        return cos, sin
        
    def forward(self, x: torch.Tensor, alphas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        cos, sin = self._phasors(x, alphas)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return shifted_real, shifted_imag

# ==============================================================================
# Experts
# ==============================================================================

class ExpertBase(nn.Module):
    def __init__(self, max_scale: float = 0.2, scale_init: float = -1.5):
        super().__init__()
        self.max_scale = max_scale
        self.raw_scale = nn.Parameter(torch.tensor(float(scale_init)))
        
    def get_scale(self):
        return self.max_scale * torch.sigmoid(self.raw_scale)

class FRESHExpertBase(ExpertBase):
    """Base logic mimicking Stage 79's branch filter and gate."""
    def __init__(self, num_branches: int, hidden_channels: int = 8, gate_hidden: int = 8, kernel_size: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=num_branches,
            out_complex_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(2 * num_branches, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=hidden_channels,
            out_complex_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        # Small random init instead of absolute zero
        nn.init.normal_(self.out_proj.real.weight, std=1e-2)
        nn.init.normal_(self.out_proj.imag.weight, std=1e-2)
        if self.out_proj.bias_real is not None:
            nn.init.zeros_(self.out_proj.bias_real)
            nn.init.zeros_(self.out_proj.bias_imag)

    def process_fresh(self, r: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        hr, hi = self.branch_filter(r, i)
        gate_input = torch.cat([r, i], dim=1)
        gate = self.gate(gate_input)
        hr = hr * gate
        hi = hi * gate
        dr, di = self.out_proj(hr, hi)
        return torch.cat([dr, di], dim=1) * self.get_scale()


class SingleAlphaFRESHExpert(FRESHExpertBase):
    def __init__(self, **kwargs):
        super().__init__(num_branches=5, **kwargs)
        self.fresh = MultiBranchFRESH(top_k=1)
        
    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        r, i = self.fresh(x, alpha)
        return self.process_fresh(r, i)


class DualAlphaFRESHExpert(FRESHExpertBase):
    def __init__(self, **kwargs):
        super().__init__(num_branches=9, **kwargs)
        self.fresh = MultiBranchFRESH(top_k=2)
        
    def forward(self, x: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        r, i = self.fresh(x, alphas)
        return self.process_fresh(r, i)


class QAMSafeExpert(ExpertBase):
    def __init__(self, hidden_channels: int = 16, dilations: List[int] = [1, 2, 4], **kwargs):
        super().__init__(**kwargs)
        
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                ComplexTiedConv1d(1, hidden_channels // len(dilations), kernel_size=5, padding=2*d, dilation=d)
            )
        
        self.filter = ComplexTiedConv1d(in_complex_channels=hidden_channels, out_complex_channels=hidden_channels, kernel_size=5)
        self.out_proj = ComplexTiedConv1d(hidden_channels, 1, kernel_size=1)
        
        nn.init.normal_(self.out_proj.real.weight, std=1e-2)
        nn.init.normal_(self.out_proj.imag.weight, std=1e-2)
        if self.out_proj.bias_real is not None:
            nn.init.zeros_(self.out_proj.bias_real)
            nn.init.zeros_(self.out_proj.bias_imag)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[:, 0:1, :]
        i = x[:, 1:2, :]
        
        branch_r, branch_i = [], []
        for branch in self.branches:
            br, bi = branch(r, i)
            branch_r.append(br)
            branch_i.append(bi)
            
        cat_r = torch.cat(branch_r, dim=1)
        cat_i = torch.cat(branch_i, dim=1)
        
        hr, hi = self.filter(cat_r, cat_i)
        dr, di = self.out_proj(hr, hi)
        delta = torch.cat([dr, di], dim=1)
        return delta * self.get_scale()

# ==============================================================================
# MoE Core (Evidence & Router)
# ==============================================================================

class PhysicalEvidenceExtractor(nn.Module):
    def __init__(self, min_freq: float = 1/64, max_freq: float = 1/8):
        super().__init__()
        self.min_freq = min_freq
        self.max_freq = max_freq
        
    def forward(self, x: torch.Tensor) -> dict:
        B, C, L = x.shape
        device = x.device
        
        # 1. Constant Modulus Score (PSK: low, QAM/APSK: high)
        amp = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + 1e-8) # [B, L]
        cm_score = amp.std(dim=-1) / (amp.mean(dim=-1) + 1e-8) # [B]
        
        # 2. Cyclic Spectrum
        x_c = torch.complex(x[:, 0], x[:, 1])
        r_0 = x_c * torch.conj(x_c)
        r_0 = r_0 - r_0.mean(dim=-1, keepdim=True)
        spec = torch.fft.fft(r_0, dim=-1).abs()[:, :L//2 + 1]
        
        freqs = torch.fft.rfftfreq(L, d=1.0).to(device)
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        
        masked_power = spec[:, mask]
        masked_freqs = freqs[mask]
        
        avg_power = masked_power.mean(dim=-1) + 1e-8
        topk_k = min(32, masked_power.shape[1])
        topk_vals, topk_idx = torch.topk(masked_power, k=topk_k, dim=-1)
        
        if topk_vals.shape[1] < 32:
            topk_vals = F.pad(topk_vals, (0, 32 - topk_vals.shape[1]))
            topk_idx = F.pad(topk_idx, (0, 32 - topk_idx.shape[1]))
            
        peak1_val = topk_vals[:, 0]
        alpha1 = masked_freqs[topk_idx[:, 0]]
        
        # NMS to find peak 2
        alpha2 = torch.zeros_like(alpha1)
        peak2_val = torch.zeros_like(peak1_val)
        valid2 = torch.zeros_like(peak1_val)
        
        for b in range(B):
            a1 = alpha1[b]
            for i in range(1, 32):
                candidate_a = masked_freqs[topk_idx[b, i]]
                # 5 bins separation instead of 2 bins
                if torch.abs(candidate_a - a1) > 5.0 / L:
                    alpha2[b] = candidate_a
                    peak2_val[b] = topk_vals[b, i]
                    # Check harmonic (5 bins separation from harmonics)
                    if torch.min(torch.abs(candidate_a - 2*a1), torch.abs(a1 - 2*candidate_a)) > 5.0 / L:
                        valid2[b] = 1.0
                    break
                    
        sharpness1 = peak1_val / avg_power
        sharpness2 = peak2_val / avg_power
        dominance = peak1_val / (peak2_val + 1e-8)
        
        harmonic_score = torch.min(torch.abs(alpha2 - 2*alpha1), torch.abs(alpha1 - 2*alpha2)) * L
        harmonic_score = torch.clamp(harmonic_score, max=10.0)
        
        # Evidence Vector
        evidence = torch.stack([
            cm_score,
            sharpness1 / 10.0,
            sharpness2 / 10.0,
            torch.clamp(dominance, max=10.0) / 10.0,
            harmonic_score / 10.0,
            valid2
        ], dim=-1) # [B, 6]
        
        alphas_single = alpha1.unsqueeze(1) # [B, 1]
        alphas_dual = torch.stack([alpha1, alpha2], dim=1) # [B, 2]
        
        # Confidences
        conf1 = torch.tanh(sharpness1 / 5.0)
        conf2 = torch.tanh(sharpness2 / 5.0)
        qam_conf = torch.clamp(cm_score / 0.5, 0.0, 1.0)
        
        return {
            "evidence": evidence,
            "alphas_single": alphas_single,
            "alphas_dual": alphas_dual,
            "conf1": conf1,
            "conf2": conf2,
            "qam_conf": qam_conf,
            "valid2": valid2
        }

class EvidenceRouter(nn.Module):
    def __init__(self, evidence_dim: int = 6, num_experts: int = 3, identity_bias: float = 1.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(evidence_dim, 16),
            nn.SiLU(),
            nn.Linear(16, num_experts) # Outputs raw logits for Expert 1, 2, 3
        )
        self.identity_bias = identity_bias
        
    def forward(self, evidence: torch.Tensor, confs: List[torch.Tensor], valids: List[torch.Tensor]) -> torch.Tensor:
        # evidence: [B, 6]
        B = evidence.shape[0]
        expert_logits = self.mlp(evidence) # [B, 3]
        
        # Inject physical confidence into logits
        # confs is [conf1, conf2, qam_conf]
        expert_logits[:, 0] += torch.log(confs[0] + 1e-8)
        expert_logits[:, 1] += torch.log(confs[1] * valids[1] + 1e-8)
        expert_logits[:, 2] += torch.log(confs[2] + 1e-8)
        
        # Mask out invalid experts structurally
        for i in range(3):
            mask = valids[i] == 0
            expert_logits[mask, i] = -1e9
            
        # Add Expert 0 (Identity) logit with a fixed bias
        identity_logit = torch.ones(B, 1, device=evidence.device) * self.identity_bias
        
        all_logits = torch.cat([identity_logit, expert_logits], dim=-1) # [B, 4]
        weights = F.softmax(all_logits, dim=-1) # [B, 4]
        
        return weights

class MoEPriorAdapter1D(nn.Module):
    def __init__(self, identity_bias: float = 1.5, max_scale: float = 0.2, scale_init: float = -1.5):
        super().__init__()
        self.extractor = PhysicalEvidenceExtractor()
        
        self.expert_single = SingleAlphaFRESHExpert(max_scale=max_scale, scale_init=scale_init)
        self.expert_dual = DualAlphaFRESHExpert(max_scale=max_scale, scale_init=scale_init)
        self.expert_qam = QAMSafeExpert(max_scale=max_scale, scale_init=scale_init)
        
        self.router = EvidenceRouter(identity_bias=identity_bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prevent gradients from flowing back through FFT evidence extraction
        with torch.no_grad():
            info = self.extractor(x.float())
            for k in info:
                info[k] = info[k].detach()
                
        # valid vectors
        B = x.shape[0]
        valid1 = torch.ones(B, device=x.device)
        valid2 = info["valid2"]
        valid3 = torch.ones(B, device=x.device)
        
        delta1 = self.expert_single(x, info["alphas_single"])
        delta2 = self.expert_dual(x, info["alphas_dual"])
        delta3 = self.expert_qam(x)
        
        # Apply confidences to deltas
        c1 = info["conf1"].unsqueeze(-1).unsqueeze(-1)
        c2 = info["conf2"].unsqueeze(-1).unsqueeze(-1)
        cq = info["qam_conf"].unsqueeze(-1).unsqueeze(-1)
        
        x1 = x + delta1 * c1 * valid1.view(-1, 1, 1)
        x2 = x + delta2 * c2 * valid2.view(-1, 1, 1)
        x3 = x + delta3 * cq * valid3.view(-1, 1, 1)
        x0 = x
        
        weights = self.router(info["evidence"], [info["conf1"], info["conf2"], info["qam_conf"]], [valid1, valid2, valid3]) # [B, 4]
        
        w0 = weights[:, 0].view(-1, 1, 1)
        w1 = weights[:, 1].view(-1, 1, 1)
        w2 = weights[:, 2].view(-1, 1, 1)
        w3 = weights[:, 3].view(-1, 1, 1)
        
        out = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
        return out

# ==============================================================================
# Full Model Wrapper
# ==============================================================================

class IQUResUNet1D_MoEPrior(nn.Module):
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
        identity_bias: float = 1.5,
        max_scale: float = 0.2,
        scale_init: float = -1.5,
        **kwargs,
    ):
        super().__init__()
        
        self.adapter = MoEPriorAdapter1D(identity_bias=identity_bias, max_scale=max_scale, scale_init=scale_init)
        
        # Correctly instantiating the standard IQUMamba1D backbone (matches Stage 79)
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
            **kwargs
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x_prior = self.adapter(x)
        return self.backbone(x_prior)
