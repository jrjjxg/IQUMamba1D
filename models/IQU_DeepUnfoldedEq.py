import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Type
from models.IQUBiMamba1D import IQUBiMamba1D

class PGDEqualizationUnfoldedHead(nn.Module):
    """
    Proximal-gradient-unfolded equalization head.
    
    Data Consistency Step:
        r^t = sum_k s_k^t - x_mix
        y_k^t = s_k^t - eta_t r^t
        
    Proximal Step (Physics-Embedded Equalization):
        Applies a learned combination of CMA (Constant Modulus) and MMA (Multi-Modulus)
        gradient updates as the proximal operator.
        
        CMA: s_k^{t+1} = y_k - mu_{cma} * y_k * (|y_k|^2 - R_cma^2)
        MMA: s_k^{t+1}_I = y_k_I - mu_{mma} * y_k_I * (y_k_I^2 - R_mma_I^2)
             s_k^{t+1}_Q = y_k_Q - mu_{mma} * y_k_Q * (y_k_Q^2 - R_mma_Q^2)
    """
    def __init__(
        self,
        num_sources: int,
        num_steps: int = 3,
        step_size_init: float = 0.5,
        eq_step_init: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.num_steps = int(num_steps)
        self.eps = float(eps)
        self.max_step_size = 2.0 / float(self.num_sources)
        
        # PGD step size for data consistency
        init_ratio = float(step_size_init) / self.max_step_size
        self.step_size_logits = nn.Parameter(torch.full((self.num_steps,), self._logit(init_ratio)))
        
        # CMA Parameters: R^2 and step size
        self.cma_r2 = nn.Parameter(torch.ones(self.num_sources))
        self.cma_mu_logits = nn.Parameter(torch.full((self.num_steps, self.num_sources), self._logit(eq_step_init)))
        
        # MMA Parameters: R_I^2, R_Q^2 and step size
        self.mma_r2_i = nn.Parameter(torch.ones(self.num_sources))
        self.mma_r2_q = nn.Parameter(torch.ones(self.num_sources))
        self.mma_mu_logits = nn.Parameter(torch.full((self.num_steps, self.num_sources), self._logit(eq_step_init)))
        
        # Learnable mixing coefficient between CMA and MMA for each source
        # sigmoid(alpha) -> 1.0 means pure CMA, 0.0 means pure MMA
        self.cma_mma_alpha = nn.Parameter(torch.zeros(self.num_sources))
        
    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(value / (1.0 - value))
        
    def _step_size(self, step_idx: int) -> torch.Tensor:
        return self.max_step_size * torch.sigmoid(self.step_size_logits[step_idx])
        
    def _cma_mu(self, step_idx: int) -> torch.Tensor:
        return torch.sigmoid(self.cma_mu_logits[step_idx]) # (K,)
        
    def _mma_mu(self, step_idx: int) -> torch.Tensor:
        return torch.sigmoid(self.mma_mu_logits[step_idx]) # (K,)
        
    @staticmethod
    def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(mixture, size=target_length, mode="linear", align_corners=False)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        # estimates: (B, 2K, L)
        # mixture: (B, 2, L)
        b, _, target_length = estimates.shape
        mixture = self._resize_mixture(mixture, target_length)
        z = estimates.reshape(b, self.num_sources, 2, target_length)
        
        alpha = torch.sigmoid(self.cma_mma_alpha).view(1, self.num_sources, 1, 1)
        
        # We ensure R2 is always positive using softplus
        cma_r2 = F.softplus(self.cma_r2).view(1, self.num_sources, 1)
        mma_r2_i = F.softplus(self.mma_r2_i).view(1, self.num_sources, 1)
        mma_r2_q = F.softplus(self.mma_r2_q).view(1, self.num_sources, 1)

        for step_idx in range(self.num_steps):
            # 1. Data Consistency Step
            residual = z.sum(dim=1) - mixture # (B, 2, L)
            y = z - self._step_size(step_idx) * residual.unsqueeze(1) # (B, K, 2, L)
            
            # 2. Equalization Proximal Step
            i_y = y[:, :, 0, :]
            q_y = y[:, :, 1, :]
            
            # CMA Update
            cma_mu = self._cma_mu(step_idx).view(1, self.num_sources, 1)
            mag_sq = i_y**2 + q_y**2 # (B, K, L)
            cma_delta = mag_sq - cma_r2
            cma_update_i = i_y - cma_mu * i_y * cma_delta
            cma_update_q = q_y - cma_mu * q_y * cma_delta
            
            # MMA Update
            mma_mu = self._mma_mu(step_idx).view(1, self.num_sources, 1)
            mma_delta_i = i_y**2 - mma_r2_i
            mma_delta_q = q_y**2 - mma_r2_q
            mma_update_i = i_y - mma_mu * i_y * mma_delta_i
            mma_update_q = q_y - mma_mu * q_y * mma_delta_q
            
            # Blend
            next_i = alpha.squeeze(2) * cma_update_i + (1 - alpha.squeeze(2)) * mma_update_i
            next_q = alpha.squeeze(2) * cma_update_q + (1 - alpha.squeeze(2)) * mma_update_q
            
            z = torch.stack((next_i, next_q), dim=2)
            
        return z.reshape(b, 2 * self.num_sources, target_length)

class IQUBiMamba1D_PGD_EQ(IQUBiMamba1D):
    """BiMamba separator followed by PGD-unfolded Communication Equalization (CMA/MMA) head."""

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
        pgd_eq_num_steps: int = 3,
        pgd_eq_step_size_init: float = 0.5,
        pgd_eq_mu_init: float = 0.1,
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
        self.pgd_eq_head = PGDEqualizationUnfoldedHead(
            num_sources=self.num_sources,
            num_steps=pgd_eq_num_steps,
            step_size_init=pgd_eq_step_size_init,
            eq_step_init=pgd_eq_mu_init,
        )

    def _refine_outputs(self, outputs, mixture: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            outputs[-1] = self.pgd_eq_head(outputs[-1], mixture)
            return outputs
        return self.pgd_eq_head(outputs, mixture)

    def forward(self, x):
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)
