"""IQUBiMamba1D — Bidirectional Mamba variant of IQUMamba1D.

The only structural change from IQUMamba1D is the replacement of the
unidirectional `MambaLayer` with a `BiMambaLayer` that scans the sequence
in **both** forward and backward directions and fuses the outputs via a
learned linear projection.  This follows the design validated in the
SPMamba paper (arXiv:2404.02063) and DPMamba.

All other components (ResidualMambaEncoder, UNetResDecoder, SkipConnection-
Processor, etc.) are inherited from IQUMamba1D.py without modification.
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import autocast

from mamba_ssm import Mamba

# Re-use all building blocks from the original file
from models.IQUMamba1D import (
    ResidualMambaEncoder,
    UNetResDecoder,
    BasicResBlock,
    SkipConnectionProcessor,
)
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

from typing import Union, Type, List, Tuple
from torch.nn.modules.conv import _ConvNd

if hasattr(torch, "bfloat16"):
    HALF_PRECISION_DTYPES = (torch.float16, torch.bfloat16)
else:
    HALF_PRECISION_DTYPES = (torch.float16,)


# ============================================================================
#  BiMambaLayer — drop-in replacement for MambaLayer
# ============================================================================

class BiMambaLayer(nn.Module):
    """Bidirectional Mamba: forward scan + backward scan → linear fusion.

    Interface is identical to the original ``MambaLayer`` so the encoder
    can use it transparently.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(int(dim))
        self.mamba_fwd = Mamba(
            d_model=int(dim),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=int(dim),
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Fuse two directions back to original dim
        self.out_proj = nn.Linear(int(dim) * 2, int(dim))
        self.channel_token = channel_token

    def _fuse(self, x):
        """Run bidirectional Mamba and fuse.  x: [B, L, D]"""
        x_norm = self.norm(x)
        h_fwd = self.mamba_fwd(x_norm)
        h_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        return self.out_proj(torch.cat([h_fwd, h_bwd], dim=-1)) + x  # residual

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        dims = x.shape[2:]
        n_tokens = dims.numel()
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)  # [B, L, D]
        out = self._fuse(x_flat)
        return out.transpose(-1, -2).reshape(B, d_model, *dims)

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        dims = x.shape[2:]
        x_flat = x.flatten(2)  # [B, L, D]
        out = self._fuse(x_flat)
        return out.reshape(B, n_tokens, *dims)

    @autocast('cuda', enabled=False)
    def forward(self, x):
        if x.dtype in HALF_PRECISION_DTYPES:
            x = x.float()
        if self.channel_token:
            return self.forward_channel_token(x)
        return self.forward_patch_token(x)


# ============================================================================
#  ResidualBiMambaEncoder — identical to ResidualMambaEncoder but uses
#  BiMambaLayer instead of MambaLayer at alternating stages.
# ============================================================================

class ResidualBiMambaEncoder(nn.Module):
    def __init__(self,
                 input_size: Tuple[int, ...],
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        kernel_sizes = [maybe_convert_scalar_to_list(conv_op, ks) for ks in kernel_sizes]
        strides = [maybe_convert_scalar_to_list(conv_op, s) for s in strides]

        features_per_stage = [features_per_stage] * n_stages if isinstance(features_per_stage, int) else features_per_stage
        n_blocks_per_stage = [n_blocks_per_stage] * n_stages if isinstance(n_blocks_per_stage, int) else n_blocks_per_stage
        strides = [strides] * n_stages if isinstance(strides, int) else strides

        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = input_size
        for s in range(n_stages):
            feature_map_sizes.append([i / j for i, j in zip(feature_map_size, strides[s])])
            feature_map_size = feature_map_sizes[-1]
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        self.conv_pad_sizes = [[k // 2 for k in ks] for ks in kernel_sizes]

        stem_channels = features_per_stage[0] if stem_channels is None else int(stem_channels)
        self.stem = nn.Sequential(
            BasicResBlock(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0][0],
                stride=1,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
            ),
            *[BasicBlockD(
                conv_op=conv_op,
                input_channels=stem_channels,
                output_channels=stem_channels,
                kernel_size=kernel_sizes[0],
                stride=1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
            ) for _ in range(n_blocks_per_stage[0] - 1)]
        )

        input_channels = stem_channels
        stages = []
        mamba_layers = []
        for s in range(n_stages):
            stage = nn.Sequential(
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s][0],
                    stride=strides[s][0],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ),
                *[BasicBlockD(
                    conv_op=conv_op,
                    input_channels=features_per_stage[s],
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                ) for _ in range(n_blocks_per_stage[s] - 1)]
            )

            # Key difference: use BiMambaLayer instead of MambaLayer
            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    BiMambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s]
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.stages = nn.ModuleList(stages)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]


# ============================================================================
#  IQUBiMamba1D — top-level model class
# ============================================================================

class IQUBiMamba1D(nn.Module):
    """Bidirectional Mamba U-Net for IQ signal separation.

    Identical to IQUMamba1D but uses ``BiMambaLayer`` (fwd+bwd scan)
    instead of the unidirectional ``MambaLayer``.
    """

    def __init__(self,
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
                 ):
        super().__init__()
        self.encoder = ResidualBiMambaEncoder(
            input_size=(input_size,),
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=[[k] for k in kernel_sizes],
            strides=[[s] for s in strides],
            n_blocks_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
        )
        self.decoder = UNetResDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)


class _ADMMProxBlock1D(nn.Module):
    """Learnable communication-prior proximal operator for one IQ source.

    In the unfolded ADMM step this approximates

        z^{t+1}_k = prox_{R_comm / rho_t}(s^{t+1}_k + u^t_k).

    The residual branch is zero-initialized, so the whole ADMM head starts as
    the mathematically exact identity-prox ADMM update and learns only the
    communication prior correction needed by data.
    """

    def __init__(
        self,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=1, bias=True),
            nn.InstanceNorm1d(hidden_channels, eps=1e-5, affine=True),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=hidden_channels,
                bias=False,
            ),
            nn.InstanceNorm1d(hidden_channels, eps=1e-5, affine=True),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ADMMCommunicationPriorUnfoldedHead(nn.Module):
    """ADMM-unfolded communication-prior separator refinement head.

    The head unfolds ADMM for the constrained separation objective

        min_{S,Z} 0.5 ||sum_k s_k - x_mix||_2^2 + R_comm(Z)
        s.t.      S = Z,

    where S are mixture-consistent source variables and Z are communication-
    prior variables. With scaled dual variables U, one layer performs:

        S^{t+1} = argmin_S 0.5 ||sum_k s_k - x||^2
                         + rho_t/2 sum_k ||s_k - z_k^t + u_k^t||^2
        Z^{t+1} = Prox_comm(S^{t+1} + U^t)
        U^{t+1} = U^t + beta_t (S^{t+1} - Z^{t+1})

    The S-update is solved exactly. Let v_k = z_k^t - u_k^t and V=sum_k v_k:

        s_k^{t+1} = v_k - (V - x_mix) / (rho_t + K).

    This is the closed-form ADMM data-consistency step for a single-channel
    additive IQ mixture, applied independently at every time sample and I/Q
    component.
    """

    def __init__(
        self,
        num_sources: int,
        num_steps: int = 3,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        dropout: float = 0.0,
        tied_steps: bool = True,
        rho_init: float = 1.0,
        dual_step_init: float = 1.0,
        prox_step_init: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if rho_init <= 0:
            raise ValueError(f"rho_init must be > 0, got {rho_init}")
        if dual_step_init <= 0:
            raise ValueError(f"dual_step_init must be > 0, got {dual_step_init}")
        if prox_step_init < 0:
            raise ValueError(f"prox_step_init must be >= 0, got {prox_step_init}")

        self.num_sources = int(num_sources)
        self.num_steps = int(num_steps)
        self.tied_steps = bool(tied_steps)
        self.eps = float(eps)

        if self.tied_steps:
            self.shared_prox = _ADMMProxBlock1D(
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.prox_blocks = None
        else:
            self.shared_prox = None
            self.prox_blocks = nn.ModuleList([
                _ADMMProxBlock1D(
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(self.num_steps)
            ])

        self.rho_logits = nn.Parameter(torch.full(
            (self.num_steps,),
            self._softplus_inverse(float(rho_init)),
        ))
        self.dual_step_logits = nn.Parameter(torch.full(
            (self.num_steps,),
            self._softplus_inverse(float(dual_step_init)),
        ))
        self.prox_step_logits = nn.Parameter(torch.full(
            (self.num_steps,),
            self._softplus_inverse(float(prox_step_init)) if prox_step_init > 0 else -20.0,
        ))

    @staticmethod
    def _softplus_inverse(value: float) -> float:
        return math.log(math.expm1(value)) if value < 20.0 else value

    @staticmethod
    def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        if mixture.size(-1) == target_length:
            return mixture
        return F.interpolate(
            mixture,
            size=target_length,
            mode="linear",
            align_corners=False,
        )

    def _get_prox_block(self, step_idx: int) -> _ADMMProxBlock1D:
        return self.shared_prox if self.tied_steps else self.prox_blocks[step_idx]

    def _positive_scalar(self, logits: torch.Tensor, step_idx: int) -> torch.Tensor:
        return F.softplus(logits[step_idx]) + self.eps

    def _s_update(
        self,
        z: torch.Tensor,
        u: torch.Tensor,
        mixture: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        # z/u: (B, K, 2, L), mixture: (B, 2, L)
        v = z - u
        v_sum = v.sum(dim=1)
        correction = (v_sum - mixture) / (rho + float(self.num_sources))
        return v - correction.unsqueeze(1)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if estimates.dim() != 3:
            raise ValueError(
                f"estimates must have shape (B, 2K, L), got {tuple(estimates.shape)}"
            )
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}"
            )
        if estimates.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} estimate channels, got {estimates.size(1)}"
            )

        b, _, target_length = estimates.shape
        mixture = self._resize_mixture(mixture, target_length)
        z = estimates.reshape(b, self.num_sources, 2, target_length)
        u = torch.zeros_like(z)

        for step_idx in range(self.num_steps):
            rho = self._positive_scalar(self.rho_logits, step_idx)
            dual_step = self._positive_scalar(self.dual_step_logits, step_idx)
            prox_step = self._positive_scalar(self.prox_step_logits, step_idx)

            s = self._s_update(z=z, u=u, mixture=mixture, rho=rho)

            prox_input = (s + u).reshape(b * self.num_sources, 2, target_length)
            prox_delta = self._get_prox_block(step_idx)(prox_input)
            z_next = prox_input + prox_step * prox_delta
            z_next = z_next.reshape(b, self.num_sources, 2, target_length)

            u = u + dual_step * (s - z_next)
            z = z_next

        return z.reshape(b, 2 * self.num_sources, target_length)


class IQUBiMamba1D_ADMM(IQUBiMamba1D):
    """BiMamba separator followed by ADMM-unfolded communication-prior refinement."""

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
        admm_num_steps: int = 3,
        admm_hidden_channels: int = 48,
        admm_kernel_size: int = 7,
        admm_dropout: float = 0.0,
        admm_tied_steps: bool = True,
        admm_rho_init: float = 1.0,
        admm_dual_step_init: float = 1.0,
        admm_prox_step_init: float = 0.25,
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
            raise ValueError(
                f"num_classes must be even for I/Q source pairs, got {num_classes}"
            )
        self.num_sources = num_classes // 2
        self.admm_head = ADMMCommunicationPriorUnfoldedHead(
            num_sources=self.num_sources,
            num_steps=admm_num_steps,
            hidden_channels=admm_hidden_channels,
            kernel_size=admm_kernel_size,
            dropout=admm_dropout,
            tied_steps=admm_tied_steps,
            rho_init=admm_rho_init,
            dual_step_init=admm_dual_step_init,
            prox_step_init=admm_prox_step_init,
        )

    def _refine_outputs(self, outputs, mixture: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            outputs[-1] = self.admm_head(outputs[-1], mixture)
            return outputs
        return self.admm_head(outputs, mixture)

    def forward(self, x):
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)


class PGDCommunicationPriorUnfoldedHead(nn.Module):
    """Proximal-gradient-unfolded communication-prior refinement head.

    The head unfolds proximal gradient descent for

        min_S 0.5 ||sum_k s_k - x_mix||_2^2 + R_comm(S).

    One layer performs a mathematically explicit data-consistency gradient
    step followed by a learned communication-prior proximal correction:

        r^t       = sum_k s_k^t - x_mix
        y_k^t     = s_k^t - eta_t r^t
        s_k^{t+1} = Prox_comm(y_k^t).

    The gradient of the additive mixture data term w.r.t. every source is the
    same residual r^t. Its Lipschitz constant is K for K sources, so eta_t is
    constrained to (0, 2/K) by construction. The learned prox residual is
    zero-initialized, which makes the initial head exactly a stable PGD data
    step before learning communication-domain corrections.
    """

    def __init__(
        self,
        num_sources: int,
        num_steps: int = 3,
        hidden_channels: int = 48,
        kernel_size: int = 7,
        dropout: float = 0.0,
        tied_steps: bool = True,
        step_size_init: float = 0.5,
        prox_step_init: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")

        self.num_sources = int(num_sources)
        self.num_steps = int(num_steps)
        self.tied_steps = bool(tied_steps)
        self.eps = float(eps)
        self.max_step_size = 2.0 / float(self.num_sources)

        if not 0.0 < float(step_size_init) < self.max_step_size:
            raise ValueError(
                f"step_size_init must be in (0, {self.max_step_size:.6g}) "
                f"for {self.num_sources} sources, got {step_size_init}"
            )
        if prox_step_init < 0:
            raise ValueError(f"prox_step_init must be >= 0, got {prox_step_init}")

        if self.tied_steps:
            self.shared_prox = _ADMMProxBlock1D(
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.prox_blocks = None
        else:
            self.shared_prox = None
            self.prox_blocks = nn.ModuleList([
                _ADMMProxBlock1D(
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(self.num_steps)
            ])

        init_ratio = float(step_size_init) / self.max_step_size
        self.step_size_logits = nn.Parameter(torch.full(
            (self.num_steps,),
            self._logit(init_ratio),
        ))
        self.prox_step_logits = nn.Parameter(torch.full(
            (self.num_steps,),
            ADMMCommunicationPriorUnfoldedHead._softplus_inverse(float(prox_step_init))
            if prox_step_init > 0 else -20.0,
        ))

    @staticmethod
    def _logit(value: float) -> float:
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(value / (1.0 - value))

    @staticmethod
    def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        return ADMMCommunicationPriorUnfoldedHead._resize_mixture(mixture, target_length)

    def _get_prox_block(self, step_idx: int) -> _ADMMProxBlock1D:
        return self.shared_prox if self.tied_steps else self.prox_blocks[step_idx]

    def _step_size(self, step_idx: int) -> torch.Tensor:
        return self.max_step_size * torch.sigmoid(self.step_size_logits[step_idx])

    def _prox_step(self, step_idx: int) -> torch.Tensor:
        return F.softplus(self.prox_step_logits[step_idx]) + self.eps

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if estimates.dim() != 3:
            raise ValueError(
                f"estimates must have shape (B, 2K, L), got {tuple(estimates.shape)}"
            )
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}"
            )
        if estimates.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} estimate channels, got {estimates.size(1)}"
            )

        b, _, target_length = estimates.shape
        mixture = self._resize_mixture(mixture, target_length)
        z = estimates.reshape(b, self.num_sources, 2, target_length)

        for step_idx in range(self.num_steps):
            residual = z.sum(dim=1) - mixture
            y = z - self._step_size(step_idx) * residual.unsqueeze(1)

            y_flat = y.reshape(b * self.num_sources, 2, target_length)
            prox_delta = self._get_prox_block(step_idx)(y_flat)
            z = y_flat + self._prox_step(step_idx) * prox_delta
            z = z.reshape(b, self.num_sources, 2, target_length)

        return z.reshape(b, 2 * self.num_sources, target_length)


class IQUBiMamba1D_PGDU(IQUBiMamba1D):
    """BiMamba separator followed by PGD-unfolded communication-prior refinement."""

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
        pgdu_num_steps: int = 3,
        pgdu_hidden_channels: int = 48,
        pgdu_kernel_size: int = 7,
        pgdu_dropout: float = 0.0,
        pgdu_tied_steps: bool = True,
        pgdu_step_size_init: float = 0.5,
        pgdu_prox_step_init: float = 0.25,
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
            raise ValueError(
                f"num_classes must be even for I/Q source pairs, got {num_classes}"
            )
        self.num_sources = num_classes // 2
        self.pgdu_head = PGDCommunicationPriorUnfoldedHead(
            num_sources=self.num_sources,
            num_steps=pgdu_num_steps,
            hidden_channels=pgdu_hidden_channels,
            kernel_size=pgdu_kernel_size,
            dropout=pgdu_dropout,
            tied_steps=pgdu_tied_steps,
            step_size_init=pgdu_step_size_init,
            prox_step_init=pgdu_prox_step_init,
        )

    def _refine_outputs(self, outputs, mixture: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            outputs[-1] = self.pgdu_head(outputs[-1], mixture)
            return outputs
        return self.pgdu_head(outputs, mixture)

    def forward(self, x):
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)


class GainPhaseChannelConsistencyHead(nn.Module):
    """Gain/phase-parameterized channel consistency layer for IQ separation.

    The layer estimates one gain and one phase offset for each separated source
    and interprets the mixture as

        x_mix ~= sum_k gain_k * exp(j phase_k) * s_k.

    It then computes the residual in the observed mixture domain and maps the
    correction back through the inverse gain/phase transform before adding it to
    each source. This injects a differentiable communication-channel prior while
    preserving the ordinary separator contract: input and output are both
    tensors of shape (B, 2K, L).
    """

    def __init__(
        self,
        num_sources: int,
        hidden_channels: int = 32,
        kernel_size: int = 7,
        max_gain_db: float = 12.0,
        max_phase_deg: float = 180.0,
        weight_mode: str = "energy",
        min_weight: float = 1e-3,
        correction_strength_init: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError(f"num_sources must be >= 1, got {num_sources}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if weight_mode not in {"energy", "uniform"}:
            raise ValueError(
                f"Unsupported weight_mode='{weight_mode}'. Use 'energy' or 'uniform'."
            )
        if min_weight < 0:
            raise ValueError(f"min_weight must be >= 0, got {min_weight}")
        if correction_strength_init <= 0:
            raise ValueError(
                f"correction_strength_init must be > 0, got {correction_strength_init}"
            )

        self.num_sources = int(num_sources)
        self.max_gain_log = math.log(10.0 ** (float(max_gain_db) / 20.0))
        self.max_phase_rad = math.radians(float(max_phase_deg))
        self.weight_mode = str(weight_mode)
        self.min_weight = float(min_weight)
        self.eps = float(eps)

        padding = kernel_size // 2
        self.param_head = nn.Sequential(
            nn.Conv1d(2, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv1d(hidden_channels, 2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.param_head[-1].weight)
        nn.init.zeros_(self.param_head[-1].bias)

        self.correction_strength_logit = nn.Parameter(torch.tensor(
            ADMMCommunicationPriorUnfoldedHead._softplus_inverse(float(correction_strength_init)),
            dtype=torch.float32,
        ))
        self.last_gain = None
        self.last_phase = None

    @staticmethod
    def _resize_mixture(mixture: torch.Tensor, target_length: int) -> torch.Tensor:
        return ADMMCommunicationPriorUnfoldedHead._resize_mixture(mixture, target_length)

    @staticmethod
    def _rotate_iq(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # x: (B, K, 2, L), phase: (B, K)
        cos_p = torch.cos(phase).unsqueeze(-1)
        sin_p = torch.sin(phase).unsqueeze(-1)
        i = x[:, :, 0, :]
        q = x[:, :, 1, :]
        return torch.stack((cos_p * i - sin_p * q, sin_p * i + cos_p * q), dim=2)

    def _estimate_params(self, est: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, _, l = est.shape
        raw = self.param_head(est.reshape(b * k, 2, l)).mean(dim=-1)
        raw = raw.reshape(b, k, 2)
        gain = torch.exp(self.max_gain_log * torch.tanh(raw[:, :, 0]))
        phase = self.max_phase_rad * torch.tanh(raw[:, :, 1])
        self.last_gain = gain.detach()
        self.last_phase = phase.detach()
        return gain, phase

    def _compute_weights(self, observed_sources: torch.Tensor) -> torch.Tensor:
        b, k, _, l = observed_sources.shape
        if self.weight_mode == "uniform":
            return observed_sources.new_full((b, k, l), 1.0 / float(k))

        energy = observed_sources.pow(2).sum(dim=2).clamp_min(self.eps)
        scores = energy + self.min_weight
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if estimates.dim() != 3:
            raise ValueError(
                f"estimates must have shape (B, 2K, L), got {tuple(estimates.shape)}"
            )
        if mixture.dim() != 3 or mixture.size(1) != 2:
            raise ValueError(
                f"mixture must have shape (B, 2, L), got {tuple(mixture.shape)}"
            )
        if estimates.size(1) != 2 * self.num_sources:
            raise ValueError(
                f"Expected {2 * self.num_sources} estimate channels, got {estimates.size(1)}"
            )

        b, _, target_length = estimates.shape
        mixture = self._resize_mixture(mixture, target_length)
        est = estimates.reshape(b, self.num_sources, 2, target_length)

        gain, phase = self._estimate_params(est)
        observed = gain.unsqueeze(-1).unsqueeze(-1) * self._rotate_iq(est, phase)
        residual = mixture - observed.sum(dim=1)
        weights = self._compute_weights(observed)
        correction_observed = weights.unsqueeze(2) * residual.unsqueeze(1)

        correction_source = self._rotate_iq(
            correction_observed / gain.unsqueeze(-1).unsqueeze(-1).clamp_min(self.eps),
            -phase,
        )
        strength = F.softplus(self.correction_strength_logit)
        refined = est + strength * correction_source
        return refined.reshape(b, 2 * self.num_sources, target_length)


class IQUBiMamba1D_GainPhase(IQUBiMamba1D):
    """BiMamba separator with gain/phase-parameterized channel consistency."""

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
        gp_hidden_channels: int = 32,
        gp_kernel_size: int = 7,
        gp_max_gain_db: float = 12.0,
        gp_max_phase_deg: float = 180.0,
        gp_weight_mode: str = "energy",
        gp_min_weight: float = 1e-3,
        gp_correction_strength_init: float = 1.0,
        gp_apply_train: bool = True,
        gp_apply_eval: bool = True,
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
            raise ValueError(
                f"num_classes must be even for I/Q source pairs, got {num_classes}"
            )
        self.num_sources = num_classes // 2
        self.gp_apply_train = bool(gp_apply_train)
        self.gp_apply_eval = bool(gp_apply_eval)
        self.gain_phase_head = GainPhaseChannelConsistencyHead(
            num_sources=self.num_sources,
            hidden_channels=gp_hidden_channels,
            kernel_size=gp_kernel_size,
            max_gain_db=gp_max_gain_db,
            max_phase_deg=gp_max_phase_deg,
            weight_mode=gp_weight_mode,
            min_weight=gp_min_weight,
            correction_strength_init=gp_correction_strength_init,
        )

    def _should_apply(self) -> bool:
        return self.gp_apply_train if self.training else self.gp_apply_eval

    def _refine_outputs(self, outputs, mixture: torch.Tensor):
        if not self._should_apply():
            return outputs
        if isinstance(outputs, (list, tuple)):
            outputs = list(outputs)
            outputs[-1] = self.gain_phase_head(outputs[-1], mixture)
            return outputs
        return self.gain_phase_head(outputs, mixture)

    def forward(self, x):
        outputs = super().forward(x)
        return self._refine_outputs(outputs, x)
