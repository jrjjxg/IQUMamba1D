"""Multi-hypothesis cyclic reliability adapter for stage-4 IQUMamba.

Unlike single-peak cyclic adapters, this module gives the model a null
hypothesis. If the mixture does not provide reliable cyclic evidence, the
softmax can put mass on the null branch and keep the backbone input unchanged.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Type, Union

import torch
from torch import nn

from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_ComplexAdapter import ComplexTiedConv1d


class MultiHypothesisCyclicReliabilityAdapter1D(nn.Module):
    """FRESH-style adapter with candidate cyclic frequencies plus null fallback."""

    def __init__(
        self,
        input_channels: int,
        freqs: Sequence[float] = (0.015625, 0.03125, 0.0625, 0.125),
        hidden_channels: int = 8,
        kernel_size: int = 9,
        scale_init: float = 0.01,
        gate_hidden: int = 8,
        temperature: float = 0.5,
        null_logit_init: float = 2.0,
        local_bins: int = 5,
        zero_init: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if input_channels != 2:
            raise ValueError(f"MultiHypothesisCyclicReliabilityAdapter1D expects one I/Q mixture with 2 channels, got {input_channels}")
        freq_values = [float(freq) for freq in freqs]
        if len(freq_values) < 1:
            raise ValueError("freqs must contain at least one cyclic-frequency candidate")

        self.num_hypotheses = len(freq_values)
        self.temperature = max(float(temperature), 1e-3)
        self.local_bins = max(1, int(local_bins))
        self.eps = float(eps)
        self.register_buffer("candidate_freqs", torch.tensor(freq_values, dtype=torch.float32))
        self.null_logit = nn.Parameter(torch.tensor(float(null_logit_init)))

        hidden_channels = max(1, int(hidden_channels))
        gate_hidden = max(1, int(gate_hidden))
        self.branch_filter = ComplexTiedConv1d(
            in_complex_channels=self.num_hypotheses,
            out_complex_channels=hidden_channels,
            kernel_size=kernel_size,
            bias=True,
        )
        self.out_proj = ComplexTiedConv1d(
            in_complex_channels=hidden_channels,
            out_complex_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )
        self.gate = nn.Sequential(
            nn.Conv1d(2 * self.num_hypotheses, gate_hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(gate_hidden, hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        if zero_init:
            nn.init.zeros_(self.out_proj.real.weight)
            nn.init.zeros_(self.out_proj.imag.weight)
            if self.out_proj.bias_real is not None:
                nn.init.zeros_(self.out_proj.bias_real)
                nn.init.zeros_(self.out_proj.bias_imag)

    def _envelope_power_spectrum(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        envelope_power = x[:, 0].square() + x[:, 1].square()
        envelope_power = envelope_power - envelope_power.mean(dim=-1, keepdim=True)
        spectrum = torch.fft.rfft(envelope_power.float(), dim=-1)
        power = spectrum.abs().square()
        freqs = torch.fft.rfftfreq(x.size(-1), d=1.0).to(device=x.device)
        return power.to(device=x.device), freqs

    def _candidate_indices(self, freqs: torch.Tensor) -> torch.Tensor:
        candidates = self.candidate_freqs.to(device=freqs.device, dtype=freqs.dtype).abs().clamp(0.0, 0.5)
        return torch.argmin(torch.abs(freqs.view(1, -1) - candidates.view(-1, 1)), dim=-1)

    def _reliability_logits(self, x: torch.Tensor) -> torch.Tensor:
        power, freqs = self._envelope_power_spectrum(x)
        indices = self._candidate_indices(freqs)
        reliability_list = []
        for idx_tensor in indices:
            idx = int(idx_tensor.item())
            start = max(1, idx - self.local_bins)
            stop = min(power.size(-1), idx + self.local_bins + 1)
            local_power = power[:, start:stop]
            floor = local_power.mean(dim=-1).clamp_min(self.eps)
            peak = power[:, idx].clamp_min(self.eps)
            reliability_list.append(torch.log(peak / floor))
        reliability_logits = torch.stack(reliability_list, dim=-1)
        return reliability_logits

    def _hypothesis_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        reliability_logits = self._reliability_logits(x)
        null_logits = self.null_logit.to(device=x.device, dtype=reliability_logits.dtype).expand(x.size(0), 1)
        logits = torch.cat([null_logits, reliability_logits / self.temperature], dim=-1)
        hypothesis_weights = torch.softmax(logits, dim=-1)
        return hypothesis_weights.to(dtype=x.dtype), reliability_logits.to(dtype=x.dtype)

    def _phasors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.size(-1)
        n = torch.arange(length, device=x.device, dtype=torch.float32)
        freqs = self.candidate_freqs.to(device=x.device, dtype=torch.float32)
        phase = -2.0 * math.pi * freqs.unsqueeze(1) * n.unsqueeze(0)
        cos = torch.cos(phase).to(dtype=x.dtype).unsqueeze(0)
        sin = torch.sin(phase).to(dtype=x.dtype).unsqueeze(0)
        return cos, sin

    def _shift_branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real = x[:, 0:1, :]
        imag = x[:, 1:2, :]
        cos, sin = self._phasors(x)
        shifted_real = real * cos - imag * sin
        shifted_imag = real * sin + imag * cos
        return shifted_real, shifted_imag

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.dim() != 3 or x.size(1) != 2:
            raise ValueError(f"Expected raw I/Q mixture with shape (B, 2, L), got {tuple(x.shape)}")

        hypothesis_weights, reliability_logits = self._hypothesis_weights(x.detach())
        candidate_weights = hypothesis_weights[:, 1:].unsqueeze(-1)
        shifted_real, shifted_imag = self._shift_branches(x)
        shifted_real = shifted_real * candidate_weights
        shifted_imag = shifted_imag * candidate_weights
        hidden_real, hidden_imag = self.branch_filter(shifted_real, shifted_imag)
        gate_input = torch.cat([shifted_real, shifted_imag], dim=1)
        gate = self.gate(gate_input)
        hidden_real = hidden_real * gate
        hidden_imag = hidden_imag * gate
        delta_real, delta_imag = self.out_proj(hidden_real, hidden_imag)
        cyclic_delta = torch.cat([delta_real, delta_imag], dim=1)
        adapted = x + self.scale * (1.0 - hypothesis_weights[:, 0:1].unsqueeze(-1)) * cyclic_delta
        aux = {
            "hypothesis_weights": hypothesis_weights,
            "reliability_logits": reliability_logits,
            "cyclic_delta": cyclic_delta,
            "candidate_freqs": self.candidate_freqs.to(device=x.device, dtype=x.dtype),
        }
        return adapted, aux


class IQUMamba1D_MultiHypCyclicReliability(nn.Module):
    """Stage-4 IQUMamba with null-aware multi-hypothesis cyclic input adapter."""

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
        norm_op_kwargs: dict = {"eps": 1e-5, "affine": True},
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {"inplace": True},
        deep_supervision: bool = False,
        multihyp_cyclic_freqs: Sequence[float] = (0.015625, 0.03125, 0.0625, 0.125),
        multihyp_cyclic_hidden_channels: int = 8,
        multihyp_cyclic_kernel_size: int = 9,
        multihyp_cyclic_scale_init: float = 0.01,
        multihyp_cyclic_gate_hidden: int = 8,
        multihyp_cyclic_temperature: float = 0.5,
        multihyp_cyclic_null_logit_init: float = 2.0,
        multihyp_cyclic_local_bins: int = 5,
        multihyp_cyclic_zero_init: bool = True,
        multihyp_cyclic_return_aux: bool = False,
        multihyp_cyclic_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.adapter = MultiHypothesisCyclicReliabilityAdapter1D(
            input_channels=input_channels,
            freqs=multihyp_cyclic_freqs,
            hidden_channels=multihyp_cyclic_hidden_channels,
            kernel_size=multihyp_cyclic_kernel_size,
            scale_init=multihyp_cyclic_scale_init,
            gate_hidden=multihyp_cyclic_gate_hidden,
            temperature=multihyp_cyclic_temperature,
            null_logit_init=multihyp_cyclic_null_logit_init,
            local_bins=multihyp_cyclic_local_bins,
            zero_init=multihyp_cyclic_zero_init,
            eps=multihyp_cyclic_eps,
        )
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
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
        )
        self.multihyp_cyclic_return_aux = bool(multihyp_cyclic_return_aux)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, List[torch.Tensor], tuple[Union[torch.Tensor, List[torch.Tensor]], dict[str, torch.Tensor]]]:
        adapted, aux = self.adapter(x)
        outputs = self.backbone(adapted)
        if self.multihyp_cyclic_return_aux:
            return outputs, aux
        return outputs
