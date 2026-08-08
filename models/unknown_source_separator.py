"""Wrapper that adds source-slot existence estimation to a fixed K_max separator."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn



def _extract_separation_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("separation", "separated", "sources", "output"):
            if key in output:
                return _extract_separation_tensor(output[key])
        raise ValueError("Base separator dictionary has no separation output")
    if isinstance(output, tuple):
        if not output:
            raise ValueError("Base separator returned an empty tuple")
        return _extract_separation_tensor(output[0])
    if isinstance(output, list):
        candidates = [_extract_separation_tensor(value) for value in output]
        return max(candidates, key=lambda tensor: tensor.size(-1))
    raise ValueError(f"Unsupported base separator output type: {type(output).__name__}")


class SlotExistenceHead(nn.Module):
    """Shared waveform classifier applied independently to every source slot."""

    def __init__(self, hidden_channels: int = 32, eps: float = 1e-8) -> None:
        super().__init__()
        hidden_channels = int(hidden_channels)
        self.eps = float(eps)
        self.encoder = nn.Sequential(
            nn.Conv1d(4, hidden_channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=5,
                stride=4,
                padding=2,
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels + 1, 1)

    def forward(self, sources: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if sources.ndim != 4 or sources.size(2) != 2:
            raise ValueError(f"Expected sources [B,K,2,L], got {tuple(sources.shape)}")
        if mixture.ndim != 3 or mixture.size(1) != 2:
            raise ValueError(f"Expected mixture [B,2,L], got {tuple(mixture.shape)}")
        batch, slots, _, length = sources.shape
        mixture_slots = mixture.unsqueeze(1).expand(-1, slots, -1, -1)
        features = torch.cat([sources, mixture_slots], dim=2).reshape(batch * slots, 4, length)
        encoded = self.encoder(features).squeeze(-1)
        log_energy = torch.log(
            sources.float().pow(2).mean(dim=(2, 3)).clamp_min(self.eps)
        ).reshape(batch * slots, 1)
        logits = self.classifier(torch.cat([encoded, log_energy.to(encoded.dtype)], dim=1))
        return logits.reshape(batch, slots)


class UnknownSourceSeparator(nn.Module):
    """Keep the base separator fixed at K_max and predict which slots are active."""

    def __init__(
        self,
        separator: nn.Module,
        max_sources: int = 3,
        existence_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.separator = separator
        self.max_sources = int(max_sources)
        self.existence_head = SlotExistenceHead(hidden_channels=existence_hidden_channels)

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_output = self.separator(mixture)
        separation = _extract_separation_tensor(raw_output)
        expected_channels = 2 * self.max_sources
        if not torch.is_tensor(separation) or separation.ndim != 3:
            raise ValueError("Base separator did not return a [B,2*K_max,L] tensor")
        if separation.size(1) != expected_channels:
            raise ValueError(
                f"Expected base separator output channels={expected_channels}, "
                f"got {tuple(separation.shape)}"
            )
        if separation.size(-1) != mixture.size(-1):
            raise ValueError(
                f"Base separator output length={separation.size(-1)} does not match "
                f"mixture length={mixture.size(-1)}"
            )
        sources = separation.reshape(
            separation.size(0), self.max_sources, 2, separation.size(-1)
        )
        existence_logits = self.existence_head(sources, mixture)
        return {
            "separation": separation,
            "existence_logits": existence_logits,
        }
