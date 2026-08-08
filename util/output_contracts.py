from __future__ import annotations

from collections.abc import Sequence

import torch


def select_finest_separation_output(outputs):
    """Select the waveform tensor with the finest temporal resolution."""
    def _collect(value):
        if torch.is_tensor(value):
            return [value] if value.ndim == 3 else []
        if isinstance(value, dict):
            return [tensor for item in value.values() for tensor in _collect(item)]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [tensor for item in value for tensor in _collect(item)]
        return []

    candidates = _collect(outputs)
    if not candidates:
        raise ValueError("separation output does not contain a [B, C, L] tensor")
    return max(candidates, key=lambda tensor: tensor.shape[-1])
