"""ICASSP 2024 RF Challenge compatibility layer for IQUMamba1D.

The package implements the public RF Challenge protocol without TensorFlow or
Sionna. It deliberately keeps the RF task separate from the project's
multi-source blind-separation pipeline: an RF Challenge model predicts one
known signal of interest as two I/Q channels.
"""

from .protocol import (
    FRAME_LENGTH,
    INTERFERENCE_TYPES,
    OFFICIAL_CASES,
    SINR_DB_VALUES,
    SOI_TYPES,
)

__all__ = [
    "FRAME_LENGTH",
    "INTERFERENCE_TYPES",
    "OFFICIAL_CASES",
    "SINR_DB_VALUES",
    "SOI_TYPES",
]
