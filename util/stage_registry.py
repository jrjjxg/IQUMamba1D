from __future__ import annotations


def supported_stage_ids() -> list[int]:
    """Return stage IDs accepted by the command-line interface."""
    standard_stages = list(range(2, 209)) + list(range(210, 257))
    # Stage 358 is reserved by the RF-Challenge-only RF-Demucs adapter and has
    # no standard training config in main.py.  Keep it out of the CLI choices.
    extended_stages = list(range(257, 358)) + list(range(359, 398))
    sequence_length_aliases = [8192, 16384, 32768]
    return standard_stages + extended_stages + sequence_length_aliases
