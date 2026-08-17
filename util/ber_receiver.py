"""Protocol-aware BER receiver for the IQUMamba datasets.

The private IQUMamba generators save pulse-shaped complex baseband streams and
the transmitted bits in separate MATLAB v7.3 files.  This module reconstructs
the digital receiver needed after source separation:

    I/Q -> CFO removal -> RRC matched filter -> timing search
       -> reference-assisted carrier/gain tracking -> hard demapper -> BER

The default evaluator is deliberately called ``reference-assisted``.  A
blind receiver cannot resolve the absolute PSK phase ambiguity without a
preamble, pilot, or other framing information.  The clean target and its known
labels are therefore used to choose the timing grid and complex reference
gain; the separated waveform is still hard-demapped against the exact protocol
constellation before the final bit comparison.  Results from this module must
retain that label when reported.

Public RML2016/RML2018/TorchSig files in this repository contain IQ snapshots
and modulation/SNR labels, but not transmitter bit streams.  The evaluator
returns an explicit ``bits_unavailable`` status for those datasets unless a
separate bit sidecar is supplied.

This file has no torch dependency so it can be used from a small standalone
audit process as well as from the training code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12
TIMING_SCORE_MAX_SYMBOLS = 4096


class BitsUnavailable(RuntimeError):
    """Raised when a strict BER comparison has no transmitter bit labels."""


@dataclass(frozen=True)
class DatasetSpec:
    """Static protocol information used when a file has no metadata field."""

    name: str
    modulations: Tuple[str, ...]
    sample_rate_hz: Optional[float] = None
    sps_by_source: Tuple[Optional[int], ...] = ()
    public: bool = False
    description: str = ""


def _private(name: str, modulations: Sequence[str], fs_hz: float, sps: Sequence[int]) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        modulations=tuple(modulations),
        sample_rate_hz=float(fs_hz),
        sps_by_source=tuple(int(value) for value in sps),
        description="MATLAB-generated private protocol with file_bits sidecars",
    )


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "8PSK-A": _private("8PSK-A", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "8PSK-B": _private("8PSK-B", ("8PSK", "8PSK"), 50e6, (20, 10)),
    "8PSK-C": _private("8PSK-C", ("8PSK", "8PSK", "8PSK"), 100e6, (20, 20, 20)),
    "8PSK-D": _private("8PSK-D", ("8PSK", "8PSK", "8PSK"), 50e6, (10, 14, 25)),
    "8PSK-E": _private("8PSK-E", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "8PSK-F": _private("8PSK-F", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "8PSK-G": _private("8PSK-G", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "8PSK-H": _private("8PSK-H", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "8PSK-I": _private("8PSK-I", ("8PSK", "8PSK"), 50e6, (20, 10)),
    "8PSK-J": _private("8PSK-J", ("8PSK", "8PSK", "8PSK"), 100e6, (20, 20, 20)),
    "8PSK-K": _private("8PSK-K", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "8PSK-L": _private("8PSK-L", ("8PSK", "8PSK"), 100e6, (20, 20)),
    "QPSK+16APSK-A": _private(
        "QPSK+16APSK-A", ("QPSK", "16APSK"), 100e6, (20, 20)
    ),
    "QPSK+16APSK-B": _private(
        "QPSK+16APSK-B", ("QPSK", "16APSK"), 100e6, (20, 20)
    ),
    "QAM-A": _private("QAM-A", ("16QAM", "64QAM"), 100e6, (20, 20)),
    "QAM-B": _private("QAM-B", ("64QAM", "64QAM"), 100e6, (20, 20)),
    "QAM-C": _private("QAM-C", ("64QAM", "128QAM"), 100e6, (20, 20)),
    "QAM-D": _private("QAM-D", ("16QAM", "64QAM", "128QAM"), 100e6, (20, 20, 20)),
    "QAM-E": _private("QAM-E", ("16QAM", "64QAM", "128QAM"), 100e6, (20, 20, 20)),
    "RML2016": DatasetSpec(
        "RML2016", ("BPSK", "QPSK"), public=True,
        description="Public IQ snapshots; the repository loader has no transmitter bits",
    ),
    "RML2018": DatasetSpec(
        "RML2018", ("BPSK", "QPSK"), public=True,
        description="Public IQ snapshots; the repository loader has no transmitter bits",
    ),
    "TorchSig": DatasetSpec(
        "TorchSig", ("BPSK", "QPSK"), public=True,
        description="Public IQ snapshots; the repository loader has no transmitter bits",
    ),
}


_DATASET_ALIASES = {
    "2016": "RML2016",
    "RML-2016": "RML2016",
    "2018": "RML2018",
    "RML-2018": "RML2018",
    "TORCH-SIG": "TorchSig",
    "QPSK16APSK-A": "QPSK+16APSK-A",
    "QPSK-16APSK-A": "QPSK+16APSK-A",
    "QPSK16APSK-B": "QPSK+16APSK-B",
    "QPSK-16APSK-B": "QPSK+16APSK-B",
}


def canonical_dataset_name(name: str) -> str:
    """Return the registry key for a dataset or raise a useful error."""
    raw = str(name).strip().replace("_", "-")
    if raw.upper() == "TORCHSIG":
        raw = "TorchSig"
    upper = raw.upper()
    alias = _DATASET_ALIASES.get(upper, _DATASET_ALIASES.get(raw, raw))
    if alias == "TorchSig":
        return alias
    if alias in DATASET_REGISTRY:
        return alias
    if upper in DATASET_REGISTRY:
        return upper
    raise ValueError(
        f"Unknown dataset '{name}'. Supported names: "
        + ", ".join(sorted(DATASET_REGISTRY))
    )


def dataset_spec(name: str) -> DatasetSpec:
    return DATASET_REGISTRY[canonical_dataset_name(name)]


def _gray_codes(bits: int) -> np.ndarray:
    return np.asarray([index ^ (index >> 1) for index in range(1 << bits)], dtype=np.int64)


def _psk_constellation(order: int, phase_offset: float, label_to_natural: Sequence[int]) -> Tuple[np.ndarray, int]:
    bits = int(round(math.log2(order)))
    natural = np.exp(1j * (float(phase_offset) + 2.0 * np.pi * np.arange(order) / order))
    return natural[np.asarray(label_to_natural, dtype=np.int64)], bits


def _qam_constellation(order: int) -> Tuple[np.ndarray, int]:
    """Match MATLAB qammod geometry used by the private generator.

    MATLAB's mapping in the saved files has ascending Gray levels on I and the
    reversed levels on Q.  The latter detail is easy to miss because a
    symmetric QAM plot looks identical under a vertical reflection.
    """
    bits = int(round(math.log2(order)))
    if 1 << bits != int(order):
        raise ValueError(f"QAM order must be a power of two, got {order}")
    i_bits = (bits + 1) // 2
    q_bits = bits // 2
    i_count = 1 << i_bits
    q_count = 1 << q_bits
    i_levels = np.arange(-(i_count - 1), i_count, 2, dtype=np.float64)
    q_levels = np.arange(-(q_count - 1), q_count, 2, dtype=np.float64)[::-1]
    avg_power = float(np.mean(i_levels ** 2) + np.mean(q_levels ** 2))
    i_levels /= math.sqrt(avg_power)
    q_levels /= math.sqrt(avg_power)

    symbols = np.zeros(order, dtype=np.complex128)
    i_gray = _gray_codes(i_bits)
    q_gray = _gray_codes(q_bits)
    for i_index, i_label in enumerate(i_gray):
        for q_index, q_label in enumerate(q_gray):
            label = (int(i_label) << q_bits) | int(q_label)
            symbols[label] = i_levels[i_index] + 1j * q_levels[q_index]
    return symbols, bits


def _qam128_constellation() -> Tuple[np.ndarray, int]:
    """MATLAB 128-QAM cross constellation and its 7-bit label order.

    ``qammod`` uses the 12-by-12 cross for the odd-bit order 128.  The 16
    corner points for which both coordinates have magnitude 9 or 11 are
    removed.  The table below is the binary-label order observed in the
    generator output: each group of eight labels follows the same Q-axis
    Gray order, while the group selects the corresponding cross-QAM I strip.
    """
    q_outer = np.asarray([9, 11, 9, 11, -9, -11, -9, -11], dtype=np.float64)
    q_inner = np.asarray([7, 5, 1, 3, -7, -5, -1, -3], dtype=np.float64)
    outer_groups = {
        0: (-7, -1),
        1: (-5, -3),
        8: (7, 1),
        9: (5, 3),
    }
    inner_groups = {
        2: -9, 3: -11, 4: -1, 5: -3, 6: -7, 7: -5,
        10: 9, 11: 11, 12: 1, 13: 3, 14: 7, 15: 5,
    }
    coordinates = np.zeros((128, 2), dtype=np.float64)
    for group, (first_i, second_i) in outer_groups.items():
        for offset in range(8):
            i_value = first_i if offset in (0, 1, 4, 5) else second_i
            coordinates[8 * group + offset] = (i_value, q_outer[offset])
    for group, i_value in inner_groups.items():
        for offset in range(8):
            coordinates[8 * group + offset] = (i_value, q_inner[offset])

    if np.any((np.abs(coordinates[:, 0]) >= 9) & (np.abs(coordinates[:, 1]) >= 9)):
        raise AssertionError("Invalid 128-QAM cross constellation corner")
    if len({tuple(row) for row in coordinates}) != 128:
        raise AssertionError("128-QAM label table contains duplicate points")
    scale = math.sqrt(float(np.mean(np.sum(coordinates ** 2, axis=1))))
    return coordinates[:, 0] / scale + 1j * coordinates[:, 1] / scale, 7


def _apsk16_constellation() -> Tuple[np.ndarray, int]:
    r1 = 1.0
    r2 = 2.85
    outer_angles = np.asarray([
        3 * np.pi / 12, 21 * np.pi / 12, 9 * np.pi / 12, 15 * np.pi / 12,
        np.pi / 12, 23 * np.pi / 12, 11 * np.pi / 12, 13 * np.pi / 12,
        5 * np.pi / 12, 19 * np.pi / 12, 7 * np.pi / 12, 17 * np.pi / 12,
    ])
    inner_angles = np.asarray([np.pi / 4, 7 * np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4])
    symbols = np.concatenate([
        r2 * np.exp(1j * outer_angles),
        r1 * np.exp(1j * inner_angles),
    ]).astype(np.complex128)
    avg_power = (4 * r1 ** 2 + 12 * r2 ** 2) / 16.0
    symbols /= math.sqrt(avg_power)
    return symbols, 4


@lru_cache(maxsize=32)
def constellation_for_modulation(modulation: str) -> Tuple[np.ndarray, int]:
    """Return ``(symbols_by_bit_label, bits_per_symbol)``."""
    key = str(modulation).upper().strip().replace("_", "-")
    base = key.split("+")[0].split("-")[0]
    if key in {"MATLAB-QPSK", "QPSK-MATLAB"}:
        value = _psk_constellation(4, np.pi / 4, (0, 1, 3, 2))
    elif key in {"MATLAB-16APSK", "16APSK-MATLAB"}:
        value = _apsk16_constellation()
    elif base == "8PSK":
        value = _psk_constellation(8, np.pi / 8, (0, 1, 3, 2, 7, 6, 4, 5))
    elif base == "QPSK":
        # The private generator uses [pi/4, 3pi/4, 7pi/4, 5pi/4].
        value = (
            np.exp(1j * np.asarray([np.pi / 4, 3 * np.pi / 4, 7 * np.pi / 4, 5 * np.pi / 4])),
            2,
        )
    elif base == "BPSK":
        value = (np.asarray([1.0 + 0j, -1.0 + 0j]), 1)
    elif base == "128QAM":
        value = _qam128_constellation()
    elif base in {"16QAM", "64QAM", "256QAM"}:
        value = _qam_constellation(int(base.replace("QAM", "")))
    elif base == "16APSK":
        value = _apsk16_constellation()
    else:
        raise ValueError(f"Unsupported modulation '{modulation}'")
    return np.asarray(value[0], dtype=np.complex128), int(value[1])


def bits_to_labels(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Convert an MSB-first flat bit vector to integer symbol labels."""
    raw = np.asarray(bits, dtype=np.uint8).reshape(-1)
    width = int(bits_per_symbol)
    count = len(raw) // width
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    matrix = raw[:count * width].reshape(count, width).astype(np.int64)
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return matrix @ weights


def labels_to_bits(labels: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Convert integer symbol labels to an MSB-first flat bit vector."""
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    width = int(bits_per_symbol)
    output = np.zeros((len(values), width), dtype=np.uint8)
    for bit in range(width):
        output[:, bit] = (values >> (width - 1 - bit)) & 1
    return output.reshape(-1)


def hard_demodulate(symbols: np.ndarray, modulation: str) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour demodulation; returns labels and flat bits."""
    constellation, bits_per_symbol = constellation_for_modulation(modulation)
    samples = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    labels = np.argmin(np.abs(samples[:, None] - constellation[None, :]), axis=1).astype(np.int64)
    return labels, labels_to_bits(labels, bits_per_symbol)


@lru_cache(maxsize=64)
def rrc_taps(alpha: float = 0.35, span: int = 20, sps: int = 20) -> np.ndarray:
    """Unit-energy root-raised-cosine taps matching MATLAB rcosdesign sqrt."""
    rolloff = float(alpha)
    span_int = int(span)
    sps_int = int(sps)
    if not 0.0 <= rolloff <= 1.0:
        raise ValueError(f"RRC roll-off must be in [0, 1], got {alpha}")
    if span_int <= 0 or sps_int <= 0:
        raise ValueError("RRC span and samples-per-symbol must be positive")
    time = np.arange(-span_int * sps_int / 2, span_int * sps_int / 2 + 1, dtype=np.float64)
    time /= float(sps_int)
    taps = np.zeros_like(time)
    for index, value in enumerate(time):
        if abs(value) < 1e-12:
            taps[index] = 1.0 + rolloff * (4.0 / np.pi - 1.0)
        elif rolloff > 0 and np.isclose(abs(value), 1.0 / (4.0 * rolloff), atol=1e-12):
            taps[index] = (
                rolloff / np.sqrt(2.0)
                * (
                    (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff))
                    + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
                )
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - rolloff))
                + 4.0 * rolloff * value * np.cos(np.pi * value * (1.0 + rolloff))
            )
            denominator = np.pi * value * (1.0 - (4.0 * rolloff * value) ** 2)
            taps[index] = numerator / denominator if abs(denominator) > 1e-12 else 0.0
    taps /= np.sqrt(np.sum(taps ** 2) + EPS)
    taps.setflags(write=False)
    return taps


def matched_filter(signal: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Same-length complex convolution using an FFT for long frames."""
    value = np.asarray(signal, dtype=np.complex128).reshape(-1)
    if len(value) == 0:
        return value.copy()
    kernel = np.asarray(taps, dtype=np.float64).reshape(-1)
    if len(kernel) == 1:
        return value * kernel[0]
    full_length = len(value) + len(kernel) - 1
    fft_length = 1 << (full_length - 1).bit_length()
    full = np.fft.ifft(np.fft.fft(value, fft_length) * np.fft.fft(kernel, fft_length))
    start = (len(kernel) - 1) // 2
    return np.asarray(full[start:start + len(value)], dtype=np.complex128)


def _nearest_labels(symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    return np.argmin(np.abs(values[:, None] - constellation[None, :]), axis=1).astype(np.int64)


def _blind_constellation_evm(symbols: np.ndarray, constellation: np.ndarray) -> float:
    """Decision EVM used only for timing-grid selection, never for BER choice."""
    values = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    if len(values) < 4 or not np.all(np.isfinite(values)):
        return float("inf")
    if len(values) > TIMING_SCORE_MAX_SYMBOLS:
        sample_index = np.linspace(
            0, len(values) - 1, TIMING_SCORE_MAX_SYMBOLS, dtype=np.int64
        )
        values = values[sample_index]
    if np.ptp(np.abs(constellation)) <= 0.05 * max(np.mean(np.abs(constellation)), EPS):
        power = float(np.mean(np.abs(values) ** 2))
        if power <= EPS:
            return float("inf")
        return float(np.var(np.abs(values)) / (power + EPS))

    gain = complex(np.sqrt(np.mean(np.abs(values) ** 2) / (np.mean(np.abs(constellation) ** 2) + EPS)))
    if abs(gain) <= EPS:
        return float("inf")
    for _ in range(4):
        decisions = constellation[_nearest_labels(values / gain, constellation)]
        den = np.sum(np.abs(decisions) ** 2)
        if abs(den) <= EPS:
            break
        gain = complex(np.sum(values * np.conj(decisions)) / den)
    if abs(gain) <= EPS:
        return float("inf")
    decisions = constellation[_nearest_labels(values / gain, constellation)]
    return float(np.mean(np.abs(values / gain - decisions) ** 2) / (np.mean(np.abs(decisions) ** 2) + EPS))


def _complex_alignment_error(pred: np.ndarray, target: np.ndarray, block_symbols: int) -> float:
    """Reference waveform error after blockwise complex carrier/gain tracking."""
    x = np.asarray(pred, dtype=np.complex128).reshape(-1)
    y = np.asarray(target, dtype=np.complex128).reshape(-1)
    count = min(len(x), len(y))
    if count < 2:
        return float("inf")
    x = x[:count]
    y = y[:count]
    block = max(8, int(block_symbols))
    errors = []
    for start in range(0, count, block):
        stop = min(count, start + block)
        xb = x[start:stop]
        yb = y[start:stop]
        den = float(np.sum(np.abs(xb) ** 2))
        if den <= EPS:
            continue
        gain = np.sum(yb * np.conj(xb)) / den
        scale = float(np.mean(np.abs(yb) ** 2) + EPS)
        errors.append(float(np.mean(np.abs(gain * xb - yb) ** 2) / scale))
    return float(np.mean(errors)) if errors else float("inf")


def _select_target_offset(
    matched: np.ndarray,
    sps: int,
    constellation: np.ndarray,
    n_symbols: int,
    guard_symbols: int,
    reference_labels: Optional[np.ndarray] = None,
) -> Optional[int]:
    candidates: List[Tuple[float, float, float, int]] = []
    for offset in range(int(sps)):
        values = matched[offset::int(sps)][:n_symbols]
        # A file can contain a few more transmitted symbols than fit in the
        # stored sample stream because MATLAB rounds symbols_per_frame before
        # slicing the continuous waveform into frames.  Keep every offset
        # with a useful symbol run and let the final comparison use the
        # available prefix.
        if len(values) < max(4, int(0.5 * n_symbols)):
            continue
        guard = min(int(guard_symbols), max(0, (len(values) - 4) // 2))
        # The generator slices a continuous stream into 4096-sample frames;
        # 4096 is not a multiple of the common 20 SPS grid.  When labels are
        # available, the reference MSE below resolves this phase of the eye;
        # power/EVM remain fallbacks for callers that only provide waveforms.
        edge = max(guard, min(16, len(values) // 20))
        body = values[edge:len(values) - edge or None]
        power_score = -float(np.mean(np.abs(body) ** 2))
        evm_score = _blind_constellation_evm(body, constellation)
        reference_score = float("inf")
        if reference_labels is not None:
            label_count = min(len(values), len(reference_labels))
            if label_count >= 4:
                reference_values = values[:label_count]
                reference_ideal = constellation[
                    np.asarray(reference_labels[:label_count], dtype=np.int64)
                ]
                reference_guard = min(guard, max(0, (label_count - 4) // 2))
                if reference_guard:
                    reference_values = reference_values[reference_guard:-reference_guard]
                    reference_ideal = reference_ideal[reference_guard:-reference_guard]
                gain = _estimate_gain_to_ideal(reference_values, reference_ideal)
                reference_score = float(
                    np.mean(np.abs(reference_values / gain - reference_ideal) ** 2)
                    / (np.mean(np.abs(reference_ideal) ** 2) + EPS)
                )
        candidates.append((reference_score, evm_score, power_score, offset))
    if not candidates:
        return None
    return int(min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[3])


def _search_reference_timing(
    pred_mf: np.ndarray,
    target_mf: np.ndarray,
    sps: int,
    target_offset: int,
    n_symbols: int,
    max_lag_samples: int,
    block_symbols: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Find an integer sample lag using waveform correlation, not bit errors."""
    best: Optional[Tuple[Tuple[float, int, int, int], np.ndarray, np.ndarray, int]] = None
    for lag in range(-int(max_lag_samples), int(max_lag_samples) + 1):
        pred_start = int(target_offset) + lag
        if pred_start >= len(pred_mf):
            continue
        k_start = max(0, int(math.ceil(-pred_start / float(sps))))
        pred_first = pred_start + k_start * int(sps)
        target_first = int(target_offset) + k_start * int(sps)
        if pred_first < 0 or target_first < 0:
            continue
        available_pred = 1 + (len(pred_mf) - 1 - pred_first) // int(sps)
        available_target = 1 + (len(target_mf) - 1 - target_first) // int(sps)
        count = min(n_symbols - k_start, available_pred, available_target)
        if count < 4:
            continue
        pred_symbols = pred_mf[pred_first:pred_first + count * int(sps):int(sps)]
        target_symbols = target_mf[target_first:target_first + count * int(sps):int(sps)]
        score = _complex_alignment_error(pred_symbols, target_symbols, block_symbols)
        metric = (score, abs(lag), lag, -count)
        if best is None or metric < best[0]:
            best = (metric, pred_symbols, target_symbols, k_start)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _estimate_gain_to_ideal(samples: np.ndarray, ideal: np.ndarray) -> complex:
    den = float(np.sum(np.abs(ideal) ** 2))
    if den <= EPS:
        return 1.0 + 0j
    gain = np.sum(samples * np.conj(ideal)) / den
    return complex(gain) if abs(gain) > EPS else 1.0 + 0j


def _estimate_gain_to_reference(pred: np.ndarray, target: np.ndarray) -> complex:
    den = float(np.sum(np.abs(pred) ** 2))
    if den <= EPS:
        return 1.0 + 0j
    gain = np.sum(target * np.conj(pred)) / den
    return complex(gain) if abs(gain) > EPS else 1.0 + 0j


def _decode_reference_aligned(
    pred_symbols: np.ndarray,
    target_symbols: np.ndarray,
    labels: np.ndarray,
    constellation: np.ndarray,
    bits_per_symbol: int,
    block_symbols: int,
) -> Tuple[int, int]:
    """Track target-referenced gain and demap against the exact constellation.

    The clean target supplies a complex gain estimate, which removes the
    source-separation scale/phase ambiguity.  Hard decisions still use the
    protocol constellation and the transmitted labels are used only for the
    final BER comparison.  This keeps the reference-assisted evaluation from
    turning each block's target labels into an oracle decision table.
    """
    count = min(len(pred_symbols), len(target_symbols), len(labels))
    if count <= 0:
        return 0, 0
    pred_symbols = np.asarray(pred_symbols[:count], dtype=np.complex128)
    target_symbols = np.asarray(target_symbols[:count], dtype=np.complex128)
    labels = np.asarray(labels[:count], dtype=np.int64)
    errors = 0
    compared = 0
    block = max(8, int(block_symbols))
    for start in range(0, count, block):
        stop = min(count, start + block)
        pred_block = pred_symbols[start:stop]
        target_block = target_symbols[start:stop]
        label_block = labels[start:stop]
        prediction_gain = _estimate_gain_to_reference(pred_block, target_block)
        target_ideal = constellation[label_block]
        target_gain = _estimate_gain_to_ideal(target_block, target_ideal)
        aligned = prediction_gain * pred_block / target_gain
        decoded = _nearest_labels(aligned, constellation)
        decoded_bits = labels_to_bits(decoded, bits_per_symbol)
        expected_bits = labels_to_bits(label_block, bits_per_symbol)
        errors += int(np.count_nonzero(decoded_bits != expected_bits))
        compared += int(len(expected_bits))
    return errors, compared


def _as_bit_frames(bits: np.ndarray, frame_count: int, bits_per_frame: Optional[int]) -> np.ndarray:
    raw = np.asarray(bits, dtype=np.uint8)
    if raw.ndim == 2:
        if raw.shape[0] == frame_count:
            return raw
        if raw.shape[1] == frame_count:
            return raw.T
    flat = raw.reshape(-1)
    if bits_per_frame is None:
        if frame_count <= 0 or len(flat) % frame_count != 0:
            raise BitsUnavailable(
                f"Cannot split {len(flat)} bits into {frame_count} frames without bits_per_frame"
            )
        bits_per_frame = len(flat) // frame_count
    expected = int(frame_count) * int(bits_per_frame)
    if len(flat) != expected:
        raise BitsUnavailable(f"Expected {expected} bits, found {len(flat)}")
    return flat.reshape(frame_count, int(bits_per_frame))


@dataclass
class BERCount:
    errors: int = 0
    compared_bits: int = 0
    active_frames: int = 0
    skipped_frames: int = 0

    @property
    def ber(self) -> float:
        return float(self.errors / self.compared_bits) if self.compared_bits else float("nan")

    def add(self, errors: int, compared_bits: int) -> None:
        self.errors += int(errors)
        self.compared_bits += int(compared_bits)


def reference_ber_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    bits: np.ndarray,
    modulation: str,
    sps: int,
    sample_rate_hz: float,
    cfo_hz: float = 0.0,
    rrc_alpha: float = 0.35,
    rrc_span: int = 20,
    max_lag_samples: int = 32,
    block_symbols: int = 128,
    guard_symbols: Optional[int] = None,
    active_threshold: float = 1e-8,
    max_frames: Optional[int] = None,
) -> BERCount:
    """Compute reference-assisted BER for arrays shaped ``(B, 2, L)``.

    ``pred`` and ``target`` may also be shaped ``(2, L)`` for one frame.
    Aggregation is by compared bit count, so unequal frame/source payload
    lengths are handled without an unweighted frame average.
    """
    constellation, bits_per_symbol = constellation_for_modulation(modulation)
    pred_frames = _normalise_frames(pred, channels=2)
    target_frames = _normalise_frames(target, channels=2)
    if pred_frames.shape[0] != target_frames.shape[0]:
        raise ValueError(f"pred/target frame count mismatch: {pred_frames.shape} vs {target_frames.shape}")
    bit_frames = _as_bit_frames(bits, target_frames.shape[0], None)
    count = BERCount()
    filter_taps = rrc_taps(rrc_alpha, rrc_span, int(sps))
    effective_guard_symbols = (
        max(0, int(rrc_span) // 2)
        if guard_symbols is None
        else max(0, int(guard_symbols))
    )
    frame_limit = target_frames.shape[0] if max_frames is None else min(target_frames.shape[0], int(max_frames))

    for frame_index in range(frame_limit):
        target_iq = target_frames[frame_index]
        pred_iq = pred_frames[frame_index]
        length = min(target_iq.shape[-1], pred_iq.shape[-1])
        target_c = target_iq[0, :length] + 1j * target_iq[1, :length]
        pred_c = pred_iq[0, :length] + 1j * pred_iq[1, :length]
        if float(np.mean(np.abs(target_c) ** 2)) <= float(active_threshold):
            count.skipped_frames += 1
            continue
        count.active_frames += 1

        sample_index = np.arange(length, dtype=np.float64)
        derotation = np.exp(
            -1j * 2.0 * np.pi * float(cfo_hz) * sample_index / float(sample_rate_hz)
        )
        target_mf = matched_filter(target_c * derotation, filter_taps)
        pred_mf = matched_filter(pred_c * derotation, filter_taps)
        frame_bits = np.asarray(bit_frames[frame_index], dtype=np.uint8).reshape(-1)
        n_label_symbols = len(frame_bits) // bits_per_symbol
        if n_label_symbols <= 0:
            count.skipped_frames += 1
            count.active_frames -= 1
            continue
        labels = bits_to_labels(frame_bits, bits_per_symbol)
        target_offset = _select_target_offset(
            target_mf,
            int(sps),
            constellation,
            n_label_symbols,
            effective_guard_symbols,
            reference_labels=labels,
        )
        if target_offset is None:
            count.skipped_frames += 1
            count.active_frames -= 1
            continue
        selected = _search_reference_timing(
            pred_mf,
            target_mf,
            int(sps),
            target_offset,
            n_label_symbols,
            int(max_lag_samples),
            int(block_symbols),
        )
        if selected is None:
            count.skipped_frames += 1
            count.active_frames -= 1
            continue
        pred_symbols, target_symbols, label_start = selected
        label_end = min(len(labels), label_start + len(pred_symbols), label_start + len(target_symbols))
        pred_symbols = pred_symbols[:label_end - label_start]
        target_symbols = target_symbols[:label_end - label_start]
        labels_used = labels[label_start:label_end]
        guard = min(effective_guard_symbols, max(0, (len(labels_used) - 4) // 2))
        if guard:
            pred_symbols = pred_symbols[guard:-guard]
            target_symbols = target_symbols[guard:-guard]
            labels_used = labels_used[guard:-guard]
        errors, compared = _decode_reference_aligned(
            pred_symbols,
            target_symbols,
            labels_used,
            constellation,
            bits_per_symbol,
            int(block_symbols),
        )
        count.add(errors, compared)
    return count


def _normalise_frames(array: np.ndarray, channels: int) -> np.ndarray:
    """Normalize common IQ layouts to ``(frames, channels, samples)``."""
    value = np.asarray(array)
    if value.ndim == 2:
        if np.iscomplexobj(value):
            if value.shape[0] == channels:
                return np.stack([value.real, value.imag], axis=0)[None].astype(np.float32)
            return np.stack([value.real, value.imag], axis=1)[None].astype(np.float64)
        if value.shape[0] == channels:
            return value[None].astype(np.float32)
        if value.shape[1] == channels:
            return value.T[None].astype(np.float32)
        raise ValueError(f"Cannot infer IQ layout from shape {value.shape}")
    if value.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D IQ array, got shape {value.shape}")
    if value.shape[1] == channels:
        result = value
    elif value.shape[2] == channels:
        result = np.transpose(value, (0, 2, 1))
    elif value.shape[0] == channels:
        result = np.transpose(value, (2, 0, 1))
    else:
        raise ValueError(f"Cannot infer IQ layout with {channels} channels from shape {value.shape}")
    return np.asarray(result, dtype=np.float32)


def _normalise_source_frames(array: np.ndarray, num_sources: int) -> np.ndarray:
    return _normalise_frames(array, channels=2 * int(num_sources))


def _open_h5(path: Path):
    try:
        import h5py
    except ImportError as exc:
        # The desktop workspace keeps optional scientific packages in a local
        # dependency directory, while the system Python may not expose it.
        dependency_root = Path(__file__).resolve().parents[2] / ".codex_pydeps"
        if dependency_root.is_dir() and str(dependency_root) not in sys.path:
            sys.path.insert(0, str(dependency_root))
        try:
            import h5py
        except ImportError:
            raise RuntimeError(
                "h5py is required for MATLAB v7.3 files. Use the repository environment "
                "or install h5py."
            ) from exc
    return h5py.File(path, "r")


def _read_scalar(handle: Any, key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in handle:
        return default
    value = np.asarray(handle[key]).reshape(-1)
    return default if value.size == 0 else float(value[0])


def _read_vector(handle: Any, key: str, default: Optional[Sequence[float]] = None) -> Optional[List[float]]:
    if key not in handle:
        return list(default) if default is not None else None
    value = np.asarray(handle[key]).reshape(-1)
    return [float(item) for item in value]


def _read_array_from_file(path: Path, key: Optional[str] = None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            selected = key or ("pred" if "pred" in archive.files else archive.files[0])
            return np.asarray(archive[selected])
        finally:
            archive.close()
    try:
        with _open_h5(path) as handle:
            candidates = [key] if key else [
                "estimated_frames", "separated", "prediction", "pred", "outputs",
                "ideal_frames", "mixed_frames", "data", "name", "MAT",
            ]
            for candidate in candidates:
                if candidate and candidate in handle:
                    return np.asarray(handle[candidate][:])
            datasets = [name for name, obj in handle.items() if hasattr(obj, "shape")]
            raise KeyError(f"No waveform dataset found in {path}; keys={datasets}")
    except (OSError, ValueError):
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not HDF5 and scipy is required to read classic MAT files"
            ) from exc
        values = loadmat(path)
        candidates = [key] if key else [
            "estimated_frames", "separated", "prediction", "pred", "outputs",
            "ideal_frames", "mixed_frames", "data", "name", "MAT",
        ]
        for candidate in candidates:
            if candidate and candidate in values:
                return np.asarray(values[candidate])
        raise KeyError(f"No waveform variable found in {path}; keys={sorted(values)}")


def load_waveform(path: Path, expected_channels: int, key: Optional[str] = None) -> np.ndarray:
    """Load HDF5/classic MAT/NPY/NPZ IQ data as ``(B, 2K, L)``."""
    raw = _read_array_from_file(Path(path), key=key)
    if np.iscomplexobj(raw):
        def complex_to_iq(value: np.ndarray) -> np.ndarray:
            value = np.asarray(value, dtype=np.complex128)
            output = np.empty((value.shape[0], 2 * value.shape[1], value.shape[2]), dtype=np.float32)
            output[:, 0::2, :] = value.real
            output[:, 1::2, :] = value.imag
            return output

        if raw.ndim == 2:
            if raw.shape[0] == expected_channels:
                value = raw[None, ...]
            else:
                value = raw[:, None, ...]
            return complex_to_iq(value)
        if raw.ndim == 3:
            if raw.shape[1] == expected_channels:
                return complex_to_iq(raw)
            if raw.shape[2] == expected_channels:
                return complex_to_iq(np.transpose(raw, (0, 2, 1)))
            if raw.shape[0] == expected_channels:
                return complex_to_iq(np.transpose(raw, (2, 0, 1)))
        raise ValueError(f"Cannot infer complex waveform layout from shape {raw.shape}")
    return _normalise_source_frames(raw, num_sources=expected_channels // 2)


def read_private_metadata(path: Path, num_sources: int) -> Dict[str, Any]:
    """Read protocol fields from a private target/mixture MATLAB file."""
    meta: Dict[str, Any] = {}
    try:
        with _open_h5(Path(path)) as handle:
            for key in (
                "frame_length", "valid_frame_length", "rrc_alpha", "rrc_span",
                "sample_rate_mhz", "Fs_sps", "symbols_per_frame", "bits_per_symbol",
                "bits_per_frame",
            ):
                value = _read_scalar(handle, key)
                if value is not None:
                    meta[key] = value
            for key in (
                "Fs_sps_by_source", "symbols_per_frame_by_source",
                "bits_per_symbol_by_source", "bits_per_frame_by_source",
                "symbol_rates_mhz", "cfo_hz", "initial_phase_rad",
                "delay_samples_by_source",
            ):
                value = _read_vector(handle, key)
                if value is not None:
                    meta[key] = value
    except (OSError, ValueError):
        # Classic MAT files do not normally carry the private generator fields.
        pass

    if "sample_rate_mhz" in meta:
        meta["sample_rate_hz"] = float(meta["sample_rate_mhz"]) * 1e6
    if "Fs_sps_by_source" not in meta and "Fs_sps" in meta:
        meta["Fs_sps_by_source"] = [meta["Fs_sps"]] * int(num_sources)
    if "bits_per_frame_by_source" not in meta and "bits_per_frame" in meta:
        meta["bits_per_frame_by_source"] = [meta["bits_per_frame"]] * int(num_sources)
    if "bits_per_symbol_by_source" not in meta and "bits_per_symbol" in meta:
        meta["bits_per_symbol_by_source"] = [meta["bits_per_symbol"]] * int(num_sources)
    return meta


def _find_bit_path(
    target_path: Path,
    source_index: int,
    bits_root: Optional[Path],
    modulations: Optional[Sequence[str]] = None,
) -> Path:
    dataset_dir = Path(bits_root) if bits_root is not None else target_path.parent.parent / "bits"
    target_name = target_path.name
    derived_name = target_name.replace("Dataset_target", "BitData")
    derived_name = re.sub(r"\.mat$", f"_Source{int(source_index) + 1}.mat", derived_name)
    candidates = [dataset_dir / derived_name]

    # Older generators saved one sidecar per modulation rather than deriving
    # the name from the target file, e.g. QPSK_BitData_1_SNR=10dB_Source1.mat.
    match = re.search(
        r"_(?P<file>\d+)_SNR=(?P<snr>[-+]?\d+(?:\.\d+)?)dB",
        target_name,
        re.IGNORECASE,
    )
    if match is not None and modulations is not None and source_index < len(modulations):
        modulation_token = str(modulations[source_index]).upper().replace("MATLAB-", "")
        generic_name = (
            f"{modulation_token}_BitData_{match.group('file')}"
            f"_SNR={match.group('snr')}dB_Source{int(source_index) + 1}.mat"
        )
        candidates.append(dataset_dir / generic_name)

    # A few original MATLAB runs kept target and sidecars in one directory.
    if target_path.parent != dataset_dir:
        candidates.extend(target_path.parent / path.name for path in candidates.copy())
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return candidate
    return candidates[0]


def _bit_dataset_key(handle: Any) -> Optional[str]:
    preferred = ("file_bits", "bits", "bit_data")
    for key in preferred:
        if key in handle and getattr(handle[key], "size", 0) > 1:
            return key
    candidates = []
    for key, value in handle.items():
        if not hasattr(value, "shape") or getattr(value, "size", 0) <= 1:
            continue
        lowered = str(key).lower()
        if lowered.startswith(("file_bits", "bit_data", "bits")):
            candidates.append((int(value.size), str(key)))
    return max(candidates)[1] if candidates else None


def _bit_value_key(values: Mapping[str, Any]) -> Optional[str]:
    preferred = ("file_bits", "bits", "bit_data")
    for key in preferred:
        if key in values and np.asarray(values[key]).size > 1:
            return key
    candidates = []
    for key, value in values.items():
        if str(key).startswith("_"):
            continue
        array = np.asarray(value)
        if array.size <= 1:
            continue
        lowered = str(key).lower()
        if lowered.startswith(("file_bits", "bit_data", "bits")):
            candidates.append((int(array.size), str(key)))
    return max(candidates)[1] if candidates else None


def _read_bits_file(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not Path(path).exists():
        raise BitsUnavailable(f"Missing bit sidecar: {path}")
    metadata: Dict[str, Any] = {}
    try:
        with _open_h5(Path(path)) as handle:
            key = _bit_dataset_key(handle)
            if key is None:
                raise BitsUnavailable(f"No bit vector in {path}")
            value = np.asarray(handle[key][:]).reshape(-1).astype(np.uint8)
            for name in (
                "source_bits_per_frame", "source_bits_per_symbol",
                "source_symbols_per_frame", "source_Fs_sps",
            ):
                scalar = _read_scalar(handle, name)
                if scalar is not None:
                    metadata[name] = scalar
            return value, metadata
    except BitsUnavailable:
        raise
    except (OSError, ValueError):
        try:
            from scipy.io import loadmat
        except ImportError as exc:
            raise BitsUnavailable(f"Cannot read classic MAT bit sidecar without scipy: {path}") from exc
        values = loadmat(path)
        key = _bit_value_key(values)
        if key is None:
            raise BitsUnavailable(f"No bit vector in {path}")
        for name in (
            "source_bits_per_frame", "source_bits_per_symbol",
            "source_symbols_per_frame", "source_Fs_sps",
        ):
            if name in values:
                metadata[name] = float(np.asarray(values[name]).reshape(-1)[0])
        return np.asarray(values[key]).reshape(-1).astype(np.uint8), metadata


def load_private_bits(
    target_path: Path,
    num_sources: int,
    frame_count: int,
    metadata: Mapping[str, Any],
    bits_root: Optional[Path] = None,
    bit_files: Optional[Sequence[Path]] = None,
    modulations: Optional[Sequence[str]] = None,
) -> List[np.ndarray]:
    """Load and split one flat ``file_bits`` sidecar per source."""
    bits_by_source: List[np.ndarray] = []
    meta_bpf = list(metadata.get("bits_per_frame_by_source", []))
    for source_index in range(int(num_sources)):
        path = (
            Path(bit_files[source_index])
            if bit_files is not None and source_index < len(bit_files)
            else _find_bit_path(
                Path(target_path), source_index, bits_root, modulations=modulations
            )
        )
        raw, bit_meta = _read_bits_file(path)
        bits_per_frame: Optional[int] = None
        if source_index < len(meta_bpf):
            bits_per_frame = int(round(float(meta_bpf[source_index])))
        if bits_per_frame is None and "source_bits_per_frame" in bit_meta:
            bits_per_frame = int(round(float(bit_meta["source_bits_per_frame"])))
        bits_by_source.append(_as_bit_frames(raw, frame_count, bits_per_frame))
    return bits_by_source


def _snr_from_name(path: Path) -> Optional[float]:
    match = re.search(r"SNR=([-+]?\d+(?:\.\d+)?)dB", path.name, re.IGNORECASE)
    if match is None:
        match = re.search(r"SNR([-+]?\d+(?:\.\d+)?)dB", path.name, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _iq_sources(frames: np.ndarray) -> np.ndarray:
    value = np.asarray(frames, dtype=np.float32)
    source_count = value.shape[1] // 2
    return value[:, 0::2, :].astype(np.complex64) + 1j * value[:, 1::2, :].astype(np.complex64)


def source_permutation(pred_frames: np.ndarray, target_frames: np.ndarray) -> Tuple[int, ...]:
    """Return a target-index -> prediction-index PIT permutation."""
    pred = _iq_sources(pred_frames)
    target = _iq_sources(target_frames)
    if pred.shape[1] != target.shape[1]:
        raise ValueError(f"Source count mismatch: pred={pred.shape[1]}, target={target.shape[1]}")
    source_count = pred.shape[1]
    scores = np.zeros((source_count, source_count), dtype=np.float64)
    for target_index in range(source_count):
        y = target[:, target_index, :].reshape(-1)
        y_norm = float(np.sqrt(np.sum(np.abs(y) ** 2)) + EPS)
        for pred_index in range(source_count):
            x = pred[:, pred_index, :].reshape(-1)
            x_norm = float(np.sqrt(np.sum(np.abs(x) ** 2)) + EPS)
            scores[target_index, pred_index] = abs(np.sum(y * np.conj(x))) / (x_norm * y_norm)
    best = max(
        permutations(range(source_count)),
        key=lambda permutation: sum(scores[target_index, permutation[target_index]] for target_index in range(source_count)),
    )
    return tuple(int(index) for index in best)


def _metadata_for_source(
    spec: DatasetSpec,
    metadata: Mapping[str, Any],
    source_index: int,
) -> Dict[str, Any]:
    def vector_value(key: str, default: Any) -> Any:
        values = metadata.get(key)
        if values is not None and source_index < len(values):
            return values[source_index]
        return default

    return {
        "sample_rate_hz": float(metadata.get("sample_rate_hz", spec.sample_rate_hz or 0.0)),
        "sps": int(round(vector_value("Fs_sps_by_source", spec.sps_by_source[source_index] if source_index < len(spec.sps_by_source) and spec.sps_by_source[source_index] is not None else 0))),
        "cfo_hz": float(vector_value("cfo_hz", 0.0)),
        "rrc_alpha": float(metadata.get("rrc_alpha", 0.35)),
        "rrc_span": int(round(float(metadata.get("rrc_span", 20)))),
        "valid_frame_length": int(round(float(metadata.get("valid_frame_length", 0)))) if metadata.get("valid_frame_length") is not None else None,
        "delay_samples": vector_value("delay_samples_by_source", 0.0),
    }


def _active_frame(target: np.ndarray, threshold: float) -> bool:
    value = np.asarray(target, dtype=np.float64)
    return float(np.mean(value[0] ** 2 + value[1] ** 2)) > float(threshold)


def _flatten_source_frames(source_frames: np.ndarray) -> np.ndarray:
    """Concatenate ``(B, 2, L)`` frames in time without interleaving channels."""
    value = np.asarray(source_frames)
    if value.ndim != 3 or value.shape[1] != 2:
        raise ValueError(f"Expected source frames shaped (B, 2, L), got {value.shape}")
    return np.transpose(value, (1, 0, 2)).reshape(1, 2, -1)


def evaluate_private_file(
    target_path: Path,
    dataset: str,
    pred_path: Optional[Path] = None,
    pred_key: Optional[str] = None,
    bits_root: Optional[Path] = None,
    bit_files: Optional[Sequence[Path]] = None,
    max_frames: Optional[int] = None,
    max_lag_samples: int = 32,
    block_symbols: int = 128,
    guard_symbols: Optional[int] = None,
    active_threshold: float = 1e-8,
    use_pit: bool = True,
) -> Dict[str, Any]:
    """Evaluate one private target file and optional separated prediction file."""
    spec = dataset_spec(dataset)
    if spec.public:
        raise ValueError(f"{spec.name} is a public snapshot dataset, not a private target file")
    target_path = Path(target_path)
    target_frames = load_waveform(target_path, 2 * len(spec.modulations), key="ideal_frames")
    metadata = read_private_metadata(target_path, len(spec.modulations))
    frame_count = target_frames.shape[0]
    bits_by_source = load_private_bits(
        target_path,
        len(spec.modulations),
        frame_count,
        metadata,
        bits_root=bits_root,
        bit_files=bit_files,
        modulations=spec.modulations,
    )
    if pred_path is None:
        pred_frames = target_frames.copy()
        target_vs_target = True
    else:
        pred_frames = load_waveform(Path(pred_path), 2 * len(spec.modulations), key=pred_key)
        target_vs_target = False
    if pred_frames.shape[0] != frame_count:
        raise ValueError(f"Prediction/target frame count mismatch: {pred_frames.shape} vs {target_frames.shape}")
    if pred_frames.shape[-1] != target_frames.shape[-1]:
        length = min(pred_frames.shape[-1], target_frames.shape[-1])
        pred_frames = pred_frames[..., :length]
        target_frames = target_frames[..., :length]

    per_source: List[Dict[str, Any]] = []
    max_frame_count = frame_count if max_frames is None else min(frame_count, int(max_frames))
    # Source matching only needs the selected prefix.  The target-vs-target
    # audit is identity by construction and avoids a second full complex copy
    # of large MATLAB files.
    if target_vs_target:
        permutation = tuple(range(len(spec.modulations)))
    else:
        permutation = (
            source_permutation(pred_frames[:max_frame_count], target_frames[:max_frame_count])
            if use_pit else tuple(range(len(spec.modulations)))
        )
    for target_index, modulation in enumerate(spec.modulations):
        pred_index = permutation[target_index]
        protocol = _metadata_for_source(spec, metadata, target_index)
        if protocol["sps"] <= 0 or protocol["sample_rate_hz"] <= 0:
            raise ValueError(
                f"Missing samples-per-symbol/sample-rate metadata for {spec.name} source {target_index + 1}"
            )
        valid_length = protocol.get("valid_frame_length")
        target_source = target_frames[:max_frame_count, 2 * target_index:2 * target_index + 2, :]
        pred_source = pred_frames[:max_frame_count, 2 * pred_index:2 * pred_index + 2, :]
        if valid_length is not None and valid_length > 0:
            target_source = target_source[..., :valid_length]
            pred_source = pred_source[..., :valid_length]
        active_flags = [_active_frame(target_source[index], active_threshold) for index in range(len(target_source))]
        if active_flags and all(active_flags):
            # The generator creates one continuous symbol stream and only then
            # cuts it into frames.  Evaluating each frame independently would
            # eventually associate frame f with the wrong 205-symbol bit block
            # because 4096/20 (and 4096/10) is not integral.  Stream-level
            # filtering preserves the true transmitter symbol order.
            count = reference_ber_arrays(
                _flatten_source_frames(pred_source),
                _flatten_source_frames(target_source),
                bits_by_source[target_index][:max_frame_count].reshape(-1),
                modulation=modulation,
                sps=protocol["sps"],
                sample_rate_hz=protocol["sample_rate_hz"],
                cfo_hz=protocol["cfo_hz"],
                rrc_alpha=protocol["rrc_alpha"],
                rrc_span=protocol["rrc_span"],
                max_lag_samples=max_lag_samples,
                block_symbols=block_symbols,
                guard_symbols=guard_symbols,
                active_threshold=active_threshold,
                max_frames=1,
            )
            count.active_frames = len(active_flags)
        else:
            # Burst datasets have deliberately muted frames.  They are kept on
            # the conservative frame path rather than pretending that bits in
            # an absent burst were transmitted.
            count = reference_ber_arrays(
                pred_source,
                target_source,
                bits_by_source[target_index][:max_frame_count],
                modulation=modulation,
                sps=protocol["sps"],
                sample_rate_hz=protocol["sample_rate_hz"],
                cfo_hz=protocol["cfo_hz"],
                rrc_alpha=protocol["rrc_alpha"],
                rrc_span=protocol["rrc_span"],
                max_lag_samples=max_lag_samples,
                block_symbols=block_symbols,
                guard_symbols=guard_symbols,
                active_threshold=active_threshold,
                max_frames=max_frame_count,
            )
        per_source.append({
            "source": target_index + 1,
            "prediction_source": pred_index + 1,
            "modulation": modulation,
            "sps": protocol["sps"],
            "sample_rate_hz": protocol["sample_rate_hz"],
            "cfo_hz": protocol["cfo_hz"],
            "guard_symbols": (
                max(0, int(protocol["rrc_span"]) // 2)
                if guard_symbols is None else max(0, int(guard_symbols))
            ),
            "errors": count.errors,
            "compared_bits": count.compared_bits,
            "ber": count.ber,
            "active_frames": count.active_frames,
            "skipped_frames": count.skipped_frames,
        })

    total_errors = sum(item["errors"] for item in per_source)
    total_bits = sum(item["compared_bits"] for item in per_source)
    return {
        "dataset": spec.name,
        "target_file": str(target_path),
        "prediction_file": str(pred_path) if pred_path is not None else None,
        "snr_db": _snr_from_name(target_path),
        "receiver": "reference_assisted",
        "target_vs_target": target_vs_target,
        "pit_permutation_target_to_prediction": [index + 1 for index in permutation],
        "errors": total_errors,
        "compared_bits": total_bits,
        "ber": float(total_errors / total_bits) if total_bits else float("nan"),
        "sources": per_source,
    }


def _aggregate_file_results(file_results: Sequence[Mapping[str, Any]], dataset: str) -> Dict[str, Any]:
    by_snr: Dict[str, Dict[str, int]] = {}
    source_totals: Dict[str, Dict[str, int]] = {}
    for result in file_results:
        snr_key = str(result.get("snr_db"))
        group = by_snr.setdefault(snr_key, {"errors": 0, "compared_bits": 0, "files": 0})
        group["errors"] += int(result.get("errors", 0))
        group["compared_bits"] += int(result.get("compared_bits", 0))
        group["files"] += 1
        for source in result.get("sources", []):
            key = str(source["source"])
            source_group = source_totals.setdefault(key, {"errors": 0, "compared_bits": 0})
            source_group["errors"] += int(source.get("errors", 0))
            source_group["compared_bits"] += int(source.get("compared_bits", 0))

    snr_results = {}
    for key, group in sorted(by_snr.items(), key=lambda item: float(item[0]) if item[0] != "None" else float("inf")):
        snr_results[key] = {
            **group,
            "ber": float(group["errors"] / group["compared_bits"]) if group["compared_bits"] else float("nan"),
        }
    sources = {}
    for key, group in sorted(source_totals.items(), key=lambda item: int(item[0])):
        sources[key] = {
            **group,
            "ber": float(group["errors"] / group["compared_bits"]) if group["compared_bits"] else float("nan"),
        }
    errors = sum(int(item.get("errors", 0)) for item in file_results)
    compared = sum(int(item.get("compared_bits", 0)) for item in file_results)
    return {
        "dataset": dataset,
        "status": "ok" if compared else "no_compared_bits",
        "receiver": "reference_assisted",
        "errors": errors,
        "compared_bits": compared,
        "ber": float(errors / compared) if compared else float("nan"),
        "files": len(file_results),
        "by_snr_db": snr_results,
        "by_source": sources,
        "file_results": list(file_results),
    }


def discover_private_target_files(root: Path, dataset: str) -> List[Path]:
    spec = dataset_spec(dataset)
    if spec.public:
        return []
    root = Path(root)
    candidates = [root / spec.name / "target", root / "target"]
    for directory in candidates:
        if directory.is_dir():
            return sorted(directory.glob("*.mat"))
    return []


def evaluate_dataset(
    dataset: str,
    root: Path,
    target_file: Optional[Path] = None,
    pred_file: Optional[Path] = None,
    pred_key: Optional[str] = None,
    bits_root: Optional[Path] = None,
    max_files: Optional[int] = None,
    max_frames: Optional[int] = None,
    max_lag_samples: int = 32,
    block_symbols: int = 128,
    guard_symbols: Optional[int] = None,
    active_threshold: float = 1e-8,
    use_pit: bool = True,
    bit_files: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    """Evaluate private files or return an explicit public-data limitation."""
    spec = dataset_spec(dataset)
    if spec.public:
        return {
            "dataset": spec.name,
            "status": "bits_unavailable",
            "receiver": "reference_assisted",
            "ber": float("nan"),
            "reason": (
                f"{spec.name} files contain IQ snapshots and modulation/SNR labels, "
                "but no transmitter bit sequence. Supply a verified bit sidecar "
                "before reporting strict BER."
            ),
        }
    if target_file is not None:
        files = [Path(target_file)]
    else:
        files = discover_private_target_files(Path(root), spec.name)
        if max_files is not None:
            files = files[:int(max_files)]
    if not files:
        raise FileNotFoundError(
            f"No target files found for {spec.name}. Searched {Path(root) / spec.name / 'target'} "
            f"and {Path(root) / 'target'}"
        )
    if pred_file is not None and len(files) != 1:
        raise ValueError("--pred-file requires --target-file when more than one target file is selected")
    results = []
    for path in files:
        results.append(
            evaluate_private_file(
                path,
                spec.name,
                pred_path=pred_file,
                pred_key=pred_key,
                bits_root=bits_root,
                bit_files=bit_files,
                max_frames=max_frames,
                max_lag_samples=max_lag_samples,
                block_symbols=block_symbols,
                guard_symbols=guard_symbols,
                active_threshold=active_threshold,
                use_pit=use_pit,
            )
        )
    return _aggregate_file_results(results, spec.name)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. 8PSK-H or RML2018")
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "synthetic",
        help="Synthetic root containing <dataset>/target and <dataset>/bits",
    )
    parser.add_argument("--target-file", type=Path, help="Evaluate one target file")
    parser.add_argument("--pred-file", type=Path, help="Optional separated output file")
    parser.add_argument("--pred-key", help="Dataset key when --pred-file is HDF5/MAT")
    parser.add_argument("--bits-root", type=Path, help="Override the automatic bits directory")
    parser.add_argument(
        "--bit-file", type=Path, action="append", default=None,
        help="Bit sidecar per source; repeat once per source",
    )
    parser.add_argument("--max-files", type=int, help="Limit the number of target files")
    parser.add_argument("--max-frames", type=int, help="Limit frames read from each file")
    parser.add_argument("--max-lag", type=int, default=32, help="Timing search range in samples")
    parser.add_argument("--block-symbols", type=int, default=128, help="Carrier/gain tracking block size")
    parser.add_argument(
        "--guard-symbols", type=int, default=None,
        help="Exclude edge symbols; default is rrc_span//2, use 0 to disable",
    )
    parser.add_argument("--active-threshold", type=float, default=1e-8)
    parser.add_argument("--no-pit", action="store_true", help="Do not source-permutation match predictions")
    parser.add_argument("--output-json", type=Path, help="Also write the JSON report to this path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        dataset = canonical_dataset_name(args.dataset)
        result = evaluate_dataset(
            dataset,
            root=args.root,
            target_file=args.target_file,
            pred_file=args.pred_file,
            pred_key=args.pred_key,
            bits_root=args.bits_root,
            max_files=args.max_files,
            max_frames=args.max_frames,
            max_lag_samples=args.max_lag,
            block_symbols=args.block_symbols,
            guard_symbols=args.guard_symbols,
            active_threshold=args.active_threshold,
            use_pit=not args.no_pit,
            bit_files=args.bit_file,
        )
    except (ValueError, FileNotFoundError, BitsUnavailable, RuntimeError) as exc:
        parser.error(str(exc))
        return 2
    safe_result = _json_safe(result)
    print(json.dumps(safe_result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(safe_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.get("status") == "bits_unavailable":
        return 3
    return 0 if result.get("status") == "ok" else 2


__all__ = [
    "BERCount",
    "BitsUnavailable",
    "DATASET_REGISTRY",
    "DatasetSpec",
    "canonical_dataset_name",
    "constellation_for_modulation",
    "dataset_spec",
    "discover_private_target_files",
    "evaluate_dataset",
    "evaluate_private_file",
    "hard_demodulate",
    "labels_to_bits",
    "bits_to_labels",
    "load_private_bits",
    "load_waveform",
    "matched_filter",
    "reference_ber_arrays",
    "rrc_taps",
    "source_permutation",
]


if __name__ == "__main__":
    sys.exit(main())
