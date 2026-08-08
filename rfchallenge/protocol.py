"""Native implementation of the public ICASSP 2024 RF Challenge DSP protocol.

This module mirrors the signal definitions in the official starter kit while
remaining runnable in the IQUMamba PyTorch environment. In particular it
implements:

* QPSK with the official 16-sample/symbol, beta=0.5, span=8 RRC waveform;
* OFDM-QPSK with NFFT=64, CP=16, guards (4, 3), and a nulled DC carrier;
* nominal-SINR mixture construction and phase randomisation used by the kit.

All public functions use batched NumPy arrays. Complex waveform arrays have
shape ``(B, L)`` and I/Q arrays have shape ``(B, 2, L)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np


SOIType = Literal["QPSK", "OFDMQPSK"]

FRAME_LENGTH = 40_960
SINR_DB_VALUES = np.arange(-30.0, 0.1, 3.0, dtype=np.float32)
SOI_TYPES: tuple[str, ...] = ("QPSK", "OFDMQPSK")
INTERFERENCE_TYPES: tuple[str, ...] = (
    "EMISignal1",
    "CommSignal2",
    "CommSignal3",
    "CommSignal5G1",
)
OFFICIAL_CASES: tuple[tuple[str, str], ...] = tuple(
    (soi, interference)
    for soi in SOI_TYPES
    for interference in INTERFERENCE_TYPES
)

QPSK_BITS_PER_SYMBOL = 2
QPSK_SAMPLES_PER_SYMBOL = 16
QPSK_RRC_SPAN_SYMBOLS = 8
QPSK_RRC_ROLLOFF = 0.5

OFDM_NFFT = 64
OFDM_CP_LENGTH = 16
OFDM_SYMBOL_LENGTH = OFDM_NFFT + OFDM_CP_LENGTH
OFDM_LEFT_GUARD = 4
OFDM_RIGHT_GUARD = 3
OFDM_DC_INDEX = OFDM_NFFT // 2
OFDM_ACTIVE_SUBCARRIERS = np.concatenate(
    (
        np.arange(OFDM_LEFT_GUARD, OFDM_DC_INDEX, dtype=np.int64),
        np.arange(OFDM_DC_INDEX + 1, OFDM_NFFT - OFDM_RIGHT_GUARD, dtype=np.int64),
    )
)
OFDM_DATA_SUBCARRIERS = int(OFDM_ACTIVE_SUBCARRIERS.size)


@dataclass(frozen=True)
class GeneratedSOI:
    """One batch of synthetic SOI waveforms and their transmitted bits."""

    waveform: np.ndarray
    bits: np.ndarray
    soi_type: str


@dataclass(frozen=True)
class MixtureBatch:
    """One RF Challenge mixture batch with protocol metadata."""

    mixture: np.ndarray
    target: np.ndarray
    bits: np.ndarray
    nominal_sinr_db: np.ndarray
    actual_sinr_db: np.ndarray
    interference_scale: np.ndarray
    phase_radians: np.ndarray


def _as_batch_complex(samples: np.ndarray | Sequence[complex]) -> np.ndarray:
    """Normalize a waveform array to contiguous complex64 ``(B, L)`` form."""

    array = np.asarray(samples)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim == 3 and array.shape[-1] == 2 and not np.iscomplexobj(array):
        array = array[..., 0] + 1j * array[..., 1]
    if array.ndim != 2:
        raise ValueError(
            "Expected a complex waveform array with shape (B, L), "
            f"got {tuple(array.shape)}"
        )
    return np.ascontiguousarray(array.astype(np.complex64, copy=False))


def complex_to_iq(samples: np.ndarray | Sequence[complex]) -> np.ndarray:
    """Convert complex waveforms ``(B, L)`` to float32 I/Q ``(B, 2, L)``."""

    array = _as_batch_complex(samples)
    return np.stack((array.real, array.imag), axis=1).astype(np.float32, copy=False)


def iq_to_complex(samples: np.ndarray) -> np.ndarray:
    """Convert float I/Q ``(B, 2, L)`` or ``(B, L, 2)`` to complex64."""

    array = np.asarray(samples)
    if array.ndim != 3:
        raise ValueError(f"Expected an I/Q array with three dimensions, got {array.shape}")
    if array.shape[1] == 2:
        real, imag = array[:, 0, :], array[:, 1, :]
    elif array.shape[-1] == 2:
        real, imag = array[..., 0], array[..., 1]
    else:
        raise ValueError(
            "Expected I/Q channels in dimension 1 or -1, "
            f"got {tuple(array.shape)}"
        )
    return np.ascontiguousarray((real + 1j * imag).astype(np.complex64, copy=False))


def root_raised_cosine_taps(
    samples_per_symbol: int = QPSK_SAMPLES_PER_SYMBOL,
    span_in_symbols: int = QPSK_RRC_SPAN_SYMBOLS,
    beta: float = QPSK_RRC_ROLLOFF,
) -> np.ndarray:
    """Return energy-normalized RRC taps used by the official QPSK helper.

    Sionna's ``RootRaisedCosineFilter`` defaults to a unit-energy, symmetric
    filter. The resulting filter has ``span * samples_per_symbol + 1`` taps.
    The closed-form special cases avoid the removable singularities at zero
    and at ``|t| = 1/(4*beta)``.
    """

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    if span_in_symbols <= 0:
        raise ValueError("span_in_symbols must be positive")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    half_width = span_in_symbols * samples_per_symbol // 2
    n = np.arange(-half_width, half_width + 1, dtype=np.float64)
    t = n / float(samples_per_symbol)

    if beta == 0.0:
        taps = np.sinc(t)
    else:
        numerator = (
            np.sin(np.pi * t * (1.0 - beta))
            + 4.0 * beta * t * np.cos(np.pi * t * (1.0 + beta))
        )
        denominator = np.pi * t * (1.0 - (4.0 * beta * t) ** 2)
        taps = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(t),
            where=np.abs(denominator) > 1e-14,
        )

        at_zero = np.isclose(t, 0.0, atol=1e-14)
        taps[at_zero] = 1.0 - beta + 4.0 * beta / np.pi

        singular_t = 1.0 / (4.0 * beta)
        at_singularity = np.isclose(np.abs(t), singular_t, atol=1e-14)
        singular_value = beta / np.sqrt(2.0) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
        )
        taps[at_singularity] = singular_value

    taps /= np.sqrt(np.sum(taps**2))
    return taps.astype(np.float32)


def _same_fir(samples: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Apply a symmetric real FIR to a complex batch with same-length output."""

    waveform = _as_batch_complex(samples)
    filter_taps = np.asarray(taps, dtype=np.float32).reshape(-1)
    if filter_taps.size % 2 != 1:
        raise ValueError("The RF Challenge RRC filter must have odd length")

    # PyTorch's direct Conv1d makes generation practical for long 40,960-point
    # frames. The fallback preserves portability for protocol-only users.
    try:
        import torch
        import torch.nn.functional as functional

        kernel = torch.from_numpy(filter_taps[::-1].copy()).view(1, 1, -1)
        real = torch.from_numpy(np.ascontiguousarray(waveform.real)).unsqueeze(1)
        imag = torch.from_numpy(np.ascontiguousarray(waveform.imag)).unsqueeze(1)
        padding = filter_taps.size // 2
        filtered_real = functional.conv1d(real, kernel, padding=padding)
        filtered_imag = functional.conv1d(imag, kernel, padding=padding)
        result = filtered_real[:, 0].numpy() + 1j * filtered_imag[:, 0].numpy()
        return np.ascontiguousarray(result.astype(np.complex64, copy=False))
    except Exception:
        filtered = [np.convolve(row, filter_taps, mode="same") for row in waveform]
        return np.ascontiguousarray(np.asarray(filtered, dtype=np.complex64))


def qpsk_symbols_from_bits(bits: np.ndarray) -> np.ndarray:
    """Map binary pairs to Sionna-compatible normalized QPSK symbols.

    The mapping is ``00 -> (+1,+1)``, ``01 -> (+1,-1)``,
    ``10 -> (-1,+1)``, and ``11 -> (-1,-1)``, scaled by ``1/sqrt(2)``.
    Bits are flattened in the same pair order used by the official helper.
    """

    bit_array = np.asarray(bits, dtype=np.uint8)
    if bit_array.ndim == 1:
        bit_array = bit_array[None, :]
    if bit_array.ndim != 2 or bit_array.shape[1] % QPSK_BITS_PER_SYMBOL:
        raise ValueError(
            "QPSK bits must have shape (B, 2*N), "
            f"got {tuple(bit_array.shape)}"
        )
    if np.any((bit_array != 0) & (bit_array != 1)):
        raise ValueError("QPSK bits must contain only zeros and ones")

    paired = bit_array.reshape(bit_array.shape[0], -1, QPSK_BITS_PER_SYMBOL)
    real = 1.0 - 2.0 * paired[..., 0].astype(np.float32)
    imag = 1.0 - 2.0 * paired[..., 1].astype(np.float32)
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def qpsk_bits_from_symbols(symbols: np.ndarray) -> np.ndarray:
    """Hard-demap normalized QPSK symbols to flattened uint8 bit pairs."""

    symbol_array = _as_batch_complex(symbols)
    bits = np.stack(
        (
            (symbol_array.real < 0.0).astype(np.uint8),
            (symbol_array.imag < 0.0).astype(np.uint8),
        ),
        axis=-1,
    )
    return bits.reshape(symbol_array.shape[0], -1)


def modulate_qpsk_bits(bits: np.ndarray) -> np.ndarray:
    """Create the official pulse-shaped QPSK waveform from a bit matrix."""

    symbols = qpsk_symbols_from_bits(bits)
    batch_size, num_symbols = symbols.shape
    pulse_train = np.zeros(
        (batch_size, num_symbols * QPSK_SAMPLES_PER_SYMBOL), dtype=np.complex64
    )
    # The official helper upsamples and then shifts right by half a symbol.
    pulse_train[:, QPSK_SAMPLES_PER_SYMBOL // 2 :: QPSK_SAMPLES_PER_SYMBOL] = symbols
    waveform = _same_fir(
        pulse_train,
        root_raised_cosine_taps(),
    )
    return (waveform * np.sqrt(QPSK_SAMPLES_PER_SYMBOL)).astype(np.complex64)


def generate_qpsk(
    batch_size: int,
    num_symbols: int,
    rng: np.random.Generator,
) -> GeneratedSOI:
    """Generate independent QPSK frames and their uncoded information bits."""

    if batch_size <= 0 or num_symbols <= 0:
        raise ValueError("batch_size and num_symbols must be positive")
    bits = rng.integers(
        0,
        2,
        size=(batch_size, num_symbols * QPSK_BITS_PER_SYMBOL),
        dtype=np.uint8,
    )
    return GeneratedSOI(modulate_qpsk_bits(bits), bits, "QPSK")


def demodulate_qpsk(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matched-filter and hard-demap a QPSK waveform.

    Returns ``(bits, sampled_symbols)``. The sampling offset is exactly the
    half-symbol offset used by the official Sionna ``Downsampling`` helper.
    """

    waveform = _as_batch_complex(samples)
    num_symbols = waveform.shape[1] // QPSK_SAMPLES_PER_SYMBOL
    if num_symbols <= 0:
        raise ValueError("QPSK waveform is shorter than one symbol")
    filtered = _same_fir(waveform, root_raised_cosine_taps())
    offset = QPSK_SAMPLES_PER_SYMBOL // 2
    sampled = filtered[:, offset : offset + num_symbols * QPSK_SAMPLES_PER_SYMBOL : QPSK_SAMPLES_PER_SYMBOL]
    sampled = (sampled / np.sqrt(QPSK_SAMPLES_PER_SYMBOL)).astype(np.complex64)
    return qpsk_bits_from_symbols(sampled), sampled


def ofdm_active_subcarriers() -> np.ndarray:
    """Return active QPSK data bins in the starter kit's FFT-shifted grid."""

    return OFDM_ACTIVE_SUBCARRIERS.copy()


def modulate_ofdm_qpsk_bits(bits: np.ndarray) -> np.ndarray:
    """Create CP-OFDM QPSK frames matching the public resource-grid settings."""

    bit_array = np.asarray(bits, dtype=np.uint8)
    if bit_array.ndim == 1:
        bit_array = bit_array[None, :]
    bits_per_ofdm_symbol = OFDM_DATA_SUBCARRIERS * QPSK_BITS_PER_SYMBOL
    if bit_array.ndim != 2 or bit_array.shape[1] % bits_per_ofdm_symbol:
        raise ValueError(
            "OFDM bits must have shape (B, N*112), "
            f"got {tuple(bit_array.shape)}"
        )

    batch_size = bit_array.shape[0]
    num_ofdm_symbols = bit_array.shape[1] // bits_per_ofdm_symbol
    mapped = qpsk_symbols_from_bits(bit_array).reshape(
        batch_size, num_ofdm_symbols, OFDM_DATA_SUBCARRIERS
    )
    # Sionna builds the resource grid in FFT-shifted frequency order before
    # its OFDM modulator applies ifftshift followed by TensorFlow's IFFT.
    grid = np.zeros((batch_size, num_ofdm_symbols, OFDM_NFFT), dtype=np.complex64)
    grid[:, :, OFDM_ACTIVE_SUBCARRIERS] = mapped
    # Sionna 0.10 uses a unitary IDFT: its ``ifft`` helper multiplies
    # TensorFlow's 1/N-normalized IFFT by sqrt(N).  Keeping that factor is
    # essential here because the starter kit applies the requested nominal
    # SINR coefficient without first normalizing SOI/interference power.
    unitary_scale = np.float32(np.sqrt(OFDM_NFFT))
    time_symbols = (
        np.fft.ifft(np.fft.ifftshift(grid, axes=-1), axis=-1)
        * unitary_scale
    ).astype(np.complex64)
    cyclic_prefix = time_symbols[:, :, -OFDM_CP_LENGTH:]
    return np.ascontiguousarray(
        np.concatenate((cyclic_prefix, time_symbols), axis=-1).reshape(batch_size, -1)
    )


def generate_ofdm_qpsk(
    batch_size: int,
    num_ofdm_symbols: int,
    rng: np.random.Generator,
) -> GeneratedSOI:
    """Generate CP-OFDM QPSK frames and their uncoded information bits."""

    if batch_size <= 0 or num_ofdm_symbols <= 0:
        raise ValueError("batch_size and num_ofdm_symbols must be positive")
    bits = rng.integers(
        0,
        2,
        size=(
            batch_size,
            num_ofdm_symbols * OFDM_DATA_SUBCARRIERS * QPSK_BITS_PER_SYMBOL,
        ),
        dtype=np.uint8,
    )
    return GeneratedSOI(modulate_ofdm_qpsk_bits(bits), bits, "OFDMQPSK")


def demodulate_ofdm_qpsk(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove CP, FFT, select data bins, and hard-demap OFDM-QPSK frames."""

    waveform = _as_batch_complex(samples)
    num_ofdm_symbols = waveform.shape[1] // OFDM_SYMBOL_LENGTH
    if num_ofdm_symbols <= 0:
        raise ValueError("OFDM waveform is shorter than one OFDM symbol")
    usable = waveform[:, : num_ofdm_symbols * OFDM_SYMBOL_LENGTH]
    blocks = usable.reshape(waveform.shape[0], num_ofdm_symbols, OFDM_SYMBOL_LENGTH)
    fft_input = blocks[:, :, OFDM_CP_LENGTH:]
    # Inverse of the starter's unitary IDFT. Sionna's normalized FFT divides
    # TensorFlow's unnormalized FFT by sqrt(N).
    unitary_scale = np.float32(np.sqrt(OFDM_NFFT))
    grid = np.fft.fftshift(
        np.fft.fft(fft_input, axis=-1) / unitary_scale,
        axes=-1,
    ).astype(np.complex64)
    data_symbols = grid[:, :, OFDM_ACTIVE_SUBCARRIERS].reshape(waveform.shape[0], -1)
    return qpsk_bits_from_symbols(data_symbols), data_symbols


def generate_soi(
    soi_type: SOIType | str,
    batch_size: int,
    frame_length: int,
    rng: np.random.Generator,
) -> GeneratedSOI:
    """Generate one of the two SOI formats used by the RF Challenge."""

    normalized_type = str(soi_type).upper()
    if normalized_type == "QPSK":
        if frame_length % QPSK_SAMPLES_PER_SYMBOL:
            raise ValueError("QPSK frame_length must be divisible by 16")
        return generate_qpsk(batch_size, frame_length // QPSK_SAMPLES_PER_SYMBOL, rng)
    if normalized_type == "OFDMQPSK":
        if frame_length % OFDM_SYMBOL_LENGTH:
            raise ValueError("OFDMQPSK frame_length must be divisible by 80")
        return generate_ofdm_qpsk(batch_size, frame_length // OFDM_SYMBOL_LENGTH, rng)
    raise ValueError(f"Unsupported SOI type '{soi_type}'. Expected one of {SOI_TYPES}")


def demodulate_soi(soi_type: SOIType | str, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the protocol-matched hard demodulator for a SOI waveform."""

    normalized_type = str(soi_type).upper()
    if normalized_type == "QPSK":
        return demodulate_qpsk(samples)
    if normalized_type == "OFDMQPSK":
        return demodulate_ofdm_qpsk(samples)
    raise ValueError(f"Unsupported SOI type '{soi_type}'. Expected one of {SOI_TYPES}")


def crop_interference_windows(
    interference_frames: np.ndarray,
    frame_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose random source frames and random contiguous windows, as in starter code."""

    bank = _as_batch_complex(interference_frames)
    if bank.shape[1] < frame_length:
        raise ValueError(
            "Interference frames must be at least as long as the requested RF frame: "
            f"{bank.shape[1]} < {frame_length}"
        )
    source_indices = rng.integers(0, bank.shape[0], size=bank.shape[0], endpoint=False)
    selected = bank[source_indices]
    max_start = selected.shape[1] - frame_length
    starts = (
        np.zeros(selected.shape[0], dtype=np.int64)
        if max_start == 0
        else rng.integers(0, max_start, size=selected.shape[0], endpoint=False)
    )
    positions = starts[:, None] + np.arange(frame_length, dtype=np.int64)[None, :]
    return np.take_along_axis(selected, positions, axis=1).astype(np.complex64, copy=False)


def crop_interference_batch(
    interference_frames: np.ndarray,
    batch_size: int,
    frame_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``batch_size`` random contiguous windows from a frame bank."""

    bank = _as_batch_complex(interference_frames)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if bank.shape[1] < frame_length:
        raise ValueError(
            "Interference frames must be at least as long as the requested RF frame: "
            f"{bank.shape[1]} < {frame_length}"
        )
    source_indices = rng.integers(0, bank.shape[0], size=batch_size, endpoint=False)
    selected = bank[source_indices]
    max_start = selected.shape[1] - frame_length
    starts = (
        np.zeros(batch_size, dtype=np.int64)
        if max_start == 0
        else rng.integers(0, max_start, size=batch_size, endpoint=False)
    )
    positions = starts[:, None] + np.arange(frame_length, dtype=np.int64)[None, :]
    return np.take_along_axis(selected, positions, axis=1).astype(np.complex64, copy=False)


def actual_sinr_db(signal: np.ndarray, interference: np.ndarray) -> np.ndarray:
    """Compute per-frame ``10*log10(P_signal/P_interference)`` in dB."""

    signal_array = _as_batch_complex(signal)
    interference_array = _as_batch_complex(interference)
    if signal_array.shape != interference_array.shape:
        raise ValueError("signal and interference must have the same shape")
    signal_power = np.mean(np.abs(signal_array) ** 2, axis=-1)
    interference_power = np.mean(np.abs(interference_array) ** 2, axis=-1)
    return (10.0 * np.log10(signal_power / np.maximum(interference_power, 1e-30))).astype(
        np.float32
    )


def build_mixtures(
    target: np.ndarray,
    interference: np.ndarray,
    nominal_sinr_db: float | Sequence[float] | np.ndarray,
    rng: np.random.Generator,
) -> MixtureBatch:
    """Mix an SOI and interference using the official nominal-SINR rule.

    The starter code deliberately uses the scalar ``sqrt(10**(-SINR/10))``
    without power-normalizing either waveform. We preserve that rule and
    record the resulting actual SINR separately.
    """

    target_array = _as_batch_complex(target)
    interference_array = _as_batch_complex(interference)
    if target_array.shape != interference_array.shape:
        raise ValueError("target and interference must have the same shape")

    nominal = np.broadcast_to(
        np.asarray(nominal_sinr_db, dtype=np.float32), (target_array.shape[0],)
    ).astype(np.float32, copy=False)
    amplitude = np.sqrt(10.0 ** (-nominal / 10.0)).astype(np.float32)
    phase = rng.uniform(0.0, 1.0, size=target_array.shape[0]).astype(np.float32)
    phase_radians = (2.0 * np.pi * phase).astype(np.float32)
    coefficient = amplitude * np.exp(1j * phase_radians)
    scaled_interference = interference_array * coefficient[:, None]
    mixture = (target_array + scaled_interference).astype(np.complex64)
    return MixtureBatch(
        mixture=np.ascontiguousarray(mixture),
        target=np.ascontiguousarray(target_array),
        bits=np.empty((target_array.shape[0], 0), dtype=np.uint8),
        nominal_sinr_db=np.ascontiguousarray(nominal),
        actual_sinr_db=actual_sinr_db(target_array, scaled_interference),
        interference_scale=np.ascontiguousarray(amplitude),
        phase_radians=np.ascontiguousarray(phase_radians),
    )


def generate_mixture_batch(
    soi_type: SOIType | str,
    interference_frames: np.ndarray,
    batch_size: int,
    frame_length: int,
    nominal_sinr_db: float | Sequence[float] | np.ndarray,
    rng: np.random.Generator,
) -> MixtureBatch:
    """Generate a supervised RF Challenge batch from a raw interference bank."""

    generated = generate_soi(soi_type, batch_size, frame_length, rng)
    interference = crop_interference_batch(interference_frames, batch_size, frame_length, rng)
    mixed = build_mixtures(generated.waveform, interference, nominal_sinr_db, rng)
    return MixtureBatch(
        mixture=mixed.mixture,
        target=generated.waveform,
        bits=generated.bits,
        nominal_sinr_db=mixed.nominal_sinr_db,
        actual_sinr_db=mixed.actual_sinr_db,
        interference_scale=mixed.interference_scale,
        phase_radians=mixed.phase_radians,
    )
