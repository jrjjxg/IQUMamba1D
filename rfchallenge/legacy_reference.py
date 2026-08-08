"""Dependency-free reference for the public ICASSP 2024 starter semantics.

The official starter uses TensorFlow/Sionna, which is intentionally not a
dependency of the IQUMamba environment. This module keeps an independent,
small NumPy translation of the public QPSK/OFDM helpers and evaluator for
compatibility testing. It must not import the native ``rfchallenge.protocol``
implementation, so a shared bug cannot make the comparison pass.
"""

from __future__ import annotations

import numpy as np


_QPSK_SPS = 16
_QPSK_SPAN = 8
_QPSK_BETA = 0.5
_OFDM_NFFT = 64
_OFDM_CP = 16
_OFDM_DATA_BINS = np.concatenate((np.arange(4, 32), np.arange(33, 61))).astype(np.int64)


def starter_rrc_taps() -> np.ndarray:
    """Compute the unit-energy RRC impulse response used by Sionna's filter."""

    n = np.arange(-64, 65, dtype=np.float64)
    t = n / _QPSK_SPS
    beta = _QPSK_BETA
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
    taps[np.isclose(t, 0.0, atol=1e-14)] = 1.0 - beta + 4.0 * beta / np.pi
    singular = np.isclose(np.abs(t), 1.0 / (4.0 * beta), atol=1e-14)
    taps[singular] = beta / np.sqrt(2.0) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
        + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
    )
    taps /= np.sqrt(np.sum(taps**2))
    return taps.astype(np.float32)


def _as_bits(bits: np.ndarray, bits_per_symbol: int = 2) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] % bits_per_symbol:
        raise ValueError("bits must be a two-dimensional integral number of QPSK bit pairs")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("bits must contain zeros and ones")
    return values


def _as_complex_batch(waveforms: np.ndarray) -> np.ndarray:
    values = np.asarray(waveforms)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not np.iscomplexobj(values):
        raise ValueError("waveforms must be a complex array with shape (batch, samples)")
    return np.ascontiguousarray(values.astype(np.complex64, copy=False))


def _map_qpsk(bits: np.ndarray) -> np.ndarray:
    pairs = _as_bits(bits).reshape(bits.shape[0], -1, 2)
    real = 1.0 - 2.0 * pairs[..., 0].astype(np.float32)
    imag = 1.0 - 2.0 * pairs[..., 1].astype(np.float32)
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def _hard_demap_qpsk(symbols: np.ndarray) -> np.ndarray:
    values = _as_complex_batch(symbols)
    return np.stack(
        ((values.real < 0.0).astype(np.uint8), (values.imag < 0.0).astype(np.uint8)),
        axis=-1,
    ).reshape(values.shape[0], -1)


def _same_rrc(waveforms: np.ndarray) -> np.ndarray:
    values = _as_complex_batch(waveforms)
    taps = starter_rrc_taps()
    return np.ascontiguousarray(
        np.asarray([np.convolve(row, taps, mode="same") for row in values], dtype=np.complex64)
    )


def starter_modulate_qpsk(bits: np.ndarray) -> np.ndarray:
    """Direct equivalent of ``modulate_qpsk_signal`` in the public helper."""

    bit_values = _as_bits(bits)
    symbols = _map_qpsk(bit_values)
    impulses = np.zeros((symbols.shape[0], symbols.shape[1] * _QPSK_SPS), dtype=np.complex64)
    impulses[:, _QPSK_SPS // 2 :: _QPSK_SPS] = symbols
    return (_same_rrc(impulses) * np.sqrt(_QPSK_SPS)).astype(np.complex64)


def starter_demodulate_qpsk(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct equivalent of the starter matched-filter/downsample hard demod."""

    values = _as_complex_batch(waveforms)
    num_symbols = values.shape[1] // _QPSK_SPS
    filtered = _same_rrc(values)
    sampled = filtered[:, 8 : 8 + num_symbols * _QPSK_SPS : _QPSK_SPS]
    sampled = (sampled / np.sqrt(_QPSK_SPS)).astype(np.complex64)
    return _hard_demap_qpsk(sampled), sampled


def starter_ofdm_data_subcarriers() -> np.ndarray:
    """Return literal Sionna ResourceGrid data-bin indices in shifted order."""

    return _OFDM_DATA_BINS.copy()


def starter_modulate_ofdm_qpsk(bits: np.ndarray) -> np.ndarray:
    """Direct equivalent of the public ResourceGridMapper + OFDMModulator path."""

    bit_values = _as_bits(bits)
    bits_per_symbol = _OFDM_DATA_BINS.size * 2
    if bit_values.shape[1] % bits_per_symbol:
        raise ValueError("OFDM bit width must be a multiple of 112")
    count = bit_values.shape[1] // bits_per_symbol
    data = _map_qpsk(bit_values).reshape(bit_values.shape[0], count, _OFDM_DATA_BINS.size)
    resource_grid = np.zeros((bit_values.shape[0], count, _OFDM_NFFT), dtype=np.complex64)
    resource_grid[:, :, _OFDM_DATA_BINS] = data
    # Sionna 0.10's signal.ifft is unitary and therefore multiplies
    # TensorFlow's IFFT by sqrt(fft_size).
    time_symbols = (
        np.fft.ifft(np.fft.ifftshift(resource_grid, axes=-1), axis=-1)
        * np.float32(np.sqrt(_OFDM_NFFT))
    ).astype(np.complex64)
    with_cp = np.concatenate((time_symbols[:, :, -_OFDM_CP:], time_symbols), axis=-1)
    return np.ascontiguousarray(with_cp.reshape(bit_values.shape[0], -1))


def starter_demodulate_ofdm_qpsk(waveforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct equivalent of OFDMDemodulator + ResourceGridDemapper + hard demod."""

    values = _as_complex_batch(waveforms)
    symbols = values.shape[1] // (_OFDM_NFFT + _OFDM_CP)
    usable = values[:, : symbols * (_OFDM_NFFT + _OFDM_CP)]
    blocks = usable.reshape(values.shape[0], symbols, _OFDM_NFFT + _OFDM_CP)
    grid = np.fft.fftshift(
        np.fft.fft(blocks[:, :, _OFDM_CP:], axis=-1)
        / np.float32(np.sqrt(_OFDM_NFFT)),
        axes=-1,
    )
    data = grid[:, :, _OFDM_DATA_BINS].reshape(values.shape[0], -1).astype(np.complex64)
    return _hard_demap_qpsk(data), data


def starter_evaluate(
    estimated_soi: np.ndarray,
    target_soi: np.ndarray,
    estimated_bits: np.ndarray,
    target_bits: np.ndarray,
    n_per_sinr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the public evaluation script's MSE/BER aggregation exactly."""

    estimate = _as_complex_batch(estimated_soi)
    target = _as_complex_batch(target_soi)
    bit_estimate = np.asarray(estimated_bits, dtype=np.uint8)
    bit_target = np.asarray(target_bits, dtype=np.uint8)
    if estimate.shape != target.shape or bit_estimate.shape != bit_target.shape:
        raise ValueError("estimate/target arrays must have matching shapes")
    if estimate.shape[0] != len(np.arange(-30, 0.1, 3)) * n_per_sinr:
        raise ValueError("frame count must contain eleven equal SINR groups")

    all_mse: list[np.ndarray] = []
    all_ber: list[np.ndarray] = []
    for index in range(11):
        begin, end = index * n_per_sinr, (index + 1) * n_per_sinr
        frame_mse = np.mean(np.abs(estimate[begin:end] - target[begin:end]) ** 2, axis=1)
        frame_ber = np.sum(
            (bit_estimate[begin:end] != bit_target[begin:end]).astype(np.float32), axis=1
        ) / bit_target.shape[1]
        all_mse.append(frame_mse)
        all_ber.append(frame_ber)
    mse_db = 10.0 * np.log10(np.mean(np.asarray(all_mse), axis=-1))
    ber = np.mean(np.asarray(all_ber), axis=-1)
    return mse_db, ber
