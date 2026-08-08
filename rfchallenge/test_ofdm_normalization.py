"""Regression tests for the official Sionna OFDM energy convention."""

from __future__ import annotations

import unittest

import numpy as np

from rfchallenge.protocol import (
    OFDM_CP_LENGTH,
    OFDM_DATA_SUBCARRIERS,
    OFDM_NFFT,
    OFDM_SYMBOL_LENGTH,
    demodulate_ofdm_qpsk,
    generate_ofdm_qpsk,
)


class OfficialOFDMNormalizationTests(unittest.TestCase):
    def test_useful_symbol_power_matches_unitary_sionna_ifft(self) -> None:
        generated = generate_ofdm_qpsk(
            batch_size=3,
            num_ofdm_symbols=7,
            rng=np.random.default_rng(100),
        )
        blocks = generated.waveform.reshape(3, 7, OFDM_SYMBOL_LENGTH)
        useful = blocks[:, :, OFDM_CP_LENGTH:]

        # Every active QPSK bin has unit energy. A unitary 64-point IDFT
        # therefore preserves total energy, giving mean time-domain power
        # 56/64 for the 56 active data subcarriers.
        expected_power = OFDM_DATA_SUBCARRIERS / OFDM_NFFT
        per_symbol_power = np.mean(np.abs(useful) ** 2, axis=-1)
        np.testing.assert_allclose(
            per_symbol_power,
            expected_power,
            rtol=2e-6,
            atol=2e-6,
        )

    def test_unitary_modulation_still_roundtrips_bits_and_symbols(self) -> None:
        generated = generate_ofdm_qpsk(
            batch_size=2,
            num_ofdm_symbols=5,
            rng=np.random.default_rng(23),
        )
        recovered_bits, recovered_symbols = demodulate_ofdm_qpsk(
            generated.waveform
        )

        np.testing.assert_array_equal(recovered_bits, generated.bits)
        np.testing.assert_allclose(
            np.abs(recovered_symbols),
            1.0,
            rtol=2e-6,
            atol=2e-6,
        )


if __name__ == "__main__":
    unittest.main()
