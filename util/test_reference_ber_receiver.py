import unittest

import numpy as np
import torch

from util.metrics import _rrc_taps, reference_ber_iq_from_bits


class ReferenceBerReceiverTest(unittest.TestCase):
    def test_identity_stream_with_cfo_and_unknown_phase_is_zero(self):
        rng = np.random.default_rng(7)
        bits_per_symbol = 3
        labels = np.tile(np.arange(8, dtype=np.int64), 32)
        bits = ((labels[:, None] >> np.arange(2, -1, -1)) & 1).astype(np.uint8)
        constellation = np.exp(1j * (np.pi / 8 + np.pi * labels / 4))
        sps = 8
        taps = _rrc_taps(0.35, 8, sps)
        upsampled = np.zeros(len(labels) * sps, dtype=np.complex128)
        upsampled[::sps] = constellation
        shaped = np.convolve(upsampled, taps, mode="same")
        sample_rate = 8e6
        cfo_hz = 1250.0
        n = np.arange(len(shaped), dtype=np.float64)
        stream = shaped * np.exp(1j * (0.73 + 2 * np.pi * cfo_hz * n / sample_rate))
        stream += 1e-5 * (
            rng.standard_normal(len(stream)) + 1j * rng.standard_normal(len(stream))
        )

        iq = torch.from_numpy(
            np.stack([stream.real, stream.imag], axis=0)[None].astype(np.float32)
        )
        bit_tensor = torch.from_numpy(bits.reshape(1, -1))
        value = reference_ber_iq_from_bits(
            iq,
            iq,
            bit_tensor,
            modulation="8PSK",
            sps=sps,
            sample_rate_hz=sample_rate,
            cfo_hz=cfo_hz,
            rrc_alpha=0.35,
            rrc_span=8,
        )
        self.assertEqual(float(value), 0.0)


if __name__ == "__main__":
    unittest.main()
