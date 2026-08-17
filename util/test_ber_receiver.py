import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.ber_receiver import (
    canonical_dataset_name,
    constellation_for_modulation,
    dataset_spec,
    evaluate_dataset,
    _find_bit_path,
    labels_to_bits,
    reference_ber_arrays,
    rrc_taps,
    source_permutation,
)


class BERReceiverTest(unittest.TestCase):
    def test_registry_aliases_and_public_bit_status(self):
        self.assertEqual(canonical_dataset_name("QPSK16APSK-A"), "QPSK+16APSK-A")
        self.assertEqual(canonical_dataset_name("2016"), "RML2016")
        self.assertTrue(dataset_spec("TorchSig").public)
        result = evaluate_dataset("RML2018", Path("missing-public-root"))
        self.assertEqual(result["status"], "bits_unavailable")

    def test_constellation_tables_round_trip_all_labels(self):
        for modulation in ("8PSK", "QPSK", "16APSK", "16QAM", "64QAM", "128QAM"):
            constellation, bits_per_symbol = constellation_for_modulation(modulation)
            labels = np.arange(len(constellation), dtype=np.int64)
            recovered = np.argmin(
                np.abs(constellation[labels, None] - constellation[None, :]), axis=1
            )
            self.assertTrue(np.array_equal(recovered, labels), modulation)
            bits = labels_to_bits(labels, bits_per_symbol)
            self.assertEqual(len(bits), len(labels) * bits_per_symbol)

        cross, _ = constellation_for_modulation("128QAM")
        scale = np.sqrt(82.0)
        self.assertTrue(np.allclose(cross[0], (-7.0 + 9.0j) / scale))
        self.assertTrue(np.allclose(cross[16], (-9.0 + 7.0j) / scale))
        self.assertTrue(np.allclose(cross[64], (7.0 + 9.0j) / scale))

    def test_reference_receiver_handles_cfo_phase_gain_and_integer_delay(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(0, 8, size=192, dtype=np.int64)
        constellation, bits_per_symbol = constellation_for_modulation("8PSK")
        bits = labels_to_bits(labels, bits_per_symbol)
        sps = 8
        sample_rate_hz = 8e6
        cfo_hz = 1250.0
        taps = rrc_taps(0.35, 8, sps)
        impulses = np.zeros(len(labels) * sps, dtype=np.complex128)
        impulses[::sps] = constellation[labels]
        tx = np.convolve(impulses, taps, mode="same")
        sample_index = np.arange(len(tx), dtype=np.float64)
        target = tx * np.exp(1j * (0.73 + 2 * np.pi * cfo_hz * sample_index / sample_rate_hz))
        delay = 3
        delayed = np.zeros_like(target)
        delayed[delay:] = target[:-delay]
        pred = delayed * (0.37 * np.exp(1j * 1.17))
        count = reference_ber_arrays(
            np.stack([pred.real, pred.imag], axis=0)[None],
            np.stack([target.real, target.imag], axis=0)[None],
            bits[None, :],
            modulation="8PSK",
            sps=sps,
            sample_rate_hz=sample_rate_hz,
            cfo_hz=cfo_hz,
            rrc_alpha=0.35,
            rrc_span=8,
            max_lag_samples=6,
            block_symbols=32,
        )
        self.assertEqual(count.errors, 0)
        self.assertGreater(count.compared_bits, 0)
        self.assertEqual(count.ber, 0.0)

    def test_default_rrc_guard_removes_qam_filter_transients(self):
        labels = np.arange(128, dtype=np.int64)
        constellation, bits_per_symbol = constellation_for_modulation("128QAM")
        bits = labels_to_bits(labels, bits_per_symbol)
        sps = 8
        taps = rrc_taps(0.35, 8, sps)
        impulses = np.zeros(len(labels) * sps, dtype=np.complex128)
        impulses[::sps] = constellation[labels]
        waveform = np.convolve(impulses, taps, mode="same")
        iq = np.stack([waveform.real, waveform.imag], axis=0)[None]

        count = reference_ber_arrays(
            iq,
            iq,
            bits[None, :],
            modulation="128QAM",
            sps=sps,
            sample_rate_hz=8e6,
            rrc_alpha=0.35,
            rrc_span=8,
        )
        self.assertEqual(count.errors, 0)
        self.assertLess(count.compared_bits, len(bits))

    def test_source_permutation_is_target_to_prediction(self):
        rng = np.random.default_rng(3)
        target = rng.standard_normal((5, 4, 32)).astype(np.float32)
        prediction = target[:, [2, 3, 0, 1], :]
        self.assertEqual(source_permutation(prediction, target), (1, 0))

    def test_legacy_modulation_named_sidecar_is_discovered(self):
        root = Path("legacy-sidecar-test")
        target = root / "target" / (
            "2Source_QPSK+16APSK-A_Dataset_target_1_SNR=-10dB.mat"
        )
        expected = root / "bits" / "QPSK_BitData_1_SNR=-10dB_Source1.mat"
        with patch.object(
            Path,
            "exists",
            autospec=True,
            side_effect=lambda path: path == expected,
        ):
            found = _find_bit_path(
                target,
                0,
                None,
                modulations=("QPSK", "16APSK"),
            )
            self.assertEqual(found, expected)


if __name__ == "__main__":
    unittest.main()
