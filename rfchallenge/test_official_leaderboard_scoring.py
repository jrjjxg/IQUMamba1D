"""Official eight-case MSE/BER leaderboard formula contracts."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rfchallenge.cli import _official_wavenet_case_directory, build_parser
from rfchallenge.metrics import (
    RFChallengeCaseMetrics,
    aggregate_case_metrics,
    official_case_ber_score_db,
)
from rfchallenge.protocol import SINR_DB_VALUES, complex_to_iq, generate_qpsk
from rfchallenge.training import TrainOptions, validate_single_soi_model_with_metrics


def _case(index: int) -> RFChallengeCaseMetrics:
    mse_db = np.full(11, -10.0 - index, dtype=np.float32)
    ber = np.full(11, 0.1, dtype=np.float32)
    # Case i first reaches BER <= 1e-2 at bin i+1 (capped at the final bin).
    first = min(index + 1, 10)
    ber[first:] = 0.01
    return RFChallengeCaseMetrics(
        soi_type="QPSK" if index < 4 else "OFDMQPSK",
        interference_type=f"I{index}",
        sinr_db=SINR_DB_VALUES.copy(),
        mse_db=mse_db,
        ber=ber,
        frame_count=np.full(11, 100, dtype=np.int64),
        raw_mse=np.power(10.0, mse_db / 10.0),
    )


class OfficialLeaderboardFormulaTests(unittest.TestCase):
    def test_ber_score_is_lowest_qualifying_sinr(self) -> None:
        ber = np.full(11, 0.02, dtype=np.float32)
        ber[6:] = 0.01
        self.assertEqual(
            official_case_ber_score_db(SINR_DB_VALUES, ber), -12.0
        )

    def test_ber_score_is_zero_when_threshold_is_never_reached(self) -> None:
        self.assertEqual(
            official_case_ber_score_db(
                SINR_DB_VALUES, np.full(11, 0.02, dtype=np.float32)
            ),
            0.0,
        )

    def test_result_is_sum_and_average_is_divided_by_eight(self) -> None:
        cases = [_case(index) for index in range(8)]
        aggregate = aggregate_case_metrics(cases)
        expected_mse = sum(result.official_mse_score_db for result in cases)
        expected_ber = sum(result.official_ber_score_db for result in cases)
        self.assertAlmostEqual(aggregate["official_mse_result_db"], expected_mse)
        self.assertAlmostEqual(aggregate["official_mse_average_db"], expected_mse / 8)
        self.assertAlmostEqual(aggregate["official_ber_result_db"], expected_ber)
        self.assertAlmostEqual(aggregate["official_ber_average_db"], expected_ber / 8)

    def test_epoch_validation_demodulates_bits_and_reports_ber(self) -> None:
        frames_per_sinr = 2
        total_frames = len(SINR_DB_VALUES) * frames_per_sinr
        generated = generate_qpsk(
            batch_size=total_frames,
            num_symbols=10,
            rng=np.random.default_rng(2024),
        )
        iq = torch.from_numpy(complex_to_iq(generated.waveform)).contiguous()
        labels = torch.repeat_interleave(
            torch.from_numpy(SINR_DB_VALUES.copy()),
            frames_per_sinr,
        )
        bits = torch.from_numpy(generated.bits.copy())
        loader = DataLoader(
            TensorDataset(iq, iq.clone(), labels, bits),
            batch_size=3,
            shuffle=False,
        )
        metrics = validate_single_soi_model_with_metrics(
            model=torch.nn.Identity(),
            loader=loader,
            device=torch.device("cpu"),
            options=TrainOptions(epochs=1, batch_size=3, amp=False),
            soi_type="QPSK",
        )
        self.assertEqual(metrics.complex_mse, 0.0)
        self.assertEqual(metrics.ber, 0.0)
        self.assertEqual(metrics.official_ber_score_db, -30.0)
        self.assertEqual(metrics.ber_by_sinr, [0.0] * len(SINR_DB_VALUES))


class OfficialBaselineAllCommandTests(unittest.TestCase):
    def test_case_directory_matches_released_naming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = (
                root
                / "dataset_ofdmqpsk_commsignal2_mixture_wavenet"
            )
            expected.mkdir()
            self.assertEqual(
                _official_wavenet_case_directory(
                    root, "OFDMQPSK", "CommSignal2"
                ),
                expected,
            )

    def test_one_command_parser_contract(self) -> None:
        args = build_parser().parse_args([
            "benchmark-baseline-all",
            "--data-root", "data",
            "--weights-root", "weights",
        ])
        self.assertEqual(args.n_per_sinr, 100)
        self.assertEqual(args.seed, 100)
        self.assertEqual(args.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
