"""Contracts for resumable training and the train-test-all command."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import TensorDataset

from rfchallenge.cli import (
    _preflight_all_case_initial_checkpoints,
    _resolve_initial_model_checkpoint,
    _train_one_case,
    build_parser,
    command_train_test_all,
)
from rfchallenge.protocol import INTERFERENCE_TYPES, OFFICIAL_CASES
from rfchallenge.training import TrainOptions, train_single_soi_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TinySingleSOIModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(2, 2, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class ResumableTrainingTests(unittest.TestCase):
    def test_checkpoint_artifacts_and_resume_to_total_epoch_count(self) -> None:
        torch.manual_seed(17)
        features = torch.randn(4, 2, 32)
        targets = torch.randn(4, 2, 32)
        dataset = TensorDataset(features, targets)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_history = train_single_soi_model(
                model=_TinySingleSOIModel(),
                train_dataset=dataset,
                validation_dataset=dataset,
                device=torch.device("cpu"),
                output_dir=root,
                options=TrainOptions(
                    epochs=1,
                    batch_size=2,
                    learning_rate=1e-3,
                    amp=False,
                    lr_patience=0,
                ),
                metadata={
                    "soi_type": "QPSK",
                    "interference_type": "EMISignal1",
                    "frame_length": 32,
                    "model_stage": 4,
                },
            )
            expected_files = (
                root / "weights" / "best_model_weights.pth",
                root / "weights" / "best_training_checkpoint.pth",
                root / "weights" / "latest_training_checkpoint.pth",
                root / "checkpoint" / "best_model_weights.pth",
                root / "best.pt",
                root / "last.pt",
                root / "training_history.json",
            )
            for path in expected_files:
                self.assertTrue(path.is_file(), path)
            self.assertEqual(len(first_history.train_loss), 1)

            latest = root / "weights" / "latest_training_checkpoint.pth"
            resumed_history = train_single_soi_model(
                model=_TinySingleSOIModel(),
                train_dataset=dataset,
                validation_dataset=dataset,
                device=torch.device("cpu"),
                output_dir=root,
                options=TrainOptions(
                    epochs=2,
                    batch_size=2,
                    learning_rate=1e-3,
                    amp=False,
                    lr_patience=0,
                ),
                metadata={
                    "soi_type": "QPSK",
                    "interference_type": "EMISignal1",
                    "frame_length": 32,
                    "model_stage": 4,
                },
                resume_checkpoint=latest,
            )
            self.assertEqual(len(resumed_history.train_loss), 2)
            try:
                payload = torch.load(latest, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(latest, map_location="cpu")
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(len(payload["history"]["train_loss"]), 2)
            self.assertIn("optimizer_state_dict", payload)
            self.assertIsNotNone(payload["scheduler_state_dict"])
            self.assertIn("scaler_state_dict", payload)
            self.assertIn("torch_random_state", payload)

    def test_resume_rejects_a_different_case(self) -> None:
        dataset = TensorDataset(torch.randn(2, 2, 8), torch.randn(2, 2, 8))
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "weights" / "latest_training_checkpoint.pth"
            train_single_soi_model(
                _TinySingleSOIModel(),
                dataset,
                dataset,
                torch.device("cpu"),
                directory,
                TrainOptions(epochs=1, batch_size=1, amp=False),
                metadata={"soi_type": "QPSK", "interference_type": "EMISignal1"},
            )
            with self.assertRaisesRegex(ValueError, "interference_type"):
                train_single_soi_model(
                    _TinySingleSOIModel(),
                    dataset,
                    dataset,
                    torch.device("cpu"),
                    directory,
                    TrainOptions(epochs=2, batch_size=1, amp=False),
                    metadata={"soi_type": "QPSK", "interference_type": "CommSignal2"},
                    resume_checkpoint=latest,
                )


class TrainTestAllParserTests(unittest.TestCase):
    def test_parser_exposes_training_test_and_resume_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "train-test-all",
                "--data-root", "data",
                "--model-stage", "342",
                "--resume",
            ]
        )
        self.assertIs(args.handler, command_train_test_all)
        self.assertEqual(args.model_stage, 342)
        self.assertTrue(args.resume)
        self.assertEqual(args.test_n_per_sinr, 100)
        self.assertEqual(args.test_seed, 100)
        self.assertEqual(args.test_batch_size, 8)

    def test_parser_exposes_model_only_wavenet_initialization(self) -> None:
        args = build_parser().parse_args(
            [
                "train-test-all",
                "--data-root", "data",
                "--model-config",
                str(PROJECT_ROOT / "config" / "model_config_icassp_baseline_wavenet.yaml"),
                "--init-checkpoint-root", "released-weights",
            ]
        )
        self.assertEqual(args.init_checkpoint_root, Path("released-weights"))
        self.assertIsNone(args.init_checkpoint)

    def test_initial_checkpoint_root_maps_all_eight_official_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_root = Path(directory) / "torchmodels"
            expected: dict[tuple[str, str], Path] = {}
            for soi, interference in OFFICIAL_CASES:
                case_directory = weights_root / (
                    f"dataset_{soi.lower()}_{interference.lower()}_mixture_wavenet"
                )
                case_directory.mkdir(parents=True)
                checkpoint = case_directory / "weights.pt"
                checkpoint.touch()
                expected[(soi, interference)] = checkpoint

            args = build_parser().parse_args(
                [
                    "train-test-all",
                    "--data-root", str(Path(directory) / "data"),
                    "--model-config",
                    str(PROJECT_ROOT / "config" / "model_config_icassp_baseline_wavenet.yaml"),
                    "--init-checkpoint-root", str(weights_root),
                ]
            )
            for case, checkpoint in expected.items():
                self.assertEqual(
                    _resolve_initial_model_checkpoint(args, *case),
                    checkpoint,
                )

    def test_all_case_command_rejects_one_checkpoint_for_eight_tasks(self) -> None:
        args = build_parser().parse_args(
            [
                "train-test-all",
                "--data-root", "data",
                "--model-config",
                str(PROJECT_ROOT / "config" / "model_config_icassp_baseline_wavenet.yaml"),
                "--init-checkpoint", "one-case.pt",
            ]
        )
        with self.assertRaisesRegex(ValueError, "only valid for the single-case"):
            _preflight_all_case_initial_checkpoints(args)

    def test_single_case_warm_start_loads_model_only_before_training(self) -> None:
        torch.manual_seed(23)
        source = _TinySingleSOIModel()
        target = _TinySingleSOIModel()
        with torch.no_grad():
            for parameter in target.parameters():
                parameter.zero_()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "released_weights.pt"
            torch.save(
                {
                    "model": source.state_dict(),
                    # This intentionally invalid state must never be passed to
                    # the fresh fine-tuning optimizer.
                    "optimizer_state_dict": {"invalid_official_state": True},
                    "step": 999_999,
                },
                checkpoint,
            )
            output_root = root / "output"
            args = build_parser().parse_args(
                [
                    "train",
                    "--data-root", str(root / "data"),
                    "--soi", "QPSK",
                    "--interference", "CommSignal2",
                    "--model-config",
                    str(PROJECT_ROOT / "config" / "model_config_icassp_baseline_wavenet.yaml"),
                    "--init-checkpoint", str(checkpoint),
                    "--output-dir", str(output_root),
                    "--learning-rate", "2e-6",
                    "--device", "cpu",
                ]
            )
            dataset = TensorDataset(
                torch.randn(2, 2, 8),
                torch.randn(2, 2, 8),
            )
            train_path = root / "train.h5"
            validation_path = root / "validation.h5"
            captured: dict[str, object] = {}

            def fake_train(**kwargs):
                captured.update(kwargs)
                output_dir = Path(kwargs["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "best.pt").touch()

            fake_datasets = (
                dataset,
                dataset,
                train_path,
                validation_path,
                None,
                0.0,
                None,
                0.0,
                {"validation_protocol": "unit-test"},
            )
            with (
                mock.patch("rfchallenge.cli._build_datasets", return_value=fake_datasets),
                mock.patch(
                    "rfchallenge.cli.build_single_soi_model",
                    return_value=(
                        target,
                        SimpleNamespace(model_type="icassp_baseline_wavenet"),
                    ),
                ),
                mock.patch(
                    "rfchallenge.cli.train_single_soi_model",
                    side_effect=fake_train,
                ),
            ):
                result = _train_one_case(
                    args,
                    "QPSK",
                    "CommSignal2",
                    logging.getLogger("test"),
                )

            self.assertEqual(result, output_root / "QPSK_CommSignal2" / "best.pt")
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], expected)
            self.assertIsNone(captured["resume_checkpoint"])
            self.assertEqual(captured["options"].learning_rate, 2e-6)
            metadata = captured["metadata"]
            self.assertEqual(
                metadata["initialization_mode"],
                "model_only_fresh_optimizer",
            )
            self.assertEqual(metadata["initial_checkpoint"], str(checkpoint.resolve()))

    def test_command_trains_eight_cases_then_tests_best_weight_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            for folder, suffix in (
                ("interferenceset_frame", "_raw_data.h5"),
                ("testset1_frame", "_test1_raw_data.h5"),
            ):
                target = data_root / folder
                target.mkdir(parents=True)
                for interference in INTERFERENCE_TYPES:
                    (target / f"{interference}{suffix}").touch()
            output_root = root / "output"
            args = build_parser().parse_args(
                [
                    "train-test-all",
                    "--data-root", str(data_root),
                    "--model-stage", "4",
                    "--output-dir", str(output_root),
                    "--epochs", "1",
                ]
            )
            calls: list[tuple[str, str]] = []
            tested: dict[tuple[str, str], Path] = {}

            def fake_train(_args, soi, interference, _logger):
                calls.append((soi, interference))
                case_dir = output_root / f"{soi}_{interference}"
                weights = case_dir / "weights" / "best_model_weights.pth"
                weights.parent.mkdir(parents=True, exist_ok=True)
                weights.touch()
                best = case_dir / "best.pt"
                best.touch()
                return best

            def fake_test(_args, _logger, checkpoints):
                tested.update(checkpoints)
                return {"case_count": 8}

            with (
                mock.patch("rfchallenge.cli._train_one_case", side_effect=fake_train),
                mock.patch("rfchallenge.cli._test_trained_all_cases", side_effect=fake_test),
            ):
                result = command_train_test_all(args, logging.getLogger("test"))

            self.assertEqual(result, 0)
            self.assertEqual(calls, list(OFFICIAL_CASES))
            self.assertEqual(set(tested), set(OFFICIAL_CASES))
            self.assertTrue(
                all(path.name == "best_model_weights.pth" for path in tested.values())
            )


if __name__ == "__main__":
    unittest.main()
