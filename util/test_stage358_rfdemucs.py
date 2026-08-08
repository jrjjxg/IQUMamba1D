"""Structure, paper-case, and RF pipeline contracts for Stage 358."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from models.rfdemucs import RFDEMUCS, RFDEMUCSBLSTM
from rfchallenge.cli import _rfdemucs_case_settings
from rfchallenge.models import (
    RFCHALLENGE_STAGE_CONFIGS,
    build_single_soi_model,
    resolve_stage_config,
)
from util.config import MambaConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RFDEMUCSModelTests(unittest.TestCase):
    def test_small_waveform_model_preserves_length_and_backpropagates(self) -> None:
        torch.manual_seed(358)
        model = RFDEMUCS(
            hidden=4,
            depth=2,
            kernel_size=8,
            stride=2,
            resample=2,
            lstm_layers=1,
        )
        inputs = torch.randn(2, 2, 65, requires_grad=True)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (2, 2, 65))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertIsInstance(model.lstm, RFDEMUCSBLSTM)
        self.assertFalse(model.normalize)

    def test_factory_builds_single_soi_contract(self) -> None:
        model, config = build_single_soi_model(
            resolve_stage_config(358),
            frame_length=64,
            device="cpu",
            model_overrides={
                "rfdemucs_hidden": 4,
                "rfdemucs_depth": 2,
                "rfdemucs_lstm_layers": 1,
            },
        )
        with torch.no_grad():
            output = model(torch.randn(1, 2, 64))
        self.assertEqual(tuple(output.shape), (1, 2, 64))
        self.assertEqual(config.model_type, "rfchallenge_rfdemucs")


class RFDEMUCSRegistrationTests(unittest.TestCase):
    def test_config_and_stage_registration(self) -> None:
        path = PROJECT_ROOT / "config" / "model_config_stage358_rfchallenge.yaml"
        self.assertEqual(RFCHALLENGE_STAGE_CONFIGS[358], path)
        config = MambaConfig(str(path)).model_config
        self.assertEqual(config["model_type"], "rfchallenge_rfdemucs")
        self.assertEqual(config["rfdemucs_depth"], 5)
        self.assertEqual(config["rfdemucs_kernel_size"], 8)
        self.assertFalse(config["rfdemucs_normalize"])

    def test_paper_case_specific_architecture_and_learning_rates(self) -> None:
        path = resolve_stage_config(358)
        default_architecture, default_lr = _rfdemucs_case_settings(
            path, "OFDMQPSK", "CommSignal2"
        )
        self.assertEqual(
            default_architecture,
            {
                "rfdemucs_hidden": 64,
                "rfdemucs_stride": 2,
                "rfdemucs_resample": 2,
            },
        )
        self.assertEqual(default_lr, 3e-5)

        comm2_architecture, comm2_lr = _rfdemucs_case_settings(
            path, "QPSK", "CommSignal2"
        )
        self.assertEqual(comm2_architecture["rfdemucs_hidden"], 64)
        self.assertEqual(comm2_lr, 3e-4)

        exceptional, exceptional_lr = _rfdemucs_case_settings(
            path, "QPSK", "CommSignal3"
        )
        self.assertEqual(
            exceptional,
            {
                "rfdemucs_hidden": 80,
                "rfdemucs_stride": 4,
                "rfdemucs_resample": 4,
            },
        )
        self.assertEqual(exceptional_lr, 3e-4)


if __name__ == "__main__":
    unittest.main()
