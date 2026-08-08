"""Contracts for loading the RF Challenge's released PyTorch WaveNet."""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from models.icassp_baseline_wavenet import ICASPBaselineWaveNet
from rfchallenge.models import (
    OFFICIAL_STARTER_ROOT,
    _register_official_checkpoint_compatibility,
    load_checkpoint,
    resolve_checkpoint_path,
)
from util.config import MambaConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> ICASPBaselineWaveNet:
    return ICASPBaselineWaveNet(
        input_channels=2,
        num_classes=2,
        residual_channels=4,
        residual_layers=2,
        dilation_cycle_length=2,
    )


class OfficialWaveNetCheckpointTests(unittest.TestCase):
    def test_reproduction_uses_official_state_dict_names(self) -> None:
        keys = set(_small_model().state_dict())
        self.assertTrue(any(key.startswith("residual_layers.0.") for key in keys))
        self.assertFalse(any(key.startswith("residual_blocks.") for key in keys))

    def test_official_directory_and_model_payload_load_strictly(self) -> None:
        source = _small_model()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            torch.save({"model": source.state_dict()}, checkpoint)
            self.assertEqual(resolve_checkpoint_path(directory), checkpoint)

            target = _small_model()
            with torch.no_grad():
                for parameter in target.parameters():
                    parameter.zero_()
            load_checkpoint(target, directory, device="cpu", strict=True)
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], expected)

    def test_old_local_residual_block_keys_remain_loadable(self) -> None:
        source = _small_model()
        legacy = {
            key.replace("residual_layers.", "residual_blocks."): value
            for key, value in source.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            torch.save({"model_state_dict": legacy}, checkpoint)
            target = _small_model()
            load_checkpoint(target, checkpoint, device="cpu", strict=True)
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], expected)

    def test_pickled_official_src_config_loads_without_external_repository(self) -> None:
        source = _small_model()
        _register_official_checkpoint_compatibility()
        official_config_module = importlib.import_module("src.config_torchwavenet")
        official_config = official_config_module.Config(model_dir="unused")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            torch.save(
                {"model": source.state_dict(), "cfg": official_config},
                checkpoint,
            )

            root_text = str(OFFICIAL_STARTER_ROOT)
            while root_text in sys.path:
                sys.path.remove(root_text)
            for module_name in tuple(sys.modules):
                if module_name == "src" or module_name.startswith("src."):
                    sys.modules.pop(module_name, None)

            target = _small_model()
            load_checkpoint(target, checkpoint, device="cpu", strict=True)
            for name, expected in source.state_dict().items():
                torch.testing.assert_close(target.state_dict()[name], expected)
            self.assertIn(root_text, sys.path)

    def test_official_architecture_config_matches_release(self) -> None:
        path = PROJECT_ROOT / "config" / "model_config_icassp_baseline_wavenet.yaml"
        config = MambaConfig(str(path)).model_config
        self.assertEqual(config["input_channels"], 2)
        self.assertEqual(config["num_classes"], 2)
        self.assertEqual(config["residual_channels"], 128)
        self.assertEqual(config["residual_layers"], 30)
        self.assertEqual(config["dilation_cycle_length"], 10)


if __name__ == "__main__":
    unittest.main()
