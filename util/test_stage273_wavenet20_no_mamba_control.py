from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "model_config_stage273_wavenet20_no_mamba.yaml"


class Stage273RegistrationTests(unittest.TestCase):
    def test_control_is_selectable_and_matches_stage261_depth(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(273, supported_stage_ids())
        self.assertIn(f'273: CONFIG_ROOT / "{CONFIG_NAME}"', main)

        config = MambaConfig(str(ROOT / "config" / CONFIG_NAME))
        config._load_enc_config()
        self.assertEqual(config.model_type, "icassp_baseline_wavenet")
        self.assertEqual(config.residual_channels, 128)
        self.assertEqual(config.residual_layers, 20)
        self.assertEqual(config.dilation_cycle_length, 10)


if __name__ == "__main__":
    unittest.main()
