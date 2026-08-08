from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    274: (
        "model_config_stage274_wavenet15_interleaved_mamba.yaml",
        "icassp_wavenet_interleaved_mamba",
    ),
    275: (
        "model_config_stage275_wavenet15_no_mamba.yaml",
        "icassp_baseline_wavenet",
    ),
}


class WaveNet15MambaAblationRegistrationTests(unittest.TestCase):
    def test_models_are_selectable_and_depth_matched(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main = (ROOT / "main.py").read_text(encoding="utf-8")
        configs = {}
        for stage, (filename, model_type) in STAGES.items():
            self.assertIn(stage, supported_stage_ids())
            self.assertIn(f'{stage}: CONFIG_ROOT / "{filename}"', main)

            config = MambaConfig(str(ROOT / "config" / filename))
            config._load_enc_config()
            self.assertEqual(config.model_type, model_type)
            self.assertEqual(config.residual_channels, 128)
            self.assertEqual(config.residual_layers, 15)
            self.assertEqual(config.dilation_cycle_length, 10)
            configs[stage] = config

        self.assertEqual(configs[274].mamba_insert_after_block, 10)
        self.assertEqual(configs[274].mamba_downsample_factor, 4)


if __name__ == "__main__":
    unittest.main()
