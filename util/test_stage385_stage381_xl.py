"""Regression tests for Stage385: the XL-scaled Stage381."""

from pathlib import Path
import unittest

import torch

# Reuse the Stage377 test module's CPU-safe Mamba stub when mamba_ssm is absent.
from util.test_stage377_stage56_complexstate_unireplk_latent_mask import (
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "model_config_stage385_stage381_xl.yaml"
XL_WIDTHS = [96, 192, 384, 768]
EXPECTED_PARAMETERS = 15_359_514


class Stage385Tests(unittest.TestCase):
    def test_registration_and_stage381_config_contract(self):
        self.assertIn(385, supported_stage_ids())
        main_text = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            f'385: CONFIG_ROOT / "{CONFIG_NAME}"',
            main_text,
        )

        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIG_NAME))
        config._load_enc_config()
        self.assertEqual(
            config.model_type,
            "resunet1d_complexstate_unireplk_no_stage1_xl",
        )
        self.assertEqual(config.features_per_stage, XL_WIDTHS)
        self.assertEqual(config.model_config["rf_apply_stages"], [0, 2])
        self.assertEqual(config.model_config["complex_state_fusion_hidden"], 192)

    def test_factory_adapts_modules_and_preserves_stage381_placement(self):
        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIG_NAME))
        config._load_enc_config()
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(
            model,
            IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
        )
        self.assertEqual(list(model.encoder.output_channels), XL_WIDTHS)
        self.assertEqual(set(model.stage_rf.keys()), {"0", "2"})
        self.assertNotIn("1", model.stage_rf)
        self.assertEqual(
            [head.in_channels for head in model.latent_mask_heads],
            XL_WIDTHS,
        )
        self.assertEqual(
            [head.out_channels for head in model.latent_mask_heads],
            [2 * width for width in XL_WIDTHS],
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            EXPECTED_PARAMETERS,
        )


if __name__ == "__main__":
    unittest.main()
