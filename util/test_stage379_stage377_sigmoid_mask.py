"""Regression tests for Stage379's independent sigmoid mask ablation."""

import unittest
from pathlib import Path

import torch

from util.test_stage377_stage56_complexstate_unireplk_latent_mask import (
    _small_model,
)
from models.IQUResUNet1D_ComplexStateUniRepLK_LatentMask import (
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Stage379Tests(unittest.TestCase):
    def test_parameter_count_matches_stage377(self):
        softmax_model = _small_model(latent_mask_mode="real")
        sigmoid_model = _small_model(latent_mask_mode="sigmoid")
        self.assertEqual(
            sum(parameter.numel() for parameter in softmax_model.parameters()),
            sum(parameter.numel() for parameter in sigmoid_model.parameters()),
        )

    def test_independent_sigmoid_masks(self):
        model = _small_model(latent_mask_mode="sigmoid").eval()
        x = torch.randn(2, 2, 64)

        with torch.no_grad():
            skips = model._encode_skips(x)
            masks = [
                model._make_masks(features, head)
                for features, head in zip(skips, model.latent_mask_heads)
            ]
            output = model(x)

        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(output).all())
        for mask in masks:
            self.assertTrue(torch.all(mask >= 0.0))
            self.assertTrue(torch.all(mask <= 1.0))
            torch.testing.assert_close(mask, torch.full_like(mask, 0.5))
            torch.testing.assert_close(
                mask.sum(dim=1),
                torch.ones_like(mask[:, 0]),
            )

        with torch.no_grad():
            model.latent_mask_heads[0].bias[0] = 2.0
            changed = model._make_masks(skips[0], model.latent_mask_heads[0])
        self.assertFalse(
            torch.allclose(
                changed.sum(dim=1),
                torch.ones_like(changed[:, 0]),
            )
        )

    def test_registration_and_factory(self):
        self.assertIn(379, supported_stage_ids())
        config = MambaConfig(
            str(
                PROJECT_ROOT
                / "config"
                / "model_config_stage379_stage377_latent_mask_sigmoid.yaml"
            )
        )
        config._load_enc_config()
        self.assertEqual(
            config.model_type,
            "resunet1d_complexstate_unireplk_latent_mask_sigmoid",
        )
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(
            model, IQUResUNet1D_ComplexStateUniRepLK_LatentMask
        )
        self.assertEqual(model.latent_mask_mode, "sigmoid")
        self.assertFalse(hasattr(model.decoder, "skip_processors"))


if __name__ == "__main__":
    unittest.main()
