"""Regression tests for Stage380's lightweight temporal mask separator."""

import unittest
from pathlib import Path

import torch

from util.test_stage377_stage56_complexstate_unireplk_latent_mask import (
    _small_model,
)
from models.IQUResUNet1D_LightMaskSeparator import (
    IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator,
    LightweightTemporalMaskEstimator,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_stage380() -> IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator:
    return IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator(
        input_size=64,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        conv_op=torch.nn.Conv1d,
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        deep_supervision=False,
        bimamba_apply_stages=(1, 3),
        bimamba_residual_scale_init=1.0,
        complex_state_d_state=2,
        complex_state_d_conv=3,
        complex_state_expand=1,
        complex_state_scan_checkpoint=False,
        complex_state_scan_backend="torch",
        complex_state_fusion_hidden=8,
        rf_apply_stages=(0, 1, 2),
        rf_residual_scale_init=0.05,
        rf_large_kernel=5,
        rf_ffn_factor=2,
        rf_layer_scale=1.0e-6,
        latent_mask_mode="real",
        separator_kernel_size=5,
        separator_dilations=(1, 2),
        separator_residual_scale_init=0.1,
    )


class Stage380Tests(unittest.TestCase):
    def test_temporal_separator_and_simplex_masks(self):
        model = _small_stage380().eval()
        x = torch.randn(2, 2, 64)
        decoder_calls = []
        hook = model.decoder.register_forward_hook(
            lambda *_: decoder_calls.append(1)
        )
        with torch.no_grad():
            skips = model._encode_skips(x)
            masks = [
                model._make_masks(features, head)
                for features, head in zip(skips, model.latent_mask_heads)
            ]
            output = model(x)
        hook.remove()

        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertEqual(len(decoder_calls), 1)
        self.assertTrue(torch.isfinite(output).all())
        for head, mask in zip(model.latent_mask_heads, masks):
            self.assertIsInstance(head, LightweightTemporalMaskEstimator)
            self.assertEqual(
                [block.dilation for block in head.blocks],
                [1, 2],
            )
            torch.testing.assert_close(
                mask.sum(dim=1),
                torch.ones_like(mask[:, 0]),
                atol=1.0e-5,
                rtol=1.0e-5,
            )

    def test_parameter_overhead_is_lightweight(self):
        stage377 = _small_model(latent_mask_mode="real")
        stage380 = _small_stage380()
        base_parameters = sum(
            parameter.numel() for parameter in stage377.parameters()
        )
        enhanced_parameters = sum(
            parameter.numel() for parameter in stage380.parameters()
        )
        self.assertGreater(enhanced_parameters, base_parameters)
        self.assertLess(enhanced_parameters, int(base_parameters * 1.20))

    def test_registration_and_factory(self):
        self.assertIn(380, supported_stage_ids())
        config = MambaConfig(
            str(
                PROJECT_ROOT
                / "config"
                / "model_config_stage380_stage377_light_separator.yaml"
            )
        )
        config._load_enc_config()
        self.assertEqual(
            config.model_type,
            "resunet1d_complexstate_unireplk_light_separator",
        )
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(
            model, IQUResUNet1D_ComplexStateUniRepLK_LightMaskSeparator
        )
        self.assertEqual(model.separator_dilations, (1, 2))
        self.assertFalse(hasattr(model.decoder, "skip_processors"))


if __name__ == "__main__":
    unittest.main()
