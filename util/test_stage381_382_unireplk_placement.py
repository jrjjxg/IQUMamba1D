"""Regression tests for Stage381/382 UniRepLK placement ablations."""

import unittest
from pathlib import Path

import torch

from util.test_stage377_stage56_complexstate_unireplk_latent_mask import (
    _small_model,
)
from models.IQUResUNet1D_UniRepLKSeparator import (
    IQUResUNet1D_ComplexStateUniRepLKSeparator,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_model(filename: str):
    config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
    config._load_enc_config()
    return config, Create_Mamba_model(
        config,
        logger=None,
        input_size_=64,
        device_override=torch.device("cpu"),
    )


def _small_stage382() -> IQUResUNet1D_ComplexStateUniRepLKSeparator:
    return IQUResUNet1D_ComplexStateUniRepLKSeparator(
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
        rf_apply_stages=(),
        separator_unireplk_stages=(0, 1, 2),
        rf_residual_scale_init=0.05,
        rf_large_kernel=5,
        rf_ffn_factor=2,
        rf_layer_scale=1.0e-6,
        latent_mask_mode="real",
    )


class Stage381Tests(unittest.TestCase):
    def test_only_stage1_unireplk_is_removed(self):
        self.assertIn(381, supported_stage_ids())
        config, model = _config_model(
            "model_config_stage381_stage377_no_stage1_unireplk.yaml"
        )
        self.assertEqual(
            config.model_type,
            "resunet1d_complexstate_unireplk_no_stage1",
        )
        self.assertEqual(set(model.stage_rf.keys()), {"0", "2"})


class Stage382Tests(unittest.TestCase):
    def test_unireplk_runs_only_in_separator_path(self):
        model = _small_stage382().eval()
        self.assertEqual(set(model.stage_rf.keys()), set())
        self.assertEqual(set(model.separator_rf.keys()), {"0", "1", "2"})

        calls = []
        hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, _output, stage=stage: calls.append(stage)
            )
            for stage, module in model.separator_rf.items()
        ]
        x = torch.randn(2, 2, 64)
        with torch.no_grad():
            model._encode_skips(x)
        self.assertEqual(calls, [])

        with torch.no_grad():
            output = model(x)
        for hook in hooks:
            hook.remove()
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertEqual(set(calls), {"0", "1", "2"})

    def test_parameter_count_matches_stage377(self):
        stage377 = _small_model(latent_mask_mode="real")
        stage382 = _small_stage382()
        self.assertEqual(
            sum(parameter.numel() for parameter in stage377.parameters()),
            sum(parameter.numel() for parameter in stage382.parameters()),
        )

    def test_registration_and_factory(self):
        self.assertIn(382, supported_stage_ids())
        config, model = _config_model(
            "model_config_stage382_stage377_unireplk_separator.yaml"
        )
        self.assertEqual(
            config.model_type,
            "resunet1d_complexstate_unireplk_separator",
        )
        self.assertIsInstance(
            model,
            IQUResUNet1D_ComplexStateUniRepLKSeparator,
        )
        self.assertEqual(set(model.stage_rf.keys()), set())
        self.assertEqual(set(model.separator_rf.keys()), {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
