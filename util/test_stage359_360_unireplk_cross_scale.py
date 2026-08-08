import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

if (
    "mamba_ssm" not in sys.modules
    and importlib.util.find_spec("mamba_ssm") is None
):
    mamba_stub = types.ModuleType("mamba_ssm")

    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(int(d_model), int(d_model))

        def forward(self, x):
            return self.projection(x)

    mamba_stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = mamba_stub

from models.IQUBiMamba1D import BiMambaLayer
from models.IQUBiMamba1D_CrossScaleAttention import (
    CompressedGlobalCrossAttention,
)
from models.IQUMamba1D import MambaLayer
from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    UniRepLKNetBlock1D,
)
from models.IQUMamba1D_UniRepLKCrossScale import (
    IQUBiMamba1D_UniRepLKCrossScale,
    IQUMamba1D_UniRepLKCrossScale,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    359: "model_config_stage359_bimamba_cross_scale_unireplk.yaml",
    360: "model_config_stage360_iqumamba_cross_scale_unireplk.yaml",
}


def _small_model(model_class):
    return model_class(
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
        cross_scale_query_stages=[2],
        cross_scale_global_stage=3,
        cross_scale_kv_tokens=8,
        cross_scale_num_heads=4,
        cross_scale_residual_scale_init=0.01,
        rf_apply_stages=[0, 1, 2],
        rf_residual_scale_init=0.05,
        rf_large_kernel=17,
        rf_ffn_factor=2,
        rf_layer_scale=1e-6,
    )


class UniRepLKCrossScaleStructureTests(unittest.TestCase):
    def test_stage359_is_bimamba_plus_unireplk_and_cross_scale(self):
        model = _small_model(IQUBiMamba1D_UniRepLKCrossScale)
        self.assertTrue(
            any(isinstance(module, BiMambaLayer) for module in model.modules())
        )
        self._assert_shared_structure(model)

    def test_stage360_is_unidirectional_plus_unireplk_and_cross_scale(self):
        model = _small_model(IQUMamba1D_UniRepLKCrossScale)
        self.assertTrue(
            any(isinstance(module, MambaLayer) for module in model.modules())
        )
        self.assertFalse(
            any(isinstance(module, BiMambaLayer) for module in model.modules())
        )
        self._assert_shared_structure(model)

    def _assert_shared_structure(self, model):
        self.assertEqual(len(model.encoder.stages), 4)
        self.assertEqual(len(model.decoder.stages), 3)
        self.assertEqual(model.rf_apply_stages, (0, 1, 2))
        self.assertEqual(set(model.stage_rf.keys()), {"0", "1", "2"})
        for block in model.stage_rf.values():
            self.assertIsInstance(block, FeatureResidualAdapter)
            self.assertIsInstance(block.operator, UniRepLKNetBlock1D)
        self.assertEqual(model.cross_scale_query_stages, (2,))
        self.assertEqual(model.cross_scale_global_stage, 3)
        self.assertIsInstance(
            model.cross_scale_blocks["2"],
            CompressedGlobalCrossAttention,
        )

    def test_both_models_run_all_unireplk_blocks_and_backpropagate(self):
        for model_class in (
            IQUBiMamba1D_UniRepLKCrossScale,
            IQUMamba1D_UniRepLKCrossScale,
        ):
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(359)
                model = _small_model(model_class)
                calls = {stage: 0 for stage in model.stage_rf}
                handles = []
                for stage, block in model.stage_rf.items():
                    handles.append(block.register_forward_hook(
                        lambda _module, _inputs, _output, stage=stage:
                        calls.__setitem__(stage, calls[stage] + 1)
                    ))
                x = torch.randn(2, 2, 64, requires_grad=True)
                output = model(x)
                for handle in handles:
                    handle.remove()
                self.assertEqual(tuple(output.shape), (2, 4, 64))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(calls, {"0": 1, "1": 1, "2": 1})
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_unireplk_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            IQUMamba1D_UniRepLKCrossScale(
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
                cross_scale_query_stages=[2],
                cross_scale_global_stage=3,
                rf_apply_stages=[4],
            )


class UniRepLKCrossScaleRegistrationTests(unittest.TestCase):
    def test_configs_are_registered_and_keep_four_stage_unet(self):
        expected_types = {
            359: "bimamba_cross_scale_unireplk",
            360: "iqumamba_cross_scale_unireplk",
        }
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(
                    f'{stage}: CONFIG_ROOT / "{filename}"',
                    main_source,
                )
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                config._load_enc_config()
                self.assertEqual(config.model_type, expected_types[stage])
                self.assertEqual(config.n_stages, 4)
                self.assertEqual(config.features_per_stage, [32, 64, 128, 256])
                self.assertEqual(config.strides, [1, 2, 2, 2])
                self.assertEqual(config.model_config["rf_apply_stages"], [0, 1, 2])

    def test_factory_dispatches_to_the_requested_direction(self):
        expected_classes = {
            359: IQUBiMamba1D_UniRepLKCrossScale,
            360: IQUMamba1D_UniRepLKCrossScale,
        }
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                model = Create_Mamba_model(
                    config,
                    logger=None,
                    input_size_=64,
                    device_override=torch.device("cpu"),
                )
                self.assertIsInstance(model, expected_classes[stage])
                self.assertEqual(len(model.encoder.stages), 4)
                self.assertEqual(set(model.stage_rf.keys()), {"0", "1", "2"})

    def test_new_configs_preserve_their_source_module_settings(self):
        source_names = {
            "cross_scale": "model_config_stage300_stage4_cross_scale_single.yaml",
            "unireplk": "model_config_stage310_stage4_unireplk1d.yaml",
        }
        sources = {
            key: MambaConfig(str(PROJECT_ROOT / "config" / name)).model_config
            for key, name in source_names.items()
        }
        cross_scale_keys = (
            "cross_scale_query_stages",
            "cross_scale_global_stage",
            "cross_scale_kv_tokens",
            "cross_scale_num_heads",
            "cross_scale_dropout",
            "cross_scale_residual_scale_init",
            "cross_scale_evidence_gate",
        )
        unireplk_keys = (
            "rf_residual_scale_init",
            "rf_apply_stages",
            "rf_large_kernel",
            "rf_ffn_factor",
            "rf_layer_scale",
        )
        for filename in CONFIGS.values():
            cfg = MambaConfig(str(PROJECT_ROOT / "config" / filename)).model_config
            self.assertEqual(
                {key: cfg[key] for key in cross_scale_keys},
                {key: sources["cross_scale"][key] for key in cross_scale_keys},
            )
            self.assertEqual(
                {key: cfg[key] for key in unireplk_keys},
                {key: sources["unireplk"][key] for key in unireplk_keys},
            )


if __name__ == "__main__":
    unittest.main()
