import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
    mamba_stub = types.ModuleType("mamba_ssm")
    mamba_stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)

    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(int(d_model), int(d_model))

        def forward(self, x):
            return self.projection(x)

    mamba_stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = mamba_stub


from models.IQUBiMamba1D_CoreUpgrades import (
    IQUBiMamba1D_IndependentComplexStateUniRepLK,
    IndependentComplexStateBiMambaLayer,
)
from models.IQUMamba1D_RecentRFModules import (
    ParallelFeatureDeltaAdapter,
    UniRepLKNetBlock1D,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "model_config_stage365_stage364_unireplk.yaml"


def _small_model():
    return IQUBiMamba1D_IndependentComplexStateUniRepLK(
        input_size=128,
        input_channels=2,
        n_stages=4,
        features_per_stage=[8, 16, 32, 64],
        conv_op=torch.nn.Conv1d,
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        deep_supervision=False,
        bimamba_apply_stages=[1, 3],
        bimamba_residual_scale_init=1.0,
        complex_state_d_state=2,
        complex_state_d_conv=4,
        complex_state_expand=1,
        complex_state_scan_checkpoint=False,
        complex_state_scan_backend="torch",
        complex_state_fusion_hidden=8,
        rf_apply_stages=[0, 1, 2],
        rf_residual_scale_init=0.05,
        rf_large_kernel=17,
        rf_ffn_factor=2,
        rf_layer_scale=1e-6,
    )


class Stage365StructureTests(unittest.TestCase):
    def test_exact_stage364_core_and_stage310_unireplk_placement(self):
        model = _small_model()
        self.assertEqual(len(model.encoder.stages), 4)
        self.assertEqual(len(model.decoder.stages), 3)
        self.assertEqual(model.stage12_active_indices, (1, 3))
        self.assertIsInstance(model.encoder.mamba_layers[0], torch.nn.Identity)
        self.assertIsInstance(
            model.encoder.mamba_layers[1], IndependentComplexStateBiMambaLayer
        )
        self.assertIsInstance(model.encoder.mamba_layers[2], torch.nn.Identity)
        self.assertIsInstance(
            model.encoder.mamba_layers[3], IndependentComplexStateBiMambaLayer
        )
        self.assertEqual(model.rf_apply_stages, (0, 1, 2))
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
        for adapter in model.stage_rf.values():
            self.assertIsInstance(adapter, ParallelFeatureDeltaAdapter)
            self.assertIsInstance(adapter.operator, UniRepLKNetBlock1D)
        self.assertFalse(hasattr(model, "cross_scale_blocks"))

    def test_unireplk_and_memory_share_the_same_stage_features(self):
        torch.manual_seed(365)
        model = _small_model()
        call_order = []
        memory_sources = {}
        rf_sources = {}
        rf_mains = {}
        handles = []
        for stage in range(4):
            handles.append(
                model.encoder.mamba_layers[stage].register_forward_hook(
                    lambda _module, inputs, _output, stage=stage: (
                        call_order.append(f"memory{stage}"),
                        memory_sources.__setitem__(stage, inputs[0]),
                        None,
                    )[-1]
                )
            )
            if str(stage) in model.stage_rf:
                handles.append(
                    model.stage_rf[str(stage)].register_forward_hook(
                        lambda _module, inputs, _output, stage=stage: (
                            call_order.append(f"rf{stage}"),
                            rf_sources.__setitem__(stage, inputs[0]),
                            rf_mains.__setitem__(stage, inputs[1]),
                            None,
                        )[-1]
                    )
                )

        x = torch.randn(1, 2, 128, requires_grad=True)
        output = model(x)
        for handle in handles:
            handle.remove()
        self.assertEqual(
            call_order,
            ["memory0", "rf0", "memory1", "rf1", "memory2", "rf2", "memory3"],
        )
        for stage in model.rf_apply_stages:
            self.assertIs(rf_sources[stage], memory_sources[stage])
            self.assertEqual(rf_mains[stage].shape, memory_sources[stage].shape)
        self.assertEqual(tuple(output.shape), (1, 4, 128))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        layer = model.encoder.mamba_layers[1]
        self.assertIsNotNone(layer.ssm_fwd.theta.grad)
        self.assertIsNotNone(layer.ssm_bwd.theta.grad)

    def test_adapter_adds_only_the_unireplk_delta_to_main(self):
        torch.manual_seed(366)
        operator = UniRepLKNetBlock1D(
            channels=8, kernel_size=17, ffn_factor=2, layer_scale=1e-3
        )
        adapter = ParallelFeatureDeltaAdapter(8, operator, scale_init=0.05)
        adapter.eval()
        source = torch.randn(2, 8, 32)
        main = torch.randn_like(source)

        with torch.no_grad():
            expected = main + adapter.residual_scale * adapter.norm(
                operator.residual_branch(source)
            )
            actual = adapter(source, main)
        torch.testing.assert_close(actual, expected)

    def test_no_weight_decay_keeps_both_component_parameter_sets(self):
        names = _small_model().no_weight_decay()
        self.assertIn("encoder.mamba_layers.1.ssm_fwd.theta", names)
        self.assertIn("encoder.mamba_layers.1.ssm_bwd.theta", names)
        self.assertIn("stage_rf.0.residual_scale", names)
        self.assertIn("stage_rf.1.residual_scale", names)
        self.assertIn("stage_rf.2.residual_scale", names)


class Stage365RegistrationTests(unittest.TestCase):
    def test_config_registry_and_main_mapping(self):
        self.assertIn(365, supported_stage_ids())
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(f'365: CONFIG_ROOT / "{CONFIG_NAME}"', main_source)

        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIG_NAME))
        config._load_enc_config()
        self.assertEqual(
            config.model_type, "bimamba_complex_state_independent_unireplk"
        )
        self.assertEqual(config.n_stages, 4)
        self.assertEqual(config.features_per_stage, [32, 64, 128, 256])
        self.assertEqual(config.strides, [1, 2, 2, 2])
        self.assertEqual(config.model_config["bimamba_apply_stages"], [1, 3])
        self.assertEqual(config.model_config["rf_apply_stages"], [0, 1, 2])

    def test_config_copies_stage364_and_stage310_component_settings(self):
        combined = MambaConfig(
            str(PROJECT_ROOT / "config" / CONFIG_NAME)
        ).model_config
        stage364 = MambaConfig(
            str(
                PROJECT_ROOT
                / "config"
                / "model_config_stage364_stage12_independent_complex_state.yaml"
            )
        ).model_config
        stage310 = MambaConfig(
            str(
                PROJECT_ROOT
                / "config"
                / "model_config_stage310_stage4_unireplk1d.yaml"
            )
        ).model_config
        core_keys = (
            "bimamba_apply_stages",
            "bimamba_residual_scale_init",
            "complex_state_d_state",
            "complex_state_d_conv",
            "complex_state_expand",
            "complex_state_scan_checkpoint",
            "complex_state_scan_backend",
            "complex_state_fusion_hidden",
        )
        rf_keys = (
            "rf_apply_stages",
            "rf_residual_scale_init",
            "rf_large_kernel",
            "rf_ffn_factor",
            "rf_layer_scale",
        )
        self.assertEqual(
            {key: combined[key] for key in core_keys},
            {key: stage364[key] for key in core_keys},
        )
        self.assertEqual(
            {key: combined[key] for key in rf_keys},
            {key: stage310[key] for key in rf_keys},
        )

    def test_factory_dispatches_stage365(self):
        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIG_NAME))
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=128,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(
            model, IQUBiMamba1D_IndependentComplexStateUniRepLK
        )
        self.assertEqual(len(model.encoder.stages), 4)
        self.assertEqual(model.stage12_active_indices, (1, 3))
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
