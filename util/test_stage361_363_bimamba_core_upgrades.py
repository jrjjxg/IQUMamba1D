import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
    mamba_stub = types.ModuleType("mamba_ssm")
    mamba_stub.__spec__ = importlib.machinery.ModuleSpec(
        "mamba_ssm", loader=None
    )

    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.d_model = int(d_model)
            self.projection = torch.nn.Linear(self.d_model, self.d_model)

        def forward(self, x):
            return self.projection(x)

    mamba_stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = mamba_stub


from models.IQUBiMamba1D_CoreUpgrades import (
    ComplexStateBiMambaLayer,
    HydraBiMambaLayer,
    IQUBiMamba1D_ComplexState,
    IQUBiMamba1D_Hydra,
    IQUBiMamba1D_IndependentComplexState,
    IQUBiMamba1D_MultiScale,
    IndependentComplexStateBiMambaLayer,
    MultiScaleBiMambaLayer,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    361: "model_config_stage361_stage12_hydra.yaml",
    362: "model_config_stage362_stage12_complex_state.yaml",
    363: "model_config_stage363_stage12_multiscale_bimamba.yaml",
    364: "model_config_stage364_stage12_independent_complex_state.yaml",
}
EXPECTED_TYPES = {
    361: "bimamba_hydra",
    362: "bimamba_complex_state",
    363: "bimamba_multiscale",
    364: "bimamba_complex_state_independent",
}
EXPECTED_CLASSES = {
    361: IQUBiMamba1D_Hydra,
    362: IQUBiMamba1D_ComplexState,
    363: IQUBiMamba1D_MultiScale,
    364: IQUBiMamba1D_IndependentComplexState,
}
EXPECTED_LAYERS = {
    361: HydraBiMambaLayer,
    362: ComplexStateBiMambaLayer,
    363: MultiScaleBiMambaLayer,
    364: IndependentComplexStateBiMambaLayer,
}


def _small_model(model_class):
    common = dict(
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
    )
    if model_class is IQUBiMamba1D_Hydra:
        common.update(
            hydra_d_state=4,
            hydra_d_conv=7,
            hydra_expand=2,
            hydra_headdim=16,
            hydra_ngroups=1,
            hydra_chunk_size=32,
            hydra_prefer_fused_scan=False,
        )
    elif model_class in {
        IQUBiMamba1D_ComplexState,
        IQUBiMamba1D_IndependentComplexState,
    }:
        common.update(
            complex_state_d_state=2,
            complex_state_d_conv=4,
            complex_state_expand=1,
            complex_state_scan_checkpoint=False,
            complex_state_scan_backend="torch",
            complex_state_fusion_hidden=8,
        )
    else:
        common.update(
            multiscale_d_state=4,
            multiscale_global_d_conv=4,
            multiscale_expand=1,
            multiscale_local_kernels=[3, 7, 15],
            multiscale_local_scale_init=0.1,
        )
    return model_class(**common)


class Stage12CoreUpgradeStructureTests(unittest.TestCase):
    def test_all_variants_keep_exact_stage12_geometry_and_placement(self):
        for stage, model_class in EXPECTED_CLASSES.items():
            with self.subTest(stage=stage):
                model = _small_model(model_class)
                self.assertEqual(len(model.encoder.stages), 4)
                self.assertEqual(len(model.encoder.mamba_layers), 4)
                self.assertEqual(len(model.decoder.stages), 3)
                self.assertEqual(model.stage12_active_indices, (1, 3))
                self.assertIsInstance(model.encoder.mamba_layers[0], torch.nn.Identity)
                self.assertIsInstance(model.encoder.mamba_layers[2], torch.nn.Identity)
                self.assertIsInstance(
                    model.encoder.mamba_layers[1], EXPECTED_LAYERS[stage]
                )
                self.assertIsInstance(
                    model.encoder.mamba_layers[3], EXPECTED_LAYERS[stage]
                )

    def test_non_four_stage_or_changed_placement_is_rejected(self):
        kwargs = dict(
            input_size=64,
            input_channels=2,
            features_per_stage=[8, 16, 32, 64],
            conv_op=torch.nn.Conv1d,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=4,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "exactly four"):
            IQUBiMamba1D_MultiScale(n_stages=5, **kwargs)
        with self.assertRaisesRegex(ValueError, "stages \\[1, 3\\]"):
            IQUBiMamba1D_MultiScale(
                n_stages=4, bimamba_apply_stages=[0, 1, 2, 3], **kwargs
            )

    def test_hydra_uses_quasiseparable_reference_scan(self):
        layer = _small_model(IQUBiMamba1D_Hydra).encoder.mamba_layers[1]
        x = torch.randn(2, layer.dim, 11, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(layer.hydra.last_scan_backend, "torch_reference")
        self.assertFalse(hasattr(layer, "mamba_fwd"))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_complex_state_is_conjugate_shared_and_adaptive(self):
        layer = _small_model(IQUBiMamba1D_ComplexState).encoder.mamba_layers[1]
        theta = layer.ssm._theta_values(1.0)
        self.assertTrue(torch.equal(layer.ssm._theta_values(-1.0), -theta))
        self.assertTrue(layer.conjugate_directions)
        x = torch.randn(2, layer.dim, 13, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(layer.last_direction_gate)
        self.assertTrue(
            torch.allclose(
                layer.last_direction_gate,
                torch.full_like(layer.last_direction_gate, 0.5),
            )
        )
        self.assertEqual(layer.last_scan_backends, ("torch", "torch"))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_multiscale_has_exact_local_widths_and_original_global_pair(self):
        layer = _small_model(IQUBiMamba1D_MultiScale).encoder.mamba_layers[1]
        self.assertEqual(layer.local_kernel_sizes, (3, 7, 15))
        self.assertEqual(
            tuple(conv.kernel_size[0] for conv in layer.local_convs),
            (3, 7, 15),
        )
        self.assertTrue(hasattr(layer, "mamba_fwd"))
        self.assertTrue(hasattr(layer, "mamba_bwd"))
        x = torch.randn(2, layer.dim, 17, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)
        weight_sum = layer.last_local_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sum, torch.ones_like(weight_sum)))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_independent_complex_state_has_two_disjoint_stage295_cores(self):
        layer = _small_model(
            IQUBiMamba1D_IndependentComplexState
        ).encoder.mamba_layers[1]
        self.assertIsInstance(layer, IndependentComplexStateBiMambaLayer)
        self.assertIsNot(layer.ssm_fwd, layer.ssm_bwd)
        self.assertNotEqual(
            layer.ssm_fwd.theta.untyped_storage().data_ptr(),
            layer.ssm_bwd.theta.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            layer.ssm_fwd.in_proj.weight.untyped_storage().data_ptr(),
            layer.ssm_bwd.in_proj.weight.untyped_storage().data_ptr(),
        )
        self.assertFalse(layer.conjugate_directions)
        self.assertTrue(layer.independent_directions)

        x = torch.randn(2, layer.dim, 13, requires_grad=True)
        output = layer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(layer.last_scan_backends, ("torch", "torch"))
        output.square().mean().backward()
        self.assertIsNotNone(layer.ssm_fwd.theta.grad)
        self.assertIsNotNone(layer.ssm_bwd.theta.grad)
        self.assertTrue(torch.isfinite(layer.ssm_fwd.theta.grad).all())
        self.assertTrue(torch.isfinite(layer.ssm_bwd.theta.grad).all())


class Stage12CoreUpgradeEndToEndTests(unittest.TestCase):
    def test_all_core_upgrade_models_forward_and_backward(self):
        for model_class in EXPECTED_CLASSES.values():
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(361)
                model = _small_model(model_class)
                x = torch.randn(1, 2, 128, requires_grad=True)
                output = model(x)
                self.assertEqual(tuple(output.shape), (1, 4, 128))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())


class Stage12CoreUpgradeRegistrationTests(unittest.TestCase):
    def test_configs_and_cli_registry(self):
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(
                    f'{stage}: CONFIG_ROOT / "{filename}"', main_source
                )
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                config._load_enc_config()
                self.assertEqual(config.model_type, EXPECTED_TYPES[stage])
                self.assertEqual(config.n_stages, 4)
                self.assertEqual(config.features_per_stage, [32, 64, 128, 256])
                self.assertEqual(config.strides, [1, 2, 2, 2])
                self.assertEqual(config.bimamba_apply_stages, [1, 3])

    def test_factory_dispatches_all_core_upgrade_configs(self):
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                model = Create_Mamba_model(
                    config,
                    logger=None,
                    input_size_=512,
                    device_override=torch.device("cpu"),
                )
                self.assertIsInstance(model, EXPECTED_CLASSES[stage])
                self.assertEqual(len(model.encoder.stages), 4)
                self.assertEqual(model.stage12_active_indices, (1, 3))
                with torch.no_grad():
                    output = model(torch.randn(1, 2, 512))
                self.assertEqual(tuple(output.shape), (1, 4, 512))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
