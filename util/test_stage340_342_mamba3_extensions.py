"""Focused tests for the Stage 340-342 Mamba-3 extensions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

# Install lightweight optional-dependency stubs before importing Stage-4.
import util.test_stage317_322_fdconv_unirep_ablation  # noqa: F401

import models.IQUMamba1D_ComplexStateMamba as complex_state_module
from models.IQUMamba1D import MambaLayer
from models.IQUMamba1D_ComplexStage4 import ComplexStem1d
from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    ComplexStateSelectiveSSM,
)
from models.IQUMamba1D_EstimatedCycloFRESH import EstimatedCycloFRESHAdapter1D
from models.IQUMamba1D_Mamba3Extensions import (
    IQUMamba1DFullRFCombination,
    IQUMamba1DOfficialMamba3,
    OfficialMamba3Layer,
    _official_mamba3_class,
)
from models.IQUMamba1D_RecentRFModules import UniRepLKNetBlock1D
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    340: "model_config_stage340_stage4_official_mamba3.yaml",
    341: "model_config_stage341_rf_mamba3_cyclic_reliability_fast.yaml",
    342: "model_config_stage342_fresh_complex_rf_mamba3_unireplk.yaml",
    349: "model_config_stage349_complex_rf_mamba3_unireplk.yaml",
}


class _FakeMamba3(nn.Module):
    instances: list["_FakeMamba3"] = []

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.projection = nn.Linear(kwargs["d_model"], kwargs["d_model"])
        self.instances.append(self)

    def forward(self, x):
        return self.projection(x)


def _small_common() -> dict:
    return dict(
        input_size=64,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
    )


class OfficialMamba3Tests(unittest.TestCase):
    def setUp(self):
        _FakeMamba3.instances.clear()
        self.patch = mock.patch.object(
            sys.modules["mamba_ssm"], "Mamba3", _FakeMamba3, create=True
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_public_class_arguments_and_token_layouts(self):
        patch_layer = OfficialMamba3Layer(
            12, d_state=32, expand=2, headdim=7, chunk_size=64
        )
        self.assertEqual(patch_layer.mamba.kwargs["d_state"], 32)
        self.assertEqual(patch_layer.mamba.kwargs["headdim"], 8)
        self.assertFalse(patch_layer.mamba.kwargs["is_mimo"])

        x = torch.randn(2, 12, 19, requires_grad=True)
        output = patch_layer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(
            patch_layer.last_scan_backend, "official_mamba3_siso_triton"
        )
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

        channel_layer = OfficialMamba3Layer(15, channel_token=True)
        channel_input = torch.randn(2, 7, 3, 5)
        self.assertEqual(channel_layer(channel_input).shape, channel_input.shape)

    def test_official_module_path_is_supported_when_top_level_is_not_exported(self):
        package = sys.modules["mamba_ssm"]
        modules_package = __import__("types").ModuleType("mamba_ssm.modules")
        direct_module = __import__("types").ModuleType("mamba_ssm.modules.mamba3")
        direct_module.Mamba3 = _FakeMamba3
        with mock.patch.dict(
            sys.modules,
            {
                "mamba_ssm.modules": modules_package,
                "mamba_ssm.modules.mamba3": direct_module,
            },
        ), mock.patch.object(package, "Mamba3", create=True) as exported:
            del package.Mamba3
            self.assertIs(_official_mamba3_class(), _FakeMamba3)

    def test_stage340_replaces_only_original_stage4_mamba(self):
        model = IQUMamba1DOfficialMamba3(**_small_common())
        self.assertEqual(model.replaced_layers, 2)
        self.assertEqual(
            sum(isinstance(layer, OfficialMamba3Layer)
                for layer in model.backbone.encoder.mamba_layers),
            2,
        )
        self.assertFalse(any(
            isinstance(layer, MambaLayer)
            for layer in model.backbone.encoder.mamba_layers
        ))
        output = model(torch.randn(1, 2, 64))
        self.assertEqual(output.shape, (1, 4, 64))

    def test_forward_failure_has_no_silent_local_fallback(self):
        layer = OfficialMamba3Layer(8)
        layer.mamba.forward = mock.Mock(side_effect=RuntimeError("kernel missing"))
        with self.assertRaisesRegex(RuntimeError, "No local fallback"):
            layer(torch.randn(1, 8, 9))


class FastRFTests(unittest.TestCase):
    def test_stage341_flags_and_fused_backend_selection(self):
        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIGS[341]))
        cfg = config.model_config
        self.assertEqual(cfg["mamba_discretization"], "exponential_euler")
        self.assertTrue(cfg["cyclic_theta_enable"])
        self.assertTrue(cfg["reliability_enable"])
        self.assertTrue(cfg["require_mamba_fused_scan"])
        self.assertFalse(cfg["scan_checkpoint"])

        layer = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            expand=1,
            discretization=cfg["mamba_discretization"],
            cyclic_theta_enable=True,
            cyclic_frequencies=cfg["cyclic_frequencies"],
            reliability_enable=True,
            scan_backend="auto",
            require_mamba_fused_scan=True,
        )
        with mock.patch.object(
            complex_state_module, "_mamba_selective_scan_fn", object()
        ):
            self.assertEqual(
                layer._select_scan_backend(mock.Mock(is_cuda=True)),
                "mamba_cuda",
            )

    def test_stage341_refuses_cuda_fallback_when_kernel_is_missing(self):
        layer = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            expand=1,
            cyclic_theta_enable=True,
            cyclic_frequencies=[0.0, 0.05],
            reliability_enable=True,
            require_mamba_fused_scan=True,
        )
        with mock.patch.object(
            complex_state_module, "_mamba_selective_scan_fn", None
        ):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                layer._select_scan_backend(mock.Mock(is_cuda=True))


class FullCombinationTests(unittest.TestCase):
    def _build(self, *, use_cyclofresh: bool = True) -> IQUMamba1DFullRFCombination:
        return IQUMamba1DFullRFCombination(
            **_small_common(),
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            scan_checkpoint=False,
            scan_backend="torch",
            cyclic_frequencies=[0.0, 1 / 64, -1 / 64, 0.05],
            estimated_cyclofresh_enable=use_cyclofresh,
            unireplk_large_kernel=9,
            unireplk_ffn_factor=2,
        )

    def test_exact_requested_module_placement(self):
        model = self._build()
        self.assertIsInstance(model.input_adapter, EstimatedCycloFRESHAdapter1D)
        self.assertIsInstance(model.encoder.stem, ComplexStem1d)
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
        self.assertTrue(all(
            isinstance(adapter.operator, UniRepLKNetBlock1D)
            for adapter in model.stage_rf.values()
        ))

        memory_indexes = [
            index
            for index, layer in enumerate(model.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertEqual(memory_indexes, [1, 3])
        self.assertFalse(any(
            isinstance(layer, MambaLayer) for layer in model.encoder.mamba_layers
        ))
        for index in memory_indexes:
            ssm = model.encoder.mamba_layers[index].ssm
            self.assertEqual(ssm.discretization, "exponential_trapezoidal")
            self.assertTrue(ssm.cyclic_theta_enable)
            self.assertTrue(ssm.reliability_enable)

    def test_forward_backward_and_weight_decay_contract(self):
        torch.manual_seed(342)
        model = self._build()
        x = torch.randn(1, 2, 64, requires_grad=True)
        output = model(x)
        self.assertEqual(output.shape, (1, 4, 64))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(sum(
            parameter.grad.abs().sum().item()
            for parameter in model.stage_rf.parameters()
            if parameter.grad is not None
        ), 0.0)
        parameters = dict(model.named_parameters())
        for name in model.no_weight_decay():
            self.assertIn(name, parameters)

    def test_stage349_removes_only_input_cyclofresh(self):
        model = self._build(use_cyclofresh=False)
        self.assertIsInstance(model.input_adapter, nn.Identity)
        self.assertFalse(model.uses_cyclofresh)
        self.assertIsInstance(model.encoder.stem, ComplexStem1d)
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
        memory_indexes = [
            index for index, layer in enumerate(model.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertEqual(memory_indexes, [1, 3])


class RegistrationAndFactoryTests(unittest.TestCase):
    def test_all_three_stages_are_independently_registered(self):
        expected_types = {
            340: "iqumamba_stage4_official_mamba3",
            341: "iqumamba_rf_mamba3_fast",
            342: "iqumamba_full_rf_mamba3_combination",
            349: "iqumamba_full_rf_mamba3_combination",
        }
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(f'{stage}: CONFIG_ROOT / "{filename}"', source)
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                config._load_enc_config()
                self.assertEqual(config.model_type, expected_types[stage])

    def test_configs_build_through_public_factory(self):
        from util.utils import Create_Mamba_model

        with mock.patch.object(
            sys.modules["mamba_ssm"], "Mamba3", _FakeMamba3, create=True
        ):
            for stage in (340, 341, 342, 349):
                with self.subTest(stage=stage):
                    config = MambaConfig(
                        str(PROJECT_ROOT / "config" / CONFIGS[stage])
                    )
                    model = Create_Mamba_model(
                        config,
                        logger=None,
                        input_size_=64,
                        device_override=torch.device("cpu"),
                    )
                    self.assertIsInstance(model, nn.Module)
                    output = model(torch.randn(1, 2, 64))
                    self.assertEqual(output.shape, (1, 4, 64))
                    self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
