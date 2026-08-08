"""Structural and runtime contracts for Stages 351-353."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

# Install lightweight optional-dependency stubs in CPU-only test environments.
import util.test_stage317_322_fdconv_unirep_ablation  # noqa: F401

from models.IQUMamba1D_ComplexStage4 import ComplexStem1d
from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    IQUMamba1DComplexStateMamba,
)
from models.IQUMamba1D_MemoryRFStages import (
    IQUMamba1DS4D,
    IQUMamba1DS4DUniRepLK,
    S4DLayer,
)
from models.IQUMamba1D_RecentRFModules import UniRepLKNetBlock1D
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    351: "model_config_stage351_complex_stem_rf_mamba3.yaml",
    352: "model_config_stage352_complex_stem_s4d.yaml",
    353: "model_config_stage353_complex_stem_s4d_unireplk.yaml",
}


def _common() -> dict:
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
        deep_supervision=False,
    )


class ComplexCombinationStructureTests(unittest.TestCase):
    def test_stage351_is_exactly_complex_stem_plus_full_stage333(self) -> None:
        model = IQUMamba1DComplexStateMamba(
            **_common(),
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            scan_backend="torch",
            scan_checkpoint=False,
            mamba_discretization="exponential_trapezoidal",
            cyclic_theta_enable=True,
            cyclic_frequencies=[0.0, 1 / 64, -1 / 64, 0.05],
            reliability_enable=True,
            complex_stem_enable=True,
        )
        self.assertIsInstance(model.backbone.encoder.stem, ComplexStem1d)
        layers = [
            layer for layer in model.backbone.encoder.mamba_layers
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertEqual(layer.ssm.discretization, "exponential_trapezoidal")
            self.assertTrue(layer.ssm.cyclic_theta_enable)
            self.assertTrue(layer.ssm.reliability_enable)
        self.assertFalse(hasattr(model, "stage_rf"))
        self.assertFalse(hasattr(model, "input_adapter"))
        self._assert_forward_backward(model)

    def test_stage352_is_complex_stem_plus_s4d_only(self) -> None:
        model = IQUMamba1DS4D(
            **_common(), d_state=8, complex_stem_enable=True
        )
        self.assertTrue(model.uses_complex_stem)
        self.assertIsInstance(model.backbone.encoder.stem, ComplexStem1d)
        self.assertEqual(
            sum(isinstance(layer, S4DLayer)
                for layer in model.backbone.encoder.mamba_layers),
            2,
        )
        self.assertFalse(hasattr(model, "stage_rf"))
        self._assert_forward_backward(model)

    def test_stage353_adds_real_unireplk_at_stages_zero_to_two(self) -> None:
        model = IQUMamba1DS4DUniRepLK(
            **_common(),
            d_state=8,
            complex_stem_enable=True,
            unireplk_large_kernel=9,
            unireplk_ffn_factor=2,
        )
        self.assertIsInstance(model.encoder.stem, ComplexStem1d)
        self.assertEqual(
            sum(isinstance(layer, S4DLayer)
                for layer in model.encoder.mamba_layers),
            2,
        )
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
        self.assertTrue(all(
            isinstance(adapter.operator, UniRepLKNetBlock1D)
            for adapter in model.stage_rf.values()
        ))
        self._assert_forward_backward(model)
        parameters = dict(model.named_parameters())
        for name in model.no_weight_decay():
            self.assertIn(name, parameters)

    def _assert_forward_backward(self, model: torch.nn.Module) -> None:
        x = torch.randn(1, 2, 64, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (1, 4, 64))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class ComplexCombinationRegistrationTests(unittest.TestCase):
    def test_main_configs_and_flags_are_independently_registered(self) -> None:
        expected_types = {
            351: "iqumamba_rf_mamba3",
            352: "iqumamba_stage4_s4d",
            353: "iqumamba_stage4_s4d_unireplk",
        }
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(f'{stage}: CONFIG_ROOT / "{filename}"', source)
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                cfg = config.model_config
                self.assertEqual(cfg["model_type"], expected_types[stage])
                self.assertTrue(cfg["complex_stem_enable"])
        self.assertNotIn("unireplk_large_kernel", MambaConfig(
            str(PROJECT_ROOT / "config" / CONFIGS[351])
        ).model_config)
        self.assertNotIn("unireplk_large_kernel", MambaConfig(
            str(PROJECT_ROOT / "config" / CONFIGS[352])
        ).model_config)
        self.assertIn("unireplk_large_kernel", MambaConfig(
            str(PROJECT_ROOT / "config" / CONFIGS[353])
        ).model_config)


if __name__ == "__main__":
    unittest.main()
