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

from models.IQUMamba1D_CombinedStages import (
    Stage301ComplexStateCrossScale,
    Stage302BiFRESHComplexBottleneck,
    Stage303ComplexStemBiMambaCrossScale,
    Stage304Stage298299Fusion,
    Stage305GatedFRESHComplexState,
)
from models.IQUMamba1D_ComplexStage4 import ComplexStem1d
from models.IQUMamba1D_ComplexStateMamba import ComplexStateMambaLayer
from util.stage_registry import supported_stage_ids


def _common():
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


def _state():
    return dict(
        mamba_d_state=2,
        mamba_d_conv=2,
        mamba_expand=1,
        scan_backend="torch",
        scan_checkpoint=False,
    )


class _Scale(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return x * self.value


class Stage301To305CombinedModelTests(unittest.TestCase):
    def test_all_stages_are_registered(self):
        supported = supported_stage_ids()
        for stage in range(301, 306):
            self.assertIn(stage, supported)

    def test_stage301_forward(self):
        model = Stage301ComplexStateCrossScale(
            **_common(),
            **_state(),
            cross_scale_query_stages=[2],
            cross_scale_global_stage=3,
            cross_scale_kv_tokens=8,
            cross_scale_num_heads=4,
        )
        output = model(torch.randn(1, 2, 64))
        self.assertEqual(tuple(output.shape), (1, 4, 64))
        self.assertIn("2", model.cross_scale_blocks)

    def test_stage302_replaces_only_deepest_bimamba(self):
        model = Stage302BiFRESHComplexBottleneck(
            **_common(),
            conv_op=torch.nn.Conv1d,
            **_state(),
        )
        layers = model.core.backbone.encoder.mamba_layers
        self.assertIsInstance(
            layers[model.complex_bottleneck_index],
            ComplexStateMambaLayer,
        )
        self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))

    def test_stage303_has_complex_stem_and_no_fresh_adapter(self):
        model = Stage303ComplexStemBiMambaCrossScale(
            **_common(),
            conv_op=torch.nn.Conv1d,
            cross_scale_query_stages=[2],
            cross_scale_global_stage=3,
            cross_scale_kv_tokens=8,
            cross_scale_num_heads=4,
        )
        self.assertIsInstance(model.core.encoder.stem, ComplexStem1d)
        self.assertFalse(hasattr(model, "estimated_cyclofresh_adapter"))
        self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))

    def test_stage304_initially_averages_two_branches(self):
        model = Stage304Stage298299Fusion(
            stage298=_Scale(2.0),
            stage299=_Scale(4.0),
            fusion_logit_init=0.0,
        )
        output = model(torch.ones(1, 2, 8))
        self.assertTrue(torch.allclose(output, torch.full_like(output, 3.0)))

    def test_stage305_starts_with_small_fresh_gate(self):
        model = Stage305GatedFRESHComplexState(
            **_common(),
            **_state(),
            fresh_gate_logit_init=-3.0,
        )
        gate = torch.sigmoid(model.fresh_gate_logit).item()
        self.assertLess(gate, 0.05)
        self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))

    def test_real_configs_build_through_model_factory(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        config_dir = Path(__file__).resolve().parents[1] / "config"
        config_names = {
            301: "model_config_stage301_stage299_cross_scale.yaml",
            302: "model_config_stage302_stage298_complex_bottleneck.yaml",
            303: "model_config_stage303_complex_bimamba_cross_scale.yaml",
            304: "model_config_stage304_stage298_stage299_fusion.yaml",
            305: "model_config_stage305_stage299_gated_fresh.yaml",
        }
        for stage, config_name in config_names.items():
            with self.subTest(stage=stage):
                config = MambaConfig(str(config_dir / config_name))
                model = Create_Mamba_model(
                    config,
                    logger=None,
                    input_size_=64,
                    device_override=torch.device("cpu"),
                )
                output = model(torch.randn(1, 2, 64))
                self.assertEqual(tuple(output.shape), (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
