import importlib.util
import sys
import types
import unittest

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
import models.IQUBiMamba1D_CrossScaleAttention as bimamba_cross_scale_module
import models.IQUMamba1D_CrossScaleAttention as iqumamba_cross_scale_module
from models.IQUBiMamba1D_CrossScaleAttention import (
    CompressedGlobalCrossAttention,
)
from models.IQUMamba1D import MambaLayer
from models.IQUMamba1D_CrossScaleAttention import (
    IQUMamba1D_CrossScaleAttention,
)
from util.stage_registry import supported_stage_ids


def _small_model():
    return IQUMamba1D_CrossScaleAttention(
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
    )


class Stage300IQUMambaCrossScaleTest(unittest.TestCase):
    def test_stage_is_registered(self):
        self.assertIn(300, supported_stage_ids())

    def test_uses_unidirectional_mamba_and_cross_scale_attention(self):
        model = _small_model()
        self.assertEqual(model.cross_scale_query_stages, (2,))
        self.assertEqual(model.cross_scale_global_stage, 3)
        self.assertIsInstance(
            model.cross_scale_blocks["2"],
            CompressedGlobalCrossAttention,
        )
        self.assertTrue(
            any(isinstance(module, MambaLayer) for module in model.modules())
        )
        self.assertFalse(
            any(isinstance(module, BiMambaLayer) for module in model.modules())
        )

    def test_stage300_and_stage235_share_the_exact_attention_class(self):
        self.assertIs(
            iqumamba_cross_scale_module.CompressedGlobalCrossAttention,
            bimamba_cross_scale_module.CompressedGlobalCrossAttention,
        )

    def test_stage300_and_stage235_attention_configs_match(self):
        from pathlib import Path
        from util.config import MambaConfig

        config_root = Path(__file__).resolve().parents[1] / "config"
        stage235 = MambaConfig(str(
            config_root / "model_config_bimamba_cross_scale_single.yaml"
        )).model_config
        stage300 = MambaConfig(str(
            config_root / "model_config_stage300_stage4_cross_scale_single.yaml"
        )).model_config
        keys = (
            "cross_scale_query_stages",
            "cross_scale_global_stage",
            "cross_scale_kv_tokens",
            "cross_scale_num_heads",
            "cross_scale_dropout",
            "cross_scale_residual_scale_init",
            "cross_scale_evidence_gate",
        )
        self.assertEqual(
            {key: stage300[key] for key in keys},
            {key: stage235[key] for key in keys},
        )

    def test_forward_backward_shape_and_finiteness(self):
        torch.manual_seed(300)
        model = _small_model()
        output = model(torch.randn(2, 2, 64))
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_invalid_query_stage_is_rejected(self):
        model_class = _small_model().__class__
        with self.assertRaises(ValueError):
            model_class(
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
                cross_scale_query_stages=[3],
                cross_scale_global_stage=3,
            )


if __name__ == "__main__":
    unittest.main()
