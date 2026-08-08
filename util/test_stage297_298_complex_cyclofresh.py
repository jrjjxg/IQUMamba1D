import unittest
import importlib.util
import sys
import types

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

from models.IQUBiMamba1D_EstimatedCycloFRESH import (
    IQUBiMamba1D_EstimatedCycloFRESH,
)
from models.IQUMamba1D_ComplexStage4 import ComplexStem1d
from models.IQUMamba1D_EstimatedCycloFRESH import (
    IQUMamba1D_EstimatedCycloFRESH,
)
from util.stage_registry import supported_stage_ids


def _small_kwargs():
    return {
        "input_size": 64,
        "input_channels": 2,
        "n_stages": 4,
        "features_per_stage": [4, 8, 16, 32],
        "conv_op": torch.nn.Conv1d,
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "n_conv_per_stage": [1, 1, 1, 1],
        "num_classes": 4,
        "n_conv_per_stage_decoder": [1, 1, 1, 1],
        "deep_supervision": False,
        "estimated_cyclofresh_hidden_channels": 2,
        "estimated_cyclofresh_kernel_size": 3,
        "complex_stem_enable": True,
    }


class ComplexCycloFRESHStagesTest(unittest.TestCase):
    def test_stage79_and_stage197_backbones_receive_c1_stem(self):
        for model_class in (
            IQUMamba1D_EstimatedCycloFRESH,
            IQUBiMamba1D_EstimatedCycloFRESH,
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class(**_small_kwargs())
                self.assertTrue(model.complex_stem_enable)
                self.assertIsInstance(model.backbone.encoder.stem, ComplexStem1d)

    def test_existing_stage79_default_is_unchanged(self):
        kwargs = _small_kwargs()
        kwargs["complex_stem_enable"] = False
        model = IQUMamba1D_EstimatedCycloFRESH(**kwargs)
        self.assertFalse(model.complex_stem_enable)
        self.assertNotIsInstance(model.backbone.encoder.stem, ComplexStem1d)
        self.assertEqual(model.no_weight_decay(), set())

    def test_stage297_preserves_stage290_no_weight_decay_policy(self):
        model = IQUMamba1D_EstimatedCycloFRESH(**_small_kwargs())
        names = model.no_weight_decay()
        self.assertTrue(
            any(name.endswith(".log_scale") for name in names),
            names,
        )
        self.assertTrue(
            any(name.endswith(".bias") for name in names),
            names,
        )
        self.assertTrue(
            all(name.startswith("backbone.encoder.stem.") for name in names),
            names,
        )

    def test_forward_backward_shape_and_finiteness(self):
        torch.manual_seed(297)
        for model_class in (
            IQUMamba1D_EstimatedCycloFRESH,
            IQUBiMamba1D_EstimatedCycloFRESH,
        ):
            with self.subTest(model=model_class.__name__):
                model = model_class(**_small_kwargs())
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

    def test_new_stage_ids_are_registered(self):
        stages = supported_stage_ids()
        self.assertIn(297, stages)
        self.assertIn(298, stages)


if __name__ == "__main__":
    unittest.main()
