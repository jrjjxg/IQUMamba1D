import unittest

import torch

from models.IQUMamba1D_ComplexStage4 import ComplexStem1d
from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    IQUMamba1DComplexStateMamba,
)
from util.stage_registry import supported_stage_ids


class Stage299HybridTests(unittest.TestCase):
    def _model(self):
        return IQUMamba1DComplexStateMamba(
            input_size=256,
            input_channels=2,
            n_stages=4,
            features_per_stage=[4, 8, 16, 32],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=4,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            scan_backend="torch",
            scan_checkpoint=False,
            complex_stem_enable=True,
        )

    def test_stage_is_registered(self):
        self.assertIn(299, supported_stage_ids())

    def test_combines_stage290_stem_and_stage295_ssm(self):
        model = self._model()
        self.assertIsInstance(model.backbone.encoder.stem, ComplexStem1d)
        complex_state_layers = [
            layer
            for layer in model.backbone.encoder.mamba_layers
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertGreater(len(complex_state_layers), 0)

    def test_forward_backward_shape_and_finiteness(self):
        torch.manual_seed(299)
        model = self._model()
        output = model(torch.randn(2, 2, 256))
        self.assertEqual(tuple(output.shape), (2, 4, 256))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_stage295_default_keeps_real_stem(self):
        model = IQUMamba1DComplexStateMamba(
            input_size=256,
            input_channels=2,
            n_stages=4,
            features_per_stage=[4, 8, 16, 32],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=4,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
            mamba_d_state=2,
            mamba_d_conv=2,
            mamba_expand=1,
            scan_backend="torch",
            scan_checkpoint=False,
        )
        self.assertFalse(model.complex_stem_enable)
        self.assertNotIsInstance(model.backbone.encoder.stem, ComplexStem1d)


if __name__ == "__main__":
    unittest.main()
