import math
import unittest

import torch

from models.icassp_baseline_wavenet import ICASPBaselineWaveNet
from models.icassp_complex_wavenet import (
    ComplexConv1d,
    ComplexWaveNetResidualBlock,
    ConjugateLeakageAdapter,
    ICASPComplexWaveNet,
)
from util.stage_registry import supported_stage_ids


def _rotate_hidden(x, angle):
    real, imag = torch.chunk(x, 2, dim=1)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.cat(
        (cosine * real - sine * imag, sine * real + cosine * imag),
        dim=1,
    )


def _rotate_public_sources(x, angle):
    batch, channels, length = x.shape
    sources = x.view(batch, channels // 2, 2, length)
    real, imag = sources[:, :, 0], sources[:, :, 1]
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.stack(
        (cosine * real - sine * imag, sine * real + cosine * imag),
        dim=2,
    ).reshape_as(x)


class ComplexPrimitiveTests(unittest.TestCase):
    def test_complex_convolution_is_rotation_equivariant(self):
        torch.manual_seed(10)
        layer = ComplexConv1d(3, 5, kernel_size=3, dilation=2, padding=2)
        x = torch.randn(2, 6, 71)
        angle = 0.81
        torch.testing.assert_close(
            layer(_rotate_hidden(x, angle)),
            _rotate_hidden(layer(x), angle),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_complex_gated_block_is_rotation_equivariant(self):
        torch.manual_seed(11)
        block = ComplexWaveNetResidualBlock(4, dilation=4)
        x = torch.randn(2, 8, 83)
        angle = -0.49
        output, skip = block(x)
        rotated_output, rotated_skip = block(_rotate_hidden(x, angle))
        torch.testing.assert_close(
            rotated_output,
            _rotate_hidden(output, angle),
            rtol=4e-5,
            atol=4e-5,
        )
        torch.testing.assert_close(
            rotated_skip,
            _rotate_hidden(skip, angle),
            rtol=4e-5,
            atol=4e-5,
        )

    def test_conjugate_adapter_starts_as_exact_identity(self):
        adapter = ConjugateLeakageAdapter(max_coefficient=0.15)
        x = torch.randn(3, 2, 47)
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
        self.assertEqual(float(adapter.coefficient_magnitude().detach()), 0.0)


class ComplexWaveNetTests(unittest.TestCase):
    def test_full_model_is_rotation_equivariant_with_public_output_layout(self):
        torch.manual_seed(12)
        model = ICASPComplexWaveNet(
            num_classes=4,
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
            complex_layers=4,
            strict_complex_output=True,
        )
        # The production output is intentionally zero initialized. Randomize
        # it here so the equivariance test cannot pass trivially.
        torch.nn.init.normal_(model.output_projection.weight_real)
        torch.nn.init.normal_(model.output_projection.weight_imag)
        x = torch.randn(2, 2, 65)
        angle = 0.37
        output = model(x)
        rotated_output = model(_rotate_public_sources(x, angle))
        torch.testing.assert_close(
            rotated_output,
            _rotate_public_sources(output, angle),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_c1_to_c7_shapes_and_backward(self):
        torch.manual_seed(13)
        variants = (
            dict(complex_layers=0, strict_complex_output=False),
            dict(complex_layers=2, strict_complex_output=False),
            dict(complex_layers=3, strict_complex_output=False),
            dict(complex_layers=4, strict_complex_output=True),
            dict(
                complex_layers=4,
                strict_complex_output=True,
                use_conjugate_adapter=True,
            ),
            # Stage 288: fully complex backbone with an unconstrained real head.
            dict(complex_layers=4, strict_complex_output=False),
            # Stage 289: adjacent-number strict-head control (same topology as 286).
            dict(complex_layers=4, strict_complex_output=True),
        )
        for kwargs in variants:
            with self.subTest(**kwargs):
                model = ICASPComplexWaveNet(
                    num_classes=4,
                    residual_channels=8,
                    residual_layers=4,
                    dilation_cycle_length=2,
                    **kwargs,
                )
                x = torch.randn(2, 2, 69)
                output = model(x)
                self.assertEqual(tuple(output.shape), (2, 4, 69))
                self.assertTrue(torch.isfinite(output).all())
                loss = output.square().mean()
                loss.backward()
                self.assertTrue(
                    all(
                        parameter.grad is None
                        or torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    )
                )

    def test_full_complex_model_does_not_increase_parameter_count(self):
        baseline = ICASPBaselineWaveNet(
            num_classes=4,
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
        )
        complex_model = ICASPComplexWaveNet(
            num_classes=4,
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
            complex_layers=4,
            strict_complex_output=True,
        )
        baseline_parameters = sum(p.numel() for p in baseline.parameters())
        complex_parameters = sum(p.numel() for p in complex_model.parameters())
        self.assertLess(complex_parameters, baseline_parameters)


class ComplexStageRegistrationTests(unittest.TestCase):
    def test_new_stage_ids_are_supported(self):
        stages = supported_stage_ids()
        for stage in range(283, 290):
            self.assertIn(stage, stages)


if __name__ == "__main__":
    unittest.main()
