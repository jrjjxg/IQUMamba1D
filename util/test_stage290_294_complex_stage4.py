import math
import unittest

import torch
from torch.nn import functional as F

from models.IQUMamba1D_ComplexStage4 import (
    IQUMamba1DComplexStage4,
    IQUMamba1DComplexStemC1,
    MagnitudeControlledMamba1d,
    complex_cat,
    hidden_to_public_sources,
    merge_complex,
    split_complex,
)
from models.icassp_complex_wavenet import ComplexConv1d
from util.stage_registry import supported_stage_ids


def rotate_hidden(x, angle):
    real, imag = split_complex(x)
    cosine, sine = math.cos(angle), math.sin(angle)
    return merge_complex(
        cosine * real - sine * imag,
        sine * real + cosine * imag,
    )


def rotate_public(x, angle):
    batch, channels, length = x.shape
    paired = x.view(batch, channels // 2, 2, length)
    real, imag = paired[:, :, 0], paired[:, :, 1]
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.stack(
        (
            cosine * real - sine * imag,
            sine * real + cosine * imag,
        ),
        dim=2,
    ).reshape_as(x)


def small_model(**overrides):
    kwargs = dict(
        input_size=64,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        strict_complex_output=False,
        use_equivariant_mamba=False,
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=1,
    )
    kwargs.update(overrides)
    return IQUMamba1DComplexStage4(**kwargs)


class ComplexConvolutionCorrectnessTests(unittest.TestCase):
    def test_fused_kernel_matches_complex_convolution_definition(self):
        torch.manual_seed(290)
        layer = ComplexConv1d(
            3, 5, kernel_size=3, dilation=2, padding=2
        )
        real = torch.randn(2, 3, 41)
        imag = torch.randn(2, 3, 41)
        actual_real, actual_imag = split_complex(
            layer(merge_complex(real, imag))
        )
        expected_real = F.conv1d(
            real,
            layer.weight_real,
            padding=2,
            dilation=2,
        ) - F.conv1d(
            imag,
            layer.weight_imag,
            padding=2,
            dilation=2,
        )
        expected_imag = F.conv1d(
            real,
            layer.weight_imag,
            padding=2,
            dilation=2,
        ) + F.conv1d(
            imag,
            layer.weight_real,
            padding=2,
            dilation=2,
        )
        torch.testing.assert_close(actual_real, expected_real)
        torch.testing.assert_close(actual_imag, expected_imag)

    def test_complex_cat_preserves_all_real_then_all_imag_layout(self):
        first = merge_complex(
            torch.full((1, 2, 3), 1.0),
            torch.full((1, 2, 3), 10.0),
        )
        second = merge_complex(
            torch.full((1, 1, 3), 2.0),
            torch.full((1, 1, 3), 20.0),
        )
        real, imag = split_complex(complex_cat((first, second)))
        self.assertEqual(real[:, :2].unique().item(), 1.0)
        self.assertEqual(real[:, 2:].unique().item(), 2.0)
        self.assertEqual(imag[:, :2].unique().item(), 10.0)
        self.assertEqual(imag[:, 2:].unique().item(), 20.0)

    def test_public_source_layout_is_interleaved(self):
        hidden = merge_complex(
            torch.tensor([[[1.0], [2.0]]]),
            torch.tensor([[[10.0], [20.0]]]),
        )
        public = hidden_to_public_sources(hidden)
        torch.testing.assert_close(
            public.flatten(),
            torch.tensor([1.0, 10.0, 2.0, 20.0]),
        )


class ComplexStage4Tests(unittest.TestCase):
    def test_c1_to_c5_shapes_finiteness_and_backward(self):
        torch.manual_seed(291)
        variants = [
            IQUMamba1DComplexStemC1(
                input_size=64,
                input_channels=2,
                n_stages=4,
                features_per_stage=[4, 8, 16, 32],
                kernel_sizes=[3, 3, 3, 3],
                strides=[1, 2, 2, 2],
                n_conv_per_stage=[1, 1, 1, 1],
                num_classes=4,
                n_conv_per_stage_decoder=[1, 1, 1, 1],
            ),
            small_model(
                strict_complex_output=False,
                use_equivariant_mamba=False,
            ),
            small_model(
                strict_complex_output=True,
                use_equivariant_mamba=False,
            ),
            small_model(
                strict_complex_output=False,
                use_equivariant_mamba=True,
            ),
            small_model(
                strict_complex_output=True,
                use_equivariant_mamba=True,
            ),
        ]
        for index, model in enumerate(variants, start=1):
            with self.subTest(complex_stage=index):
                x = torch.randn(2, 2, 64)
                output = model(x)
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

    def test_magnitude_controlled_mamba_is_rotation_equivariant(self):
        torch.manual_seed(292)
        layer = MagnitudeControlledMamba1d(
            8, d_state=4, d_conv=2, expand=1
        )
        x = torch.randn(2, 8, 37)
        angle = 0.63
        torch.testing.assert_close(
            layer(rotate_hidden(x, angle)),
            rotate_hidden(layer(x), angle),
            rtol=2e-4,
            atol=2e-4,
        )

    def test_c5_is_end_to_end_rotation_equivariant(self):
        torch.manual_seed(293)
        model = small_model(
            strict_complex_output=True,
            use_equivariant_mamba=True,
        )
        torch.nn.init.normal_(model.decoder.output_head.weight_real)
        torch.nn.init.normal_(model.decoder.output_head.weight_imag)
        x = torch.randn(1, 2, 64)
        angle = -0.47
        torch.testing.assert_close(
            model(rotate_public(x, angle)),
            rotate_public(model(x), angle),
            rtol=8e-4,
            atol=8e-4,
        )


class ComplexStage4RegistrationTests(unittest.TestCase):
    def test_c1_to_c5_stage_ids_are_registered(self):
        stages = supported_stage_ids()
        for stage in range(290, 295):
            self.assertIn(stage, stages)


if __name__ == "__main__":
    unittest.main()
