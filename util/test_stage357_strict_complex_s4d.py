"""Mathematical and integration contracts for strict-complex Stage 357."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from models.IQUMamba1D_MemoryRFStages import (
    IQUMamba1DStrictComplexS4D,
    StrictComplexS4DKernel,
    StrictComplexS4DLayer,
)
from rfchallenge.models import RFCHALLENGE_STAGE_CONFIGS
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rotate_hidden(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate hidden ``[R..., I...]`` complex features by one global phase."""

    real, imag = torch.chunk(x, 2, dim=1)
    cosine = torch.cos(torch.as_tensor(angle, dtype=x.dtype, device=x.device))
    sine = torch.sin(torch.as_tensor(angle, dtype=x.dtype, device=x.device))
    return torch.cat(
        (cosine * real - sine * imag, sine * real + cosine * imag), dim=1
    )


def _rotate_public(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate public interleaved ``[S1-I,S1-Q,...]`` source channels."""

    batch, channels, length = x.shape
    pairs = x.reshape(batch, channels // 2, 2, length)
    real, imag = pairs[:, :, 0], pairs[:, :, 1]
    cosine = torch.cos(torch.as_tensor(angle, dtype=x.dtype, device=x.device))
    sine = torch.sin(torch.as_tensor(angle, dtype=x.dtype, device=x.device))
    rotated = torch.stack(
        (cosine * real - sine * imag, sine * real + cosine * imag), dim=2
    )
    return rotated.reshape(batch, channels, length)


def _small_model(*, num_classes: int = 4) -> IQUMamba1DStrictComplexS4D:
    return IQUMamba1DStrictComplexS4D(
        input_size=64,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=num_classes,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        deep_supervision=False,
        d_state=8,
    )


class StrictComplexS4DMathTests(unittest.TestCase):
    def test_fft_kernel_equals_zoh_state_recurrence(self) -> None:
        torch.manual_seed(357)
        kernel_module = StrictComplexS4DKernel(2, d_state=4)
        length = 11
        inputs = torch.complex(
            torch.randn(3, 2, length), torch.randn(3, 2, length)
        )

        kernel = kernel_module(length)
        fft_output = torch.fft.ifft(
            torch.fft.fft(inputs, n=2 * length)
            * torch.fft.fft(kernel, n=2 * length),
            n=2 * length,
        )[..., :length]

        poles, coefficients, dt = kernel_module.continuous_parameters()
        discrete_a = torch.exp(poles * dt.unsqueeze(-1))
        discrete_b = torch.expm1(poles * dt.unsqueeze(-1)) / poles
        state = torch.zeros(
            inputs.size(0), 2, 4, dtype=inputs.dtype
        )
        recurrence = []
        for index in range(length):
            state = (
                discrete_a.unsqueeze(0) * state
                + discrete_b.unsqueeze(0) * inputs[:, :, index].unsqueeze(-1)
            )
            recurrence.append(
                (coefficients.unsqueeze(0) * state).sum(dim=-1)
            )
        recurrent_output = torch.stack(recurrence, dim=-1)

        torch.testing.assert_close(
            fft_output, recurrent_output, rtol=2e-5, atol=2e-5
        )

    def test_memory_layer_is_global_phase_equivariant(self) -> None:
        torch.manual_seed(357)
        layer = StrictComplexS4DLayer(8, d_state=8).eval()
        inputs = torch.randn(2, 8, 31)
        angle = 0.731
        with torch.no_grad():
            reference = layer(inputs)
            rotated = layer(_rotate_hidden(inputs, angle))
        torch.testing.assert_close(
            rotated, _rotate_hidden(reference, angle), rtol=3e-4, atol=3e-4
        )


class StrictComplexS4DModelTests(unittest.TestCase):
    def test_full_model_is_phase_equivariant_and_trainable(self) -> None:
        torch.manual_seed(357)
        model = _small_model().eval()
        inputs = torch.randn(1, 2, 64)
        angle = -0.417
        with torch.no_grad():
            reference = model(inputs)
            rotated = model(_rotate_public(inputs, angle))
        self.assertEqual(tuple(reference.shape), (1, 4, 64))
        torch.testing.assert_close(
            rotated, _rotate_public(reference, angle), rtol=8e-4, atol=8e-4
        )

        model.train()
        output = model(inputs.requires_grad_(True))
        output.square().mean().backward()
        self.assertIsNotNone(inputs.grad)
        self.assertTrue(torch.isfinite(inputs.grad).all())
        memory_layers = [
            layer for layer in model.encoder.mamba_layers
            if isinstance(layer, StrictComplexS4DLayer)
        ]
        self.assertEqual(len(memory_layers), 2)
        for layer in memory_layers:
            self.assertIsNotNone(layer.kernel.C.grad)
            self.assertTrue(torch.isfinite(layer.kernel.C.grad).all())

    def test_registration_and_rf_contract(self) -> None:
        self.assertIn(357, supported_stage_ids())
        self.assertIn(357, RFCHALLENGE_STAGE_CONFIGS)
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            '357: CONFIG_ROOT / "model_config_stage357_strict_complex_s4d.yaml"',
            main_source,
        )
        expected = {
            "model_config_stage357_strict_complex_s4d.yaml": 4,
            "model_config_stage357_rfchallenge.yaml": 2,
        }
        for filename, classes in expected.items():
            with self.subTest(filename=filename):
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                cfg = config.model_config
                self.assertEqual(
                    cfg["model_type"], "iqumamba_stage4_complex_s4d"
                )
                self.assertEqual(cfg["num_classes"], classes)
                self.assertTrue(all(width % 2 == 0 for width in cfg[
                    "features_per_stage"
                ]))


if __name__ == "__main__":
    unittest.main()
