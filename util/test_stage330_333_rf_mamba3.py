"""Tests for the controlled RF Mamba-3 Stage 330-333 ablations."""

from __future__ import annotations

import math
import os
import unittest
from unittest import mock
from pathlib import Path

import torch

from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    ComplexStateSelectiveSSM,
    IQUMamba1DRealStateTrapReliability,
    complex_prefix_scan,
)
from models.complex_scan_cuda import native_complex_scan
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sequential_complex_scan(a_real, a_imag, u_real, u_imag):
    h_real = torch.zeros_like(u_real[:, 0])
    h_imag = torch.zeros_like(u_imag[:, 0])
    real_outputs = []
    imag_outputs = []
    for index in range(u_real.shape[1]):
        next_real = (
            a_real[:, index] * h_real
            - a_imag[:, index] * h_imag
            + u_real[:, index]
        )
        next_imag = (
            a_real[:, index] * h_imag
            + a_imag[:, index] * h_real
            + u_imag[:, index]
        )
        h_real, h_imag = next_real, next_imag
        real_outputs.append(h_real)
        imag_outputs.append(h_imag)
    return torch.stack(real_outputs, 1), torch.stack(imag_outputs, 1)


class ComplexInputScanTests(unittest.TestCase):
    def test_parallel_scan_supports_complex_state_input(self):
        torch.manual_seed(330)
        shape = (2, 19, 3, 4)
        magnitude = 0.95 * torch.rand(shape)
        angle = torch.randn(shape)
        a_real = magnitude * torch.cos(angle)
        a_imag = magnitude * torch.sin(angle)
        u_real = torch.randn(shape)
        u_imag = torch.randn(shape)
        actual = complex_prefix_scan(a_real, a_imag, u_real, u_imag)
        expected = _sequential_complex_scan(
            a_real, a_imag, u_real, u_imag
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-5)

    def test_omitted_imaginary_input_preserves_stage295_contract(self):
        torch.manual_seed(331)
        shape = (1, 13, 2, 3)
        a_real = 0.8 * torch.rand(shape)
        a_imag = 0.2 * torch.rand(shape)
        u_real = torch.randn(shape)
        implicit = complex_prefix_scan(a_real, a_imag, u_real)
        explicit = complex_prefix_scan(
            a_real, a_imag, u_real, torch.zeros_like(u_real)
        )
        torch.testing.assert_close(implicit, explicit)


class TrapezoidalSSMTests(unittest.TestCase):
    def test_lambda_one_reduces_to_exponential_euler(self):
        torch.manual_seed(332)
        euler = ComplexStateSelectiveSSM(
            6,
            d_state=3,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            discretization="exponential_euler",
        )
        trapezoidal = ComplexStateSelectiveSSM(
            6,
            d_state=3,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            discretization="exponential_trapezoidal",
        )
        trapezoidal.load_state_dict(euler.state_dict(), strict=False)
        with torch.no_grad():
            trapezoidal.trapezoid_lambda_weight.zero_()
            trapezoidal.trapezoid_lambda_bias.fill_(30.0)
        x = torch.randn(2, 23, 6)
        torch.testing.assert_close(
            trapezoidal(x), euler(x), rtol=2e-4, atol=2e-5
        )

    def test_trapezoid_parameters_and_input_receive_gradients(self):
        torch.manual_seed(333)
        model = ComplexStateSelectiveSSM(
            8,
            d_state=4,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            discretization="exponential_trapezoidal",
        )
        x = torch.randn(2, 31, 8, requires_grad=True)
        model(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.trapezoid_lambda_weight.grad)
        self.assertGreater(model.trapezoid_lambda_weight.grad.abs().sum(), 0.0)
        self.assertIsNotNone(model.last_trapezoid_lambda)


class CyclicPoleTests(unittest.TestCase):
    def test_poles_initialize_at_requested_input_frequencies(self):
        frequencies = [0.0, 1.0 / 64.0, -1.0 / 64.0, 1.0 / 20.0]
        model = ComplexStateSelectiveSSM(
            6,
            d_state=4,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            cyclic_theta_enable=True,
            cyclic_frequencies=frequencies,
            cyclic_max_frequency_delta=0.01,
            token_stride=2,
        )
        actual = model.effective_cyclic_frequencies().mean(0)
        torch.testing.assert_close(
            actual,
            torch.tensor(frequencies),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_learned_pole_residual_is_bounded(self):
        model = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            cyclic_theta_enable=True,
            cyclic_frequencies=[0.0, 0.05],
            cyclic_max_frequency_delta=0.01,
        )
        initial = model.effective_cyclic_frequencies()
        with torch.no_grad():
            model.theta.fill_(100.0)
        shifted = model.effective_cyclic_frequencies()
        self.assertLessEqual((shifted - initial).abs().max(), 0.010001)

    def test_derived_cyclic_anchors_do_not_change_checkpoint_schema(self):
        baseline = ComplexStateSelectiveSSM(
            4, d_state=2, d_conv=2, expand=1, scan_backend="torch"
        )
        cyclic = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            cyclic_theta_enable=True,
            cyclic_frequencies=[0.0, 0.05],
        )
        self.assertEqual(baseline.state_dict().keys(), cyclic.state_dict().keys())


class ReliabilityConditioningTests(unittest.TestCase):
    def test_reliability_initialization_is_near_stage295(self):
        model = ComplexStateSelectiveSSM(
            8,
            d_state=4,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            reliability_enable=True,
            reliability_floor=0.05,
            reliability_init=0.995,
        )
        model(torch.randn(2, 17, 8))
        expected = torch.full_like(model.last_reliability, 0.995)
        torch.testing.assert_close(model.last_reliability, expected, atol=1e-6, rtol=0)

    def test_reliability_network_trains_through_state_update(self):
        torch.manual_seed(334)
        model = ComplexStateSelectiveSSM(
            8,
            d_state=4,
            d_conv=2,
            expand=1,
            scan_backend="torch",
            reliability_enable=True,
        )
        output = model(torch.randn(2, 29, 8))
        output.square().mean().backward()
        final_layer = model.reliability_net[-1]
        self.assertIsNotNone(final_layer.weight.grad)
        self.assertGreater(final_layer.weight.grad.abs().sum(), 0.0)

    def test_forced_cuda_rejects_cpu_tensor(self):
        model = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            d_conv=2,
            expand=1,
            scan_backend="cuda",
            reliability_enable=True,
        )
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA tensor"):
            model(torch.randn(1, 8, 4))

    def test_reliability_euler_can_use_official_fused_scan(self):
        model = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            expand=1,
            reliability_enable=True,
            scan_backend="auto",
        )
        fake_cuda_tensor = mock.Mock(is_cuda=True)
        with mock.patch(
            "models.IQUMamba1D_ComplexStateMamba._mamba_selective_scan_fn",
            object(),
        ):
            self.assertEqual(
                model._select_scan_backend(fake_cuda_tensor), "mamba_cuda"
            )

    def test_trapezoidal_cuda_selects_native_scan(self):
        model = ComplexStateSelectiveSSM(
            4,
            d_state=2,
            expand=1,
            discretization="exponential_trapezoidal",
            scan_backend="auto",
        )
        self.assertEqual(
            model._select_scan_backend(mock.Mock(is_cuda=True)), "native_cuda"
        )


@unittest.skipUnless(
    torch.cuda.is_available()
    and os.environ.get("RUN_IQUMAMBA_CUDA_EXTENSION_TESTS") == "1",
    "set RUN_IQUMAMBA_CUDA_EXTENSION_TESTS=1 on a CUDA build host",
)
class NativeCudaScanTests(unittest.TestCase):
    def test_forward_and_backward_match_reference_scan(self):
        torch.manual_seed(335)
        shape = (2, 37, 3, 4)
        magnitude = 0.95 * torch.rand(shape, device="cuda")
        angle = torch.randn(shape, device="cuda")
        values = [
            (magnitude * torch.cos(angle)).requires_grad_(True),
            (magnitude * torch.sin(angle)).requires_grad_(True),
            torch.randn(shape, device="cuda", requires_grad=True),
            torch.randn(shape, device="cuda", requires_grad=True),
        ]

        actual = native_complex_scan(*values)
        actual_loss = actual[0].square().mean() + actual[1].square().mean()
        actual_grads = torch.autograd.grad(actual_loss, values)

        references = [value.detach().clone().requires_grad_(True) for value in values]
        expected = complex_prefix_scan(*references)
        expected_loss = expected[0].square().mean() + expected[1].square().mean()
        expected_grads = torch.autograd.grad(expected_loss, references)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3e-4, atol=3e-5
            )


class StageRegistrationTests(unittest.TestCase):
    def test_stage_ids_and_configs_are_registered(self):
        for stage in range(330, 334):
            self.assertIn(stage, supported_stage_ids())
            paths = [
                path
                for path in (PROJECT_ROOT / "config").glob(
                    f"model_config_stage{stage}_*.yaml"
                )
                if not path.stem.endswith("_rfchallenge")
            ]
            self.assertEqual(len(paths), 1)
            config = MambaConfig(str(paths[0]), train=True)
            config._load_enc_config()
            self.assertEqual(config.model_type, "iqumamba_rf_mamba3")

    def test_ablation_flags_are_isolated(self):
        expected = {
            330: ("exponential_trapezoidal", False, False),
            331: ("exponential_euler", True, False),
            332: ("exponential_euler", False, True),
            333: ("exponential_trapezoidal", True, True),
        }
        for stage, flags in expected.items():
            path = next(
                path
                for path in (PROJECT_ROOT / "config").glob(
                    f"model_config_stage{stage}_*.yaml"
                )
                if not path.stem.endswith("_rfchallenge")
            )
            cfg = MambaConfig(str(path), train=True).model_config
            actual = (
                cfg["mamba_discretization"],
                cfg["cyclic_theta_enable"],
                cfg["reliability_enable"],
            )
            self.assertEqual(actual, flags)


class Stage347RealStateAblationTests(unittest.TestCase):
    def test_registered_config_is_stage333_scale_real_state_ablation(self):
        path = PROJECT_ROOT / "config" / "model_config_stage347_real_state_trap_reliability.yaml"
        config = MambaConfig(str(path), train=True)
        config._load_enc_config()
        cfg = config.model_config
        self.assertEqual(config.model_type, "iqumamba_real_state_trap_reliability")
        self.assertEqual(cfg["mamba_d_state"], 8)
        self.assertEqual(cfg["mamba_discretization"], "exponential_trapezoidal")
        self.assertTrue(cfg["reliability_enable"])
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(
            '347: CONFIG_ROOT / "model_config_stage347_real_state_trap_reliability.yaml"',
            main_source,
        )

    def test_all_replaced_states_are_real_and_frozen(self):
        from util.utils import Create_Mamba_model

        config = MambaConfig(
            str(PROJECT_ROOT / "config" / "model_config_stage347_real_state_trap_reliability.yaml")
        )
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(model, IQUMamba1DRealStateTrapReliability)
        layers = [
            layer for layer in model.encoder.mamba_layers
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertEqual(layer.ssm.discretization, "exponential_trapezoidal")
            self.assertTrue(layer.ssm.reliability_enable)
            self.assertFalse(layer.ssm.theta.requires_grad)
            torch.testing.assert_close(
                layer.ssm.theta, torch.zeros_like(layer.ssm.theta)
            )
        parameters = dict(model.named_parameters())
        for name in model.no_weight_decay():
            self.assertIn(name, parameters)


if __name__ == "__main__":
    unittest.main()
