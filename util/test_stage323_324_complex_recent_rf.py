from pathlib import Path
import unittest

import torch

# Install the same CPU-only dependency stubs before importing model modules.
import util.test_stage317_322_fdconv_unirep_ablation  # noqa: F401

from models.IQUMamba1D_ComplexRecentRF import (
    ComplexDilatedReparamBlock1D,
    ComplexFrequencyDynamicConv1D,
    ComplexRecentRFIQAdapter,
    ComplexUniRepLKNetBlock1D,
)
from util.stage_registry import supported_stage_ids


class Stage323To324Tests(unittest.TestCase):
    def test_complex_fdconv_experts_are_disjoint_complete_and_reconstruct(self):
        operator = ComplexFrequencyDynamicConv1D(4, kernel_size=15, bands=3)
        coverage = operator.frequency_masks.sum(0)
        torch.testing.assert_close(coverage, torch.ones_like(coverage))
        torch.testing.assert_close(
            operator.expert_kernels().sum(0), operator.complex_weight,
            atol=3e-6, rtol=3e-6,
        )

    def test_complex_fdconv_is_global_phase_equivariant(self):
        torch.manual_seed(323)
        operator = ComplexFrequencyDynamicConv1D(4, kernel_size=15, bands=3).eval()
        z = torch.complex(torch.randn(2, 4, 65), torch.randn(2, 4, 65))
        phase = torch.exp(torch.tensor(0.73j))
        torch.testing.assert_close(
            operator(z * phase), operator(z) * phase, atol=2e-5, rtol=2e-5,
        )

    def test_complex_unireplk_uses_official_branches_and_reparameterizes(self):
        torch.manual_seed(324)
        block = ComplexDilatedReparamBlock1D(4, 17)
        self.assertEqual(block.branch_kernels, (5, 9, 3, 3, 3))
        self.assertEqual(block.dilations, (1, 2, 4, 5, 7))
        # Update running complex means and scalar variances before deployment.
        for _ in range(3):
            block(torch.complex(torch.randn(2, 4, 64), torch.randn(2, 4, 64)))
        block.eval()
        z = torch.complex(torch.randn(2, 4, 64), torch.randn(2, 4, 64))
        expected = block(z)
        block.reparameterize()
        torch.testing.assert_close(block(z), expected, atol=4e-6, rtol=4e-6)

    def test_complex_unireplk_is_phase_equivariant_at_initialization(self):
        torch.manual_seed(324)
        operator = ComplexUniRepLKNetBlock1D(4, 17, ffn_factor=2)
        z = torch.complex(torch.randn(2, 4, 65), torch.randn(2, 4, 65))
        phase = torch.exp(torch.tensor(-0.51j))
        expected = operator(z) * phase
        actual = operator(z * phase)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    def test_both_complex_adapters_preserve_shape_dtype_and_train_imaginary_weights(self):
        operators = {
            "fdconv": ComplexFrequencyDynamicConv1D(4, 15, 3),
            "unireplk": ComplexUniRepLKNetBlock1D(4, 17, ffn_factor=2),
        }
        for name, operator in operators.items():
            with self.subTest(name=name):
                adapter = ComplexRecentRFIQAdapter(4, operator)
                x = torch.randn(2, 2, 129, requires_grad=True)
                output = adapter(x)
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.dtype, x.dtype)
                output.square().mean().backward()
                imaginary_grad = sum(
                    parameter.grad.abs().sum().item()
                    for parameter_name, parameter in adapter.named_parameters()
                    if "imag" in parameter_name and parameter.grad is not None
                )
                self.assertGreater(imaginary_grad, 0.0)

                low_precision = adapter(torch.randn(1, 2, 129, dtype=torch.bfloat16))
                self.assertEqual(low_precision.dtype, torch.bfloat16)

    def test_complex_iq_adapter_is_phase_equivariant(self):
        torch.manual_seed(323)
        adapter = ComplexRecentRFIQAdapter(
            4, ComplexFrequencyDynamicConv1D(4, 15, 3),
        ).eval()
        x = torch.randn(2, 2, 65)
        angle = 0.61
        cosine, sine = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rotated = torch.stack((
            cosine * x[:, 0] - sine * x[:, 1],
            sine * x[:, 0] + cosine * x[:, 1],
        ), dim=1)
        y = adapter(x)
        expected = torch.stack((
            cosine * y[:, 0] - sine * y[:, 1],
            sine * y[:, 0] + cosine * y[:, 1],
        ), dim=1)
        torch.testing.assert_close(adapter(rotated), expected, atol=3e-5, rtol=3e-5)

    def test_stage_configs_register_build_and_run_length4096(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        root = Path(__file__).resolve().parents[1]
        configs = {
            323: "model_config_stage323_complex_fdconv.yaml",
            324: "model_config_stage324_complex_unireplk.yaml",
        }
        for stage, filename in configs.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                path = root / "config" / filename
                self.assertTrue(path.is_file())
                model = Create_Mamba_model(
                    MambaConfig(str(path)), logger=None, input_size_=4096,
                    device_override=torch.device("cpu"),
                ).eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 2, 4096))
                self.assertEqual(output.shape, (1, 4, 4096))
                model.train()
                model.zero_grad(set_to_none=True)
                x = torch.randn(2, 2, 64, requires_grad=True)
                model(x).square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                imaginary_grad = sum(
                    parameter.grad.abs().sum().item()
                    for name, parameter in model.complex_rf.named_parameters()
                    if "imag" in name and parameter.grad is not None
                )
                self.assertGreater(imaginary_grad, 0.0)


if __name__ == "__main__":
    unittest.main()
