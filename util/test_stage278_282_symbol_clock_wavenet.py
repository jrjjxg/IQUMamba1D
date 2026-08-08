import math
import unittest

import torch
import torch.nn as nn

from models import icassp_wavenet_mamba as legacy_wavenet_mamba


class _FakeMamba(nn.Module):
    """CPU-only shape-preserving stand-in for unit tests."""

    def __init__(self, d_model, **_kwargs):
        super().__init__()
        self.projection = nn.Linear(int(d_model), int(d_model))

    def forward(self, x):
        return self.projection(x)


legacy_wavenet_mamba.Mamba = _FakeMamba

from models.icassp_symbol_clock_wavenet import (  # noqa: E402
    ICASPAntiAliasedInterleavedMamba,
    ICASPSymbolClockWaveNet,
    ICASPTemporalPhysicalControllerWaveNet,
    TemporalPhysicalMambaController,
    WidelyLinearComplexStem,
)
from util.loss import _phase_increment_pair_per_item  # noqa: E402
from util.stage_registry import supported_stage_ids  # noqa: E402


class StageRegistrationTests(unittest.TestCase):
    def test_new_stage_ids_are_supported(self):
        stages = supported_stage_ids()
        for stage in range(278, 283):
            self.assertIn(stage, stages)


class ComplexStemTests(unittest.TestCase):
    def test_zero_conjugate_branch_is_rotation_equivariant(self):
        torch.manual_seed(1)
        stem = WidelyLinearComplexStem(8)
        with torch.no_grad():
            stem.bias_real.zero_()
            stem.bias_imag.zero_()
        x = torch.randn(2, 2, 97)
        angle = 0.73
        c, s = math.cos(angle), math.sin(angle)
        rotated = torch.stack(
            (c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]),
            dim=1,
        )
        y = stem(x)
        y_rotated = stem(rotated)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        expected = torch.cat(
            (c * y_real - s * y_imag, s * y_real + c * y_imag), dim=1
        )
        torch.testing.assert_close(y_rotated, expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(float(stem.conjugate_ratio().detach()), 0.0)


class ContextAndControllerTests(unittest.TestCase):
    def test_antialiased_stage_has_finite_diagnostics(self):
        torch.manual_seed(2)
        model = ICASPAntiAliasedInterleavedMamba(
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
            mamba_insert_after_block=2,
            mamba_channels=4,
            mamba_downsample_factor=4,
            antialias_taps_per_phase=2,
        )
        output = model(torch.randn(2, 2, 129))
        self.assertEqual(tuple(output.shape), (2, 4, 129))
        for value in model.diagnostics().values():
            self.assertTrue(torch.isfinite(value).all())

    def test_temporal_controller_preserves_order_and_probability_simplex(self):
        torch.manual_seed(3)
        controller = TemporalPhysicalMambaController(
            feature_channels=8,
            controlled_blocks=3,
            token_channels=6,
            chunk_size=32,
            chunk_hop=16,
            physical_lags=(1, 2, 4, 8),
            candidate_periods=(2, 4, 8),
            d_state=4,
            d_conv=2,
            expand=1,
        )
        controls, probabilities = controller(
            torch.randn(2, 8, 95),
            torch.randn(2, 2, 95),
            output_length=95,
        )
        self.assertEqual(tuple(controls.shape), (2, 3, 4, 95))
        self.assertEqual(tuple(probabilities.shape), (2, 3, 95))
        torch.testing.assert_close(
            probabilities.sum(dim=1),
            torch.ones(2, 95),
            rtol=1e-5,
            atol=1e-5,
        )


class EndToEndModelTests(unittest.TestCase):
    def test_stage279_forward(self):
        torch.manual_seed(4)
        model = ICASPTemporalPhysicalControllerWaveNet(
            residual_channels=8,
            residual_layers=6,
            dilation_cycle_length=3,
            controller_insert_after_block=3,
            token_channels=6,
            chunk_size=32,
            chunk_hop=16,
            physical_lags=(1, 2, 4, 8),
            candidate_periods=(2, 4, 8),
            mamba_d_state=4,
            mamba_d_conv=2,
            mamba_expand=1,
        )
        output = model(torch.randn(2, 2, 96))
        self.assertEqual(tuple(output.shape), (2, 4, 96))
        self.assertTrue(torch.isfinite(output).all())

    def test_symbol_clock_ablation_flags(self):
        torch.manual_seed(5)
        x = torch.randn(2, 2, 96)
        for complex_stem, temporal_controls in (
            (False, False),
            (True, False),
            (True, True),
        ):
            with self.subTest(
                complex_stem=complex_stem,
                temporal_controls=temporal_controls,
            ):
                model = ICASPSymbolClockWaveNet(
                    residual_channels=8,
                    pre_residual_layers=3,
                    pre_dilation_cycle_length=3,
                    adaptive_layers=2,
                    candidate_periods=(2, 4, 8),
                    dilation_multipliers=(1, 2),
                    max_dilation=16,
                    use_widely_linear_stem=complex_stem,
                    use_temporal_controls=temporal_controls,
                    token_channels=6,
                    chunk_size=32,
                    chunk_hop=16,
                    physical_lags=(1, 2, 4, 8),
                    mamba_d_state=4,
                    mamba_d_conv=2,
                    mamba_expand=1,
                )
                output = model(x)
                self.assertEqual(tuple(output.shape), (2, 4, 96))
                self.assertTrue(torch.isfinite(output).all())
                diagnostics = model.diagnostics()
                self.assertIn("symbol_router_entropy", diagnostics)
                if complex_stem:
                    self.assertIn("widely_linear_conjugate_ratio", diagnostics)


class PhaseIncrementLossTests(unittest.TestCase):
    def test_phase_increment_penalizes_conjugated_trajectory(self):
        n = torch.arange(128, dtype=torch.float32)
        phase = 0.17 * n
        target = torch.stack((phase.cos(), phase.sin()), dim=0).unsqueeze(0)
        conjugated = target.clone()
        conjugated[:, 1].neg_()
        exact_loss = _phase_increment_pair_per_item(target, target)
        wrong_loss = _phase_increment_pair_per_item(conjugated, target)
        self.assertLess(float(exact_loss), 1e-7)
        self.assertGreater(float(wrong_loss), 0.01)


if __name__ == "__main__":
    unittest.main()
