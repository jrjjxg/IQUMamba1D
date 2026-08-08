"""Focused tests for Stage 334 blind phase-folded Mamba."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from models.IQUMamba1D_PhaseFoldedMamba import PhaseFoldedMambaResidual
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config_stage334_phase_folded_mamba.yaml"


class PhaseGeometryTests(unittest.TestCase):
    def test_fold_unfold_is_exact_for_nondivisible_length(self):
        x = torch.randn(2, 5, 103)
        folded, length = PhaseFoldedMambaResidual.fold(x, 12)
        self.assertEqual(tuple(folded.shape), (2, 12, 9, 5))
        torch.testing.assert_close(
            PhaseFoldedMambaResidual.unfold(folded, length), x
        )

    def test_nominal_fractional_warp_is_identity(self):
        model = PhaseFoldedMambaResidual(
            hidden_channels=4,
            candidate_periods=(8,),
            d_state=2,
            d_conv=2,
            scan_checkpoint=False,
        )
        x = torch.randn(2, 2, 61)
        frequency = torch.full((2,), 1.0 / 8.0)
        warped = model._canonical_warp(x, frequency, 8)
        restored = model._inverse_warp(warped, frequency, 8)
        torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


class BlindRoutingTests(unittest.TestCase):
    def test_envelope_evidence_identifies_period_eight(self):
        length = 512
        time = torch.arange(length, dtype=torch.float32)
        amplitude = 1.5 + 0.8 * torch.cos(2.0 * math.pi * time / 8.0)
        x = torch.stack([amplitude, torch.zeros_like(amplitude)]).unsqueeze(0)
        model = PhaseFoldedMambaResidual(
            hidden_channels=4,
            candidate_periods=(8, 16, 32),
            null_logit_init=-10.0,
            d_state=2,
            d_conv=2,
            scan_checkpoint=False,
        )
        frequencies, routes = model._period_evidence(x)
        self.assertAlmostEqual(float(frequencies[0, 0]), 1.0 / 8.0, places=3)
        self.assertGreater(float(routes[0, 1]), float(routes[0, 2]))
        self.assertGreater(float(routes[0, 1]), float(routes[0, 3]))
        torch.testing.assert_close(routes.sum(-1), torch.ones(1))

    def test_sparse_router_evaluates_configured_candidate_count(self):
        model = PhaseFoldedMambaResidual(
            hidden_channels=4,
            candidate_periods=(8, 12, 16, 24),
            candidate_top_k=2,
            d_state=2,
            d_conv=2,
            scan_checkpoint=False,
        )
        output = model(torch.randn(2, 2, 65))
        self.assertEqual(tuple(output.shape), (2, 4, 65))
        self.assertEqual(len(model.last_selected_candidates), 2)


class ConservativeInitializationTests(unittest.TestCase):
    def test_zero_initialized_branch_is_exactly_inert(self):
        model = PhaseFoldedMambaResidual(
            hidden_channels=4,
            candidate_periods=(8, 16),
            candidate_top_k=1,
            d_state=2,
            d_conv=2,
            scan_checkpoint=False,
            zero_init=True,
        )
        output = model(torch.randn(2, 2, 63))
        torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
        self.assertEqual(model.phase_ssm.last_scan_backend, "torch")

    def test_output_projection_receives_gradient_on_first_step(self):
        model = PhaseFoldedMambaResidual(
            hidden_channels=4,
            candidate_periods=(8,),
            d_state=2,
            d_conv=2,
            scan_checkpoint=False,
            zero_init=True,
        )
        model(torch.randn(2, 2, 64)).sum().backward()
        self.assertGreater(model.output_proj.weight.grad.abs().sum(), 0.0)


class Stage334RegistrationTests(unittest.TestCase):
    def test_stage_and_config_are_registered(self):
        self.assertIn(334, supported_stage_ids())
        config = MambaConfig(str(CONFIG_PATH), train=True)
        config._load_enc_config()
        self.assertEqual(config.model_type, "iqumamba_phase_folded_mamba")
        self.assertEqual(config.model_config["phase_fold_candidate_top_k"], 3)
        self.assertEqual(config.model_config["phase_fold_scan_backend"], "auto")


if __name__ == "__main__":
    unittest.main()
