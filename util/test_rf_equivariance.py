import unittest
from pathlib import Path

import torch

from util.rf_equivariance import (
    apply_fixed_slot_rf_transform,
    build_fixed_slot_rf_view,
    fixed_slot_rf_equivariance_consistency_loss,
    sample_fixed_slot_rf_parameters,
)


class FixedSlotRFEquivarianceTests(unittest.TestCase):
    def _parameters(self, batch=3, sources=2, mode="per_source"):
        torch.manual_seed(7)
        return sample_fixed_slot_rf_parameters(
            batch,
            sources,
            max_phase_degrees=180.0,
            max_cfo_cycles_per_sample=2.0e-3,
            max_gain_db=3.0,
            max_shift_samples=7,
            conjugate_probability=0.5,
            source_mode=mode,
            device=torch.device("cpu"),
        )

    def test_forward_inverse_round_trip(self):
        x = torch.randn(3, 4, 97)
        parameters = self._parameters()
        transformed = apply_fixed_slot_rf_transform(x, parameters, num_sources=2)
        restored = apply_fixed_slot_rf_transform(
            transformed, parameters, num_sources=2, inverse=True
        )
        self.assertTrue(torch.allclose(restored, x, atol=2e-5, rtol=2e-5))

    def test_per_source_view_preserves_receiver_residual(self):
        targets = torch.randn(3, 4, 64)
        clean = targets.reshape(3, 2, 2, 64).sum(dim=1)
        residual = 0.1 * torch.randn_like(clean)
        inputs = clean + residual
        transformed_inputs, transformed_targets = build_fixed_slot_rf_view(
            inputs,
            targets,
            self._parameters(),
            num_sources=2,
            source_mode="per_source",
        )
        transformed_clean = transformed_targets.reshape(3, 2, 2, 64).sum(dim=1)
        self.assertTrue(
            torch.allclose(transformed_inputs - transformed_clean, residual, atol=1e-6)
        )

    def test_exact_equivariance_has_zero_consistency_loss(self):
        original = torch.randn(3, 4, 64)
        parameters = self._parameters()
        transformed = apply_fixed_slot_rf_transform(
            original, parameters, num_sources=2
        )
        loss = fixed_slot_rf_equivariance_consistency_loss(
            original, transformed, parameters, num_sources=2
        )
        self.assertLess(float(loss), 1e-10)

    def test_slot_swap_is_not_silently_pit_aligned(self):
        original = torch.randn(3, 4, 64)
        parameters = self._parameters()
        transformed = apply_fixed_slot_rf_transform(
            original, parameters, num_sources=2
        ).reshape(3, 2, 2, 64)
        swapped = transformed[:, [1, 0]].reshape(3, 4, 64)
        loss = fixed_slot_rf_equivariance_consistency_loss(
            original, swapped, parameters, num_sources=2
        )
        self.assertGreater(float(loss), 1e-3)

    def test_global_sampling_reuses_one_transform_for_all_slots(self):
        parameters = self._parameters(mode="global")
        for value in parameters.values():
            self.assertTrue(torch.equal(value[:, :1], value[:, 1:2]))

    def test_training_and_cli_are_wired_without_pit(self):
        project_root = Path(__file__).resolve().parents[1]
        training = (project_root / "util" / "training.py").read_text(encoding="utf-8")
        main = (project_root / "main.py").read_text(encoding="utf-8")
        self.assertIn("_compute_rf_equiv_extra_loss", training)
        self.assertIn("fixed_slot_rf_equivariance_consistency_loss", training)
        self.assertIn("--rf_equiv_enable", main)
        helper = (project_root / "util" / "rf_equivariance.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_pit_reorder", helper)
        self.assertNotIn("itertools.permutations", helper)


if __name__ == "__main__":
    unittest.main()
