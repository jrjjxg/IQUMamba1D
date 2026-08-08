import unittest

import torch

from util.loss import (
    pit_si_snr_huber_identity_anchor_loss,
    pit_si_snr_huber_loss,
)


def _swap_two_sources(x: torch.Tensor) -> torch.Tensor:
    return x[:, [2, 3, 0, 1], :]


class PITIdentityAnchorLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(290)
        source_1 = torch.randn(4, 2, 97)
        source_2 = torch.randn(4, 2, 97)
        self.targets = torch.cat((source_1, source_2), dim=1)
        self.identity_outputs = self.targets + 0.02 * torch.randn_like(self.targets)
        self.swapped_outputs = _swap_two_sources(self.identity_outputs)

    def test_zero_anchor_weight_exactly_matches_plain_pit(self):
        plain = pit_si_snr_huber_loss(
            self.identity_outputs,
            self.targets,
            alpha=1.0,
            beta=0.5,
            delta=1.0,
        )
        anchored = pit_si_snr_huber_identity_anchor_loss(
            self.identity_outputs,
            self.targets,
            alpha=1.0,
            beta=0.5,
            delta=1.0,
            identity_anchor_weight=0.0,
        )
        torch.testing.assert_close(anchored, plain, rtol=0, atol=0)

    def test_plain_pit_is_permutation_invariant(self):
        identity = pit_si_snr_huber_loss(
            self.identity_outputs, self.targets, beta=0.5
        )
        swapped = pit_si_snr_huber_loss(
            self.swapped_outputs, self.targets, beta=0.5
        )
        torch.testing.assert_close(identity, swapped, rtol=1e-6, atol=1e-6)

    def test_anchor_prefers_fixed_identity_without_changing_pit_branch(self):
        identity = pit_si_snr_huber_identity_anchor_loss(
            self.identity_outputs,
            self.targets,
            beta=0.5,
            identity_anchor_weight=0.05,
            identity_anchor_margin=0.2,
            identity_anchor_temperature=0.5,
        )
        swapped = pit_si_snr_huber_identity_anchor_loss(
            self.swapped_outputs,
            self.targets,
            beta=0.5,
            identity_anchor_weight=0.05,
            identity_anchor_margin=0.2,
            identity_anchor_temperature=0.5,
        )
        self.assertLess(float(identity), float(swapped))

    def test_backward_is_finite_for_ambiguous_initial_outputs(self):
        outputs = torch.zeros_like(self.targets, requires_grad=True)
        loss = pit_si_snr_huber_identity_anchor_loss(
            outputs,
            self.targets,
            beta=0.5,
            identity_anchor_weight=0.05,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(outputs.grad)
        self.assertTrue(torch.isfinite(outputs.grad).all())

    def test_invalid_anchor_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            pit_si_snr_huber_identity_anchor_loss(
                self.identity_outputs,
                self.targets,
                identity_anchor_weight=-0.01,
            )
        with self.assertRaises(ValueError):
            pit_si_snr_huber_identity_anchor_loss(
                self.identity_outputs,
                self.targets,
                identity_anchor_temperature=0.0,
            )


if __name__ == "__main__":
    unittest.main()
