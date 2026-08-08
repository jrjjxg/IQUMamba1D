"""Unit tests for stage 296 (FSQ tokenizer + token cross-entropy prior).

Run from the repository root:

    python -m unittest util.test_stage296_fsq_token_prior -v

CPU-only, small sizes, no dataset access required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.FSQTokenizer1D import (  # noqa: E402
    FSQ,
    FSQTokenizer1D,
    load_fsq_tokenizer,
    save_fsq_tokenizer,
)
from util.fsq_token_prior import (  # noqa: E402
    fsq_token_accuracy,
    fsq_token_ce_loss,
    load_frozen_tokenizer,
)
from util.stage_registry import supported_stage_ids  # noqa: E402


class FSQQuantizerTests(unittest.TestCase):
    def test_indices_cover_all_levels_and_match_grid(self):
        for levels in ([8, 5, 5, 5], [7, 7], [4], [3, 2]):
            fsq = FSQ(levels)
            z = torch.linspace(-6.0, 6.0, steps=501).view(1, 1, -1).repeat(1, len(levels), 1)
            q, indices = fsq(z)
            for dim, level in enumerate(levels):
                idx = indices[0, dim]
                self.assertEqual(int(idx.min()), 0)
                self.assertEqual(int(idx.max()), level - 1)
                # quantized values must sit exactly on the per-dim grid
                positions = fsq.positions(dim)
                gathered = positions[idx]
                self.assertTrue(torch.allclose(q[0, dim], gathered, atol=1e-5))

    def test_straight_through_gradient_is_nonzero(self):
        fsq = FSQ([8, 5, 5, 5])
        z = torch.randn(2, 4, 16, requires_grad=True)
        q, _ = fsq(z)
        q.sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum()), 0.0)

    def test_codebook_size(self):
        self.assertEqual(FSQ([8, 5, 5, 5]).codebook_size, 1000)


class FSQTokenizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tokenizer = FSQTokenizer1D(levels=[8, 5, 5, 5], base_channels=8)

    def test_forward_shapes_and_backward(self):
        x = torch.randn(3, 2, 256)
        recon, aux = self.tokenizer(x)
        self.assertEqual(tuple(recon.shape), (3, 2, 256))
        self.assertEqual(tuple(aux['indices'].shape), (3, 4, 256 // 8))
        self.assertTrue(torch.isfinite(recon).all())
        (recon - x).pow(2).mean().backward()
        grads = [p.grad for p in self.tokenizer.parameters() if p.grad is not None]
        self.assertGreater(len(grads), 0)

    def test_encode_paths_are_consistent(self):
        x = torch.randn(2, 2, 128)
        bounded = self.tokenizer.encode_bounded(x)
        indices = self.tokenizer.encode_indices(x)
        self.assertEqual(tuple(bounded.shape), (2, 4, 16))
        self.assertEqual(tuple(indices.shape), (2, 4, 16))
        q, idx2 = self.tokenizer.fsq.quantize(bounded)
        self.assertTrue(torch.equal(indices, idx2))

    def test_tokens_are_power_invariant(self):
        x = torch.randn(2, 2, 128)
        idx_a = self.tokenizer.encode_indices(x)
        idx_b = self.tokenizer.encode_indices(4.0 * x)
        self.assertTrue(torch.equal(idx_a, idx_b))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode_indices(torch.randn(2, 3, 128))
        with self.assertRaises(ValueError):
            self.tokenizer.encode_indices(torch.randn(2, 2, 130))

    def test_save_and_load_roundtrip(self):
        x = torch.randn(1, 2, 128)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tok.pth')
            save_fsq_tokenizer(path, self.tokenizer, extra_meta={'note': 'test'})
            loaded = load_fsq_tokenizer(path)
            self.assertEqual(loaded.levels, self.tokenizer.levels)
            self.assertTrue(
                torch.equal(
                    loaded.encode_indices(x), self.tokenizer.encode_indices(x)
                )
            )
            # frozen-loader path used by the training wrapper
            frozen = load_frozen_tokenizer(path)
            self.assertFalse(any(p.requires_grad for p in frozen.parameters()))


class FSQTokenCELossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.tokenizer = FSQTokenizer1D(levels=[8, 5, 5, 5], base_channels=8)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad_(False)

    def test_scalar_finite_and_grad_flows_to_prediction(self):
        targets = torch.randn(2, 4, 128)  # B=2, K=2 sources, layout [I0,Q0,I1,Q1]
        prediction = torch.randn(2, 4, 128, requires_grad=True)
        loss = fsq_token_ce_loss(self.tokenizer, prediction, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)
        # frozen tokenizer must stay gradient-free
        self.assertTrue(
            all(p.grad is None for p in self.tokenizer.parameters())
        )

    def test_perfect_prediction_beats_random(self):
        targets = torch.randn(2, 4, 128)
        loss_perfect = fsq_token_ce_loss(self.tokenizer, targets.clone(), targets)
        loss_random = fsq_token_ce_loss(self.tokenizer, torch.randn(2, 4, 128), targets)
        self.assertLess(float(loss_perfect), float(loss_random))
        acc = fsq_token_accuracy(self.tokenizer, targets.clone(), targets)
        self.assertAlmostEqual(float(acc), 1.0, places=5)

    def test_eval_style_no_grad_call_is_safe(self):
        targets = torch.randn(1, 4, 128)
        with torch.no_grad():
            loss = fsq_token_ce_loss(self.tokenizer, targets.clone(), targets)
        self.assertFalse(loss.requires_grad)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fsq_token_ce_loss(
                self.tokenizer, torch.randn(1, 4, 128), torch.randn(1, 4, 64)
            )


class Stage296RegistrationTests(unittest.TestCase):
    def test_stage_296_is_registered(self):
        self.assertIn(296, supported_stage_ids())

    def test_stage_296_config_exists(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(
            root, 'config', 'model_config_stage296_fsq_token_ce.yaml'
        )
        self.assertTrue(os.path.isfile(config_path), config_path)


if __name__ == '__main__':
    unittest.main()
