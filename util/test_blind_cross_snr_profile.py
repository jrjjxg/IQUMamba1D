import unittest
from types import SimpleNamespace

from util.blind_cross_snr import apply_blind_cross_snr_profile


class BlindCrossSNRProfileTests(unittest.TestCase):
    def test_disabled_profile_is_a_noop(self):
        args = SimpleNamespace(blind_cross_snr_distill=False, cross_snr_enable=None)
        self.assertIs(apply_blind_cross_snr_profile(args), args)
        self.assertIsNone(args.cross_snr_enable)

    def test_checkpoint_free_profile_defaults_to_ema(self):
        args = SimpleNamespace(
            blind_cross_snr_distill=True,
            cross_snr_teacher_checkpoint=None,
            cross_snr_teacher_mode=None,
        )
        apply_blind_cross_snr_profile(args)
        self.assertTrue(args.cross_snr_enable)
        self.assertEqual(args.cross_snr_teacher_mode, "ema")
        self.assertTrue(args.cross_snr_ema_teacher_enable)
        self.assertEqual(args.cross_snr_teacher_view, "clean")
        self.assertEqual(args.cross_snr_pair_mode, "curriculum_student")
        self.assertTrue(args.cross_snr_shared_permutation)

    def test_frozen_profile_uses_same_backbone_checkpoint_without_sync_labels(self):
        args = SimpleNamespace(
            blind_cross_snr_distill=True,
            cross_snr_teacher_checkpoint="stage365_best_model_weights.pth",
            cross_snr_teacher_mode="frozen",
            sync_snr_aux_weight=0.05,
            sync_cross_snr_consistency_weight=0.05,
            sync_physical_supervision_weight=0.10,
            sync_physical_require_metadata=True,
            sync_metadata_enable=True,
        )
        apply_blind_cross_snr_profile(args)
        self.assertFalse(args.cross_snr_ema_teacher_enable)
        self.assertFalse(args.cross_snr_shared_permutation)
        self.assertEqual(args.sync_snr_aux_weight, 0.0)
        self.assertEqual(args.sync_cross_snr_consistency_weight, 0.0)
        self.assertEqual(args.sync_physical_supervision_weight, 0.0)
        self.assertFalse(args.sync_physical_require_metadata)
        self.assertFalse(args.sync_metadata_enable)

    def test_frozen_profile_requires_checkpoint(self):
        args = SimpleNamespace(
            blind_cross_snr_distill=True,
            cross_snr_teacher_checkpoint=None,
            cross_snr_teacher_mode="frozen",
        )
        with self.assertRaisesRegex(ValueError, "same backbone stage"):
            apply_blind_cross_snr_profile(args)


if __name__ == "__main__":
    unittest.main()
