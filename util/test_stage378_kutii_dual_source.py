from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Stage378RegistrationTests(unittest.TestCase):
    def test_stage_and_config_are_registered(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        filename = "model_config_stage378_kutii_dual_source_wavenet.yaml"
        main_text = (ROOT / "main.py").read_text(encoding="utf-8")
        utils_text = (ROOT / "util" / "utils.py").read_text(encoding="utf-8")
        self.assertIn(378, supported_stage_ids())
        self.assertIn(
            f'378: CONFIG_ROOT / "{filename}"',
            main_text,
        )
        self.assertIn('"kutii_dual_source_wavenet"', utils_text)

        config = MambaConfig(str(ROOT / "config" / filename))
        config._load_enc_config()
        self.assertEqual(config.model_type, "kutii_dual_source_wavenet")
        self.assertEqual(config.residual_channels, 256)
        self.assertEqual(config.residual_layers, 30)
        self.assertEqual(config.dilation_cycle_length, 10)


class Stage378NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError:
            raise unittest.SkipTest("torch is not installed")
        from models.kutii_learnable_dilation_wavenet import KUTIIDualSourceWaveNet

        cls.torch = torch
        cls.model_class = KUTIIDualSourceWaveNet

    def test_two_source_forward_backward_and_dilation_contract(self):
        model = self.model_class(
            input_channels=2,
            num_classes=4,
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
            max_dilation=8,
        )
        x = self.torch.randn(2, 2, 32, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 4, 32))
        self.assertTrue(self.torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(self.torch.isfinite(x.grad).all())
        self.assertEqual(tuple(model.effective_dilations().shape), (4,))

    def test_three_source_slots_are_supported(self):
        model = self.model_class(
            input_channels=2,
            num_classes=6,
            residual_channels=4,
            residual_layers=2,
            dilation_cycle_length=2,
            max_dilation=4,
        )
        output = model(self.torch.randn(1, 2, 16))
        self.assertEqual(tuple(output.shape), (1, 6, 16))


if __name__ == "__main__":
    unittest.main()
