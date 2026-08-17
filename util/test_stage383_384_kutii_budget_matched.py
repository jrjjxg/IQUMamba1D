from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class KUTIIBudgetMatchedRegistrationTests(unittest.TestCase):
    CASES = {
        383: (
            "model_config_stage383_kutii_param_match.yaml",
            "param_match",
            108,
            2_824_882,
        ),
        384: (
            "model_config_stage384_kutii_flops_match.yaml",
            "flops_match_stage381_enhanced",
            52,
            None,
        ),
    }

    def test_stage_configs_are_registered(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main_text = (ROOT / "main.py").read_text(encoding="utf-8")
        supported = supported_stage_ids()
        for stage, (filename, variant, channels, _params) in self.CASES.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported)
                self.assertIn(
                    f'{stage}: CONFIG_ROOT / "{filename}"',
                    main_text,
                )
                config = MambaConfig(str(ROOT / "config" / filename))
                config._load_enc_config()
                self.assertEqual(config.model_type, "kutii_dual_source_wavenet")
                self.assertEqual(config.comparison_variant, variant)
                self.assertEqual(config.residual_channels, channels)
                self.assertEqual(config.residual_layers, 30)
                self.assertEqual(config.dilation_cycle_length, 10)
                self.assertEqual(config.max_dilation, 1024)
                if stage == 384:
                    self.assertTrue(config.model_config["enhanced_stage381"])


class KUTIIBudgetMatchedNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError:
            raise unittest.SkipTest("torch is not installed")
        from models.kutii_learnable_dilation_wavenet import KUTIIDualSourceWaveNet

        cls.torch = torch
        cls.model_class = KUTIIDualSourceWaveNet

    def test_parameter_counts_and_source_slot_forward(self):
        cases = ((108, 2_824_882),)
        for channels, expected_parameters in cases:
            with self.subTest(channels=channels):
                model = self.model_class(
                    input_channels=2,
                    num_classes=4,
                    residual_channels=channels,
                    residual_layers=30,
                    dilation_cycle_length=10,
                    max_dilation=1024,
                )
                parameter_count = sum(p.numel() for p in model.parameters())
                self.assertEqual(parameter_count, expected_parameters)
                output = model(self.torch.randn(1, 2, 16))
                self.assertEqual(tuple(output.shape), (1, 4, 16))
                self.assertTrue(self.torch.isfinite(output).all())

                expected_cycle = self.torch.tensor(
                    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] * 3,
                    dtype=model.effective_dilations().dtype,
                )
                self.assertTrue(
                    self.torch.equal(model.effective_dilations(), expected_cycle)
                )

    def test_stage4_flops_budget_width(self):
        # The 52-channel trunk leaves room for the three light adapters and
        # original Stage4 301.86 GFLOPs budget before the three light adapters.
        batch_size = 32
        signal_length = 4096
        channels = 52
        residual_layers = 30
        macs = batch_size * signal_length * (
            (14 * residual_layers + 1) * channels * channels + 6 * channels
        )
        forward_flops = 2 * macs
        stage4_forward_flops = 301.86e9
        relative_error = abs(forward_flops - stage4_forward_flops) / stage4_forward_flops
        self.assertLess(relative_error, 0.06)

    def test_stage384_enhanced_modules_and_forward(self):
        model = self.model_class(
            input_channels=2,
            num_classes=4,
            residual_channels=52,
            residual_layers=30,
            dilation_cycle_length=10,
            max_dilation=1024,
            enhanced_stage381=True,
        )
        self.assertIsNotNone(model.enhanced_bottleneck)
        self.assertTrue(hasattr(model.enhanced_bottleneck, "complex_bimamba"))
        self.assertTrue(hasattr(model.enhanced_bottleneck, "unireplk"))
        self.assertTrue(hasattr(model.enhanced_bottleneck, "mask_head"))
        output = model(self.torch.randn(1, 2, 64))
        self.assertEqual(tuple(output.shape), (1, 4, 64))
        self.assertTrue(self.torch.isfinite(output).all())
        with self.torch.no_grad():
            sample = self.torch.randn(1, 2, 64)
            hidden = self.torch.relu(model.input_projection(sample))
            skips = []
            for block in model.residual_layers:
                hidden, skip = block(hidden)
                skips.append(skip)
            trunk = self.torch.relu(
                model.skip_projection(sum(skips) / len(skips) ** 0.5)
            )
            context = self.torch.nn.functional.avg_pool1d(
                trunk, kernel_size=8, stride=8, ceil_mode=True
            )
            _, masks = model.enhanced_bottleneck(context, trunk)
        simplex = masks.sum(dim=1)
        self.assertTrue(self.torch.allclose(simplex, self.torch.ones_like(simplex)))


if __name__ == "__main__":
    unittest.main()
