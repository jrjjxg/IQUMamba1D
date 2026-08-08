from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    269: (
        "model_config_stage269_wavenet_mamba_film_controller.yaml",
        "icassp_wavenet_mamba_film_controller",
    ),
    270: (
        "model_config_stage270_wavenet_mamba_dilation_skip_router.yaml",
        "icassp_wavenet_mamba_dilation_skip_router",
    ),
    271: (
        "model_config_stage271_wavenet_phase_aware_reverse_mamba.yaml",
        "icassp_wavenet_interleaved_phase_aware_reverse_mamba",
    ),
}


class Stage269271RegistrationTests(unittest.TestCase):
    def test_all_variants_are_selectable_and_configured(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main = (ROOT / "main.py").read_text(encoding="utf-8")
        utils = (ROOT / "util" / "utils.py").read_text(encoding="utf-8")
        for stage, (filename, model_type) in STAGES.items():
            self.assertIn(stage, supported_stage_ids())
            self.assertIn(
                f'{stage}: CONFIG_ROOT / "{filename}"',
                main,
            )
            self.assertIn(f'"{model_type}"', utils)
            config = MambaConfig(str(ROOT / "config" / filename))
            config._load_enc_config()
            self.assertEqual(config.model_type, model_type)
            self.assertEqual(config.residual_layers, 20)
            self.assertEqual(config.mamba_insert_after_block, 10)


class Stage269271NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError:
            raise unittest.SkipTest("torch is not installed")

        import models.icassp_wavenet_mamba as wavenet_mamba

        class FakeMamba(torch.nn.Module):
            def __init__(self, d_model, **_kwargs):
                super().__init__()
                self.proj = torch.nn.Linear(int(d_model), int(d_model))

            def forward(self, x):
                return self.proj(x)

        # The numerical contracts below concern WaveNet integration.  A small
        # deterministic stand-in avoids requiring a compiled Mamba extension.
        wavenet_mamba.Mamba = FakeMamba
        cls.torch = torch
        cls.film_class = wavenet_mamba.ICASPBaselineWaveNetMambaFiLMController
        cls.router_class = wavenet_mamba.ICASPBaselineWaveNetMambaDilationSkipRouter
        cls.phase_class = (
            wavenet_mamba.ICASPBaselineWaveNetInterleavedPhaseAwareReverseMamba
        )
        cls.phase_context = wavenet_mamba.PhaseAwareReverseMambaContext
        from models.icassp_baseline_wavenet import ICASPBaselineWaveNet

        cls.baseline_class = ICASPBaselineWaveNet

    def _common_kwargs(self):
        return {
            "input_channels": 2,
            "num_classes": 4,
            "residual_channels": 8,
            "residual_layers": 4,
            "dilation_cycle_length": 2,
            "mamba_insert_after_block": 2,
            "mamba_channels": 4,
            "mamba_downsample_factor": 4,
            "mamba_d_state": 4,
            "mamba_d_conv": 2,
            "mamba_expand": 1,
            "mamba_controller_hidden": 8,
        }

    def test_phase_reverse_is_complex_conjugate_time_reversal(self):
        mixture = self.torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        )
        expected = self.torch.tensor(
            [[[3.0, 2.0, 1.0], [-6.0, -5.0, -4.0]]]
        )
        self.assertTrue(
            self.torch.equal(self.phase_context.physical_reverse(mixture), expected)
        )

    def test_zero_initialized_controllers_match_plain_wavenet(self):
        common = self._common_kwargs()
        for controlled_class in (self.film_class, self.router_class):
            controlled = controlled_class(**common).eval()
            baseline = self.baseline_class(
                input_channels=2,
                num_classes=4,
                residual_channels=8,
                residual_layers=4,
                dilation_cycle_length=2,
            ).eval()
            baseline.load_state_dict(controlled.state_dict(), strict=False)
            with self.torch.no_grad():
                controlled.output_projection.weight.normal_(std=0.05)
                controlled.output_projection.bias.zero_()
            baseline.load_state_dict(controlled.state_dict(), strict=False)

            x = self.torch.randn(2, 2, 32)
            self.assertTrue(
                self.torch.allclose(controlled(x), baseline(x), atol=1e-6, rtol=1e-5)
            )

    def test_all_variants_support_batch_forward_and_backward(self):
        common = self._common_kwargs()
        phase_common = {
            key: value
            for key, value in common.items()
            if key != "mamba_controller_hidden"
        }
        models = (
            self.film_class(**common),
            self.router_class(**common),
            self.phase_class(**phase_common),
        )
        for model in models:
            with self.torch.no_grad():
                model.output_projection.weight.normal_(std=0.05)
                model.output_projection.bias.zero_()
            x = self.torch.randn(2, 2, 32, requires_grad=True)
            output = model(x)
            self.assertEqual(tuple(output.shape), (2, 4, 32))
            self.assertTrue(self.torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(self.torch.isfinite(x.grad).all())
            if hasattr(model, "no_weight_decay"):
                parameter_names = {name for name, _ in model.named_parameters()}
                self.assertTrue(model.no_weight_decay() <= parameter_names)


if __name__ == "__main__":
    unittest.main()
