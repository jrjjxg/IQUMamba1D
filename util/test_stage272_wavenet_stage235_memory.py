from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "model_config_stage272_wavenet_stage235_memory.yaml"
MODEL_TYPE = "icassp_wavenet_interleaved_stage235_memory"


class Stage272RegistrationTests(unittest.TestCase):
    def test_stage_is_selectable_and_configured(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main = (ROOT / "main.py").read_text(encoding="utf-8")
        utils = (ROOT / "util" / "utils.py").read_text(encoding="utf-8")
        self.assertIn(272, supported_stage_ids())
        self.assertIn(f'272: CONFIG_ROOT / "{CONFIG_NAME}"', main)
        self.assertIn(f'"{MODEL_TYPE}"', utils)

        config = MambaConfig(str(ROOT / "config" / CONFIG_NAME))
        config._load_enc_config()
        self.assertEqual(config.model_type, MODEL_TYPE)
        self.assertEqual(config.residual_layers, 20)
        self.assertEqual(config.mamba_insert_after_block, 10)


class Stage272NumericalTests(unittest.TestCase):
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

        wavenet_mamba.Mamba = FakeMamba
        cls.torch = torch
        cls.model_class = wavenet_mamba.ICASPBaselineWaveNetInterleavedStage235Memory

    def test_memory_replaces_reconstructed_context_and_backpropagates(self):
        model = self.model_class(
            input_channels=2,
            num_classes=4,
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
            mamba_insert_after_block=2,
            mamba_channels=4,
            mamba_downsample_factor=4,
            mamba_d_state=4,
            mamba_d_conv=2,
            mamba_expand=1,
            cross_scale_kv_tokens=4,
            cross_scale_num_heads=2,
        )
        self.assertFalse(hasattr(model, "interleaved_mamba"))
        self.assertTrue(hasattr(model.stage235_memory, "global_memory"))
        self.assertFalse(hasattr(model.stage235_memory.global_memory, "upsample"))

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
        self.assertIsNotNone(
            model.stage235_memory.global_memory.mamba.forward_mamba.proj.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
