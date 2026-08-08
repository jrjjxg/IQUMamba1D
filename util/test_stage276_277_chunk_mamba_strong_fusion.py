from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    276: (
        "model_config_stage276_wavenet5_chunk_mamba_wavenet5.yaml",
        10,
        5,
        5,
    ),
    277: (
        "model_config_stage277_wavenet10_chunk_mamba_wavenet10.yaml",
        20,
        10,
        10,
    ),
}
MODEL_TYPE = "icassp_wavenet_chunk_mamba_strong_fusion"


class ChunkMambaStrongFusionRegistrationTests(unittest.TestCase):
    def test_stages_are_selectable_and_configured(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main = (ROOT / "main.py").read_text(encoding="utf-8")
        utils = (ROOT / "util" / "utils.py").read_text(encoding="utf-8")
        self.assertIn(f'"{MODEL_TYPE}"', utils)

        for stage, (filename, layers, insert_after, cycle) in STAGES.items():
            self.assertIn(stage, supported_stage_ids())
            self.assertIn(f'{stage}: CONFIG_ROOT / "{filename}"', main)

            config = MambaConfig(str(ROOT / "config" / filename))
            config._load_enc_config()
            self.assertEqual(config.model_type, MODEL_TYPE)
            self.assertEqual(config.residual_layers, layers)
            self.assertEqual(config.mamba_insert_after_block, insert_after)
            self.assertEqual(config.dilation_cycle_length, cycle)
            self.assertEqual(config.mamba_chunk_size, 64)
            self.assertEqual(config.mamba_chunk_hop, 32)
            self.assertEqual(config.mamba_fusion_gain_init, 1.0)


class ChunkMambaStrongFusionNumericalTests(unittest.TestCase):
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
        cls.model_class = (
            wavenet_mamba.ICASPBaselineWaveNetChunkMambaStrongFusion
        )

    def _make_model(self):
        return self.model_class(
            input_channels=2,
            num_classes=4,
            residual_channels=8,
            residual_layers=4,
            dilation_cycle_length=2,
            mamba_insert_after_block=2,
            mamba_channels=4,
            mamba_chunk_size=8,
            mamba_chunk_hop=4,
            mamba_d_state=4,
            mamba_d_conv=2,
            mamba_expand=1,
            mamba_fusion_gain_init=1.0,
        )

    def test_gain_is_strong_at_initialization(self):
        model = self._make_model()
        gain = model.chunk_mamba_fusion.fusion_gain_values()
        self.assertTrue(
            self.torch.allclose(gain, self.torch.ones_like(gain), atol=1e-6)
        )

    def test_unaligned_length_supports_forward_and_backward(self):
        model = self._make_model()
        with self.torch.no_grad():
            model.output_projection.weight.normal_(std=0.05)
            model.output_projection.bias.zero_()

        x = self.torch.randn(2, 2, 35, requires_grad=True)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 4, 35))
        self.assertTrue(self.torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(self.torch.isfinite(x.grad).all())

        mamba_weight = (
            model.chunk_mamba_fusion.mamba.forward_mamba.proj.weight
        )
        self.assertIsNotNone(mamba_weight.grad)
        self.assertGreater(float(mamba_weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.chunk_mamba_fusion.fusion_gain_raw.grad)

    def test_shorter_than_one_chunk_preserves_length(self):
        model = self._make_model().eval()
        output = model(self.torch.randn(1, 2, 5))
        self.assertEqual(tuple(output.shape), (1, 4, 5))


if __name__ == "__main__":
    unittest.main()
