"""Regression tests for Stage377's strict no-ASC Stage371 combination."""

import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
    mamba_stub = types.ModuleType("mamba_ssm")
    mamba_stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)

    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(int(d_model), int(d_model))

        def forward(self, x):
            return self.projection(x)

    mamba_stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = mamba_stub


from models.IQUResUNet1D_ComplexStateUniRepLK_LatentMask import (
    IQUResUNet1D_ComplexStateUniRepLK_LatentMask,
)
from util.config import MambaConfig
from util.loss import si_snr_huber_loss
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model(
    latent_mask_mode: str = "real",
) -> IQUResUNet1D_ComplexStateUniRepLK_LatentMask:
    return IQUResUNet1D_ComplexStateUniRepLK_LatentMask(
        input_size=64,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        conv_op=torch.nn.Conv1d,
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        deep_supervision=False,
        bimamba_apply_stages=(1, 3),
        bimamba_residual_scale_init=1.0,
        complex_state_d_state=2,
        complex_state_d_conv=3,
        complex_state_expand=1,
        complex_state_scan_checkpoint=False,
        complex_state_scan_backend="torch",
        complex_state_fusion_hidden=8,
        rf_apply_stages=(0, 1, 2),
        rf_residual_scale_init=0.05,
        rf_large_kernel=5,
        rf_ffn_factor=2,
        rf_layer_scale=1.0e-6,
        latent_mask_mode=latent_mask_mode,
    )


class Stage377Tests(unittest.TestCase):
    def test_forward_modules_and_simplex(self):
        model = _small_model().eval()
        x = torch.randn(2, 2, 64)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(hasattr(model.decoder, "skip_processors"))
        self.assertEqual(
            model.encoder.mamba_layers[1].__class__.__name__,
            "IndependentComplexStateBiMambaLayer",
        )
        self.assertEqual(
            model.encoder.mamba_layers[3].__class__.__name__,
            "IndependentComplexStateBiMambaLayer",
        )
        self.assertEqual(set(model.stage_rf.keys()), {"0", "1", "2"})

        with torch.no_grad():
            skips = model._encode_skips(x)
            masks = [
                model._make_masks(features, head)
                for features, head in zip(skips, model.latent_mask_heads)
            ]
        for mask in masks:
            self.assertEqual(mask.shape[1], 2)
            torch.testing.assert_close(
                mask.sum(dim=1),
                torch.ones_like(mask.sum(dim=1)),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_shared_decoder_and_fixed_slot_gradient(self):
        model = _small_model().train()
        x = torch.randn(2, 2, 64)
        skips = model._encode_skips(x)
        masks = [
            model._make_masks(features, head)
            for features, head in zip(skips, model.latent_mask_heads)
        ]
        batched = model.decoder(model._flatten_source_slots(skips, masks))
        per_slot = torch.stack(
            [
                model.decoder(
                    [features * scale_masks[:, slot] for features, scale_masks in zip(skips, masks)]
                )
                for slot in range(model.latent_mask_num_sources)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            batched.reshape_as(per_slot), per_slot, atol=1e-6, rtol=1e-6
        )

        model.zero_grad(set_to_none=True)
        output = model(x)
        target = torch.randn_like(output)
        loss = si_snr_huber_loss(output, target, beta=0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for head in model.latent_mask_heads:
            self.assertIsNotNone(head.weight.grad)
            self.assertGreater(float(head.weight.grad.norm()), 0.0)

    def test_three_source_and_registration(self):
        model = IQUResUNet1D_ComplexStateUniRepLK_LatentMask(
            input_size=64,
            input_channels=2,
            n_stages=4,
            features_per_stage=[4, 8, 16, 32],
            conv_op=torch.nn.Conv1d,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=6,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
            deep_supervision=False,
            bimamba_apply_stages=(1, 3),
            complex_state_d_state=2,
            complex_state_d_conv=3,
            complex_state_expand=1,
            complex_state_scan_checkpoint=False,
            complex_state_scan_backend="torch",
            complex_state_fusion_hidden=8,
            rf_apply_stages=(0, 1, 2),
            rf_large_kernel=5,
            rf_ffn_factor=2,
            latent_mask_mode="real",
        ).eval()
        self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 6, 64))
        self.assertIn(377, supported_stage_ids())

        config = MambaConfig(
            str(
                PROJECT_ROOT
                / "config"
                / "model_config_stage377_stage56_complexstate_unireplk_latent_mask_real.yaml"
            )
        )
        config._load_enc_config()
        self.assertEqual(
            config.model_type,
            "resunet1d_complexstate_unireplk_latent_mask_real",
        )
        factory_model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(
            factory_model, IQUResUNet1D_ComplexStateUniRepLK_LatentMask
        )
        self.assertFalse(hasattr(factory_model.decoder, "skip_processors"))


if __name__ == "__main__":
    unittest.main()
