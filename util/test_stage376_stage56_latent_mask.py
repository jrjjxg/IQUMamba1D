"""Regression tests for Stage376 (Stage56 + Stage371-style latent masks)."""

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


from models.IQUResUNet1D_LatentMask import IQUResUNet1D_NoASC_LatentMask
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> IQUResUNet1D_NoASC_LatentMask:
    return IQUResUNet1D_NoASC_LatentMask(
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
        latent_mask_mode="real",
    )


class Stage376Tests(unittest.TestCase):
    def test_forward_and_simplex_contract(self):
        model = _small_model().eval()
        x = torch.randn(2, 2, 64)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(hasattr(model.encoder, "mamba_layers"))
        self.assertFalse(hasattr(model.decoder, "skip_processors"))

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

    def test_shared_decoder_matches_per_slot_decoder(self):
        model = _small_model().eval()
        x = torch.randn(2, 2, 64)
        with torch.no_grad():
            skips = model._encode_skips(x)
            masks = [
                model._make_masks(features, head)
                for features, head in zip(skips, model.latent_mask_heads)
            ]
            batched = model.decoder(model._flatten_source_slots(skips, masks))
            per_slot = []
            for slot in range(model.latent_mask_num_sources):
                slot_skips = [
                    features * scale_masks[:, slot]
                    for features, scale_masks in zip(skips, masks)
                ]
                per_slot.append(model.decoder(slot_skips))
            per_slot = torch.stack(per_slot, dim=1).reshape_as(
                batched.reshape(x.shape[0], -1, x.shape[-1])
            )
            batched = batched.reshape_as(per_slot)
        torch.testing.assert_close(batched, per_slot, atol=1e-6, rtol=1e-6)

    def test_mask_route_is_differentiable(self):
        model = _small_model()
        x = torch.randn(1, 2, 64, requires_grad=True)
        output = model(x)
        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        for head in model.latent_mask_heads:
            self.assertIsNotNone(head.weight.grad)
            self.assertTrue(torch.isfinite(head.weight.grad).all())

    def test_registration(self):
        self.assertIn(376, supported_stage_ids())
        config = MambaConfig(
            str(
                PROJECT_ROOT
                / "config"
                / "model_config_stage376_stage56_latent_mask_real.yaml"
            )
        )
        config._load_enc_config()
        self.assertEqual(config.model_type, "resunet1d_noasc_latent_mask_real")
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(model, IQUResUNet1D_NoASC_LatentMask)
        self.assertEqual(model.decoder.seg_layers[-1].out_channels, 2)


if __name__ == "__main__":
    unittest.main()
