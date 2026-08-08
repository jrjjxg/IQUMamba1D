import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


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


from models.IQUBiMamba1D_LatentMask import IQUBiMamba1D_ComplexLatentMask
from models.IQUBiMamba1D_BottleneckMask import IQUBiMamba1D_BottleneckRealMask
from util.config import MambaConfig
from util.loss import si_snr_huber_loss

try:
    from util.evaluation import test_model
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise
    test_model = None
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model(mode):
    return IQUBiMamba1D_ComplexLatentMask(
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
        bimamba_apply_stages=[1, 3],
        bimamba_residual_scale_init=1.0,
        complex_state_d_state=2,
        complex_state_d_conv=4,
        complex_state_expand=1,
        complex_state_scan_checkpoint=False,
        complex_state_scan_backend="torch",
        complex_state_fusion_hidden=8,
        rf_apply_stages=[0, 1, 2],
        rf_residual_scale_init=0.05,
        rf_large_kernel=17,
        rf_ffn_factor=2,
        rf_layer_scale=1e-6,
        latent_mask_mode=mode,
        latent_mask_residual_weight=0.1,
        latent_mask_mixture_weight=0.1,
    )


class Stage371To374ForwardTests(unittest.TestCase):
    def test_all_variants_preserve_fixed_source_contract(self):
        torch.manual_seed(371)
        x = torch.randn(2, 2, 64)
        for mode in (
            "real",
            "complex_ratio",
            "complex_residual",
            "complex_conservation",
        ):
            with self.subTest(mode=mode):
                model = _small_model(mode).eval()
                output = model(x)
                if mode in {"complex_residual", "complex_conservation"}:
                    sources, auxiliary = output
                    self.assertEqual(tuple(auxiliary["residual_output"].shape), (2, 2, 64))
                else:
                    sources = output
                self.assertEqual(tuple(sources.shape), (2, 4, 64))
                self.assertTrue(torch.isfinite(sources).all())

                if mode == "complex_conservation":
                    self.assertLess(
                        float(auxiliary["latent_mask_sum_max_error"].detach()), 1e-5
                    )
                    for mask in auxiliary["latent_masks"]:
                        torch.testing.assert_close(
                            mask.sum(dim=1),
                            torch.ones_like(mask.sum(dim=1)),
                            atol=1e-5,
                            rtol=1e-5,
                        )

    def test_residual_slot_is_differentiable(self):
        model = _small_model("complex_residual")
        x = torch.randn(1, 2, 64, requires_grad=True)
        sources, auxiliary = model(x)
        (sources.square().mean() + auxiliary["residual_output"].square().mean()).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipIf(test_model is None, "evaluation regression requires pandas")
    def test_evaluation_consumes_source_tensor_not_auxiliary_tuple(self):
        class _Logger:
            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

        model = _small_model("complex_conservation").eval()
        inputs = torch.randn(2, 2, 64)
        targets = torch.randn(2, 4, 64)
        snr = torch.zeros(2)
        loader = DataLoader(TensorDataset(inputs, targets, snr), batch_size=1)
        metrics = test_model(
            model,
            {0.0: loader},
            si_snr_huber_loss,
            torch.device("cpu"),
            _Logger(),
            str(PROJECT_ROOT / "tmp_stage371_eval"),
            num_plots=0,
            num_points=64,
            input_size=64,
            data_choice="debug_random",
            signal_names=["S1", "S2"],
            save_artifacts=False,
        )
        self.assertIn(0.0, metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics[0.0]["Loss"])))


class Stage371To374RegistrationTests(unittest.TestCase):
    def test_configs_registry_and_factory(self):
        for stage, filename, model_type, mode in (
            (371, "model_config_stage371_stage365_latent_mask_real.yaml", "bimamba_complex_latent_mask_real", "real"),
            (372, "model_config_stage372_stage365_latent_mask_complex.yaml", "bimamba_complex_latent_mask_ratio", "complex_ratio"),
            (373, "model_config_stage373_stage365_latent_mask_residual.yaml", "bimamba_complex_latent_mask_residual", "complex_residual"),
            (374, "model_config_stage374_stage365_latent_mask_conservation.yaml", "bimamba_complex_latent_mask_conservation", "complex_conservation"),
        ):
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                config._load_enc_config()
                self.assertEqual(config.model_type, model_type)
                self.assertEqual(config.model_config["latent_mask_mode"], mode)
                self.assertIn("latent_mask_residual_weight", config.__dict__)

        model = Create_Mamba_model(
            MambaConfig(
                str(
                    PROJECT_ROOT
                    / "config"
                    / "model_config_stage371_stage365_latent_mask_real.yaml"
                )
            ),
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(model, IQUBiMamba1D_ComplexLatentMask)
        self.assertEqual(model.latent_mask_mode, "real")
        self.assertEqual(model.decoder.seg_layers[-1].out_channels, 2)


class Stage375BottleneckMaskTests(unittest.TestCase):
    def _small_model(self):
        return IQUBiMamba1D_BottleneckRealMask(
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
            bimamba_apply_stages=[1, 3],
            bimamba_residual_scale_init=1.0,
            complex_state_d_state=2,
            complex_state_d_conv=4,
            complex_state_expand=1,
            complex_state_scan_checkpoint=False,
            complex_state_scan_backend="torch",
            complex_state_fusion_hidden=8,
            rf_apply_stages=[0, 1, 2],
            rf_residual_scale_init=0.05,
            rf_large_kernel=17,
            rf_ffn_factor=2,
            rf_layer_scale=1e-6,
        )

    def test_forward_preserves_fixed_source_contract(self):
        model = self._small_model().eval()
        output = model(torch.randn(2, 2, 64))
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertTrue(torch.isfinite(output).all())

    def test_bottleneck_masks_are_simplex(self):
        model = self._small_model().eval()
        x = torch.randn(2, 2, 64)
        with torch.no_grad():
            skips = model._encode_skips(x)
            masks = model._make_bottleneck_masks(skips[-1])
        self.assertEqual(tuple(masks.shape[:2]), (2, 2))
        torch.testing.assert_close(
            masks.sum(dim=1),
            torch.ones_like(masks.sum(dim=1)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_batched_decoder_matches_per_slot_decoder(self):
        model = self._small_model().eval()
        x = torch.randn(2, 2, 64)
        with torch.no_grad():
            skips = model._encode_skips(x)
            masks = model._make_bottleneck_masks(skips[-1])
            batched = model.decoder(model._flatten_source_slots(skips, masks))
            per_slot = []
            for slot in range(model.bottleneck_mask_num_sources):
                slot_skips = list(skips[:-1]) + [skips[-1] * masks[:, slot]]
                per_slot.append(model.decoder(slot_skips))
            per_slot = torch.stack(per_slot, dim=1).reshape_as(batched)
        torch.testing.assert_close(batched, per_slot, atol=1e-6, rtol=1e-6)

    def test_registration(self):
        self.assertIn(375, supported_stage_ids())
        config = MambaConfig(
            str(PROJECT_ROOT / "config" / "model_config_stage375_stage365_bottleneck_mask_real.yaml")
        )
        config._load_enc_config()
        self.assertEqual(config.model_type, "bimamba_bottleneck_mask_real")
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(model, IQUBiMamba1D_BottleneckRealMask)
        self.assertEqual(model.decoder.seg_layers[-1].out_channels, 2)


if __name__ == "__main__":
    unittest.main()
