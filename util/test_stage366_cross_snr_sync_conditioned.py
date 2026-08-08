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


from models.IQUMamba1D import MambaLayer
from models.IQUMamba1D_SyncConditioned import (
    IQUMamba1D_SyncConditioned,
    SyncFiLM1D,
)
from util.config import MambaConfig
from util.low_snr_training import (
    cross_snr_teacher_consistency_loss,
    sync_parameter_cross_snr_consistency_loss,
    sync_parameter_snr_supervision_loss,
)
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "model_config_stage366_stage4_cross_snr_sync_conditioned.yaml"


def _small_model(input_size=64):
    return IQUMamba1D_SyncConditioned(
        input_size=input_size,
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
        sync_hidden=16,
        sync_lags=[1, 2, 4],
        sync_sps_candidates=[8, 10, 20],
    )


class Stage366StructureTests(unittest.TestCase):
    def test_keeps_exact_four_level_stage4_memory_placement(self):
        model = _small_model()
        self.assertEqual(len(model.backbone.encoder.stages), 4)
        self.assertEqual(len(model.backbone.decoder.stages), 3)
        self.assertEqual(len(model.sync_film), 4)
        self.assertTrue(all(isinstance(layer, SyncFiLM1D) for layer in model.sync_film))
        self.assertIsInstance(model.backbone.encoder.mamba_layers[0], torch.nn.Identity)
        self.assertIsInstance(model.backbone.encoder.mamba_layers[1], MambaLayer)
        self.assertIsInstance(model.backbone.encoder.mamba_layers[2], torch.nn.Identity)
        self.assertIsInstance(model.backbone.encoder.mamba_layers[3], MambaLayer)

    def test_zero_initialized_film_is_exact_stage4_at_initialization(self):
        torch.manual_seed(366)
        model = _small_model().eval()
        x = torch.randn(2, 2, 64)
        with torch.no_grad():
            baseline = model.backbone(x)
            conditioned, auxiliary = model(x)
        self.assertTrue(torch.equal(conditioned, baseline))
        self.assertEqual(tuple(auxiliary["sync_condition"].shape), (2, 9))

    def test_explicit_parameters_have_valid_domains_and_odd_length_is_restored(self):
        model = _small_model(input_size=65).eval()
        with torch.no_grad():
            output, auxiliary = model(torch.randn(2, 2, 65))
        self.assertEqual(tuple(output.shape), (2, 4, 65))
        self.assertTrue(((auxiliary["snr_prediction"] >= -10.0) & (auxiliary["snr_prediction"] <= 30.0)).all())
        self.assertTrue((auxiliary["cfo_cycles_per_sample"].abs() <= 0.25).all())
        self.assertTrue((auxiliary["timing_offset_unit"] >= 0.0).all())
        self.assertTrue((auxiliary["timing_offset_unit"] <= 1.0).all())
        self.assertTrue(torch.allclose(auxiliary["phase_vector"].norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(auxiliary["sps_probabilities"].sum(dim=-1), torch.ones(2), atol=1e-6))
        self.assertTrue((auxiliary["phase_drift_rad_per_sample"].abs() <= 0.05).all())

    def test_separation_and_snr_supervision_backpropagate(self):
        torch.manual_seed(367)
        model = _small_model()
        x = torch.randn(2, 2, 64, requires_grad=True)
        output, auxiliary = model(x)
        loss = output.square().mean() + sync_parameter_snr_supervision_loss(
            auxiliary,
            torch.tensor([-6.0, 10.0]),
        )
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(model.sync_estimator.head.weight.grad)
        self.assertIsNotNone(model.sync_film[0].projection.weight.grad)

    def test_rejects_accidental_fifth_unet_level(self):
        with self.assertRaisesRegex(ValueError, "four-level"):
            IQUMamba1D_SyncConditioned(
                input_size=64,
                input_channels=2,
                n_stages=5,
                features_per_stage=[4, 8, 16, 32, 64],
                conv_op=torch.nn.Conv1d,
                kernel_sizes=[3] * 5,
                strides=[1, 2, 2, 2, 2],
                n_conv_per_stage=[1] * 5,
                num_classes=4,
                n_conv_per_stage_decoder=[1] * 5,
            )


class Stage366DistillationTests(unittest.TestCase):
    @staticmethod
    def _aux(offset, requires_grad=False):
        def value(shape, scale=1.0):
            return (torch.full(shape, float(offset)) * scale).requires_grad_(requires_grad)

        return {
            "cfo_cycles_per_sample": value((2,), 0.01),
            "phase_vector": value((2, 2), 0.1),
            "timing_offset_unit": value((2,), 0.1),
            "sps_probabilities": value((2, 3), 0.1),
            "phase_drift_rad_per_sample": value((2,), 0.001),
        }

    def test_low_view_receives_gradients_and_ema_teacher_is_detached(self):
        original = self._aux(1.0, requires_grad=True)
        partner = self._aux(2.0, requires_grad=True)
        teacher = self._aux(0.0, requires_grad=True)
        loss = sync_parameter_cross_snr_consistency_loss(
            original,
            partner,
            torch.tensor([-10.0, 10.0]),
            torch.tensor([10.0, -10.0]),
            teacher_high_auxiliary=teacher,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(original["cfo_cycles_per_sample"].grad)
        self.assertIsNotNone(partner["cfo_cycles_per_sample"].grad)
        self.assertIsNone(teacher["cfo_cycles_per_sample"].grad)

    def test_separation_distillation_accepts_detached_ema_high_view(self):
        torch.manual_seed(368)
        original = torch.randn(2, 4, 32, requires_grad=True)
        partner = torch.randn(2, 4, 32, requires_grad=True)
        teacher = torch.randn(2, 4, 32, requires_grad=True)
        targets = torch.randn(2, 4, 32)
        loss = cross_snr_teacher_consistency_loss(
            original,
            partner,
            targets,
            torch.tensor([-10.0, 10.0]),
            torch.tensor([10.0, -10.0]),
            num_sources=2,
            shared_permutation=True,
            teacher_high_outputs=teacher,
        )
        loss.backward()
        self.assertIsNotNone(original.grad)
        self.assertIsNotNone(partner.grad)
        self.assertIsNone(teacher.grad)

    def test_ema_teacher_is_saved_restored_and_updated(self):
        source = (PROJECT_ROOT / "util" / "training.py").read_text(encoding="utf-8")
        self.assertIn("'cross_snr_teacher_state_dict'", source)
        self.assertIn("resume_cross_snr_teacher_state", source)
        self.assertIn("def _update_cross_snr_teacher", source)
        self.assertGreaterEqual(source.count("_update_cross_snr_teacher()"), 3)


class Stage366RegistrationTests(unittest.TestCase):
    def test_config_registry_and_training_controls(self):
        self.assertIn(366, supported_stage_ids())
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn(f'366: CONFIG_ROOT / "{CONFIG_NAME}"', main_source)
        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIG_NAME))
        config._load_enc_config()
        self.assertEqual(config.model_type, "iqumamba_sync_conditioned")
        self.assertEqual(config.n_stages, 4)
        self.assertTrue(config.cross_snr_enable)
        self.assertTrue(config.cross_snr_ema_teacher_enable)
        self.assertEqual(config.cross_snr_low_final_db, -10.0)
        self.assertGreater(config.sync_snr_aux_weight, 0.0)
        self.assertGreater(config.sync_cross_snr_consistency_weight, 0.0)

    def test_factory_dispatches_stage366(self):
        config = MambaConfig(str(PROJECT_ROOT / "config" / CONFIG_NAME))
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=64,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(model, IQUMamba1D_SyncConditioned)
        self.assertEqual(len(model.backbone.encoder.stages), 4)
        self.assertEqual(len(model.backbone.decoder.stages), 3)


if __name__ == "__main__":
    unittest.main()
