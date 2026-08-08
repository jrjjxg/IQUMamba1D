import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import numpy as np


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

if "scipy.io" not in sys.modules and importlib.util.find_spec("scipy") is None:
    scipy_stub = types.ModuleType("scipy")
    scipy_stub.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None)
    scipy_io_stub = types.ModuleType("scipy.io")
    scipy_io_stub.__spec__ = importlib.machinery.ModuleSpec("scipy.io", loader=None)
    scipy_io_stub.loadmat = lambda *args, **kwargs: None
    scipy_stub.io = scipy_io_stub
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.io"] = scipy_io_stub

if "tqdm" not in sys.modules and importlib.util.find_spec("tqdm") is None:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.__spec__ = importlib.machinery.ModuleSpec("tqdm", loader=None)
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_stub


from models.IQUMamba1D_PhysicalSyncRTN import IQUMamba1D_PhysicalSyncRTN
from data_loader.dataloader import (
    LightweightRFTrainAugmentDataset,
    MATLABSignalDataset,
    _fixed_protocol_sync_defaults,
)
from util.config import MambaConfig
from util.low_snr_training import (
    build_snr_view,
    cross_snr_feature_consistency_loss,
    cross_snr_teacher_consistency_loss,
    pit_align_sync_auxiliary,
    sample_progressive_snr_range,
    sync_parameter_cross_snr_consistency_loss,
    sync_parameter_physical_supervision_loss,
)
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model(num_sources=2, input_size=64):
    return IQUMamba1D_PhysicalSyncRTN(
        input_size=input_size,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        conv_op=torch.nn.Conv1d,
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=2 * num_sources,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        deep_supervision=False,
        sync_hidden=16,
        sync_lags=[1, 2, 4],
        sync_sps_candidates=[8, 10, 20],
    )


def _metadata(batch_size=2, num_sources=2):
    shape = (batch_size, num_sources)
    valid = torch.ones(shape, dtype=torch.bool)
    return {
        "sync_metadata_version": torch.ones(batch_size, dtype=torch.int64),
        "cfo_cycles_per_sample": torch.zeros(shape),
        "cfo_valid": valid.clone(),
        "phase_rad": torch.zeros(shape),
        "phase_valid": valid.clone(),
        "timing_offset_samples": torch.zeros(shape),
        "timing_valid": valid.clone(),
        "samples_per_symbol": torch.full(shape, 8.0),
        "sps_valid": valid.clone(),
        "phase_drift_rad_per_sample": torch.zeros(shape),
        "drift_valid": torch.zeros(shape, dtype=torch.bool),
    }


class PhysicalSyncRTNTests(unittest.TestCase):
    def test_four_level_backbone_and_per_source_outputs(self):
        model = _small_model(num_sources=3).eval()
        with torch.no_grad():
            output, auxiliary = model(torch.randn(2, 2, 64))
        self.assertEqual(tuple(output.shape), (2, 6, 64))
        self.assertEqual(tuple(auxiliary["cfo_cycles_per_sample"].shape), (2, 3))
        self.assertEqual(tuple(auxiliary["phase_vector"].shape), (2, 3, 2))
        self.assertEqual(tuple(auxiliary["sps_logits"].shape), (2, 3, 3))
        self.assertEqual(len(model.backbone.encoder.stages), 4)
        self.assertEqual(len(model.backbone.decoder.stages), 3)

    def test_zero_initialized_rtn_and_film_preserve_stage4_output(self):
        torch.manual_seed(369)
        model = _small_model().eval()
        x = torch.randn(2, 2, 64)
        with torch.no_grad():
            baseline = model.backbone(x)
            output, _ = model(x)
        self.assertTrue(torch.equal(output, baseline))

    def test_pit_aligned_physical_loss_backpropagates(self):
        model = _small_model()
        x = torch.randn(2, 2, 64)
        targets = torch.randn(2, 4, 64)
        outputs, auxiliary = model(x)
        loss = sync_parameter_physical_supervision_loss(
            auxiliary,
            _metadata(),
            outputs,
            targets,
            num_sources=2,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.sync_estimator.source_head.weight.grad)
        self.assertGreater(model.sync_estimator.source_head.weight.grad.abs().sum().item(), 0.0)

    def test_feature_distillation_detaches_teacher(self):
        student = torch.randn(2, 8, 10, requires_grad=True)
        teacher = torch.randn(2, 8, 10, requires_grad=True)
        loss = cross_snr_feature_consistency_loss(
            {"distillation_feature": student},
            {"distillation_feature": teacher},
        )
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_frozen_teacher_and_student_use_independent_pit(self):
        source_a = torch.zeros(1, 2, 16)
        source_b = torch.ones(1, 2, 16)
        targets = torch.cat([source_a, source_b], dim=1)
        student = targets.clone().requires_grad_(True)
        teacher = torch.cat([source_b, source_a], dim=1).requires_grad_(True)
        loss = cross_snr_teacher_consistency_loss(
            student,
            student,
            targets,
            torch.tensor([20.0]),
            torch.tensor([-5.0]),
            num_sources=2,
            shared_permutation=False,
            teacher_high_outputs=teacher,
            force_partner_student=True,
        )
        self.assertLess(float(loss.detach()), 1e-7)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_sync_states_follow_each_models_own_pit(self):
        source_a = torch.zeros(1, 2, 16)
        source_b = torch.ones(1, 2, 16)
        targets = torch.cat([source_a, source_b], dim=1)
        student_outputs = targets.clone()
        teacher_outputs = torch.cat([source_b, source_a], dim=1)

        def auxiliary(values):
            values = torch.tensor(values, dtype=torch.float32).view(1, 2)
            return {
                "cfo_cycles_per_sample": values * 1e-5,
                "phase_vector": torch.stack([values, values], dim=-1),
                "timing_offset_unit": values,
                "sps_probabilities": torch.stack([values, values, values], dim=-1),
                "phase_drift_rad_per_sample": values * 1e-5,
            }

        student_aux = pit_align_sync_auxiliary(
            auxiliary([1.0, 2.0]), student_outputs, targets, num_sources=2
        )
        teacher_aux = pit_align_sync_auxiliary(
            auxiliary([2.0, 1.0]), teacher_outputs, targets, num_sources=2
        )
        loss = sync_parameter_cross_snr_consistency_loss(
            student_aux,
            student_aux,
            torch.tensor([20.0]),
            torch.tensor([-5.0]),
            teacher_high_auxiliary=teacher_aux,
            force_partner_student=True,
            cfo_scale=1e-4,
            phase_drift_scale=1e-4,
        )
        self.assertLess(float(loss.detach()), 1e-7)

    def test_progressive_ranges_and_exact_snr_view(self):
        ranges = ((10.0, 30.0), (2.0, 30.0), (-10.0, 30.0))
        early = sample_progressive_snr_range(32, 0, 100, ranges)
        middle = sample_progressive_snr_range(32, 30, 100, ranges)
        late = sample_progressive_snr_range(32, 80, 100, ranges)
        self.assertTrue(bool(((early >= 10.0) & (early <= 30.0)).all()))
        self.assertTrue(bool(((middle >= 2.0) & (middle <= 30.0)).all()))
        self.assertTrue(bool(((late >= -10.0) & (late <= 30.0)).all()))
        targets = torch.randn(2, 4, 64)
        inputs = targets.view(2, 2, 2, 64).sum(dim=1) + 0.1 * torch.randn(2, 2, 64)
        view, snr = build_snr_view(
            inputs, targets, torch.tensor([-6.0, 10.0]), num_sources=2
        )
        self.assertEqual(tuple(view.shape), (2, 2, 64))
        self.assertTrue(torch.equal(snr, torch.tensor([-6.0, 10.0])))

    def test_cross_sample_remix_invalidates_stale_physical_labels(self):
        class _Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                targets = torch.randn(4, 32)
                mixture = targets.view(2, 2, 32).sum(dim=0)
                metadata = {
                    key: value[0].clone()
                    for key, value in _metadata(batch_size=1).items()
                }
                return mixture, targets, torch.tensor(20.0), metadata

        augmented = LightweightRFTrainAugmentDataset(
            _Dataset(),
            num_sources=2,
            source_phase_jitter_deg=0.0,
            source_gain_jitter_db=0.0,
            max_common_time_shift=0,
            global_phase_rotation=False,
            mix_enable=True,
            mix_prob=1.0,
            mix_cross_sample=True,
        )
        *_waveforms, metadata = augmented[0]
        for key in ("cfo_valid", "phase_valid", "timing_valid", "sps_valid", "drift_valid"):
            self.assertFalse(bool(metadata[key].any()))

    def test_matlab_metadata_is_per_source_and_calibrated(self):
        dataset = MATLABSignalDataset.__new__(MATLABSignalDataset)
        dataset.num_sources = 2
        dataset.data_choice = "8PSK-A"
        dataset._file_meta = [{
            "start": 0,
            "end": 3,
            "sample_rate_hz": 1_000.0,
            "cfo_hz_by_source": [10.0, -20.0],
            "initial_phase_rad_by_source": [0.2, -0.3],
            "delay_samples_by_source": [1, 2],
            "samples_per_symbol_by_source": [8, 10],
            "samples_per_symbol": None,
            "phase_drift_rad_per_sample_by_source": None,
            "frame_length": 64,
            "frame_initial_phase_rad_by_source": np.array(
                [[0.2, -0.3], [0.7, -0.8], [1.2, -1.3]], dtype=np.float64
            ),
        }]
        metadata = dataset._sync_metadata_for_index(1)
        self.assertTrue(
            torch.allclose(
                metadata["cfo_cycles_per_sample"], torch.tensor([0.01, -0.02])
            )
        )
        self.assertTrue(bool(metadata["cfo_valid"].all()))
        self.assertTrue(torch.allclose(metadata["phase_rad"], torch.tensor([0.7, -0.8])))
        self.assertTrue(torch.equal(metadata["samples_per_symbol"], torch.tensor([8.0, 10.0])))
        self.assertFalse(bool(metadata["drift_valid"].any()))

    def test_old_fixed_protocol_metadata_is_recoverable_without_guessing(self):
        defaults = _fixed_protocol_sync_defaults("8PSK-A", 2)
        self.assertEqual(defaults["sample_rate_hz"], 100e6)
        self.assertEqual(defaults["cfo_hz_by_source"], [-500.0, 500.0])
        self.assertEqual(defaults["delay_samples_by_source"], [0.0, 1.0])
        self.assertTrue(
            np.allclose(defaults["initial_phase_rad_by_source"], [0.0, np.pi / 3.0])
        )
        self.assertEqual(defaults["phase_first_sample_offset"], 0)
        self.assertIsNone(_fixed_protocol_sync_defaults("8PSK-H", 2))

    def test_nominal_frame_phase_is_advanced_and_drift_files_fail_closed(self):
        dataset = MATLABSignalDataset.__new__(MATLABSignalDataset)
        dataset.num_sources = 2
        dataset.data_choice = "8PSK-A"
        dataset._file_meta = [{
            "start": 0,
            "end": 3,
            "frame_length": 64,
            "sample_rate_hz": 1_000.0,
            "cfo_hz_by_source": [10.0, -20.0],
            "initial_phase_rad_by_source": [0.2, -0.3],
            "delay_samples_by_source": [0, 0],
            "samples_per_symbol_by_source": [8, 10],
            "samples_per_symbol": None,
            "phase_drift_rad_per_sample_by_source": None,
            "frame_initial_phase_rad_by_source": None,
        }]
        metadata = dataset._sync_metadata_for_index(1)
        expected = torch.tensor([0.2, -0.3]) + 2.0 * torch.pi * torch.tensor(
            [0.01, -0.02]
        ) * 65.0
        expected = torch.atan2(torch.sin(expected), torch.cos(expected))
        self.assertTrue(torch.allclose(metadata["phase_rad"], expected))
        self.assertTrue(bool(metadata["phase_valid"].all()))

        dataset.data_choice = "8PSK-H"
        metadata = dataset._sync_metadata_for_index(1)
        self.assertFalse(bool(metadata["phase_valid"].any()))

    def test_configs_factory_and_registry(self):
        self.assertIn(369, supported_stage_ids())
        self.assertIn(370, supported_stage_ids())
        for stage, filename in (
            (369, "model_config_stage369_physical_sync_teacher.yaml"),
            (370, "model_config_stage370_physical_sync_clean_teacher.yaml"),
        ):
            config = MambaConfig(str(PROJECT_ROOT / "config" / filename), train=True)
            config._load_enc_config()
            self.assertEqual(config.model_type, "iqumamba_physical_sync_rtn")
            self.assertIn(14, config.model_config["sync_sps_candidates"])
            self.assertIn(25, config.model_config["sync_sps_candidates"])
            self.assertAlmostEqual(
                config.model_config["sync_max_cfo_cycles_per_sample"], 1e-4
            )
            import util.utils as utils_module

            previous_input_size = getattr(utils_module, "input_size", None)
            previous_device = getattr(utils_module, "device", None)
            try:
                model = Create_Mamba_model(
                    config, logger=None, input_size_=64, device_override=torch.device("cpu")
                )
            finally:
                utils_module.input_size = previous_input_size
                utils_module.device = previous_device
            self.assertIsInstance(model, IQUMamba1D_PhysicalSyncRTN, msg=f"stage {stage}")

    def test_frozen_teacher_path_is_not_ema_updated(self):
        source = (PROJECT_ROOT / "util" / "training.py").read_text(encoding="utf-8")
        self.assertIn("cross_snr_teacher_mode != 'ema'", source)
        self.assertIn("--cross_snr_teacher_checkpoint from the same backbone architecture", source)
        self.assertIn("checkpoint is not architecture-compatible", source)
        student_config = MambaConfig(
            str(PROJECT_ROOT / "config" / "model_config_stage370_physical_sync_clean_teacher.yaml"),
            train=True,
        )
        student_config._load_enc_config()
        self.assertEqual(student_config.cross_snr_teacher_mode, "frozen")
        self.assertEqual(student_config.cross_snr_teacher_view, "clean")
        self.assertEqual(student_config.cross_snr_pair_mode, "curriculum_student")
        self.assertFalse(student_config.cross_snr_shared_permutation)
        teacher_config = MambaConfig(
            str(PROJECT_ROOT / "config" / "model_config_stage369_physical_sync_teacher.yaml"),
            train=True,
        )
        teacher_config._load_enc_config()
        self.assertEqual(teacher_config.training_snr_floor_db, 10.0)
        self.assertEqual(teacher_config.validation_snr_floor_db, 10.0)

if __name__ == "__main__":
    unittest.main()
