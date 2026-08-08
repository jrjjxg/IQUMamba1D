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


from models.IQUMamba1D import IQUMamba1D
from models.IQUMamba1D_SyncConditioned import IQUMamba1D_SyncConditioned
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids
from util.utils import Create_Mamba_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    366: "model_config_stage366_stage4_cross_snr_sync_conditioned.yaml",
    367: "model_config_stage367_stage4_cross_snr_ema.yaml",
    368: "model_config_stage368_stage4_sync_conditioned.yaml",
}


def _config(stage):
    return MambaConfig(str(PROJECT_ROOT / "config" / CONFIGS[stage]))


def _model(stage, input_size=64):
    return Create_Mamba_model(
        _config(stage),
        logger=None,
        input_size_=input_size,
        device_override=torch.device("cpu"),
    )


class Stage367368RegistrationTests(unittest.TestCase):
    def test_stages_are_registered_and_mapped(self):
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage in (367, 368):
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(
                    f'{stage}: CONFIG_ROOT / "{CONFIGS[stage]}"',
                    main_source,
                )

    def test_both_ablations_keep_the_exact_four_level_stage4_shape(self):
        for stage in (367, 368):
            with self.subTest(stage=stage):
                config = _config(stage)
                config._load_enc_config()
                self.assertEqual(config.n_stages, 4)
                self.assertEqual(config.features_per_stage, [32, 64, 128, 256])
                self.assertEqual(config.strides, [1, 2, 2, 2])
                model = _model(stage)
                if stage == 367:
                    encoder, decoder = model.encoder, model.decoder
                else:
                    encoder, decoder = model.backbone.encoder, model.backbone.decoder
                self.assertEqual(len(encoder.stages), 4)
                self.assertEqual(len(decoder.stages), 3)

    def test_both_ablations_copy_the_plain_stage4_backbone_config(self):
        baseline = MambaConfig(
            str(PROJECT_ROOT / "config" / "model_config_IQ_stage4.yaml")
        ).model_config
        keys = (
            "input_channels",
            "num_classes",
            "n_stages",
            "features_per_stage",
            "kernel_sizes",
            "strides",
            "n_conv_per_stage",
            "n_conv_per_stage_decoder",
            "deep_supervision",
        )
        for stage in (367, 368):
            with self.subTest(stage=stage):
                ablation = _config(stage).model_config
                self.assertEqual(
                    {key: ablation[key] for key in keys},
                    {key: baseline[key] for key in keys},
                )


class Stage367CrossSNROnlyTests(unittest.TestCase):
    def test_factory_builds_plain_stage4_without_sync_modules(self):
        model = _model(367)
        self.assertIsInstance(model, IQUMamba1D)
        self.assertNotIsInstance(model, IQUMamba1D_SyncConditioned)
        self.assertFalse(hasattr(model, "sync_estimator"))
        with torch.no_grad():
            output = model(torch.randn(1, 2, 64))
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (1, 4, 64))

    def test_cross_snr_and_ema_settings_exactly_match_stage366(self):
        full = _config(366).model_config
        ablation = _config(367).model_config
        keys = (
            "cross_snr_enable",
            "cross_snr_probability",
            "cross_snr_high_db",
            "cross_snr_low_start_db",
            "cross_snr_low_middle_db",
            "cross_snr_low_final_db",
            "cross_snr_first_fraction",
            "cross_snr_second_fraction",
            "cross_snr_pair_weight",
            "cross_snr_consistency_weight",
            "cross_snr_consistency_beta",
            "cross_snr_eps",
            "cross_snr_shared_permutation",
            "cross_snr_ema_teacher_enable",
            "cross_snr_ema_decay",
        )
        self.assertEqual(
            {key: ablation[key] for key in keys},
            {key: full[key] for key in keys},
        )
        self.assertEqual(ablation["sync_snr_aux_weight"], 0.0)
        self.assertEqual(ablation["sync_cross_snr_consistency_weight"], 0.0)
        self.assertNotIn("sync_hidden", ablation)


class Stage368SyncOnlyTests(unittest.TestCase):
    def test_factory_builds_sync_conditioned_stage4_without_cross_snr(self):
        config = _config(368)
        config._load_enc_config()
        self.assertEqual(config.model_type, "iqumamba_sync_conditioned")
        self.assertFalse(config.cross_snr_enable)
        self.assertFalse(config.cross_snr_ema_teacher_enable)
        self.assertEqual(config.sync_cross_snr_consistency_weight, 0.0)

        model = _model(368)
        self.assertIsInstance(model, IQUMamba1D_SyncConditioned)
        with torch.no_grad():
            output, auxiliary = model(torch.randn(1, 2, 64))
        self.assertEqual(tuple(output.shape), (1, 4, 64))
        self.assertIn("snr_prediction", auxiliary)
        self.assertIn("cfo_cycles_per_sample", auxiliary)

    def test_sync_architecture_and_snr_supervision_exactly_match_stage366(self):
        full = _config(366).model_config
        ablation = _config(368).model_config
        keys = (
            "sync_hidden",
            "sync_lags",
            "sync_sps_candidates",
            "sync_snr_min_db",
            "sync_snr_max_db",
            "sync_max_cfo_cycles_per_sample",
            "sync_max_phase_drift_rad_per_sample",
            "sync_sps_temperature",
            "sync_film_max_delta",
            "sync_eps",
            "sync_snr_aux_weight",
            "sync_snr_aux_min_db",
            "sync_snr_aux_max_db",
            "sync_snr_aux_beta",
        )
        self.assertEqual(
            {key: ablation[key] for key in keys},
            {key: full[key] for key in keys},
        )

        full_model = _model(366)
        ablation_model = _model(368)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in ablation_model.state_dict().items()},
            {key: tuple(value.shape) for key, value in full_model.state_dict().items()},
        )


if __name__ == "__main__":
    unittest.main()
