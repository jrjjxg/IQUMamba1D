"""Contract tests for IQUMamba stages exposed by the RF Challenge pipeline."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest

import torch


# CPU-only CI can validate the pipeline contract without compiling the CUDA
# mamba_ssm extension.  Real training environments continue to use the actual
# package because this stub is installed only when the dependency is absent.
if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
    mamba_stub = types.ModuleType("mamba_ssm")

    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model: int, *_args, **_kwargs) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(int(d_model), int(d_model))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.projection(x)

    mamba_stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = mamba_stub


from rfchallenge.models import (  # noqa: E402
    RFCHALLENGE_STAGE_CONFIGS,
    build_single_soi_model,
    extract_single_soi_output,
    resolve_stage_config,
    supported_rfchallenge_stages,
)
from util.config import MambaConfig  # noqa: E402


NEW_STAGES = (
    235, 290, 295, 299, 309, 310, 333, 336, 342, 350, 351, 352, 353,
    354, 355, 356, 357, 358,
)
EXPECTED_MODEL_TYPES = {
    235: "bimamba_cross_scale_single",
    290: "iqumamba_stage4_complex_c1",
    295: "iqumamba_stage4_complex_state",
    299: "iqumamba_stage4_complex_stem_complex_state",
    309: "iqumamba_recent_rf",
    310: "iqumamba_recent_rf",
    333: "iqumamba_rf_mamba3",
    336: "iqumamba_stage4_s4d",
    342: "iqumamba_full_rf_mamba3_combination",
    350: "iqumamba_strong_rf_combination",
    351: "iqumamba_rf_mamba3",
    352: "iqumamba_stage4_s4d",
    353: "iqumamba_stage4_s4d_unireplk",
    354: "iqumamba_rf_mamba3",
    355: "iqumamba_rf_mamba3",
    356: "iqumamba_rf_mamba3",
    357: "iqumamba_stage4_complex_s4d",
    358: "rfchallenge_rfdemucs",
}


class RFChallengeStageConfigTests(unittest.TestCase):
    def test_new_stages_have_dedicated_single_soi_configs(self) -> None:
        self.assertTrue(set(NEW_STAGES).issubset(supported_rfchallenge_stages()))
        for stage in NEW_STAGES:
            with self.subTest(stage=stage):
                path = resolve_stage_config(stage)
                self.assertEqual(path, RFCHALLENGE_STAGE_CONFIGS[stage])
                self.assertTrue(path.is_file())
                config = MambaConfig(str(path)).model_config
                self.assertEqual(config["model_type"], EXPECTED_MODEL_TYPES[stage])
                self.assertEqual(config["input_channels"], 2)
                self.assertEqual(config["num_classes"], 2)
                self.assertFalse(config["deep_supervision"])

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not registered"):
            resolve_stage_config(9999)


class RFChallengeStageForwardTests(unittest.TestCase):
    def test_new_stages_build_and_preserve_single_soi_shape(self) -> None:
        overrides = {
            "features_per_stage": [4, 8, 16, 32],
            "n_conv_per_stage": [1, 1, 1, 1],
            "n_conv_per_stage_decoder": [1, 1, 1, 1],
            "scan_backend": "torch",
            "scan_checkpoint": False,
            "memory_d_state": 8,
            "rf_ffn_factor": 2,
            "unireplk_ffn_factor": 2,
            "rfdemucs_hidden": 4,
            "rfdemucs_depth": 2,
            "rfdemucs_kernel_size": 8,
            "rfdemucs_stride": 2,
            "rfdemucs_resample": 2,
            "rfdemucs_lstm_layers": 1,
        }
        for stage in NEW_STAGES:
            with self.subTest(stage=stage):
                torch.manual_seed(stage)
                model, config = build_single_soi_model(
                    resolve_stage_config(stage),
                    frame_length=64,
                    device="cpu",
                    model_overrides=overrides,
                )
                model.eval()
                with torch.no_grad():
                    output = extract_single_soi_output(
                        model(torch.randn(1, 2, 64))
                    )
                self.assertEqual(tuple(output.shape), (1, 2, 64))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(config.input_channels, 2)
                self.assertEqual(config.num_classes, 2)
                self.assertFalse(config.deep_supervision)
                del model


if __name__ == "__main__":
    unittest.main()
