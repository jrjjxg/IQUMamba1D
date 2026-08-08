"""Strict pairwise ablation checks for the three Stage-333 mechanisms."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import yaml

from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

STANDARD_CONFIGS = {
    354: CONFIG_ROOT / "model_config_stage354_rf_mamba3_a2_a3.yaml",
    355: CONFIG_ROOT / "model_config_stage355_rf_mamba3_a2_a4.yaml",
    356: CONFIG_ROOT / "model_config_stage356_rf_mamba3_a3_a4.yaml",
}
RFCHALLENGE_CONFIGS = {
    stage: CONFIG_ROOT / f"model_config_stage{stage}_rfchallenge.yaml"
    for stage in STANDARD_CONFIGS
}
EXPECTED_FLAGS = {
    # discretization, cyclic theta (A3), reliability (A4)
    354: ("exponential_trapezoidal", True, False),   # A2 + A3
    355: ("exponential_trapezoidal", False, True),   # A2 + A4
    356: ("exponential_euler", True, True),          # A3 + A4
}
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _model_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["model_config"]


class Stage333StrictPairwiseConfigTests(unittest.TestCase):
    def _assert_exact_stage333_ablation(
        self,
        reference_path: Path,
        variant_paths: dict[int, Path],
    ) -> None:
        reference = _model_config(reference_path)
        for stage, path in variant_paths.items():
            expected = deepcopy(reference)
            discretization, cyclic_enable, reliability_enable = EXPECTED_FLAGS[stage]
            expected["mamba_discretization"] = discretization
            expected["cyclic_theta_enable"] = cyclic_enable
            expected["reliability_enable"] = reliability_enable
            self.assertEqual(
                _model_config(path),
                expected,
                f"Stage {stage} must differ from Stage 333 only by its missing mechanism",
            )

    def test_standard_configs_are_exact_stage333_pairwise_ablations(self):
        self._assert_exact_stage333_ablation(
            CONFIG_ROOT / "model_config_stage333_rf_mamba3_combined.yaml",
            STANDARD_CONFIGS,
        )

    def test_rfchallenge_configs_are_exact_stage333_pairwise_ablations(self):
        self._assert_exact_stage333_ablation(
            CONFIG_ROOT / "model_config_stage333_rfchallenge.yaml",
            RFCHALLENGE_CONFIGS,
        )

    def test_stages_are_registered_in_both_pipelines(self):
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        rfchallenge_source = (
            PROJECT_ROOT / "rfchallenge" / "models.py"
        ).read_text(encoding="utf-8")
        for stage, path in STANDARD_CONFIGS.items():
            self.assertIn(stage, supported_stage_ids())
            self.assertIn(
                f'{stage}: CONFIG_ROOT / "{path.name}"',
                main_source,
            )
            self.assertIn(
                f'{stage}: PACKAGE_ROOT / "config" / '
                f'"{RFCHALLENGE_CONFIGS[stage].name}"',
                rfchallenge_source,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required for model construction")
class Stage333StrictPairwiseModelTests(unittest.TestCase):
    def test_runtime_modules_match_the_pairwise_flags(self):
        import torch

        from models.IQUMamba1D_ComplexStateMamba import ComplexStateMambaLayer
        from util.utils import Create_Mamba_model

        for stage, path in STANDARD_CONFIGS.items():
            config = MambaConfig(str(path))
            model = Create_Mamba_model(
                config,
                logger=None,
                input_size_=64,
                device_override=torch.device("cpu"),
            )
            layers = [
                layer
                for layer in model.encoder.mamba_layers
                if isinstance(layer, ComplexStateMambaLayer)
            ]
            self.assertEqual(len(layers), 2)
            expected_discretization, expected_cyclic, expected_reliability = (
                EXPECTED_FLAGS[stage]
            )
            for layer in layers:
                ssm = layer.ssm
                self.assertEqual(ssm.discretization, expected_discretization)
                self.assertEqual(ssm.cyclic_theta_enable, expected_cyclic)
                self.assertEqual(ssm.reliability_enable, expected_reliability)
                self.assertTrue(ssm.scan_checkpoint)
                self.assertEqual(ssm.scan_backend, "auto")
                self.assertEqual(
                    ssm.trapezoid_lambda_weight is not None,
                    expected_discretization == "exponential_trapezoidal",
                )
                self.assertEqual(ssm.reliability_net is not None, expected_reliability)
                self.assertEqual(ssm.theta_anchor_angular.numel() > 0, expected_cyclic)


if __name__ == "__main__":
    unittest.main()
