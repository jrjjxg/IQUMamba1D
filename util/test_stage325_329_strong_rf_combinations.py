import unittest
from pathlib import Path

import torch

# Install dependency stubs when the optional CUDA/runtime packages are absent.
import util.test_stage317_322_fdconv_unirep_ablation  # noqa: F401

from models.IQUBiMamba1D import BiMambaLayer
from models.IQUMamba1D_ComplexStage4 import ComplexStem1d
from models.IQUMamba1D_EstimatedCycloFRESH import EstimatedCycloFRESHAdapter1D
from models.IQUMamba1D_RecentRFModules import (
    FrequencyDynamicConv1D,
    UniRepLKNetBlock1D,
)
from models.IQUMamba1D_StrongRFCombinations import (
    IQUMamba1DStrongRFCombination,
)
from util.stage_registry import supported_stage_ids


CONFIGS = {
    325: "model_config_stage325_stage290_fdconv.yaml",
    326: "model_config_stage326_stage290_unireplk.yaml",
    327: "model_config_stage327_stage290_fdconv_unireplk.yaml",
    328: "model_config_stage328_stage197_unireplk.yaml",
    329: "model_config_stage329_stage79_unireplk.yaml",
    350: "model_config_stage350_fresh_complex_unireplk.yaml",
}


def build_small(variant: str) -> IQUMamba1DStrongRFCombination:
    return IQUMamba1DStrongRFCombination(
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
        combination_variant=variant,
        rf_module_config={
            "fdconv_kernel_size": 15,
            "fdconv_bands": 3,
            "unireplk_large_kernel": 17,
            "unireplk_ffn_factor": 2,
        },
    )


class Stage325To329Tests(unittest.TestCase):
    def test_complex_variants_reuse_stage290_stem(self):
        for variant in (
            "complex_fdconv", "complex_unireplk", "complex_hierarchical",
            "complex_cyclofresh_unireplk",
        ):
            with self.subTest(variant=variant):
                model = build_small(variant)
                self.assertIsInstance(model.encoder.stem, ComplexStem1d)
                self.assertEqual(
                    model.uses_cyclofresh,
                    variant == "complex_cyclofresh_unireplk",
                )

    def test_rf_placements_match_requested_combinations(self):
        fdconv = build_small("complex_fdconv")
        self.assertEqual(set(fdconv.stage_rf), {"0"})
        self.assertIsInstance(fdconv.stage_rf["0"].operator, FrequencyDynamicConv1D)

        for variant in (
            "complex_unireplk",
            "bimamba_cyclofresh_unireplk",
            "mamba_cyclofresh_unireplk",
            "complex_cyclofresh_unireplk",
        ):
            with self.subTest(variant=variant):
                model = build_small(variant)
                self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
                self.assertTrue(all(
                    isinstance(adapter.operator, UniRepLKNetBlock1D)
                    for adapter in model.stage_rf.values()
                ))

        hierarchical = build_small("complex_hierarchical")
        self.assertIsInstance(
            hierarchical.stage_rf["0"].operator, FrequencyDynamicConv1D
        )
        self.assertTrue(all(
            isinstance(hierarchical.stage_rf[str(stage)].operator, UniRepLKNetBlock1D)
            for stage in (1, 2)
        ))

    def test_stage328_is_bimamba_and_328_329_use_cyclofresh(self):
        stage328 = build_small("bimamba_cyclofresh_unireplk")
        stage329 = build_small("mamba_cyclofresh_unireplk")
        self.assertIsInstance(stage328.input_adapter, EstimatedCycloFRESHAdapter1D)
        self.assertIsInstance(stage329.input_adapter, EstimatedCycloFRESHAdapter1D)
        self.assertTrue(any(
            isinstance(layer, BiMambaLayer)
            for layer in stage328.encoder.mamba_layers
        ))
        self.assertFalse(any(
            isinstance(layer, BiMambaLayer)
            for layer in stage329.encoder.mamba_layers
        ))

    def test_stage350_combines_fresh_complex_stem_and_real_unireplk(self):
        model = build_small("complex_cyclofresh_unireplk")
        self.assertIsInstance(model.input_adapter, EstimatedCycloFRESHAdapter1D)
        self.assertIsInstance(model.encoder.stem, ComplexStem1d)
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
        self.assertTrue(all(
            isinstance(adapter.operator, UniRepLKNetBlock1D)
            for adapter in model.stage_rf.values()
        ))
        self.assertFalse(any(
            isinstance(layer, BiMambaLayer)
            for layer in model.encoder.mamba_layers
        ))

    def test_all_variants_forward_backward(self):
        for variant in sorted(IQUMamba1DStrongRFCombination.VARIANTS):
            with self.subTest(variant=variant):
                torch.manual_seed(325)
                model = build_small(variant)
                x = torch.randn(2, 2, 64, requires_grad=True)
                output = model(x)
                self.assertEqual(output.shape, (2, 4, 64))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                rf_gradient = sum(
                    parameter.grad.abs().sum().item()
                    for parameter in model.stage_rf.parameters()
                    if parameter.grad is not None
                )
                self.assertGreater(rf_gradient, 0.0)

    def test_no_weight_decay_names_resolve(self):
        for variant in IQUMamba1DStrongRFCombination.VARIANTS:
            model = build_small(variant)
            parameters = dict(model.named_parameters())
            for name in model.no_weight_decay():
                self.assertIn(name, parameters)

    def test_real_configs_build_through_factory(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        root = Path(__file__).resolve().parents[1]
        supported = supported_stage_ids()
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported)
                config = MambaConfig(str(root / "config" / filename))
                model = Create_Mamba_model(
                    config,
                    logger=None,
                    input_size_=64,
                    device_override=torch.device("cpu"),
                )
                output = model(torch.randn(1, 2, 64))
                self.assertEqual(output.shape, (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
