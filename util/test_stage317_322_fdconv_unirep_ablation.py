import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
    mamba_stub = types.ModuleType("mamba_ssm")

    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.projection = torch.nn.Linear(int(d_model), int(d_model))

        def forward(self, x):
            return self.projection(x)

    mamba_stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = mamba_stub


if (
    "dynamic_network_architectures" not in sys.modules
    and importlib.util.find_spec("dynamic_network_architectures") is None
):
    root = types.ModuleType("dynamic_network_architectures")
    blocks = types.ModuleType("dynamic_network_architectures.building_blocks")
    helper = types.ModuleType("dynamic_network_architectures.building_blocks.helper")
    residual = types.ModuleType("dynamic_network_architectures.building_blocks.residual")

    def maybe_convert_scalar_to_list(_conv_op, value):
        return [value] if isinstance(value, int) else value

    class BasicBlockD(torch.nn.Module):
        def __init__(self, conv_op, input_channels, output_channels, kernel_size,
                     stride=1, conv_bias=False, **_kwargs):
            super().__init__()
            kernel = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
            stride_value = stride[0] if isinstance(stride, (list, tuple)) else stride
            self.conv = conv_op(
                input_channels, output_channels, kernel, stride=stride_value,
                padding=kernel // 2, bias=conv_bias,
            )

        def forward(self, x):
            return self.conv(x)

    helper.maybe_convert_scalar_to_list = maybe_convert_scalar_to_list
    residual.BasicBlockD = BasicBlockD
    sys.modules[root.__name__] = root
    sys.modules[blocks.__name__] = blocks
    sys.modules[helper.__name__] = helper
    sys.modules[residual.__name__] = residual


from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    FrequencyDynamicConv1D,
    IQUMamba1DFDConvUniRepAblation,
    ScaleAwareParallelFusion1D,
    UniRepLKNetBlock1D,
)
from util.stage_registry import supported_stage_ids


CONFIGS = {
    317: "model_config_stage317_fdconv_unireplk_serial.yaml",
    318: "model_config_stage318_fdconv_unireplk_hierarchical.yaml",
    319: "model_config_stage319_fdconv_unireplk_parallel.yaml",
    320: "model_config_stage320_stage4_no_mamba.yaml",
    321: "model_config_stage321_no_mamba_fdconv.yaml",
    322: "model_config_stage322_no_mamba_unireplk.yaml",
}


def build_small(variant: str) -> IQUMamba1DFDConvUniRepAblation:
    return IQUMamba1DFDConvUniRepAblation(
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
        rf_variant=variant,
        rf_module_config={
            "fdconv_kernel_size": 15,
            "fdconv_bands": 3,
            "unireplk_large_kernel": 17,
            "unireplk_ffn_factor": 2,
        },
    )


class Stage317To322Tests(unittest.TestCase):
    def test_serial_uses_full_309_then_310_placements(self):
        model = build_small("serial")
        self.assertEqual(set(model.stage_rf), {"0", "1", "2"})
        stage0 = model.stage_rf["0"]
        self.assertIsInstance(stage0, torch.nn.Sequential)
        self.assertIsInstance(stage0[0].operator, FrequencyDynamicConv1D)
        self.assertIsInstance(stage0[1].operator, UniRepLKNetBlock1D)
        self.assertIsInstance(model.stage_rf["1"].operator, UniRepLKNetBlock1D)
        self.assertIsInstance(model.stage_rf["2"].operator, UniRepLKNetBlock1D)

    def test_hierarchical_has_no_stage0_unireplk(self):
        model = build_small("hierarchical")
        self.assertIsInstance(model.stage_rf["0"].operator, FrequencyDynamicConv1D)
        self.assertIsInstance(model.stage_rf["1"].operator, UniRepLKNetBlock1D)
        self.assertIsInstance(model.stage_rf["2"].operator, UniRepLKNetBlock1D)

    def test_parallel_fusion_matches_scales_and_starts_balanced(self):
        model = build_small("parallel")
        fusion = model.stage_rf["0"]
        self.assertIsInstance(fusion, ScaleAwareParallelFusion1D)
        x = torch.randn(2, 4, 65)
        branch = 20 * torch.randn_like(x)
        matched = fusion._match_input_scale(branch, x)
        torch.testing.assert_close(
            matched.float().square().mean(-1).sqrt(),
            x.float().square().mean(-1).sqrt(),
            atol=2e-5, rtol=2e-5,
        )
        output = fusion(x)
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(
            fusion.last_gates, torch.full_like(fusion.last_gates, 0.5),
        )

    def test_no_mamba_variants_remove_every_mamba_layer(self):
        expected_stages = {
            "no_mamba": set(),
            "no_mamba_fdconv": {"0"},
            "no_mamba_unireplk": {"0", "1", "2"},
        }
        for variant, stages in expected_stages.items():
            with self.subTest(variant=variant):
                model = build_small(variant)
                self.assertTrue(all(
                    isinstance(layer, torch.nn.Identity)
                    for layer in model.encoder.mamba_layers
                ))
                self.assertEqual(set(model.stage_rf), stages)

    def test_mamba_is_retained_in_combination_variants(self):
        for variant in ("serial", "hierarchical", "parallel"):
            with self.subTest(variant=variant):
                model = build_small(variant)
                self.assertTrue(any(
                    not isinstance(layer, torch.nn.Identity)
                    for layer in model.encoder.mamba_layers
                ))

    def test_all_variants_forward_backward_and_new_paths_train(self):
        for variant in sorted(IQUMamba1DFDConvUniRepAblation.VARIANTS):
            with self.subTest(variant=variant):
                torch.manual_seed(317)
                model = build_small(variant)
                x = torch.randn(2, 2, 64, requires_grad=True)
                output = model(x)
                self.assertEqual(output.shape, (2, 4, 64))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                if model.stage_rf:
                    rf_grad = sum(
                        parameter.grad.abs().sum().item()
                        for parameter in model.stage_rf.parameters()
                        if parameter.grad is not None
                    )
                    self.assertGreater(rf_grad, 0.0)

    def test_no_weight_decay_names_resolve_to_parameters(self):
        for variant in ("serial", "hierarchical", "parallel"):
            model = build_small(variant)
            parameter_names = dict(model.named_parameters())
            for name in model.no_weight_decay():
                self.assertIn(name, parameter_names)

    def test_stage_configs_register_and_build_through_factory(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        root = Path(__file__).resolve().parents[1]
        supported = supported_stage_ids()
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported)
                path = root / "config" / filename
                self.assertTrue(path.is_file())
                config = MambaConfig(str(path))
                model = Create_Mamba_model(
                    config, logger=None, input_size_=64,
                    device_override=torch.device("cpu"),
                )
                output = model(torch.randn(1, 2, 64))
                self.assertEqual(output.shape, (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
