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


if importlib.util.find_spec("dynamic_network_architectures") is None:
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
    DCLSConv1D,
    DeformableConvV4OneD,
    DilatedReparamBlock1D,
    FrequencyDynamicConv1D,
    IQUMamba1DRecentRF,
    MixtureOfReceptiveFields1D,
    ModernTCNBlock1D,
    ReparamLargeKernelConv1D,
    RecentRFInputAdapter,
    ShiftwiseConv1D,
    WTConv1D,
    build_recent_rf_operator,
)
from util.stage_registry import supported_stage_ids


MODULE_CONFIGS = {
    "fadc": {"rf_kernel_size": 5, "rf_cutoffs": [2, 4]},
    "fdconv": {"rf_kernel_size": 15, "rf_bands": 3},
    "unireplk": {"rf_large_kernel": 17, "rf_ffn_factor": 2},
    "shiftwise": {"rf_big_kernel": 15, "rf_small_kernel": 3, "rf_paths": 2},
    "moderntcn": {"rf_kernel_size": 15, "rf_small_kernel": 3, "rf_expansion": 2},
    "morf": {"rf_kernels": [3, 7, 15], "rf_top_k": 2, "rf_routing": "soft"},
    "dcnv4": {"rf_points": 5, "rf_offset_scale": 1.0, "rf_groups": 4},
    "wtconv": {"rf_levels": 2, "rf_kernel_size": 3},
    "dcls": {"rf_taps": 5, "rf_max_offset": 8.0, "rf_dcls_version": "gauss"},
}


class RecentRFOperatorTests(unittest.TestCase):
    def test_all_operators_preserve_shape_and_backpropagate(self):
        for kind, config in MODULE_CONFIGS.items():
            with self.subTest(kind=kind):
                torch.manual_seed(308)
                operator = build_recent_rf_operator(kind, 8, config)
                x = torch.randn(2, 8, 65, requires_grad=True)
                output = operator(x)
                self.assertEqual(output.shape, x.shape)
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_all_operators_support_length4096_and_low_precision_dtype(self):
        for kind, config in MODULE_CONFIGS.items():
            with self.subTest(kind=kind):
                operator = build_recent_rf_operator(kind, 8, config).to(torch.bfloat16)
                x = torch.randn(1, 8, 4096, dtype=torch.bfloat16, requires_grad=True)
                output = operator(x)
                loss = output.float().square().mean()
                self.assertEqual(output.shape, x.shape)
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output).all())
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_unireplk_uses_official_kernel17_branches_and_reparameterizes(self):
        torch.manual_seed(310)
        block = DilatedReparamBlock1D(4, large_kernel=17)
        self.assertEqual(block.branch_kernels, (5, 9, 3, 3, 3))
        self.assertEqual(block.dilations, (1, 2, 4, 5, 7))
        block.eval()
        x = torch.randn(2, 4, 64)
        expected = block(x)
        block.reparameterize()
        actual = block(x)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_adapter_gives_operator_gradient_on_first_step(self):
        operator = build_recent_rf_operator("fadc", 8, MODULE_CONFIGS["fadc"])
        adapter = RecentRFInputAdapter(2, 8, operator, residual_scale_init=0.05)
        x = torch.randn(2, 2, 64, requires_grad=True)
        delta = adapter(x) - x
        self.assertLess(delta.square().mean().sqrt().item(), 0.25)
        adapter(x).square().mean().backward()
        operator_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in operator.parameters() if parameter.grad is not None
        )
        self.assertGreater(operator_grad, 0.0)

    def test_fdconv_fourier_experts_are_disjoint_and_complete(self):
        operator = FrequencyDynamicConv1D(4, kernel_size=15, bands=3)
        coverage = operator.frequency_masks.sum(dim=0)
        torch.testing.assert_close(coverage, torch.ones_like(coverage))
        self.assertTrue((operator.frequency_masks.flatten(1).sum(1) > 0).all())
        torch.testing.assert_close(
            operator.expert_kernels().sum(0), operator.weight, atol=2e-6, rtol=2e-6,
        )

    def test_shiftwise_preserves_ghost_channels(self):
        operator = ShiftwiseConv1D(8, big_kernel=15, small_kernel=3, ghost_ratio=0.25)
        x = torch.randn(2, 8, 33)
        y = operator(x)
        torch.testing.assert_close(y[:, operator.rep_channels:], x[:, operator.rep_channels:])
        self.assertEqual(operator.nk, 5)

    def test_moderntcn_large_kernel_reparameterization_is_exact(self):
        torch.manual_seed(312)
        block = ReparamLargeKernelConv1D(4, 15, 3).eval()
        x = torch.randn(2, 4, 64)
        expected = block(x)
        block.reparameterize()
        torch.testing.assert_close(block(x), expected, atol=2e-6, rtol=2e-6)
        modern = ModernTCNBlock1D(8, kernel_size=15, small_kernel=3, nvars=2)
        self.assertEqual((modern.nvars, modern.dmodel), (2, 4))

    def test_morf_kernel_fusion_matches_explicit_experts(self):
        torch.manual_seed(313)
        operator = MixtureOfReceptiveFields1D(4, (3, 7, 15), routing="soft")
        x = torch.randn(2, 4, 37)
        route = operator.routing_weights(x)
        explicit = 0
        for index, weight in enumerate(operator.expert_weights):
            response = torch.nn.functional.conv1d(
                x, weight, padding=weight.size(-1) // 2, groups=4,
            )
            explicit = explicit + route[:, index, None, None] * response
        torch.testing.assert_close(operator(x), explicit, atol=2e-6, rtol=2e-6)

    def test_dcnv4_masks_are_group_specific_and_not_softmaxed(self):
        operator = DeformableConvV4OneD(
            4, points=3, groups=2, offset_range=1.0, center_feature_scale=False,
        )
        with torch.no_grad():
            operator.value_proj.weight.zero_()
            operator.output_proj.weight.zero_()
            for index in range(4):
                operator.value_proj.weight[index, index, 0] = 1
                operator.output_proj.weight[index, index, 0] = 1
            operator.value_proj.bias.zero_()
            operator.output_proj.bias.zero_()
            bias = operator.offset_mask.bias.reshape(operator.groups, operator.points, 2)
            bias[..., 1] = 1.0
        x = torch.ones(1, 4, 9)
        y = operator(x)
        # Interior points sum three unit samples. A softmax implementation gives one.
        torch.testing.assert_close(y[..., 2:-2], torch.full_like(y[..., 2:-2], 3.0))
        self.assertEqual(operator.offset_mask.out_channels, 2 * 3 * 2)

    def test_wtconv_retains_official_base_path(self):
        operator = WTConv1D(4, levels=2, kernel_size=3)
        even = torch.randn(2, 4, 32)
        torch.testing.assert_close(
            operator._synthesis(operator._analysis(even)), even, atol=2e-6, rtol=2e-6,
        )
        for conv in operator.wavelet_convs:
            torch.nn.init.zeros_(conv.weight)
        x = torch.randn(2, 4, 35)
        torch.testing.assert_close(operator(x), operator.base_scale(operator.base_conv(x)))

    def test_dcls_gaussian_basis_is_normalized_and_positions_train(self):
        operator = DCLSConv1D(4, taps=5, max_offset=8, version="gauss")
        kernel = operator.constructed_kernel()
        torch.testing.assert_close(
            kernel.sum(-1), operator.weight.sum(-1), atol=2e-6, rtol=2e-6,
        )
        operator(torch.randn(2, 4, 33)).square().mean().backward()
        self.assertIsNotNone(operator.P.grad)
        self.assertGreater(operator.P.grad.abs().sum().item(), 0.0)

    def test_fadc_has_no_cross_sample_leakage(self):
        torch.manual_seed(308)
        operator = build_recent_rf_operator("fadc", 8, MODULE_CONFIGS["fadc"])
        first = torch.randn(1, 8, 65)
        batch_a = torch.cat((first, torch.randn_like(first)), dim=0)
        batch_b = torch.cat((first, 10.0 * torch.randn_like(first)), dim=0)
        torch.testing.assert_close(
            operator(batch_a)[:1], operator(batch_b)[:1], atol=1e-6, rtol=1e-6,
        )
        torch.testing.assert_close(
            operator.frequency_selection(first), first, atol=2e-5, rtol=2e-5,
        )

    def test_all_stage_configs_and_ids_exist(self):
        root = Path(__file__).resolve().parents[1]
        names = {
            308: "model_config_stage308_stage4_fadc1d.yaml",
            309: "model_config_stage309_stage4_fdconv1d.yaml",
            310: "model_config_stage310_stage4_unireplk1d.yaml",
            311: "model_config_stage311_stage4_shiftwise1d.yaml",
            312: "model_config_stage312_stage4_moderntcn1d.yaml",
            313: "model_config_stage313_stage4_morf1d.yaml",
            314: "model_config_stage314_stage4_dcnv4_1d.yaml",
            315: "model_config_stage315_stage4_wtconv1d.yaml",
            316: "model_config_stage316_stage4_dcls1d.yaml",
        }
        supported = supported_stage_ids()
        for stage, filename in names.items():
            self.assertIn(stage, supported)
            self.assertTrue((root / "config" / filename).is_file())

    def test_full_stage4_wrapper_forward_backward(self):
        model = IQUMamba1DRecentRF(
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
            rf_module_type="fadc",
            rf_hidden_channels=8,
            rf_module_config=MODULE_CONFIGS["fadc"],
        )
        self.assertEqual(model.rf_apply_stages, (0, 1))
        self.assertEqual(set(model.stage_rf), {"0", "1"})
        x = torch.randn(2, 2, 64, requires_grad=True)
        output = model(x)
        self.assertEqual(output.shape, (2, 4, 64))
        output.mean().backward()
        self.assertIsNotNone(x.grad)
        operator_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.stage_rf["0"].operator.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(operator_grad, 0.0)

    def test_real_configs_build_through_factory(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        config_dir = Path(__file__).resolve().parents[1] / "config"
        config_names = [
            "model_config_stage308_stage4_fadc1d.yaml",
            "model_config_stage309_stage4_fdconv1d.yaml",
            "model_config_stage310_stage4_unireplk1d.yaml",
            "model_config_stage311_stage4_shiftwise1d.yaml",
            "model_config_stage312_stage4_moderntcn1d.yaml",
            "model_config_stage313_stage4_morf1d.yaml",
            "model_config_stage314_stage4_dcnv4_1d.yaml",
            "model_config_stage315_stage4_wtconv1d.yaml",
            "model_config_stage316_stage4_dcls1d.yaml",
        ]
        for config_name in config_names:
            with self.subTest(config=config_name):
                config = MambaConfig(str(config_dir / config_name))
                model = Create_Mamba_model(
                    config, logger=None, input_size_=64,
                    device_override=torch.device("cpu"),
                )
                output = model(torch.randn(1, 2, 64))
                self.assertEqual(output.shape, (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
