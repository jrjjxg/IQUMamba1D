"""Focused tests for Stage 335-339 memory and role-RF experiments."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import models.IQUMamba1D_MemoryRFStages as memory_rf_module
from models.IQUMamba1D_MemoryRFStages import (
    IQUMamba1DMamba2SSD,
    IQUMamba1DRoleRF,
    IQUMamba1DReliabilityS4D,
    IQUMamba1DS4D,
    IQUMamba1DS4DUniRepLK,
    Mamba2SSDLayer,
    RoleRFMambaLayer,
    RoleRFSelectiveSSM,
    ReliabilitySelectiveS4DLayer,
    S4DKernel,
    S4DLayer,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = {
    335: "model_config_stage335_stage4_mamba2_ssd.yaml",
    336: "model_config_stage336_stage4_s4d.yaml",
    337: "model_config_stage337_stage4_shared_multiscale_mamba.yaml",
    338: "model_config_stage338_stage4_fixed_role_rf_mamba.yaml",
    339: "model_config_stage339_stage4_routed_role_rf_mamba.yaml",
    348: "model_config_stage348_s4d_reliability.yaml",
}


class S4DReproductionTests(unittest.TestCase):
    def test_kernel_matches_official_vandermonde_formula(self):
        torch.manual_seed(335)
        module = S4DKernel(d_model=3, d_state=6)
        length = 11
        dt = module.log_dt.exp()
        coefficients = torch.view_as_complex(module.C)
        poles = -module.log_A_real.exp() + 1j * module.A_imag
        dt_poles = poles * dt[:, None]
        steps = torch.arange(length, dtype=dt.dtype)
        expected = 2 * torch.einsum(
            "hn,hnl->hl",
            coefficients * (dt_poles.exp() - 1.0) / poles,
            torch.exp(dt_poles[..., None] * steps),
        ).real
        torch.testing.assert_close(module(length), expected)

    def test_layer_preserves_shape_and_has_finite_gradients(self):
        layer = S4DLayer(8, d_state=8)
        x = torch.randn(2, 8, 31, requires_grad=True)
        output = layer(x)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertEqual(layer.last_scan_backend, "torch_fft_cpu")
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)

    def test_channel_token_adapter_preserves_stage4_layout(self):
        layer = S4DLayer(12, d_state=8, channel_token=True)
        x = torch.randn(2, 5, 3, 4)
        self.assertEqual(tuple(layer(x).shape), tuple(x.shape))


class ReliabilitySelectiveS4DTests(unittest.TestCase):
    def test_initial_reliability_controls_delta_and_preserves_shape(self):
        layer = ReliabilitySelectiveS4DLayer(
            8, d_state=8, reliability_init=0.995
        )
        x = torch.randn(2, 8, 31, requires_grad=True)
        output = layer(x)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        self.assertEqual(layer.last_scan_backend, "torch_reference")
        torch.testing.assert_close(
            layer.last_reliability,
            torch.full_like(layer.last_reliability, 0.995),
            atol=1e-6,
            rtol=0.0,
        )
        expected_delta = (
            layer.kernel.log_dt.exp().view(1, 1, -1)
            * layer.last_reliability / layer.reliability_init
        )
        torch.testing.assert_close(layer.last_delta, expected_delta)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(layer.reliability_net[-1].weight.grad)

    def test_reliability_changes_state_recurrence_not_only_output(self):
        layer = ReliabilitySelectiveS4DLayer(4, d_state=4)
        x = torch.randn(1, 4, 17)
        original = layer(x)
        original_delta = layer.last_delta.clone()
        with torch.no_grad():
            layer.reliability_net[-1].bias.fill_(-8.0)
        changed = layer(x)
        self.assertTrue((layer.last_delta < original_delta).all())
        self.assertFalse(torch.allclose(changed, original))

    def test_centered_gate_core_matches_stage336_zoh_kernel(self):
        torch.manual_seed(348)
        layer = ReliabilitySelectiveS4DLayer(3, d_state=6)
        u = torch.randn(2, 3, 19)
        base_dt = layer.kernel.log_dt.exp()
        poles = torch.complex(
            -layer.kernel.log_A_real.exp(), layer.kernel.A_imag
        )
        dt_poles = base_dt.unsqueeze(-1) * poles
        coefficients = (
            torch.view_as_complex(layer.kernel.C)
            * torch.expm1(dt_poles) / dt_poles
        )
        actual = layer._reference_scan(
            u,
            base_dt.view(1, -1, 1).expand(u.size(0), -1, u.size(-1)),
            poles,
            coefficients,
        )
        kernel = layer.kernel(u.size(-1))
        expected = torch.fft.irfft(
            torch.fft.rfft(u, n=2 * u.size(-1))
            * torch.fft.rfft(kernel, n=2 * u.size(-1)),
            n=2 * u.size(-1),
        )[..., :u.size(-1)]
        expected = expected + u * layer.D.view(1, -1, 1)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_cuda_policy_requires_official_fused_scan(self):
        with mock.patch.object(memory_rf_module, "selective_scan_fn", None), \
             mock.patch.object(
                 memory_rf_module,
                 "_SELECTIVE_SCAN_IMPORT_ERROR",
                 "test import failure",
             ):
            with self.assertRaisesRegex(RuntimeError, "refusing to fall back"):
                memory_rf_module._use_fused_selective_scan(
                    mock.Mock(is_cuda=True)
                )


class RoleRFTests(unittest.TestCase):
    def test_cuda_policy_never_silently_uses_reference_scan(self):
        with mock.patch.object(memory_rf_module, "selective_scan_fn", None), \
             mock.patch.object(
                 memory_rf_module,
                 "_SELECTIVE_SCAN_IMPORT_ERROR",
                 "test import failure",
             ):
            with self.assertRaisesRegex(RuntimeError, "refusing to fall back"):
                memory_rf_module._use_fused_selective_scan(
                    mock.Mock(is_cuda=True)
                )
            self.assertFalse(
                memory_rf_module._use_fused_selective_scan(
                    mock.Mock(is_cuda=False)
                )
            )

    def _run_variant(self, variant: str) -> RoleRFSelectiveSSM:
        module = RoleRFSelectiveSSM(
            8, d_state=4, expand=1, context_kernels=(3, 7, 15), variant=variant
        )
        x = torch.randn(2, 23, 8, requires_grad=True)
        output = module(x)
        self.assertEqual(tuple(output.shape), tuple(x.shape))
        output.square().mean().backward()
        self.assertEqual(module.last_scan_backend, "torch_reference")
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()), 0.0)
        return module

    def test_all_variants_preserve_shape_and_gradient(self):
        for variant in sorted(RoleRFSelectiveSSM.VARIANTS):
            with self.subTest(variant=variant):
                self._run_variant(variant)

    def test_stage337_uses_one_shared_router_for_all_roles(self):
        module = self._run_variant("shared")
        self.assertEqual(len(module.routers), 1)
        torch.testing.assert_close(
            module.last_role_weights["B"], module.last_role_weights["C"]
        )
        torch.testing.assert_close(
            module.last_role_weights["B"], module.last_role_weights["delta"]
        )

    def test_stage338_fixes_b_short_c_medium_delta_long(self):
        module = self._run_variant("fixed_role")
        self.assertEqual(len(module.routers), 0)
        self.assertEqual(int(module.last_role_weights["B"].argmax()), 0)
        self.assertEqual(int(module.last_role_weights["C"].argmax()), 1)
        self.assertEqual(int(module.last_role_weights["delta"].argmax()), 2)

    def test_stage339_has_distinct_trainable_role_routers(self):
        module = self._run_variant("routed_role")
        self.assertEqual(len(module.routers), 3)
        self.assertEqual(len({id(router.weight) for router in module.routers}), 3)
        self.assertTrue(all(router.weight.requires_grad for router in module.routers))
        self.assertTrue(all(router.weight.grad is not None for router in module.routers))
        self.assertTrue(all(torch.isfinite(router.weight.grad).all()
                            for router in module.routers))
        self.assertEqual(
            [
                int(module.last_role_weights[name].mean((0, 1)).argmax())
                for name in ("B", "C", "delta")
            ],
            [0, 1, 2],
        )

    def test_channel_token_adapter_preserves_stage4_layout(self):
        layer = RoleRFMambaLayer(
            12,
            channel_token=True,
            d_state=4,
            expand=1,
            context_kernels=(3, 7, 15),
            variant="fixed_role",
        )
        x = torch.randn(2, 5, 3, 4)
        self.assertEqual(tuple(layer(x).shape), tuple(x.shape))


class _FakeMambaLayer(nn.Module):
    def __init__(self, dim: int = 8, channel_token: bool = False):
        super().__init__()
        self.dim = dim
        self.channel_token = channel_token

    def forward(self, x):
        return x


class _FakeIQUMamba1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.mamba_layers = nn.ModuleList(
            [nn.Identity(), _FakeMambaLayer(), nn.Identity(), _FakeMambaLayer()]
        )

    def forward(self, x):
        for layer in self.encoder.mamba_layers:
            x = layer(x)
        return x


class _FakeMamba2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x):
        return x


class Stage4ReplacementTests(unittest.TestCase):
    def setUp(self):
        fake_backbone = types.ModuleType("models.IQUMamba1D")
        fake_backbone.IQUMamba1D = _FakeIQUMamba1D
        fake_backbone.MambaLayer = _FakeMambaLayer
        self.backbone_patch = mock.patch.dict(
            sys.modules, {"models.IQUMamba1D": fake_backbone}
        )
        self.backbone_patch.start()
        self.backbone_kwargs = dict(
            input_size=64,
            input_channels=2,
            n_stages=4,
            features_per_stage=[32, 64, 128, 256],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[2, 2, 2, 2],
            num_classes=4,
            n_conv_per_stage_decoder=[2, 2, 1, 1],
        )

    def tearDown(self):
        self.backbone_patch.stop()

    def test_s4d_and_role_rf_replace_exactly_stage4_sequence_layers(self):
        cases = (
            (IQUMamba1DS4D, S4DLayer, {}),
            (
                IQUMamba1DReliabilityS4D,
                ReliabilitySelectiveS4DLayer,
                {},
            ),
            (IQUMamba1DRoleRF, RoleRFMambaLayer, {"variant": "fixed_role"}),
        )
        for wrapper, layer_type, extra in cases:
            with self.subTest(wrapper=wrapper.__name__):
                model = wrapper(**self.backbone_kwargs, **extra)
                self.assertEqual(model.replaced_layers, 2)
                self.assertEqual(
                    sum(isinstance(layer, layer_type)
                        for layer in model.backbone.encoder.mamba_layers),
                    2,
                )
                output = model(torch.randn(2, 8, 19))
                self.assertEqual(tuple(output.shape), (2, 8, 19))

    def test_mamba2_wrapper_invokes_official_public_class(self):
        fake_package = types.ModuleType("mamba_ssm")
        fake_package.Mamba2 = _FakeMamba2
        with mock.patch.dict(sys.modules, {"mamba_ssm": fake_package}):
            model = IQUMamba1DMamba2SSD(**self.backbone_kwargs, d_state=64)
        self.assertEqual(model.replaced_layers, 2)
        layers = [
            layer for layer in model.backbone.encoder.mamba_layers
            if isinstance(layer, Mamba2SSDLayer)
        ]
        self.assertEqual(len(layers), 2)
        self.assertTrue(all(layer.mamba.kwargs["d_state"] == 64 for layer in layers))
        self.assertTrue(all(layer.mamba.kwargs["use_mem_eff_path"] for layer in layers))


class StageRegistrationTests(unittest.TestCase):
    def test_stages_and_configs_are_registered_independently(self):
        expected_types = {
            335: "iqumamba_stage4_mamba2_ssd",
            336: "iqumamba_stage4_s4d",
            337: "iqumamba_stage4_role_rf",
            338: "iqumamba_stage4_role_rf",
            339: "iqumamba_stage4_role_rf",
            348: "iqumamba_stage4_s4d_reliability",
        }
        expected_variants = {337: "shared", 338: "fixed_role", 339: "routed_role"}
        main_source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage, filename in CONFIG_NAMES.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(f"{stage}: CONFIG_ROOT / \"{filename}\"", main_source)
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                config._load_enc_config()
                self.assertEqual(config.model_type, expected_types[stage])
                if stage in expected_variants:
                    self.assertEqual(
                        config.model_config["role_rf_variant"],
                        expected_variants[stage],
                    )

    def test_factory_wires_all_model_types_and_hyperparameters(self):
        source = (PROJECT_ROOT / "util" / "utils.py").read_text(encoding="utf-8")
        for model_type in (
            "iqumamba_stage4_mamba2_ssd",
            "iqumamba_stage4_s4d",
            "iqumamba_stage4_s4d_unireplk",
            "iqumamba_stage4_s4d_reliability",
            "iqumamba_stage4_role_rf",
        ):
            self.assertIn(model_type, source)
        for parameter in (
            "memory_d_state",
            "memory_chunk_size",
            "memory_dt_min",
            "reliability_hidden",
            "role_rf_variant",
            "role_rf_context_kernels",
        ):
            self.assertIn(parameter, source)


if __name__ == "__main__":
    unittest.main()
