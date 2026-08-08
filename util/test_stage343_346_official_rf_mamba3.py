"""Tests for Stage 343-346 official fused RF-aware Mamba-3 variants."""

from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

import util.test_stage317_322_fdconv_unirep_ablation  # noqa: F401

import models.IQUMamba1D_OfficialRFMamba3 as rf_module
from models.IQUMamba1D_EstimatedCycloFRESH import (
    EstimatedCycloFRESHAdapter1D,
    estimate_cyclic_frequency_with_confidence,
)
from models.IQUMamba1D_OfficialRFMamba3 import (
    IQUMamba1DOfficialRFMamba3,
    OfficialRFMamba3Core,
    OfficialRFMamba3Layer,
)
from util.config import MambaConfig
from util.stage_registry import supported_stage_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    343: "model_config_stage343_official_mamba3_cyclic_anchor.yaml",
    344: "model_config_stage344_official_mamba3_reliability.yaml",
    345: "model_config_stage345_official_mamba3_cyclic_reliability.yaml",
    346: "model_config_stage346_official_mamba3_shared_cyclo_conditioning.yaml",
}


class _FakeOfficialMamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        is_outproj_norm=False,
        is_mimo=False,
        chunk_size=64,
        **_kwargs,
    ):
        super().__init__()
        assert not is_mimo
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.expand = int(expand)
        self.d_inner = self.d_model * self.expand
        self.headdim = int(headdim)
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = int(ngroups)
        self.num_rope_angles = max(1, int(self.d_state * rope_fraction) // 2)
        self.A_floor = 1e-4
        self.is_outproj_norm = bool(is_outproj_norm)
        self.chunk_size = int(chunk_size)
        projection_size = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.full((self.nheads,), -2.0))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.B_norm = nn.LayerNorm(self.d_state)
        self.C_norm = nn.LayerNorm(self.d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)


class _FakeOfficialModule(types.ModuleType):
    def __init__(self):
        super().__init__("mamba_ssm.modules.mamba3")
        self.last_kernel_kwargs = None

    @staticmethod
    def heavy_tail_activation(x):
        negative = x.clamp_max(0)
        positive = x.clamp_min(0)
        return positive + torch.reciprocal(1 - negative)

    def mamba3_siso_combined(self, **kwargs):
        self.last_kernel_kwargs = kwargs
        dt = kwargs["DT"].transpose(1, 2)[..., None]
        angle = kwargs["Angles"].mean(dim=-1, keepdim=True)
        trap = kwargs["Trap"].transpose(1, 2)[..., None]
        return kwargs["V"] + 0.01 * dt + 0.001 * angle + 0.001 * trap


def _small_common() -> dict:
    return dict(
        input_size=64,
        input_channels=2,
        n_stages=4,
        features_per_stage=[4, 8, 16, 32],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        n_conv_per_stage=[1, 1, 1, 1],
        num_classes=4,
        n_conv_per_stage_decoder=[1, 1, 1, 1],
        d_state=8,
        expand=1,
        headdim=4,
        chunk_size=8,
    )


class _OfficialPatchMixin:
    def setUp(self):
        self.official_module = _FakeOfficialModule()
        self.class_patch = mock.patch.object(
            rf_module, "_official_mamba3_class", return_value=_FakeOfficialMamba3
        )
        self.module_patch = mock.patch.dict(
            sys.modules,
            {"mamba_ssm.modules.mamba3": self.official_module},
        )
        self.class_patch.start()
        self.module_patch.start()

    def tearDown(self):
        self.module_patch.stop()
        self.class_patch.stop()


class SharedEstimatorTests(unittest.TestCase):
    def test_per_sample_frequency_and_confidence(self):
        length = 256
        steps = torch.arange(length, dtype=torch.float32)
        requested = torch.tensor([1 / 32, 1 / 16])
        envelopes = torch.stack([
            torch.sqrt(1.0 + 0.8 * torch.cos(2 * math.pi * f * steps))
            for f in requested
        ])
        mixtures = torch.stack((envelopes, torch.zeros_like(envelopes)), dim=1)
        frequency, confidence = estimate_cyclic_frequency_with_confidence(
            mixtures, min_freq=1 / 64, max_freq=1 / 8
        )
        torch.testing.assert_close(frequency, requested, atol=1e-6, rtol=0)
        self.assertTrue(torch.all(confidence > 0.5))

    def test_conditioned_fresh_uses_exact_shared_frequency(self):
        adapter = EstimatedCycloFRESHAdapter1D(2, zero_init=False)
        x = torch.randn(2, 2, 64)
        frequency = torch.tensor([1 / 32, 1 / 16])
        confidence = torch.tensor([0.8, 0.4])
        output = adapter.forward_conditioned(x, frequency, confidence)
        self.assertEqual(output.shape, x.shape)
        torch.testing.assert_close(adapter.last_frequency, frequency)
        torch.testing.assert_close(adapter.last_confidence, confidence)


class RFCoreTests(_OfficialPatchMixin, unittest.TestCase):
    def test_fixed_anchor_and_reliability_reach_official_kernel(self):
        core = OfficialRFMamba3Core(
            8,
            d_state=8,
            expand=1,
            headdim=4,
            token_stride=2,
            cyclic_anchor_enable=True,
            cyclic_frequencies=[0.0, 1 / 32, -1 / 32],
            reliability_enable=True,
        )
        tokens = torch.randn(2, 17, 8, requires_grad=True)
        output = core(tokens)
        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(
            core.last_scan_backend,
            "official_mamba3_siso_triton_rf_conditioned",
        )
        call = self.official_module.last_kernel_kwargs
        self.assertEqual(call["Angles"].shape[:3], (2, 17, 2))
        torch.testing.assert_close(call["ADT"], call["DT"] * (
            call["ADT"] / call["DT"].clamp_min(1e-8)
        ))
        self.assertIsNotNone(core.last_anchor_frequencies)
        self.assertIsNotNone(core.last_reliability)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertIsNotNone(core.reliability_net[-1].weight.grad)

    def test_zero_confidence_disables_shared_changes(self):
        core = OfficialRFMamba3Core(
            8,
            d_state=8,
            expand=1,
            headdim=4,
            cyclic_anchor_enable=True,
            dynamic_cyclic_enable=True,
            reliability_enable=True,
            shared_confidence_enable=True,
        )
        core(
            torch.randn(2, 11, 8),
            cyclic_frequency=torch.tensor([1 / 32, 1 / 16]),
            confidence=torch.zeros(2),
        )
        torch.testing.assert_close(
            core.last_reliability, torch.ones_like(core.last_reliability)
        )

    def test_real_state_keeps_trapezoid_but_zeros_all_rotation_angles(self):
        core = OfficialRFMamba3Core(
            8,
            d_state=8,
            expand=1,
            headdim=4,
            force_real_state=True,
            reliability_enable=True,
        )
        output = core(torch.randn(2, 13, 8))
        self.assertTrue(torch.isfinite(output).all())
        call = self.official_module.last_kernel_kwargs
        torch.testing.assert_close(
            call["Angles"], torch.zeros_like(call["Angles"])
        )
        self.assertEqual(tuple(call["Trap"].shape), (2, 2, 13))
        self.assertGreater(float(call["Trap"].abs().sum()), 0.0)
        self.assertIsNotNone(core.last_reliability)


class StageStructureTests(_OfficialPatchMixin, unittest.TestCase):
    def _build(self, **flags) -> IQUMamba1DOfficialRFMamba3:
        return IQUMamba1DOfficialRFMamba3(
            **_small_common(),
            cyclic_frequencies=[0.0, 1 / 32, -1 / 32],
            **flags,
        )

    def test_variants_replace_stage1_and_stage3_only(self):
        model = self._build(cyclic_anchor_enable=True)
        indexes = [
            index for index, layer in enumerate(model.encoder.mamba_layers)
            if isinstance(layer, OfficialRFMamba3Layer)
        ]
        self.assertEqual(indexes, [1, 3])
        self.assertTrue(all(
            not model.encoder.mamba_layers[index].channel_token
            for index in indexes
        ))
        self.assertEqual(
            [model.encoder.mamba_layers[index].ssm.token_stride for index in indexes],
            [2, 8],
        )

    def test_shared_frequency_reaches_fresh_and_both_memory_layers(self):
        model = self._build(
            cyclic_anchor_enable=True,
            dynamic_cyclic_enable=True,
            reliability_enable=True,
            shared_conditioning_enable=True,
        )
        x = torch.randn(2, 2, 64)
        output = model(x)
        self.assertEqual(output.shape, (2, 4, 64))
        self.assertIsInstance(model.input_adapter, EstimatedCycloFRESHAdapter1D)
        torch.testing.assert_close(
            model.input_adapter.last_frequency, model.last_cyclic_frequency
        )
        for layer in model.encoder.mamba_layers:
            if isinstance(layer, OfficialRFMamba3Layer):
                expected = model.last_cyclic_frequency[:, None].expand_as(
                    layer.ssm.last_anchor_frequencies
                )
                pattern = layer.ssm.last_anchor_frequencies
                torch.testing.assert_close(pattern[:, 1], expected[:, 1])


class RegistrationAndFactoryTests(_OfficialPatchMixin, unittest.TestCase):
    def test_configs_have_isolated_flags(self):
        expected = {
            343: (False, True, False, False, False),
            344: (False, False, False, True, False),
            345: (False, True, False, True, False),
            346: (False, True, True, True, True),
        }
        source = (PROJECT_ROOT / "main.py").read_text(encoding="utf-8")
        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                self.assertIn(stage, supported_stage_ids())
                self.assertIn(f'{stage}: CONFIG_ROOT / "{filename}"', source)
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                config._load_enc_config()
                cfg = config.model_config
                self.assertEqual(config.model_type, "iqumamba_official_rf_mamba3")
                actual = (
                    cfg.get("force_real_state", False),
                    cfg["cyclic_anchor_enable"],
                    cfg["dynamic_cyclic_enable"],
                    cfg["reliability_enable"],
                    cfg["shared_conditioning_enable"],
                )
                self.assertEqual(actual, expected[stage])

    def test_all_configs_build_and_forward_through_factory(self):
        from util.utils import Create_Mamba_model

        for stage, filename in CONFIGS.items():
            with self.subTest(stage=stage):
                config = MambaConfig(str(PROJECT_ROOT / "config" / filename))
                model = Create_Mamba_model(
                    config,
                    logger=None,
                    input_size_=64,
                    device_override=torch.device("cpu"),
                )
                output = model(torch.randn(1, 2, 64))
                self.assertEqual(output.shape, (1, 4, 64))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
