from __future__ import annotations

import importlib.util
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    252: ("model_config_bimamba_unified_physical_global_kv.yaml",
          "bimamba_unified_physical_global_kv"),
    253: ("model_config_bimamba_physical_film_global_memory.yaml",
          "bimamba_physical_film_global_memory"),
    254: ("model_config_bimamba_scale_isolated_physical_fusion.yaml",
          "bimamba_scale_isolated_physical_fusion"),
    255: ("model_config_bimamba_identity_aware_physical_moe.yaml",
          "bimamba_identity_aware_physical_moe"),
    256: ("model_config_bimamba_cross_gated_dual_memory.yaml",
          "bimamba_cross_gated_dual_memory"),
}


class Stage252256RegistrationTests(unittest.TestCase):
    def test_variants_are_registered_with_comparable_strengths(self):
        from util.config import MambaConfig
        from util.stage_registry import supported_stage_ids

        main = (ROOT / "main.py").read_text(encoding="utf-8")
        posthoc = (ROOT / "util" / "posthoc_sweep.py").read_text(encoding="utf-8")
        utils = (ROOT / "util" / "utils.py").read_text(encoding="utf-8")
        for stage, (filename, model_type) in STAGES.items():
            mapping = f'{stage}: CONFIG_ROOT / "{filename}"'
            self.assertIn(stage, supported_stage_ids())
            self.assertIn(mapping, main)
            self.assertIn(mapping, posthoc)
            self.assertIn(f'"{model_type}"', utils)
            config = MambaConfig(str(ROOT / "config" / filename))
            config._load_enc_config()
            self.assertEqual(config.model_type, model_type)
            self.assertEqual(config.fusion_channel_scale_init, 0.1)
            self.assertEqual(config.fusion_channel_scale_max, 0.5)
            self.assertEqual(config.fusion_bottleneck_scale_init, 0.1)

    def test_component_checkpoint_cli_and_alias_loading_are_present(self):
        main = (ROOT / "main.py").read_text(encoding="utf-8")
        training = (ROOT / "util" / "training.py").read_text(encoding="utf-8")
        model = (ROOT / "models" / "IQUBiMamba1D_HierarchicalKVFusion.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("--component_checkpoints", main)
        self.assertIn("aliases_only=True", training)
        self.assertIn("checkpoint_prefix_aliases", model)


class Stage252256NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ModuleNotFoundError:
            raise unittest.SkipTest("torch is not installed")
        if importlib.util.find_spec("dynamic_network_architectures") is None:
            raise unittest.SkipTest("dynamic_network_architectures is not installed")
        if importlib.util.find_spec("mamba_ssm") is None:
            fake_mamba_ssm = types.ModuleType("mamba_ssm")
            fake_mamba_ssm.__spec__ = importlib.machinery.ModuleSpec(
                "mamba_ssm", loader=None
            )

            class FakeMamba(torch.nn.Module):
                def __init__(self, d_model, **_kwargs):
                    super().__init__()
                    self.proj = torch.nn.Linear(int(d_model), int(d_model))

                def forward(self, x):
                    return torch.cumsum(self.proj(x), dim=1)

            fake_mamba_ssm.Mamba = FakeMamba
            sys.modules["mamba_ssm"] = fake_mamba_ssm

        cls.torch = torch
        from models.IQUBiMamba1D_HierarchicalKVFusion import (
            IQUBiMamba1D_CrossGatedDualMemory,
            IQUBiMamba1D_IdentityAwarePhysicalMoE,
            IQUBiMamba1D_PhysicalFiLMGlobalMemory,
            IQUBiMamba1D_ScaleIsolatedPhysicalFusion,
            IQUBiMamba1D_UnifiedPhysicalGlobalKV,
        )
        cls.model_classes = (
            IQUBiMamba1D_UnifiedPhysicalGlobalKV,
            IQUBiMamba1D_PhysicalFiLMGlobalMemory,
            IQUBiMamba1D_ScaleIsolatedPhysicalFusion,
            IQUBiMamba1D_IdentityAwarePhysicalMoE,
            IQUBiMamba1D_CrossGatedDualMemory,
        )

    def _common(self):
        return {
            "input_size": 128, "input_channels": 2, "n_stages": 4,
            "features_per_stage": [8, 16, 32, 64], "conv_op": self.torch.nn.Conv1d,
            "kernel_sizes": [3, 3, 3, 3], "strides": [1, 2, 2, 2],
            "n_conv_per_stage": [1, 1, 1, 1], "num_classes": 4,
            "n_conv_per_stage_decoder": [1, 1, 1, 1],
            "fusion_global_kv_tokens": 8, "fusion_num_heads": 4,
            "physical_cyclic_lags": [0, 1], "physical_polyphase_branches": 4,
            "physical_symbol_orders": [2, 4],
        }

    def _models(self):
        return [model_class(**self._common()) for model_class in self.model_classes]

    def test_stable_initialization_contracts(self):
        unified, film, isolated, moe, cross_gated = self._models()
        tokens = self.torch.randn(2, film.physical_token_extractor.token_count, 8)
        feature = self.torch.randn(2, film.global_channels, 16)
        self.assertTrue(self.torch.equal(film.physical_film(feature, tokens), feature))
        isolated_feature = self.torch.randn(2, 16, 32)
        self.assertTrue(self.torch.equal(
            isolated.physical_skip_film(isolated_feature, tokens), isolated_feature
        ))
        query = self.torch.randn(2, moe.query_channels, 32)
        global_feature = self.torch.randn(2, moe.global_channels, 16)
        global_delta = self.torch.randn_like(query)
        physical_delta = self.torch.randn_like(query)
        condition = self.torch.zeros(2, 16)
        route, uncertainty, agreement = moe.expert_router(
            tokens, query, global_feature, global_delta, physical_delta, condition
        )
        self.assertTrue(self.torch.allclose(route.sum(dim=1), self.torch.ones(2)))
        self.assertTrue(self.torch.all(route[:, 0] > route[:, 1]))
        self.assertTrue(self.torch.all(route[:, 2:] < 0.1))
        self.assertTrue(self.torch.allclose(uncertainty, self.torch.full_like(uncertainty, 0.5)))
        self.assertEqual(tuple(agreement.shape), (2, 4))
        self.assertTrue(self.torch.allclose(
            cross_gated.physical_to_global_gate(tokens), self.torch.ones(2, 32)
        ))
        for scale in (
            unified.channel_scale.values(), film.channel_scale.values(),
            isolated.channel_scale.values(), moe.channel_scale.values(),
            cross_gated.global_scale.values(), cross_gated.physical_scale.values(),
        ):
            self.assertTrue(self.torch.allclose(scale, self.torch.full_like(scale, 0.1)))
            self.assertTrue(self.torch.all((scale > 0.0) & (scale < 0.5)))
        for model in (unified, film, isolated, moe, cross_gated):
            parameter_names = {name for name, _ in model.named_parameters()}
            self.assertTrue(model.no_weight_decay() <= parameter_names)

    def test_all_variants_forward_and_backward(self):
        for model in self._models():
            x = self.torch.randn(1, 2, 128, requires_grad=True)
            output = model(x)
            self.assertEqual(output.shape, (1, 4, 128))
            self.assertTrue(self.torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(self.torch.isfinite(x.grad).all())

    def test_factory_builds_all_variants(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        for (_, (filename, _)), model_class in zip(STAGES.items(), self.model_classes):
            config = MambaConfig(str(ROOT / "config" / filename))
            model = Create_Mamba_model(
                config, logger=None, input_size_=128,
                device_override=self.torch.device("cpu"),
            )
            self.assertIsInstance(model, model_class)


if __name__ == "__main__":
    unittest.main()
