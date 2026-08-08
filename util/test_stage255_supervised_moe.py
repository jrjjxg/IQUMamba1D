from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Stage255SupervisedMoETests(unittest.TestCase):
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
            fake_mamba_ssm.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)

            class FakeMamba(torch.nn.Module):
                def __init__(self, d_model, **_kwargs):
                    super().__init__()
                    self.proj = torch.nn.Linear(int(d_model), int(d_model))

                def forward(self, x):
                    return self.proj(x)

            fake_mamba_ssm.Mamba = FakeMamba
            sys.modules["mamba_ssm"] = fake_mamba_ssm
        cls.torch = torch

    def _model(self, **overrides):
        from models.IQUBiMamba1D_HierarchicalKVFusion import (
            IQUBiMamba1D_IdentityAwarePhysicalMoE,
        )

        parameters = dict(
            input_size=64,
            input_channels=2,
            n_stages=4,
            features_per_stage=[8, 16, 32, 64],
            conv_op=self.torch.nn.Conv1d,
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=4,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
            fusion_global_kv_tokens=8,
            fusion_num_heads=4,
            physical_cyclic_lags=[0, 1],
            physical_polyphase_branches=4,
            physical_symbol_orders=[2, 4],
            fusion_trust_penalty_enable=True,
            fusion_condition_routing_enable=True,
            fusion_counterfactual_enable=True,
            fusion_return_route_aux=True,
            fusion_route_candidate_probability=1.0,
        )
        parameters.update(overrides)
        return IQUBiMamba1D_IdentityAwarePhysicalMoE(**parameters)

    def test_training_returns_four_counterfactual_paths_and_condition_aux(self):
        model = self._model().train()
        separation, auxiliary = model(self.torch.randn(2, 2, 64))
        self.assertEqual(tuple(separation.shape), (2, 4, 64))
        self.assertEqual(len(auxiliary["candidate_outputs"]), 4)
        self.assertTrue(all(tuple(item.shape) == (2, 4, 64) for item in auxiliary["candidate_outputs"]))
        self.assertEqual(tuple(auxiliary["route_weights"].shape), (2, 4))
        self.assertEqual(tuple(auxiliary["snr_prediction"].shape), (2,))
        self.assertFalse(any(item.requires_grad for item in auxiliary["candidate_outputs"]))

    def test_disabling_counterfactual_and_condition_avoids_auxiliary_decoder_work(self):
        model = self._model(
            fusion_counterfactual_enable=False,
            fusion_condition_routing_enable=False,
            fusion_return_route_aux=False,
        ).train()
        decoder_calls = []
        handle = model.decoder.register_forward_hook(lambda *_args: decoder_calls.append(1))
        try:
            output = model(self.torch.randn(2, 2, 64))
        finally:
            handle.remove()
        self.assertTrue(self.torch.is_tensor(output))
        self.assertEqual(len(decoder_calls), 1)

    def test_trust_penalty_is_independently_switchable(self):
        model = self._model(
            fusion_counterfactual_enable=False,
            fusion_condition_routing_enable=False,
            fusion_return_route_aux=True,
        ).train()
        inputs = self.torch.randn(2, 2, 64)
        model.trust_penalty_enable = False
        _, without_trust = model(inputs)
        model.trust_penalty_enable = True
        _, with_trust = model(inputs)
        self.assertTrue(
            self.torch.all(
                with_trust["route_weights"][:, 2:].sum(dim=1)
                < without_trust["route_weights"][:, 2:].sum(dim=1)
            )
        )

    def test_counterfactual_quality_target_backpropagates_only_to_router(self):
        from util.evidence_moe_loss import counterfactual_route_loss

        target = self.torch.randn(2, 4, 32)
        candidates = [target + scale * self.torch.randn_like(target) for scale in (0.0, 0.2, 0.5, 1.0)]
        logits = self.torch.zeros(2, 4, requires_grad=True)
        route = logits.softmax(dim=-1)
        loss, candidate_losses, target_route = counterfactual_route_loss(
            candidates,
            target,
            route,
            quality_loss="pit_si_snr_huber",
            temperature=0.25,
        )
        loss.backward()
        self.assertTrue(self.torch.isfinite(loss))
        self.assertTrue(self.torch.isfinite(logits.grad).all())
        self.assertTrue(self.torch.equal(target_route.argmax(dim=1), candidate_losses.argmin(dim=1)))
        self.assertTrue(self.torch.equal(target_route.argmax(dim=1), self.torch.zeros(2, dtype=self.torch.long)))

    def test_configuration_enables_all_four_requested_training_features(self):
        from util.config import MambaConfig

        config = (ROOT / "config" / "model_config_bimamba_identity_aware_physical_moe.yaml").read_text(
            encoding="utf-8"
        )
        training = (ROOT / "util" / "training.py").read_text(encoding="utf-8")
        for field in (
            "evidence_moe_route_supervision_enable: true",
            "fusion_trust_penalty_init",
            "stage255_router_warmup_epochs",
            "stage255_snr_aux_weight",
            "stage255_snr_curriculum_enable: true",
        ):
            self.assertIn(field, config)
        self.assertIn("_set_stage255_router_phase", training)
        self.assertIn("_apply_stage255_snr_curriculum", training)
        self.assertIn("_compute_stage255_snr_aux_loss", training)
        loaded = MambaConfig(str(ROOT / "config" / "model_config_bimamba_identity_aware_physical_moe.yaml"))
        loaded._load_enc_config()
        self.assertFalse(loaded.fusion_return_route_aux)
        self.assertEqual(loaded.fusion_route_candidate_probability, 0.25)
        self.assertEqual(loaded.stage255_expert_pretrain_epochs, 5)
        self.assertEqual(loaded.stage255_router_warmup_epochs, 5)
        self.assertTrue(loaded.stage255_snr_curriculum_enable)
        self.assertTrue(loaded.fusion_counterfactual_enable)
        self.assertTrue(loaded.fusion_trust_penalty_enable)
        self.assertTrue(loaded.fusion_condition_routing_enable)

    def test_independent_cli_switches_are_registered(self):
        main_text = (ROOT / "main.py").read_text(encoding="utf-8")
        for flag in (
            "--stage255_counterfactual_disable",
            "--stage255_trust_disable",
            "--stage255_condition_disable",
            "--evidence_moe_route_supervision_disable",
            "--stage255_snr_curriculum_disable",
            "--stage255_route_candidate_probability",
            "--stage255_expert_pretrain_epochs",
        ):
            self.assertIn(flag, main_text)


if __name__ == "__main__":
    unittest.main()
