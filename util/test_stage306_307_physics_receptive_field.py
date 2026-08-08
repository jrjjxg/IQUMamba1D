import importlib.util
import math
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
                input_channels, output_channels, kernel,
                stride=stride_value, padding=kernel // 2, bias=conv_bias,
            )

        def forward(self, x):
            return self.conv(x)

    helper.maybe_convert_scalar_to_list = maybe_convert_scalar_to_list
    residual.BasicBlockD = BasicBlockD
    sys.modules[root.__name__] = root
    sys.modules[blocks.__name__] = blocks
    sys.modules[helper.__name__] = helper
    sys.modules[residual.__name__] = residual


from models.IQUMamba1D_PhysicsReceptiveField import (
    IQUMamba1DPhysicalCanonical,
    IQUMamba1DSymbolDelayDopplerRF,
    PhysicalSourceCanonicalizer,
    SymbolNormalizedDelayDopplerRF,
    carrier_phase_increment,
    phase_increment,
)
from models.IQUMamba1D import IQUMamba1D
from util.stage_registry import supported_stage_ids


def _tone(frequency, length=128, batch=1):
    phase = frequency * torch.arange(length, dtype=torch.float32)
    iq = torch.stack((torch.cos(phase), torch.sin(phase)), dim=0)
    return iq.unsqueeze(0).repeat(batch, 1, 1)


def _rotate(x, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.stack(
        (cosine * x[:, 0] - sine * x[:, 1],
         sine * x[:, 0] + cosine * x[:, 1]),
        dim=1,
    )


def _small_model(model_class, **extra):
    return model_class(
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
        **extra,
    )


class PhysicalCanonicalTests(unittest.TestCase):
    def test_phase_increment_recovers_tone_frequency(self):
        actual = phase_increment(_tone(0.07))
        torch.testing.assert_close(actual, torch.tensor([0.07]), atol=1e-5, rtol=1e-5)

    def test_power_transform_removes_random_8psk_symbols(self):
        torch.manual_seed(306)
        length = 4096
        omega = 0.012
        symbols = torch.randint(0, 8, (length,)) * (2.0 * math.pi / 8.0)
        phase = symbols + omega * torch.arange(length)
        source = torch.stack((torch.cos(phase), torch.sin(phase)), dim=0)[None]
        actual = carrier_phase_increment(source, symbol_orders=(2, 4, 8))
        torch.testing.assert_close(actual, torch.tensor([omega]), atol=2e-4, rtol=2e-4)

    def test_two_sources_are_sorted_by_ascending_phase_increment(self):
        high = _tone(0.09)
        low = _tone(-0.04)
        output = torch.cat((high, low), dim=1)
        canonicalizer = PhysicalSourceCanonicalizer()
        sorted_output = canonicalizer(output).view(1, 2, 2, -1)
        scores = canonicalizer.source_scores(sorted_output)
        self.assertLessEqual(float(scores[0, 0]), float(scores[0, 1]))
        torch.testing.assert_close(sorted_output[:, 0], low)
        torch.testing.assert_close(sorted_output[:, 1], high)

    def test_stage306_forward_backward(self):
        torch.manual_seed(306)
        model = _small_model(IQUMamba1DPhysicalCanonical)
        output = model(torch.randn(2, 2, 64))
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        output.square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_stage306_loads_stage4_state_dict_strictly(self):
        stage4 = _small_model(IQUMamba1D)
        stage306 = _small_model(IQUMamba1DPhysicalCanonical)
        stage306.load_state_dict(stage4.state_dict(), strict=True)


class SymbolDelayDopplerTests(unittest.TestCase):
    def test_operator_is_global_phase_equivariant(self):
        torch.manual_seed(307)
        module = SymbolNormalizedDelayDopplerRF(
            sps_candidates=(4, 8, 12), symbol_spans=(1, 2, 4),
            default_sps=8, residual_scale_init=0.1,
        )
        x = torch.randn(2, 2, 96)
        angle = 0.73
        torch.testing.assert_close(
            module(_rotate(x, angle)), _rotate(module(x), angle),
            atol=2e-5, rtol=2e-5,
        )

    def test_zero_residual_scale_is_exact_identity(self):
        module = SymbolNormalizedDelayDopplerRF(
            sps_candidates=(4, 8), symbol_spans=(1, 2),
            default_sps=8, residual_scale_init=0.0,
        )
        x = torch.randn(2, 2, 64)
        torch.testing.assert_close(module(x), x)

    def test_stage307_forward_backward_and_diagnostics(self):
        torch.manual_seed(307)
        model = _small_model(
            IQUMamba1DSymbolDelayDopplerRF,
            rf_sps_candidates=(4, 8, 12),
            rf_symbol_spans=(1, 2, 4),
            rf_default_sps=8,
            rf_gate_hidden=8,
        )
        output = model(torch.randn(2, 2, 64))
        self.assertEqual(tuple(output.shape), (2, 4, 64))
        self.assertEqual(tuple(model.symbol_delay_doppler_rf.last_sps.shape), (2,))
        self.assertTrue(model.no_weight_decay() <= dict(model.named_parameters()).keys())
        output.square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_stages_are_registered(self):
        self.assertIn(306, supported_stage_ids())
        self.assertIn(307, supported_stage_ids())

    def test_real_configs_build_through_factory(self):
        from util.config import MambaConfig
        from util.utils import Create_Mamba_model

        config_dir = Path(__file__).resolve().parents[1] / "config"
        for config_name in (
            "model_config_stage306_stage4_physical_canonical.yaml",
            "model_config_stage307_stage4_symbol_delay_doppler_rf.yaml",
        ):
            with self.subTest(config=config_name):
                config = MambaConfig(str(config_dir / config_name))
                model = Create_Mamba_model(
                    config, logger=None, input_size_=64,
                    device_override=torch.device("cpu"),
                )
                self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
