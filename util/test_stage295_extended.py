"""Extended verification for Stage 295 (complex-state selective SSM).

Complements util/test_stage295_complex_state_mamba.py with:

* analytic gradient checking of the parallel complex prefix scan (float64),
* gradient equivalence of the checkpointed vs direct scan path,
* long / non-power-of-two length agreement with the sequential recurrence,
* YAML -> MambaConfig -> constructor integration at full Stage-4 width,
* the real ``Create_Mamba_model`` factory path (skipped when heavy deps
  such as ``mamba_ssm`` are unavailable, e.g. CPU-only containers),
* parameter parity against the Stage-4 baseline (same skip rule),
* an oscillation-matters check (zeroing ``theta`` must change the output),
* an autocast smoke test.

Run from the project root:

    python -m pytest util/test_stage295_extended.py -v
"""

import math
import unittest
from pathlib import Path

import torch

from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    ComplexStateSelectiveSSM,
    IQUMamba1DComplexStateMamba,
    complex_prefix_scan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE295_YAML = (
    PROJECT_ROOT / "config" / "model_config_stage295_stage4_complex_state_mamba.yaml"
)

try:  # heavy: pulls in every model family incl. mamba_ssm-backed ones
    from util.utils import Create_Mamba_model

    HAVE_FULL_UTILS = True
except Exception:  # pragma: no cover - CPU-only or slim environments
    HAVE_FULL_UTILS = False

try:
    from mamba_ssm import Mamba  # noqa: F401 - presence check only

    HAVE_MAMBA_SSM = True
except Exception:  # pragma: no cover
    HAVE_MAMBA_SSM = False


def sequential_reference(a_real, a_imag, u_real):
    h_real = torch.zeros_like(u_real[:, 0])
    h_imag = torch.zeros_like(u_real[:, 0])
    outs_real, outs_imag = [], []
    for t in range(u_real.shape[1]):
        new_real = a_real[:, t] * h_real - a_imag[:, t] * h_imag + u_real[:, t]
        new_imag = a_real[:, t] * h_imag + a_imag[:, t] * h_real
        h_real, h_imag = new_real, new_imag
        outs_real.append(h_real)
        outs_imag.append(h_imag)
    return torch.stack(outs_real, dim=1), torch.stack(outs_imag, dim=1)


def stage295_kwargs_from_yaml():
    """Mirror utils._create_iqumamba_stage4_complex_state_model's mapping."""

    from util.config import MambaConfig

    config = MambaConfig(str(STAGE295_YAML), train=True)
    config._load_enc_config()
    model_cfg = config.model_config
    if config.model_type != "iqumamba_stage4_complex_state":
        raise AssertionError(f"unexpected model_type {config.model_type!r}")
    return dict(
        input_channels=config.input_channels,
        n_stages=config.n_stages,
        features_per_stage=config.features_per_stage,
        kernel_sizes=config.kernel_sizes,
        strides=config.strides,
        n_conv_per_stage=config.n_conv_per_stage,
        num_classes=config.num_classes,
        n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
        mamba_d_state=int(model_cfg.get("mamba_d_state", 8)),
        mamba_d_conv=int(model_cfg.get("mamba_d_conv", 4)),
        mamba_expand=int(model_cfg.get("mamba_expand", 2)),
        scan_checkpoint=bool(model_cfg.get("scan_checkpoint", True)),
        scan_backend=str(model_cfg.get("scan_backend", "auto")),
    )


class ScanGradientTests(unittest.TestCase):
    def test_gradcheck_float64(self):
        torch.manual_seed(2950)
        shape = (1, 5, 2, 3)
        magnitude = 0.9 * torch.rand(shape, dtype=torch.float64)
        angle = (2 * torch.rand(shape, dtype=torch.float64) - 1) * math.pi
        a_real = (magnitude * torch.cos(angle)).requires_grad_(True)
        a_imag = (magnitude * torch.sin(angle)).requires_grad_(True)
        u_real = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                complex_prefix_scan,
                (a_real, a_imag, u_real),
                eps=1e-6,
                atol=1e-8,
            )
        )

    def test_checkpointed_gradients_match_direct(self):
        torch.manual_seed(2951)
        ssm = ComplexStateSelectiveSSM(8, d_state=4, d_conv=2, expand=1)
        x = torch.randn(2, 33, 8)

        def run(checkpointed):
            ssm.scan_checkpoint = checkpointed
            ssm.zero_grad(set_to_none=True)
            x_local = x.clone().requires_grad_(True)
            ssm(x_local).square().mean().backward()
            grads = {
                name: parameter.grad.detach().clone()
                for name, parameter in ssm.named_parameters()
                if parameter.grad is not None
            }
            return x_local.grad.detach().clone(), grads

        x_grad_ckpt, param_grads_ckpt = run(True)
        x_grad_direct, param_grads_direct = run(False)
        torch.testing.assert_close(x_grad_ckpt, x_grad_direct)
        self.assertEqual(param_grads_ckpt.keys(), param_grads_direct.keys())
        for name in param_grads_ckpt:
            torch.testing.assert_close(
                param_grads_ckpt[name],
                param_grads_direct[name],
                msg=f"gradient mismatch for {name}",
            )

    def test_long_nonpow2_length_matches_sequential(self):
        torch.manual_seed(2952)
        shape = (1, 1000, 2, 4)
        magnitude = 0.99 * torch.rand(shape)
        angle = (2 * torch.rand(shape) - 1) * math.pi
        a_real = magnitude * torch.cos(angle)
        a_imag = magnitude * torch.sin(angle)
        u_real = torch.randn(shape)
        h_real, h_imag = complex_prefix_scan(a_real, a_imag, u_real)
        ref_real, ref_imag = sequential_reference(a_real, a_imag, u_real)
        torch.testing.assert_close(h_real, ref_real, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(h_imag, ref_imag, rtol=1e-4, atol=1e-4)


class OscillationBehaviourTests(unittest.TestCase):
    def test_zeroing_theta_changes_output(self):
        """The imaginary (rotation) part must actually shape the output."""

        torch.manual_seed(2953)
        ssm = ComplexStateSelectiveSSM(8, d_state=4, d_conv=2, expand=1)
        x = torch.randn(2, 64, 8)
        with torch.no_grad():
            baseline = ssm(x)
            ssm.theta.zero_()
            no_rotation = ssm(x)
        self.assertFalse(
            torch.allclose(baseline, no_rotation, rtol=1e-3, atol=1e-4),
            "output is insensitive to theta; complex state is inert",
        )

    def test_autocast_smoke(self):
        torch.manual_seed(2954)
        layer = ComplexStateMambaLayer(8, d_state=4, d_conv=2, expand=1)
        x = torch.randn(2, 8, 40)
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        layer = layer.to(device_type)
        x = x.to(device_type)
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            out = layer(x.to(dtype))
        self.assertTrue(torch.isfinite(out.float()).all())
        self.assertEqual(tuple(out.shape), (2, 8, 40))


class YamlIntegrationTests(unittest.TestCase):
    def test_yaml_reaches_constructor_at_full_width(self):
        kwargs = stage295_kwargs_from_yaml()
        self.assertEqual(kwargs["features_per_stage"], [32, 64, 128, 256])
        self.assertEqual(kwargs["mamba_d_state"], 8)
        self.assertTrue(kwargs["scan_checkpoint"])
        self.assertEqual(kwargs["scan_backend"], "auto")

        torch.manual_seed(2955)
        model = IQUMamba1DComplexStateMamba(input_size=2048, **kwargs)

        replaced = [
            index
            for index, layer in enumerate(model.backbone.encoder.mamba_layers)
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertEqual(replaced, [1, 3])

        x = torch.randn(1, 2, 2048)
        output = model(x)
        self.assertEqual(tuple(output.shape), (1, 4, 2048))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        gradless = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (parameter.grad is None or not torch.isfinite(parameter.grad).all())
        ]
        # Decoder deep-supervision heads for unused resolutions may stay
        # gradless; the complex-state parameters themselves must all train.
        self.assertFalse(
            [name for name in gradless if ".ssm." in name],
            f"complex-state parameters without finite gradients: {gradless}",
        )

    @unittest.skipUnless(
        HAVE_FULL_UTILS, "util.utils (and its model zoo deps) not importable here"
    )
    def test_real_factory_builds_stage295(self):
        from util.config import MambaConfig

        config = MambaConfig(str(STAGE295_YAML), train=True)
        model = Create_Mamba_model(
            config,
            logger=None,
            input_size_=1024,
            device_override=torch.device("cpu"),
        )
        self.assertIsInstance(model, IQUMamba1DComplexStateMamba)
        out = model(torch.randn(1, 2, 1024))
        self.assertEqual(tuple(out.shape), (1, 4, 1024))

    @unittest.skipUnless(HAVE_MAMBA_SSM, "mamba_ssm unavailable for baseline")
    def test_parameter_parity_with_stage4_baseline(self):
        """d_state=8 complex vs baseline d_state=16 real: ~equal budget."""

        from torch import nn

        from models.IQUMamba1D import IQUMamba1D

        kwargs = stage295_kwargs_from_yaml()
        torch.manual_seed(2956)
        complex_model = IQUMamba1DComplexStateMamba(input_size=2048, **kwargs)
        baseline = IQUMamba1D(
            input_size=2048,
            input_channels=kwargs["input_channels"],
            n_stages=kwargs["n_stages"],
            features_per_stage=list(kwargs["features_per_stage"]),
            conv_op=nn.Conv1d,
            kernel_sizes=list(kwargs["kernel_sizes"]),
            strides=list(kwargs["strides"]),
            n_conv_per_stage=list(kwargs["n_conv_per_stage"]),
            num_classes=kwargs["num_classes"],
            n_conv_per_stage_decoder=list(kwargs["n_conv_per_stage_decoder"]),
            deep_supervision=False,
        )
        complex_total = sum(p.numel() for p in complex_model.parameters())
        base_total = sum(p.numel() for p in baseline.parameters())
        ratio = complex_total / base_total
        print(
            f"\nstage295 params: {complex_total:,} | stage4 baseline: "
            f"{base_total:,} | ratio {ratio:.4f}"
        )
        self.assertLess(abs(ratio - 1.0), 0.02)


if __name__ == "__main__":
    unittest.main()
