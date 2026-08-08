import math
import unittest

import torch

import models.IQUMamba1D_ComplexStateMamba as complex_state_module
from models.IQUMamba1D_ComplexStateMamba import (
    ComplexStateMambaLayer,
    ComplexStateSelectiveSSM,
    IQUMamba1DComplexStateMamba,
    _mamba_selective_scan_fn,
    _pack_complex_sequence,
    complex_prefix_scan,
)
from util.stage_registry import supported_stage_ids


def sequential_reference(a_real, a_imag, u_real):
    """Naive O(L) loop implementing h_t = a_t * h_{t-1} + u_t."""

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


class ComplexPrefixScanTests(unittest.TestCase):
    def test_cuda_complex_sequence_packing_round_trip(self):
        real = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        imag = -real - 0.5
        packed = _pack_complex_sequence(real, imag)
        self.assertEqual(tuple(packed.shape), (2, 3, 8))
        unpacked = torch.view_as_complex(
            packed.reshape(2, 3, 4, 2)
        ).permute(0, 2, 1)
        torch.testing.assert_close(unpacked.real, real)
        torch.testing.assert_close(unpacked.imag, imag)

    def test_parallel_scan_matches_sequential_recurrence(self):
        torch.manual_seed(295)
        for length in (1, 2, 7, 64, 100):
            magnitude = 0.97 * torch.rand(2, length, 3, 4)
            angle = (2 * torch.rand(2, length, 3, 4) - 1) * math.pi
            a_real = magnitude * torch.cos(angle)
            a_imag = magnitude * torch.sin(angle)
            u_real = torch.randn(2, length, 3, 4)
            h_real, h_imag = complex_prefix_scan(a_real, a_imag, u_real)
            ref_real, ref_imag = sequential_reference(a_real, a_imag, u_real)
            torch.testing.assert_close(h_real, ref_real, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(h_imag, ref_imag, rtol=1e-4, atol=1e-5)

    def test_impulse_response_is_decaying_rotation(self):
        decay, theta, length = 0.9, 0.35, 32
        a_real = torch.full((1, length, 1, 1), decay * math.cos(theta))
        a_imag = torch.full((1, length, 1, 1), decay * math.sin(theta))
        u_real = torch.zeros(1, length, 1, 1)
        u_real[:, 0] = 1.0
        h_real, h_imag = complex_prefix_scan(a_real, a_imag, u_real)
        steps = torch.arange(length, dtype=torch.float32)
        expected_real = decay**steps * torch.cos(theta * steps)
        expected_imag = decay**steps * torch.sin(theta * steps)
        torch.testing.assert_close(
            h_real.flatten(), expected_real, rtol=1e-4, atol=1e-5
        )
        torch.testing.assert_close(
            h_imag.flatten(), expected_imag, rtol=1e-4, atol=1e-5
        )


class ComplexStateSSMTests(unittest.TestCase):
    def test_fused_argument_packing_matches_existing_recurrence(self):
        torch.manual_seed(2957)
        ssm = ComplexStateSelectiveSSM(
            6, d_state=3, d_conv=2, expand=1, scan_backend="torch"
        )
        batch, length, inner, state = 2, 11, ssm.d_inner, ssm.d_state
        x_conv = torch.randn(batch, length, inner)
        z = torch.randn_like(x_conv)
        dt = 0.01 + 0.1 * torch.rand_like(x_conv)
        b_in = torch.randn(batch, length, state)
        c_real = torch.randn(batch, length, state)
        c_imag = torch.randn(batch, length, state)

        decay = torch.exp(-dt.unsqueeze(-1) * torch.exp(ssm.a_log))
        angle = dt.unsqueeze(-1) * ssm.theta
        h_real, h_imag = complex_prefix_scan(
            decay * torch.cos(angle),
            decay * torch.sin(angle),
            (dt * x_conv).unsqueeze(-1) * b_in.unsqueeze(2),
        )
        expected = (
            (h_real * c_real.unsqueeze(2)).sum(-1)
            + (h_imag * c_imag.unsqueeze(2)).sum(-1)
            + ssm.D * x_conv
        ) * torch.nn.functional.silu(z)

        def fake_selective_scan(u, delta, A, B, C, D, z, **_kwargs):
            unpacked_B = torch.view_as_complex(
                B.reshape(batch, state, length, 2)
            ).permute(0, 2, 1)
            unpacked_C = torch.view_as_complex(
                C.reshape(batch, state, length, 2)
            ).permute(0, 2, 1)
            transition = torch.exp(
                delta.transpose(1, 2).unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
            )
            injection = (
                delta.transpose(1, 2) * u.transpose(1, 2)
            ).unsqueeze(-1) * unpacked_B.unsqueeze(2)
            h_r, h_i = complex_prefix_scan(
                transition.real, transition.imag, injection.real
            )
            h = torch.complex(h_r, h_i)
            y = 2.0 * (h * unpacked_C.unsqueeze(2)).sum(-1).real
            y = y + D * u.transpose(1, 2)
            y = y * torch.nn.functional.silu(z.transpose(1, 2))
            return y.transpose(1, 2)

        original = complex_state_module._mamba_selective_scan_fn
        complex_state_module._mamba_selective_scan_fn = fake_selective_scan
        try:
            actual = ssm._fused_scan(
                x_conv, z, dt, b_in, c_real, c_imag
            )
        finally:
            complex_state_module._mamba_selective_scan_fn = original
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)

    def test_ssm_shapes_backward_and_finiteness(self):
        torch.manual_seed(296)
        ssm = ComplexStateSelectiveSSM(16, d_state=4, d_conv=2, expand=2)
        x = torch.randn(2, 37, 16, requires_grad=True)
        y = ssm(x)
        self.assertEqual(tuple(y.shape), (2, 37, 16))
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        for parameter in ssm.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(ssm.last_scan_backend, "torch")

    def test_explicit_cuda_backend_rejects_cpu_tensor(self):
        ssm = ComplexStateSelectiveSSM(
            8,
            d_state=4,
            d_conv=2,
            expand=1,
            scan_backend="cuda",
        )
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA tensor"):
            ssm(torch.randn(1, 12, 8))

    @unittest.skipUnless(
        torch.cuda.is_available() and _mamba_selective_scan_fn is not None,
        "complex selective_scan CUDA extension unavailable",
    )
    def test_fused_cuda_matches_torch_backend(self):
        torch.manual_seed(2958)
        torch_model = ComplexStateSelectiveSSM(
            8, d_state=4, d_conv=2, expand=1,
            scan_checkpoint=False, scan_backend="torch",
        ).cuda()
        cuda_model = ComplexStateSelectiveSSM(
            8, d_state=4, d_conv=2, expand=1,
            scan_checkpoint=False, scan_backend="cuda",
        ).cuda()
        cuda_model.load_state_dict(torch_model.state_dict())
        x = torch.randn(2, 65, 8, device="cuda")
        expected = torch_model(x)
        actual = cuda_model(x)
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        actual.square().mean().backward()
        self.assertEqual(cuda_model.last_scan_backend, "mamba_cuda")
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in cuda_model.parameters()
            )
        )

    def test_checkpointed_scan_matches_direct_scan(self):
        torch.manual_seed(297)
        ssm = ComplexStateSelectiveSSM(8, d_state=4, d_conv=2, expand=1)
        x = torch.randn(2, 21, 8)
        ssm.scan_checkpoint = True
        y_checkpointed = ssm(x)
        ssm.scan_checkpoint = False
        y_direct = ssm(x)
        torch.testing.assert_close(y_checkpointed, y_direct)

    def test_layer_preserves_channel_first_layout(self):
        torch.manual_seed(298)
        layer = ComplexStateMambaLayer(12, d_state=4, d_conv=2, expand=1)
        x = torch.randn(2, 12, 45)
        out = layer(x)
        self.assertEqual(tuple(out.shape), (2, 12, 45))
        self.assertTrue(torch.isfinite(out).all())


class ComplexStateModelTests(unittest.TestCase):
    def test_stage295_shapes_backward_and_replacement(self):
        torch.manual_seed(299)
        from models.IQUMamba1D import MambaLayer

        model = IQUMamba1DComplexStateMamba(
            input_size=512,
            input_channels=2,
            n_stages=4,
            features_per_stage=[4, 8, 16, 32],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=4,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
            mamba_d_state=4,
            mamba_d_conv=2,
            mamba_expand=1,
        )
        remaining = [
            layer
            for layer in model.backbone.encoder.mamba_layers
            if isinstance(layer, MambaLayer)
        ]
        self.assertEqual(len(remaining), 0)
        replaced = [
            layer
            for layer in model.backbone.encoder.mamba_layers
            if isinstance(layer, ComplexStateMambaLayer)
        ]
        self.assertEqual(len(replaced), 2)

        x = torch.randn(2, 2, 512)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 4, 512))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_no_weight_decay_targets_state_parameters(self):
        model_names = IQUMamba1DComplexStateMamba(
            input_size=512,
            input_channels=2,
            n_stages=4,
            features_per_stage=[4, 8, 16, 32],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2],
            n_conv_per_stage=[1, 1, 1, 1],
            num_classes=4,
            n_conv_per_stage_decoder=[1, 1, 1, 1],
            mamba_d_state=4,
            mamba_d_conv=2,
            mamba_expand=1,
        ).no_weight_decay()
        self.assertTrue(any(name.endswith(".a_log") for name in model_names))
        self.assertTrue(any(name.endswith(".theta") for name in model_names))
        self.assertTrue(any(name.endswith(".D") for name in model_names))


class ComplexStateRegistrationTests(unittest.TestCase):
    def test_stage_295_is_registered(self):
        self.assertIn(295, supported_stage_ids())


if __name__ == "__main__":
    unittest.main()
