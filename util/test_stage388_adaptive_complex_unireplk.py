import importlib.machinery
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

if "mamba_ssm" not in sys.modules and importlib.util.find_spec("mamba_ssm") is None:
    stub = types.ModuleType("mamba_ssm")
    stub.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None)
    class _MambaStub(torch.nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__(); self.projection = torch.nn.Linear(int(d_model), int(d_model))
        def forward(self, x): return self.projection(x)
    stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = stub

from models.IQUResUNet1D_UniRepLKBackbone import AdaptiveComplexUniRepLK
from util.config import MambaConfig
from util.utils import Create_Mamba_model


class Stage388Tests(unittest.TestCase):
    def test_forward_and_phase_equivariance(self):
        block = AdaptiveComplexUniRepLK(4).eval()
        x = torch.randn(2, 8, 32)
        phase = 0.71
        c, s = torch.cos(torch.tensor(phase)), torch.sin(torch.tensor(phase))
        xr, xi = x[:, :4], x[:, 4:]
        rotated = torch.cat((c * xr - s * xi, s * xr + c * xi), dim=1)
        with torch.no_grad():
            y = block(x)
            yr = block(rotated)
        expected = torch.cat((c * y[:, :4] - s * y[:, 4:], s * y[:, :4] + c * y[:, 4:]), dim=1)
        self.assertTrue(torch.allclose(yr, expected, atol=2e-4, rtol=2e-4))

    def test_stage_forward(self):
        root = Path(__file__).resolve().parents[1]
        config = MambaConfig(str(root / "config" / "model_config_stage388_adaptive_complex_unireplk.yaml"))
        config._load_enc_config()
        model = Create_Mamba_model(config, logger=None, input_size_=64, device_override=torch.device("cpu"))
        self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
