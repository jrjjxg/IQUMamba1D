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

from models.IQUResUNet1D_UniRepLKBackbone import (
    AdaptiveComplexUniRepLK, ComplexUniRepLKBlock, IQUResUNet1D_Stage388,
    IQUResUNet1D_Stage389, IQUResUNet1D_Stage390,
)
from util.config import MambaConfig
from util.utils import Create_Mamba_model


class Stage388AblationTests(unittest.TestCase):
    def test_models_preserve_stage381_shape_and_rf_slots(self):
        root = Path(__file__).resolve().parents[1]
        names = {
            388: "model_config_stage388_adaptive_complex_unireplk.yaml",
            389: "model_config_stage389_adaptive_real_unireplk.yaml",
            390: "model_config_stage390_fixed_complex_unireplk.yaml",
        }
        classes = {388: IQUResUNet1D_Stage388, 389: IQUResUNet1D_Stage389, 390: IQUResUNet1D_Stage390}
        for stage, name in names.items():
            config = MambaConfig(str(root / "config" / name)); config._load_enc_config()
            model = Create_Mamba_model(config, logger=None, input_size_=64, device_override=torch.device("cpu"))
            self.assertIsInstance(model, classes[stage])
            self.assertEqual(set(model.stage_rf.keys()), {"0", "2"})
            self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))

    def test_complex_block_phase_equivariance(self):
        block = ComplexUniRepLKBlock(4).eval()
        x = torch.randn(1, 8, 32); phase = torch.tensor(0.63)
        c, s = torch.cos(phase), torch.sin(phase)
        xr, xi = x[:, :4], x[:, 4:]
        rotated = torch.cat((c * xr - s * xi, s * xr + c * xi), dim=1)
        with torch.no_grad(): y, yr = block(x), block(rotated)
        expected = torch.cat((c * y[:, :4] - s * y[:, 4:], s * y[:, :4] + c * y[:, 4:]), dim=1)
        self.assertTrue(torch.allclose(yr, expected, atol=3e-4, rtol=3e-4))


if __name__ == "__main__":
    unittest.main()
