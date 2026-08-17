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
            super().__init__()
            self.projection = torch.nn.Linear(int(d_model), int(d_model))

        def forward(self, x):
            return self.projection(x)

    stub.Mamba = _MambaStub
    sys.modules["mamba_ssm"] = stub

from util.config import MambaConfig
from util.utils import Create_Mamba_model


class Stage386Tests(unittest.TestCase):
    def test_factory_and_forward(self):
        root = Path(__file__).resolve().parents[1]
        config = MambaConfig(str(root / "config" / "model_config_stage386_stage381_unireplk_backbone.yaml"))
        config._load_enc_config()
        model = Create_Mamba_model(config, logger=None, input_size_=64, device_override=torch.device("cpu"))
        self.assertEqual(model.__class__.__name__, "IQUResUNet1D_Stage386")
        self.assertEqual(set(model.stage_rf.keys()), set())
        self.assertEqual(len(model.encoder.stages[0]), 2)
        self.assertEqual(model.encoder.stages[0][0].__class__.__name__, "UniRepLKNetBlock1D")
        self.assertEqual(model.encoder.stages[0][1].__class__.__name__, "UniRepLKNetBlock1D")
        self.assertEqual(model.encoder.stages[1][0].__class__.__name__, "_TransitionConv")
        self.assertEqual(model.encoder.stages[1][1].__class__.__name__, "UniRepLKNetBlock1D")
        self.assertEqual(model.decoder.stages[0][0].__class__.__name__, "_TransitionConv")
        self.assertEqual(model.decoder.stages[0][1].__class__.__name__, "UniRepLKNetBlock1D")
        output = model(torch.randn(1, 2, 64))
        self.assertEqual(tuple(output.shape), (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
