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

from models.IQUMamba1D_RecentRFModules import (
    FeatureResidualAdapter,
    ParallelFeatureDeltaAdapter,
    ParallelFeatureFullAdapter,
    UniRepLKNetBlock1D,
)
from models.IQUResUNet1D_UniRepLKBackbone import AdaptiveRealUniRepLK
from models.IQUMamba1D_Stage391Ablations import (
    IQUMamba1D_Stage391,
    IQUMamba1D_Stage392,
    IQUMamba1D_Stage393,
    IQUMamba1D_Stage394,
    IQUMamba1D_Stage395,
    IQUMamba1D_Stage396,
    IQUMamba1D_Stage397,
)
from models.IQUBiMamba1D_CoreUpgrades import IndependentComplexStateBiMambaLayer
from util.config import MambaConfig
from util.utils import Create_Mamba_model


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x


class Stage391To394Tests(unittest.TestCase):
    def _make(self, stage):
        root = Path(__file__).resolve().parents[1]
        names = {
            391: "model_config_stage391_stage310_no_stage1_unireplk.yaml",
            392: "model_config_stage392_stage391_parallel_delta.yaml",
            393: "model_config_stage393_stage391_adaptive_rf.yaml",
            394: "model_config_stage394_stage392_adaptive_rf.yaml",
            395: "model_config_stage395_delta_post.yaml",
            396: "model_config_stage396_full_pre.yaml",
            397: "model_config_stage397_stage394_complex_bimamba.yaml",
        }
        config = MambaConfig(str(root / "config" / names[stage]))
        return Create_Mamba_model(
            config, logger=None, input_size_=64, device_override=torch.device("cpu")
        )

    def test_controlled_structure(self):
        classes = {
            391: IQUMamba1D_Stage391,
            392: IQUMamba1D_Stage392,
            393: IQUMamba1D_Stage393,
            394: IQUMamba1D_Stage394,
            395: IQUMamba1D_Stage395,
            396: IQUMamba1D_Stage396,
            397: IQUMamba1D_Stage397,
        }
        for stage in range(391, 398):
            model = self._make(stage)
            self.assertIsInstance(model, classes[stage])
            self.assertEqual(set(model.stage_rf), {"0", "2"})
            adapter = model.stage_rf["0"]
            self.assertIsInstance(
                adapter,
                (ParallelFeatureDeltaAdapter if stage in (392, 394, 395, 397)
                 else ParallelFeatureFullAdapter if stage == 396
                 else FeatureResidualAdapter),
            )
            self.assertIsInstance(
                adapter.operator,
                AdaptiveRealUniRepLK if stage in (393, 394)
                else UniRepLKNetBlock1D,
            )
            if stage in (393, 394):
                self.assertEqual(
                    tuple(expert.dwconv.large_kernel for expert in adapter.operator.experts),
                    (9, 17),
                )
            self.assertEqual(tuple(model(torch.randn(1, 2, 64)).shape), (1, 4, 64))
            if stage == 397:
                self.assertIsInstance(model.encoder.mamba_layers[1], IndependentComplexStateBiMambaLayer)
                self.assertIsInstance(model.encoder.mamba_layers[3], IndependentComplexStateBiMambaLayer)

    def test_connection_inputs_are_semantically_distinct(self):
        source = torch.randn(1, 2, 64)
        for stage, expects_pre_mamba in (
            (391, False), (392, True), (393, False), (394, True),
            (395, False), (396, True),
            (397, True),
        ):
            model = self._make(stage).eval()
            recorder = _Recorder()
            if stage in (392, 394, 395, 396):
                model.stage_rf["0"].operator = recorder
            else:
                model.stage_rf["0"] = recorder
            with torch.no_grad():
                stem = model.encoder.stem(source)
                conv = model.encoder.stages[0](stem)
                mamba = model.encoder.mamba_layers[0](conv)
                model(source)
            expected = conv if expects_pre_mamba else mamba
            self.assertTrue(torch.allclose(recorder.seen, expected))


if __name__ == "__main__":
    unittest.main()
