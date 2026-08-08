"""Full-frame RF Challenge inference and official-format output generation."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .metrics import save_submission_artifacts
from .models import extract_single_soi_output
from .protocol import complex_to_iq, demodulate_soi


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def predict_soi(
    model: torch.nn.Module,
    mixtures: np.ndarray,
    device: torch.device,
    batch_size: int = 1,
    amp: bool = True,
) -> np.ndarray:
    """Infer full 40,960-point SOI estimates with no chunk boundary artifacts."""

    mixture_array = np.asarray(mixtures)
    if mixture_array.ndim != 2 or not np.iscomplexobj(mixture_array):
        raise ValueError("mixtures must be a complex (B, L) NumPy array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    input_iq = complex_to_iq(mixture_array)
    output = np.empty(mixture_array.shape, dtype=np.complex64)
    model.eval()
    for start in range(0, input_iq.shape[0], batch_size):
        stop = min(start + batch_size, input_iq.shape[0])
        batch = torch.from_numpy(input_iq[start:stop]).to(device, non_blocking=True)
        with _autocast_context(device, amp):
            estimated_iq = extract_single_soi_output(model(batch))
        estimated_iq = estimated_iq.detach().float().cpu().numpy()
        output[start:stop] = estimated_iq[:, 0] + 1j * estimated_iq[:, 1]
    return output


def infer_and_save_submission(
    model: torch.nn.Module,
    mixtures: np.ndarray,
    device: torch.device,
    output_dir: str | Path,
    method_id: str,
    testset_identifier: str,
    soi_type: str,
    interference_type: str,
    batch_size: int = 1,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Path]]:
    """Run separation, hard demodulation, and save the required pair of files."""

    estimated_soi = predict_soi(model, mixtures, device=device, batch_size=batch_size, amp=amp)
    estimated_bits, _ = demodulate_soi(soi_type, estimated_soi)
    paths = save_submission_artifacts(
        output_dir=output_dir,
        method_id=method_id,
        testset_identifier=testset_identifier,
        soi_type=soi_type,
        interference_type=interference_type,
        estimated_soi=estimated_soi,
        estimated_bits=estimated_bits,
    )
    return estimated_soi, estimated_bits, paths
