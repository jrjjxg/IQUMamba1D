"""Lazy JIT loader and autograd binding for the Stage 330/333 CUDA scan."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import torch


_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None
_LOAD_LOCK = threading.Lock()


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("native complex scan extension is unavailable") from _EXTENSION_ERROR
    with _LOAD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        try:
            from torch.utils.cpp_extension import load

            source_root = Path(__file__).resolve().parent / "csrc"
            _EXTENSION = load(
                name="iqumamba_complex_scan_cuda_v1",
                sources=[
                    str(source_root / "complex_scan.cpp"),
                    str(source_root / "complex_scan_cuda.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=os.environ.get("IQUMAMBA_CUDA_BUILD_VERBOSE", "0") == "1",
            )
        except Exception as exc:
            _EXTENSION_ERROR = exc
            raise RuntimeError("failed to build native complex scan extension") from exc
    return _EXTENSION


class _NativeComplexScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_real, a_imag, u_real, u_imag):
        extension = _load_extension()
        tensors = tuple(
            value.contiguous() for value in (a_real, a_imag, u_real, u_imag)
        )
        h_real, h_imag = extension.forward(*tensors)
        ctx.save_for_backward(tensors[0], tensors[1], h_real, h_imag)
        return h_real, h_imag

    @staticmethod
    def backward(ctx, grad_h_real, grad_h_imag):
        extension = _load_extension()
        a_real, a_imag, h_real, h_imag = ctx.saved_tensors
        gradients = extension.backward(
            a_real,
            a_imag,
            h_real,
            h_imag,
            grad_h_real.contiguous(),
            grad_h_imag.contiguous(),
        )
        return tuple(gradients)


def native_complex_scan(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    u_real: torch.Tensor,
    u_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the linear-time native CUDA complex recurrence."""

    if not all(value.is_cuda for value in (a_real, a_imag, u_real, u_imag)):
        raise RuntimeError("native_complex_scan requires CUDA tensors")
    return _NativeComplexScan.apply(a_real, a_imag, u_real, u_imag)
