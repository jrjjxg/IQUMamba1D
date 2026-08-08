"""CUDA microbenchmark for optimized complex scan backends.

Run on the training host after installing the project CUDA dependencies:

    python util/benchmark_complex_scan_backends.py --length 2048 --batch 4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.IQUMamba1D_ComplexStateMamba import ComplexStateSelectiveSSM


def _measure(model, x, warmup, iterations):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        optimizer.step()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--state", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    x = torch.randn(args.batch, args.length, args.dim, device="cuda")
    cases = (
        ("euler-auto", "exponential_euler", False, "auto"),
        ("reliability-auto", "exponential_euler", True, "auto"),
        ("trapezoid-auto", "exponential_trapezoidal", False, "auto"),
        ("trapezoid-reference", "exponential_trapezoidal", False, "torch"),
    )
    for name, discretization, reliability, backend in cases:
        model = ComplexStateSelectiveSSM(
            args.dim,
            d_state=args.state,
            discretization=discretization,
            reliability_enable=reliability,
            scan_backend=backend,
        ).cuda()
        elapsed = _measure(model, x, args.warmup, args.iterations)
        print(
            f"{name:24s} {elapsed:10.2f} ms/step "
            f"backend={model.last_scan_backend}"
        )


if __name__ == "__main__":
    main()
