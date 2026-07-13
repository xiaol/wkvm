#!/usr/bin/env python
"""Bounded microbenchmark for native Gemma's fused final residual boundary."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from typing import Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "wkvm.native_gemma_fused_residual_bench.v1"
GEMMA_HIDDEN_SIZE = 2560
GEMMA_LAYER_COUNT = 42


def _baseline(hidden_states, weight, residual, scalar, eps: float):
    import torch.nn.functional as F

    normalized = F.rms_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight,
        eps,
    )
    return (normalized + residual) * scalar


def _median_event_us(
    function: Callable[[], object],
    *,
    warmup: int,
    repetitions: int,
    samples: int,
) -> float:
    import torch

    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / repetitions)
    return float(statistics.median(timings))


def _capture(function: Callable[[], object]):
    import torch

    for _ in range(10):
        output = function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    torch.cuda.synchronize()
    return graph, output


def run_benchmark(
    batches: Sequence[int],
    *,
    warmup: int,
    repetitions: int,
    samples: int,
) -> dict[str, object]:
    import torch
    from wkvm.runner.gemma_fused_ops import rms_norm_residual_scalar

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    rows = []
    with torch.inference_mode():
        for batch in batches:
            torch.manual_seed(61 + int(batch))
            hidden_states = torch.randn(
                int(batch),
                1,
                GEMMA_HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
            )
            residual = torch.randn_like(hidden_states)
            weight = torch.randn(
                GEMMA_HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
            )
            scalar = torch.tensor([0.75], dtype=torch.bfloat16, device=device)
            baseline_call = lambda: _baseline(
                hidden_states,
                weight,
                residual,
                scalar,
                1e-6,
            )
            fused_call = lambda: rms_norm_residual_scalar(
                hidden_states,
                weight,
                residual,
                scalar,
                1e-6,
            )
            expected = baseline_call()
            actual = fused_call()
            torch.cuda.synchronize()
            max_abs_error = float((expected - actual).abs().max().item())
            tolerance = float(torch.finfo(torch.bfloat16).eps)

            eager_baseline_us = _median_event_us(
                baseline_call,
                warmup=warmup,
                repetitions=repetitions,
                samples=samples,
            )
            eager_fused_us = _median_event_us(
                fused_call,
                warmup=warmup,
                repetitions=repetitions,
                samples=samples,
            )
            baseline_graph, _ = _capture(baseline_call)
            fused_graph, _ = _capture(fused_call)
            graph_baseline_us = _median_event_us(
                baseline_graph.replay,
                warmup=warmup,
                repetitions=repetitions,
                samples=samples,
            )
            graph_fused_us = _median_event_us(
                fused_graph.replay,
                warmup=warmup,
                repetitions=repetitions,
                samples=samples,
            )
            rows.append(
                {
                    "batch_size": int(batch),
                    "max_abs_error": max_abs_error,
                    "tolerance": tolerance,
                    "within_tolerance": max_abs_error <= tolerance,
                    "bit_exact": bool(torch.equal(expected, actual)),
                    "eager_baseline_us": eager_baseline_us,
                    "eager_fused_us": eager_fused_us,
                    "eager_speedup": eager_baseline_us / eager_fused_us,
                    "graph_baseline_us": graph_baseline_us,
                    "graph_fused_us": graph_fused_us,
                    "graph_speedup": graph_baseline_us / graph_fused_us,
                    "estimated_graph_us_saved_per_42_layer_step": (
                        graph_baseline_us - graph_fused_us
                    )
                    * GEMMA_LAYER_COUNT,
                }
            )
    return {
        "schema": SCHEMA,
        "status": (
            "pass" if all(row["within_tolerance"] for row in rows) else "fail"
        ),
        "shape": {
            "hidden_size": GEMMA_HIDDEN_SIZE,
            "layers": GEMMA_LAYER_COUNT,
            "dtype": "bfloat16",
        },
        "measurement": {
            "warmup": int(warmup),
            "repetitions": int(repetitions),
            "samples": int(samples),
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
        "rows": rows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if any(batch < 1 or batch > 256 for batch in args.batches):
        raise ValueError("batch sizes must be in [1, 256]")
    if args.warmup < 0 or not 1 <= args.repetitions <= 100_000:
        raise ValueError("invalid warmup or repetitions")
    if not 1 <= args.samples <= 21:
        raise ValueError("samples must be in [1, 21]")
    payload = run_benchmark(
        args.batches,
        warmup=args.warmup,
        repetitions=args.repetitions,
        samples=args.samples,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
