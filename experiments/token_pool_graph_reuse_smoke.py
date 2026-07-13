#!/usr/bin/env python
"""Bounded CUDA smoke for cross-cohort token-pool graph reuse.

This intentionally does not load Gemma weights.  It drives the production
``GemmaRoutedSpanRunner.decode_batch_padded_persistent`` path with tiny native
cache objects and a synthetic CUDA model whose output depends on token-pool
metadata.  Fresh, same-shape metadata tensors are used for every cohort so a
successful hit proves that the cached graph refreshed captured metadata.

Example:

    python experiments/token_pool_graph_reuse_smoke.py --hits 4
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA = "wkvm.token_pool_graph_reuse_smoke.v1"
MAX_BATCH_SIZE = 8
MAX_ROW_WIDTH = 32
MAX_HIT_COHORTS = 16
MAX_GRAPH_WARMUP_ITERS = 3

_POSITION_SCALE = 16
_OUT_SLOT_SCALE = 1024


class SmokeInvariantError(RuntimeError):
    """Raised when the graph-reuse proof is incomplete."""


@dataclass(frozen=True)
class SmokeConfig:
    batch_size: int = 2
    row_width: int = 3
    hit_cohorts: int = 4
    graph_warmup_iters: int = 1
    device: str = "cuda"

    def validate(self) -> "SmokeConfig":
        _bounded_int("batch_size", self.batch_size, 1, MAX_BATCH_SIZE)
        _bounded_int("row_width", self.row_width, 1, MAX_ROW_WIDTH)
        _bounded_int("hit_cohorts", self.hit_cohorts, 1, MAX_HIT_COHORTS)
        _bounded_int(
            "graph_warmup_iters",
            self.graph_warmup_iters,
            0,
            MAX_GRAPH_WARMUP_ITERS,
        )
        if not str(self.device).startswith("cuda"):
            raise ValueError("device must be a CUDA device")
        return self

    @property
    def cohort_count(self) -> int:
        return int(self.hit_cohorts) + 1


@dataclass(frozen=True)
class CohortPlan:
    cohort_index: int
    token_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    token_slot_rows: tuple[tuple[int, ...], ...]
    out_cache_loc: tuple[int, ...]


@dataclass(frozen=True)
class CohortMeasurement:
    cohort_index: int
    captured: int
    cache_hit: int
    synchronized_wall_s: float
    runner_graph_prepare_wall_s: float
    runner_decode_wall_s: float
    runner_replay_dispatch_wall_s: float
    runner_metadata_copy_wall_s: float
    metadata_tensor_copies: int
    metadata_tensor_copy_skips: int
    actual_signals: tuple[float, ...]


def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> int:
    value = int(value)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def build_cohort_plans(config: SmokeConfig) -> tuple[CohortPlan, ...]:
    """Build deterministic, disjoint cohorts with identical tensor shapes."""

    config.validate()
    token_ids = tuple(101 + row for row in range(config.batch_size))
    position_ids = tuple(17 + row for row in range(config.batch_size))
    cohort_stride = config.batch_size * config.row_width
    plans = []
    for cohort_index in range(config.cohort_count):
        base = cohort_index * cohort_stride
        rows = tuple(
            tuple(
                range(
                    base + row * config.row_width,
                    base + (row + 1) * config.row_width,
                )
            )
            for row in range(config.batch_size)
        )
        plans.append(
            CohortPlan(
                cohort_index=cohort_index,
                token_ids=token_ids,
                position_ids=position_ids,
                token_slot_rows=rows,
                out_cache_loc=tuple(row[-1] for row in rows),
            )
        )
    return tuple(plans)


def expected_signals(plan: CohortPlan) -> tuple[float, ...]:
    """Return the exact signal computed by the synthetic CUDA model."""

    return tuple(
        float(
            token_id
            + position_id * _POSITION_SCALE
            + out_slot * _OUT_SLOT_SCALE
        )
        for token_id, position_id, out_slot in zip(
            plan.token_ids,
            plan.position_ids,
            plan.out_cache_loc,
        )
    )


def _signals_match(actual: Sequence[float], expected: Sequence[float]) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=0.25)
        for got, want in zip(actual, expected)
    )


def _timing_ms(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    milliseconds = [float(value) * 1000.0 for value in values]
    return {
        "min": min(milliseconds),
        "median": statistics.median(milliseconds),
        "max": max(milliseconds),
    }


def validate_and_summarize(
    plans: Sequence[CohortPlan],
    measurements: Sequence[CohortMeasurement],
    *,
    graph_cache_entries: int,
) -> dict[str, Any]:
    """Validate the smoke proof and summarize synchronized timings."""

    errors = []
    if len(plans) != len(measurements):
        errors.append(
            f"planned {len(plans)} cohorts but measured {len(measurements)}"
        )
    paired = list(zip(plans, measurements))
    for plan, measurement in paired:
        if measurement.cohort_index != plan.cohort_index:
            errors.append(
                f"cohort index mismatch: {plan.cohort_index} != "
                f"{measurement.cohort_index}"
            )
        if not _signals_match(
            measurement.actual_signals,
            expected_signals(plan),
        ):
            errors.append(
                f"cohort {plan.cohort_index} output did not reflect its metadata"
            )

    captures = sum(int(item.captured) for item in measurements)
    cache_hits = sum(int(item.cache_hit) for item in measurements)
    if not measurements or int(measurements[0].captured) != 1:
        errors.append("first cohort did not capture a graph")
    if measurements and int(measurements[0].cache_hit) != 0:
        errors.append("first cohort unexpectedly reported a cache hit")
    if captures != 1:
        errors.append(f"expected exactly one capture, observed {captures}")
    expected_hits = max(0, len(measurements) - 1)
    if cache_hits != expected_hits or any(
        int(item.cache_hit) != 1 for item in measurements[1:]
    ):
        errors.append(
            f"expected {expected_hits} later cache hits, observed {cache_hits}"
        )
    if any(int(item.captured) != 0 for item in measurements[1:]):
        errors.append("a later cohort recaptured the graph")
    if int(graph_cache_entries) != 1:
        errors.append(
            f"expected one retained graph-cache entry, observed {graph_cache_entries}"
        )
    if any(int(item.metadata_tensor_copies) < 1 for item in measurements[1:]):
        errors.append("a cache hit did not copy fresh cohort metadata")

    expected_by_cohort = [expected_signals(plan) for plan in plans]
    if len(expected_by_cohort) > 1 and any(
        values == expected_by_cohort[0] for values in expected_by_cohort[1:]
    ):
        errors.append("cohort metadata signals were not distinct")
    if errors:
        raise SmokeInvariantError("; ".join(errors))

    first = measurements[0]
    hits = list(measurements[1:])
    return {
        "proof": {
            "exactly_one_capture": True,
            "all_later_cohorts_hit": True,
            "fresh_metadata_copied_on_every_hit": True,
            "outputs_match_refreshed_metadata": True,
            "capture_count": captures,
            "cache_hit_count": cache_hits,
            "graph_cache_entries": int(graph_cache_entries),
            "hit_metadata_tensor_copies": sum(
                int(item.metadata_tensor_copies) for item in hits
            ),
        },
        "timing_ms": {
            "capture_cohort_synchronized_wall": first.synchronized_wall_s * 1000.0,
            "capture_runner_graph_prepare": (
                first.runner_graph_prepare_wall_s * 1000.0
            ),
            "hit_cohort_synchronized_wall": _timing_ms(
                [item.synchronized_wall_s for item in hits]
            ),
            "hit_runner_graph_lookup": _timing_ms(
                [item.runner_graph_prepare_wall_s for item in hits]
            ),
            "hit_runner_decode_dispatch": _timing_ms(
                [item.runner_decode_wall_s for item in hits]
            ),
            "hit_runner_replay_dispatch": _timing_ms(
                [item.runner_replay_dispatch_wall_s for item in hits]
            ),
            "hit_runner_metadata_copy": _timing_ms(
                [item.runner_metadata_copy_wall_s for item in hits]
            ),
        },
    }


class _SyntheticTokenPoolModel:
    """Two-logit CUDA model whose first logit encodes decode metadata."""

    wkvm_no_hf_transformer_forward = True

    def __init__(self, *, device: Any, config: Any) -> None:
        self.device = device
        self.config = config

    def __call__(
        self,
        *,
        input_ids,
        position_ids,
        wkvm_token_pool_decode,
        **_kwargs,
    ):
        import torch

        metadata = wkvm_token_pool_decode.metadata_by_layer_type[
            "sliding_attention"
        ]
        signal = input_ids[:, 0].to(dtype=torch.float32)
        signal = signal + position_ids[:, 0].to(dtype=torch.float32) * _POSITION_SCALE
        signal = signal + metadata.out_cache_loc.to(
            dtype=torch.float32
        ) * _OUT_SLOT_SCALE
        logits = torch.stack((signal, -signal), dim=-1).unsqueeze(1)
        return SimpleNamespace(logits=logits)


def _build_native_cache_cohort(
    plan: CohortPlan,
    *,
    hf_config: Any,
    native_config: Any,
    device: Any,
) -> list[Any]:
    import torch

    from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

    caches = []
    width = len(plan.token_slot_rows[0])
    for row in range(len(plan.token_ids)):
        cache = NativeGemmaRoutedCache(hf_config, native_config)
        keys = torch.full(
            (1, 1, width, 1),
            float(plan.cohort_index + row + 1),
            dtype=torch.float32,
            device=device,
        )
        values = torch.full_like(keys, float(plan.cohort_index + row + 11))
        cache.update(keys, values, layer_idx=0)
        caches.append(cache)
    return caches


def _build_token_pool_context(
    plan: CohortPlan,
    *,
    device: Any,
    kv_pool: Any,
    attention_workspace: Any,
):
    from wkvm.runner.gemma_token_pool import (
        TokenPoolDecodeContext,
        build_decode_metadata_from_token_slot_rows,
    )

    metadata = build_decode_metadata_from_token_slot_rows(
        plan.token_slot_rows,
        out_cache_loc=plan.out_cache_loc,
        device=device,
    )
    return TokenPoolDecodeContext(
        metadata_by_layer_type={"sliding_attention": metadata},
        metadata_by_layer_id={0: metadata},
        kv_pool=kv_pool,
        attention_workspace=attention_workspace,
        covered_layer_types=frozenset({"sliding_attention"}),
    )


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    """Run one capture plus bounded cross-cohort cache hits on CUDA."""

    config.validate()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the graph-reuse smoke")
    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("device must resolve to CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)

    from wkvm.models.gemma import gemma4_e4b_routed_span_config
    from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

    plans = build_cohort_plans(config)
    sliding_window = max(64, config.row_width + 1)
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        num_kv_shared_layers=0,
        layer_types=("sliding_attention",),
        sliding_window=sliding_window,
    )
    native_config = gemma4_e4b_routed_span_config(
        num_hidden_layers=1,
        num_kv_shared_layers=0,
        layer_types=("sliding_attention",),
        sliding_window=sliding_window,
    )
    model = _SyntheticTokenPoolModel(device=device, config=hf_config)
    runner = GemmaRoutedSpanRunner(
        model,
        SimpleNamespace(),
        persistent_padded_decode_cuda_graph=True,
        persistent_padded_decode_graph_warmup_iters=config.graph_warmup_iters,
    )
    kv_pool = SimpleNamespace(layer_specs={})
    attention_workspace = object()
    prepared = [
        (
            plan,
            _build_native_cache_cohort(
                plan,
                hf_config=hf_config,
                native_config=native_config,
                device=device,
            ),
            _build_token_pool_context(
                plan,
                device=device,
                kv_pool=kv_pool,
                attention_workspace=attention_workspace,
            ),
        )
        for plan in plans
    ]

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measurements = []
    for plan, caches, context in prepared:
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        logits, _merged_cache = runner.decode_batch_padded_persistent(
            caches,
            list(plan.token_ids),
            position_ids=list(plan.position_ids),
            reserve_steps=1,
            token_pool_decode=context,
        )
        torch.cuda.synchronize(device)
        synchronized_wall_s = time.perf_counter() - started
        actual_signals = tuple(float(value) for value in logits[:, 0].cpu().tolist())
        info = dict(runner.last_decode_batch_info)
        measurements.append(
            CohortMeasurement(
                cohort_index=plan.cohort_index,
                captured=int(
                    info.get("persistent_padded_decode_cuda_graph_captured", 0)
                ),
                cache_hit=int(
                    info.get("persistent_padded_decode_cuda_graph_cache_hit", 0)
                ),
                synchronized_wall_s=synchronized_wall_s,
                runner_graph_prepare_wall_s=float(
                    info.get("cuda_graph_capture_wall_s", 0.0) or 0.0
                ),
                runner_decode_wall_s=float(
                    info.get("cuda_graph_decode_wall_s_total", 0.0) or 0.0
                ),
                runner_replay_dispatch_wall_s=float(
                    info.get("cuda_graph_replay_wall_s", 0.0) or 0.0
                ),
                runner_metadata_copy_wall_s=float(
                    info.get("cuda_graph_metadata_copy_wall_s", 0.0) or 0.0
                ),
                metadata_tensor_copies=int(
                    info.get("cuda_graph_metadata_tensor_copies", 0) or 0
                ),
                metadata_tensor_copy_skips=int(
                    info.get("cuda_graph_metadata_tensor_copy_skips", 0) or 0
                ),
                actual_signals=actual_signals,
            )
        )

    summary = validate_and_summarize(
        plans,
        measurements,
        graph_cache_entries=len(runner._token_pool_decode_graph_cache),
    )
    return {
        "schema": SCHEMA,
        "status": "pass",
        "config": asdict(config),
        "bounds": {
            "max_batch_size": MAX_BATCH_SIZE,
            "max_row_width": MAX_ROW_WIDTH,
            "max_hit_cohorts": MAX_HIT_COHORTS,
            "max_graph_warmup_iters": MAX_GRAPH_WARMUP_ITERS,
        },
        "runtime": {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        **summary,
        "cohorts": [
            {
                **asdict(measurement),
                "expected_signals": list(expected_signals(plan)),
            }
            for plan, measurement in zip(plans, measurements)
        ],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--row-width", type=int, default=3)
    parser.add_argument("--hits", type=int, default=4, dest="hit_cohorts")
    parser.add_argument("--graph-warmup-iters", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None, dest="json_path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = SmokeConfig(
        batch_size=args.batch_size,
        row_width=args.row_width,
        hit_cohorts=args.hit_cohorts,
        graph_warmup_iters=args.graph_warmup_iters,
        device=args.device,
    )
    try:
        payload = run_smoke(config)
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    payload["launch_command"] = " ".join(
        shlex.quote(value) for value in [sys.executable, *sys.argv]
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
