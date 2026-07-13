#!/usr/bin/env python
"""Native routed-span quality smoke for Gemma 4 E4B.

This reuses the synthetic t1/t2/t3 builders from ``quality_eval.py`` but runs
them through the same native checkpoint, routed cache, token-pool attention,
Triton, and persistent decode path used by the architecture benchmark.

Example:

  HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python \
      experiments/native_gemma_quality.py \
      --ctxs 8192 --depths 0.1,0.5,0.9 \
      --json /tmp/wkvm-native-quality.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bench_prompt_utils import (  # noqa: E402
    generated_output_fingerprint,
    prompt_set_fingerprint,
)


TaskBuilder = Callable[[Any, random.Random, int, float], tuple[Any, Callable[[str], float], int]]

FULL_GRID_CONTEXTS = (8192, 16384, 32768)
FULL_GRID_DEPTHS = (0.1, 0.3, 0.5, 0.7, 0.9)
FULL_GRID_T12_SEEDS = 3
FULL_GRID_T3_SEEDS = 1
HISTORICAL_QUALITY_GATE = {
    "overall_cell_mean_score": 0.90,
    "task_mean_scores": {
        "t1-needle": 1.0,
        "t2-multikey": 8.0 / 9.0,
        "t3-aggregate": 0.80,
    },
    "task_minimum_cell_scores": {
        "t1-needle": 1.0,
        "t2-multikey": 2.0 / 3.0,
        "t3-aggregate": 2.0 / 3.0,
    },
}
RUNTIME_SOURCE_PATHS = (
    "experiments/bench_prompt_utils.py",
    "experiments/native_gemma_engine_smoke.py",
    "experiments/native_gemma_quality.py",
    "experiments/native_gemma_smoke.py",
    "experiments/quality_eval.py",
    "wkvm/core/arena.py",
    "wkvm/core/request.py",
    "wkvm/core/scheduler.py",
    "wkvm/gemma_engine.py",
    "wkvm/models/gemma.py",
    "wkvm/runner/gemma_fused_ops.py",
    "wkvm/runner/gemma_native_forward.py",
    "wkvm/runner/gemma_runner.py",
    "wkvm/runner/gemma_state.py",
    "wkvm/runner/gemma_token_pool.py",
    "wkvm/runner/gemma_token_pool_attention.py",
    "wkvm/runner/gemma_token_pool_triton.py",
)


@dataclass(frozen=True)
class QualityCase:
    req_id: str
    task: str
    context_tokens: int
    depth: float
    seed: int
    seed_key: str
    prompt_token_ids: tuple[int, ...]
    break_mask: tuple[bool, ...]
    max_new_tokens: int
    scorer: Callable[[str], float]
    scorer_metadata: dict[str, Any]


def parse_int_csv(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_depth_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in values
    ):
        raise argparse.ArgumentTypeError("depths must be finite values in [0, 1]")
    return values


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


def scorer_metadata(scorer: Callable[[str], float]) -> dict[str, Any]:
    explicit = getattr(scorer, "meta", None)
    if explicit is not None:
        return {"source": "scorer.meta", "value": _json_safe(explicit)}
    closure = getattr(scorer, "__closure__", None)
    freevars = getattr(getattr(scorer, "__code__", None), "co_freevars", ())
    if not closure or not freevars:
        return {"source": "unavailable", "value": None}
    values = {
        name: _json_safe(cell.cell_contents)
        for name, cell in zip(freevars, closure)
    }
    return {"source": "scorer.closure", "value": values}


def _prompt_ids(prompt: Any) -> list[int]:
    if hasattr(prompt, "detach"):
        prompt = prompt.detach().cpu()
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if (
        isinstance(prompt, (list, tuple))
        and len(prompt) == 1
        and isinstance(prompt[0], (list, tuple))
    ):
        prompt = prompt[0]
    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("quality task returned an empty or unsupported prompt")
    return [int(token_id) for token_id in prompt]


def _legacy_tasks_and_break_mask():
    from experiments import quality_eval

    return quality_eval.TASKS, quality_eval.break_mask_for


def build_quality_cases(
    tokenizer,
    *,
    contexts: Sequence[int],
    depths: Sequence[float],
    t12_seeds: int,
    t3_seeds: int,
    tasks: Sequence[tuple[str, TaskBuilder]] | None = None,
    break_mask_builder: Callable[[Any, list[int]], Iterable[bool]] | None = None,
) -> list[QualityCase]:
    if t12_seeds < 1 or t3_seeds < 1:
        raise ValueError("quality seed counts must be >= 1")
    if tasks is None or break_mask_builder is None:
        legacy_tasks, legacy_break_mask = _legacy_tasks_and_break_mask()
        tasks = legacy_tasks if tasks is None else tasks
        break_mask_builder = legacy_break_mask if break_mask_builder is None else break_mask_builder

    cases: list[QualityCase] = []
    for context_tokens in contexts:
        for task_name, task_builder in tasks:
            seed_count = t3_seeds if task_name == "t3-aggregate" else t12_seeds
            for depth in depths:
                for seed in range(seed_count):
                    seed_key = f"{context_tokens}/{task_name}/{depth}/{seed}"
                    prompt, scorer, max_new_tokens = task_builder(
                        tokenizer,
                        random.Random(seed_key),
                        int(context_tokens),
                        float(depth),
                    )
                    token_ids = _prompt_ids(prompt)
                    if len(token_ids) != int(context_tokens):
                        raise ValueError(
                            f"{task_name} prompt length {len(token_ids)} != ctx {context_tokens}"
                        )
                    breaks = tuple(
                        bool(value)
                        for value in break_mask_builder(tokenizer, token_ids)
                    )
                    if len(breaks) != len(token_ids):
                        raise ValueError(f"{task_name} break mask length does not match prompt")
                    req_id = (
                        f"quality-c{context_tokens}-{task_name}-"
                        f"d{float(depth):.6f}-s{seed}"
                    )
                    cases.append(
                        QualityCase(
                            req_id=req_id,
                            task=task_name,
                            context_tokens=int(context_tokens),
                            depth=float(depth),
                            seed=seed,
                            seed_key=seed_key,
                            prompt_token_ids=tuple(token_ids),
                            break_mask=breaks,
                            max_new_tokens=int(max_new_tokens),
                            scorer=scorer,
                            scorer_metadata=scorer_metadata(scorer),
                        )
                    )
    if len({case.req_id for case in cases}) != len(cases):
        raise ValueError("quality request IDs are not unique")
    return cases


def new_routed_cache_observations() -> dict[str, Any]:
    return {
        "cache_samples": 0,
        "routed_layer_samples": 0,
        "max_cumulative_tokens": 0,
        "max_materialized_tokens": 0,
        "max_pending_tokens": 0,
        "max_evicted_tokens": 0,
        "max_active_route_slots": 0,
        "minimum_materialized_fraction": None,
        "dense_storage_release_observed": False,
    }


def observe_routed_caches(engine, observations: dict[str, Any]) -> None:
    caches = getattr(engine, "_caches", {})
    for cache in caches.values():
        observations["cache_samples"] += 1
        for layer in getattr(cache, "layers", ()):
            if not hasattr(layer, "_evicted") or not hasattr(layer, "_pend_k"):
                continue
            observations["routed_layer_samples"] += 1
            cumulative = int(getattr(layer, "cumulative_length", 0))
            pending = int(layer._pend_k.shape[2])
            evicted = int(getattr(layer, "_evicted", 0))
            active = int(getattr(layer, "_n_active", 0))
            materialized_fn = getattr(layer, "materialized_tokens", None)
            materialized = int(materialized_fn()) if callable(materialized_fn) else 0
            observations["max_cumulative_tokens"] = max(
                observations["max_cumulative_tokens"], cumulative
            )
            observations["max_materialized_tokens"] = max(
                observations["max_materialized_tokens"], materialized
            )
            observations["max_pending_tokens"] = max(
                observations["max_pending_tokens"], pending
            )
            observations["max_evicted_tokens"] = max(
                observations["max_evicted_tokens"], evicted
            )
            observations["max_active_route_slots"] = max(
                observations["max_active_route_slots"], active
            )
            if cumulative > 0:
                fraction = materialized / cumulative
                current = observations["minimum_materialized_fraction"]
                observations["minimum_materialized_fraction"] = (
                    fraction if current is None else min(current, fraction)
                )
            observations["dense_storage_release_observed"] = bool(
                observations["dense_storage_release_observed"]
                or getattr(layer, "_dense_storage_released", False)
            )


def _decode(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def break_mask_fingerprint(cases: Sequence[QualityCase]) -> dict[str, Any]:
    digest = hashlib.sha256()
    true_count = 0
    total_count = 0
    for case in cases:
        row = bytes(bool(value) for value in case.break_mask)
        digest.update(len(row).to_bytes(8, "little", signed=False))
        digest.update(row)
        true_count += sum(row)
        total_count += len(row)
    return {
        "schema": "wkvm.quality_break_masks.sha256.v1",
        "request_count": len(cases),
        "mask_value_count": total_count,
        "true_value_count": true_count,
        "break_masks_sha256": digest.hexdigest(),
    }


def run_engine_cases(
    engine,
    tokenizer,
    cases: Sequence[QualityCase],
    *,
    max_steps: int,
    score_stop_token_ids: frozenset[int] = frozenset({1, 106}),
    observations: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[Any]]:
    from wkvm.core.request import Request, RequestStatus

    requests = [
        Request(
            prompt_token_ids=list(case.prompt_token_ids),
            max_new_tokens=case.max_new_tokens,
            req_id=case.req_id,
        )
        for case in cases
    ]
    if observations is None:
        observations = new_routed_cache_observations()
    for request, case in zip(requests, cases):
        engine.add_request(request, break_mask=list(case.break_mask))

    steps = 0
    while engine.has_unfinished:
        observe_routed_caches(engine, observations)
        engine.step()
        observe_routed_caches(engine, observations)
        steps += 1
        if steps > max_steps:
            raise RuntimeError("native quality smoke did not converge")

    rows: list[dict[str, Any]] = []
    for request, case in zip(requests, cases):
        output_ids = [int(token_id) for token_id in request.output_token_ids]
        cut = next(
            (
                index
                for index, token_id in enumerate(output_ids)
                if token_id in score_stop_token_ids
            ),
            len(output_ids),
        )
        scored_ids = output_ids[:cut]
        output_text = _decode(tokenizer, scored_ids)
        successful = bool(
            request.status is RequestStatus.FINISHED_LENGTH
            and len(output_ids) == case.max_new_tokens
        )
        score = None
        score_error = None
        if successful:
            try:
                score = float(case.scorer(output_text))
            except Exception as exc:
                score_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

        prompt_fingerprint = prompt_set_fingerprint(
            [case.prompt_token_ids],
            prompt_token_source="hf_tokenizer",
        )
        output_fingerprint = generated_output_fingerprint(
            [(case.req_id, output_ids)]
        )
        rows.append(
            {
                "req_id": case.req_id,
                "task": case.task,
                "context_tokens": case.context_tokens,
                "depth": case.depth,
                "seed": case.seed,
                "seed_key": case.seed_key,
                "max_new_tokens": case.max_new_tokens,
                "status": request.status.name.lower(),
                "successful": successful,
                "score": score,
                "score_error": score_error,
                "scorer_metadata": case.scorer_metadata,
                "score_stop_token_ids": sorted(score_stop_token_ids),
                "scored_output_token_count": len(scored_ids),
                "output_text": output_text,
                "output_text_sha256": _text_sha256(output_text),
                "output_token_ids": output_ids,
                "prompt_fingerprint": prompt_fingerprint,
                "prompt_token_ids_sha256": prompt_fingerprint[
                    "prompt_token_ids_sha256"
                ],
                "generated_output_fingerprint": output_fingerprint,
                "request_output_token_ids_sha256": output_fingerprint[
                    "request_output_token_ids_sha256"
                ],
            }
        )
    return rows, observations, requests


def summarize_scores(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    all_scores: list[float] = []
    cells: list[dict[str, Any]] = []
    cell_keys = sorted(
        {
            (int(row["context_tokens"]), str(row["task"]), float(row["depth"]))
            for row in rows
        }
    )
    for context_tokens, task, depth in cell_keys:
        cell_rows = [
            row
            for row in rows
            if int(row["context_tokens"]) == context_tokens
            and str(row["task"]) == task
            and float(row["depth"]) == depth
        ]
        scores = [
            float(row["score"])
            for row in cell_rows
            if row.get("score") is not None
        ]
        cells.append(
            {
                "context_tokens": context_tokens,
                "task": task,
                "depth": depth,
                "case_count": len(cell_rows),
                "successful_count": sum(
                    bool(row.get("successful")) for row in cell_rows
                ),
                "scored_count": len(scores),
                "mean_score": None if not scores else sum(scores) / len(scores),
                "minimum_score": None if not scores else min(scores),
            }
        )
    for task in sorted({str(row["task"]) for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        scores = [float(row["score"]) for row in task_rows if row.get("score") is not None]
        all_scores.extend(scores)
        by_task[task] = {
            "case_count": len(task_rows),
            "successful_count": sum(bool(row.get("successful")) for row in task_rows),
            "scored_count": len(scores),
            "mean_score": None if not scores else sum(scores) / len(scores),
            "minimum_score": None if not scores else min(scores),
        }
    cell_scores = [
        float(cell["mean_score"])
        for cell in cells
        if cell["mean_score"] is not None
    ]
    return {
        "case_count": len(rows),
        "successful_count": sum(bool(row.get("successful")) for row in rows),
        "scored_count": len(all_scores),
        "overall_mean_score": None if not all_scores else sum(all_scores) / len(all_scores),
        "cell_count": len(cells),
        "scored_cell_count": len(cell_scores),
        "overall_cell_mean_score": (
            None if not cell_scores else sum(cell_scores) / len(cell_scores)
        ),
        "by_task": by_task,
        "cells": cells,
    }


def quality_validation(
    *,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    workload: dict[str, Any],
    require_full_grid: bool,
) -> dict[str, Any]:
    violations: list[str] = []
    scores = [row.get("score") for row in rows]
    if any(not row.get("successful") for row in rows):
        violations.append("one_or_more_requests_failed")
    if any(score is None for score in scores):
        violations.append("one_or_more_scores_missing")
    elif any(not math.isfinite(float(score)) for score in scores):
        violations.append("one_or_more_scores_nonfinite")
    elif any(float(score) < 0.0 or float(score) > 1.0 for score in scores):
        violations.append("one_or_more_scores_out_of_range")

    overall = summary.get("overall_cell_mean_score")
    overall_floor = float(HISTORICAL_QUALITY_GATE["overall_cell_mean_score"])
    if overall is None or float(overall) + 1e-12 < overall_floor:
        violations.append("overall_cell_mean_below_gate")

    task_mean_floors = HISTORICAL_QUALITY_GATE["task_mean_scores"]
    task_cell_floors = HISTORICAL_QUALITY_GATE["task_minimum_cell_scores"]
    by_task = summary.get("by_task", {})
    cells = summary.get("cells", [])
    for task, floor in task_mean_floors.items():
        task_summary = by_task.get(task)
        mean_score = None if task_summary is None else task_summary.get("mean_score")
        if mean_score is None or float(mean_score) + 1e-12 < float(floor):
            violations.append(f"{task}_mean_below_gate")
        task_cells = [cell for cell in cells if cell.get("task") == task]
        minimum_cell = min(
            (
                float(cell["mean_score"])
                for cell in task_cells
                if cell.get("mean_score") is not None
            ),
            default=None,
        )
        if minimum_cell is None or minimum_cell + 1e-12 < float(task_cell_floors[task]):
            violations.append(f"{task}_cell_below_gate")

    full_grid_observed = bool(
        tuple(workload.get("contexts", ())) == FULL_GRID_CONTEXTS
        and tuple(workload.get("depths", ())) == FULL_GRID_DEPTHS
        and int(workload.get("t12_seeds", 0)) == FULL_GRID_T12_SEEDS
        and int(workload.get("t3_seeds", 0)) == FULL_GRID_T3_SEEDS
        and int(workload.get("case_count", 0)) == 105
        and int(summary.get("cell_count", 0)) == 45
    )
    if require_full_grid and not full_grid_observed:
        violations.append("full_quality_grid_not_observed")

    return {
        "passed": not violations,
        "violations": violations,
        "gate": HISTORICAL_QUALITY_GATE,
        "full_grid_required": bool(require_full_grid),
        "full_grid_observed": full_grid_observed,
    }


def runtime_validation(
    *,
    config: dict[str, Any],
    engine_stats: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    observations: dict[str, Any],
    triton_stats: dict[str, Any] | None,
    require_benchmark_shape: bool = False,
) -> dict[str, Any]:
    violations: list[str] = []
    if config.get("chunk") != 2048:
        violations.append("chunk_is_not_2048")
    if config.get("route_chunk") != 512:
        violations.append("route_chunk_is_not_512")
    if engine_stats.get("uses_hf_transformer_forward") is not False:
        violations.append("hf_transformer_forward_not_disabled")
    if engine_stats.get("uses_hf_model_construction") is not False:
        violations.append("hf_model_construction_not_disabled")
    if engine_stats.get("native_gemma_checkpoint_loader") is not True:
        violations.append("native_checkpoint_loader_not_proven")
    if engine_stats.get("token_pool_attention_enabled") is not True:
        violations.append("token_pool_attention_not_enabled")
    if int(engine_stats.get("error_count", 0)) != 0:
        violations.append("engine_errors_observed")
    if any(not row.get("successful") for row in rows):
        violations.append("one_or_more_requests_failed")
    if any(row.get("score") is None for row in rows):
        violations.append("one_or_more_scores_missing")

    fold_expected = max((int(row["context_tokens"]) for row in rows), default=0) >= (
        int(config.get("sink", 0))
        + int(config.get("window", 0))
        + int(config.get("route_chunk", 0))
    )
    fold_observed = int(observations.get("max_evicted_tokens", 0)) > 0
    if fold_expected and not fold_observed:
        violations.append("routed_fold_not_observed")
    pending_bounded = int(observations.get("max_pending_tokens", 0)) < int(
        config.get("route_chunk", 0)
    )
    if observations.get("routed_layer_samples", 0) < 1:
        violations.append("routed_cache_not_observed")
    elif not pending_bounded:
        violations.append("pending_tail_reached_route_chunk")

    batched_prefill_expected = bool(
        int(config.get("prefill_microbatch_rows") or 0) > 1
        and int(config.get("slots") or 0) > 1
        and len(rows) > 1
    )
    batched_prefill_observed = int(engine_stats.get("max_prefill_batch_rows", 0)) > 1
    if batched_prefill_expected and not batched_prefill_observed:
        violations.append("prefill_microbatch_not_observed")

    if triton_stats is not None:
        if triton_stats.get("effective_enabled") is not True:
            violations.append("token_pool_triton_not_effectively_enabled")
        if int(triton_stats.get("runtime_errors", 0)) != 0:
            violations.append("token_pool_triton_runtime_errors")
        if triton_stats.get("fallback_reasons"):
            violations.append("token_pool_triton_fallback_observed")

    request_traces = engine_stats.get("requests")
    trace_contract_observed = bool(
        isinstance(request_traces, dict)
        and len(request_traces) == len(rows)
        and all(
            row.get("req_id") in request_traces
            and request_traces[row["req_id"]].get("error") is None
            and request_traces[row["req_id"]].get("finish_reason") == "length"
            and int(request_traces[row["req_id"]].get("output_tokens", -1))
            == int(request_traces[row["req_id"]].get("target_output_tokens", -2))
            for row in rows
        )
    )
    if request_traces is not None and not trace_contract_observed:
        violations.append("request_trace_contract_failed")

    benchmark_shape_observed = bool(
        int(config.get("sink", 0)) == 16
        and int(config.get("window", 0)) == 1024
        and int(config.get("m_slots", 0)) == 64
        and int(config.get("slots", 0)) == 16
        and int(config.get("prefill_microbatch_rows", 0)) == 2
        and int(config.get("decode_microbatch_rows", 0)) == 16
        and int(config.get("persistent_padded_decode_steps", 0)) == 128
        and int(config.get("persistent_padded_decode_graph_warmup_iters", -1)) == 0
        and config.get("native_gemma_checkpoint_loader") is True
        and config.get("use_native_gemma_forward") is True
        and config.get("native_gemma_attention_backend") == "sdpa_single_gqa"
        and config.get("native_gemma_projection_backend") == "separate"
        and config.get("enable_token_pool_attention") is True
        and int(config.get("token_pool_capacity", 0)) == 36_864
        and int(config.get("token_pool_max_context_len", 0)) >= 33_024
        and int(config.get("token_pool_paged_block_size", 0)) == 16
        and config.get("persistent_padded_sliding_metadata_padding") is True
        and int(engine_stats.get("max_prefill_batch_rows", 0)) == 2
        and int(engine_stats.get("max_decode_batch_rows", 0)) == 16
        and int(engine_stats.get("persistent_padded_decode_cuda_graph_captures", 0)) > 0
        and int(engine_stats.get("persistent_padded_decode_cuda_graph_replays", 0)) > 0
        and trace_contract_observed
        and bool(observations.get("dense_storage_release_observed"))
        and triton_stats is not None
        and int(triton_stats.get("successes", 0)) > 0
        and int(triton_stats.get("paged_successes", 0)) > 0
    )
    if require_benchmark_shape and not benchmark_shape_observed:
        violations.append("benchmark_b16_winning_shape_not_observed")

    return {
        "passed": not violations,
        "violations": violations,
        "winning_path_configured": (
            config.get("chunk") == 2048 and config.get("route_chunk") == 512
        ),
        "native_checkpoint_and_forward_proven": (
            engine_stats.get("uses_hf_transformer_forward") is False
            and engine_stats.get("uses_hf_model_construction") is False
            and engine_stats.get("native_gemma_checkpoint_loader") is True
        ),
        "fold_expected": fold_expected,
        "fold_observed": fold_observed,
        "pending_tail_bounded": pending_bounded,
        "batched_prefill_expected": batched_prefill_expected,
        "batched_prefill_observed": batched_prefill_observed,
        "benchmark_shape_required": bool(require_benchmark_shape),
        "benchmark_shape_observed": benchmark_shape_observed,
        "request_trace_contract_observed": trace_contract_observed,
    }


def _git_bytes(*args: str) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL
        )
    except Exception:
        return None


def provenance() -> dict[str, Any]:
    commit = _git_bytes("rev-parse", "HEAD")
    status = _git_bytes("status", "--porcelain=v1", "-z")
    diff = _git_bytes("diff", "--binary", "HEAD")
    source = Path(__file__).read_bytes()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "git_commit": None if commit is None else commit.decode().strip(),
        "git_worktree_clean": None if status is None else not bool(status),
        "git_status_sha256": None if status is None else hashlib.sha256(status).hexdigest(),
        "git_tracked_diff_sha256": None if diff is None else hashlib.sha256(diff).hexdigest(),
        "harness_source_sha256": hashlib.sha256(source).hexdigest(),
    }


def runtime_source_identity() -> dict[str, Any]:
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative_path in RUNTIME_SOURCE_PATHS:
        path = ROOT / relative_path
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        files[relative_path] = digest
        encoded_path = relative_path.encode("utf-8")
        combined.update(len(encoded_path).to_bytes(4, "little", signed=False))
        combined.update(encoded_path)
        combined.update(bytes.fromhex(digest))
    return {
        "schema": "wkvm.runtime_source_identity.sha256.v1",
        "file_count": len(files),
        "files": files,
        "combined_sha256": combined.hexdigest(),
    }


def finalize_provenance(
    provenance_before: dict[str, Any],
    source_identity_before: dict[str, Any],
) -> dict[str, Any]:
    provenance_after = provenance()
    source_identity_after = runtime_source_identity()
    provenance_after["pre_run_timestamp_utc"] = provenance_before["timestamp_utc"]
    provenance_after["pre_run_git_commit"] = provenance_before["git_commit"]
    provenance_after["pre_run_git_status_sha256"] = provenance_before[
        "git_status_sha256"
    ]
    provenance_after["pre_run_git_tracked_diff_sha256"] = provenance_before[
        "git_tracked_diff_sha256"
    ]
    provenance_after["git_identity_unchanged_during_run"] = all(
        provenance_before[key] == provenance_after[key]
        for key in (
            "git_commit",
            "git_status_sha256",
            "git_tracked_diff_sha256",
            "harness_source_sha256",
        )
    )
    provenance_after["runtime_source_identity_before"] = source_identity_before
    provenance_after["runtime_source_identity_after"] = source_identity_after
    provenance_after["runtime_source_unchanged_during_run"] = (
        source_identity_before == source_identity_after
    )
    return provenance_after


def apply_winning_path_environment() -> dict[str, str]:
    values = {
        "WKVM_ENABLE_TOKEN_POOL_TRITON": "1",
        "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON": "1",
        "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON": "1",
        "WKVM_TOKEN_POOL_TRITON_STRICT": "1",
        "WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY": "1",
    }
    os.environ.update(values)
    try:
        from wkvm.runner.gemma_token_pool_attention import (
            reset_token_pool_triton_dispatch_plan_cache,
        )

        reset_token_pool_triton_dispatch_plan_cache()
    except Exception:
        pass
    return values


def build_native_engine(model, cases: Sequence[QualityCase], args):
    from native_gemma_engine_smoke import chunked_scheduler_config
    from wkvm.gemma_engine import GemmaNativeEngine
    from wkvm.models.gemma import gemma4_e4b_routed_span_config

    config = gemma4_e4b_routed_span_config(
        num_hidden_layers=model.config.num_hidden_layers,
        num_kv_shared_layers=getattr(model.config, "num_kv_shared_layers", 0),
        layer_types=tuple(model.config.layer_types),
        num_kv_heads=getattr(model.config, "num_global_key_value_heads", None)
        or getattr(model.config, "num_key_value_heads", 2),
        head_dim=getattr(model.config, "global_head_dim", None)
        or getattr(model.config, "head_dim", 512),
        sink_tokens=args.sink,
        ring_tokens=args.window,
        routed_slots=args.m_slots,
        pending_tokens=args.route_chunk,
        sliding_window=getattr(model.config, "sliding_window", 1024),
    )
    prompts = [list(case.prompt_token_ids) for case in cases]
    scheduler_config = chunked_scheduler_config(
        prompts,
        slots=args.slots,
        token_budget=None,
        chunk=args.chunk,
    )
    engine = GemmaNativeEngine(
        model,
        config,
        num_slots=args.slots,
        scheduler_config=scheduler_config,
        prefill_chunk=args.chunk,
        prefill_microbatch_rows=args.prefill_microbatch_rows,
        decode_microbatch_rows=args.decode_microbatch_rows,
        decode_batch_planner="scheduler",
        persistent_exact_decode=True,
        persistent_padded_decode=True,
        persistent_padded_decode_steps=args.persistent_padded_decode_steps,
        persistent_padded_full_attention_rows=None,
        persistent_padded_sliding_metadata_padding=True,
        persistent_padded_decode_cuda_graph=True,
        persistent_padded_decode_graph_warmup_iters=args.graph_warmup_iters,
        use_native_gemma_forward=True,
        native_gemma_attention_backend=args.native_gemma_attention_backend,
        native_gemma_projection_backend=args.native_gemma_projection_backend,
        native_gemma_weight_backend="hf_live",
        enable_token_pool_attention=True,
        token_pool_max_context_len=args.token_pool_max_context_len,
        token_pool_capacity=args.token_pool_capacity,
        token_pool_paged_block_size=16,
    )
    return engine


def configuration(args, runtime_environment: dict[str, str]) -> dict[str, Any]:
    return {
        "sink": args.sink,
        "window": args.window,
        "m_slots": args.m_slots,
        "route_chunk": args.route_chunk,
        "chunk": args.chunk,
        "slots": args.slots,
        "prefill_microbatch_rows": args.prefill_microbatch_rows,
        "decode_microbatch_rows": args.decode_microbatch_rows,
        "persistent_padded_decode_steps": args.persistent_padded_decode_steps,
        "persistent_padded_decode_cuda_graph": True,
        "persistent_padded_decode_graph_warmup_iters": args.graph_warmup_iters,
        "persistent_padded_sliding_metadata_padding": True,
        "native_gemma_checkpoint_loader": True,
        "use_native_gemma_forward": True,
        "native_gemma_attention_backend": args.native_gemma_attention_backend,
        "native_gemma_projection_backend": args.native_gemma_projection_backend,
        "enable_token_pool_attention": True,
        "token_pool_max_context_len": args.token_pool_max_context_len,
        "token_pool_capacity": args.token_pool_capacity,
        "token_pool_paged_block_size": 16,
        "runtime_environment": runtime_environment,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def emit_payload(payload: dict[str, Any], output_path: str | None) -> None:
    if output_path is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    atomic_write_json(Path(output_path), payload)
    print(f"WROTE {output_path}")


def run(args) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from native_gemma_smoke import load_model, resolve_model_path

    started = time.perf_counter()
    provenance_before = provenance()
    source_identity_before = runtime_source_identity()
    runtime_environment = apply_winning_path_environment()
    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cases = build_quality_cases(
        tokenizer,
        contexts=args.ctxs,
        depths=args.depths,
        t12_seeds=args.t12_seeds,
        t3_seeds=args.t3_seeds,
    )
    model = load_model(
        model_path,
        args.device,
        "sdpa",
        native_checkpoint_loader=True,
        native_gemma_attention_backend=args.native_gemma_attention_backend,
        native_gemma_projection_backend=args.native_gemma_projection_backend,
    )
    engine = build_native_engine(model, cases, args)
    config = configuration(args, runtime_environment)
    workload = {
        "contexts": list(args.ctxs),
        "depths": list(args.depths),
        "t12_seeds": args.t12_seeds,
        "t3_seeds": args.t3_seeds,
        "case_count": len(cases),
    }
    observations = new_routed_cache_observations()
    try:
        rows, observations, requests = run_engine_cases(
            engine,
            tokenizer,
            cases,
            max_steps=args.max_steps,
            observations=observations,
        )
    except Exception as exc:
        failure_payload = {
            "schema": "wkvm.native_gemma_quality.v1",
            "status": "failed",
            "engine": "wkvm-native",
            "model_path": model_path,
            "dtype": "bfloat16",
            "device": args.device,
            "failure": {
                "type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
            },
            "workload": workload,
            "config": config,
            "prompt_fingerprint": prompt_set_fingerprint(
                [case.prompt_token_ids for case in cases],
                prompt_token_source="hf_tokenizer",
            ),
            "break_mask_fingerprint": break_mask_fingerprint(cases),
            "routed_cache_observations": observations,
            "engine_stats": engine.stats(),
            "provenance": finalize_provenance(
                provenance_before,
                source_identity_before,
            ),
            "runtime_seconds": time.perf_counter() - started,
        }
        emit_payload(failure_payload, args.json)
        raise
    engine_stats = engine.stats()
    try:
        from wkvm.runner.gemma_native_forward import token_pool_triton_stats

        triton_stats = token_pool_triton_stats()
    except Exception as exc:
        triton_stats = {"available": False, "error": str(exc).splitlines()[0]}

    validation = runtime_validation(
        config=config,
        engine_stats=engine_stats,
        rows=rows,
        observations=observations,
        triton_stats=triton_stats,
        require_benchmark_shape=args.require_benchmark_shape,
    )
    prompts = [case.prompt_token_ids for case in cases]
    outputs = [
        (request.req_id, request.output_token_ids)
        for request in requests
    ]
    prompt_fingerprint = prompt_set_fingerprint(
        prompts,
        prompt_token_source="hf_tokenizer",
    )
    output_fingerprint = generated_output_fingerprint(outputs)
    provenance_after = finalize_provenance(
        provenance_before,
        source_identity_before,
    )
    summary = summarize_scores(rows)
    quality = quality_validation(
        rows=rows,
        summary=summary,
        workload=workload,
        require_full_grid=args.require_full_grid,
    )
    payload = {
        "schema": "wkvm.native_gemma_quality.v1",
        "engine": "wkvm-native",
        "model_path": model_path,
        "dtype": "bfloat16",
        "device": args.device,
        "task_builder_source": "experiments/quality_eval.py:TASKS",
        "scoring": {
            "method": "legacy_substring_scorers",
            "generation": "greedy_fixed_length",
            "score_stop_token_ids": [1, 106],
        },
        "workload": workload,
        "config": config,
        "summary": summary,
        "quality_validation": quality,
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_token_ids_sha256": prompt_fingerprint["prompt_token_ids_sha256"],
        "generated_output_fingerprint": output_fingerprint,
        "request_output_token_ids_sha256": output_fingerprint[
            "request_output_token_ids_sha256"
        ],
        "routed_cache_observations": observations,
        "break_mask_fingerprint": break_mask_fingerprint(cases),
        "token_pool_triton": triton_stats,
        "engine_stats": engine_stats,
        "runtime_validation": validation,
        "provenance": provenance_after,
        "runtime_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    emit_payload(payload, args.json)
    if args.require_winning_path and not validation["passed"]:
        raise RuntimeError(
            f"native winning-path validation failed: {validation['violations']}"
        )
    if args.require_quality_gate and not quality["passed"]:
        raise RuntimeError(f"native quality gate failed: {quality['violations']}")
    torch.cuda.empty_cache()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctxs", type=parse_int_csv, default=parse_int_csv("8192"))
    parser.add_argument("--depths", type=parse_depth_csv, default=parse_depth_csv("0.5"))
    parser.add_argument("--t12-seeds", type=int, default=1)
    parser.add_argument("--t3-seeds", type=int, default=1)
    parser.add_argument("--json", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--prefill-microbatch-rows", type=int, default=2)
    parser.add_argument("--decode-microbatch-rows", type=int, default=16)
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--m-slots", type=int, default=64)
    parser.add_argument("--route-chunk", type=int, default=512)
    parser.add_argument("--token-pool-max-context-len", type=int, default=None)
    parser.add_argument("--token-pool-capacity", type=int, default=36_864)
    parser.add_argument("--persistent-padded-decode-steps", type=int, default=128)
    parser.add_argument("--graph-warmup-iters", type=int, default=1)
    parser.add_argument(
        "--native-gemma-attention-backend",
        choices=["manual", "manual_gqa", "sdpa", "sdpa_single_gqa", "triton_dense_gqa"],
        default="sdpa_single_gqa",
    )
    parser.add_argument(
        "--native-gemma-projection-backend",
        choices=["separate", "qkv_packed", "gate_up_packed", "qkv_gate_up_packed"],
        default="separate",
    )
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--require-winning-path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-full-grid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-benchmark-shape",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def normalize_args(args) -> None:
    if args.t12_seeds < 1 or args.t3_seeds < 1:
        raise ValueError("seed counts must be >= 1")
    if args.slots < 1:
        raise ValueError("slots must be >= 1")
    if args.prefill_microbatch_rows < 1:
        raise ValueError("prefill microbatch rows must be >= 1")
    if args.decode_microbatch_rows < 1:
        raise ValueError("decode microbatch rows must be >= 1")
    if min(args.chunk, args.sink, args.window, args.m_slots, args.route_chunk) < 1:
        raise ValueError("chunk and routed-cache dimensions must be >= 1")
    if args.token_pool_capacity < 1:
        raise ValueError("token pool capacity must be >= 1")
    if args.persistent_padded_decode_steps < 1:
        raise ValueError("persistent padded decode steps must be >= 1")
    if args.graph_warmup_iters < 0:
        raise ValueError("graph warmup iterations must be >= 0")
    if args.token_pool_max_context_len is None:
        args.token_pool_max_context_len = max(args.ctxs) + 256
    if args.token_pool_max_context_len < max(args.ctxs) + 48:
        raise ValueError("token pool max context length must cover prompt plus output")


def main() -> None:
    args = build_parser().parse_args()
    normalize_args(args)
    run(args)


if __name__ == "__main__":
    main()
