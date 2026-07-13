#!/usr/bin/env python
"""Unified direct-engine multi-turn Gemma benchmark.

The workload keeps ``B`` independent sessions active across deterministic
turns. WKVM advances parked session state with token deltas, while vLLM and
SGLang receive the equivalent cumulative token histories so their native
prefix caches can report actual cached-token counts.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bench_prompt_utils import (
    generated_output_fingerprint,
    prompt_set_fingerprint,
)


SCHEMA = "wkvm.gemma_multiturn_bench.v1"
PROMPT_TOKEN_SOURCE = "synthetic_lcg"


@dataclass(frozen=True)
class MultiTurnWorkload:
    initial_prompts: list[list[int]]
    turn_deltas: list[list[list[int]]]


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_tree_state() -> dict[str, Any]:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {
            "clean": None,
            "status_sha256": None,
            "changed_path_count": None,
        }
    lines = [line for line in status.splitlines() if line]
    return {
        "clean": not lines,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "changed_path_count": len(lines),
    }


def session_id(index: int) -> str:
    return f"session-{index:04d}"


def request_order_indices(
    session_count: int,
    turn_index: int,
    policy: str,
    seed: int = 0,
) -> list[int]:
    if session_count < 1:
        raise ValueError("session_count must be >= 1")
    if turn_index < 0:
        raise ValueError("turn_index must be >= 0")
    order = list(range(session_count))
    if policy == "forward":
        return order
    if policy == "alternating":
        return order if turn_index % 2 == 0 else list(reversed(order))
    if policy != "seeded-shuffle":
        raise ValueError(f"unknown request order policy {policy!r}")
    state = (
        int(seed)
        ^ ((turn_index + 1) * 0x9E3779B1)
        ^ (session_count * 0x85EBCA77)
    ) & 0xFFFF_FFFF
    for index in range(session_count - 1, 0, -1):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
        swap_index = state % (index + 1)
        order[index], order[swap_index] = order[swap_index], order[index]
    return order


def restore_logical_order(values: Sequence[Any], order: Sequence[int]) -> list[Any]:
    if len(values) != len(order):
        raise ValueError("ordered values and request order must have equal length")
    if sorted(int(index) for index in order) != list(range(len(order))):
        raise ValueError("request order must be a complete permutation")
    logical: list[Any] = [None] * len(order)
    for ordered_index, logical_index in enumerate(order):
        logical[int(logical_index)] = values[ordered_index]
    return logical


def deterministic_token_sequence(
    *,
    session_index: int,
    turn_index: int,
    token_count: int,
    vocab_size: int,
    include_bos: bool = False,
) -> list[int]:
    if session_index < 0:
        raise ValueError("session_index must be >= 0")
    if turn_index < -1:
        raise ValueError("turn_index must be >= -1")
    if token_count < 1:
        raise ValueError("token_count must be >= 1")
    if vocab_size < 16:
        raise ValueError("vocab_size must be >= 16")
    state = (
        0x6D2B79F5
        ^ ((session_index + 1) * 0x9E3779B1)
        ^ ((turn_index + 2) * 0x85EBCA77)
    ) & 0xFFFF_FFFF
    tokens: list[int] = []
    if include_bos:
        tokens.append(2)
    while len(tokens) < token_count:
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
        tokens.append(4 + state % (vocab_size - 4))
    return tokens


def build_workload(
    *,
    sessions: int,
    turns: int,
    initial_context_tokens: int,
    turn_input_tokens: int,
    vocab_size: int,
) -> MultiTurnWorkload:
    if sessions < 1:
        raise ValueError("sessions must be >= 1")
    if turns < 1:
        raise ValueError("turns must be >= 1")
    initial_prompts = [
        deterministic_token_sequence(
            session_index=index,
            turn_index=-1,
            token_count=initial_context_tokens,
            vocab_size=vocab_size,
            include_bos=True,
        )
        for index in range(sessions)
    ]
    turn_deltas = [
        [
            deterministic_token_sequence(
                session_index=index,
                turn_index=turn_index,
                token_count=turn_input_tokens,
                vocab_size=vocab_size,
            )
            for index in range(sessions)
        ]
        for turn_index in range(1, turns)
    ]
    return MultiTurnWorkload(
        initial_prompts=initial_prompts,
        turn_deltas=turn_deltas,
    )


def workload_fingerprints(workload: MultiTurnWorkload) -> dict[str, Any]:
    return {
        "initial_prompts": prompt_set_fingerprint(
            workload.initial_prompts,
            prompt_token_source=PROMPT_TOKEN_SOURCE,
        ),
        "turn_deltas": [
            prompt_set_fingerprint(
                deltas,
                prompt_token_source=PROMPT_TOKEN_SOURCE,
            )
            for deltas in workload.turn_deltas
        ],
    }


def _metric_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _metric_integer(value: Any) -> int | None:
    number = _metric_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def extract_vllm_cached_tokens(outputs: Sequence[Any]) -> list[int | None]:
    cached_tokens: list[int | None] = []
    for output in outputs:
        value = getattr(output, "num_cached_tokens", None)
        if value is None:
            metrics = getattr(output, "metrics", None)
            value = getattr(metrics, "num_cached_tokens", None)
            if value is None:
                value = getattr(metrics, "cached_tokens", None)
        cached_tokens.append(_metric_integer(value))
    return cached_tokens


def extract_sglang_cached_tokens(outputs: Sequence[Any]) -> list[int | None]:
    cached_tokens: list[int | None] = []
    for output in outputs:
        meta = output.get("meta_info") if isinstance(output, dict) else None
        value = meta.get("cached_tokens") if isinstance(meta, dict) else None
        cached_tokens.append(_metric_integer(value))
    return cached_tokens


def extract_vllm_latencies(
    outputs: Sequence[Any],
) -> tuple[list[float | None], list[float | None]]:
    ttfts: list[float | None] = []
    e2es: list[float | None] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        ttft = _metric_number(getattr(metrics, "first_token_latency", None))
        first_token = _metric_number(getattr(metrics, "first_token_ts", None))
        last_token = _metric_number(getattr(metrics, "last_token_ts", None))
        e2e = None
        if (
            ttft is not None
            and first_token is not None
            and last_token is not None
            and last_token >= first_token
        ):
            e2e = ttft + last_token - first_token
        ttfts.append(ttft)
        e2es.append(e2e)
    return ttfts, e2es


def extract_sglang_latencies(
    outputs: Sequence[Any],
) -> tuple[list[float | None], list[float | None]]:
    ttfts: list[float | None] = []
    e2es: list[float | None] = []
    for output in outputs:
        meta = output.get("meta_info") if isinstance(output, dict) else None
        meta = meta if isinstance(meta, dict) else {}
        ttft = None
        for field in ("first_token_latency", "time_to_first_token", "ttft"):
            ttft = _metric_number(meta.get(field))
            if ttft is not None:
                break
        e2e = _metric_number(meta.get("e2e_latency"))
        ttfts.append(ttft)
        e2es.append(e2e)
    return ttfts, e2es


def _normalized_optional_values(
    values: Sequence[Any] | None,
    count: int,
) -> list[Any]:
    if values is None:
        return [None] * count
    if len(values) != count:
        raise ValueError(f"expected {count} values, received {len(values)}")
    return list(values)


def summarize_turn(
    *,
    turn_index: int,
    session_ids: Sequence[str],
    prompts: Sequence[Sequence[int]],
    deltas: Sequence[Sequence[int]],
    outputs: Sequence[Sequence[int] | None],
    expected_output_tokens: int,
    new_input_tokens: Sequence[int],
    wall_s: float,
    ttft_s: Sequence[float | None] | None = None,
    e2e_s: Sequence[float | None] | None = None,
    cached_tokens: Sequence[int | None] | None = None,
    errors: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    request_count = len(session_ids)
    for name, values in (
        ("prompts", prompts),
        ("deltas", deltas),
        ("outputs", outputs),
        ("new_input_tokens", new_input_tokens),
    ):
        if len(values) != request_count:
            raise ValueError(
                f"{name} has {len(values)} rows, expected {request_count}"
            )
    if expected_output_tokens < 1:
        raise ValueError("expected_output_tokens must be >= 1")
    if wall_s < 0 or not math.isfinite(wall_s):
        raise ValueError("wall_s must be finite and >= 0")

    normalized_ttft = _normalized_optional_values(ttft_s, request_count)
    normalized_e2e = _normalized_optional_values(e2e_s, request_count)
    normalized_cached = _normalized_optional_values(cached_tokens, request_count)
    normalized_errors = _normalized_optional_values(errors, request_count)
    requests: list[dict[str, Any]] = []
    successful_outputs: list[tuple[str, list[int]]] = []
    successful_new_input_tokens = 0
    output_token_count = 0
    valid_ttft: list[float] = []
    valid_e2e: list[float] = []
    valid_cached: list[int] = []

    for index, request_id in enumerate(session_ids):
        token_ids = None if outputs[index] is None else [int(x) for x in outputs[index]]
        error = normalized_errors[index]
        if error is None and token_ids is None:
            error = "engine returned no output"
        if error is None and len(token_ids or ()) != expected_output_tokens:
            error = (
                f"expected {expected_output_tokens} output tokens, "
                f"received {len(token_ids or ())}"
            )
        success = error is None
        request_ttft = _metric_number(normalized_ttft[index])
        request_e2e = _metric_number(normalized_e2e[index])
        request_cached = _metric_integer(normalized_cached[index])
        if success:
            assert token_ids is not None
            successful_outputs.append((request_id, token_ids))
            output_token_count += len(token_ids)
            successful_new_input_tokens += int(new_input_tokens[index])
            if request_ttft is not None:
                valid_ttft.append(request_ttft)
            if request_e2e is not None:
                valid_e2e.append(request_e2e)
            if request_cached is not None:
                valid_cached.append(request_cached)
        requests.append(
            {
                "session_id": request_id,
                "success": success,
                "error": error,
                "prompt_tokens": len(prompts[index]),
                "delta_tokens": len(deltas[index]),
                "new_input_tokens": int(new_input_tokens[index]),
                "output_tokens": len(token_ids or ()),
                "ttft_s": round_or_none(request_ttft),
                "e2e_latency_s": round_or_none(request_e2e),
                "cached_tokens": request_cached,
            }
        )

    success_count = len(successful_outputs)
    useful_new_tokens = successful_new_input_tokens + output_token_count
    prompt_fingerprint = prompt_set_fingerprint(
        prompts,
        prompt_token_source=PROMPT_TOKEN_SOURCE,
    )
    delta_fingerprint = prompt_set_fingerprint(
        deltas,
        prompt_token_source=PROMPT_TOKEN_SOURCE,
    )
    output_fingerprint = generated_output_fingerprint(successful_outputs)
    return {
        "turn_index": turn_index,
        "request_count": request_count,
        "success_count": success_count,
        "error_count": request_count - success_count,
        "wall_s": round_or_none(wall_s),
        "logical_prompt_tokens": sum(len(prompt) for prompt in prompts),
        "offered_new_input_tokens": sum(int(value) for value in new_input_tokens),
        "successful_new_input_tokens": successful_new_input_tokens,
        "output_tokens": output_token_count,
        "useful_new_tokens": useful_new_tokens,
        "output_tok_s": round_or_none(
            output_token_count / wall_s if wall_s > 0 else None,
            3,
        ),
        "useful_new_token_tok_s": round_or_none(
            useful_new_tokens / wall_s if wall_s > 0 else None,
            3,
        ),
        "p50_ttft_s": round_or_none(percentile(valid_ttft, 0.50)),
        "p95_ttft_s": round_or_none(percentile(valid_ttft, 0.95)),
        "ttft_available_count": len(valid_ttft),
        "p50_e2e_latency_s": round_or_none(percentile(valid_e2e, 0.50)),
        "p95_e2e_latency_s": round_or_none(percentile(valid_e2e, 0.95)),
        "e2e_latency_available_count": len(valid_e2e),
        "cached_tokens_total": sum(valid_cached),
        "cached_tokens_available_count": len(valid_cached),
        "p50_cached_tokens": round_or_none(percentile(valid_cached, 0.50), 3),
        "p95_cached_tokens": round_or_none(percentile(valid_cached, 0.95), 3),
        "prompt_fingerprint": prompt_fingerprint,
        "prompt_token_ids_sha256": prompt_fingerprint[
            "prompt_token_ids_sha256"
        ],
        "delta_fingerprint": delta_fingerprint,
        "delta_token_ids_sha256": delta_fingerprint[
            "prompt_token_ids_sha256"
        ],
        "generated_output_fingerprint": output_fingerprint,
        "request_output_token_ids_sha256": output_fingerprint[
            "request_output_token_ids_sha256"
        ],
        "output_fingerprint_complete": success_count == request_count,
        "errors": [
            {
                "session_id": request["session_id"],
                "error": request["error"],
            }
            for request in requests
            if not request["success"]
        ],
        "requests": requests,
    }


def _aggregate_turn_rows(turn_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_wall = sum(float(row.get("wall_s") or 0.0) for row in turn_rows)
    total_output = sum(int(row.get("output_tokens") or 0) for row in turn_rows)
    total_useful = sum(int(row.get("useful_new_tokens") or 0) for row in turn_rows)
    total_success = sum(int(row.get("success_count") or 0) for row in turn_rows)
    total_errors = sum(int(row.get("error_count") or 0) for row in turn_rows)
    total_requests = sum(int(row.get("request_count") or 0) for row in turn_rows)
    cached_available = sum(
        int(row.get("cached_tokens_available_count") or 0) for row in turn_rows
    )
    all_ttft = [
        float(request["ttft_s"])
        for row in turn_rows
        for request in row.get("requests", ())
        if request.get("success") and request.get("ttft_s") is not None
    ]
    all_e2e = [
        float(request["e2e_latency_s"])
        for row in turn_rows
        for request in row.get("requests", ())
        if request.get("success") and request.get("e2e_latency_s") is not None
    ]
    return {
        "turn_rows": len(turn_rows),
        "request_count": total_requests,
        "success_count": total_success,
        "error_count": total_errors,
        "wall_s": round_or_none(total_wall),
        "wall_scope": "sum_of_synchronized_engine_turn_barriers",
        "output_tokens": total_output,
        "useful_new_tokens": total_useful,
        "output_tok_s": round_or_none(
            total_output / total_wall if total_wall > 0 else None,
            3,
        ),
        "completed_requests_per_s": round_or_none(
            total_success / total_wall if total_wall > 0 else None,
            3,
        ),
        "useful_new_token_tok_s": round_or_none(
            total_useful / total_wall if total_wall > 0 else None,
            3,
        ),
        "p50_ttft_s": round_or_none(percentile(all_ttft, 0.50)),
        "p95_ttft_s": round_or_none(percentile(all_ttft, 0.95)),
        "ttft_available_count": len(all_ttft),
        "p50_e2e_latency_s": round_or_none(percentile(all_e2e, 0.50)),
        "p95_e2e_latency_s": round_or_none(percentile(all_e2e, 0.95)),
        "e2e_latency_available_count": len(all_e2e),
        "cached_tokens_total": sum(
            int(row.get("cached_tokens_total") or 0) for row in turn_rows
        ),
        "cached_tokens_available_count": cached_available,
        "cache_telemetry_complete": cached_available == total_success,
    }


def summarize_run(turn_rows: Sequence[dict[str, Any]], requested_turns: int) -> dict[str, Any]:
    aggregate = _aggregate_turn_rows(turn_rows)
    return {
        "requested_turns": requested_turns,
        "completed_turn_rows": len(turn_rows),
        "all_turns_recorded": len(turn_rows) == requested_turns,
        **aggregate,
        "turn_0": _aggregate_turn_rows(turn_rows[:1]),
        "continuation_turns": _aggregate_turn_rows(turn_rows[1:]),
    }


def _append_deltas(
    histories: list[list[int]],
    deltas: Sequence[Sequence[int]],
) -> list[list[int]]:
    for history, delta in zip(histories, deltas, strict=True):
        history.extend(int(token) for token in delta)
    return [list(history) for history in histories]


def _turn_prompts_and_deltas(
    workload: MultiTurnWorkload,
    histories: list[list[int]],
    turn_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if turn_index == 0:
        return [list(history) for history in histories], [
            [] for _ in histories
        ]
    deltas = workload.turn_deltas[turn_index - 1]
    return _append_deltas(histories, deltas), deltas


def _append_outputs(
    histories: list[list[int]],
    outputs: Sequence[Sequence[int] | None],
) -> None:
    for history, output in zip(histories, outputs, strict=True):
        if output is not None:
            history.extend(int(token) for token in output)


def _turn_errors_for_outputs(
    outputs: Sequence[Sequence[int] | None],
    expected_output_tokens: int,
) -> list[str | None]:
    errors: list[str | None] = []
    for output in outputs:
        if output is None:
            errors.append("engine returned no output")
        elif len(output) != expected_output_tokens:
            errors.append(
                f"expected {expected_output_tokens} output tokens, "
                f"received {len(output)}"
            )
        else:
            errors.append(None)
    return errors


def _print_turn(engine: str, row: dict[str, Any]) -> None:
    print(
        f"[{engine} turn={row['turn_index']} B={row['request_count']}] "
        f"success={row['success_count']}/{row['request_count']} "
        f"wall={row['wall_s']}s output={row['output_tok_s']}tok/s "
        f"useful={row['useful_new_token_tok_s']}tok/s "
        f"cached={row['cached_tokens_total']}"
    )


def _wkvm_stats_snapshot(engine: Any) -> dict[str, Any]:
    stats = engine.stats()
    keys = (
        "queue_depth",
        "runnable_rows",
        "parked_sessions",
        "resident_sessions",
        "resident_state_slots",
        "free_state_slots",
        "active_cache_bytes",
        "state_bytes_per_request",
        "max_waiting",
        "max_running",
        "max_runnable_rows",
        "max_resident_state_slots",
        "cache_builds",
        "sessions_opened",
        "session_turns_completed",
        "sessions_closed",
        "session_reuse_hits",
        "session_reuse_misses",
        "continuation_input_tokens_computed",
        "prefix_tokens_reused",
        "full_reprefill_turns",
        "session_sliding_tail_restores",
        "session_sliding_tail_tokens_restored",
        "max_resident_sessions",
        "backpressure_events",
        "retraction_events",
        "token_pool_slot_high_watermark",
        "model_forward_backend",
        "uses_hf_transformer_forward",
        "uses_hf_model_construction",
        "native_gemma_checkpoint_loader",
    )
    snapshot = {key: stats.get(key) for key in keys}
    snapshot["state"] = stats.get("state")
    snapshot["token_pool"] = stats.get("token_pool")
    snapshot["gpu_memory"] = stats.get("gpu_memory")
    return snapshot


def _wkvm_reuse_invariants(
    stats: dict[str, Any],
    *,
    sessions: int,
    turn_index: int,
    turn_input_tokens: int,
) -> dict[str, Any]:
    token_pool = stats.get("token_pool") or {}
    checks = {
        "resident_sessions": stats.get("resident_sessions") == sessions,
        "parked_sessions": stats.get("parked_sessions") == sessions,
        "cache_builds": stats.get("cache_builds") == sessions,
        "token_pool_active_requests": (
            token_pool.get("active_request_slots") == sessions
        ),
        "session_turns_completed": (
            stats.get("session_turns_completed") == sessions * (turn_index + 1)
        ),
        "session_reuse_hits": (
            stats.get("session_reuse_hits") == sessions * turn_index
        ),
        "continuation_input_tokens_computed": (
            stats.get("continuation_input_tokens_computed")
            == sessions * turn_index * (turn_input_tokens + 1)
        ),
        "full_reprefill_turns": stats.get("full_reprefill_turns") == 0,
        "session_sliding_tail_restores": (
            stats.get("session_sliding_tail_restores") == sessions * turn_index
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def run_wkvm(args: argparse.Namespace, workload: MultiTurnWorkload) -> dict[str, Any]:
    import torch

    from bench_prompt_utils import SyntheticBenchTokenizer
    from native_gemma_bench import (
        apply_token_pool_triton_bench_env,
        build_native_config,
        make_engine,
    )
    from native_gemma_smoke import break_mask_for, load_model
    from wkvm.core.request import Request

    args.slots = args.slots or args.sessions
    args.token_pool_max_context_len = (
        args.token_pool_max_context_len or args.required_model_len + 16
    )
    args.token_pool_capacity = (
        args.token_pool_capacity or args.sessions * 2048
    )
    token_pool_environment = apply_token_pool_triton_bench_env(args)
    model = load_model(
        args.model_path,
        args.device,
        args.attn,
        native_checkpoint_loader=args.native_gemma_checkpoint_loader,
        native_gemma_attention_backend=args.native_gemma_attention_backend,
        native_gemma_projection_backend=args.native_gemma_projection_backend,
    )
    config = build_native_config(model, args)
    engine = make_engine(model, config, workload.initial_prompts, args)
    tokenizer = SyntheticBenchTokenizer(vocab_size=args.synthetic_vocab_size)
    histories = [list(prompt) for prompt in workload.initial_prompts]
    request_ids = [session_id(index) for index in range(args.sessions)]
    requests: dict[str, Any] = {}
    turn_rows: list[dict[str, Any]] = []
    turn_prompt_sets: list[list[list[int]]] = []
    turn_output_sets: list[list[list[int]]] = []
    before_close: dict[str, Any] | None = None
    after_close: dict[str, Any] | None = None
    fresh_parity: dict[str, Any] | None = None
    closed = False
    try:
        for turn_index in range(args.turns):
            prompts, deltas = _turn_prompts_and_deltas(
                workload,
                histories,
                turn_index,
            )
            request_order = request_order_indices(
                args.sessions,
                turn_index,
                args.request_order_policy,
                args.request_order_seed,
            )
            turn_prompt_sets.append([list(prompt) for prompt in prompts])
            started = time.perf_counter()
            if turn_index == 0:
                for index in request_order:
                    request_id = request_ids[index]
                    request = Request(
                        prompt_token_ids=list(prompts[index]),
                        max_new_tokens=args.output_tokens_per_turn,
                        req_id=request_id,
                    )
                    requests[request_id] = request
                    engine.add_session_request(
                        request,
                        break_mask=break_mask_for(tokenizer, prompts[index]),
                    )
            else:
                continuations = {
                    request_ids[index]: list(deltas[index])
                    for index in request_order
                }
                break_masks = {
                    request_ids[index]: break_mask_for(tokenizer, prompts[index])
                    for index in request_order
                }
                engine.continue_session_requests(
                    continuations,
                    max_new_tokens=args.output_tokens_per_turn,
                    break_masks=break_masks,
                )
            for index, request_id in enumerate(request_ids):
                if list(requests[request_id].prompt_token_ids) != prompts[index]:
                    raise RuntimeError(
                        f"{request_id} logical prompt diverged before turn {turn_index}"
                    )
            steps_before = engine.metrics.steps
            while engine.has_unfinished:
                engine.step()
                if engine.metrics.steps - steps_before > args.max_steps:
                    raise RuntimeError(
                        f"WKVM turn {turn_index} did not converge within "
                        f"{args.max_steps} steps"
                    )
            wall_s = time.perf_counter() - started
            outputs = [
                [int(token) for token in requests[request_id].output_token_ids]
                for request_id in request_ids
            ]
            turn_output_sets.append([list(output) for output in outputs])
            ttfts: list[float | None] = []
            e2es: list[float | None] = []
            reused_prefix_tokens: list[int] = []
            computed_input_tokens: list[int] = []
            errors = _turn_errors_for_outputs(
                outputs,
                args.output_tokens_per_turn,
            )
            for index, request_id in enumerate(request_ids):
                request = requests[request_id]
                if request.status.name != "PARKED" and errors[index] is None:
                    errors[index] = f"unexpected request status {request.status.name}"
                trace = engine.finished_traces.get(request_id)
                trace_values = trace.as_dict() if trace is not None else {}
                ttfts.append(trace_values.get("first_token_latency_s"))
                e2es.append(trace_values.get("total_latency_s"))
                reused_prefix_tokens.append(
                    int(trace_values.get("reused_prefix_tokens") or 0)
                )
                computed_input_tokens.append(
                    int(trace_values.get("computed_input_tokens") or 0)
                )
                if trace_values.get("error") and errors[index] is None:
                    errors[index] = str(trace_values["error"])
            row = summarize_turn(
                turn_index=turn_index,
                session_ids=request_ids,
                prompts=prompts,
                deltas=deltas,
                outputs=outputs,
                expected_output_tokens=args.output_tokens_per_turn,
                new_input_tokens=(
                    [len(prompt) for prompt in prompts]
                    if turn_index == 0
                    else [len(delta) for delta in deltas]
                ),
                wall_s=wall_s,
                ttft_s=ttfts,
                e2e_s=e2es,
                cached_tokens=reused_prefix_tokens,
                errors=errors,
            )
            row["reuse_kind"] = "wkvm_parked_session_state"
            row["request_order_policy"] = args.request_order_policy
            row["request_order"] = [request_ids[index] for index in request_order]
            row["cached_tokens_source"] = (
                "GemmaRequestTrace.reused_prefix_tokens"
            )
            row["reused_prefix_tokens_total"] = sum(reused_prefix_tokens)
            row["computed_input_tokens_total"] = sum(computed_input_tokens)
            for request_row, reused_tokens, computed_tokens in zip(
                row["requests"],
                reused_prefix_tokens,
                computed_input_tokens,
                strict=True,
            ):
                request_row["reused_prefix_tokens"] = reused_tokens
                request_row["computed_input_tokens"] = computed_tokens
            barrier_stats = _wkvm_stats_snapshot(engine)
            row["engine_metrics_at_barrier"] = barrier_stats
            row["reuse_invariants"] = _wkvm_reuse_invariants(
                barrier_stats,
                sessions=args.sessions,
                turn_index=turn_index,
                turn_input_tokens=args.turn_input_tokens,
            )
            turn_rows.append(row)
            _print_turn("wkvm", row)
            if not row["reuse_invariants"]["passed"]:
                raise RuntimeError(
                    f"WKVM turn {turn_index} reuse invariants failed: "
                    f"{row['reuse_invariants']['checks']}"
                )
            _append_outputs(histories, outputs)
            if row["error_count"]:
                break
        before_close = _wkvm_stats_snapshot(engine)
        engine.close_sessions(request_ids)
        closed = True
        after_close = _wkvm_stats_snapshot(engine)
        if (
            after_close.get("resident_sessions") != 0
            or after_close.get("parked_sessions") != 0
            or (after_close.get("token_pool") or {}).get("active_request_slots") != 0
            or after_close.get("sessions_closed") != args.sessions
        ):
            raise RuntimeError(
                f"WKVM session close invariants failed: {after_close}"
            )
        if args.wkvm_verify_fresh_parity:
            parity_turns: list[dict[str, Any]] = []
            for turn_index, (prompts, expected_outputs) in enumerate(
                zip(turn_prompt_sets, turn_output_sets, strict=True)
            ):
                parity_requests = []
                parity_started = time.perf_counter()
                for index, prompt in enumerate(prompts):
                    request = Request(
                        prompt_token_ids=list(prompt),
                        max_new_tokens=args.output_tokens_per_turn,
                        req_id=f"fresh-{turn_index}-{index}",
                    )
                    parity_requests.append(request)
                    engine.add_request(
                        request,
                        break_mask=break_mask_for(tokenizer, prompt),
                    )
                steps_before = engine.metrics.steps
                while engine.has_unfinished:
                    engine.step()
                    if engine.metrics.steps - steps_before > args.max_steps:
                        raise RuntimeError(
                            f"WKVM fresh parity turn {turn_index} did not converge"
                        )
                parity_wall_s = time.perf_counter() - parity_started
                actual_outputs = [
                    [int(token) for token in request.output_token_ids]
                    for request in parity_requests
                ]
                exact_rows = [
                    actual == expected
                    for actual, expected in zip(
                        actual_outputs,
                        expected_outputs,
                        strict=True,
                    )
                ]
                parity_turns.append(
                    {
                        "turn_index": turn_index,
                        "exact_rows": sum(exact_rows),
                        "request_count": args.sessions,
                        "passed": all(exact_rows),
                        "wall_s": round_or_none(parity_wall_s),
                        "prompt_fingerprint": prompt_set_fingerprint(
                            prompts,
                            prompt_token_source=PROMPT_TOKEN_SOURCE,
                        ),
                        "stateful_output_fingerprint": generated_output_fingerprint(
                            zip(request_ids, expected_outputs, strict=True)
                        ),
                        "fresh_output_fingerprint": generated_output_fingerprint(
                            zip(request_ids, actual_outputs, strict=True)
                        ),
                    }
                )
            fresh_parity = {
                "enabled": True,
                "passed": all(row["passed"] for row in parity_turns),
                "turns": parity_turns,
            }
            if not fresh_parity["passed"]:
                raise RuntimeError(f"WKVM fresh-history parity failed: {fresh_parity}")
    finally:
        if not closed and requests:
            with contextlib.suppress(Exception):
                engine.close_sessions(list(requests))
        del engine
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    engine_config = {
        "slots": args.slots,
        "token_budget": args.token_budget,
        "chunk": args.chunk,
        "prefill_microbatch_rows": args.prefill_microbatch_rows,
        "decode_microbatch_rows": args.decode_microbatch_rows,
        "decode_microbatch_bytes": args.decode_microbatch_bytes,
        "decode_batch_planner": args.decode_batch_planner,
        "decode_workspace_bytes": args.decode_workspace_bytes,
        "decode_workspace_width_bucket": args.decode_workspace_width_bucket,
        "persistent_exact_decode": not args.disable_persistent_exact_decode,
        "persistent_padded_decode": not args.disable_persistent_padded_decode,
        "persistent_padded_decode_steps": args.persistent_padded_decode_steps,
        "persistent_padded_full_attention_rows": (
            args.persistent_padded_full_attention_rows
        ),
        "persistent_padded_sliding_metadata_padding": (
            args.persistent_padded_sliding_metadata_padding
        ),
        "persistent_padded_decode_cuda_graph": (
            args.persistent_padded_decode_cuda_graph
        ),
        "persistent_padded_decode_graph_warmup_iters": (
            args.persistent_padded_decode_graph_warmup_iters
        ),
        "native_gemma_checkpoint_loader": args.native_gemma_checkpoint_loader,
        "use_native_gemma_forward": args.use_native_gemma_forward,
        "native_gemma_attention_backend": args.native_gemma_attention_backend,
        "native_gemma_projection_backend": args.native_gemma_projection_backend,
        "native_gemma_weight_backend": args.native_gemma_weight_backend,
        "enable_token_pool_metadata": args.enable_token_pool_metadata,
        "enable_token_pool_attention": args.enable_token_pool_attention,
        "token_pool_max_context_len": args.token_pool_max_context_len,
        "token_pool_capacity": args.token_pool_capacity,
        "token_pool_paged_block_size": args.token_pool_paged_block_size,
        "sink": args.sink,
        "window": args.window,
        "m_slots": args.m_slots,
        "route_chunk": args.route_chunk,
        "verify_fresh_history_parity": args.wkvm_verify_fresh_parity,
        "request_order_policy": args.request_order_policy,
        "request_order_seed": args.request_order_seed,
        "device": args.device,
        "dtype": "bfloat16",
    }
    return {
        "turns": turn_rows,
        "engine_config": engine_config,
        "engine_version": None,
        "launch_environment": token_pool_environment,
        "engine_metrics_before_close": before_close,
        "engine_metrics_after_close": after_close,
        "fresh_history_parity": fresh_parity,
    }


def _vllm_turn_outputs(raw_outputs: Sequence[Any]) -> list[list[int] | None]:
    outputs: list[list[int] | None] = []
    for output in raw_outputs:
        choices = getattr(output, "outputs", None)
        if not choices:
            outputs.append(None)
            continue
        token_ids = getattr(choices[0], "token_ids", None)
        outputs.append(
            None if token_ids is None else [int(token) for token in token_ids]
        )
    return outputs


def run_vllm(args: argparse.Namespace, workload: MultiTurnWorkload) -> dict[str, Any]:
    import vllm
    from vllm import LLM, SamplingParams

    from incumbent_gemma_bench import (
        cleanup_cuda,
        synchronize_cuda,
        vllm_capacity_telemetry,
    )

    max_model_len = args.max_model_len or args.required_model_len + 16
    kwargs: dict[str, Any] = {
        "model": args.model_path,
        "max_model_len": max_model_len,
        "max_num_seqs": args.sessions,
        "gpu_memory_utilization": args.vllm_gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": True,
        "swap_space": 0,
        "disable_log_stats": False,
        "dtype": "bfloat16",
    }
    if args.vllm_language_model_only:
        kwargs["language_model_only"] = True
    compilation_config = None
    if args.vllm_disable_inductor:
        capture_sizes = sorted({1, 2, 4, 8, 16, args.sessions})
        compilation_config = {
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": max(capture_sizes),
        }
        kwargs["compilation_config"] = compilation_config
    try:
        llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "audio": 0})
        multimodal_config = {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    except TypeError:
        llm = LLM(**kwargs)
        multimodal_config = {"limit_mm_per_prompt": "unsupported"}
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.output_tokens_per_turn,
        ignore_eos=True,
        seed=0,
    )
    histories = [list(prompt) for prompt in workload.initial_prompts]
    request_ids = [session_id(index) for index in range(args.sessions)]
    turn_rows: list[dict[str, Any]] = []
    capacity = vllm_capacity_telemetry(llm, max_model_len=max_model_len)
    try:
        for turn_index in range(args.turns):
            prompts, deltas = _turn_prompts_and_deltas(
                workload,
                histories,
                turn_index,
            )
            request_order = request_order_indices(
                args.sessions,
                turn_index,
                args.request_order_policy,
                args.request_order_seed,
            )
            requests = [
                {"prompt_token_ids": prompts[index]}
                for index in request_order
            ]
            synchronize_cuda()
            started = time.perf_counter()
            raw_outputs = llm.generate(requests, sampling, use_tqdm=False)
            synchronize_cuda()
            wall_s = time.perf_counter() - started
            if len(raw_outputs) != args.sessions:
                outputs: list[list[int] | None] = [None] * args.sessions
                ttfts: list[float | None] = [None] * args.sessions
                e2es: list[float | None] = [None] * args.sessions
                cached: list[int | None] = [None] * args.sessions
                errors: list[str | None] = [
                    f"vLLM returned {len(raw_outputs)} outputs for {args.sessions} prompts"
                ] * args.sessions
            else:
                outputs = restore_logical_order(
                    _vllm_turn_outputs(raw_outputs),
                    request_order,
                )
                ordered_ttfts, ordered_e2es = extract_vllm_latencies(raw_outputs)
                ttfts = restore_logical_order(ordered_ttfts, request_order)
                e2es = restore_logical_order(ordered_e2es, request_order)
                cached = restore_logical_order(
                    extract_vllm_cached_tokens(raw_outputs),
                    request_order,
                )
                errors = _turn_errors_for_outputs(
                    outputs,
                    args.output_tokens_per_turn,
                )
            row = summarize_turn(
                turn_index=turn_index,
                session_ids=request_ids,
                prompts=prompts,
                deltas=deltas,
                outputs=outputs,
                expected_output_tokens=args.output_tokens_per_turn,
                new_input_tokens=(
                    [len(prompt) for prompt in prompts]
                    if turn_index == 0
                    else [len(delta) for delta in deltas]
                ),
                wall_s=wall_s,
                ttft_s=ttfts,
                e2e_s=e2es,
                cached_tokens=cached,
                errors=errors,
            )
            row["reuse_kind"] = "vllm_prefix_cache"
            row["request_order_policy"] = args.request_order_policy
            row["request_order"] = [request_ids[index] for index in request_order]
            row["cached_tokens_source"] = "RequestOutput.num_cached_tokens"
            turn_rows.append(row)
            _print_turn("vllm", row)
            _append_outputs(histories, outputs)
            if row["error_count"]:
                break
    finally:
        with contextlib.suppress(Exception):
            del llm
        cleanup_cuda()
    kwargs.update(multimodal_config)
    return {
        "turns": turn_rows,
        "engine_config": {
            **kwargs,
            "compilation_config": compilation_config,
            "prefix_caching": True,
            "request_order_policy": args.request_order_policy,
            "request_order_seed": args.request_order_seed,
            "capacity_telemetry": capacity,
        },
        "engine_version": getattr(vllm, "__version__", "unknown"),
        "launch_environment": {},
    }


def _sglang_items(raw_output: Any, expected: int) -> list[dict[str, Any]]:
    if expected == 1 and isinstance(raw_output, dict):
        return [raw_output]
    if not isinstance(raw_output, list):
        raise TypeError(
            f"unexpected SGLang output type: {type(raw_output).__name__}"
        )
    if len(raw_output) != expected:
        raise RuntimeError(
            f"SGLang returned {len(raw_output)} outputs for {expected} prompts"
        )
    if not all(isinstance(item, dict) for item in raw_output):
        raise TypeError("SGLang batch output contains a non-object item")
    return raw_output


def _sglang_output_tokens(output: dict[str, Any]) -> list[int] | None:
    for container in (output, output.get("meta_info") or {}):
        if not isinstance(container, dict):
            continue
        for field in ("output_ids", "output_token_ids"):
            if container.get(field) is not None:
                return [int(token) for token in container[field]]
    return None


def run_sglang(args: argparse.Namespace, workload: MultiTurnWorkload) -> dict[str, Any]:
    import sglang as sgl

    from incumbent_gemma_bench import (
        cleanup_cuda,
        sglang_capacity_telemetry,
        sglang_language_model_override,
        synchronize_cuda,
    )

    context_length = args.sglang_context_length or args.required_model_len + 16
    kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "mem_fraction_static": args.sglang_mem_fraction,
        "context_length": context_length,
        "max_total_tokens": args.sglang_max_total_tokens,
        "disable_radix_cache": False,
        "enable_multimodal": False,
        "log_level": args.sglang_log_level,
        "max_running_requests": args.sglang_max_running_requests or args.sessions,
        "cuda_graph_backend_decode": args.sglang_decode_graph,
        "cuda_graph_backend_prefill": args.sglang_prefill_graph,
    }
    if args.sglang_attention_backend:
        kwargs["attention_backend"] = args.sglang_attention_backend
    model_override = None
    if args.sglang_language_model_only:
        model_override = sglang_language_model_override(args.model_path)
        kwargs["json_model_override_args"] = json.dumps(model_override)
    engine = sgl.Engine(**kwargs)
    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": args.output_tokens_per_turn,
        "ignore_eos": True,
    }
    histories = [list(prompt) for prompt in workload.initial_prompts]
    request_ids = [session_id(index) for index in range(args.sessions)]
    turn_rows: list[dict[str, Any]] = []
    capacity = sglang_capacity_telemetry(engine)
    try:
        for turn_index in range(args.turns):
            prompts, deltas = _turn_prompts_and_deltas(
                workload,
                histories,
                turn_index,
            )
            request_order = request_order_indices(
                args.sessions,
                turn_index,
                args.request_order_policy,
                args.request_order_seed,
            )
            synchronize_cuda()
            started = time.perf_counter()
            raw_output = engine.generate(
                input_ids=[prompts[index] for index in request_order],
                sampling_params=sampling,
            )
            synchronize_cuda()
            wall_s = time.perf_counter() - started
            try:
                items = _sglang_items(raw_output, args.sessions)
                outputs = restore_logical_order(
                    [_sglang_output_tokens(item) for item in items],
                    request_order,
                )
                ordered_ttfts, ordered_e2es = extract_sglang_latencies(items)
                ttfts = restore_logical_order(ordered_ttfts, request_order)
                e2es = restore_logical_order(ordered_e2es, request_order)
                cached = restore_logical_order(
                    extract_sglang_cached_tokens(items),
                    request_order,
                )
                errors = _turn_errors_for_outputs(
                    outputs,
                    args.output_tokens_per_turn,
                )
            except Exception as exc:
                outputs = [None] * args.sessions
                ttfts = [None] * args.sessions
                e2es = [None] * args.sessions
                cached = [None] * args.sessions
                errors = [str(exc).splitlines()[0]] * args.sessions
            row = summarize_turn(
                turn_index=turn_index,
                session_ids=request_ids,
                prompts=prompts,
                deltas=deltas,
                outputs=outputs,
                expected_output_tokens=args.output_tokens_per_turn,
                new_input_tokens=(
                    [len(prompt) for prompt in prompts]
                    if turn_index == 0
                    else [len(delta) for delta in deltas]
                ),
                wall_s=wall_s,
                ttft_s=ttfts,
                e2e_s=e2es,
                cached_tokens=cached,
                errors=errors,
            )
            row["reuse_kind"] = "sglang_radix_cache"
            row["request_order_policy"] = args.request_order_policy
            row["request_order"] = [request_ids[index] for index in request_order]
            row["cached_tokens_source"] = "meta_info.cached_tokens"
            turn_rows.append(row)
            _print_turn("sglang", row)
            _append_outputs(histories, outputs)
            if row["error_count"]:
                break
    finally:
        with contextlib.suppress(Exception):
            engine.shutdown()
        cleanup_cuda()
    return {
        "turns": turn_rows,
        "engine_config": {
            **kwargs,
            "disable_radix_cache": False,
            "request_order_policy": args.request_order_policy,
            "request_order_seed": args.request_order_seed,
            "model_override": model_override,
            "capacity_telemetry": capacity,
        },
        "engine_version": getattr(sgl, "__version__", "unknown"),
        "launch_environment": {},
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "sessions",
        "turns",
        "initial_context_tokens",
        "turn_input_tokens",
        "output_tokens_per_turn",
        "synthetic_vocab_size",
        "max_steps",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.synthetic_vocab_size < 16:
        raise ValueError("--synthetic-vocab-size must be >= 16")
    if args.gpu_memory_sample_interval_s <= 0:
        raise ValueError("--gpu-memory-sample-interval-s must be > 0")
    if args.slots is not None and args.slots < args.sessions:
        raise ValueError("--slots must be >= --sessions")
    args.required_model_len = (
        args.initial_context_tokens
        + args.turns * args.output_tokens_per_turn
        + (args.turns - 1) * args.turn_input_tokens
    )
    if args.max_model_len is not None and args.max_model_len < args.required_model_len:
        raise ValueError(
            f"--max-model-len must be >= {args.required_model_len}"
        )
    if (
        args.sglang_context_length is not None
        and args.sglang_context_length < args.required_model_len
    ):
        raise ValueError(
            f"--sglang-context-length must be >= {args.required_model_len}"
        )


def build_payload(
    args: argparse.Namespace,
    workload: MultiTurnWorkload,
    result: dict[str, Any],
    gpu_memory: dict[str, Any],
    *,
    fatal_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turns = result.get("turns", [])
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "engine": args.engine,
        "engine_version": result.get("engine_version"),
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "prompt_token_source": PROMPT_TOKEN_SOURCE,
        "workload": {
            "sessions": args.sessions,
            "turns": args.turns,
            "initial_context_tokens": args.initial_context_tokens,
            "turn_input_tokens": args.turn_input_tokens,
            "output_tokens_per_turn": args.output_tokens_per_turn,
            "required_model_len": args.required_model_len,
            "history_policy": (
                "parked_state_plus_delta"
                if args.engine == "wkvm"
                else "cumulative_full_token_history"
            ),
            "request_order_policy": args.request_order_policy,
            "request_order_seed": args.request_order_seed,
            "fingerprints": workload_fingerprints(workload),
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "max_output_tokens_per_turn": args.output_tokens_per_turn,
        },
        "engine_config": result.get("engine_config", {}),
        "gpu_memory": gpu_memory,
        "git_commit": git_commit(),
        "git_tree_state": git_tree_state(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "launch_config": {
            "argv": [sys.executable, *sys.argv],
            "working_directory": os.getcwd(),
            "environment": result.get("launch_environment", {}),
        },
        "turns": turns,
        "summary": summarize_run(turns, args.turns),
    }
    for field in (
        "engine_metrics_before_close",
        "engine_metrics_after_close",
        "fresh_history_parity",
    ):
        if field in result:
            payload[field] = result[field]
    if fatal_error is not None:
        payload["fatal_error"] = fatal_error
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    from native_gemma_smoke import resolve_model_path
    from wkvm_serving_bench import WholeGpuMemoryMonitor

    validate_args(args)
    args.synthetic_prompts = True
    args.model_path = resolve_model_path(args.model_path)
    workload = build_workload(
        sessions=args.sessions,
        turns=args.turns,
        initial_context_tokens=args.initial_context_tokens,
        turn_input_tokens=args.turn_input_tokens,
        vocab_size=args.synthetic_vocab_size,
    )
    runners = {
        "wkvm": run_wkvm,
        "vllm": run_vllm,
        "sglang": run_sglang,
    }
    monitor = WholeGpuMemoryMonitor(
        str(args.gpu_memory_device),
        float(args.gpu_memory_sample_interval_s),
    )
    result: dict[str, Any] = {"turns": [], "engine_config": {}}
    fatal_error = None
    pending_error: BaseException | None = None
    with monitor:
        try:
            result = runners[args.engine](args, workload)
        except BaseException as exc:
            pending_error = exc
            fatal_error = {
                "type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
                "phase": "engine_run",
            }
    payload = build_payload(
        args,
        workload,
        result,
        monitor.result(),
        fatal_error=fatal_error,
    )
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if pending_error is not None:
        raise pending_error
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["wkvm", "vllm", "sglang"], required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--initial-context-tokens", type=int, default=13_824)
    parser.add_argument("--turn-input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens-per-turn", type=int, default=128)
    parser.add_argument("--synthetic-vocab-size", type=int, default=262_144)
    parser.add_argument("--gpu-memory-device", default="0")
    parser.add_argument("--gpu-memory-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--json", default=None)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument(
        "--request-order-policy",
        choices=["forward", "alternating", "seeded-shuffle"],
        default="alternating",
        help=(
            "Submission order within each turn. Alternating reverses odd turns "
            "to avoid deterministic forward-scan cache thrashing."
        ),
    )
    parser.add_argument("--request-order-seed", type=int, default=0)
    parser.add_argument(
        "--wkvm-verify-fresh-parity",
        action="store_true",
        help=(
            "After closing parked sessions, rerun every turn from its full "
            "history and require token-exact outputs. Intended for B1/B3 gates."
        ),
    )

    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--prefill-microbatch-rows", type=int, default=2)
    parser.add_argument("--decode-microbatch-rows", type=int, default=16)
    parser.add_argument("--decode-microbatch-bytes", type=int, default=None)
    parser.add_argument("--decode-batch-planner", choices=["scheduler", "length_bucketed"], default="scheduler")
    parser.add_argument("--decode-workspace-bytes", type=int, default=None)
    parser.add_argument("--decode-workspace-width-bucket", type=int, default=16)
    parser.add_argument("--disable-persistent-exact-decode", action="store_true")
    parser.add_argument("--disable-persistent-padded-decode", action="store_true")
    parser.add_argument("--persistent-padded-decode-steps", type=int, default=128)
    parser.add_argument("--persistent-padded-full-attention-rows", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-padded-sliding-metadata-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-padded-decode-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-padded-decode-graph-warmup-iters", type=int, default=0)
    parser.add_argument("--cuda-phase-metrics", action="store_true")
    parser.add_argument("--use-native-gemma-forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-gemma-checkpoint-loader", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-gemma-attention-backend", choices=["manual", "manual_gqa", "sdpa", "sdpa_single_gqa", "triton_dense_gqa"], default="sdpa_single_gqa")
    parser.add_argument("--native-gemma-projection-backend", choices=["separate", "qkv_packed", "gate_up_packed", "qkv_gate_up_packed"], default="separate")
    parser.add_argument("--native-gemma-weight-backend", choices=["hf_live", "owned", "owned_cpu"], default="hf_live")
    parser.add_argument("--native-gemma-release-hf-decoder-layers", action="store_true")
    parser.add_argument("--enable-token-pool-metadata", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-token-pool-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-pool-max-context-len", type=int, default=None)
    parser.add_argument("--token-pool-capacity", type=int, default=None)
    parser.add_argument("--token-pool-paged-block-size", type=int, default=16)
    parser.add_argument("--enable-token-pool-triton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-token-pool-paged-triton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-token-pool-paged-split-triton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-pool-triton-strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-pool-sliding-paged-metadata-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument("--sink", type=int, default=16)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--m-slots", type=int, default=64)
    parser.add_argument("--route-chunk", type=int, default=512)

    parser.add_argument("--vllm-gpu-mem-util", type=float, default=0.74)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--vllm-language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm-disable-inductor", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--sglang-mem-fraction", type=float, default=0.82)
    parser.add_argument("--sglang-context-length", type=int, default=None)
    parser.add_argument("--sglang-max-total-tokens", type=int, default=None)
    parser.add_argument("--sglang-attention-backend", default="triton")
    parser.add_argument("--sglang-language-model-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sglang-max-running-requests", type=int, default=None)
    parser.add_argument("--sglang-decode-graph", default="full")
    parser.add_argument("--sglang-prefill-graph", default="disabled")
    parser.add_argument("--sglang-log-level", default="warning")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
