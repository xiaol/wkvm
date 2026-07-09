#!/usr/bin/env python
"""Observability smoke for the native Gemma wkvm endpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_server_smoke import get_json, make_engine, post_json

from wkvm.gemma_server import BoundedGemmaService, serve


REQUEST_METRIC_KEYS = (
    "req_id",
    "prompt_tokens",
    "target_output_tokens",
    "output_tokens",
    "finish_reason",
    "error",
    "queue_time_s",
    "prefill_time_s",
    "decode_time_s",
    "first_token_latency_s",
    "total_latency_s",
)

ENGINE_METRIC_KEYS = (
    "steps",
    "scheduled_tokens",
    "admitted_requests",
    "finished_requests",
    "error_count",
    "prefill_calls",
    "decode_batches",
    "decode_rows",
    "max_decode_batch_rows",
    "distinct_history_decode_batches",
    "max_waiting",
    "max_running",
    "max_runnable_rows",
    "max_resident_state_slots",
    "max_active_cache_bytes",
    "backpressure_events",
    "retraction_events",
    "backpressure_reasons",
    "queue_depth",
    "runnable_rows",
    "resident_state_slots",
    "free_state_slots",
    "active_cache_bytes",
    "state_bytes_per_request",
    "gpu_memory",
    "state",
    "requests",
)

SERVER_METRIC_KEYS = (
    "ready",
    "total_requests",
    "total_errors",
    "total_cancelled",
    "max_queue",
    "batch_wait_s",
    "worker_alive",
    "last_error",
)

STEP_METRIC_KEYS = (
    "scheduled_rows_max",
    "token_budget_used",
    "waiting_count_max",
    "runnable_count_max",
    "graph_bucket",
    "graph_hits",
    "graph_misses",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_mapping(obj: Any, label: str) -> dict[str, Any]:
    require(isinstance(obj, dict), f"{label} is not an object")
    return obj


def require_keys(obj: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in obj]
    require(not missing, f"{label} missing keys: {missing}")


def require_nonnegative(value: Any, label: str) -> None:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric: {value!r}",
    )
    require(value >= 0, f"{label} is negative: {value!r}")


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    first = out.strip().splitlines()[0] if out.strip() else ""
    try:
        return int(first)
    except ValueError:
        return None


def cuda_memory_snapshot(device_arg: str) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "device_arg": device_arg,
            "cuda_available": False,
            "error": str(exc),
            "device_used_mib": gpu_mem_used_mib(),
        }

    snap: dict[str, Any] = {
        "device_arg": device_arg,
        "cuda_available": torch.cuda.is_available(),
        "device_used_mib": gpu_mem_used_mib(),
    }
    if not torch.cuda.is_available():
        return snap

    torch.cuda.synchronize()
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    snap.update(
        {
            "device_index": idx,
            "device_name": props.name,
            "allocated_gib": round(torch.cuda.memory_allocated(idx) / 2**30, 6),
            "reserved_gib": round(torch.cuda.memory_reserved(idx) / 2**30, 6),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(idx) / 2**30, 6),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved(idx) / 2**30, 6),
        }
    )
    return snap


def state_summary(engine, resident_sessions: int) -> dict[str, Any]:
    slots = list(engine.bank.slots[1:])
    per_slot = [
        {
            "slot_id": slot.slot_id,
            "resident_tokens": slot.resident_tokens,
            "ring_tokens": slot.ring_tokens,
            "span_bank_tokens": slot.span_bank_tokens,
            "pending_tokens": slot.pending_tokens,
        }
        for slot in slots
    ]
    occupied = [slot for slot in per_slot if slot["resident_tokens"] > 0]
    return {
        "slots_with_state": len(occupied),
        "resident_tokens": sum(slot["resident_tokens"] for slot in per_slot),
        "ring_tokens": sum(slot["ring_tokens"] for slot in per_slot),
        "span_bank_tokens": sum(slot["span_bank_tokens"] for slot in per_slot),
        "pending_tokens": sum(slot["pending_tokens"] for slot in per_slot),
        "per_slot": per_slot,
        "memory_accounting": engine.bank.memory_accounting(
            resident_sessions=resident_sessions
        ),
    }


def derived_step_metrics(engine_metrics: dict[str, Any]) -> dict[str, Any]:
    decode_batches = int(engine_metrics["decode_batches"])
    rows = int(engine_metrics["max_decode_batch_rows"])
    return {
        "scheduled_rows_max": int(engine_metrics["max_runnable_rows"]),
        "token_budget_used": int(engine_metrics["scheduled_tokens"]),
        "waiting_count_max": int(engine_metrics["max_waiting"]),
        "runnable_count_max": int(engine_metrics["max_running"]),
        "graph_bucket": f"eager:native-gemma:b{rows}",
        "graph_hits": 0,
        "graph_misses": decode_batches,
    }


def response_summary(results: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for code, body in results:
        rows.append(
            {
                "status_code": code,
                "req_id": body.get("req_id"),
                "token_count": len(body.get("tokens", [])),
                "finish_reason": body.get("finish_reason"),
                "latency_s": body.get("latency_s"),
                "has_metrics": isinstance(body.get("metrics"), dict),
            }
        )
    return sorted(rows, key=lambda row: str(row["req_id"]))


def validate_request_metrics(
    *,
    payloads: list[dict[str, Any]],
    results: list[tuple[int, dict[str, Any]]],
    engine_metrics: dict[str, Any],
    expected_out: int,
) -> None:
    requests = require_mapping(engine_metrics["requests"], "engine.requests")
    by_req = {payload["req_id"]: payload for payload in payloads}
    result_bodies = {body.get("req_id"): (code, body) for code, body in results}

    require(len(result_bodies) == len(payloads), "duplicate or missing response req_id")
    for req_id, payload in by_req.items():
        require(req_id in result_bodies, f"missing response for {req_id}")
        code, body = result_bodies[req_id]
        require(code == 200, f"{req_id} returned HTTP {code}: {body}")
        require(len(body.get("tokens", [])) == expected_out, f"{req_id} token count mismatch")
        require(body.get("finish_reason") == "length", f"{req_id} finish reason mismatch")
        body_metrics = require_mapping(body.get("metrics"), f"{req_id} response metrics")
        require_keys(body_metrics, REQUEST_METRIC_KEYS, f"{req_id} response metrics")

        require(req_id in requests, f"missing engine request trace for {req_id}")
        metrics = require_mapping(requests[req_id], f"{req_id} engine request metrics")
        require_keys(metrics, REQUEST_METRIC_KEYS, f"{req_id} engine request metrics")
        require(metrics["req_id"] == req_id, f"{req_id} trace id mismatch")
        require(
            metrics["prompt_tokens"] == len(payload["prompt_ids"]),
            f"{req_id} prompt token count mismatch",
        )
        require(
            metrics["target_output_tokens"] == expected_out,
            f"{req_id} target output count mismatch",
        )
        require(metrics["output_tokens"] == expected_out, f"{req_id} output count mismatch")
        require(metrics["finish_reason"] == "length", f"{req_id} trace finish mismatch")
        require(metrics["error"] is None, f"{req_id} trace error: {metrics['error']}")
        for key in (
            "queue_time_s",
            "prefill_time_s",
            "decode_time_s",
            "first_token_latency_s",
            "total_latency_s",
        ):
            require_nonnegative(metrics[key], f"{req_id}.{key}")
        require(
            metrics["total_latency_s"] >= metrics["first_token_latency_s"],
            f"{req_id} total latency shorter than first-token latency",
        )


def validate_server_metrics(
    *,
    health: dict[str, Any],
    server_metrics: dict[str, Any],
    concurrency: int,
    max_queue: int,
) -> None:
    require_keys(server_metrics, SERVER_METRIC_KEYS, "server metrics")
    require(health.get("ok") is True, f"health not ready: {health}")
    require(health.get("last_error") is None, f"health last_error: {health}")
    require(health.get("queue_depth") == 0, f"health queue not drained: {health}")
    require(health.get("running") == 0, f"health running not drained: {health}")
    require(server_metrics["ready"] is True, "server metrics not ready")
    require(server_metrics["worker_alive"] is True, "server worker is not alive")
    require(server_metrics["total_requests"] == concurrency, "server request count mismatch")
    require(server_metrics["total_errors"] == 0, "server observed errors")
    require(server_metrics["total_cancelled"] >= 1, "cancel metric was not recorded")
    require(server_metrics["max_queue"] == max_queue, "server max_queue mismatch")
    require(server_metrics["last_error"] is None, f"server last_error: {server_metrics}")


def validate_engine_metrics(
    *,
    engine_metrics: dict[str, Any],
    state: dict[str, Any],
    gpu: dict[str, Any],
    args,
    total_prompt_tokens: int,
) -> None:
    require_keys(engine_metrics, ENGINE_METRIC_KEYS, "engine metrics")
    for key in (
        "steps",
        "scheduled_tokens",
        "admitted_requests",
        "finished_requests",
        "error_count",
        "prefill_calls",
        "decode_batches",
        "decode_rows",
        "max_decode_batch_rows",
        "distinct_history_decode_batches",
        "max_waiting",
        "max_running",
        "max_runnable_rows",
        "max_resident_state_slots",
        "max_active_cache_bytes",
        "backpressure_events",
        "retraction_events",
        "queue_depth",
        "runnable_rows",
        "resident_state_slots",
        "free_state_slots",
        "active_cache_bytes",
        "state_bytes_per_request",
    ):
        require_nonnegative(engine_metrics[key], f"engine.{key}")

    expected_decode_rows = args.concurrency * max(args.out - 1, 0)
    expected_scheduled = total_prompt_tokens + expected_decode_rows
    expected_resident = min(args.concurrency, args.slots)
    require(engine_metrics["error_count"] == 0, "engine error_count is non-zero")
    require(engine_metrics["admitted_requests"] == args.concurrency, "admission count mismatch")
    require(engine_metrics["finished_requests"] == args.concurrency, "finish count mismatch")
    require(engine_metrics["prefill_calls"] == args.concurrency, "prefill count mismatch")
    require(
        engine_metrics["scheduled_tokens"] >= expected_scheduled,
        "scheduled token count is too small",
    )
    require(engine_metrics["queue_depth"] == 0, "engine queue not drained")
    require(engine_metrics["runnable_rows"] == 0, "engine runnable rows not drained")
    require(engine_metrics["resident_state_slots"] == 0, "engine resident slots not released")
    require(
        engine_metrics["max_resident_state_slots"] >= expected_resident,
        "resident slot high-water mark too small",
    )
    require(
        engine_metrics["max_decode_batch_rows"] >= expected_resident,
        "decode batch high-water mark too small",
    )
    require(
        engine_metrics["decode_rows"] >= expected_decode_rows,
        "decode row count is too small",
    )
    if args.out > 1 and args.concurrency > 1:
        require(
            engine_metrics["distinct_history_decode_batches"] >= 1,
            "no distinct-history decode batch observed",
        )
    require(engine_metrics["max_active_cache_bytes"] > 0, "active cache bytes never rose")
    require(engine_metrics["state_bytes_per_request"] > 0, "state bytes per request missing")

    require(state["slots_with_state"] >= expected_resident, "state slot traces missing")
    require(state["resident_tokens"] > 0, "state resident token accounting missing")
    require(state["ring_tokens"] > 0, "state ring token accounting missing")
    require_nonnegative(state["span_bank_tokens"], "state.span_bank_tokens")
    require_nonnegative(state["pending_tokens"], "state.pending_tokens")
    require(
        state["memory_accounting"]["resident_sessions"] == expected_resident,
        "state memory resident session count mismatch",
    )
    require(state["memory_accounting"]["estimated_bytes"] > 0, "state memory bytes missing")

    if args.device.startswith("cuda"):
        require(gpu.get("cuda_available") is True, "CUDA memory metrics unavailable")
        require_nonnegative(gpu.get("allocated_gib"), "gpu.allocated_gib")
        require_nonnegative(gpu.get("reserved_gib"), "gpu.reserved_gib")
        require_nonnegative(gpu.get("peak_allocated_gib"), "gpu.peak_allocated_gib")
        require_nonnegative(gpu.get("peak_reserved_gib"), "gpu.peak_reserved_gib")
        if gpu.get("device_used_mib") is not None:
            require_nonnegative(gpu["device_used_mib"], "gpu.device_used_mib")


def validate_step_metrics(
    step_metrics: dict[str, Any],
    engine_metrics: dict[str, Any],
) -> None:
    require_keys(step_metrics, STEP_METRIC_KEYS, "derived step metrics")
    for key in (
        "scheduled_rows_max",
        "token_budget_used",
        "waiting_count_max",
        "runnable_count_max",
        "graph_hits",
        "graph_misses",
    ):
        require_nonnegative(step_metrics[key], f"step_metrics.{key}")
    require(
        isinstance(step_metrics["graph_bucket"], str) and step_metrics["graph_bucket"],
        "step_metrics.graph_bucket missing",
    )
    require(step_metrics["graph_hits"] == 0, "unexpected graph hits before engine graph dispatch")
    require(
        step_metrics["graph_misses"] == engine_metrics["decode_batches"],
        "graph miss count does not match eager decode batches",
    )


def run(args) -> None:
    import torch

    engine, payloads = make_engine(args)
    service = BoundedGemmaService(
        engine,
        max_queue=args.max_queue,
        batch_wait_s=args.batch_wait_s,
    )
    server = serve(service, port=0)
    host, port = server.server_address
    base = f"http://{host}:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    service_closed = False
    thread.start()

    try:
        code, initial_health = get_json(f"{base}/health")
        require(code == 200 and initial_health.get("ok"), f"initial health failed: {initial_health}")

        start = time.perf_counter()
        results: list[tuple[int, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(post_json, f"{base}/v1/generate", payload, args.timeout)
                for payload in payloads
            ]
            for fut in as_completed(futures):
                results.append(fut.result())
        elapsed_s = time.perf_counter() - start

        cancel_code, cancel_body = post_json(
            f"{base}/v1/cancel",
            {"req_id": "metrics-cancel-noop"},
            args.timeout,
        )
        require(cancel_code == 200 and "cancelled" in cancel_body, "cancel endpoint failed")

        health_code, health = get_json(f"{base}/health")
        require(health_code == 200, "health endpoint failed after generation")
        metrics_code, metrics = get_json(f"{base}/metrics")
        require(metrics_code == 200, "metrics endpoint failed")

        server_metrics = require_mapping(metrics.get("server"), "metrics.server")
        engine_metrics = require_mapping(metrics.get("engine"), "metrics.engine")
        total_prompt_tokens = sum(len(payload["prompt_ids"]) for payload in payloads)
        state = state_summary(engine, min(args.concurrency, args.slots))
        step_metrics = derived_step_metrics(engine_metrics)
        gpu = cuda_memory_snapshot(args.device)

        validate_request_metrics(
            payloads=payloads,
            results=results,
            engine_metrics=engine_metrics,
            expected_out=args.out,
        )
        validate_server_metrics(
            health=health,
            server_metrics=server_metrics,
            concurrency=args.concurrency,
            max_queue=args.max_queue,
        )
        validate_engine_metrics(
            engine_metrics=engine_metrics,
            state=state,
            gpu=gpu,
            args=args,
            total_prompt_tokens=total_prompt_tokens,
        )
        validate_step_metrics(step_metrics, engine_metrics)

        service.close()
        service_closed = True
        shutdown_metrics = service.metrics()["server"]
        require(service.health()["ok"] is False, "service stayed ready after close")
        require(shutdown_metrics["worker_alive"] is False, "worker stayed alive after close")

        report = {
            "schema": "wkvm.native_gemma_metrics_smoke.v1",
            "engine": "wkvm-native-gemma",
            "elapsed_s": round(elapsed_s, 6),
            "requested": {
                "ctx": args.ctx,
                "out": args.out,
                "concurrency": args.concurrency,
                "slots": args.slots,
            },
            "responses": response_summary(results),
            "health": health,
            "metrics": metrics,
            "state": state,
            "step_metrics": step_metrics,
            "gpu": gpu,
            "shutdown": {
                "ready_after_close": shutdown_metrics["ready"],
                "worker_alive_after_close": shutdown_metrics["worker_alive"],
            },
        }
        print(json.dumps(report, sort_keys=True))
        print(f"success_count={len(results)}")
        print("error_count=0")
        print(f"server_total_requests={server_metrics['total_requests']}")
        print(f"engine_steps={engine_metrics['steps']}")
        print(f"engine_scheduled_tokens={engine_metrics['scheduled_tokens']}")
        print(f"engine_decode_batches={engine_metrics['decode_batches']}")
        print(f"engine_max_decode_batch_rows={engine_metrics['max_decode_batch_rows']}")
        print(f"state_slots_with_state={state['slots_with_state']}")
        print(f"state_ring_tokens={state['ring_tokens']}")
        print(f"state_span_bank_tokens={state['span_bank_tokens']}")
        print(f"gpu_peak_reserved_gib={gpu.get('peak_reserved_gib')}")
        print(f"gpu_device_used_mib={gpu.get('device_used_mib')}")
        print("NATIVE_METRICS_OK")
    finally:
        if not service_closed:
            service.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=256)
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--batch-wait-s", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()
    if args.slots is None:
        args.slots = args.concurrency
    if args.max_queue < args.concurrency:
        raise SystemExit("--max-queue must be >= --concurrency for this smoke")
    run(args)


if __name__ == "__main__":
    main()
