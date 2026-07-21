#!/usr/bin/env python
"""Build a conservative provider-HTTP 10x multi-turn comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from gemma_multiturn_bench import atomic_write_json, percentile


SCHEMA = "wkvm.multiturn_http_10x_report.v1"
BENCH_SCHEMA = "wkvm.gemma_multiturn_http_bench.v1"
TRACE_SCHEMA = "wkvm.gemma_shared_history_trace.v1"
OUTPUT_FINGERPRINT_SCHEMA = "wkvm.generated_output_token_ids.sha256.v1"
WHOLE_GPU_MEMORY_SCHEMA = "wkvm.whole_gpu_memory.v1"
ENGINES = ("wkvm", "vllm", "sglang")
INCUMBENTS = ("vllm", "sglang")
CLAIM_SCOPES = ("continuation", "full-session")
PUBLICATION_MIN_REPEATS = 3
THRESHOLD = 10.0
SLOW_REPEAT_FRACTION = 0.80
LOW_CLOCK_PEER_FRACTION = 0.95
LOW_ACTIVITY_PEER_FRACTION = 0.90
HOT_PEER_DELTA_C = 5.0
SOURCE_ROLE = "http_trace_source"
REPLAY_ROLES = frozenset({"http_teacher_forced_replay", "http_trace_replay"})
EXACT_OUTPUT_SOURCES = frozenset(
    {"response_token_ids", "teacher_trace_hook_contract"}
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _valid_git_commit(value: Any) -> bool:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _number(value: Any, *, positive: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        return None
    return result


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _normalized_text(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (str, int)):
        return None
    result = str(value).strip()
    return result or None


def _engine_version_identity(payload: dict[str, Any]) -> tuple[str, str] | None:
    engine = _normalized_text(payload.get("engine"))
    version = _normalized_text(payload.get("engine_version"))
    provenance = payload.get("provenance")
    engine_provenance = (
        provenance.get("engine") if isinstance(provenance, dict) else None
    )
    if not isinstance(engine_provenance, dict):
        return None
    provenance_label = _normalized_text(engine_provenance.get("label"))
    provenance_version = _normalized_text(engine_provenance.get("version"))
    version_source = _normalized_text(engine_provenance.get("version_source"))
    if (
        engine is None
        or version is None
        or provenance_label != engine
        or provenance_version != version
        or version_source in {None, "unreported"}
    ):
        return None
    return version, version_source


def _target_server_provenance(
    payload: dict[str, Any],
) -> tuple[str, str] | None:
    provenance = payload.get("provenance")
    target_server = (
        provenance.get("target_server") if isinstance(provenance, dict) else None
    )
    if not isinstance(target_server, dict):
        return None
    launch_command = _normalized_text(target_server.get("launch_command"))
    launch_source = _normalized_text(target_server.get("launch_command_source"))
    config = target_server.get("config")
    config_source = _normalized_text(target_server.get("config_source"))
    if (
        launch_command is None
        or launch_source in {None, "unreported"}
        or not isinstance(config, dict)
        or config_source in {None, "unreported"}
    ):
        return None
    try:
        return (
            _payload_sha256(
                {"launch_command": launch_command, "source": launch_source}
            ),
            _payload_sha256({"config": config, "source": config_source}),
        )
    except (TypeError, ValueError):
        return None


def _median(values: Sequence[float]) -> float | None:
    return percentile(values, 0.5)


def _rounded_equal(value: Any, expected: float, digits: int) -> bool:
    observed = _number(value)
    return observed is not None and math.isclose(
        observed,
        round(expected, digits),
        rel_tol=0.0,
        abs_tol=10 ** (-(digits + 1)),
    )


def _workload_signature(payload: dict[str, Any]) -> str | None:
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        return None
    fingerprints = workload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        return None
    initial = fingerprints.get("initial_prompts")
    deltas = fingerprints.get("turn_deltas")
    if not isinstance(initial, dict) or not isinstance(deltas, list):
        return None
    fields = (
        "sessions",
        "turns",
        "initial_context_tokens",
        "turn_input_tokens",
        "output_tokens_per_turn",
        "required_model_len",
        "request_order_policy",
        "request_order_seed",
        "synchronized_turn_barriers",
    )
    contract = {
        "fields": {field: workload.get(field) for field in fields},
        "prompt_token_source": payload.get("prompt_token_source"),
        "initial_prompts": initial,
        "turn_deltas": deltas,
    }
    try:
        return _payload_sha256(contract)
    except (TypeError, ValueError):
        return None


def _fingerprint_hash(
    fingerprint: Any,
    *,
    sessions: int,
    output_tokens: int,
) -> str | None:
    if not isinstance(fingerprint, dict):
        return None
    expected_ids = [f"session-{index:04d}" for index in range(sessions)]
    digest = fingerprint.get("request_output_token_ids_sha256")
    if (
        fingerprint.get("schema") != OUTPUT_FINGERPRINT_SCHEMA
        or fingerprint.get("request_count") != sessions
        or fingerprint.get("output_token_count") != sessions * output_tokens
        or fingerprint.get("request_ids") != expected_ids
        or fingerprint.get("output_token_counts")
        != [output_tokens] * sessions
        or not _valid_sha256(digest)
    ):
        return None
    return str(digest)


def _trace_sha256(payload: dict[str, Any]) -> tuple[str | None, bool]:
    values: list[str] = []
    for field in ("history_trace", "emitted_history_trace"):
        metadata = payload.get(field)
        if not isinstance(metadata, dict):
            continue
        value = metadata.get("trace_sha256")
        if _valid_sha256(value):
            values.append(str(value))
    unique = set(values)
    return (next(iter(unique)), True) if len(unique) == 1 else (None, not unique)


def _trace_metadata_for_role(
    payload: dict[str, Any],
    artifact_role: str | None,
) -> dict[str, Any] | None:
    fields = (
        ("emitted_history_trace",)
        if artifact_role == SOURCE_ROLE
        else ("history_trace",)
    )
    for field in fields:
        metadata = payload.get(field)
        if isinstance(metadata, dict):
            return metadata
    return None


def _trace_output_signature(
    metadata: dict[str, Any] | None,
    *,
    sessions: int | None,
    turns: int | None,
    output_tokens: int | None,
) -> tuple[str, ...] | None:
    if metadata is None or sessions is None or turns is None or output_tokens is None:
        return None
    fingerprints = metadata.get("output_fingerprints")
    if not isinstance(fingerprints, list) or len(fingerprints) != turns:
        return None
    hashes: list[str] = []
    for fingerprint in fingerprints:
        digest = _fingerprint_hash(
            fingerprint,
            sessions=sessions,
            output_tokens=output_tokens,
        )
        if digest is None:
            return None
        hashes.append(digest)
    return tuple(hashes)


def _trace_metadata_contract(
    payload: dict[str, Any],
    *,
    artifact_role: str | None,
    workload: dict[str, Any],
    output_signature: tuple[str, ...] | None,
) -> tuple[bool, tuple[str, ...] | None]:
    metadata = _trace_metadata_for_role(payload, artifact_role)
    sessions = _integer(workload.get("sessions"), minimum=1)
    turns = _integer(workload.get("turns"), minimum=2)
    output_tokens = _integer(
        workload.get("output_tokens_per_turn"),
        minimum=1,
    )
    metadata_signature = _trace_output_signature(
        metadata,
        sessions=sessions,
        turns=turns,
        output_tokens=output_tokens,
    )
    valid = (
        metadata is not None
        and metadata.get("schema") == TRACE_SCHEMA
        and metadata.get("shared") is True
        and metadata.get("teacher_forced") is True
        and metadata.get("turn_count") == turns
        and _valid_sha256(metadata.get("trace_sha256"))
        and metadata_signature is not None
        and output_signature is not None
        and metadata_signature == output_signature
    )
    return valid, metadata_signature


def _tree_state(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("git_tree_state")
    if not isinstance(raw, dict):
        return {
            "classification": "unknown",
            "clean": None,
            "tracked_clean": None,
            "changed_path_count": None,
            "tracked_changed_path_count": None,
            "untracked_path_count": None,
            "status_sha256": None,
            "tracked_status_sha256": None,
        }
    clean = raw.get("clean")
    tracked_clean = raw.get("tracked_clean")
    classification = (
        "clean"
        if clean is True and tracked_clean is True
        else "dirty"
        if clean is False or tracked_clean is False
        else "unknown"
    )
    return {
        "classification": classification,
        "clean": clean if isinstance(clean, bool) else None,
        "tracked_clean": (
            tracked_clean if isinstance(tracked_clean, bool) else None
        ),
        "changed_path_count": raw.get("changed_path_count"),
        "tracked_changed_path_count": raw.get("tracked_changed_path_count"),
        "untracked_path_count": raw.get("untracked_path_count"),
        "status_sha256": raw.get("status_sha256"),
        "tracked_status_sha256": raw.get("tracked_status_sha256"),
    }


def _telemetry_stat(
    telemetry: dict[str, Any],
    section: str,
    metric: str,
    statistic: str,
) -> float | None:
    metrics = telemetry.get(section)
    if not isinstance(metrics, dict):
        return None
    values = metrics.get(metric)
    if not isinstance(values, dict):
        return None
    return _number(values.get(statistic))


def _runtime_telemetry(gpu_memory: dict[str, Any]) -> dict[str, Any]:
    raw = gpu_memory.get("gpu_runtime_telemetry")
    if not isinstance(raw, dict):
        raw = {}
    sample_count = _integer(raw.get("sample_count"))
    active_sample_count = _integer(raw.get("active_sample_count"))
    query_error_count = _integer(gpu_memory.get("query_error_count"))
    result = {
        "sample_count": sample_count,
        "active_sample_count": active_sample_count,
        "active_sample_fraction": (
            None
            if sample_count in {None, 0} or active_sample_count is None
            else active_sample_count / sample_count
        ),
        "query_error_count": query_error_count,
        "active_sm_clock_mhz_mean": _telemetry_stat(
            raw,
            "active_metrics",
            "sm_clock_mhz",
            "mean",
        ),
        "active_gpu_utilization_percent_mean": _telemetry_stat(
            raw,
            "active_metrics",
            "gpu_utilization_percent",
            "mean",
        ),
        "active_power_draw_w_mean": _telemetry_stat(
            raw,
            "active_metrics",
            "power_draw_w",
            "mean",
        ),
        "temperature_gpu_c_max": _telemetry_stat(
            raw,
            "metrics",
            "temperature_gpu_c",
            "max",
        ),
        "power_limit_w_mean": _telemetry_stat(
            raw,
            "metrics",
            "power_limit_w",
            "mean",
        ),
        "active_pstates": (
            sorted(
                value
                for value in raw.get("active_pstates", [])
                if isinstance(value, str) and value
            )
            if isinstance(raw.get("active_pstates"), list)
            else []
        ),
    }
    result["complete"] = (
        sample_count is not None
        and sample_count >= 2
        and active_sample_count is not None
        and active_sample_count >= 1
        and query_error_count == 0
        and all(
            result[field] is not None
            for field in (
                "active_sm_clock_mhz_mean",
                "active_gpu_utilization_percent_mean",
                "active_power_draw_w_mean",
                "temperature_gpu_c_max",
                "power_limit_w_mean",
            )
        )
    )
    return result


def _measure_http_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    workload = payload.get("workload")
    turns_payload = payload.get("turns")
    summary = payload.get("summary")
    if not isinstance(workload, dict):
        workload = {}
        errors.append("missing_workload")
    if not isinstance(turns_payload, list):
        turns_payload = []
        errors.append("missing_turn_rows")
    if not isinstance(summary, dict):
        summary = {}
        errors.append("missing_summary")
    sessions = _integer(workload.get("sessions"), minimum=1)
    turns = _integer(workload.get("turns"), minimum=2)
    output_per_turn = _integer(
        workload.get("output_tokens_per_turn"),
        minimum=1,
    )
    if sessions is None or turns is None or output_per_turn is None:
        errors.append("invalid_workload_shape")
        return {
            "errors": sorted(set(errors)),
            "success_complete": False,
            "exact_output_ids": False,
            "latency_complete": False,
            "continuation_rate": None,
            "continuation_wall_s": None,
            "continuation_output_tokens": None,
            "full_session_rate": None,
            "full_session_wall_s": None,
            "full_session_output_tokens": None,
            "p50_ttft_s": None,
            "p95_ttft_s": None,
            "p50_e2e_s": None,
            "p95_e2e_s": None,
            "output_signature": None,
            "observed_output_id_requests": 0,
            "exact_output_id_requests": 0,
        }

    expected_turn_output = sessions * output_per_turn
    success_complete = len(turns_payload) == turns
    exact_output_ids = len(turns_payload) == turns
    latency_complete = len(turns_payload) == turns
    walls: list[float] = []
    output_signature: list[str] = []
    continuation_ttft: list[float] = []
    continuation_e2e: list[float] = []
    observed_output_id_requests = 0
    exact_output_id_requests = 0

    for fallback_index, turn_row in enumerate(turns_payload):
        if not isinstance(turn_row, dict):
            success_complete = False
            exact_output_ids = False
            latency_complete = False
            continue
        wall_s = _number(turn_row.get("wall_s"), positive=True)
        if wall_s is None:
            success_complete = False
        else:
            walls.append(wall_s)
        if (
            turn_row.get("turn_index") != fallback_index
            or turn_row.get("request_count") != sessions
            or turn_row.get("success_count") != sessions
            or turn_row.get("error_count") != 0
            or turn_row.get("output_tokens") != expected_turn_output
        ):
            success_complete = False

        generated_hash = _fingerprint_hash(
            turn_row.get("generated_output_fingerprint"),
            sessions=sessions,
            output_tokens=output_per_turn,
        )
        response_hash = _fingerprint_hash(
            turn_row.get("response_output_fingerprint"),
            sessions=sessions,
            output_tokens=output_per_turn,
        )
        if (
            turn_row.get("output_fingerprint_complete") is not True
            or turn_row.get("response_output_fingerprint_complete") is not True
            or generated_hash is None
            or response_hash != generated_hash
            or turn_row.get("request_output_token_ids_sha256") != generated_hash
        ):
            exact_output_ids = False
        else:
            output_signature.append(generated_hash)

        request_rows = turn_row.get("requests")
        if not isinstance(request_rows, list) or len(request_rows) != sessions:
            success_complete = False
            exact_output_ids = False
            latency_complete = False
            continue
        for request in request_rows:
            if not isinstance(request, dict):
                success_complete = False
                exact_output_ids = False
                latency_complete = False
                continue
            if (
                request.get("success") is not True
                or request.get("error") is not None
                or request.get("output_tokens") != output_per_turn
                or request.get("observed_output_tokens") != output_per_turn
            ):
                success_complete = False
            source = request.get("output_token_ids_source")
            if source not in EXACT_OUTPUT_SOURCES:
                exact_output_ids = False
            else:
                exact_output_id_requests += 1
            if request.get("output_token_ids_observed") is True:
                observed_output_id_requests += 1
            ttft = _number(request.get("ttft_s"))
            e2e = _number(request.get("e2e_latency_s"))
            if ttft is None or e2e is None:
                latency_complete = False
            elif fallback_index > 0:
                continuation_ttft.append(ttft)
                continuation_e2e.append(e2e)

    if not success_complete:
        errors.append("incomplete_success")
    if not exact_output_ids or len(output_signature) != turns:
        errors.append("inexact_output_ids")
    if not latency_complete:
        errors.append("incomplete_latency")
    if len(walls) != turns:
        errors.append("invalid_turn_walls")

    expected_total_requests = sessions * turns
    expected_continuation_requests = sessions * (turns - 1)
    expected_total_output = expected_turn_output * turns
    continuation_output = expected_turn_output * (turns - 1)
    continuation_wall = sum(walls[1:]) if len(walls) == turns else None
    continuation_rate = (
        continuation_output / continuation_wall
        if continuation_wall is not None and continuation_wall > 0
        else None
    )
    full_session_wall = sum(walls) if len(walls) == turns else None
    full_session_rate = (
        expected_total_output / full_session_wall
        if full_session_wall is not None and full_session_wall > 0
        else None
    )
    continuation = summary.get("continuation_turns")
    if not isinstance(continuation, dict):
        continuation = {}
    expected_p50_ttft = percentile(continuation_ttft, 0.50)
    expected_p95_ttft = percentile(continuation_ttft, 0.95)
    expected_p50_e2e = percentile(continuation_e2e, 0.50)
    expected_p95_e2e = percentile(continuation_e2e, 0.95)
    continuation_accounting_valid = (
        summary.get("requested_turns") == turns
        and summary.get("completed_turn_rows") == turns
        and summary.get("all_turns_recorded") is True
        and summary.get("request_count") == expected_total_requests
        and summary.get("success_count") == expected_total_requests
        and summary.get("error_count") == 0
        and summary.get("output_tokens") == expected_total_output
        and continuation.get("turn_rows") == turns - 1
        and continuation.get("request_count") == expected_continuation_requests
        and continuation.get("success_count") == expected_continuation_requests
        and continuation.get("error_count") == 0
        and continuation.get("output_tokens") == continuation_output
        and continuation.get("ttft_available_count")
        == expected_continuation_requests
        and continuation.get("e2e_latency_available_count")
        == expected_continuation_requests
        and continuation.get("wall_scope")
        == "sum_of_synchronized_engine_turn_barriers"
        and continuation_wall is not None
        and continuation_rate is not None
        and _rounded_equal(continuation.get("wall_s"), continuation_wall, 6)
        and _rounded_equal(continuation.get("output_tok_s"), continuation_rate, 3)
        and expected_p50_ttft is not None
        and expected_p95_ttft is not None
        and expected_p50_e2e is not None
        and expected_p95_e2e is not None
        and _rounded_equal(continuation.get("p50_ttft_s"), expected_p50_ttft, 6)
        and _rounded_equal(continuation.get("p95_ttft_s"), expected_p95_ttft, 6)
        and _rounded_equal(
            continuation.get("p50_e2e_latency_s"),
            expected_p50_e2e,
            6,
        )
        and _rounded_equal(
            continuation.get("p95_e2e_latency_s"),
            expected_p95_e2e,
            6,
        )
    )
    full_session_accounting_valid = (
        summary.get("turn_rows") == turns
        and summary.get("request_count") == expected_total_requests
        and summary.get("success_count") == expected_total_requests
        and summary.get("error_count") == 0
        and summary.get("output_tokens") == expected_total_output
        and summary.get("ttft_available_count") == expected_total_requests
        and summary.get("e2e_latency_available_count") == expected_total_requests
        and summary.get("wall_scope")
        == "sum_of_synchronized_engine_turn_barriers"
        and full_session_wall is not None
        and full_session_rate is not None
        and _rounded_equal(summary.get("wall_s"), full_session_wall, 6)
        and _rounded_equal(summary.get("output_tok_s"), full_session_rate, 3)
    )
    accounting_valid = (
        continuation_accounting_valid and full_session_accounting_valid
    )
    if not continuation_accounting_valid:
        errors.append("invalid_continuation_accounting")
    if not full_session_accounting_valid:
        errors.append("invalid_full_session_accounting")
    return {
        "errors": sorted(set(errors)),
        "success_complete": success_complete,
        "exact_output_ids": exact_output_ids and len(output_signature) == turns,
        "latency_complete": latency_complete and accounting_valid,
        "continuation_rate": continuation_rate if accounting_valid else None,
        "continuation_wall_s": continuation_wall if accounting_valid else None,
        "continuation_output_tokens": (
            continuation_output if accounting_valid else None
        ),
        "full_session_rate": full_session_rate if accounting_valid else None,
        "full_session_wall_s": (
            full_session_wall if accounting_valid else None
        ),
        "full_session_output_tokens": (
            expected_total_output if accounting_valid else None
        ),
        "p50_ttft_s": expected_p50_ttft if accounting_valid else None,
        "p95_ttft_s": expected_p95_ttft if accounting_valid else None,
        "p50_e2e_s": expected_p50_e2e if accounting_valid else None,
        "p95_e2e_s": expected_p95_e2e if accounting_valid else None,
        "output_signature": (
            tuple(output_signature) if len(output_signature) == turns else None
        ),
        "observed_output_id_requests": observed_output_id_requests,
        "exact_output_id_requests": exact_output_id_requests,
    }


def artifact_record(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    errors: list[str] = []
    if payload.get("schema") != BENCH_SCHEMA:
        errors.append("schema_mismatch")
    engine = payload.get("engine")
    if engine not in ENGINES:
        errors.append("unknown_engine")
        engine = str(engine or "")
    measurement = _measure_http_artifact(payload)
    errors.extend(measurement["errors"])
    workload_signature = _workload_signature(payload)
    if workload_signature is None:
        errors.append("invalid_workload_identity")
    trace_sha256, trace_metadata_unambiguous = _trace_sha256(payload)
    if not trace_metadata_unambiguous:
        errors.append("ambiguous_trace_metadata")
    identity = payload.get("benchmark_identity")
    if not isinstance(identity, dict):
        identity = {}
    campaign_id = _normalized_text(identity.get("campaign_id"))
    repeat_id = _normalized_text(identity.get("repeat_id"))
    run_id = _normalized_text(identity.get("run_id"))
    artifact_role = _normalized_text(identity.get("artifact_role"))
    if artifact_role not in {SOURCE_ROLE, *REPLAY_ROLES}:
        errors.append("invalid_artifact_role")
    trace_metadata_valid, trace_output_signature = _trace_metadata_contract(
        payload,
        artifact_role=artifact_role,
        workload=payload.get("workload")
        if isinstance(payload.get("workload"), dict)
        else {},
        output_signature=measurement["output_signature"],
    )
    gpu_memory = payload.get("gpu_memory")
    if not isinstance(gpu_memory, dict):
        gpu_memory = {}
    peak_used_mib = _number(gpu_memory.get("peak_used_mib"))
    memory_scope_valid = (
        gpu_memory.get("schema") == WHOLE_GPU_MEMORY_SCHEMA
        and gpu_memory.get("scope") == "whole_device"
        and peak_used_mib is not None
    )
    runtime_telemetry = _runtime_telemetry(gpu_memory)
    tree_state = _tree_state(payload)
    target_server_provenance = _target_server_provenance(payload)
    return {
        "path": str(artifact_path),
        "payload_sha256": _payload_sha256(payload),
        "engine": engine,
        "campaign_id": campaign_id,
        "repeat_id": repeat_id,
        "run_id": run_id,
        "artifact_role": artifact_role,
        "semantic_mode": _normalized_text(payload.get("semantic_mode")),
        "model_identity": _normalized_text(payload.get("model")),
        "git_commit": _normalized_text(payload.get("git_commit")),
        "engine_version_identity": _engine_version_identity(payload),
        "target_server_launch_signature": (
            target_server_provenance[0]
            if target_server_provenance is not None
            else None
        ),
        "target_server_config_signature": (
            target_server_provenance[1]
            if target_server_provenance is not None
            else None
        ),
        "workload": payload.get("workload"),
        "workload_signature": workload_signature,
        "trace_sha256": trace_sha256,
        "trace_metadata_valid": trace_metadata_valid,
        "trace_output_signature": trace_output_signature,
        "trace_output_binding": (
            trace_metadata_valid
            and trace_output_signature == measurement["output_signature"]
        ),
        "output_signature": measurement["output_signature"],
        "success_complete": measurement["success_complete"],
        "exact_output_ids": measurement["exact_output_ids"],
        "latency_complete": measurement["latency_complete"],
        "continuation_rate": measurement["continuation_rate"],
        "continuation_wall_s": measurement["continuation_wall_s"],
        "continuation_output_tokens": measurement[
            "continuation_output_tokens"
        ],
        "full_session_rate": measurement["full_session_rate"],
        "full_session_wall_s": measurement["full_session_wall_s"],
        "full_session_output_tokens": measurement[
            "full_session_output_tokens"
        ],
        "p50_ttft_s": measurement["p50_ttft_s"],
        "p95_ttft_s": measurement["p95_ttft_s"],
        "p50_e2e_s": measurement["p50_e2e_s"],
        "p95_e2e_s": measurement["p95_e2e_s"],
        "observed_output_id_requests": measurement[
            "observed_output_id_requests"
        ],
        "exact_output_id_requests": measurement["exact_output_id_requests"],
        "memory_scope_valid": memory_scope_valid,
        "peak_used_mib": peak_used_mib,
        "baseline_used_mib": _number(gpu_memory.get("baseline_used_mib")),
        "peak_delta_mib": _number(gpu_memory.get("peak_delta_mib")),
        "memory_schema": gpu_memory.get("schema"),
        "memory_scope": gpu_memory.get("scope"),
        "memory_source": gpu_memory.get("source"),
        "memory_error": gpu_memory.get("error"),
        "memory_query_error_count": _integer(
            gpu_memory.get("query_error_count")
        ),
        "gpu_name": gpu_memory.get("gpu_name"),
        "gpu_uuid": gpu_memory.get("device_uuid"),
        "gpu_memory_total_mib": _number(gpu_memory.get("memory_total_mib")),
        "gpu_driver_version": _normalized_text(gpu_memory.get("driver_version")),
        "runtime_telemetry": runtime_telemetry,
        "tree_state": tree_state,
        "errors": sorted(set(errors)),
    }


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: benchmark artifact must be a JSON object")
        records.append(artifact_record(path, payload))
    if not records:
        raise ValueError("at least one HTTP benchmark artifact is required")
    return records


def _repeat_key(record: dict[str, Any]) -> str:
    campaign_id = record.get("campaign_id")
    repeat_id = record.get("repeat_id")
    if campaign_id is not None and repeat_id is not None:
        return f"campaign:{campaign_id}:repeat:{repeat_id}"
    trace_sha256 = record.get("trace_sha256")
    if trace_sha256 is not None:
        return f"trace:{trace_sha256}"
    output_signature = record.get("output_signature")
    if output_signature is not None:
        return f"outputs:{_payload_sha256(output_signature)}"
    return f"artifact:{record['payload_sha256']}"


def _build_repeat_groups(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_repeat_key(record), []).append(record)
    groups: list[dict[str, Any]] = []
    for key, group_records in sorted(grouped.items()):
        engine_counts = {
            engine: sum(record.get("engine") == engine for record in group_records)
            for engine in ENGINES
        }
        trace_values = {record.get("trace_sha256") for record in group_records}
        trace_linked = None not in trace_values and len(trace_values) == 1
        output_values = {record.get("output_signature") for record in group_records}
        output_linked = None not in output_values and len(output_values) == 1
        linkage_method = (
            "shared_trace_sha256"
            if trace_linked and output_linked
            else "per_turn_output_fingerprints"
            if output_linked
            else None
        )
        source_records = [
            record
            for record in group_records
            if record.get("artifact_role") == SOURCE_ROLE
        ]
        replay_records = [
            record
            for record in group_records
            if record.get("artifact_role") in REPLAY_ROLES
        ]
        source_replay_contract = (
            len(source_records) == 1
            and len(replay_records) == len(ENGINES) - 1
            and len(group_records) == len(ENGINES)
        )
        complete = (
            all(engine_counts[engine] == 1 for engine in ENGINES)
            and linkage_method is not None
            and source_replay_contract
            and all(record.get("errors") == [] for record in group_records)
        )
        groups.append(
            {
                "key": key,
                "campaign_id": next(
                    (
                        record.get("campaign_id")
                        for record in group_records
                        if record.get("campaign_id") is not None
                    ),
                    None,
                ),
                "repeat_id": next(
                    (
                        record.get("repeat_id")
                        for record in group_records
                        if record.get("repeat_id") is not None
                    ),
                    None,
                ),
                "engine_counts": engine_counts,
                "linkage_method": linkage_method,
                "trace_sha256": (
                    next(iter(trace_values)) if trace_linked else None
                ),
                "source_engine": (
                    source_records[0].get("engine")
                    if len(source_records) == 1
                    else None
                ),
                "source_replay_contract": source_replay_contract,
                "complete": complete,
                "artifacts": [record["path"] for record in group_records],
            }
        )
    return groups


def _metric_summary(values: Sequence[float]) -> dict[str, float | None]:
    normalized = [float(value) for value in values]
    return {
        "min": min(normalized) if normalized else None,
        "median": _median(normalized),
        "max": max(normalized) if normalized else None,
    }


def _witness(
    record: dict[str, Any] | None,
    *,
    rate_field: str,
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "artifact": record.get("path"),
        "campaign_id": record.get("campaign_id"),
        "repeat_id": record.get("repeat_id"),
        "run_id": record.get("run_id"),
        "claim_output_tok_s": record.get(rate_field),
        "continuation_output_tok_s": record.get("continuation_rate"),
        "full_session_output_tok_s": record.get("full_session_rate"),
        "full_session_wall_s": record.get("full_session_wall_s"),
    }


def _telemetry_median(
    records: Sequence[dict[str, Any]],
    field: str,
) -> float | None:
    values = [
        float(record["runtime_telemetry"][field])
        for record in records
        if isinstance(record.get("runtime_telemetry"), dict)
        and record["runtime_telemetry"].get(field) is not None
    ]
    return _median(values)


def _build_stability_diagnostics(
    engine_groups: dict[str, list[dict[str, Any]]],
    *,
    rate_field: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    engine_stability: dict[str, Any] = {}
    repeat_diagnostics: list[dict[str, Any]] = []
    peer_fields = (
        "active_sample_fraction",
        "active_sm_clock_mhz_mean",
        "active_gpu_utilization_percent_mean",
        "active_power_draw_w_mean",
        "temperature_gpu_c_max",
    )
    for engine in ENGINES:
        rate_records = [
            record
            for record in engine_groups[engine]
            if record.get(rate_field) is not None
        ]
        rates = [float(record[rate_field]) for record in rate_records]
        median_rate = _median(rates)
        minimum_rate = min(rates) if rates else None
        maximum_rate = max(rates) if rates else None
        peer_medians = {
            field: _telemetry_median(rate_records, field)
            for field in peer_fields
        }
        slow_count = 0
        for record in rate_records:
            rate = float(record[rate_field])
            fraction = (
                None
                if median_rate is None or median_rate <= 0
                else rate / median_rate
            )
            slow_candidate = (
                len(rate_records) >= 2
                and fraction is not None
                and fraction < SLOW_REPEAT_FRACTION
            )
            telemetry = record.get("runtime_telemetry")
            if not isinstance(telemetry, dict):
                telemetry = {}
            signals: list[str] = []
            if slow_candidate:
                slow_count += 1
                signals.append("output_rate_below_80pct_of_engine_median")
                if telemetry.get("complete") is not True:
                    signals.append("runtime_telemetry_incomplete")
                clock = _number(telemetry.get("active_sm_clock_mhz_mean"))
                median_clock = peer_medians["active_sm_clock_mhz_mean"]
                if (
                    clock is not None
                    and median_clock is not None
                    and clock < median_clock * LOW_CLOCK_PEER_FRACTION
                ):
                    signals.append("active_sm_clock_below_peer_median")
                for field, signal in (
                    (
                        "active_sample_fraction",
                        "active_sample_fraction_below_peer_median",
                    ),
                    (
                        "active_gpu_utilization_percent_mean",
                        "active_gpu_utilization_below_peer_median",
                    ),
                    (
                        "active_power_draw_w_mean",
                        "active_power_draw_below_peer_median",
                    ),
                ):
                    value = _number(telemetry.get(field))
                    peer_median = peer_medians[field]
                    if (
                        value is not None
                        and peer_median is not None
                        and value < peer_median * LOW_ACTIVITY_PEER_FRACTION
                    ):
                        signals.append(signal)
                temperature = _number(telemetry.get("temperature_gpu_c_max"))
                median_temperature = peer_medians["temperature_gpu_c_max"]
                if (
                    temperature is not None
                    and median_temperature is not None
                    and temperature > median_temperature + HOT_PEER_DELTA_C
                ):
                    signals.append("gpu_temperature_above_peer_median")
            repeat_diagnostics.append(
                {
                    "engine": engine,
                    "artifact": record.get("path"),
                    "repeat_id": record.get("repeat_id"),
                    "claim_output_tok_s": rate,
                    "fraction_of_engine_median": fraction,
                    "slow_repeat_candidate": slow_candidate,
                    "signals": signals,
                    "runtime_telemetry": telemetry,
                }
            )
        engine_stability[engine] = {
            "sample_count": len(rate_records),
            "minimum_over_median": (
                None
                if minimum_rate is None or median_rate is None or median_rate <= 0
                else minimum_rate / median_rate
            ),
            "maximum_over_minimum": (
                None
                if minimum_rate is None
                or maximum_rate is None
                or minimum_rate <= 0
                else maximum_rate / minimum_rate
            ),
            "slow_repeat_candidate_count": slow_count,
            "runtime_telemetry_complete_count": sum(
                record.get("runtime_telemetry", {}).get("complete") is True
                for record in rate_records
            ),
        }
    return engine_stability, repeat_diagnostics


def _publication_artifact_checks(record: dict[str, Any]) -> dict[str, bool]:
    tree = record.get("tree_state")
    clean_tree = (
        isinstance(tree, dict)
        and tree.get("clean") is True
        and tree.get("tracked_clean") is True
        and tree.get("classification") == "clean"
    )
    gpu_identity = bool(record.get("gpu_uuid")) and bool(record.get("gpu_name"))
    baseline = _number(record.get("baseline_used_mib"))
    idle_gpu_baseline = (
        baseline is not None
        and baseline <= 1024.0
        and record.get("memory_error") in (None, "")
        and int(record.get("memory_query_error_count") or 0) == 0
    )
    peak = _number(record.get("peak_used_mib"))
    delta = _number(record.get("peak_delta_mib"))
    memory_total = _number(record.get("gpu_memory_total_mib"))
    memory_delta = (
        record.get("memory_schema") == WHOLE_GPU_MEMORY_SCHEMA
        and record.get("memory_scope") == "whole_device"
        and record.get("memory_source") == "nvidia-smi"
        and peak is not None
        and delta is not None
        and baseline is not None
        and memory_total is not None
        and peak <= memory_total
        and math.isclose(delta, peak - baseline, rel_tol=0.0, abs_tol=1.0)
        and record.get("memory_error") in (None, "")
        and int(record.get("memory_query_error_count") or 0) == 0
    )
    return {
        "clean_worktree": clean_tree,
        "gpu_identity": gpu_identity,
        "idle_gpu_baseline": idle_gpu_baseline,
        "memory_delta": memory_delta,
        "trace_metadata": record.get("trace_metadata_valid") is True,
        "trace_output_binding": record.get("trace_output_binding") is True,
    }


def _publication_campaign_checks(
    records: Sequence[dict[str, Any]],
    *,
    min_repeats: int,
    memory_ceiling_mib: float | None,
) -> dict[str, bool]:
    identities_complete = bool(records) and all(
        record.get(field) is not None
        for record in records
        for field in ("campaign_id", "repeat_id", "run_id")
    )
    campaign_ids = {record.get("campaign_id") for record in records}
    same_campaign = identities_complete and len(campaign_ids) == 1
    repeat_groups: dict[str, list[dict[str, Any]]] = {}
    if identities_complete:
        for record in records:
            repeat_groups.setdefault(str(record["repeat_id"]), []).append(record)
    exact_repeat_matrix = (
        identities_complete
        and min_repeats >= PUBLICATION_MIN_REPEATS
        and len(records) == min_repeats * len(ENGINES)
        and len(repeat_groups) == min_repeats
        and all(
            len(group) == len(ENGINES)
            and sorted(record.get("engine") for record in group)
            == sorted(ENGINES)
            for group in repeat_groups.values()
        )
    )
    unique_artifacts = (
        len({record.get("path") for record in records}) == len(records)
        and len({record.get("payload_sha256") for record in records}) == len(records)
        and len({record.get("run_id") for record in records}) == len(records)
    )
    artifact_checks = [_publication_artifact_checks(record) for record in records]
    clean_worktree = bool(artifact_checks) and all(
        checks["clean_worktree"] for checks in artifact_checks
    )
    idle_gpu_baseline = bool(artifact_checks) and all(
        checks["idle_gpu_baseline"] for checks in artifact_checks
    )
    memory_delta = bool(artifact_checks) and all(
        checks["memory_delta"] for checks in artifact_checks
    )
    trace_metadata = bool(artifact_checks) and all(
        checks["trace_metadata"] for checks in artifact_checks
    )
    trace_output_binding = bool(artifact_checks) and all(
        checks["trace_output_binding"] for checks in artifact_checks
    )
    gpu_values = {
        (record.get("gpu_uuid"), record.get("gpu_name")) for record in records
    }
    same_gpu = (
        bool(gpu_values)
        and all(uuid and name for uuid, name in gpu_values)
        and len(gpu_values) == 1
    )
    driver_values = {record.get("gpu_driver_version") for record in records}
    driver_identity = bool(records) and None not in driver_values
    same_driver = driver_identity and len(driver_values) == 1
    commits = {record.get("git_commit") for record in records}
    same_git_commit = (
        bool(records)
        and all(_valid_git_commit(commit) for commit in commits)
        and len(commits) == 1
    )
    model_identities = {record.get("model_identity") for record in records}
    stable_model_identity = (
        bool(records)
        and None not in model_identities
        and len(model_identities) == 1
    )

    def stable_per_engine(field: str) -> bool:
        return all(
            bool(engine_records)
            and None not in {record.get(field) for record in engine_records}
            and len({record.get(field) for record in engine_records}) == 1
            for engine in ENGINES
            for engine_records in (
                [record for record in records if record.get("engine") == engine],
            )
        )

    target_server_provenance = bool(records) and all(
        record.get("target_server_launch_signature") is not None
        and record.get("target_server_config_signature") is not None
        for record in records
    )
    stable_engine_versions = stable_per_engine("engine_version_identity")
    stable_engine_launches = stable_per_engine("target_server_launch_signature")
    stable_engine_configs = stable_per_engine("target_server_config_signature")
    same_trace = True
    same_outputs = True
    trace_role_contract = exact_repeat_matrix
    if trace_role_contract:
        for group in repeat_groups.values():
            by_engine = {record.get("engine"): record for record in group}
            source = by_engine.get("sglang")
            replays = [by_engine.get(engine) for engine in ("wkvm", "vllm")]
            if (
                source is None
                or source.get("artifact_role") != SOURCE_ROLE
                or source.get("trace_sha256") is None
                or any(
                    replay is None
                    or replay.get("artifact_role") != "http_teacher_forced_replay"
                    for replay in replays
                )
            ):
                trace_role_contract = False
                break
            group_trace_values = {
                record.get("trace_sha256") for record in group
            }
            group_output_values = {
                record.get("output_signature") for record in group
            }
            if len(group_trace_values) != 1:
                same_trace = False
            if len(group_output_values) != 1:
                same_outputs = False
            if not all(
                record.get("trace_output_binding") is True for record in group
            ):
                trace_role_contract = False
            if not same_trace or not same_outputs:
                trace_role_contract = False
                break
    else:
        same_trace = False
        same_outputs = False

    within_memory_ceiling = (
        memory_ceiling_mib is not None
        and bool(records)
        and all(
            record.get("memory_scope_valid") is True
            and record.get("peak_used_mib") is not None
            and float(record["peak_used_mib"]) <= float(memory_ceiling_mib)
            for record in records
        )
    )
    return {
        "benchmark_identity": identities_complete,
        "unique_artifacts": unique_artifacts,
        "same_campaign": same_campaign,
        "minimum_repeats": min_repeats >= PUBLICATION_MIN_REPEATS,
        "exact_repeat_matrix": exact_repeat_matrix,
        "clean_worktree": clean_worktree,
        "same_gpu": same_gpu,
        "driver_identity": driver_identity,
        "same_driver": same_driver,
        "same_git_commit": same_git_commit,
        "stable_model_identity": stable_model_identity,
        "stable_engine_versions": stable_engine_versions,
        "target_server_provenance": target_server_provenance,
        "stable_engine_launches": stable_engine_launches,
        "stable_engine_configs": stable_engine_configs,
        "idle_gpu_baseline": idle_gpu_baseline,
        "memory_delta": memory_delta,
        "memory_ceiling_configured": memory_ceiling_mib is not None,
        "within_memory_ceiling": within_memory_ceiling,
        "trace_metadata": trace_metadata,
        "trace_output_binding": trace_output_binding,
        "same_trace": same_trace,
        "same_output_fingerprints": same_outputs,
        "trace_role_contract": trace_role_contract,
    }


def build_report(
    records: Iterable[dict[str, Any]],
    *,
    min_repeats: int = 3,
    whole_device_memory_ceiling_mib: float | None = None,
    claim_scope: str = "continuation",
    strict: bool = False,
) -> dict[str, Any]:
    if min_repeats < 1:
        raise ValueError("min_repeats must be >= 1")
    if whole_device_memory_ceiling_mib is not None and (
        not math.isfinite(float(whole_device_memory_ceiling_mib))
        or whole_device_memory_ceiling_mib <= 0
    ):
        raise ValueError("whole-device memory ceiling must be finite and > 0")
    if claim_scope not in CLAIM_SCOPES:
        raise ValueError(
            f"claim_scope must be one of {', '.join(CLAIM_SCOPES)}"
        )
    rate_field = (
        "continuation_rate"
        if claim_scope == "continuation"
        else "full_session_rate"
    )
    claim_scope_id = (
        "provider_http_warm_stateful_continuation_e2e"
        if claim_scope == "continuation"
        else "provider_http_complete_session_e2e"
    )
    record_list = list(records)
    if not record_list:
        raise ValueError("at least one artifact record is required")
    engine_groups = {
        engine: [
            record for record in record_list if record.get("engine") == engine
        ]
        for engine in ENGINES
    }
    engine_summaries: dict[str, dict[str, Any]] = {}
    for engine, engine_records in engine_groups.items():
        rate_records = [
            record
            for record in engine_records
            if record.get(rate_field) is not None
        ]
        engine_summaries[engine] = {
            "sample_count": len(engine_records),
            "valid_sample_count": len(rate_records),
            "semantic_modes": sorted(
                {
                    str(record["semantic_mode"])
                    for record in engine_records
                    if record.get("semantic_mode") is not None
                }
            ),
            "claim_output_tok_s": _metric_summary(
                [float(record[rate_field]) for record in rate_records]
            ),
            "continuation_output_tok_s": _metric_summary(
                [float(record["continuation_rate"]) for record in rate_records]
            ),
            "full_session_output_tok_s": _metric_summary(
                [float(record["full_session_rate"]) for record in rate_records]
            ),
            "full_session_wall_s": _metric_summary(
                [float(record["full_session_wall_s"]) for record in rate_records]
            ),
            "p50_ttft_s": _metric_summary(
                [float(record["p50_ttft_s"]) for record in rate_records]
            ),
            "p95_ttft_s": _metric_summary(
                [float(record["p95_ttft_s"]) for record in rate_records]
            ),
            "p50_e2e_s": _metric_summary(
                [float(record["p50_e2e_s"]) for record in rate_records]
            ),
            "p95_e2e_s": _metric_summary(
                [float(record["p95_e2e_s"]) for record in rate_records]
            ),
            "errors": sorted(
                {
                    error
                    for record in engine_records
                    for error in record.get("errors", ())
                }
            ),
            "artifacts": [record["path"] for record in engine_records],
        }
    engine_stability, repeat_diagnostics = _build_stability_diagnostics(
        engine_groups,
        rate_field=rate_field,
    )

    wkvm_rate_records = [
        record
        for record in engine_groups["wkvm"]
        if record.get(rate_field) is not None
    ]
    incumbent_rate_records = {
        engine: [
            record
            for record in engine_groups[engine]
            if record.get(rate_field) is not None
        ]
        for engine in INCUMBENTS
    }
    wkvm_min_record = min(
        wkvm_rate_records,
        key=lambda record: float(record[rate_field]),
        default=None,
    )
    incumbent_max_records = {
        engine: max(
            incumbent_rate_records[engine],
            key=lambda record: float(record[rate_field]),
            default=None,
        )
        for engine in INCUMBENTS
    }
    wkvm_min = (
        None
        if wkvm_min_record is None
        else float(wkvm_min_record[rate_field])
    )
    ratios = {
        engine: (
            None
            if wkvm_min is None or incumbent_max_records[engine] is None
            else wkvm_min
            / float(incumbent_max_records[engine][rate_field])
        )
        for engine in INCUMBENTS
    }

    repeat_groups = _build_repeat_groups(record_list)
    complete_repeat_groups = [group for group in repeat_groups if group["complete"]]
    publication_checks = _publication_campaign_checks(
        record_list,
        min_repeats=min_repeats,
        memory_ceiling_mib=whole_device_memory_ceiling_mib,
    )
    workload_signatures = {record.get("workload_signature") for record in record_list}
    artifact_paths = [record.get("path") for record in record_list]
    artifact_hashes = [record.get("payload_sha256") for record in record_list]
    run_ids = [record.get("run_id") for record in record_list]
    memory_ceiling_configured = whole_device_memory_ceiling_mib is not None
    within_memory_ceiling = (
        True
        if not memory_ceiling_configured
        else all(
            record.get("memory_scope_valid") is True
            and record.get("peak_used_mib") is not None
            and float(record["peak_used_mib"])
            <= float(whole_device_memory_ceiling_mib)
            for record in record_list
        )
    )
    dirty_tree_artifacts = [
        {
            "artifact": record["path"],
            **record["tree_state"],
        }
        for record in record_list
        if record["tree_state"]["classification"] != "clean"
    ]
    caveats = [
        f"{item['classification']}_worktree:{item['artifact']}"
        for item in dirty_tree_artifacts
    ]
    fallback_groups = [
        group
        for group in repeat_groups
        if group["linkage_method"] == "per_turn_output_fingerprints"
    ]
    if fallback_groups:
        caveats.append(
            "trace_linkage_fallback:per-turn output fingerprints were used "
            "where one common trace SHA was unavailable"
        )
    identity_fallback = any(
        group["campaign_id"] is None or group["repeat_id"] is None
        for group in repeat_groups
    )
    if identity_fallback:
        caveats.append(
            "repeat_identity_fallback:trace/output identity grouped one or more repeats"
        )
    semantic_modes = {
        engine: engine_summaries[engine]["semantic_modes"]
        for engine in ENGINES
    }
    flattened_semantic_modes = {
        mode for modes in semantic_modes.values() for mode in modes
    }
    if len(flattened_semantic_modes) > 1:
        caveats.append(
            "semantic_mode_difference:"
            + ",".join(
                f"{engine}={'/'.join(semantic_modes[engine]) or 'unknown'}"
                for engine in ENGINES
            )
        )
    for diagnostic in repeat_diagnostics:
        if diagnostic["slow_repeat_candidate"]:
            caveats.append(
                "slow_repeat_candidate:"
                f"{diagnostic['engine']}:"
                f"{diagnostic.get('repeat_id') or diagnostic['artifact']}:"
                f"fraction_of_median={diagnostic['fraction_of_engine_median']:.6f}:"
                f"signals={','.join(diagnostic['signals'])}"
            )

    checks = {
        "all_engines_present": all(engine_groups[engine] for engine in ENGINES),
        "unique_artifacts": (
            len(set(artifact_paths)) == len(artifact_paths)
            and len(set(artifact_hashes)) == len(artifact_hashes)
            and None not in run_ids
            and len(set(run_ids)) == len(run_ids)
        ),
        "minimum_repeats": (
            len(complete_repeat_groups) >= min_repeats
            and all(
                engine_summaries[engine]["valid_sample_count"] >= min_repeats
                for engine in ENGINES
            )
        ),
        "workload_identity": (
            None not in workload_signatures and len(workload_signatures) == 1
        ),
        "complete_success": all(
            record.get("success_complete") is True for record in record_list
        ),
        "exact_output_ids": all(
            record.get("exact_output_ids") is True for record in record_list
        ),
        "semantic_modes_recorded": all(
            record.get("semantic_mode") is not None for record in record_list
        ),
        "continuation_latency_metrics": all(
            record.get("latency_complete") is True for record in record_list
        ),
        "trace_or_output_linkage": (
            bool(repeat_groups)
            and all(group["linkage_method"] is not None for group in repeat_groups)
        ),
        "source_replay_contract": (
            bool(repeat_groups)
            and all(group["source_replay_contract"] for group in repeat_groups)
        ),
        "valid_http_artifacts": all(
            record.get("errors") == [] for record in record_list
        ),
        "within_memory_ceiling": within_memory_ceiling,
        "wkvm_vs_vllm_10x": (
            ratios["vllm"] is not None and ratios["vllm"] >= THRESHOLD
        ),
        "wkvm_vs_sglang_10x": (
            ratios["sglang"] is not None and ratios["sglang"] >= THRESHOLD
        ),
    }
    workload = next(
        (
            record.get("workload")
            for record in record_list
            if isinstance(record.get("workload"), dict)
        ),
        {},
    )
    core_passed = all(checks.values())
    publication_passed = all(publication_checks.values())
    return {
        "schema": SCHEMA,
        "strict": bool(strict),
        "claim_scope": claim_scope_id,
        "claim_scope_short": claim_scope,
        "claim_rate_field": rate_field,
        "threshold": THRESHOLD,
        "minimum_repeats": min_repeats,
        "whole_device_memory_ceiling_mib": whole_device_memory_ceiling_mib,
        "memory_ceiling_configured": memory_ceiling_configured,
        "workload": workload,
        "semantic_comparison": {
            "engines": semantic_modes,
            "identical_modes": len(flattened_semantic_modes) == 1,
        },
        "engines": engine_summaries,
        "stability": {
            "slow_repeat_fraction_threshold": SLOW_REPEAT_FRACTION,
            "low_clock_peer_fraction_threshold": LOW_CLOCK_PEER_FRACTION,
            "low_activity_peer_fraction_threshold": LOW_ACTIVITY_PEER_FRACTION,
            "hot_peer_delta_c_threshold": HOT_PEER_DELTA_C,
            "engines": engine_stability,
            "repeat_diagnostics": repeat_diagnostics,
            "gating_policy": (
                "diagnostic_only; conservative claim ratios already use "
                "min(WKVM) and max(incumbent)"
            ),
        },
        "conservative": {
            "wkvm_min_output_tok_s": wkvm_min,
            "vllm_max_output_tok_s": (
                None
                if incumbent_max_records["vllm"] is None
                else float(incumbent_max_records["vllm"][rate_field])
            ),
            "sglang_max_output_tok_s": (
                None
                if incumbent_max_records["sglang"] is None
                else float(incumbent_max_records["sglang"][rate_field])
            ),
        },
        "ratios": ratios,
        "ratio_witnesses": {
            "wkvm_min": _witness(wkvm_min_record, rate_field=rate_field),
            "vllm_max": _witness(
                incumbent_max_records["vllm"], rate_field=rate_field
            ),
            "sglang_max": _witness(
                incumbent_max_records["sglang"], rate_field=rate_field
            ),
        },
        "repeat_groups": repeat_groups,
        "complete_repeat_count": len(complete_repeat_groups),
        "checks": checks,
        "publication_checks": publication_checks,
        "publication_passed": publication_passed,
        "dirty_tree_artifacts": dirty_tree_artifacts,
        "caveats": caveats,
        "artifacts": [
            {
                key: record.get(key)
                for key in (
                    "path",
                    "engine",
                    "campaign_id",
                    "repeat_id",
                    "run_id",
                    "artifact_role",
                    "semantic_mode",
                    "model_identity",
                    "git_commit",
                    "engine_version_identity",
                    "target_server_launch_signature",
                    "target_server_config_signature",
                    "trace_sha256",
                    "continuation_rate",
                    "continuation_wall_s",
                    "full_session_rate",
                    "full_session_wall_s",
                    "p50_ttft_s",
                    "p95_ttft_s",
                    "p50_e2e_s",
                    "p95_e2e_s",
                    "observed_output_id_requests",
                    "exact_output_id_requests",
                    "peak_used_mib",
                    "baseline_used_mib",
                    "peak_delta_mib",
                    "gpu_name",
                    "gpu_uuid",
                    "gpu_driver_version",
                    "runtime_telemetry",
                    "errors",
                )
            }
            for record in record_list
        ],
        "core_passed": core_passed,
        "passed": core_passed and (publication_passed if strict else True),
    }


def _format_number(value: Any, digits: int = 3) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    scope_short = report.get("claim_scope_short", "continuation")
    semantic_modes = report.get("semantic_comparison", {}).get("engines", {})
    semantic_summary = ", ".join(
        f"{engine}=`{'/'.join(semantic_modes.get(engine, [])) or 'unknown'}`"
        for engine in ENGINES
    )
    lines = [
        f"# Provider-HTTP 10x {scope_short} gate",
        "",
        f"Gate mode: `{'strict publication' if report.get('strict') else 'exploratory'}`.",
        f"Scope: `{report['claim_scope']}`.",
        f"Required repeats: `{report['minimum_repeats']}`.",
        f"Whole-device memory ceiling: `{report['whole_device_memory_ceiling_mib']}` MiB.",
        f"Semantic modes: {semantic_summary}.",
        (
            "Semantic modes are identical."
            if report.get("semantic_comparison", {}).get("identical_modes")
            else "Semantic modes differ; ratios apply only to the named modes."
        ),
        "",
        "| Engine | Valid samples | Claim output tok/s min / median / max | Full wall s min / median / max | Worst continuation p50 / p95 TTFT (s) | Worst continuation p50 / p95 E2E (s) | Errors |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for engine in ENGINES:
        metrics = report["engines"][engine]
        rates = metrics["claim_output_tok_s"]
        full_walls = metrics["full_session_wall_s"]
        lines.append(
            f"| {engine} | {metrics['valid_sample_count']} | "
            f"{_format_number(rates['min'])} / "
            f"{_format_number(rates['median'])} / "
            f"{_format_number(rates['max'])} | "
            f"{_format_number(full_walls['min'], 6)} / "
            f"{_format_number(full_walls['median'], 6)} / "
            f"{_format_number(full_walls['max'], 6)} | "
            f"{_format_number(metrics['p50_ttft_s']['max'], 6)} / "
            f"{_format_number(metrics['p95_ttft_s']['max'], 6)} | "
            f"{_format_number(metrics['p50_e2e_s']['max'], 6)} / "
            f"{_format_number(metrics['p95_e2e_s']['max'], 6)} | "
            f"{', '.join(metrics['errors']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "| Conservative ratio | Value | Gate |",
            "|---|---:|---|",
            f"| min(WKVM) / max(vLLM) | {_format_number(report['ratios']['vllm'], 6)} | "
            f"{'PASS' if report['checks']['wkvm_vs_vllm_10x'] else 'FAIL'} |",
            f"| min(WKVM) / max(SGLang) | {_format_number(report['ratios']['sglang'], 6)} | "
            f"{'PASS' if report['checks']['wkvm_vs_sglang_10x'] else 'FAIL'} |",
            "",
            "## Repeat stability diagnostics",
            "",
            (
                "Diagnostics are non-gating; the claim gate already uses the "
                "slowest WKVM repeat and fastest incumbent repeat."
            ),
            "",
            "| Engine | Repeat | Output tok/s | Fraction of median | Active samples | Active SM MHz | Active GPU util % | Active power W | Max temp C | Signals |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for diagnostic in report["stability"]["repeat_diagnostics"]:
        telemetry = diagnostic["runtime_telemetry"]
        signals = ", ".join(diagnostic["signals"]) or "none"
        lines.append(
            f"| {diagnostic['engine']} | "
            f"{diagnostic.get('repeat_id') or 'n/a'} | "
            f"{_format_number(diagnostic['claim_output_tok_s'])} | "
            f"{_format_number(diagnostic['fraction_of_engine_median'], 6)} | "
            f"{_format_number(telemetry.get('active_sample_count'), 0)} | "
            f"{_format_number(telemetry.get('active_sm_clock_mhz_mean'))} | "
            f"{_format_number(telemetry.get('active_gpu_utilization_percent_mean'))} | "
            f"{_format_number(telemetry.get('active_power_draw_w_mean'))} | "
            f"{_format_number(telemetry.get('temperature_gpu_c_max'))} | "
            f"{signals} |"
        )
    lines.extend(
        [
            "",
            "## Trace linkage",
            "",
            "| Repeat | Source | Link method | Complete |",
            "|---|---|---|---|",
        ]
    )
    for group in report["repeat_groups"]:
        repeat = group.get("repeat_id") or group["key"]
        lines.append(
            f"| {repeat} | {group.get('source_engine') or 'n/a'} | "
            f"{group.get('linkage_method') or 'none'} | "
            f"{'yes' if group.get('complete') else 'no'} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Publication checks", ""])
    for name, passed in report.get("publication_checks", {}).items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Dirty-tree caveats", ""])
    if report["dirty_tree_artifacts"]:
        for item in report["dirty_tree_artifacts"]:
            lines.append(
                f"- `{item['classification']}`: `{item['artifact']}` "
                f"(tracked changes: `{item['tracked_changed_path_count']}`, "
                f"untracked paths: `{item['untracked_path_count']}`, "
                f"status SHA: `{item['status_sha256']}`)"
            )
    else:
        lines.append("- None recorded.")
    for caveat in report["caveats"]:
        if caveat.startswith("dirty_worktree:") or caveat.startswith(
            "unknown_worktree:"
        ):
            continue
        lines.append(f"- {caveat}")
    lines.extend(
        [
            "",
            f"**Gate: {'PASS' if report['passed'] else 'FAIL'}**",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument(
        "--claim-scope",
        choices=CLAIM_SCOPES,
        default="continuation",
    )
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--whole-device-memory-ceiling-mib",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="write reports and return success even when the gate fails",
    )
    parser.add_argument(
        "--strict",
        "--strict-publication",
        dest="strict",
        action="store_true",
        help="require the publication provenance and repeat-matrix checks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        load_records(args.artifacts),
        min_repeats=args.min_repeats,
        whole_device_memory_ceiling_mib=(
            args.whole_device_memory_ceiling_mib
        ),
        claim_scope=args.claim_scope,
        strict=args.strict,
    )
    markdown = render_markdown(report)
    if args.summary_json is not None:
        atomic_write_json(args.summary_json, report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0 if report["passed"] or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
