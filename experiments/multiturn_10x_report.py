#!/usr/bin/env python
"""Build a conservative 10x gate from multi-turn benchmark artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from experiments.bench_prompt_utils import generated_output_fingerprint
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from bench_prompt_utils import generated_output_fingerprint


SCHEMA = "wkvm.gemma_multiturn_10x_report.v1"
BENCH_SCHEMA = "wkvm.gemma_multiturn_bench.v1"
ENGINES = ("wkvm", "vllm", "sglang")
MAX_IDLE_BASELINE_MIB = 1024.0
SEMANTIC_MODES = {"routed_span_approximate", "full_kv"}
SHARED_HISTORY_TRACE_SCHEMA = "wkvm.gemma_shared_history_trace.v1"
SHARED_TEACHER_HISTORY_MODE = "shared_teacher_forced"
SHARED_TEACHER_HISTORY_POLICY = "shared_teacher_forced_token_history"
ENGINE_GENERATED_HISTORY_MODE = "engine_generated"
NATIVE_TRACE_SOURCE_ROLE = "native_trace_source"
TEACHER_FORCED_REPLAY_ROLE = "teacher_forced_replay"
ENGINE_GENERATED_HISTORY_POLICIES = {
    "wkvm": "parked_state_plus_delta",
    "vllm": "cumulative_full_token_history",
    "sglang": "cumulative_full_token_history",
}
TEACHER_FORCING_BACKENDS = {
    "wkvm": "post_sample_pending_token_override",
    "vllm": "vllm_sequence_logits_processor",
    "sglang": "sglang_sequence_logits_processor",
}
GENERATED_OUTPUT_FINGERPRINT_SCHEMA = (
    "wkvm.generated_output_token_ids.sha256.v1"
)
WHOLE_GPU_MEMORY_SCHEMA = "wkvm.whole_gpu_memory.v1"
STRICT_GPU_NAME = "NVIDIA GeForce RTX 4090"
STRICT_WORKLOAD = {
    "sessions": 16,
    "turns": 8,
    "initial_context_tokens": 36_864,
    "turn_input_tokens": 32,
    "output_tokens_per_turn": 64,
    "request_order_policy": "alternating",
    "request_order_seed": 0,
}
STRICT_REQUIRED_MODEL_LEN = (
    STRICT_WORKLOAD["initial_context_tokens"]
    + STRICT_WORKLOAD["turns"] * STRICT_WORKLOAD["output_tokens_per_turn"]
    + (STRICT_WORKLOAD["turns"] - 1) * STRICT_WORKLOAD["turn_input_tokens"]
)
DYNAMIC_ENGINE_CONFIG_FIELDS = {
    "capacity_telemetry",
    "session_telemetry",
}


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _normalized_identity_value(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int)):
        normalized = str(value).strip()
        return normalized or None
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _workload_signature(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        return None
    fingerprints = workload.get("fingerprints")
    if (
        isinstance(fingerprints, dict)
        and "teacher_forced_turn_outputs" not in fingerprints
        and _native_trace_source_identity(payload) is not None
    ):
        emitted_trace = payload.get("emitted_history_trace")
        if isinstance(emitted_trace, dict):
            fingerprints = {
                **fingerprints,
                "teacher_forced_turn_outputs": emitted_trace.get(
                    "output_fingerprints"
                ),
            }
    try:
        fingerprints_key = json.dumps(
            fingerprints,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        fingerprints_key = repr(fingerprints)
    fields = (
        "sessions",
        "turns",
        "initial_context_tokens",
        "turn_input_tokens",
        "output_tokens_per_turn",
        "request_order_policy",
        "request_order_seed",
    )
    return tuple(workload.get(field) for field in fields) + (fingerprints_key,)


def _structural_workload_signature(
    payload: dict[str, Any],
) -> tuple[Any, ...] | None:
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        return None
    fingerprints = workload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        return None
    structural_fingerprints = {
        "initial_prompts": fingerprints.get("initial_prompts"),
        "turn_deltas": fingerprints.get("turn_deltas"),
    }
    try:
        fingerprints_key = json.dumps(
            structural_fingerprints,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
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
    )
    return tuple(workload.get(field) for field in fields) + (fingerprints_key,)


def _history_trace_signature(payload: dict[str, Any]) -> tuple[tuple[int, str], ...] | None:
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return None
    signature: list[tuple[int, str]] = []
    for fallback_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return None
        prompt_hash = turn.get("prompt_token_ids_sha256")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            fingerprint = turn.get("prompt_fingerprint")
            if isinstance(fingerprint, dict):
                prompt_hash = fingerprint.get("prompt_token_ids_sha256")
        if not isinstance(prompt_hash, str) or not prompt_hash:
            return None
        turn_index = turn.get("turn_index", fallback_index)
        try:
            turn_index = int(turn_index)
        except (TypeError, ValueError):
            return None
        signature.append((turn_index, prompt_hash))
    return tuple(signature)


def _shared_trace_metadata_identity(
    trace: Any,
) -> tuple[str, str] | None:
    if not isinstance(trace, dict):
        return None
    schema = trace.get("schema")
    trace_sha256 = trace.get("trace_sha256")
    if schema != SHARED_HISTORY_TRACE_SCHEMA:
        return None
    if not _valid_sha256(trace_sha256):
        return None
    if trace.get("mode") != SHARED_TEACHER_HISTORY_MODE:
        return None
    if trace.get("shared") is not True or trace.get("teacher_forced") is not True:
        return None
    return schema, trace_sha256


def _teacher_replay_trace_identity(
    payload: dict[str, Any],
) -> tuple[str, str] | None:
    return _shared_trace_metadata_identity(payload.get("history_trace"))


def _native_trace_source_identity(
    payload: dict[str, Any],
) -> tuple[str, str] | None:
    history_trace = payload.get("history_trace")
    if not isinstance(history_trace, dict):
        return None
    if history_trace.get("mode") != ENGINE_GENERATED_HISTORY_MODE:
        return None
    if history_trace.get("shared") is not False:
        return None
    if history_trace.get("teacher_forced") is not False:
        return None
    return _shared_trace_metadata_identity(payload.get("emitted_history_trace"))


def _teacher_trace_identity(payload: dict[str, Any]) -> tuple[str, str] | None:
    replay_identity = _teacher_replay_trace_identity(payload)
    source_identity = _native_trace_source_identity(payload)
    if replay_identity is not None and source_identity is not None:
        return None
    return replay_identity or source_identity


def _history_trace_role(payload: dict[str, Any]) -> str | None:
    if _native_trace_source_identity(payload) is not None:
        return NATIVE_TRACE_SOURCE_ROLE
    if _teacher_replay_trace_identity(payload) is not None:
        return TEACHER_FORCED_REPLAY_ROLE
    return None


def _benchmark_identity(payload: dict[str, Any]) -> dict[str, Any] | None:
    identity = payload.get("benchmark_identity")
    if not isinstance(identity, dict):
        return None
    campaign_id = _normalized_identity_value(identity.get("campaign_id"))
    repeat_id = _normalized_identity_value(identity.get("repeat_id"))
    run_id = _normalized_identity_value(identity.get("run_id"))
    source_run_id = _normalized_identity_value(identity.get("source_run_id"))
    memory_ceiling_mib = _positive_number(identity.get("memory_ceiling_mib"))
    if any(
        value is None
        for value in (
            campaign_id,
            repeat_id,
            run_id,
            source_run_id,
            memory_ceiling_mib,
        )
    ):
        return None
    return {
        "campaign_id": campaign_id,
        "repeat_id": repeat_id,
        "run_id": run_id,
        "source_run_id": source_run_id,
        "memory_ceiling_mib": memory_ceiling_mib,
    }


def _model_identity(payload: dict[str, Any]) -> tuple[str, str] | None:
    model_path = payload.get("model_path")
    dtype = payload.get("dtype")
    if not isinstance(model_path, str) or not model_path:
        return None
    if not isinstance(dtype, str) or not dtype:
        return None
    return model_path, dtype


def _engine_version_identity(
    payload: dict[str, Any],
    *,
    engine: str,
) -> str | None:
    version = payload.get("engine_version")
    if isinstance(version, str) and version:
        return version
    commit = payload.get("git_commit")
    if engine == "wkvm" and _valid_git_commit(commit):
        return f"git:{commit}"
    return None


def _engine_config_signature(payload: dict[str, Any]) -> str | None:
    engine_config = payload.get("engine_config")
    if not isinstance(engine_config, dict):
        return None
    stable_config = {
        key: value
        for key, value in engine_config.items()
        if key not in DYNAMIC_ENGINE_CONFIG_FIELDS
    }
    launch_config = payload.get("launch_config")
    launch_environment = (
        launch_config.get("environment")
        if isinstance(launch_config, dict)
        else None
    )
    if launch_environment is not None and not isinstance(launch_environment, dict):
        return None
    try:
        return _canonical_json_sha256(
            {
                "engine_config": stable_config,
                "launch_environment": launch_environment or {},
            }
        )
    except (TypeError, ValueError):
        return None


def _trace_metadata(payload: dict[str, Any]) -> dict[str, Any] | None:
    role = _history_trace_role(payload)
    raw = (
        payload.get("emitted_history_trace")
        if role == NATIVE_TRACE_SOURCE_ROLE
        else payload.get("history_trace")
        if role == TEACHER_FORCED_REPLAY_ROLE
        else None
    )
    return raw if isinstance(raw, dict) else None


def _raw_trace_source_identity(payload: dict[str, Any]) -> dict[str, str] | None:
    source = payload.get("source")
    if not isinstance(source, dict):
        return None
    nested = source.get("benchmark_identity")
    identity = nested if isinstance(nested, dict) else source
    campaign_id = _normalized_identity_value(identity.get("campaign_id"))
    repeat_id = _normalized_identity_value(identity.get("repeat_id"))
    run_id = _normalized_identity_value(identity.get("run_id"))
    if campaign_id is None or repeat_id is None or run_id is None:
        return None
    return {
        "campaign_id": campaign_id,
        "repeat_id": repeat_id,
        "run_id": run_id,
    }


def _raw_trace_validation(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    invalid = {
        "valid": False,
        "path": None,
        "logical_sha256": None,
        "file_sha256": None,
        "source_identity": None,
    }
    metadata = _trace_metadata(payload)
    workload = payload.get("workload")
    if not isinstance(metadata, dict) or not isinstance(workload, dict):
        return invalid
    source_path = metadata.get("source_path")
    if not isinstance(source_path, str) or not source_path:
        return invalid
    trace_path = Path(source_path).expanduser()
    if not trace_path.is_absolute():
        trace_path = path.parent / trace_path
    trace_path = trace_path.resolve()
    try:
        encoded = trace_path.read_bytes()
        raw_trace = json.loads(encoded)
    except (OSError, json.JSONDecodeError):
        return {**invalid, "path": str(trace_path)}
    if not isinstance(raw_trace, dict):
        return {**invalid, "path": str(trace_path)}
    fingerprints = workload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        return {**invalid, "path": str(trace_path)}
    expected_trace_workload = {
        "sessions": workload.get("sessions"),
        "turns": workload.get("turns"),
        "output_tokens_per_turn": workload.get("output_tokens_per_turn"),
        "prompt_token_source": payload.get("prompt_token_source"),
        "fingerprints": {
            "initial_prompts": fingerprints.get("initial_prompts"),
            "turn_deltas": fingerprints.get("turn_deltas"),
        },
    }
    if raw_trace.get("schema") != SHARED_HISTORY_TRACE_SCHEMA:
        return {**invalid, "path": str(trace_path)}
    if raw_trace.get("workload") != expected_trace_workload:
        return {**invalid, "path": str(trace_path)}
    sessions = workload.get("sessions")
    turns = workload.get("turns")
    output_tokens = workload.get("output_tokens_per_turn")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (sessions, turns, output_tokens)
    ):
        return {**invalid, "path": str(trace_path)}
    raw_outputs = raw_trace.get("turn_outputs")
    if not isinstance(raw_outputs, list) or len(raw_outputs) != turns:
        return {**invalid, "path": str(trace_path)}
    request_ids = [f"session-{index:04d}" for index in range(sessions)]
    output_fingerprints: list[dict[str, Any]] = []
    for outputs in raw_outputs:
        if not isinstance(outputs, list) or len(outputs) != sessions:
            return {**invalid, "path": str(trace_path)}
        normalized_outputs: list[list[int]] = []
        for output in outputs:
            if not isinstance(output, list) or len(output) != output_tokens:
                return {**invalid, "path": str(trace_path)}
            if any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in output
            ):
                return {**invalid, "path": str(trace_path)}
            normalized_outputs.append([int(token) for token in output])
        try:
            output_fingerprints.append(
                generated_output_fingerprint(
                    zip(request_ids, normalized_outputs, strict=True)
                )
            )
        except ValueError:
            return {**invalid, "path": str(trace_path)}
    contract = {
        "schema": SHARED_HISTORY_TRACE_SCHEMA,
        "workload": expected_trace_workload,
        "turn_outputs": raw_outputs,
    }
    logical_sha256 = _canonical_json_sha256(contract)
    if raw_trace.get("trace_sha256") != logical_sha256:
        return {**invalid, "path": str(trace_path)}
    if metadata.get("trace_sha256") != logical_sha256:
        return {**invalid, "path": str(trace_path)}
    if raw_trace.get("output_fingerprints") != output_fingerprints:
        return {**invalid, "path": str(trace_path)}
    if metadata.get("output_fingerprints") != output_fingerprints:
        return {**invalid, "path": str(trace_path)}
    if metadata.get("turn_count") != turns:
        return {**invalid, "path": str(trace_path)}
    source_identity = _raw_trace_source_identity(raw_trace)
    if source_identity is None:
        return {**invalid, "path": str(trace_path)}
    return {
        "valid": True,
        "path": str(trace_path),
        "logical_sha256": logical_sha256,
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_identity": source_identity,
    }


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _valid_output_fingerprint(
    fingerprint: Any,
    *,
    sessions: int,
    output_tokens_per_turn: int,
) -> bool:
    if not isinstance(fingerprint, dict):
        return False
    request_ids = [f"session-{index:04d}" for index in range(sessions)]
    return (
        fingerprint.get("schema") == GENERATED_OUTPUT_FINGERPRINT_SCHEMA
        and fingerprint.get("request_count") == sessions
        and fingerprint.get("output_token_count")
        == sessions * output_tokens_per_turn
        and fingerprint.get("request_ids") == request_ids
        and fingerprint.get("output_token_counts")
        == [output_tokens_per_turn] * sessions
        and _valid_sha256(fingerprint.get("request_output_token_ids_sha256"))
    )


def _rounded_equal(value: Any, expected: float, digits: int) -> bool:
    number = _number(value)
    return number is not None and math.isclose(
        number,
        round(expected, digits),
        rel_tol=0.0,
        abs_tol=10 ** (-(digits + 1)),
    )


def _continuation_measurement(payload: dict[str, Any]) -> dict[str, Any]:
    invalid = {
        "valid": False,
        "rate": None,
        "output_tokens": None,
        "wall_s": None,
    }
    workload = payload.get("workload")
    summary = payload.get("summary")
    turn_rows = payload.get("turns")
    if not all(
        isinstance(value, expected_type)
        for value, expected_type in (
            (workload, dict),
            (summary, dict),
            (turn_rows, list),
        )
    ):
        return invalid
    assert isinstance(workload, dict)
    assert isinstance(summary, dict)
    assert isinstance(turn_rows, list)
    sessions = _integer(workload.get("sessions"), minimum=1)
    turns = _integer(workload.get("turns"), minimum=2)
    output_per_turn = _integer(
        workload.get("output_tokens_per_turn"),
        minimum=1,
    )
    if sessions is None or turns is None or output_per_turn is None:
        return invalid
    if len(turn_rows) != turns:
        return invalid
    expected_turn_output = sessions * output_per_turn
    walls: list[float] = []
    for turn_index, turn in enumerate(turn_rows):
        if not isinstance(turn, dict):
            return invalid
        wall_s = _positive_number(turn.get("wall_s"))
        fingerprint = turn.get("generated_output_fingerprint")
        if (
            turn.get("turn_index") != turn_index
            or turn.get("request_count") != sessions
            or turn.get("success_count") != sessions
            or turn.get("error_count") != 0
            or turn.get("output_tokens") != expected_turn_output
            or wall_s is None
            or turn.get("output_fingerprint_complete") is not True
            or not _valid_output_fingerprint(
                fingerprint,
                sessions=sessions,
                output_tokens_per_turn=output_per_turn,
            )
            or turn.get("request_output_token_ids_sha256")
            != fingerprint.get("request_output_token_ids_sha256")
        ):
            return invalid
        walls.append(wall_s)
    continuation_walls = walls[1:]
    continuation_wall = sum(continuation_walls)
    continuation_output = expected_turn_output * (turns - 1)
    if continuation_wall <= 0:
        return invalid
    exact_rate = continuation_output / continuation_wall
    continuation = summary.get("continuation_turns")
    if not isinstance(continuation, dict):
        return invalid
    expected_total_requests = sessions * turns
    expected_continuation_requests = sessions * (turns - 1)
    expected_total_output = expected_turn_output * turns
    if (
        summary.get("requested_turns") != turns
        or summary.get("completed_turn_rows") != turns
        or summary.get("all_turns_recorded") is not True
        or summary.get("turn_rows") != turns
        or summary.get("request_count") != expected_total_requests
        or summary.get("success_count") != expected_total_requests
        or summary.get("error_count") != 0
        or summary.get("output_tokens") != expected_total_output
        or continuation.get("turn_rows") != turns - 1
        or continuation.get("request_count") != expected_continuation_requests
        or continuation.get("success_count") != expected_continuation_requests
        or continuation.get("error_count") != 0
        or continuation.get("output_tokens") != continuation_output
        or continuation.get("wall_scope")
        != "sum_of_synchronized_engine_turn_barriers"
        or not _rounded_equal(continuation.get("wall_s"), continuation_wall, 6)
        or not _rounded_equal(continuation.get("output_tok_s"), exact_rate, 3)
        or continuation.get("cache_telemetry_complete") is not True
    ):
        return invalid
    return {
        "valid": True,
        "rate": exact_rate,
        "output_tokens": continuation_output,
        "wall_s": continuation_wall,
    }


def _expected_request_order(turn_index: int, sessions: int) -> list[str]:
    request_ids = [f"session-{index:04d}" for index in range(sessions)]
    return request_ids if turn_index % 2 == 0 else list(reversed(request_ids))


def _strict_target_artifact(payload: dict[str, Any]) -> bool:
    workload = payload.get("workload")
    sampling = payload.get("sampling")
    turns = payload.get("turns")
    if not isinstance(workload, dict) or not isinstance(sampling, dict):
        return False
    if any(workload.get(key) != value for key, value in STRICT_WORKLOAD.items()):
        return False
    if workload.get("required_model_len") != STRICT_REQUIRED_MODEL_LEN:
        return False
    if (
        payload.get("prompt_token_source") != "synthetic_lcg"
        or sampling.get("temperature") != 0.0
        or sampling.get("top_p") != 1.0
        or sampling.get("ignore_eos") is not True
        or sampling.get("max_output_tokens_per_turn")
        != STRICT_WORKLOAD["output_tokens_per_turn"]
    ):
        return False
    if not isinstance(turns, list) or len(turns) != STRICT_WORKLOAD["turns"]:
        return False
    request_ids = [
        f"session-{index:04d}" for index in range(STRICT_WORKLOAD["sessions"])
    ]
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return False
        requests = turn.get("requests")
        if (
            turn.get("request_order_policy") != "alternating"
            or turn.get("request_order")
            != _expected_request_order(turn_index, STRICT_WORKLOAD["sessions"])
            or not _valid_sha256(turn.get("prompt_token_ids_sha256"))
            or not isinstance(requests, list)
            or len(requests) != STRICT_WORKLOAD["sessions"]
        ):
            return False
        if not all(isinstance(request, dict) for request in requests):
            return False
        if [request.get("session_id") for request in requests] != request_ids:
            return False
        if any(
            request.get("success") is not True
            or request.get("error") is not None
            or request.get("output_tokens")
            != STRICT_WORKLOAD["output_tokens_per_turn"]
            for request in requests
        ):
            return False
    return _continuation_measurement(payload)["valid"] is True


def _engine_limits_sufficient(payload: dict[str, Any], *, engine: str) -> bool:
    config = payload.get("engine_config")
    if not isinstance(config, dict):
        return False
    if engine == "wkvm":
        max_length = _integer(config.get("token_pool_max_context_len"), minimum=1)
        concurrency = _integer(config.get("slots"), minimum=1)
    elif engine == "vllm":
        max_length = _integer(config.get("max_model_len"), minimum=1)
        concurrency = _integer(config.get("max_num_seqs"), minimum=1)
    elif engine == "sglang":
        max_length = _integer(config.get("context_length"), minimum=1)
        concurrency = _integer(config.get("max_running_requests"), minimum=1)
    else:
        return False
    return (
        max_length is not None
        and max_length >= STRICT_REQUIRED_MODEL_LEN
        and concurrency is not None
        and concurrency >= STRICT_WORKLOAD["sessions"]
    )


def _gpu_identity(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    memory = payload.get("gpu_memory")
    if not isinstance(memory, dict):
        return None
    identity = (
        memory.get("device_uuid"),
        memory.get("gpu_name"),
        memory.get("memory_total_mib"),
    )
    if any(value in (None, "") for value in identity):
        return None
    return identity


def _memory_delta_safe(payload: dict[str, Any]) -> bool:
    memory = payload.get("gpu_memory")
    if not isinstance(memory, dict):
        return False
    baseline = _number(memory.get("baseline_used_mib"))
    peak = _number(memory.get("peak_used_mib"))
    delta = _number(memory.get("peak_delta_mib"))
    total = _number(memory.get("memory_total_mib"))
    if any(value is None for value in (baseline, peak, delta, total)):
        return False
    if memory.get("error") not in (None, ""):
        return False
    if int(memory.get("query_error_count") or 0) != 0:
        return False
    if memory.get("schema") != WHOLE_GPU_MEMORY_SCHEMA:
        return False
    if memory.get("scope") != "whole_device" or memory.get("source") != "nvidia-smi":
        return False
    if _positive_number(memory.get("sample_count")) is None:
        return False
    if _positive_number(memory.get("sample_interval_s")) is None:
        return False
    if peak > total:
        return False
    return math.isclose(
        delta,
        peak - baseline,
        rel_tol=0.0,
        abs_tol=1.0,
    )


def _idle_gpu_baseline_safe(payload: dict[str, Any]) -> bool:
    memory = payload.get("gpu_memory")
    if not isinstance(memory, dict):
        return False
    baseline = _number(memory.get("baseline_used_mib"))
    if baseline is None:
        return False
    if memory.get("error") not in (None, ""):
        return False
    if int(memory.get("query_error_count") or 0) != 0:
        return False
    return baseline <= MAX_IDLE_BASELINE_MIB


def _positive_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _cache_enabled(payload: dict[str, Any], *, engine: str) -> bool:
    engine_config = payload.get("engine_config")
    workload = payload.get("workload")
    if not isinstance(engine_config, dict) or not isinstance(workload, dict):
        return False
    if engine == "vllm":
        return (
            engine_config.get("enable_prefix_caching") is True
            and engine_config.get("prefix_caching") is True
        )
    if engine == "sglang":
        return engine_config.get("disable_radix_cache") is False
    if engine != "wkvm":
        return False
    sessions = _positive_number(workload.get("sessions"))
    slots = _positive_number(engine_config.get("slots"))
    return (
        engine_config.get("enable_token_pool_attention") is True
        and _positive_number(engine_config.get("token_pool_capacity")) is not None
        and sessions is not None
        and slots is not None
        and slots >= sessions
    )


def _capacity_telemetry_complete(payload: dict[str, Any], *, engine: str) -> bool:
    engine_config = payload.get("engine_config")
    workload = payload.get("workload")
    if not isinstance(engine_config, dict) or not isinstance(workload, dict):
        return False
    sessions = _positive_number(workload.get("sessions"))
    if sessions is None:
        return False
    if engine == "vllm":
        capacity = engine_config.get("capacity_telemetry")
        return (
            isinstance(capacity, dict)
            and _positive_number(capacity.get("kv_token_capacity")) is not None
            and _positive_number(capacity.get("kv_max_concurrency")) is not None
            and isinstance(capacity.get("capacity_source"), str)
            and bool(capacity.get("capacity_source"))
            and isinstance(capacity.get("capacity_estimated"), bool)
        )
    if engine == "sglang":
        capacity = engine_config.get("capacity_telemetry")
        return (
            isinstance(capacity, dict)
            and _positive_number(capacity.get("effective_token_capacity")) is not None
            and _positive_number(
                capacity.get("configured_max_running_requests")
            )
            is not None
            and isinstance(capacity.get("capacity_source"), str)
            and bool(capacity.get("capacity_source"))
            and capacity.get("capacity_error") in (None, "")
        )
    if engine != "wkvm":
        return False
    metrics = payload.get("engine_metrics_after_close")
    if not isinstance(metrics, dict):
        return False
    token_pool = metrics.get("token_pool")
    return (
        _positive_number(metrics.get("max_resident_sessions")) is not None
        and float(metrics["max_resident_sessions"]) >= sessions
        and _positive_number(metrics.get("max_resident_state_slots")) is not None
        and float(metrics["max_resident_state_slots"]) >= sessions
        and isinstance(token_pool, dict)
        and token_pool.get("enabled") is True
        and _positive_number(token_pool.get("token_slot_capacity")) is not None
    )


def _within_memory_ceiling(
    record: dict[str, Any],
    ceiling_mib: float | None,
) -> bool:
    if ceiling_mib is None:
        return False
    peak = _number(record.get("memory_peak_used_mib"))
    total = _number(record.get("memory_total_mib"))
    return (
        peak is not None
        and total is not None
        and peak <= ceiling_mib <= total
    )


def _teacher_forced_fixed_history(
    payload: dict[str, Any],
    *,
    engine: str,
) -> bool:
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        return False
    if engine not in ENGINES:
        return False
    if workload.get("history_policy") != SHARED_TEACHER_HISTORY_POLICY:
        return False
    trace_identity = _teacher_replay_trace_identity(payload)
    if trace_identity is None:
        return False
    _, trace_sha256 = trace_identity
    if workload.get("history_trace_sha256") != trace_sha256:
        return False
    expected_backend = TEACHER_FORCING_BACKENDS[engine]
    engine_config = payload.get("engine_config")
    if not isinstance(engine_config, dict):
        return False
    if engine_config.get("teacher_forcing_backend") != expected_backend:
        return False
    fingerprints = workload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    if not fingerprints.get("initial_prompts"):
        return False
    turn_deltas = fingerprints.get("turn_deltas")
    turns = workload.get("turns")
    if not isinstance(turn_deltas, list) or not isinstance(turns, int):
        return False
    if turns < 2 or len(turn_deltas) != turns - 1:
        return False
    if not all(isinstance(delta, dict) for delta in turn_deltas):
        return False
    teacher_outputs = fingerprints.get("teacher_forced_turn_outputs")
    if not isinstance(teacher_outputs, list) or len(teacher_outputs) != turns:
        return False
    trace = payload.get("history_trace")
    if not isinstance(trace, dict):
        return False
    if trace.get("turn_count") != turns:
        return False
    if trace.get("output_fingerprints") != teacher_outputs:
        return False
    turn_rows = payload.get("turns")
    if not isinstance(turn_rows, list) or len(turn_rows) != turns:
        return False
    sessions = workload.get("sessions")
    if not isinstance(sessions, int) or sessions < 1:
        return False
    for turn_index, (turn, expected_fingerprint) in enumerate(
        zip(turn_rows, teacher_outputs, strict=True)
    ):
        if not isinstance(turn, dict) or not isinstance(expected_fingerprint, dict):
            return False
        forcing = turn.get("teacher_forcing")
        if not isinstance(forcing, dict):
            return False
        if forcing.get("enabled") is not True:
            return False
        if forcing.get("mode") != SHARED_TEACHER_HISTORY_MODE:
            return False
        if forcing.get("backend") != expected_backend:
            return False
        if forcing.get("trace_sha256") != trace_sha256:
            return False
        if forcing.get("selected_outputs_match_trace") is not True:
            return False
        if forcing.get("selected_output_exact_rows") != sessions:
            return False
        if forcing.get("request_count") != sessions:
            return False
        if forcing.get("teacher_output_fingerprint") != expected_fingerprint:
            return False
        if turn.get("request_count") != sessions:
            return False
        if turn.get("success_count") != sessions or turn.get("error_count") != 0:
            return False
        if turn.get("output_fingerprint_complete") is not True:
            return False
        if turn.get("generated_output_fingerprint") != expected_fingerprint:
            return False
        expected_hash = expected_fingerprint.get(
            "request_output_token_ids_sha256"
        )
        if not isinstance(expected_hash, str) or not expected_hash:
            return False
        if turn.get("request_output_token_ids_sha256") != expected_hash:
            return False
        try:
            recorded_turn_index = int(turn.get("turn_index", -1))
        except (TypeError, ValueError):
            return False
        if recorded_turn_index != turn_index:
            return False
    return True


def _native_trace_source_zero_overhead(payload: dict[str, Any]) -> bool:
    if _native_trace_source_identity(payload) is None:
        return False
    sampling = payload.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("teacher_forced") is not False:
        return False
    engine_config = payload.get("engine_config")
    if not isinstance(engine_config, dict):
        return False
    if engine_config.get("history_mode") != ENGINE_GENERATED_HISTORY_MODE:
        return False
    if engine_config.get("teacher_forcing_backend") is not None:
        return False
    if engine_config.get("teacher_forcing_overhead_contract") is not None:
        return False
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return False
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        forcing = turn.get("teacher_forcing")
        if not isinstance(forcing, dict):
            return False
        if forcing.get("enabled") is not False:
            return False
        if forcing.get("mode") != ENGINE_GENERATED_HISTORY_MODE:
            return False
        if forcing.get("backend") is not None:
            return False
        if forcing.get("overhead_contract") is not None:
            return False
        if forcing.get("trace_sha256") is not None:
            return False
    return True


def _native_trace_source_fixed_history(
    payload: dict[str, Any],
    *,
    engine: str,
) -> bool:
    if engine not in ENGINES:
        return False
    trace_identity = _native_trace_source_identity(payload)
    if trace_identity is None or not _native_trace_source_zero_overhead(payload):
        return False
    _, trace_sha256 = trace_identity
    workload = payload.get("workload")
    if not isinstance(workload, dict):
        return False
    if workload.get("history_policy") != ENGINE_GENERATED_HISTORY_POLICIES[engine]:
        return False
    if "history_trace_sha256" not in workload:
        return False
    if workload.get("history_trace_sha256") is not None:
        return False
    fingerprints = workload.get("fingerprints")
    if not isinstance(fingerprints, dict):
        return False
    if not fingerprints.get("initial_prompts"):
        return False
    if "teacher_forced_turn_outputs" in fingerprints:
        return False
    turns = workload.get("turns")
    turn_deltas = fingerprints.get("turn_deltas")
    if not isinstance(turns, int) or not isinstance(turn_deltas, list):
        return False
    if turns < 2 or len(turn_deltas) != turns - 1:
        return False
    if not all(isinstance(delta, dict) for delta in turn_deltas):
        return False
    sessions = workload.get("sessions")
    if not isinstance(sessions, int) or sessions < 1:
        return False
    emitted_trace = payload.get("emitted_history_trace")
    if not isinstance(emitted_trace, dict):
        return False
    if emitted_trace.get("trace_sha256") != trace_sha256:
        return False
    if emitted_trace.get("turn_count") != turns:
        return False
    emitted_outputs = emitted_trace.get("output_fingerprints")
    if not isinstance(emitted_outputs, list) or len(emitted_outputs) != turns:
        return False
    turn_rows = payload.get("turns")
    if not isinstance(turn_rows, list) or len(turn_rows) != turns:
        return False
    for turn_index, (turn, expected_fingerprint) in enumerate(
        zip(turn_rows, emitted_outputs, strict=True)
    ):
        if not isinstance(turn, dict) or not isinstance(expected_fingerprint, dict):
            return False
        try:
            recorded_turn_index = int(turn.get("turn_index", -1))
        except (TypeError, ValueError):
            return False
        if recorded_turn_index != turn_index:
            return False
        if turn.get("request_count") != sessions:
            return False
        if turn.get("success_count") != sessions:
            return False
        if turn.get("error_count") != 0:
            return False
        if turn.get("output_fingerprint_complete") is not True:
            return False
        if turn.get("generated_output_fingerprint") != expected_fingerprint:
            return False
        expected_hash = expected_fingerprint.get(
            "request_output_token_ids_sha256"
        )
        if not isinstance(expected_hash, str) or not expected_hash:
            return False
        if turn.get("request_output_token_ids_sha256") != expected_hash:
            return False
    return True


def _fixed_history(
    payload: dict[str, Any],
    *,
    engine: str,
) -> bool:
    role = _history_trace_role(payload)
    if role == NATIVE_TRACE_SOURCE_ROLE:
        return _native_trace_source_fixed_history(payload, engine=engine)
    if role == TEACHER_FORCED_REPLAY_ROLE:
        return _teacher_forced_fixed_history(payload, engine=engine)
    return False


def _bounded_teacher_forcing(payload: dict[str, Any]) -> bool:
    role = _history_trace_role(payload)
    if role == NATIVE_TRACE_SOURCE_ROLE:
        return _native_trace_source_zero_overhead(payload)
    if role != TEACHER_FORCED_REPLAY_ROLE:
        return False
    engine_config = payload.get("engine_config")
    if not isinstance(engine_config, dict):
        return False
    expected = engine_config.get("teacher_forcing_overhead_contract")
    if not isinstance(expected, dict):
        return False
    if expected.get("timed") is not True:
        return False
    if expected.get("full_vocabulary_mask") is not False:
        return False
    mutated = expected.get("gpu_logit_elements_mutated_per_row")
    if isinstance(mutated, bool) or not isinstance(mutated, int):
        return False
    if mutated < 0 or mutated > 1:
        return False
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        return False
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        forcing = turn.get("teacher_forcing")
        if not isinstance(forcing, dict):
            return False
        if forcing.get("overhead_contract") != expected:
            return False
    return True


def _semantic_mode(payload: dict[str, Any], *, engine: str) -> tuple[str | None, bool]:
    raw = payload.get("semantic_mode")
    explicit = isinstance(raw, str) and bool(raw)
    if not explicit:
        raw = "routed_span_approximate" if engine == "wkvm" else "full_kv"
    return raw if isinstance(raw, str) else None, explicit


def _zero_metric(metrics: dict[str, Any], name: str) -> bool:
    if name not in metrics:
        return False
    try:
        return int(metrics[name]) == 0
    except (TypeError, ValueError):
        return False


def _empty_metric_map(metrics: dict[str, Any], name: str) -> bool:
    return name in metrics and isinstance(metrics[name], dict) and not metrics[name]


def _publication_checks(
    payload: dict[str, Any],
    *,
    engine: str,
) -> dict[str, bool]:
    tree = payload.get("git_tree_state")
    tree_clean = isinstance(tree, dict) and tree.get("clean") is True
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        benchmark = provenance.get("benchmark")
        if isinstance(benchmark, dict) and benchmark.get("git_worktree_dirty") is True:
            tree_clean = False

    metrics = payload.get("engine_metrics_after_close")
    if not isinstance(metrics, dict):
        metrics = {}
    if engine == "wkvm":
        no_fallbacks = (
            _zero_metric(metrics, "fallback_decode_model_calls")
            and _zero_metric(metrics, "mixed_batch_fallbacks")
            and _empty_metric_map(metrics, "decode_batch_fallback_reasons")
        )
        no_coverage_splits = _zero_metric(
            metrics, "token_pool_full_attention_coverage_splits"
        )
        no_graph_skips = (
            _zero_metric(metrics, "persistent_padded_decode_cuda_graph_skips")
            and _empty_metric_map(
                metrics, "persistent_padded_decode_cuda_graph_skip_reasons"
            )
            and _zero_metric(metrics, "token_pool_decode_graph_shape_mismatches")
            and _empty_metric_map(
                metrics, "token_pool_decode_graph_shape_mismatch_reasons"
            )
        )
        mixed_calls = _integer(metrics.get("mixed_batch_model_calls"))
        mixed_opportunities = _integer(metrics.get("mixed_batch_opportunities"))
        execution_mode = metrics.get("execution_mode")
        execution_contract = (
            mixed_calls is not None
            and mixed_opportunities is not None
            and (
                (
                    mixed_opportunities == 0
                    and mixed_calls == 0
                    and execution_mode == "partitioned_prefill_decode"
                )
                or (
                    mixed_opportunities > 0
                    and mixed_calls == mixed_opportunities
                    and execution_mode == "mixed_ragged"
                    and _zero_metric(metrics, "mixed_batch_fallbacks")
                )
            )
        )
    else:
        no_fallbacks = True
        no_coverage_splits = True
        no_graph_skips = True
        execution_contract = True

    semantic_mode, semantic_explicit = _semantic_mode(payload, engine=engine)
    semantic_scope = (
        semantic_explicit
        and semantic_mode in SEMANTIC_MODES
        and semantic_mode
        == ("routed_span_approximate" if engine == "wkvm" else "full_kv")
    )
    return {
        "clean_worktree": tree_clean,
        "idle_gpu_baseline": _idle_gpu_baseline_safe(payload),
        "memory_delta": _memory_delta_safe(payload),
        "cache_enabled": _cache_enabled(payload, engine=engine),
        "capacity_telemetry": _capacity_telemetry_complete(
            payload,
            engine=engine,
        ),
        "no_fallbacks": no_fallbacks,
        "no_full_attention_coverage_splits": no_coverage_splits,
        "no_graph_skips": no_graph_skips,
        "fixed_history": _fixed_history(payload, engine=engine),
        "bounded_teacher_forcing": _bounded_teacher_forcing(payload),
        "semantic_scope": semantic_scope,
        "execution_contract": execution_contract,
        "benchmark_identity": _benchmark_identity(payload) is not None,
        "exact_target_workload": _strict_target_artifact(payload),
        "engine_limits": _engine_limits_sufficient(payload, engine=engine),
    }


def _artifact_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    errors: list[str] = []
    if payload.get("schema") != BENCH_SCHEMA:
        errors.append("schema_mismatch")
    engine = str(payload.get("engine", ""))
    if engine not in ENGINES:
        errors.append("unknown_engine")
    workload = payload.get("workload")
    summary = payload.get("summary")
    if not isinstance(workload, dict):
        errors.append("missing_workload")
        workload = {}
    if not isinstance(summary, dict):
        errors.append("missing_summary")
        summary = {}
    continuation = summary.get("continuation_turns")
    if not isinstance(continuation, dict):
        errors.append("missing_continuation_summary")
        continuation = {}
    measurement = _continuation_measurement(payload)
    rate = measurement.get("rate")
    if measurement.get("valid") is not True or rate is None:
        errors.append("invalid_continuation_accounting")
    if summary.get("all_turns_recorded") is not True:
        errors.append("incomplete_turns")
    if int(summary.get("error_count") or 0) != 0:
        errors.append("request_errors")
    if continuation.get("cache_telemetry_complete") is not True:
        errors.append("incomplete_cache_telemetry")

    reuse_failures = 0
    if engine == "wkvm":
        turns = payload.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            errors.append("missing_wkvm_turns")
        else:
            for turn in turns[1:]:
                invariants = turn.get("reuse_invariants")
                if not isinstance(invariants, dict) or invariants.get("passed") is not True:
                    reuse_failures += 1
            if reuse_failures:
                errors.append("wkvm_reuse_invariant_failure")

    publication_checks = _publication_checks(payload, engine=engine)
    semantic_mode, semantic_explicit = _semantic_mode(payload, engine=engine)
    workload_dict = workload if isinstance(workload, dict) else {}
    identity = _benchmark_identity(payload)
    raw_trace = _raw_trace_validation(path, payload)
    engine_metrics = payload.get("engine_metrics_after_close")
    if not isinstance(engine_metrics, dict):
        engine_metrics = {}
    try:
        payload_digest = _canonical_json_sha256(payload)
    except (TypeError, ValueError):
        payload_digest = None

    return {
        "path": str(path),
        "payload_digest": payload_digest,
        "engine": engine,
        "engine_version": payload.get("engine_version"),
        "engine_version_identity": _engine_version_identity(
            payload,
            engine=engine,
        ),
        "engine_config_signature": _engine_config_signature(payload),
        "model_path": payload.get("model_path"),
        "model_name": Path(str(payload.get("model_path", ""))).name,
        "model_identity": _model_identity(payload),
        "workload_signature": _workload_signature(payload),
        "structural_workload_signature": _structural_workload_signature(payload),
        "history_trace_signature": _history_trace_signature(payload),
        "teacher_trace_identity": _teacher_trace_identity(payload),
        "history_trace_role": _history_trace_role(payload),
        "raw_trace": raw_trace,
        "benchmark_identity": identity,
        "campaign_id": identity.get("campaign_id") if identity else None,
        "repeat_id": identity.get("repeat_id") if identity else None,
        "run_id": identity.get("run_id") if identity else None,
        "source_run_id": identity.get("source_run_id") if identity else None,
        "identity_memory_ceiling_mib": (
            identity.get("memory_ceiling_mib") if identity else None
        ),
        "continuation_rate": rate,
        "continuation_output_tokens": measurement.get("output_tokens"),
        "continuation_wall_s": measurement.get("wall_s"),
        "execution_mode": engine_metrics.get("execution_mode"),
        "mixed_batch_model_calls": engine_metrics.get("mixed_batch_model_calls"),
        "mixed_batch_opportunities": engine_metrics.get("mixed_batch_opportunities"),
        "errors": sorted(set(errors)),
        "reuse_failures": reuse_failures,
        "workload": workload_dict,
        "history_policy": workload_dict.get("history_policy"),
        "semantic_mode": semantic_mode,
        "semantic_mode_explicit": semantic_explicit,
        "gpu_identity": _gpu_identity(payload),
        "gpu_name": (
            payload.get("gpu_memory", {}).get("gpu_name")
            if isinstance(payload.get("gpu_memory"), dict)
            else None
        ),
        "git_commit": payload.get("git_commit"),
        "memory_peak_used_mib": (
            payload.get("gpu_memory", {}).get("peak_used_mib")
            if isinstance(payload.get("gpu_memory"), dict)
            else None
        ),
        "memory_total_mib": (
            payload.get("gpu_memory", {}).get("memory_total_mib")
            if isinstance(payload.get("gpu_memory"), dict)
            else None
        ),
        "publication_checks": publication_checks,
    }


def load_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: benchmark artifact must be an object")
        records.append(_artifact_record(path, payload))
    if not records:
        raise ValueError("at least one benchmark artifact is required")
    return records


def _valid_trace_role_contract(records: Iterable[dict[str, Any]]) -> bool:
    role_sets: dict[str, set[str]] = {engine: set() for engine in ENGINES}
    for record in records:
        engine = record.get("engine")
        role = record.get("history_trace_role")
        if engine not in role_sets:
            return False
        if role not in {NATIVE_TRACE_SOURCE_ROLE, TEACHER_FORCED_REPLAY_ROLE}:
            return False
        role_sets[engine].add(role)
    if not all(role_sets[engine] for engine in ENGINES):
        return False
    source_engines = [
        engine
        for engine, roles in role_sets.items()
        if NATIVE_TRACE_SOURCE_ROLE in roles
    ]
    if not source_engines:
        return all(
            roles == {TEACHER_FORCED_REPLAY_ROLE}
            for roles in role_sets.values()
        )
    if len(source_engines) != 1:
        return False
    source_engine = source_engines[0]
    return all(
        roles
        == (
            {NATIVE_TRACE_SOURCE_ROLE}
            if engine == source_engine
            else {TEACHER_FORCED_REPLAY_ROLE}
        )
        for engine, roles in role_sets.items()
    )


def _campaign_publication_checks(
    records: list[dict[str, Any]],
    *,
    min_repeats: int,
    memory_ceiling_mib: float | None,
) -> dict[str, bool]:
    identity_complete = bool(records) and all(
        isinstance(record.get("benchmark_identity"), dict) for record in records
    )
    unique_artifacts = (
        bool(records)
        and len({record.get("path") for record in records}) == len(records)
        and None not in {record.get("payload_digest") for record in records}
        and len({record.get("payload_digest") for record in records}) == len(records)
        and None not in {record.get("run_id") for record in records}
        and len({record.get("run_id") for record in records}) == len(records)
    )
    campaign_ids = {record.get("campaign_id") for record in records}
    same_campaign = identity_complete and None not in campaign_ids and len(campaign_ids) == 1
    repeat_groups: dict[str, list[dict[str, Any]]] = {}
    if identity_complete:
        for record in records:
            repeat_groups.setdefault(str(record["repeat_id"]), []).append(record)
    repeat_matrix = (
        identity_complete
        and len(records) == min_repeats * len(ENGINES)
        and len(repeat_groups) == min_repeats
        and all(
            len(group) == len(ENGINES)
            and sorted(record.get("engine") for record in group) == sorted(ENGINES)
            for group in repeat_groups.values()
        )
    )

    paired_traces = repeat_matrix and same_campaign
    paired_prompts = repeat_matrix
    if paired_traces:
        for repeat_id, group in repeat_groups.items():
            by_engine = {str(record["engine"]): record for record in group}
            source = by_engine["sglang"]
            source_run_id = source.get("run_id")
            expected_source_identity = {
                "campaign_id": source.get("campaign_id"),
                "repeat_id": repeat_id,
                "run_id": source_run_id,
            }
            if (
                source.get("history_trace_role") != NATIVE_TRACE_SOURCE_ROLE
                or source.get("source_run_id") != source_run_id
                or any(
                    by_engine[engine].get("history_trace_role")
                    != TEACHER_FORCED_REPLAY_ROLE
                    for engine in ("wkvm", "vllm")
                )
                or any(
                    record.get("source_run_id") != source_run_id for record in group
                )
                or not all(record.get("raw_trace", {}).get("valid") is True for record in group)
                or len(
                    {
                        record.get("raw_trace", {}).get("logical_sha256")
                        for record in group
                    }
                )
                != 1
                or len(
                    {
                        record.get("raw_trace", {}).get("file_sha256")
                        for record in group
                    }
                )
                != 1
                or any(
                    record.get("raw_trace", {}).get("source_identity")
                    != expected_source_identity
                    for record in group
                )
            ):
                paired_traces = False
                break
    if paired_prompts:
        paired_prompts = all(
            all(record.get("history_trace_signature") is not None for record in group)
            and len(
                {record.get("history_trace_signature") for record in group}
            )
            == 1
            for group in repeat_groups.values()
        )

    commits = {record.get("git_commit") for record in records}
    same_git_commit = (
        bool(records)
        and all(_valid_git_commit(commit) for commit in commits)
        and len(commits) == 1
    )
    model_identities = {record.get("model_identity") for record in records}
    stable_model = (
        bool(records)
        and None not in model_identities
        and len(model_identities) == 1
    )
    stable_versions = all(
        len(
            {
                record.get("engine_version_identity")
                for record in records
                if record.get("engine") == engine
            }
        )
        == 1
        and all(
            record.get("engine_version_identity") is not None
            for record in records
            if record.get("engine") == engine
        )
        for engine in ENGINES
    )
    stable_configs = all(
        len(
            {
                record.get("engine_config_signature")
                for record in records
                if record.get("engine") == engine
            }
        )
        == 1
        and all(
            record.get("engine_config_signature") is not None
            for record in records
            if record.get("engine") == engine
        )
        for engine in ENGINES
    )
    identity_ceiling = (
        identity_complete
        and memory_ceiling_mib is not None
        and all(
            math.isclose(
                float(record["identity_memory_ceiling_mib"]),
                float(memory_ceiling_mib),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            for record in records
        )
    )
    expected_gpu = bool(records) and all(
        record.get("gpu_name") == STRICT_GPU_NAME for record in records
    )
    return {
        "benchmark_identity": identity_complete,
        "unique_artifacts": unique_artifacts,
        "same_campaign": same_campaign,
        "exact_repeat_matrix": repeat_matrix,
        "trace_role_contract": paired_traces,
        "same_teacher_trace": paired_traces,
        "same_prompt_trace": paired_prompts,
        "raw_trace_integrity": paired_traces,
        "same_git_commit": same_git_commit,
        "stable_model_identity": stable_model,
        "stable_engine_versions": stable_versions,
        "stable_engine_configs": stable_configs,
        "identity_memory_ceiling": identity_ceiling,
        "expected_gpu": expected_gpu,
    }


def build_report(
    records: Iterable[dict[str, Any]],
    *,
    min_repeats: int = 3,
    threshold: float = 10.0,
    strict: bool = False,
    whole_device_memory_ceiling_mib: float | None = None,
) -> dict[str, Any]:
    if min_repeats < 1:
        raise ValueError("min_repeats must be >= 1")
    if threshold <= 0 or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be finite and > 0")
    if whole_device_memory_ceiling_mib is not None and (
        not math.isfinite(float(whole_device_memory_ceiling_mib))
        or whole_device_memory_ceiling_mib <= 0
    ):
        raise ValueError("whole-device memory ceiling must be finite and > 0")
    record_list = list(records)
    campaign_records = bool(record_list) and all(
        isinstance(record.get("benchmark_identity"), dict)
        for record in record_list
    )
    groups: dict[str, list[dict[str, Any]]] = {engine: [] for engine in ENGINES}
    for record in record_list:
        engine = record.get("engine")
        if engine in groups:
            groups[engine].append(record)

    signature_key = (
        "structural_workload_signature" if campaign_records else "workload_signature"
    )
    signatures = [record.get(signature_key) for record in record_list]
    common_workload = (
        bool(signatures)
        and all(signature is not None for signature in signatures)
        and len(set(signatures)) == 1
    )
    model_values = [
        record.get("model_identity") if campaign_records else record.get("model_name")
        for record in record_list
    ]
    common_model = (
        bool(model_values)
        and all(model_value is not None and bool(model_value) for model_value in model_values)
        and len(set(model_values)) == 1
    )
    gpu_identities = [record.get("gpu_identity") for record in record_list]
    same_gpu = (
        bool(gpu_identities)
        and all(identity is not None for identity in gpu_identities)
        and len(set(gpu_identities)) == 1
    )
    history_trace_signatures = [
        record.get("history_trace_signature") for record in record_list
    ]
    same_prompt_trace = (
        bool(history_trace_signatures)
        and all(signature is not None for signature in history_trace_signatures)
        and len(set(history_trace_signatures)) == 1
    )
    teacher_trace_identities = [
        record.get("teacher_trace_identity") for record in record_list
    ]
    same_teacher_trace = (
        bool(teacher_trace_identities)
        and all(identity is not None for identity in teacher_trace_identities)
        and len(set(teacher_trace_identities)) == 1
    )
    complete = all(
        record.get("errors") == []
        for record in record_list
    )
    sample_counts = {engine: len(groups[engine]) for engine in ENGINES}
    engine_metrics: dict[str, dict[str, Any]] = {}
    for engine in ENGINES:
        rates = [
            float(record["continuation_rate"])
            for record in groups[engine]
            if record.get("continuation_rate") is not None
        ]
        engine_metrics[engine] = {
            "sample_count": len(groups[engine]),
            "valid_rate_count": len(rates),
            "min_tok_s": min(rates) if rates else None,
            "max_tok_s": max(rates) if rates else None,
            "median_tok_s": (
                (
                    sorted(rates)[len(rates) // 2]
                    if len(rates) % 2
                    else (
                        sorted(rates)[len(rates) // 2 - 1]
                        + sorted(rates)[len(rates) // 2]
                    )
                    / 2.0
                )
                if rates
                else None
            ),
            "errors": sorted(
                {
                    error
                    for record in groups[engine]
                    for error in record.get("errors", ())
                }
            ),
            "artifacts": [record["path"] for record in groups[engine]],
            "run_ids": [record.get("run_id") for record in groups[engine]],
            "repeat_ids": [record.get("repeat_id") for record in groups[engine]],
            "history_policies": sorted(
                {
                    str(record["history_policy"])
                    for record in groups[engine]
                    if record.get("history_policy") is not None
                }
            ),
            "semantic_modes": sorted(
                {
                    str(record["semantic_mode"])
                    for record in groups[engine]
                    if record.get("semantic_mode") is not None
                }
            ),
            "history_trace_roles": sorted(
                {
                    str(record["history_trace_role"])
                    for record in groups[engine]
                    if record.get("history_trace_role") is not None
                }
            ),
            "execution_modes": sorted(
                {
                    str(record["execution_mode"])
                    for record in groups[engine]
                    if record.get("execution_mode") is not None
                }
            ),
        }

    enough_repeats = all(
        engine_metrics[engine]["valid_rate_count"] >= min_repeats
        and not engine_metrics[engine]["errors"]
        for engine in ENGINES
    )

    wkvm_min = engine_metrics["wkvm"]["min_tok_s"]
    incumbent_maxes = {
        engine: engine_metrics[engine]["max_tok_s"]
        for engine in ("vllm", "sglang")
    }
    ratios = {
        engine: (
            None
            if wkvm_min is None or incumbent_maxes[engine] in (None, 0)
            else wkvm_min / incumbent_maxes[engine]
        )
        for engine in incumbent_maxes
    }
    ratio_checks = {
        engine: ratios[engine] is not None and ratios[engine] >= threshold
        for engine in incumbent_maxes
    }
    wkvm_min_record = min(
        (
            record
            for record in groups["wkvm"]
            if record.get("continuation_rate") is not None
        ),
        key=lambda record: float(record["continuation_rate"]),
        default=None,
    )
    incumbent_max_records = {
        engine: max(
            (
                record
                for record in groups[engine]
                if record.get("continuation_rate") is not None
            ),
            key=lambda record: float(record["continuation_rate"]),
            default=None,
        )
        for engine in ("vllm", "sglang")
    }

    def witness(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "run_id": record.get("run_id"),
            "repeat_id": record.get("repeat_id"),
            "artifact": record.get("path"),
            "tok_s": record.get("continuation_rate"),
        }

    workload = next(
        (
            record.get("workload")
            for record in record_list
            if isinstance(record.get("workload"), dict)
        ),
        {},
    )
    sessions = int(workload.get("sessions") or 0)
    output_per_turn = int(workload.get("output_tokens_per_turn") or 0)
    continuation_output_tokens = sessions * output_per_turn
    target_tok_s = (
        max(float(value) for value in incumbent_maxes.values() if value is not None)
        * float(threshold)
        if all(value is not None for value in incumbent_maxes.values())
        else None
    )
    target_wall = (
        continuation_output_tokens / target_tok_s
        if target_tok_s and continuation_output_tokens
        else None
    )
    checks = {
        "all_engines_present": all(sample_counts[engine] > 0 for engine in ENGINES),
        "minimum_repeats": enough_repeats,
        "same_workload": common_workload,
        "same_model": common_model,
        "complete_artifacts": complete,
        "wkvm_reuse_invariants": not any(
            record.get("reuse_failures", 0) for record in groups["wkvm"]
        ),
        "wkvm_vs_vllm": ratio_checks["vllm"],
        "wkvm_vs_sglang": ratio_checks["sglang"],
    }
    publication_check_names = (
        "clean_worktree",
        "idle_gpu_baseline",
        "memory_delta",
        "cache_enabled",
        "capacity_telemetry",
        "no_fallbacks",
        "no_full_attention_coverage_splits",
        "no_graph_skips",
        "fixed_history",
        "bounded_teacher_forcing",
        "semantic_scope",
        "execution_contract",
        "benchmark_identity",
        "exact_target_workload",
        "engine_limits",
    )
    publication_checks = {
        name: bool(record_list)
        and all(
            record.get("publication_checks", {}).get(name) is True
            for record in record_list
        )
        for name in publication_check_names
    }
    publication_checks["same_gpu"] = same_gpu
    publication_checks["memory_ceiling_configured"] = (
        whole_device_memory_ceiling_mib is not None
    )
    publication_checks["within_memory_ceiling"] = bool(record_list) and all(
        _within_memory_ceiling(record, whole_device_memory_ceiling_mib)
        for record in record_list
    )
    campaign_checks = _campaign_publication_checks(
        record_list,
        min_repeats=min_repeats,
        memory_ceiling_mib=whole_device_memory_ceiling_mib,
    )
    if campaign_records:
        publication_checks.update(campaign_checks)
    else:
        publication_checks["same_prompt_trace"] = same_prompt_trace
        publication_checks["same_teacher_trace"] = same_teacher_trace
        publication_checks["trace_role_contract"] = _valid_trace_role_contract(
            record_list
        )
    core_passed = all(checks.values())
    publication_passed = all(publication_checks.values())
    return {
        "schema": SCHEMA,
        "claim_scope": "warm_stateful_continuation_e2e",
        "strict": bool(strict),
        "whole_device_memory_ceiling_mib": whole_device_memory_ceiling_mib,
        "threshold": float(threshold),
        "minimum_repeats": int(min_repeats),
        "workload": workload,
        "engines": engine_metrics,
        "ratios": ratios,
        "ratio_witnesses": {
            "wkvm_min": witness(wkvm_min_record),
            "vllm_max": witness(incumbent_max_records["vllm"]),
            "sglang_max": witness(incumbent_max_records["sglang"]),
        },
        "target": {
            "continuation_output_tokens_per_turn": continuation_output_tokens,
            "minimum_wkvm_tok_s_for_threshold": target_tok_s,
            "maximum_turn_wall_s_for_threshold": target_wall,
        },
        "checks": checks,
        "publication_checks": publication_checks,
        "core_passed": core_passed,
        "publication_passed": publication_passed,
        "passed": core_passed and (publication_passed if strict else True),
    }


def render_markdown(report: dict[str, Any]) -> str:
    engines = report["engines"]
    lines = [
        "# Stateful 10x E2E gate",
        "",
        "Scope: `warm_stateful_continuation_e2e`.",
        f"Gate mode: `{'strict publication' if report.get('strict') else 'exploratory'}`.",
        f"Whole-device memory ceiling: `{report.get('whole_device_memory_ceiling_mib')}` MiB.",
        "",
        "| Engine | Samples | Min tok/s | Median tok/s | Max tok/s | Execution modes | Errors |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for engine in ENGINES:
        metrics = engines[engine]
        lines.append(
            f"| {engine} | {metrics['sample_count']} | "
            f"{metrics['min_tok_s']} | {metrics['median_tok_s']} | "
            f"{metrics['max_tok_s']} | "
            f"{', '.join(metrics['execution_modes']) or 'n/a'} | "
            f"{', '.join(metrics['errors']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "| Ratio | Value | Pass |",
            "|---|---:|---|",
            f"| WKVM / vLLM | {report['ratios']['vllm']} | "
            f"{report['checks']['wkvm_vs_vllm']} |",
            f"| WKVM / SGLang | {report['ratios']['sglang']} | "
            f"{report['checks']['wkvm_vs_sglang']} |",
            "",
            f"**Gate: {'PASS' if report['passed'] else 'FAIL'}**",
            "",
            "Checks:",
        ]
    )
    for name, passed in report["checks"].items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "Publication checks:"])
    for name, passed in report.get("publication_checks", {}).items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    target = report["target"]
    lines.extend(
        [
            "",
            f"Required WKVM rate at threshold: `{target['minimum_wkvm_tok_s_for_threshold']}` tok/s.",
            f"Equivalent continuation-turn wall target: `{target['maximum_turn_wall_s_for_threshold']}` s.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument(
        "--whole-device-memory-ceiling-mib",
        type=float,
        default=None,
        help=(
            "Apply one explicit whole-device peak-memory ceiling to every "
            "artifact. Strict publication mode requires this option."
        ),
    )
    parser.add_argument(
        "--strict",
        "--strict-publication",
        action="store_true",
        help=(
            "require clean provenance, idle-GPU memory evidence, fixed semantic "
            "scope, zero WKVM fallbacks/skips, and a valid recorded execution contract"
        ),
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="write the report without returning a failing exit status",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        load_records(args.artifacts),
        min_repeats=args.min_repeats,
        threshold=args.threshold,
        strict=args.strict,
        whole_device_memory_ceiling_mib=(
            args.whole_device_memory_ceiling_mib
        ),
    )
    markdown = render_markdown(report)
    if args.summary_json is not None:
        _write_json(args.summary_json, report)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0 if report["passed"] or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
