#!/usr/bin/env python
"""Validation helpers for the exploratory A800 incumbent profile sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for import_path in (ROOT, EXPERIMENTS):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


HTTP_ARTIFACT_SCHEMA = "wkvm.gemma_multiturn_http_bench.v1"


class ValidationError(ValueError):
    pass


def _fail(message: str) -> None:
    raise ValidationError(message)


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{field} must be a JSON object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{field} must be a JSON array")
    return value


def _equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        _fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")


def _float_equal(actual: Any, expected: Any, field: str) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9)
    ):
        _fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")


def _dtype_equal(actual: Any, expected: str, field: str) -> None:
    normalized = str(actual).lower().removeprefix("torch.")
    if normalized != expected:
        _fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")


def _path_equal(actual: Any, expected: str, field: str) -> None:
    if not isinstance(actual, str):
        _fail(f"{field} must be a path string")
    if Path(actual).resolve() != Path(expected).resolve():
        _fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")


def _load_json(path: str | Path, field: str) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return _object(json.load(stream), field)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot load {field}: {exc}") from exc


def validate_trace(
    path: str | Path,
    *,
    sessions: int,
    turns: int,
    initial_context_tokens: int,
    turn_input_tokens: int,
    output_tokens_per_turn: int,
) -> str:
    from gemma_multiturn_bench import build_workload, load_shared_history_trace

    workload = build_workload(
        sessions=sessions,
        turns=turns,
        initial_context_tokens=initial_context_tokens,
        turn_input_tokens=turn_input_tokens,
        vocab_size=262_144,
    )
    trace = load_shared_history_trace(
        path,
        workload,
        sessions=sessions,
        turns=turns,
        output_tokens_per_turn=output_tokens_per_turn,
        vocab_size=262_144,
    )
    return trace.trace_sha256


def assert_port_unbound(host: str, port: int) -> None:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        try:
            listener.bind((host, port))
        except OSError as exc:
            raise ValidationError(f"TCP port {host}:{port} is already bound: {exc}") from exc


def _listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            lines = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            columns = line.split()
            if len(columns) < 10 or columns[3] != "0A":
                continue
            try:
                local_port = int(columns[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port:
                inodes.add(columns[9])
    return inodes


def prove_listener_owned(port: int, process_group: int) -> list[int]:
    inodes = _listening_socket_inodes(port)
    if not inodes:
        _fail(f"no listening socket found for TCP port {port}")
    owners: list[int] = []
    for process_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(process_dir.name)
            if os.getpgid(pid) != process_group:
                continue
            descriptors = (process_dir / "fd").iterdir()
        except (OSError, ValueError):
            continue
        try:
            for descriptor in descriptors:
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    owners.append(pid)
                    break
        except OSError:
            continue
    if not owners:
        _fail(
            f"TCP port {port} is not owned by runner process group {process_group}"
        )
    return sorted(owners)


def _validate_vllm_info(
    payload: dict[str, Any],
    requested: dict[str, Any],
    *,
    model_path: str,
    served_model_name: str,
) -> None:
    config = _object(payload.get("vllm_config"), "server_info.vllm_config")
    model = _object(config.get("model_config"), "vllm_config.model_config")
    cache = _object(config.get("cache_config"), "vllm_config.cache_config")
    scheduler = _object(
        config.get("scheduler_config"), "vllm_config.scheduler_config"
    )
    compilation = _object(
        config.get("compilation_config"), "vllm_config.compilation_config"
    )
    environment = _object(payload.get("vllm_env"), "server_info.vllm_env")
    _path_equal(model.get("model"), model_path, "vllm_config.model_config.model")
    served = model.get("served_model_name")
    if isinstance(served, list):
        if served_model_name not in served:
            _fail("vllm_config.model_config.served_model_name does not contain target")
    else:
        _equal(served, served_model_name, "vllm_config.model_config.served_model_name")
    _dtype_equal(model.get("dtype"), "bfloat16", "vllm_config.model_config.dtype")
    logits_processors = model.get("logits_processors")
    _equal(
        bool(logits_processors),
        requested["custom_logits_processors_enabled"],
        "vllm_config.model_config.logits_processors enabled",
    )
    _equal(
        model.get("max_model_len"),
        requested["max_model_len"],
        "vllm_config.model_config.max_model_len",
    )
    multimodal = _object(
        model.get("multimodal_config"), "vllm_config.model_config.multimodal_config"
    )
    _equal(
        multimodal.get("language_model_only"),
        True,
        "vllm_config.model_config.multimodal_config.language_model_only",
    )
    for field in ("enable_prefix_caching", "kv_sharing_fast_prefill"):
        _equal(cache.get(field), requested[field], f"vllm_config.cache_config.{field}")
    _float_equal(
        cache.get("gpu_memory_utilization"),
        requested["gpu_memory_utilization"],
        "vllm_config.cache_config.gpu_memory_utilization",
    )
    for field in ("enable_chunked_prefill", "max_num_batched_tokens", "max_num_seqs"):
        _equal(
            scheduler.get(field),
            requested[field],
            f"vllm_config.scheduler_config.{field}",
        )
    requested_compilation = _object(
        requested.get("compilation_config"), "requested.compilation_config"
    )
    _equal(
        compilation.get("mode"),
        requested_compilation["mode"],
        "vllm_config.compilation_config.mode",
    )
    graph_values = {
        "FULL_AND_PIECEWISE": [2, 1],
        "FULL_DECODE_ONLY": [2, 0],
    }
    expected_graph = graph_values[requested_compilation["cudagraph_mode"]]
    _equal(
        compilation.get("cudagraph_mode"),
        expected_graph,
        "vllm_config.compilation_config.cudagraph_mode",
    )
    _equal(
        compilation.get("cudagraph_capture_sizes"),
        requested_compilation["cudagraph_capture_sizes"],
        "vllm_config.compilation_config.cudagraph_capture_sizes",
    )
    expected_v2 = requested["use_v2_model_runner"]
    _equal(
        requested["VLLM_USE_V2_MODEL_RUNNER"],
        expected_v2,
        "requested.VLLM_USE_V2_MODEL_RUNNER",
    )
    _equal(
        requested["model_runner_generation"],
        "v2" if expected_v2 else "v1",
        "requested.model_runner_generation",
    )
    raw_runner = environment.get("VLLM_USE_V2_MODEL_RUNNER")
    if isinstance(raw_runner, str):
        normalized = raw_runner.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            raw_runner = True
        elif normalized in {"0", "false", "no", "off"}:
            raw_runner = False
    _equal(
        raw_runner,
        expected_v2,
        "vllm_env.VLLM_USE_V2_MODEL_RUNNER",
    )
    _equal(
        bool(raw_runner),
        expected_v2,
        "vllm_config.use_v2_model_runner derived from explicit runtime env",
    )


def _validate_sglang_info(
    payload: dict[str, Any],
    requested: dict[str, Any],
    *,
    version: str,
    model_path: str,
    served_model_name: str,
) -> None:
    _equal(payload.get("version"), version, "server_info.version")
    _path_equal(payload.get("model_path"), model_path, "server_info.model_path")
    _equal(payload.get("served_model_name"), served_model_name, "server_info.served_model_name")
    _dtype_equal(payload.get("dtype"), "bfloat16", "server_info.dtype")
    for field in (
        "context_length",
        "max_total_tokens",
        "chunked_prefill_size",
        "max_running_requests",
        "disable_radix_cache",
        "disable_chunked_prefix_cache",
        "disable_overlap_schedule",
        "disable_cuda_graph",
        "disable_decode_cuda_graph",
        "disable_prefill_cuda_graph",
        "enable_cache_report",
        "enable_custom_logit_processor",
        "enable_multimodal",
        "enable_torch_compile",
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "skip_tokenizer_init",
        "sampling_defaults",
    ):
        _equal(payload.get(field), requested[field], f"server_info.{field}")
    _float_equal(
        payload.get("mem_fraction_static"),
        requested["mem_fraction_static"],
        "server_info.mem_fraction_static",
    )
    expected_backend = requested["attention_backend_requested"]
    if expected_backend == "auto":
        resolved_backend = payload.get("attention_backend")
        if (
            not isinstance(resolved_backend, str)
            or not resolved_backend.strip()
            or resolved_backend.lower() == "auto"
        ):
            _fail(
                "server_info.attention_backend must record the backend resolved "
                "from the requested auto setting"
            )
    else:
        _equal(
            payload.get("attention_backend"),
            expected_backend,
            "server_info.attention_backend",
        )
    graph = _object(payload.get("cuda_graph_config"), "server_info.cuda_graph_config")
    decode = _object(graph.get("decode"), "server_info.cuda_graph_config.decode")
    prefill = _object(graph.get("prefill"), "server_info.cuda_graph_config.prefill")
    _equal(
        decode.get("backend"),
        requested["cuda_graph_backend_decode"],
        "server_info.cuda_graph_config.decode.backend",
    )
    _equal(
        prefill.get("backend"),
        requested["cuda_graph_backend_prefill"],
        "server_info.cuda_graph_config.prefill.backend",
    )


def validate_server_info_payload(
    payload: dict[str, Any],
    *,
    engine: str,
    requested: dict[str, Any],
    version: str,
    model_path: str,
    served_model_name: str,
) -> None:
    if engine == "vllm":
        _validate_vllm_info(
            payload,
            requested,
            model_path=model_path,
            served_model_name=served_model_name,
        )
    elif engine == "sglang":
        _validate_sglang_info(
            payload,
            requested,
            version=version,
            model_path=model_path,
            served_model_name=served_model_name,
        )
    else:
        _fail(f"unsupported incumbent engine: {engine!r}")


def validate_server_info(
    path: str | Path,
    *,
    engine: str,
    requested: dict[str, Any],
    version: str,
    model_path: str,
    served_model_name: str,
) -> None:
    validate_server_info_payload(
        _load_json(path, "server info"),
        engine=engine,
        requested=requested,
        version=version,
        model_path=model_path,
        served_model_name=served_model_name,
    )


def validate_artifact(
    path: str | Path,
    *,
    engine: str,
    profile: str,
    campaign_id: str,
    run_id: str,
    trace_mode: str,
    trace_sha256: str,
    trace_path: str,
    version: str,
    model_path: str,
    served_model_name: str,
    gpu_selector: str,
    sessions: int,
    turns: int,
    initial_context_tokens: int,
    turn_input_tokens: int,
    output_tokens_per_turn: int,
    requested: dict[str, Any],
) -> None:
    payload = _load_json(path, "benchmark artifact")
    actual_trace_sha256 = validate_trace(
        trace_path,
        sessions=sessions,
        turns=turns,
        initial_context_tokens=initial_context_tokens,
        turn_input_tokens=turn_input_tokens,
        output_tokens_per_turn=output_tokens_per_turn,
    )
    _equal(actual_trace_sha256, trace_sha256, "validated trace SHA-256")
    autonomous_source = trace_mode == "autonomous_source"
    _equal(payload.get("schema"), HTTP_ARTIFACT_SCHEMA, "artifact.schema")
    _equal(payload.get("engine"), engine, "artifact.engine")
    _equal(payload.get("engine_version"), version, "artifact.engine_version")
    _equal(payload.get("semantic_mode"), "full_kv", "artifact.semantic_mode")
    _equal(payload.get("model"), served_model_name, "artifact.model")
    if payload.get("fatal_error") is not None:
        _fail(f"artifact contains fatal_error: {payload['fatal_error']!r}")

    identity = _object(payload.get("benchmark_identity"), "artifact.benchmark_identity")
    _equal(identity.get("campaign_id"), campaign_id, "benchmark_identity.campaign_id")
    _equal(identity.get("repeat_id"), profile, "benchmark_identity.repeat_id")
    _equal(identity.get("run_id"), run_id, "benchmark_identity.run_id")
    _equal(
        identity.get("artifact_role"),
        "http_trace_source" if autonomous_source else "http_teacher_forced_replay",
        "benchmark_identity.artifact_role",
    )

    trace = _object(payload.get("history_trace"), "artifact.history_trace")
    if autonomous_source:
        _equal(trace.get("shared"), False, "history_trace.shared")
        _equal(trace.get("teacher_forced"), False, "history_trace.teacher_forced")
        _equal(trace.get("mode"), "engine_generated", "history_trace.mode")
        emitted = _object(
            payload.get("emitted_history_trace"),
            "artifact.emitted_history_trace",
        )
        _equal(
            emitted.get("schema"),
            "wkvm.gemma_shared_history_trace.v1",
            "emitted_history_trace.schema",
        )
        _equal(
            emitted.get("trace_sha256"),
            trace_sha256,
            "emitted_history_trace.trace_sha256",
        )
        _equal(emitted.get("turn_count"), turns, "emitted_history_trace.turn_count")
        _path_equal(
            emitted.get("source_path"),
            trace_path,
            "emitted_history_trace.source_path",
        )
        trace_payload = _load_json(trace_path, "emitted shared-history trace")
        trace_source = _object(
            trace_payload.get("source"), "emitted shared-history trace.source"
        )
        for field, expected in (
            ("engine", engine),
            ("engine_version", version),
            ("model", served_model_name),
            ("campaign_id", campaign_id),
            ("repeat_id", profile),
            ("run_id", run_id),
        ):
            _equal(trace_source.get(field), expected, f"emitted trace source.{field}")
        _path_equal(
            trace_source.get("benchmark_artifact"),
            str(path),
            "emitted trace source.benchmark_artifact",
        )
    else:
        _equal(trace.get("shared"), True, "history_trace.shared")
        _equal(trace.get("teacher_forced"), True, "history_trace.teacher_forced")
        _equal(trace.get("mode"), "shared_teacher_forced_http", "history_trace.mode")
        _equal(trace.get("trace_sha256"), trace_sha256, "history_trace.trace_sha256")
        _equal(trace.get("turn_count"), turns, "history_trace.turn_count")
        _path_equal(trace.get("source_path"), trace_path, "history_trace.source_path")

    workload = _object(payload.get("workload"), "artifact.workload")
    expected_workload = {
        "sessions": sessions,
        "turns": turns,
        "initial_context_tokens": initial_context_tokens,
        "turn_input_tokens": turn_input_tokens,
        "output_tokens_per_turn": output_tokens_per_turn,
    }
    for field, expected in expected_workload.items():
        _equal(workload.get(field), expected, f"workload.{field}")
    _equal(workload.get("synchronized_turn_barriers"), True, "workload.synchronized_turn_barriers")

    hook = _object(payload.get("teacher_forcing_hook"), "artifact.teacher_forcing_hook")
    _equal(
        hook.get("enabled"),
        not autonomous_source,
        "teacher_forcing_hook.enabled",
    )
    _equal(
        hook.get("processor_present"),
        engine == "sglang" and not autonomous_source,
        "teacher_forcing_hook.processor_present",
    )

    provenance = _object(payload.get("provenance"), "artifact.provenance")
    engine_info = _object(provenance.get("engine"), "provenance.engine")
    _equal(engine_info.get("label"), engine, "provenance.engine.label")
    _equal(engine_info.get("version"), version, "provenance.engine.version")
    _equal(engine_info.get("version_source"), "runtime_import", "provenance.engine.version_source")
    client = _object(provenance.get("client_environment"), "provenance.client_environment")
    _equal(client.get("cuda_visible_devices"), gpu_selector, "provenance.client_environment.cuda_visible_devices")
    target = _object(provenance.get("target_server"), "provenance.target_server")
    _equal(target.get("config"), requested, "provenance.target_server.config")
    if not target.get("launch_command") or not target.get("launch_profile"):
        _fail("provenance.target_server launch command/profile is missing")

    gpu_memory = _object(payload.get("gpu_memory"), "artifact.gpu_memory")
    _equal(gpu_memory.get("enabled"), True, "gpu_memory.enabled")
    _equal(gpu_memory.get("within_memory_ceiling"), True, "gpu_memory.within_memory_ceiling")
    if int(gpu_memory.get("sample_count") or 0) < 1:
        _fail("gpu_memory.sample_count must be >= 1")

    expected_requests = sessions * turns
    expected_outputs = expected_requests * output_tokens_per_turn
    summary = _object(payload.get("summary"), "artifact.summary")
    summary_expectations = {
        "all_turns_recorded": True,
        "completed_turn_rows": turns,
        "requested_turns": turns,
        "turn_rows": turns,
        "request_count": expected_requests,
        "success_count": expected_requests,
        "error_count": 0,
        "output_tokens": expected_outputs,
    }
    for field, expected in summary_expectations.items():
        _equal(summary.get(field), expected, f"summary.{field}")

    rows = _list(payload.get("turns"), "artifact.turns")
    _equal(len(rows), turns, "artifact.turns length")
    for index, raw_row in enumerate(rows):
        row = _object(raw_row, f"artifact.turns[{index}]")
        for field, expected in (
            ("turn_index", index),
            ("request_count", sessions),
            ("success_count", sessions),
            ("error_count", 0),
            ("output_tokens", sessions * output_tokens_per_turn),
            ("response_output_fingerprint_complete", True),
        ):
            _equal(row.get(field), expected, f"turns[{index}].{field}")
        teacher = _object(row.get("teacher_forcing"), f"turns[{index}].teacher_forcing")
        _equal(
            teacher.get("requested"),
            not autonomous_source,
            f"turns[{index}].teacher_forcing.requested",
        )
        _equal(
            teacher.get("trace_sha256"),
            None if autonomous_source else trace_sha256,
            f"turns[{index}].teacher_forcing.trace_sha256",
        )
        verification_count = int(teacher.get("exact_response_verification_count") or 0)
        verification_count += int(teacher.get("hook_contract_verification_count") or 0)
        _equal(
            verification_count,
            0 if autonomous_source else sessions,
            f"turns[{index}].teacher_forcing verification count",
        )
        if autonomous_source:
            _equal(
                row.get("response_token_ids_observed_count"),
                sessions,
                f"turns[{index}].response_token_ids_observed_count",
            )
            requests = _list(row.get("requests"), f"turns[{index}].requests")
            _equal(len(requests), sessions, f"turns[{index}].requests length")
            for request_index, raw_request in enumerate(requests):
                request = _object(
                    raw_request,
                    f"turns[{index}].requests[{request_index}]",
                )
                _equal(
                    request.get("output_token_ids_observed"),
                    True,
                    f"turns[{index}].requests[{request_index}].output_token_ids_observed",
                )
                _equal(
                    request.get("output_token_ids_source"),
                    "response_token_ids",
                    f"turns[{index}].requests[{request_index}].output_token_ids_source",
                )

    _equal(payload.get("server_metrics_error"), None, "artifact.server_metrics_error")
    metrics = _object(payload.get("server_metrics_after_run"), "artifact.server_metrics_after_run")
    validate_server_info_payload(
        metrics,
        engine=engine,
        requested=requested,
        version=version,
        model_path=model_path,
        served_model_name=served_model_name,
    )


def _json_argument(raw: str) -> dict[str, Any]:
    try:
        return _object(json.loads(raw), "--config-json")
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace")
    trace.add_argument("--path", required=True)
    trace.add_argument("--sessions", type=int, required=True)
    trace.add_argument("--turns", type=int, required=True)
    trace.add_argument("--initial-context-tokens", type=int, required=True)
    trace.add_argument("--turn-input-tokens", type=int, required=True)
    trace.add_argument("--output-tokens-per-turn", type=int, required=True)

    port = subparsers.add_parser("port-unbound")
    port.add_argument("--host", required=True)
    port.add_argument("--port", type=int, required=True)

    listener = subparsers.add_parser("listener-owned")
    listener.add_argument("--port", type=int, required=True)
    listener.add_argument("--process-group", type=int, required=True)

    server = subparsers.add_parser("server-info")
    server.add_argument("--path", required=True)
    server.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    server.add_argument("--config-json", type=_json_argument, required=True)
    server.add_argument("--version", required=True)
    server.add_argument("--model-path", required=True)
    server.add_argument("--served-model-name", required=True)

    artifact = subparsers.add_parser("artifact")
    artifact.add_argument("--path", required=True)
    artifact.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    artifact.add_argument("--profile", required=True)
    artifact.add_argument("--campaign-id", required=True)
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument(
        "--trace-mode",
        choices=("teacher_forced_replay", "autonomous_source"),
        required=True,
    )
    artifact.add_argument("--trace-sha256", required=True)
    artifact.add_argument("--trace-path", required=True)
    artifact.add_argument("--version", required=True)
    artifact.add_argument("--model-path", required=True)
    artifact.add_argument("--served-model-name", required=True)
    artifact.add_argument("--gpu-selector", required=True)
    artifact.add_argument("--sessions", type=int, required=True)
    artifact.add_argument("--turns", type=int, required=True)
    artifact.add_argument("--initial-context-tokens", type=int, required=True)
    artifact.add_argument("--turn-input-tokens", type=int, required=True)
    artifact.add_argument("--output-tokens-per-turn", type=int, required=True)
    artifact.add_argument("--config-json", type=_json_argument, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "trace":
        print(
            validate_trace(
                args.path,
                sessions=args.sessions,
                turns=args.turns,
                initial_context_tokens=args.initial_context_tokens,
                turn_input_tokens=args.turn_input_tokens,
                output_tokens_per_turn=args.output_tokens_per_turn,
            )
        )
    elif args.command == "port-unbound":
        assert_port_unbound(args.host, args.port)
    elif args.command == "listener-owned":
        print(",".join(str(pid) for pid in prove_listener_owned(args.port, args.process_group)))
    elif args.command == "server-info":
        validate_server_info(
            args.path,
            engine=args.engine,
            requested=args.config_json,
            version=args.version,
            model_path=args.model_path,
            served_model_name=args.served_model_name,
        )
    elif args.command == "artifact":
        validate_artifact(
            args.path,
            engine=args.engine,
            profile=args.profile,
            campaign_id=args.campaign_id,
            run_id=args.run_id,
            trace_mode=args.trace_mode,
            trace_sha256=args.trace_sha256,
            trace_path=args.trace_path,
            version=args.version,
            model_path=args.model_path,
            served_model_name=args.served_model_name,
            gpu_selector=args.gpu_selector,
            sessions=args.sessions,
            turns=args.turns,
            initial_context_tokens=args.initial_context_tokens,
            turn_input_tokens=args.turn_input_tokens,
            output_tokens_per_turn=args.output_tokens_per_turn,
            requested=args.config_json,
        )


if __name__ == "__main__":
    try:
        main()
    except ValidationError as exc:
        raise SystemExit(f"validation failed: {exc}") from exc
