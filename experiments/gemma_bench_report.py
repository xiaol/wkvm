#!/usr/bin/env python
"""Render a normalized Gemma throughput report from benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from .benchmark_contract import ComparisonContractResult, validate_comparable
except ImportError:  # Direct script execution.
    from benchmark_contract import ComparisonContractResult, validate_comparable


SERVING_PROVENANCE_SCHEMAS = frozenset(
    {
        "wkvm.serving_bench.provenance.v1",
        "wkvm.serving_bench.provenance.v2",
    }
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def fmt(x: Any, suffix: str = "") -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}{suffix}"
    return f"{x}{suffix}"


def fmt_bytes(x: Any) -> str:
    if x is None:
        return "-"
    try:
        value = int(x)
    except (TypeError, ValueError):
        return "-"
    if value == 0:
        return "0"
    gib = value / 2**30
    if gib >= 1:
        return f"{gib:.3f} GiB"
    mib = value / 2**20
    if mib >= 1:
        return f"{mib:.1f} MiB"
    return f"{value} B"


def fmt_mib_as_gib(value: Any) -> str:
    if isinstance(value, bool):
        return "-"
    try:
        gib = float(value) / 1024.0
    except (TypeError, ValueError):
        return "-"
    return f"{gib:.3f} GiB" if math.isfinite(gib) else "-"


def load_payloads(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, json.loads(path.read_text())) for path in paths]


def shape_key(data: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        data.get("context_tokens_per_session"),
        data.get("decode_tokens_per_session"),
        data.get("prompt_lengths_mode"),
    )


def fmt_shape(shape: tuple[Any, Any, Any]) -> str:
    ctx, out, prompt_mode = shape
    return f"ctx={fmt(ctx)} out={fmt(out)} prompt={prompt_mode or '-'}"


def validate_same_shape(payloads: list[tuple[Path, dict[str, Any]]]) -> None:
    groups: dict[tuple[Any, Any, Any], list[Path]] = {}
    for path, data in payloads:
        groups.setdefault(shape_key(data), []).append(path)
    if len(groups) <= 1:
        return
    parts = []
    for shape, paths in sorted(groups.items(), key=lambda item: fmt_shape(item[0])):
        parts.append(
            f"{fmt_shape(shape)}: "
            + ", ".join(path.as_posix() for path in sorted(paths))
        )
    raise ValueError(
        "same-shape requirement failed; inputs contain multiple benchmark shapes: "
        + "; ".join(parts)
    )


def row_prompt_fingerprint(row: dict[str, Any]) -> dict[str, Any] | None:
    fingerprint = row.get("prompt_fingerprint")
    if isinstance(fingerprint, dict):
        return fingerprint
    if row.get("prompt_token_ids_sha256") is None:
        return None
    return {
        "schema": row.get("prompt_fingerprint_schema")
        or "wkvm.prompt_token_ids.sha256.v1",
        "prompt_token_source": row.get("prompt_token_source"),
        "prompt_count": row.get("prompt_count"),
        "prompt_total_tokens": row.get("prompt_total_tokens"),
        "prompt_lengths": row.get("prompt_lengths"),
        "prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
    }


def fingerprint_compare_key(fingerprint: dict[str, Any]) -> tuple[Any, ...]:
    lengths = fingerprint.get("prompt_lengths")
    if isinstance(lengths, list):
        lengths_key = tuple(lengths)
    else:
        lengths_key = lengths
    return (
        fingerprint.get("schema"),
        fingerprint.get("prompt_count"),
        fingerprint.get("prompt_total_tokens"),
        lengths_key,
        fingerprint.get("prompt_token_ids_sha256"),
    )


def validate_same_prompt_fingerprint(
    payloads: list[tuple[Path, dict[str, Any]]],
) -> None:
    violations = []
    groups: dict[tuple[Any, Any, Any, Any], list[tuple[Path, dict[str, Any], int]]] = {}
    for path, data in payloads:
        payload_rows = 0
        for index, row in enumerate(data.get("rows", [])):
            B = row.get("B")
            if B is None:
                continue
            payload_rows += 1
            fingerprint = row_prompt_fingerprint(row)
            if fingerprint is None:
                violations.append(
                    f"{path.as_posix()} row={index} B={fmt(B)} missing prompt fingerprint"
                )
                continue
            groups.setdefault((*shape_key(data), B), []).append((path, fingerprint, index))
        if payload_rows == 0:
            violations.append(f"{path.as_posix()} has no benchmark rows to check")
    for group_key, entries in groups.items():
        keys = {fingerprint_compare_key(fingerprint) for _path, fingerprint, _index in entries}
        if len(keys) <= 1:
            continue
        ctx, out, prompt_mode, B = group_key
        parts = []
        for path, fingerprint, index in entries:
            digest = fingerprint.get("prompt_token_ids_sha256")
            digest = "-" if digest is None else str(digest)[:12]
            source = fingerprint.get("prompt_token_source") or "-"
            parts.append(
                f"{path.as_posix()} row={index} source={source} hash={digest}"
            )
        violations.append(
            "shape="
            f"{fmt_shape((ctx, out, prompt_mode))} B={fmt(B)} "
            "prompt fingerprints differ: "
            + ", ".join(parts)
        )
    if violations:
        raise ValueError(
            "same-prompt-fingerprint requirement failed; "
            + "; ".join(violations)
        )


def row_has_full_success(row: dict[str, Any]) -> bool:
    return row.get("success_count") == row.get("B")


def native_no_hf_row_problems(row: dict[str, Any]) -> list[str]:
    problems = []
    if row.get("uses_hf_transformer_forward") is not False:
        problems.append("uses_hf_transformer_forward_not_false")
    if row.get("uses_hf_model_construction") is not False:
        problems.append("uses_hf_model_construction_not_false")
    if row.get("native_gemma_checkpoint_loader") is not True:
        problems.append("native_gemma_checkpoint_loader_not_true")
    return problems


def native_no_hf_setup_problems(data: dict[str, Any]) -> list[str]:
    problems = []
    if data.get("uses_hf_tokenizer") is not False:
        problems.append("uses_hf_tokenizer_not_false")
    if data.get("uses_hf_config") is not False:
        problems.append("uses_hf_config_not_false")
    if data.get("native_gemma_config_loader") is not True:
        problems.append("native_gemma_config_loader_not_true")
    return problems


def validate_native_no_hf(payloads: list[tuple[Path, dict[str, Any]]]) -> None:
    violations = []
    checked_rows = 0
    native_payloads = [
        (path, data) for path, data in payloads if data.get("engine") == "wkvm-native"
    ]
    for path, data in native_payloads:
        setup_problems = native_no_hf_setup_problems(data)
        if setup_problems:
            violations.append((path, "setup", setup_problems))
        payload_checked_rows = 0
        for row in data.get("rows", []):
            if not row_has_full_success(row):
                continue
            checked_rows += 1
            payload_checked_rows += 1
            problems = native_no_hf_row_problems(row)
            if problems:
                violations.append((path, row.get("B"), problems))
        if payload_checked_rows == 0:
            violations.append((path, None, ["no_successful_rows_to_check"]))
    if not native_payloads:
        violations.append((None, None, ["no_wkvm_native_payloads_to_check"]))
    if not violations and checked_rows > 0:
        return
    parts = []
    for path, B, problems in violations:
        source = path.as_posix() if path is not None else "<inputs>"
        parts.append(f"{source} B={fmt(B)} problems={','.join(problems)}")
    raise ValueError(
        "native no-HF requirement failed; "
        "wkvm-native payloads must prove no HF tokenizer/config, no HF model "
        "construction/forward, and native checkpoint/config loading: "
        + "; ".join(parts)
    )


def engine_label(data: dict[str, Any]) -> str:
    engine = data.get("engine", "unknown")
    if engine == "hf-transformers":
        return f"HF Transformers ({data.get('mode', 'unknown')})"
    if engine == "wkvm-native":
        cfg = data.get("config") or {}
        cap = cfg.get("decode_microbatch_bytes")
        rows = cfg.get("decode_microbatch_rows")
        suffix = ""
        if cfg.get("persistent_padded_decode") is False:
            suffix = ", persistent padded off"
        if cap:
            return f"wkvm-native byte-cap {cap}{suffix}"
        return f"wkvm-native row-cap {rows}{suffix}"
    if engine == "wkvm-native-http-stream":
        return "wkvm-native HTTP stream"
    if engine == "wkvm-native-openai-completions":
        return "wkvm-native OpenAI completions"
    if engine == "vllm-http-stream":
        return "vLLM HTTP stream"
    if engine == "sglang-http-stream":
        return "SGLang HTTP stream"
    return str(engine)


def structured_provenance(data: dict[str, Any]) -> dict[str, Any] | None:
    provenance = data.get("provenance")
    if (
        isinstance(provenance, dict)
        and provenance.get("schema") in SERVING_PROVENANCE_SCHEMAS
    ):
        return provenance
    return None


def engine_version_summary(data: dict[str, Any]) -> tuple[str, str]:
    provenance = structured_provenance(data)
    if provenance is not None:
        engine = provenance.get("engine")
        if isinstance(engine, dict) and engine.get("version"):
            return str(engine["version"]), str(engine.get("version_source") or "-")
    config = data.get("engine_config")
    if isinstance(config, dict):
        for key in ("vllm_version", "sglang_version", "engine_version"):
            if config.get(key):
                return str(config[key]), f"engine_config.{key}"
    return "-", "-"


def gpu_provenance_summary(data: dict[str, Any]) -> tuple[str, str, str]:
    provenance = structured_provenance(data)
    gpu = provenance.get("gpu") if provenance is not None else None
    if not isinstance(gpu, dict):
        return "-", "-", "-"
    identity = str(gpu.get("name") or "-")
    if gpu.get("index") is not None:
        identity += f" (index {gpu['index']})"
    total_mib = gpu.get("memory_total_mib")
    total = fmt_mib_as_gib(total_mib)
    return identity, str(gpu.get("driver_version") or "-"), total


def client_runtime_summary(data: dict[str, Any]) -> str:
    provenance = structured_provenance(data)
    client = provenance.get("client_environment") if provenance is not None else None
    if not isinstance(client, dict):
        return "-"
    parts = []
    if client.get("python_version"):
        parts.append(f"Python {client['python_version']}")
    packages = client.get("packages")
    if isinstance(packages, dict):
        parts.extend(
            f"{name} {version}"
            for name, version in packages.items()
            if version is not None
        )
    return "; ".join(parts) or "-"


def target_server_summary(data: dict[str, Any]) -> tuple[str, str, str]:
    provenance = structured_provenance(data)
    target_server = (
        provenance.get("target_server") if provenance is not None else None
    )
    if not isinstance(target_server, dict):
        return "-", "-", "-"
    launch_command = str(target_server.get("launch_command") or "-")
    launch_source = str(target_server.get("launch_command_source") or "-")
    config = target_server.get("config")
    config_summary = (
        json.dumps(config, sort_keys=True, separators=(",", ":"))
        if isinstance(config, dict)
        else "-"
    )
    return launch_command, launch_source, config_summary


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def row_or_payload_value(
    data: dict[str, Any], row: dict[str, Any], key: str
) -> Any:
    if key in row:
        return row.get(key)
    boundary = data.get("hf_boundary")
    if isinstance(boundary, dict) and key in boundary:
        return boundary.get(key)
    return data.get(key)


def no_hf_guard_summary(data: dict[str, Any], row: dict[str, Any]) -> str:
    if data.get("engine") != "wkvm-native":
        return "n/a"
    report = data.get("native_no_hf_requirement")
    if isinstance(report, dict):
        status = "pass" if report.get("passed") is True else "fail"
        checked = fmt(report.get("checked_successful_rows"))
        required = ", required" if report.get("required") else ""
        return f"{status} ({checked} rows{required})"
    if row_has_full_success(row):
        return "pass (row)" if not native_no_hf_row_problems(row) else "fail (row)"
    return "-"


def row_memory(row: dict[str, Any]) -> tuple[str, str]:
    gpu_memory = row.get("gpu_memory")
    if (
        isinstance(gpu_memory, dict)
        and gpu_memory.get("schema") == "wkvm.whole_gpu_memory.v1"
        and gpu_memory.get("scope") == "whole_device"
    ):
        peak_mib = gpu_memory.get("peak_used_mib")
        baseline_mib = gpu_memory.get("baseline_used_mib")
        delta_mib = gpu_memory.get("peak_delta_mib")
        samples = gpu_memory.get("sample_count")
        if peak_mib is None:
            return "whole GPU", f"unavailable ({fmt(samples)} samples)"
        detail = fmt_mib_as_gib(peak_mib)
        if baseline_mib is not None and delta_mib is not None:
            detail += (
                f" (baseline {fmt_mib_as_gib(baseline_mib)}, "
                f"delta {fmt_mib_as_gib(delta_mib)}, {fmt(samples)} samples)"
            )
        return "whole GPU peak", detail
    if row.get("peak_engine_delta_gib") is not None:
        return "engine delta", fmt(row.get("peak_engine_delta_gib"), " GiB")
    if row.get("peak_reserved_gib") is not None:
        return "reserved", fmt(row.get("peak_reserved_gib"), " GiB")
    if row.get("peak_alloc_gib") is not None:
        return "allocated", fmt(row.get("peak_alloc_gib"), " GiB")
    return "-", "-"


def model_batch_summary(row: dict[str, Any]) -> str:
    rows = row.get("max_decode_model_batch_rows")
    bytes_ = row.get("max_decode_model_batch_bytes")
    if rows is None and bytes_ is None:
        return "-"
    if bytes_ is None:
        return f"{fmt(rows)} rows"
    if rows is None:
        return fmt_bytes(bytes_)
    return f"{fmt(rows)} rows / {fmt_bytes(bytes_)}"


def persistent_summary(row: dict[str, Any]) -> str:
    starts = row.get("persistent_padded_decode_starts")
    reuses = row.get("persistent_padded_decode_reuses")
    if starts is None and reuses is None:
        return "-"
    return f"{fmt(starts)} starts / {fmt(reuses)} reuses"


def decode_timing_summary(row: dict[str, Any]) -> str:
    keys = (
        ("merge", "decode_timing_merge_s"),
        ("model", "decode_timing_model_forward_s"),
        ("commit", "decode_timing_commit_s"),
        ("split", "decode_timing_split_s"),
        ("mask", "decode_timing_mask_s"),
        ("total", "decode_timing_total_s"),
    )
    if all(row.get(key) is None for _label, key in keys):
        return "-"
    return " / ".join(f"{label} {fmt(row.get(key))}" for label, key in keys)


def prompt_fingerprint_summary(row: dict[str, Any]) -> str:
    fingerprint = row_prompt_fingerprint(row)
    if fingerprint is None:
        return "-"
    source = fingerprint.get("prompt_token_source") or row.get("prompt_token_source") or "-"
    digest = fingerprint.get("prompt_token_ids_sha256")
    digest = "-" if digest is None else str(digest)[:12]
    count = fingerprint.get("prompt_count")
    total = fingerprint.get("prompt_total_tokens")
    return f"{source} {digest} ({fmt(count)} prompts / {fmt(total)} tok)"


def _summary_value(
    row: dict[str, Any],
    summary: dict[str, Any],
    row_key: str,
    summary_key: str,
) -> Any:
    if row.get(row_key) is not None:
        return row.get(row_key)
    return summary.get(summary_key)


def request_timing_summary(row: dict[str, Any]) -> str:
    summary = row.get("request_trace_summary")
    if not isinstance(summary, dict):
        summary = {}
    q50 = _summary_value(row, summary, "queue_time_p50_s", "queue_time_s_p50")
    q95 = _summary_value(row, summary, "queue_time_p95_s", "queue_time_s_p95")
    ttft50 = _summary_value(
        row,
        summary,
        "first_token_latency_p50_s",
        "first_token_latency_s_p50",
    )
    ttft95 = _summary_value(
        row,
        summary,
        "first_token_latency_p95_s",
        "first_token_latency_s_p95",
    )
    if ttft50 is None:
        ttft50 = row.get("p50_ttft_s")
    if ttft95 is None:
        ttft95 = row.get("p95_ttft_s")
    decode50 = _summary_value(row, summary, "decode_time_p50_s", "decode_time_s_p50")
    decode95 = _summary_value(row, summary, "decode_time_p95_s", "decode_time_s_p95")
    if all(
        value is None
        for value in (q50, q95, ttft50, ttft95, decode50, decode95)
    ):
        return "-"
    return (
        f"q {fmt(q50)}/{fmt(q95)}; "
        f"ttft {fmt(ttft50)}/{fmt(ttft95)}; "
        f"dec {fmt(decode50)}/{fmt(decode95)}"
    )


def itl_validity_summary(row: dict[str, Any]) -> str:
    valid_requests = row.get("itl_valid_request_count")
    request_count = row.get("request_count")
    sample_count = row.get("itl_sample_count")
    p50 = row.get("p50_itl_s")
    p95 = row.get("p95_itl_s")
    exact_counts = row.get("output_token_count_exact_requests")
    sources = row.get("output_token_count_sources")
    if all(
        value is None
        for value in (
            valid_requests,
            request_count,
            sample_count,
            p50,
            p95,
            exact_counts,
            sources,
        )
    ):
        return "-"
    source_text = ",".join(str(value) for value in sources) if sources else "-"
    return (
        f"{fmt(valid_requests)}/{fmt(request_count)} req; "
        f"{fmt(sample_count)} samples; p50/p95 {fmt(p50)}/{fmt(p95)}; "
        f"count {fmt(exact_counts)}/{fmt(request_count)} exact ({source_text})"
    )


def scheduler_summary(row: dict[str, Any]) -> str:
    keys = (
        ("wait", "max_waiting"),
        ("run", "max_running"),
        ("runnable", "max_runnable_rows"),
        ("resident", "max_resident_state_slots"),
        ("bp", "backpressure_events"),
        ("ret", "retraction_events"),
    )
    if all(row.get(key) is None for _label, key in keys):
        return "-"
    return " / ".join(f"{label} {fmt(row.get(key))}" for label, key in keys)


def graph_summary(row: dict[str, Any]) -> str:
    cuda_keys = (
        "persistent_padded_decode_cuda_graph_captures",
        "persistent_padded_decode_cuda_graph_cache_hits",
        "persistent_padded_decode_cuda_graph_replays",
        "persistent_padded_decode_cuda_graph_skips",
    )
    pool_keys = (
        "token_pool_decode_graph_static_shape_starts",
        "token_pool_decode_graph_static_shape_reuses",
        "token_pool_decode_graph_shape_mismatches",
    )
    if all(row.get(key) is None for key in (*cuda_keys, *pool_keys)):
        return "-"
    return (
        "cuda cap "
        f"{fmt(row.get('persistent_padded_decode_cuda_graph_captures'))} / "
        f"hit {fmt(row.get('persistent_padded_decode_cuda_graph_cache_hits'))} / "
        f"replay {fmt(row.get('persistent_padded_decode_cuda_graph_replays'))} / "
        f"skip {fmt(row.get('persistent_padded_decode_cuda_graph_skips'))}; "
        "pool start "
        f"{fmt(row.get('token_pool_decode_graph_static_shape_starts'))} / "
        f"reuse {fmt(row.get('token_pool_decode_graph_static_shape_reuses'))} / "
        f"mismatch {fmt(row.get('token_pool_decode_graph_shape_mismatches'))}"
    )


def fatal_error_row(data: dict[str, Any]) -> dict[str, Any] | None:
    fatal = data.get("fatal_error")
    if not isinstance(fatal, dict):
        return None
    parts = []
    phase = fatal.get("phase")
    if phase:
        parts.append(str(phase))
    error_type = fatal.get("type")
    if error_type:
        parts.append(str(error_type))
    message = fatal.get("message")
    if message:
        parts.append(str(message))
    return {
        "B": None,
        "success_count": 0,
        "green": False,
        "error": ": ".join(parts) if parts else "fatal setup error",
    }


def load_rows(path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source_rows = list(data.get("rows", []))
    if not source_rows:
        fatal_row = fatal_error_row(data)
        if fatal_row is not None:
            source_rows = [fatal_row]
    for row in source_rows:
        mem_kind, mem_value = row_memory(row)
        agg_decode = row.get("agg_decode_tok_s")
        if agg_decode is None:
            agg_decode = row.get("request_output_tok_s")
        p50 = row.get("p50_latency_s")
        if p50 is None:
            p50 = row.get("p50_e2e_latency_s")
        p95 = row.get("p95_latency_s")
        if p95 is None:
            p95 = row.get("p95_e2e_latency_s")
        rows.append(
            {
                "path": path,
                "schema": data.get("schema"),
                "engine": engine_label(data),
                "shape": fmt_shape(shape_key(data)),
                "semantics": data.get("semantics") or "-",
                "ctx": data.get("context_tokens_per_session"),
                "out": data.get("decode_tokens_per_session"),
                "prompt_mode": data.get("prompt_lengths_mode"),
                "prompt_fingerprint": prompt_fingerprint_summary(row),
                "model_forward_backend": row_or_payload_value(
                    data, row, "model_forward_backend"
                ),
                "uses_hf_transformer_forward": row_or_payload_value(
                    data, row, "uses_hf_transformer_forward"
                ),
                "uses_hf_model_construction": row_or_payload_value(
                    data, row, "uses_hf_model_construction"
                ),
                "uses_hf_tokenizer": row_or_payload_value(
                    data, row, "uses_hf_tokenizer"
                ),
                "uses_hf_config": row_or_payload_value(
                    data, row, "uses_hf_config"
                ),
                "native_gemma_config_loader": row_or_payload_value(
                    data, row, "native_gemma_config_loader"
                ),
                "native_gemma_checkpoint_loader": row_or_payload_value(
                    data, row, "native_gemma_checkpoint_loader"
                ),
                "no_hf_guard": no_hf_guard_summary(data, row),
                "B": row.get("B"),
                "success": (
                    f"{row.get('success_count', 0)}/"
                    f"{fmt(row.get('request_count', row.get('B')))}"
                ),
                "green": row.get("green"),
                "agg_decode": agg_decode,
                "decode_timing": decode_timing_summary(row),
                "request_timing": request_timing_summary(row),
                "itl_validity": itl_validity_summary(row),
                "scheduler": scheduler_summary(row),
                "graph": graph_summary(row),
                "e2e_output": row.get("e2e_output_tok_s"),
                "p50": p50,
                "p95": p95,
                "mem_kind": mem_kind,
                "mem_value": mem_value,
                "model_batch": model_batch_summary(row),
                "padded_temp": fmt_bytes(row.get("padded_decode_temp_bytes")),
                "persistent_padded": persistent_summary(row),
                "error": row.get("error"),
            }
        )
    return rows


def render(
    paths: list[Path],
    *,
    require_same_shape: bool = False,
    require_same_prompt_fingerprint: bool = False,
    require_native_no_hf: bool = False,
    require_comparable: bool = False,
) -> str:
    payloads = load_payloads(paths)
    contract: ComparisonContractResult | None = None
    if require_comparable:
        contract = validate_comparable(payloads)
    if require_same_shape:
        validate_same_shape(payloads)
    if require_same_prompt_fingerprint:
        validate_same_prompt_fingerprint(payloads)
    if require_native_no_hf:
        validate_native_no_hf(payloads)
    rows: list[dict[str, Any]] = []
    for path, data in payloads:
        rows.extend(load_rows(path, data))
    rows.sort(
        key=lambda r: (
            r["ctx"] or 0,
            r["out"] or 0,
            r["B"] or 0,
            r["engine"],
            str(r["path"]),
        )
    )
    lines = [
        "# Gemma Throughput Report",
        "",
        "Rows are normalized across wkvm-native, HF Transformers, vLLM, and SGLang JSON schemas. "
        "Serving rows use HTTP stream output throughput in the `agg decode tok/s` column. "
        "Only rows with the same `ctx`, `out`, prompt mode, and benchmark path should be treated as same-shape comparisons. "
        "When present, `prompt fingerprint` is a SHA-256 over the exact prompt token IDs for the row. "
        "The native no-HF columns are applicable to wkvm-native rows and are `n/a` for incumbent engines. "
        "`request timing s` reports p50/p95 queue, first-token, and decode timings when present. "
        "Memory labelled `whole GPU peak` is opt-in `nvidia-smi` instrumentation: it includes every process "
        "on the selected device, and its baseline/delta are not process-attributed.",
    ]
    if contract is not None:
        lines.extend(["", contract.markdown_summary()])
    lines.extend(
        [
            "",
            "## Environment Provenance",
            "",
            "Engine versions are target-server values reported by the benchmark operator; client package versions describe only the benchmark process.",
            "",
            "| engine | engine version | version source | GPU | driver | GPU memory | client runtime | source |",
            "|---|---|---|---|---|---:|---|---|",
        ]
    )
    for path, data in payloads:
        version, version_source = engine_version_summary(data)
        gpu, driver, gpu_memory = gpu_provenance_summary(data)
        source = f"[json]({path.as_posix()})"
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    engine_label(data),
                    version,
                    version_source,
                    gpu,
                    driver,
                    gpu_memory,
                    client_runtime_summary(data),
                    source,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Launch Provenance",
            "",
            "Target-server commands and configurations are operator supplied and recorded verbatim; benchmark-client commands reproduce the measurement harness invocation.",
            "",
            "| engine | target server launch | launch source | server config | benchmark client launch | source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for path, data in payloads:
        server_launch, launch_source, server_config = target_server_summary(data)
        source = f"[json]({path.as_posix()})"
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(value)
                for value in (
                    engine_label(data),
                    server_launch,
                    launch_source,
                    server_config,
                    data.get("launch_command") or "-",
                    source,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Benchmark Rows",
            "",
            "| engine | shape | semantics | B | success | green | agg decode tok/s | prompt fingerprint | forward backend | HF fwd | HF construct | HF tok | HF cfg | native cfg | native ckpt | no-HF guard | decode timing s | request timing s | ITL exact | scheduler | graph | e2e output tok/s | p50 s | p95 s | memory | max model batch | padded temp | persistent padded | error | source |",
            "|---|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        path = row["path"]
        source = f"[json]({path.as_posix()})"
        memory = (
            row["mem_value"]
            if row["mem_kind"] == "-"
            else f"{row['mem_value']} {row['mem_kind']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["engine"]),
                    row["shape"],
                    row["semantics"],
                    fmt(row["B"]),
                    str(row["success"]),
                    fmt(row["green"]),
                    fmt(row["agg_decode"]),
                    row["prompt_fingerprint"],
                    str(row["model_forward_backend"] or "-"),
                    fmt(row["uses_hf_transformer_forward"]),
                    fmt(row["uses_hf_model_construction"]),
                    fmt(row["uses_hf_tokenizer"]),
                    fmt(row["uses_hf_config"]),
                    fmt(row["native_gemma_config_loader"]),
                    fmt(row["native_gemma_checkpoint_loader"]),
                    row["no_hf_guard"],
                    row["decode_timing"],
                    row["request_timing"],
                    row["itl_validity"],
                    row["scheduler"],
                    row["graph"],
                    fmt(row["e2e_output"]),
                    fmt(row["p50"]),
                    fmt(row["p95"]),
                    memory,
                    row["model_batch"],
                    row["padded_temp"],
                    row["persistent_padded"],
                    str(row["error"] or "-"),
                    source,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--require-same-shape",
        action="store_true",
        help="fail if the input JSON files do not share one ctx/out/prompt-mode shape",
    )
    ap.add_argument(
        "--require-same-prompt-fingerprint",
        action="store_true",
        help=(
            "fail unless every benchmark row has a prompt-token fingerprint and "
            "rows with matching ctx/out/prompt-mode/B have the same prompt tokens"
        ),
    )
    ap.add_argument(
        "--require-native-no-hf",
        action="store_true",
        help=(
            "fail unless every fully successful wkvm-native row proves no HF "
            "tokenizer/config setup, no HF Transformer forward, no HF model "
            "construction, and native checkpoint/config loading"
        ),
    )
    ap.add_argument(
        "--require-comparable",
        action="store_true",
        help=(
            "fail unless existing artifact fields prove a shared benchmark path, "
            "shape, batch size, prompt lengths, successful outputs, throughput, "
            "and baseline provenance"
        ),
    )
    args = ap.parse_args()
    text = render(
        args.json_files,
        require_same_shape=args.require_same_shape,
        require_same_prompt_fingerprint=args.require_same_prompt_fingerprint,
        require_native_no_hf=args.require_native_no_hf,
        require_comparable=args.require_comparable,
    )
    if args.out:
        atomic_write_text(args.out, text)
        print(f"WROTE {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
