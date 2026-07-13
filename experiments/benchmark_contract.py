"""Strict, schema-preserving comparability checks for benchmark artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


DIRECT_SCHEMAS = frozenset(
    {
        "wkvm.native_gemma_bench.v1",
        "wkvm.hf_gemma_bench.v1",
        "wkvm.incumbent_gemma_bench.v1",
    }
)
SERVING_SCHEMA = "wkvm.serving_bench.v1"
SERVING_PROVENANCE_SCHEMA = "wkvm.serving_bench.provenance.v2"
LEGACY_SERVING_PROVENANCE_SCHEMA = "wkvm.serving_bench.provenance.v1"
SUPPORTED_SERVING_PROVENANCE_SCHEMAS = frozenset(
    {SERVING_PROVENANCE_SCHEMA, LEGACY_SERVING_PROVENANCE_SCHEMA}
)
WHOLE_GPU_MEMORY_SCHEMA = "wkvm.whole_gpu_memory.v1"


class ComparisonContractError(ValueError):
    """Raised when result artifacts cannot support a strict comparison."""


@dataclass(frozen=True)
class ComparisonContractResult:
    benchmark_path: str
    shape: tuple[int, int, str]
    paired_batch_sizes: tuple[int, ...]
    warnings: tuple[str, ...]

    def markdown_summary(self) -> str:
        batches = ",".join(str(value) for value in self.paired_batch_sizes)
        summary = (
            "Comparison contract: **PASS (Stage 1 workload)** "
            f"(`{self.benchmark_path}`, paired `B={batches}`)."
        )
        if self.warnings:
            summary += " Warnings: " + "; ".join(self.warnings) + "."
        return summary


def _source(path: Path, data: dict[str, Any]) -> str:
    engine = data.get("engine") or "unknown-engine"
    return f"{path.as_posix()} ({engine})"


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_string(
    path: Path,
    data: dict[str, Any],
    field: str,
) -> str:
    value = data.get(field)
    if not _nonempty_string(value):
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} "
            f"is missing non-empty {field!r}"
        )
    return str(value)


def benchmark_path(data: dict[str, Any]) -> str:
    schema = data.get("schema")
    if schema in DIRECT_SCHEMAS:
        return "direct-offline"
    if schema == SERVING_SCHEMA:
        backend = data.get("backend")
        if not _nonempty_string(backend):
            raise ComparisonContractError(
                "comparable requirement failed; serving payload is missing a backend"
            )
        return f"http:{backend}"
    raise ComparisonContractError(
        "comparable requirement failed; unsupported benchmark schema "
        f"{schema!r}"
    )


def _shape(path: Path, data: dict[str, Any]) -> tuple[int, int, str]:
    ctx = data.get("context_tokens_per_session")
    out = data.get("decode_tokens_per_session")
    prompt_mode = data.get("prompt_lengths_mode")
    if (
        not isinstance(ctx, int)
        or isinstance(ctx, bool)
        or ctx < 1
        or not isinstance(out, int)
        or isinstance(out, bool)
        or out < 1
        or not _nonempty_string(prompt_mode)
    ):
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} has an invalid "
            "ctx/out/prompt-mode shape"
        )
    return ctx, out, str(prompt_mode)


def _rows_by_batch(
    path: Path,
    data: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} has no benchmark rows"
        )
    indexed: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ComparisonContractError(
                f"comparable requirement failed; {_source(path, data)} row={index} "
                "is not an object"
            )
        batch = row.get("B")
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ComparisonContractError(
                f"comparable requirement failed; {_source(path, data)} row={index} "
                "has an invalid B"
            )
        if batch in indexed:
            raise ComparisonContractError(
                f"comparable requirement failed; {_source(path, data)} has duplicate "
                f"rows for B={batch}"
            )
        indexed[batch] = row
    return indexed


def _engine_family(data: dict[str, Any]) -> str:
    engine = str(data.get("engine") or "").lower()
    if engine.startswith("vllm"):
        return "vllm"
    if engine.startswith("sglang"):
        return "sglang"
    return engine


def _structured_provenance(data: dict[str, Any]) -> dict[str, Any] | None:
    provenance = data.get("provenance")
    if (
        isinstance(provenance, dict)
        and provenance.get("schema") in SUPPORTED_SERVING_PROVENANCE_SCHEMAS
    ):
        return provenance
    return None


def _validate_target_server_provenance(
    path: Path,
    data: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    target_server = provenance.get("target_server")
    if (
        not isinstance(target_server, dict)
        or not _nonempty_string(target_server.get("launch_command"))
        or target_server.get("launch_command_source") != "operator_supplied"
    ):
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} is missing "
            "operator-supplied target server launch provenance"
        )
    config = target_server.get("config")
    if config is not None and (
        not isinstance(config, dict)
        or target_server.get("config_source") != "operator_supplied"
    ):
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} has invalid "
            "operator-supplied target server config provenance"
        )


def _engine_version(data: dict[str, Any]) -> str | None:
    provenance = _structured_provenance(data)
    if provenance is not None:
        engine = provenance.get("engine")
        if isinstance(engine, dict) and _nonempty_string(engine.get("version")):
            return str(engine["version"])
    family = _engine_family(data)
    config = data.get("engine_config")
    key = f"{family}_version"
    if isinstance(config, dict) and _nonempty_string(config.get(key)):
        return str(config[key])
    return None


def _gpu_monitor_policy(data: dict[str, Any]) -> tuple[Any, ...]:
    provenance = _structured_provenance(data)
    monitor = provenance.get("gpu_memory_monitor") if provenance is not None else None
    if not isinstance(monitor, dict) or monitor.get("enabled") is not True:
        return (False, None, None, None)
    return (
        True,
        monitor.get("scope"),
        monitor.get("source"),
        monitor.get("sample_interval_s"),
    )


def _whole_gpu_memory(row: dict[str, Any]) -> dict[str, Any] | None:
    memory = row.get("gpu_memory")
    if (
        isinstance(memory, dict)
        and memory.get("schema") == WHOLE_GPU_MEMORY_SCHEMA
        and memory.get("scope") == "whole_device"
        and _numeric(memory.get("peak_used_mib"))
    ):
        return memory
    return None


def _require_engine_version(path: Path, data: dict[str, Any]) -> None:
    family = _engine_family(data)
    if family not in {"vllm", "sglang"}:
        return
    key = f"{family}_version"
    if _engine_version(data) is None:
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} is missing "
            f"structured engine version or engine_config.{key}"
        )


def _numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _memory_kind(row: dict[str, Any]) -> str | None:
    if _whole_gpu_memory(row) is not None:
        return "whole-gpu-used"
    if row.get("peak_engine_delta_gib") is not None:
        return "engine-delta"
    if row.get("peak_reserved_gib") is not None:
        return "torch-reserved"
    if row.get("peak_alloc_gib") is not None:
        return "torch-allocated"
    return None


def _validate_paired_row(
    *,
    path: Path,
    data: dict[str, Any],
    row: dict[str, Any],
    batch: int,
    throughput_field: str,
) -> tuple[int, tuple[int, ...]]:
    request_count = row.get("request_count", batch)
    if (
        not isinstance(request_count, int)
        or isinstance(request_count, bool)
        or request_count < batch
    ):
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} B={batch} has "
            "an invalid request_count"
        )
    if row.get("success_count") != request_count or row.get("error_count") != 0:
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} B={batch} is not "
            "fully successful"
        )
    if throughput_field == "request_output_tok_s":
        if row.get("output_token_count_exact_requests") != request_count:
            raise ComparisonContractError(
                f"comparable requirement failed; {_source(path, data)} B={batch} "
                "does not have exact output-token counts for every request"
            )
        sources = row.get("output_token_count_sources")
        if not isinstance(sources, list) or not sources:
            raise ComparisonContractError(
                f"comparable requirement failed; {_source(path, data)} B={batch} "
                "is missing output-token count provenance"
            )
    throughput = row.get(throughput_field)
    if not _numeric(throughput) or float(throughput) <= 0:
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} B={batch} is "
            f"missing a positive {throughput_field}"
        )
    prompt_lengths = row.get("prompt_lengths")
    if (
        not isinstance(prompt_lengths, list)
        or len(prompt_lengths) != request_count
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in prompt_lengths
        )
    ):
        raise ComparisonContractError(
            f"comparable requirement failed; {_source(path, data)} B={batch} has "
            "invalid prompt_lengths"
        )
    return request_count, tuple(prompt_lengths)


def validate_comparable(
    payloads: list[tuple[Path, dict[str, Any]]],
) -> ComparisonContractResult:
    """Validate the strongest comparison supported by existing result schemas.

    Every payload must contain the same concurrency ladder; each row is then
    subjected to success, workload, output-accounting, and throughput checks.
    """

    if len(payloads) < 2:
        raise ComparisonContractError(
            "comparable requirement failed; at least two result payloads are required"
        )

    path_keys: list[str] = []
    shapes: list[tuple[int, int, str]] = []
    rows_by_payload: list[dict[int, dict[str, Any]]] = []
    dtypes: list[str] = []
    model_paths: list[str] = []
    commits: list[str] = []
    missing_http_engine_versions: list[str] = []
    missing_structured_provenance: list[str] = []
    legacy_structured_provenance: list[str] = []
    missing_gpu_provenance: list[str] = []
    gpu_uuids: set[str] = set()
    gpu_names: set[str] = set()
    driver_versions: set[str] = set()
    serving_policies: list[tuple[Any, ...]] = []
    semantic_modes: list[str] = []

    for path, data in payloads:
        _require_string(path, data, "schema")
        _require_string(path, data, "engine")
        _require_string(path, data, "launch_command")
        commits.append(_require_string(path, data, "git_commit"))
        model_paths.append(_require_string(path, data, "model_path"))
        path_key = benchmark_path(data)
        family = _engine_family(data)
        if path_key == "direct-offline":
            _require_engine_version(path, data)
        elif family in {"vllm", "sglang"}:
            if _engine_version(data) is None:
                missing_http_engine_versions.append(str(data.get("engine")))
        if path_key.startswith("http:"):
            provenance = _structured_provenance(data)
            if provenance is None:
                missing_structured_provenance.append(str(data.get("engine")))
            else:
                if provenance.get("schema") == SERVING_PROVENANCE_SCHEMA:
                    _validate_target_server_provenance(path, data, provenance)
                else:
                    legacy_structured_provenance.append(str(data.get("engine")))
                gpu = provenance.get("gpu")
                if isinstance(gpu, dict):
                    if _nonempty_string(gpu.get("uuid")):
                        gpu_uuids.add(str(gpu["uuid"]))
                    if _nonempty_string(gpu.get("name")):
                        gpu_names.add(str(gpu["name"]))
                    if _nonempty_string(gpu.get("driver_version")):
                        driver_versions.add(str(gpu["driver_version"]))
                elif _gpu_monitor_policy(data)[0]:
                    missing_gpu_provenance.append(str(data.get("engine")))
            served_model = _require_string(path, data, "served_model")
            semantics = _require_string(path, data, "semantics")
            prompt_source = _require_string(path, data, "prompt_token_source")
            prompt_reuse_policy = _require_string(
                path,
                data,
                "prompt_reuse_policy",
            )
            warmup_requests = data.get("warmup_requests")
            warmup_output_tokens = data.get("warmup_output_tokens")
            warmup_row_offset = data.get("warmup_row_offset")
            if (
                not isinstance(warmup_requests, int)
                or isinstance(warmup_requests, bool)
                or warmup_requests < 0
                or not isinstance(warmup_output_tokens, int)
                or isinstance(warmup_output_tokens, bool)
                or warmup_output_tokens < 1
                or not isinstance(warmup_row_offset, int)
                or isinstance(warmup_row_offset, bool)
                or warmup_row_offset < 0
            ):
                raise ComparisonContractError(
                    f"comparable requirement failed; {_source(path, data)} has "
                    "invalid warmup policy fields"
                )
            extra_body = data.get("extra_body")
            if extra_body is not None and not isinstance(extra_body, dict):
                raise ComparisonContractError(
                    f"comparable requirement failed; {_source(path, data)} has "
                    "a non-object extra_body"
                )
            sampling = data.get("sampling")
            if not isinstance(sampling, dict) or not sampling:
                raise ComparisonContractError(
                    f"comparable requirement failed; {_source(path, data)} is "
                    "missing structured sampling settings"
                )
            semantic_modes.append(semantics)
            serving_policies.append(
                (
                    served_model,
                    prompt_source,
                    prompt_reuse_policy,
                    data.get("requests_per_row"),
                    warmup_requests,
                    warmup_output_tokens,
                    warmup_row_offset,
                    json.dumps(sampling, sort_keys=True),
                    json.dumps(extra_body, sort_keys=True),
                    _gpu_monitor_policy(data),
                )
            )
        path_keys.append(path_key)
        shapes.append(_shape(path, data))
        rows_by_payload.append(_rows_by_batch(path, data))
        if path_key == "direct-offline":
            dtypes.append(_require_string(path, data, "dtype"))

    if len(set(path_keys)) != 1:
        raise ComparisonContractError(
            "comparable requirement failed; inputs mix benchmark paths: "
            + ", ".join(sorted(set(path_keys)))
        )
    if len(set(shapes)) != 1:
        raise ComparisonContractError(
            "comparable requirement failed; inputs have different ctx/out/prompt-mode shapes"
        )
    if dtypes and len(set(dtypes)) != 1:
        raise ComparisonContractError(
            "comparable requirement failed; direct benchmark inputs have different dtypes: "
            + ", ".join(sorted(set(dtypes)))
        )
    if serving_policies and len(set(serving_policies)) != 1:
        raise ComparisonContractError(
            "comparable requirement failed; serving inputs have different model, "
            "prompt-source, request-count, warmup, cache-reuse, extra-body, or "
            "GPU-monitoring policies"
        )

    if path_keys[0] == "direct-offline" and any(
        any(row.get("green") is not None for row in rows.values())
        for rows in rows_by_payload
    ):
        budgets = []
        for path, data in payloads:
            cap = data.get("mem_cap_gib")
            headroom = data.get("headroom_gib")
            if not _numeric(cap) or not _numeric(headroom):
                raise ComparisonContractError(
                    f"comparable requirement failed; {_source(path, data)} is missing "
                    "a numeric mem_cap_gib/headroom_gib for green rows"
                )
            budgets.append((float(cap), float(headroom)))
        if len(set(budgets)) != 1:
            raise ComparisonContractError(
                "comparable requirement failed; green rows use different memory "
                "caps/headroom"
            )

    batch_sets = [set(rows) for rows in rows_by_payload]
    if any(values != batch_sets[0] for values in batch_sets[1:]):
        raise ComparisonContractError(
            "comparable requirement failed; inputs have different concurrency ladders"
        )
    paired = batch_sets[0]

    throughput_field = (
        "request_output_tok_s"
        if path_keys[0].startswith("http:")
        else "agg_decode_tok_s"
    )
    memory_kinds: set[str] = set()
    whole_gpu_memory_coverage: list[tuple[str, int, int]] = []
    for (path, data), rows in zip(payloads, rows_by_payload, strict=True):
        measured = sum(1 for row in rows.values() if _whole_gpu_memory(row) is not None)
        whole_gpu_memory_coverage.append((_source(path, data), measured, len(rows)))
    for batch in sorted(paired):
        expected_request_count: int | None = None
        expected_prompts: tuple[int, ...] | None = None
        for (path, data), rows in zip(payloads, rows_by_payload, strict=True):
            row = rows[batch]
            request_count, prompts = _validate_paired_row(
                path=path,
                data=data,
                row=row,
                batch=batch,
                throughput_field=throughput_field,
            )
            if expected_request_count is None:
                expected_request_count = request_count
                expected_prompts = prompts
            elif request_count != expected_request_count:
                raise ComparisonContractError(
                    "comparable requirement failed; paired rows have different "
                    f"request_count at B={batch}"
                )
            elif prompts != expected_prompts:
                raise ComparisonContractError(
                    "comparable requirement failed; paired rows have different "
                    f"prompt_lengths at B={batch}"
                )
            kind = _memory_kind(row)
            if kind is not None:
                memory_kinds.add(kind)

    warnings: list[str] = [
        "model identity is path-only; no checkpoint digest is available",
        "prompt identity is length-only unless the prompt-fingerprint guard is also used",
        "git_commit identifies the benchmark harness, not necessarily engine source",
    ]
    if path_keys[0] == "direct-offline":
        warnings.append(
            "semantic mode and sampling settings are not structured in direct schemas"
        )
    elif len(set(semantic_modes)) > 1:
        warnings.append(
            "semantic modes differ ("
            + ", ".join(sorted(set(semantic_modes)))
            + "); throughput is not quality-equivalent"
        )
    if len(set(model_paths)) != 1:
        warnings.append("model paths differ across inputs")
    if len(set(commits)) != 1:
        warnings.append("benchmark harness commits differ across inputs")
    if len(memory_kinds) > 1:
        warnings.append(
            "memory metric kinds differ; green/memory values are not cross-engine comparable"
        )
    if missing_http_engine_versions:
        warnings.append(
            "server engine versions are missing for "
            + ", ".join(sorted(missing_http_engine_versions))
        )
    if missing_structured_provenance:
        warnings.append(
            "structured serving provenance is missing for "
            + ", ".join(sorted(missing_structured_provenance))
        )
    if legacy_structured_provenance:
        warnings.append(
            "legacy structured serving provenance does not prove target server "
            "launch commands for "
            + ", ".join(sorted(legacy_structured_provenance))
        )
    if len(gpu_uuids) > 1 or (not gpu_uuids and len(gpu_names) > 1):
        warnings.append(
            "GPU devices differ across inputs; whole-GPU memory is hardware-dependent"
        )
    if len(driver_versions) > 1:
        warnings.append("GPU driver versions differ across inputs")
    monitoring_requested = any(_gpu_monitor_policy(data)[0] for _path, data in payloads)
    if monitoring_requested or any(
        measured for _source_name, measured, _total in whole_gpu_memory_coverage
    ):
        incomplete = [
            source_name
            for source_name, measured, total in whole_gpu_memory_coverage
            if measured != total
        ]
        if incomplete:
            warnings.append(
                "whole-GPU memory is missing for some rows or inputs: "
                + ", ".join(incomplete)
            )
    if missing_gpu_provenance:
        warnings.append(
            "selected GPU/driver provenance is missing for "
            + ", ".join(sorted(missing_gpu_provenance))
        )

    return ComparisonContractResult(
        benchmark_path=path_keys[0],
        shape=shapes[0],
        paired_batch_sizes=tuple(sorted(paired)),
        warnings=tuple(warnings),
    )
