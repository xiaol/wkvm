#!/usr/bin/env python
"""Build a strict repeated-run Gemma comparison report.

The report accepts direct native-WKVM and incumbent vLLM/SGLang benchmark
artifacts.  It validates workload identity, request completion, token counts,
GPU provenance, and whole-device memory telemetry before aggregating repeated
runs.  Decode ratios are emitted only for same-run decode intervals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import statistics
from typing import Any, Iterable


SUMMARY_SCHEMA = "wkvm.reliable_gemma_report.v1"
PROMPT_FINGERPRINT_SCHEMA = "wkvm.prompt_token_ids.sha256.v1"
OUTPUT_FINGERPRINT_SCHEMA = "wkvm.generated_output_token_ids.sha256.v1"
MEMORY_SCHEMA = "wkvm.whole_gpu_memory.v1"
SOURCE_IDENTITY_SCHEMA = "wkvm.git_worktree_identity.sha256.v1"
MODEL_IDENTITY_SCHEMA = "wkvm.model_checkpoint_identity.sha256.v1"
REQUIRED_ENGINES = frozenset({"wkvm-native", "vllm", "sglang"})
MINIMUM_PUBLIC_REPEATS = 3
MAXIMUM_IDLE_BASELINE_GIB = 1.0
PUBLIC_E2E_CLAIM_RATIO = 10.0
SOURCE_EXCLUDED_PATH_PATTERNS = [
    "experiments/results/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    "**/*.egg-info/**",
    ".venv/**",
    "build/**",
    "dist/**",
]
MODEL_EXCLUDED_PATH_PATTERNS = [".cache/**"]
SOURCE_IDENTITY_SCOPE = (
    "git tracked and all untracked worktree files excluding declared generated artifacts"
)
SUPPORTED_SCHEMAS = frozenset(
    {
        "wkvm.native_gemma_bench.v1",
        "wkvm.incumbent_gemma_bench.v1",
    }
)
SEMANTICS_ROUTED = "routed_span_approximate"
SEMANTICS_FULL_KV = "full_kv"
SEQUENTIAL_RUNBOOK = """Sequential evidence runbook:
  1. Select one idle physical GPU, export CUDA_VISIBLE_DEVICES=<GPU>, and pass
     that same physical index or UUID through --gpu-memory-device. Confirm the
     pre-load whole-device baseline is at most 1 GiB.
  2. Activate each engine venv, or prepend its bin directory to PATH so helper
     tools such as ninja are discoverable (PATH=<venv>/bin:$PATH).
  3. Run native WKVM, vLLM, and SGLang sequentially on that same GPU with the
     same model, ctx/out/prompt fingerprint, memory policy, and cold policy;
     pass --no-warmup to the incumbent harness for cold one-shot evidence and
     set explicit --vllm-max-num-batched-tokens / --sglang-chunked-prefill-size.
  4. Produce at least three distinct JSON artifacts for every engine/B.
     Write them outside the source tree or under experiments/results/.
  5. Run this command with every artifact and both output paths.
"""


class ReliableReportError(ValueError):
    """Raised when input evidence cannot support a reliable comparison."""


@dataclass(frozen=True)
class ValidatedSample:
    path: Path
    engine: str
    engine_version: str
    semantics: str
    batch: int
    prompt_fingerprint: tuple[Any, ...]
    prompt_digest: str
    output_digest: str
    launch_command: str
    normalized_launch_command: str
    configuration_signature: str
    provenance: dict[str, Any]
    git_commit: str
    source_identity_sha256: str
    model_identity_sha256: str
    gpu_model: str
    gpu_uuid: str
    driver_version: str
    gpu_memory_sample_interval_s: float
    metrics: dict[str, float | None]
    cohort_prefill_comparable: bool
    cohort_prefill_method: str
    decode_comparable: bool
    decode_exclusion_reason: str | None


def _fail(message: str) -> None:
    raise ReliableReportError(message)


def _number(value: Any, *, positive: bool = False) -> float | None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        return None
    number = float(value)
    if positive and number <= 0:
        return None
    return number


def _required_number(
    value: Any,
    *,
    source: str,
    field: str,
    positive: bool = False,
) -> float:
    number = _number(value, positive=positive)
    if number is None:
        qualifier = "positive finite" if positive else "finite"
        _fail(f"{source}: {field} must be a {qualifier} number")
    return number


def _required_string(value: Any, *, source: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{source}: missing non-empty {field}")
    return value.strip()


def _digest(value: Any, *, source: str, field: str) -> str:
    digest = _required_string(value, source=source, field=field).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _fail(f"{source}: {field} must be a 64-character SHA-256 digest")
    return digest


def _git_oid(value: Any, *, source: str, field: str) -> str:
    object_id = _required_string(value, source=source, field=field).lower()
    if len(object_id) not in {40, 64} or any(
        char not in "0123456789abcdef" for char in object_id
    ):
        _fail(f"{source}: {field} must be a 40- or 64-character Git object ID")
    return object_id


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _close(actual: float, expected: float) -> bool:
    tolerance = max(0.02, abs(expected) * 0.01)
    return abs(actual - expected) <= tolerance


def _engine_name(data: dict[str, Any], source: str) -> str:
    engine = _required_string(data.get("engine"), source=source, field="engine")
    lowered = engine.lower()
    if data.get("schema") == "wkvm.native_gemma_bench.v1":
        if lowered != "wkvm-native":
            _fail(f"{source}: native schema requires engine='wkvm-native'")
        return "wkvm-native"
    if lowered.startswith("vllm"):
        return "vllm"
    if lowered.startswith("sglang"):
        return "sglang"
    _fail(f"{source}: unsupported incumbent engine {engine!r}")


def _semantics(data: dict[str, Any], engine: str, source: str) -> str:
    reported = data.get("semantics")
    if reported is not None:
        reported = _required_string(reported, source=source, field="semantics")
        aliases = {
            "routed-span-approximate": SEMANTICS_ROUTED,
            "routed_span_approximate": SEMANTICS_ROUTED,
            "full-kv": SEMANTICS_FULL_KV,
            "full_kv": SEMANTICS_FULL_KV,
        }
        normalized = aliases.get(reported.lower())
        if normalized is None:
            _fail(f"{source}: unsupported semantics {reported!r}")
    else:
        normalized = SEMANTICS_ROUTED if engine == "wkvm-native" else SEMANTICS_FULL_KV
    expected = SEMANTICS_ROUTED if engine == "wkvm-native" else SEMANTICS_FULL_KV
    if normalized != expected:
        _fail(
            f"{source}: engine {engine!r} cannot be reported as semantics {normalized!r}"
        )
    return normalized


def _source_identity(
    provenance: dict[str, Any],
    *,
    git_commit: str,
    source: str,
) -> tuple[dict[str, Any], str]:
    benchmark = provenance["benchmark"]
    identity = benchmark.get("source_identity")
    if not isinstance(identity, dict):
        _fail(f"{source}: missing benchmark source_identity")
    if identity.get("schema") != SOURCE_IDENTITY_SCHEMA:
        _fail(f"{source}: invalid benchmark source_identity schema")
    if identity.get("error") is not None:
        _fail(f"{source}: benchmark source_identity contains an error")
    identity_sha256 = _digest(
        identity.get("identity_sha256"),
        source=source,
        field="source_identity.identity_sha256",
    )
    if identity.get("excluded_path_patterns") != SOURCE_EXCLUDED_PATH_PATTERNS:
        _fail(f"{source}: source identity must exclude only declared generated artifacts")
    if identity.get("scope") != SOURCE_IDENTITY_SCOPE:
        _fail(f"{source}: source identity scope is not exact-worktree scope")
    if identity.get("excluded_paths") not in ([], None):
        _fail(f"{source}: source identity contains unsupported extra exclusions")
    identity_commit = _git_oid(
        identity.get("git_commit"),
        source=source,
        field="source_identity.git_commit",
    )
    if identity_commit != git_commit:
        _fail(f"{source}: source identity commit disagrees with provenance commit")
    _git_oid(
        identity.get("git_head_tree"),
        source=source,
        field="source_identity.git_head_tree",
    )
    for field in (
        "git_status_sha256",
        "git_tracked_diff_sha256",
        "worktree_manifest_sha256",
    ):
        _digest(identity.get(field), source=source, field=f"source_identity.{field}")
    identity_fields = {
        "git_commit": identity_commit,
        "git_head_tree": identity["git_head_tree"].lower(),
        "git_status_sha256": identity["git_status_sha256"].lower(),
        "git_tracked_diff_sha256": identity["git_tracked_diff_sha256"].lower(),
        "worktree_manifest_sha256": identity["worktree_manifest_sha256"].lower(),
    }
    if identity_sha256 != _canonical_sha256(identity_fields):
        _fail(f"{source}: source identity digest is inconsistent")
    for field in (
        "tracked_file_count",
        "untracked_file_count",
        "worktree_file_count",
    ):
        value = identity.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            _fail(f"{source}: source_identity.{field} must be a non-negative integer")
    if identity.get("worktree_file_count", 0) < 1:
        _fail(f"{source}: source identity worktree manifest is empty")
    if identity["worktree_file_count"] != (
        identity["tracked_file_count"] + identity["untracked_file_count"]
    ):
        _fail(f"{source}: source identity worktree counts are inconsistent")
    dirty = identity.get("git_worktree_dirty")
    if not isinstance(dirty, bool):
        _fail(f"{source}: source_identity.git_worktree_dirty must be boolean")
    empty_digest = hashlib.sha256(b"").hexdigest()
    expected_dirty = (
        identity_fields["git_status_sha256"] != empty_digest
        or identity["untracked_file_count"] > 0
    )
    if dirty is not expected_dirty:
        _fail(f"{source}: source identity dirty state is inconsistent")
    if benchmark.get("git_worktree_dirty") is not dirty:
        _fail(f"{source}: source identity dirty state disagrees with provenance")
    if dirty:
        _fail(f"{source}: benchmark source worktree must be clean")
    pre_run_identity = _digest(
        benchmark.get("pre_run_source_identity_sha256"),
        source=source,
        field="provenance.benchmark.pre_run_source_identity_sha256",
    )
    if pre_run_identity != identity_sha256:
        _fail(f"{source}: source identity changed during benchmark run")
    if benchmark.get("source_identity_unchanged_during_run") is not True:
        _fail(f"{source}: source identity must be unchanged during benchmark run")
    return identity, identity_sha256


def _model_identity(data: dict[str, Any], source: str) -> tuple[dict[str, Any], str]:
    identity = data.get("model_identity")
    if not isinstance(identity, dict):
        _fail(f"{source}: missing structured model_identity")
    if identity.get("schema") != MODEL_IDENTITY_SCHEMA:
        _fail(f"{source}: invalid model_identity schema")
    if identity.get("error") is not None:
        _fail(f"{source}: model_identity contains an error")
    if identity.get("excluded_path_patterns") != MODEL_EXCLUDED_PATH_PATTERNS:
        _fail(f"{source}: model identity must exclude only downloader cache metadata")
    _required_string(
        identity.get("model_root"),
        source=source,
        field="model_identity.model_root",
    )
    files = identity.get("files")
    if not isinstance(files, list) or not files:
        _fail(f"{source}: model_identity.files must be a non-empty list")
    normalized_files: list[dict[str, Any]] = []
    paths: list[str] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            _fail(f"{source}: model_identity.files[{index}] must be an object")
        path = _required_string(
            entry.get("path"),
            source=source,
            field=f"model_identity.files[{index}].path",
        )
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _fail(
                f"{source}: model_identity.files[{index}].size_bytes must be "
                "a non-negative integer"
            )
        digest = _digest(
            entry.get("sha256"),
            source=source,
            field=f"model_identity.files[{index}].sha256",
        )
        paths.append(path)
        normalized_files.append(
            {"path": path, "size_bytes": size, "sha256": digest}
        )
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        _fail(f"{source}: model_identity file paths must be unique and sorted")
    if identity.get("file_count") != len(normalized_files):
        _fail(f"{source}: model_identity.file_count is inconsistent")
    total_bytes = sum(entry["size_bytes"] for entry in normalized_files)
    if identity.get("total_bytes") != total_bytes:
        _fail(f"{source}: model_identity.total_bytes is inconsistent")
    manifest_sha256 = _digest(
        identity.get("manifest_sha256"),
        source=source,
        field="model_identity.manifest_sha256",
    )
    if manifest_sha256 != _canonical_sha256(normalized_files):
        _fail(f"{source}: model identity manifest digest is inconsistent")
    return identity, manifest_sha256


def _validate_model_root(
    identity: dict[str, Any],
    *,
    model_path: str,
    source: str,
) -> None:
    reported_root = Path(
        _required_string(
            identity.get("model_root"),
            source=source,
            field="model_identity.model_root",
        )
    ).expanduser().resolve()
    requested_root = Path(model_path).expanduser().resolve()
    if reported_root != requested_root:
        _fail(f"{source}: model_identity.model_root disagrees with model_path")


def _compute_gpu_uuid(
    provenance: dict[str, Any],
    *,
    engine: str,
    gpu: dict[str, Any],
    source: str,
) -> str:
    runtime_key = "environment" if engine == "wkvm-native" else "runtime"
    runtime = provenance.get(runtime_key)
    if not isinstance(runtime, dict):
        _fail(f"{source}: provenance.{runtime_key} must be an object")
    visible = _required_string(
        runtime.get("cuda_visible_devices"),
        source=source,
        field=f"provenance.{runtime_key}.cuda_visible_devices",
    )
    selectors = [value.strip() for value in visible.split(",") if value.strip()]
    if len(selectors) != 1:
        _fail(f"{source}: CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    selector = selectors[0]
    gpu_index = gpu.get("device_index", gpu.get("index"))
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
        _fail(f"{source}: provenance GPU index must be a non-negative integer")
    gpu_uuid = _required_string(
        gpu.get("device_uuid", gpu.get("uuid")),
        source=source,
        field="provenance.gpu UUID",
    )
    if selector.isdecimal():
        if int(selector) != gpu_index:
            _fail(f"{source}: monitored GPU does not match CUDA_VISIBLE_DEVICES index")
    elif selector != gpu_uuid:
        _fail(f"{source}: monitored GPU does not match CUDA_VISIBLE_DEVICES UUID")
    return gpu_uuid


def _provenance(
    data: dict[str, Any],
    *,
    engine: str,
    source: str,
) -> tuple[dict[str, Any], str, str, str, str, str, str]:
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        _fail(f"{source}: missing structured provenance")
    schema = provenance.get("schema")
    allowed_schema = (
        "wkvm.native_gemma_bench.provenance.v1"
        if engine == "wkvm-native"
        else "wkvm.incumbent_gemma_bench.provenance.v1"
    )
    if schema != allowed_schema:
        _fail(f"{source}: provenance.schema must be {allowed_schema!r}")
    benchmark = provenance.get("benchmark")
    if not isinstance(benchmark, dict):
        _fail(f"{source}: provenance.benchmark must be an object")
    git_commit = _git_oid(
        benchmark.get("git_commit"),
        source=source,
        field="provenance.benchmark.git_commit",
    )
    gpu = provenance.get("gpu")
    if not isinstance(gpu, dict):
        _fail(f"{source}: provenance.gpu must be an object")
    gpu_model = gpu.get("gpu_name", gpu.get("name"))
    gpu_model = _required_string(
        gpu_model,
        source=source,
        field="provenance.gpu model",
    )
    driver_version = _required_string(
        gpu.get("driver_version"),
        source=source,
        field="provenance.gpu.driver_version",
    )
    gpu_uuid = _compute_gpu_uuid(
        provenance,
        engine=engine,
        gpu=gpu,
        source=source,
    )
    _source_identity_object, source_identity_sha256 = _source_identity(
        provenance,
        git_commit=git_commit,
        source=source,
    )
    if engine == "wkvm-native":
        engine_version = benchmark.get("wkvm_package_version")
    else:
        engine_provenance = provenance.get("engine")
        if not isinstance(engine_provenance, dict):
            _fail(f"{source}: provenance.engine must be an object")
        engine_version = engine_provenance.get("version")
    engine_version = _required_string(
        engine_version,
        source=source,
        field="engine package version",
    )
    return (
        provenance,
        git_commit,
        source_identity_sha256,
        gpu_model,
        gpu_uuid,
        driver_version,
        engine_version,
    )


def _shape(data: dict[str, Any], source: str) -> tuple[int, int, str, str]:
    context = data.get("context_tokens_per_session")
    output = data.get("decode_tokens_per_session")
    mode = data.get("prompt_lengths_mode")
    dtype = data.get("dtype")
    if not isinstance(context, int) or isinstance(context, bool) or context < 1:
        _fail(f"{source}: invalid context_tokens_per_session")
    if not isinstance(output, int) or isinstance(output, bool) or output < 1:
        _fail(f"{source}: invalid decode_tokens_per_session")
    mode = _required_string(mode, source=source, field="prompt_lengths_mode")
    dtype = _required_string(dtype, source=source, field="dtype")
    return context, output, mode, dtype


def _validate_compute_device(
    data: dict[str, Any],
    *,
    engine: str,
    source: str,
) -> None:
    if engine != "wkvm-native":
        return
    device = _required_string(
        data.get("device"),
        source=source,
        field="device",
    ).lower()
    if device not in {"cuda", "cuda:0"}:
        _fail(
            f"{source}: native compute device must be cuda or cuda:0 when "
            "CUDA_VISIBLE_DEVICES selects one physical GPU"
        )


def _common_policy(data: dict[str, Any], source: str) -> tuple[Any, ...]:
    mem_cap = _required_number(
        data.get("mem_cap_gib"),
        source=source,
        field="mem_cap_gib",
        positive=True,
    )
    headroom = _required_number(
        data.get("headroom_gib"),
        source=source,
        field="headroom_gib",
    )
    if headroom < 0 or headroom >= mem_cap:
        _fail(f"{source}: headroom_gib must be non-negative and below mem_cap_gib")
    prompt_source = _required_string(
        data.get("prompt_token_source"),
        source=source,
        field="prompt_token_source",
    )
    tokenizer = data.get("uses_hf_tokenizer")
    if not isinstance(tokenizer, bool):
        _fail(f"{source}: uses_hf_tokenizer must be boolean")
    warmup = data.get("warmup")
    if not isinstance(warmup, bool):
        _fail(f"{source}: warmup must be boolean")
    if warmup:
        _fail(f"{source}: reliable cold comparison requires warmup=false")
    max_baseline = _required_number(
        data.get("max_baseline_gpu_used_gib"),
        source=source,
        field="max_baseline_gpu_used_gib",
    )
    if max_baseline < 0:
        _fail(f"{source}: max_baseline_gpu_used_gib must be non-negative")
    if max_baseline > MAXIMUM_IDLE_BASELINE_GIB:
        _fail(
            f"{source}: max_baseline_gpu_used_gib must be <= "
            f"{MAXIMUM_IDLE_BASELINE_GIB:.1f} GiB"
        )
    return mem_cap, headroom, prompt_source, tokenizer, warmup, max_baseline


def _configuration(
    data: dict[str, Any],
    *,
    engine: str,
    source: str,
) -> tuple[dict[str, Any], str]:
    key = "config" if engine == "wkvm-native" else "engine_config"
    configuration = data.get(key)
    if not isinstance(configuration, dict) or not configuration:
        _fail(f"{source}: missing non-empty {key}")
    required_scheduler_field = {
        "vllm": "max_num_batched_tokens",
        "sglang": "chunked_prefill_size",
    }.get(engine)
    if required_scheduler_field is not None:
        scheduler_value = configuration.get(required_scheduler_field)
        if (
            not isinstance(scheduler_value, int)
            or isinstance(scheduler_value, bool)
            or scheduler_value < 1
        ):
            _fail(
                f"{source}: {key}.{required_scheduler_field} must be an "
                "explicit positive integer"
            )
    comparable_configuration = json.loads(json.dumps(configuration))
    if engine != "wkvm-native":
        comparable_configuration.pop("residency_telemetry_capacity", None)
    signature = json.dumps(
        comparable_configuration,
        sort_keys=True,
        separators=(",", ":"),
    )
    return configuration, signature


def _normalized_launch_command(command: str, source: str) -> str:
    try:
        arguments = shlex.split(command)
    except ValueError as exc:
        _fail(f"{source}: launch_command cannot be parsed: {exc}")
    normalized: list[str] = []
    skip_next = False
    for argument in arguments:
        if skip_next:
            skip_next = False
            continue
        if argument == "--json":
            skip_next = True
            continue
        if argument.startswith("--json="):
            continue
        normalized.append(argument)
    if skip_next:
        _fail(f"{source}: launch_command has --json without a path")
    return shlex.join(normalized)


def _prompt_fingerprint(
    row: dict[str, Any],
    *,
    batch: int,
    prompt_lengths: tuple[int, ...],
    source: str,
) -> tuple[tuple[Any, ...], str]:
    fingerprint = row.get("prompt_fingerprint")
    if not isinstance(fingerprint, dict):
        _fail(f"{source}: missing row prompt_fingerprint")
    if fingerprint.get("schema") != PROMPT_FINGERPRINT_SCHEMA:
        _fail(f"{source}: invalid prompt fingerprint schema")
    if fingerprint.get("prompt_count") != batch:
        _fail(f"{source}: prompt fingerprint count does not equal B={batch}")
    fingerprint_lengths = fingerprint.get("prompt_lengths")
    if fingerprint_lengths != list(prompt_lengths):
        _fail(f"{source}: prompt fingerprint lengths do not match row prompt_lengths")
    prompt_total = sum(prompt_lengths)
    if fingerprint.get("prompt_total_tokens") != prompt_total:
        _fail(f"{source}: prompt fingerprint total does not match prompt lengths")
    token_source = _required_string(
        fingerprint.get("prompt_token_source"),
        source=source,
        field="prompt_fingerprint.prompt_token_source",
    )
    digest = _digest(
        fingerprint.get("prompt_token_ids_sha256"),
        source=source,
        field="prompt_fingerprint.prompt_token_ids_sha256",
    )
    return (
        (
            fingerprint.get("schema"),
            token_source,
            batch,
            prompt_total,
            prompt_lengths,
            digest,
        ),
        digest,
    )


def _output_counts(
    row: dict[str, Any],
    *,
    batch: int,
    output_tokens: int,
    source: str,
) -> tuple[tuple[int, ...], str]:
    fingerprint = row.get("generated_output_fingerprint")
    if not isinstance(fingerprint, dict):
        _fail(f"{source}: missing generated_output_fingerprint")
    if fingerprint.get("schema") != OUTPUT_FINGERPRINT_SCHEMA:
        _fail(f"{source}: invalid generated output fingerprint schema")
    digest = _digest(
        fingerprint.get("request_output_token_ids_sha256"),
        source=source,
        field="generated output fingerprint digest",
    )
    if fingerprint.get("request_count") != batch:
        _fail(f"{source}: generated output request_count does not equal B={batch}")
    request_ids = fingerprint.get("request_ids")
    if (
        not isinstance(request_ids, list)
        or len(request_ids) != batch
        or len(set(request_ids)) != batch
        or any(not isinstance(value, str) or not value for value in request_ids)
    ):
        _fail(f"{source}: generated output request_ids must identify B={batch} requests")
    counts = fingerprint.get("output_token_counts")
    if not isinstance(counts, list) or len(counts) != batch:
        _fail(f"{source}: generated output counts must contain B={batch} entries")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value != output_tokens
        for value in counts
    ):
        _fail(
            f"{source}: every successful request must contain exactly "
            f"{output_tokens} output tokens"
        )
    if fingerprint.get("output_token_count") != sum(counts):
        _fail(f"{source}: generated output total does not match per-request counts")
    row_counts = row.get("output_token_counts")
    if row_counts is not None and row_counts != counts:
        _fail(f"{source}: row output_token_counts disagree with fingerprint")
    return tuple(counts), digest


def _memory_metrics(
    row: dict[str, Any],
    *,
    provenance: dict[str, Any],
    max_baseline_gpu_used_gib: float,
    source: str,
) -> tuple[float, float, float]:
    memory = row.get("gpu_memory")
    if not isinstance(memory, dict):
        _fail(f"{source}: missing row gpu_memory telemetry")
    if memory.get("schema") != MEMORY_SCHEMA:
        _fail(f"{source}: invalid gpu_memory schema")
    if memory.get("scope") != "whole_device" or memory.get("source") != "nvidia-smi":
        _fail(f"{source}: gpu_memory must be whole-device nvidia-smi telemetry")
    if memory.get("error") is not None or memory.get("query_error_count") != 0:
        _fail(f"{source}: gpu_memory telemetry contains query errors")
    sample_count = memory.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 2:
        _fail(f"{source}: gpu_memory requires at least two valid samples")
    sample_interval_s = _required_number(
        memory.get("sample_interval_s"),
        source=source,
        field="gpu_memory.sample_interval_s",
        positive=True,
    )
    baseline = _required_number(
        memory.get("baseline_used_mib"),
        source=source,
        field="gpu_memory.baseline_used_mib",
    )
    peak = _required_number(
        memory.get("peak_used_mib"),
        source=source,
        field="gpu_memory.peak_used_mib",
        positive=True,
    )
    delta = _required_number(
        memory.get("peak_delta_mib"),
        source=source,
        field="gpu_memory.peak_delta_mib",
    )
    if baseline < 0 or peak < baseline or delta < 0 or abs(delta - (peak - baseline)) > 1:
        _fail(f"{source}: inconsistent gpu_memory baseline/peak/delta values")
    if baseline / 1024.0 > max_baseline_gpu_used_gib:
        _fail(
            f"{source}: pre-load GPU baseline {baseline / 1024.0:.3f} GiB "
            f"exceeds idle ceiling {max_baseline_gpu_used_gib:.3f} GiB"
        )
    selector = _required_string(
        memory.get("device_selector"),
        source=source,
        field="gpu_memory.device_selector",
    )
    memory_index = memory.get("device_index")
    if (
        not isinstance(memory_index, int)
        or isinstance(memory_index, bool)
        or memory_index < 0
    ):
        _fail(f"{source}: gpu_memory.device_index must be a non-negative integer")
    gpu = provenance["gpu"]
    provenance_index = gpu.get("device_index", gpu.get("index"))
    if memory_index != provenance_index:
        _fail(f"{source}: memory index disagrees with GPU provenance")
    provenance_selector = gpu.get("device_selector")
    if provenance_selector is not None and str(provenance_selector) != selector:
        _fail(f"{source}: memory selector disagrees with GPU provenance")
    memory_uuid = _required_string(
        memory.get("device_uuid"),
        source=source,
        field="gpu_memory.device_uuid",
    )
    provenance_uuid = _required_string(
        gpu.get("device_uuid", gpu.get("uuid")),
        source=source,
        field="provenance.gpu UUID",
    )
    if memory_uuid != provenance_uuid:
        _fail(f"{source}: memory UUID disagrees with GPU provenance")
    if selector not in {str(memory_index), memory_uuid}:
        _fail(f"{source}: memory selector must be the monitored GPU index or UUID")
    memory_name = _required_string(
        memory.get("gpu_name"),
        source=source,
        field="gpu_memory.gpu_name",
    )
    provenance_name = _required_string(
        gpu.get("gpu_name", gpu.get("name")),
        source=source,
        field="provenance.gpu model",
    )
    if memory_name != provenance_name:
        _fail(f"{source}: memory GPU model disagrees with GPU provenance")
    memory_driver = _required_string(
        memory.get("driver_version"),
        source=source,
        field="gpu_memory.driver_version",
    )
    provenance_driver = _required_string(
        gpu.get("driver_version"),
        source=source,
        field="provenance.gpu.driver_version",
    )
    if memory_driver != provenance_driver:
        _fail(f"{source}: memory driver disagrees with GPU provenance")
    memory_total = memory.get("memory_total_mib")
    provenance_total = gpu.get("memory_total_mib")
    if (
        not isinstance(memory_total, int)
        or isinstance(memory_total, bool)
        or memory_total <= 0
    ):
        _fail(f"{source}: gpu_memory.memory_total_mib must be a positive integer")
    if (
        not isinstance(provenance_total, int)
        or isinstance(provenance_total, bool)
        or provenance_total <= 0
    ):
        _fail(f"{source}: provenance GPU memory_total_mib must be a positive integer")
    if memory_total != provenance_total:
        _fail(f"{source}: memory total disagrees with GPU provenance")
    return peak / 1024.0, delta / 1024.0, sample_interval_s


def _ttft_metrics(
    row: dict[str, Any],
    *,
    engine: str,
    batch: int,
    source: str,
) -> tuple[float | None, float | None, float | None, str | None]:
    p50 = _number(row.get("p50_ttft_s"), positive=True)
    p95 = _number(row.get("p95_ttft_s"), positive=True)
    maximum = _number(row.get("max_ttft_s"), positive=True)
    if engine == "sglang":
        if p50 is not None or p95 is not None or maximum is not None:
            _fail(f"{source}: SGLang separate-run prefill cannot report request TTFT")
        return None, None, None, "request_ttft_unavailable"
    if p50 is None or p95 is None or maximum is None:
        _fail(f"{source}: missing positive request TTFT p50/p95/max evidence")
    count = row.get("ttft_request_count", row.get("ttft_metric_count"))
    if count != batch:
        _fail(f"{source}: TTFT metric count does not equal B={batch}")
    if p50 > p95 or p95 > maximum:
        _fail(f"{source}: TTFT p50/p95/max ordering is invalid")
    return p50, p95, maximum, None


def _decode_metrics(
    row: dict[str, Any],
    *,
    engine: str,
    decode_token_count: int,
    source: str,
) -> tuple[bool, float | None, float | None, str | None]:
    method = row.get("decode_timing_method")
    comparable_flag = row.get("decode_timing_comparable")
    scope = row.get("decode_interval_scope")
    if engine == "wkvm-native":
        comparable = scope == "batch_earliest_first_to_latest_last"
    else:
        comparable = (
            comparable_flag is True
            and method == "same_run_request_metrics"
            and scope == "batch_earliest_first_to_latest_last"
        )
    if engine == "sglang":
        if (
            method != "separate_run_subtraction"
            or comparable_flag is not False
            or scope != "separate_run_wall_time_subtraction"
        ):
            _fail(
                f"{source}: SGLang decode must be labeled as non-comparable "
                "separate-run subtraction"
            )
        return False, None, None, "separate_run_subtraction"
    if not comparable:
        _fail(f"{source}: missing comparable same-run decode interval evidence")
    interval = row.get("decode_interval_s", row.get("decode_seconds"))
    interval = _required_number(
        interval,
        source=source,
        field="decode_interval_s",
        positive=True,
    )
    throughput = _required_number(
        row.get("agg_decode_tok_s"),
        source=source,
        field="agg_decode_tok_s",
        positive=True,
    )
    expected = decode_token_count / interval
    if not _close(throughput, expected):
        _fail(f"{source}: agg_decode_tok_s disagrees with decode token accounting")
    return True, interval, throughput, None


def _validate_row(
    *,
    path: Path,
    data: dict[str, Any],
    row: dict[str, Any],
    engine: str,
    engine_version: str,
    semantics: str,
    provenance: dict[str, Any],
    git_commit: str,
    source_identity_sha256: str,
    model_identity_sha256: str,
    gpu_model: str,
    gpu_uuid: str,
    driver_version: str,
    output_tokens: int,
    launch_command: str,
    normalized_launch_command: str,
    configuration_signature: str,
) -> ValidatedSample:
    batch = row.get("B")
    source = f"{path.as_posix()} engine={engine} B={batch}"
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        _fail(f"{source}: invalid B")
    if (
        row.get("success_count") != batch
        or row.get("error_count") != 0
        or row.get("error") is not None
    ):
        _fail(f"{source}: row is not fully successful")
    prompt_lengths_raw = row.get("prompt_lengths")
    if (
        not isinstance(prompt_lengths_raw, list)
        or len(prompt_lengths_raw) != batch
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in prompt_lengths_raw
        )
    ):
        _fail(f"{source}: prompt_lengths must contain B={batch} positive integers")
    prompt_lengths = tuple(prompt_lengths_raw)
    prompt_key, prompt_digest = _prompt_fingerprint(
        row,
        batch=batch,
        prompt_lengths=prompt_lengths,
        source=source,
    )
    counts, output_digest = _output_counts(
        row,
        batch=batch,
        output_tokens=output_tokens,
        source=source,
    )
    total_output_tokens = sum(counts)
    decode_token_count = sum(max(0, count - 1) for count in counts)
    batch_wall = _required_number(
        row.get("batch_wall_s", row.get("elapsed_s")),
        source=source,
        field="batch_wall_s",
        positive=True,
    )
    if row.get("batch_wall_scope") != "synchronous_batch_completion":
        _fail(f"{source}: batch_wall_scope must be synchronous_batch_completion")
    e2e_throughput = _required_number(
        row.get("e2e_output_tok_s"),
        source=source,
        field="e2e_output_tok_s",
        positive=True,
    )
    if not _close(e2e_throughput, total_output_tokens / batch_wall):
        _fail(f"{source}: e2e_output_tok_s disagrees with output token accounting")
    cohort_wall = _required_number(
        row.get("cohort_prefill_wall_s"),
        source=source,
        field="cohort_prefill_wall_s",
        positive=True,
    )
    cohort_input = _required_number(
        row.get("cohort_input_tok_s"),
        source=source,
        field="cohort_input_tok_s",
        positive=True,
    )
    cohort_scope = row.get("cohort_prefill_scope")
    cohort_reported_comparable = row.get("cohort_prefill_comparable")
    if engine == "wkvm-native":
        if cohort_scope != "same_run_max_request_ttft":
            _fail(f"{source}: native cohort prefill must use same-run maximum TTFT")
        cohort_prefill_comparable = True
        cohort_prefill_method = "same_run_max_request_ttft"
    elif engine == "vllm":
        if cohort_scope != "max_request_ttft_synchronous_cohort":
            _fail(f"{source}: vLLM cohort prefill must use same-run maximum TTFT")
        if cohort_reported_comparable is not True:
            _fail(f"{source}: vLLM cohort prefill must be marked comparable")
        cohort_prefill_comparable = True
        cohort_prefill_method = "same_run_max_request_ttft"
    else:
        if cohort_scope != "separate_run_batch_wall":
            _fail(f"{source}: SGLang cohort prefill must identify separate-run batch wall")
        if cohort_reported_comparable is not False:
            _fail(f"{source}: SGLang separate-run cohort prefill must be non-comparable")
        if row.get("separate_timing_probe_order") != "full_then_max_tokens_1":
            _fail(
                f"{source}: SGLang cold full run must precede its separate "
                "one-token timing probe"
            )
        cohort_prefill_comparable = False
        cohort_prefill_method = "separate_run_batch_wall"
    prompt_total = sum(prompt_lengths)
    reported_prompt_total = row.get(
        "cohort_input_token_count",
        row.get("cohort_input_tokens"),
    )
    if reported_prompt_total != prompt_total:
        _fail(f"{source}: cohort input token count disagrees with prompt lengths")
    if not _close(cohort_input, prompt_total / cohort_wall):
        _fail(f"{source}: cohort_input_tok_s disagrees with prompt token accounting")
    p50_ttft, p95_ttft, max_ttft, _ttft_exclusion = _ttft_metrics(
        row,
        engine=engine,
        batch=batch,
        source=source,
    )
    if max_ttft is not None and not _close(max_ttft, cohort_wall):
        _fail(f"{source}: max_ttft_s disagrees with cohort_prefill_wall_s")
    peak_used_gib, peak_delta_gib, memory_sample_interval_s = _memory_metrics(
        row,
        provenance=provenance,
        max_baseline_gpu_used_gib=_required_number(
            data.get("max_baseline_gpu_used_gib"),
            source=source,
            field="max_baseline_gpu_used_gib",
        ),
        source=source,
    )
    mem_cap_gib = _required_number(
        data.get("mem_cap_gib"), source=source, field="mem_cap_gib", positive=True
    )
    headroom_gib = _required_number(
        data.get("headroom_gib"), source=source, field="headroom_gib"
    )
    memory_gate_gib = mem_cap_gib - headroom_gib
    reported_engine_delta_gib = _required_number(
        row.get("peak_engine_delta_gib"),
        source=source,
        field="peak_engine_delta_gib",
    )
    if not _close(reported_engine_delta_gib, peak_delta_gib):
        _fail(f"{source}: peak_engine_delta_gib disagrees with GPU telemetry")
    if peak_delta_gib > memory_gate_gib + 0.02 or row.get("green") is not True:
        _fail(
            f"{source}: row failed memory gate; peak delta "
            f"{peak_delta_gib:.3f} GiB exceeds {memory_gate_gib:.3f} GiB or "
            "green is not true"
        )
    decode_comparable, decode_interval, decode_throughput, decode_exclusion = (
        _decode_metrics(
            row,
            engine=engine,
            decode_token_count=decode_token_count,
            source=source,
        )
    )
    return ValidatedSample(
        path=path,
        engine=engine,
        engine_version=engine_version,
        semantics=semantics,
        batch=batch,
        prompt_fingerprint=prompt_key,
        prompt_digest=prompt_digest,
        output_digest=output_digest,
        launch_command=launch_command,
        normalized_launch_command=normalized_launch_command,
        configuration_signature=configuration_signature,
        provenance=provenance,
        git_commit=git_commit,
        source_identity_sha256=source_identity_sha256,
        model_identity_sha256=model_identity_sha256,
        gpu_model=gpu_model,
        gpu_uuid=gpu_uuid,
        driver_version=driver_version,
        gpu_memory_sample_interval_s=memory_sample_interval_s,
        metrics={
            "cohort_input_tok_s": cohort_input,
            "cohort_prefill_wall_s": cohort_wall,
            "e2e_output_tok_s": e2e_throughput,
            "p50_ttft_s": p50_ttft,
            "p95_ttft_s": p95_ttft,
            "max_ttft_s": max_ttft,
            "batch_wall_s": batch_wall,
            "peak_gpu_used_gib": peak_used_gib,
            "peak_gpu_delta_gib": peak_delta_gib,
            "comparable_decode_interval_s": decode_interval,
            "comparable_decode_tok_s": decode_throughput,
        },
        cohort_prefill_comparable=cohort_prefill_comparable,
        cohort_prefill_method=cohort_prefill_method,
        decode_comparable=decode_comparable,
        decode_exclusion_reason=decode_exclusion,
    )


def load_and_validate(paths: Iterable[Path]) -> tuple[list[ValidatedSample], dict[str, Any]]:
    path_list = [Path(path) for path in paths]
    if not path_list:
        _fail("no benchmark artifacts supplied")
    if len({path.resolve() for path in path_list}) != len(path_list):
        _fail("duplicate benchmark artifact paths are not allowed")
    samples: list[ValidatedSample] = []
    shapes: dict[tuple[int, int, str, str], list[str]] = {}
    model_paths: dict[str, list[str]] = {}
    model_identities: dict[str, dict[str, Any]] = {}
    policies: dict[tuple[Any, ...], list[str]] = {}
    artifact_rows: dict[Path, set[int]] = {}
    artifact_metadata: list[dict[str, Any]] = []
    for path in path_list:
        source = path.as_posix()
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            _fail(f"{source}: cannot load JSON: {exc}")
        if not isinstance(data, dict):
            _fail(f"{source}: artifact root must be an object")
        if data.get("schema") not in SUPPORTED_SCHEMAS:
            _fail(f"{source}: unsupported schema {data.get('schema')!r}")
        engine = _engine_name(data, source)
        semantics = _semantics(data, engine, source)
        _validate_compute_device(data, engine=engine, source=source)
        shape = _shape(data, source)
        shapes.setdefault(shape, []).append(source)
        policy = _common_policy(data, source)
        policies.setdefault(policy, []).append(source)
        model_path = _required_string(
            data.get("model_path"),
            source=source,
            field="model_path",
        )
        model_paths.setdefault(model_path, []).append(source)
        model_identity, model_identity_sha256 = _model_identity(data, source)
        _validate_model_root(
            model_identity,
            model_path=model_path,
            source=source,
        )
        model_identities.setdefault(model_identity_sha256, model_identity)
        launch_command = _required_string(
            data.get("launch_command"),
            source=source,
            field="launch_command",
        )
        normalized_launch_command = _normalized_launch_command(
            launch_command,
            source,
        )
        configuration, configuration_signature = _configuration(
            data,
            engine=engine,
            source=source,
        )
        (
            provenance,
            commit,
            source_identity_sha256,
            gpu_model,
            gpu_uuid,
            driver,
            engine_version,
        ) = _provenance(data, engine=engine, source=source)
        rows = data.get("rows")
        if not isinstance(rows, list) or not rows:
            _fail(f"{source}: artifact has no benchmark rows")
        artifact_rows[path] = set()
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                _fail(f"{source}: rows[{row_index}] must be an object")
            batch = row.get("B")
            if batch in artifact_rows[path]:
                _fail(f"{source}: duplicate row for B={batch}")
            artifact_rows[path].add(batch)
            samples.append(
                _validate_row(
                    path=path,
                    data=data,
                    row=row,
                    engine=engine,
                    engine_version=engine_version,
                    semantics=semantics,
                    provenance=provenance,
                    git_commit=commit,
                    source_identity_sha256=source_identity_sha256,
                    model_identity_sha256=model_identity_sha256,
                    gpu_model=gpu_model,
                    gpu_uuid=gpu_uuid,
                    driver_version=driver,
                    output_tokens=shape[1],
                    launch_command=launch_command,
                    normalized_launch_command=normalized_launch_command,
                    configuration_signature=configuration_signature,
                )
            )
        artifact_metadata.append(
            {
                "path": source,
                "engine": engine,
                "engine_version": engine_version,
                "semantics": semantics,
                "batch_sizes": sorted(artifact_rows[path]),
                "launch_command": launch_command,
                "normalized_launch_command": normalized_launch_command,
                "benchmark_config": configuration,
                "model_identity": model_identity,
                "provenance": provenance,
            }
        )
    if len(shapes) != 1:
        details = "; ".join(
            f"{shape}: {', '.join(paths_for_shape)}"
            for shape, paths_for_shape in sorted(shapes.items())
        )
        _fail(f"same-shape requirement failed: {details}")
    shape = next(iter(shapes))
    if len(policies) != 1:
        details = "; ".join(
            f"{policy}: {', '.join(policy_sources)}"
            for policy, policy_sources in sorted(policies.items(), key=lambda item: str(item[0]))
        )
        _fail(f"common benchmark policy differs across artifacts: {details}")
    if len(model_paths) != 1:
        details = "; ".join(
            f"{model}: {', '.join(model_sources)}"
            for model, model_sources in sorted(model_paths.items())
        )
        _fail(f"same-model requirement failed: {details}")
    if len(model_identities) != 1:
        _fail("model checkpoint manifests differ across artifacts")
    gpu_keys = {
        (sample.gpu_uuid, sample.gpu_model, sample.driver_version)
        for sample in samples
    }
    if len(gpu_keys) != 1:
        details = ", ".join(
            f"{uuid} {model} driver={driver}"
            for uuid, model, driver in sorted(gpu_keys)
        )
        _fail(f"physical GPU UUID/model/driver comparability failed: {details}")
    memory_sample_intervals = {
        sample.gpu_memory_sample_interval_s for sample in samples
    }
    if len(memory_sample_intervals) != 1:
        _fail("GPU memory sample intervals differ across artifacts")
    commits = {sample.git_commit for sample in samples}
    if len(commits) != 1:
        _fail("benchmark git commits differ across artifacts")
    source_identities = {sample.source_identity_sha256 for sample in samples}
    if len(source_identities) != 1:
        _fail("exact source/worktree identities differ across artifacts")
    engine_versions: dict[str, set[str]] = {}
    semantics_by_engine: dict[str, set[str]] = {}
    for sample in samples:
        engine_versions.setdefault(sample.engine, set()).add(sample.engine_version)
        semantics_by_engine.setdefault(sample.engine, set()).add(sample.semantics)
    for engine, versions in sorted(engine_versions.items()):
        if len(versions) != 1:
            _fail(f"engine {engine!r} mixes package versions: {sorted(versions)}")
    for engine, values in sorted(semantics_by_engine.items()):
        if len(values) != 1:
            _fail(f"engine {engine!r} mixes semantics: {sorted(values)}")
    if set(engine_versions) != REQUIRED_ENGINES:
        _fail(
            "public comparison requires wkvm-native, vllm, and sglang evidence"
        )
    if {sample.semantics for sample in samples} != {
        SEMANTICS_ROUTED,
        SEMANTICS_FULL_KV,
    }:
        _fail("comparison requires both routed_span_approximate and full_kv evidence")
    prompt_groups: dict[int, set[tuple[Any, ...]]] = {}
    output_groups: dict[int, set[str]] = {}
    engines_by_batch: dict[int, set[str]] = {}
    for sample in samples:
        prompt_groups.setdefault(sample.batch, set()).add(sample.prompt_fingerprint)
        output_groups.setdefault(sample.batch, set()).add(sample.output_digest)
        engines_by_batch.setdefault(sample.batch, set()).add(sample.engine)
    for batch, fingerprints in sorted(prompt_groups.items()):
        if len(fingerprints) != 1:
            _fail(f"same-prompt-fingerprint requirement failed for B={batch}")
        if len(output_groups[batch]) != 1:
            _fail(f"same-output-fingerprint requirement failed for B={batch}")
        if engines_by_batch[batch] != REQUIRED_ENGINES:
            _fail(
                f"B={batch} requires wkvm-native, vllm, and sglang evidence"
            )
    gpu_uuid, gpu_model, driver = next(iter(gpu_keys))
    (
        mem_cap,
        headroom,
        prompt_source,
        tokenizer,
        warmup,
        max_baseline,
    ) = next(iter(policies))
    model_identity_sha256, model_identity = next(iter(model_identities.items()))
    source_identity = samples[0].provenance["benchmark"]["source_identity"]
    contract = {
        "shape": {
            "context_tokens_per_session": shape[0],
            "decode_tokens_per_session": shape[1],
            "prompt_lengths_mode": shape[2],
            "dtype": shape[3],
        },
        "model_path": next(iter(model_paths)),
        "model_identity": model_identity,
        "model_identity_sha256": model_identity_sha256,
        "model_identity_excluded_path_patterns": model_identity[
            "excluded_path_patterns"
        ],
        "policy": {
            "mem_cap_gib": mem_cap,
            "headroom_gib": headroom,
            "prompt_token_source": prompt_source,
            "uses_hf_tokenizer": tokenizer,
            "warmup": warmup,
            "max_baseline_gpu_used_gib": max_baseline,
            "gpu_memory_sample_interval_s": next(iter(memory_sample_intervals)),
        },
        "gpu": {
            "uuid": gpu_uuid,
            "model": gpu_model,
            "driver_version": driver,
        },
        "git_commit": next(iter(commits)),
        "source_identity_sha256": next(iter(source_identities)),
        "source_identity_excluded_path_patterns": source_identity[
            "excluded_path_patterns"
        ],
        "prompt_fingerprints_by_batch": {
            str(batch): next(
                sample.prompt_digest for sample in samples if sample.batch == batch
            )
            for batch in sorted(prompt_groups)
        },
        "output_fingerprints_by_batch": {
            str(batch): next(iter(output_groups[batch]))
            for batch in sorted(output_groups)
        },
        "artifacts": artifact_metadata,
    }
    return samples, contract


def aggregate_values(values: Iterable[float | None]) -> dict[str, Any] | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return {
        "count": len(present),
        "median": statistics.median(present),
        "min": min(present),
        "max": max(present),
    }


def _engine_order(engine: str) -> tuple[int, str]:
    priorities = {"wkvm-native": 0, "vllm": 1, "sglang": 2}
    return priorities.get(engine, 99), engine


def build_summary(
    paths: Iterable[Path],
    *,
    min_samples: int = 3,
) -> dict[str, Any]:
    if (
        not isinstance(min_samples, int)
        or isinstance(min_samples, bool)
        or min_samples < MINIMUM_PUBLIC_REPEATS
    ):
        _fail(
            f"min_samples must be an integer >= {MINIMUM_PUBLIC_REPEATS} "
            "for a public comparison"
        )
    samples, contract = load_and_validate(paths)
    grouped: dict[tuple[str, int], list[ValidatedSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.engine, sample.batch), []).append(sample)
    for (engine, batch), group in sorted(grouped.items()):
        if len(group) < min_samples:
            _fail(
                f"insufficient repeated evidence for engine={engine} B={batch}: "
                f"got {len(group)}, require at least {min_samples}"
            )
        if len({sample.launch_command for sample in group}) != len(group):
            _fail(
                f"duplicate launch commands cannot count as independent samples "
                f"for engine={engine} B={batch}"
            )
        if len({sample.normalized_launch_command for sample in group}) != 1:
            _fail(
                f"launch policy/config differs across repeats for "
                f"engine={engine} B={batch}"
            )
        if len({sample.configuration_signature for sample in group}) != 1:
            _fail(
                f"engine configuration differs across repeats for "
                f"engine={engine} B={batch}"
            )
    engines_per_batch: dict[int, set[str]] = {}
    for engine, batch in grouped:
        engines_per_batch.setdefault(batch, set()).add(engine)
    for batch, engines in sorted(engines_per_batch.items()):
        semantics = {
            grouped[(engine, batch)][0].semantics for engine in engines
        }
        if semantics != {SEMANTICS_ROUTED, SEMANTICS_FULL_KV}:
            _fail(f"B={batch} does not contain both routed and full-KV evidence")
    metric_names = (
        "cohort_input_tok_s",
        "cohort_prefill_wall_s",
        "e2e_output_tok_s",
        "p50_ttft_s",
        "p95_ttft_s",
        "max_ttft_s",
        "batch_wall_s",
        "peak_gpu_used_gib",
        "peak_gpu_delta_gib",
        "comparable_decode_interval_s",
        "comparable_decode_tok_s",
    )
    result_groups: list[dict[str, Any]] = []
    for engine, batch in sorted(grouped, key=lambda value: (value[1], _engine_order(value[0]))):
        group = grouped[(engine, batch)]
        decode_included = sum(sample.decode_comparable for sample in group)
        decode_reasons = sorted(
            {
                sample.decode_exclusion_reason
                for sample in group
                if sample.decode_exclusion_reason is not None
            }
        )
        metrics = {
            name: aggregate_values(sample.metrics[name] for sample in group)
            for name in metric_names
        }
        result_groups.append(
            {
                "engine": engine,
                "engine_version": group[0].engine_version,
                "semantics": group[0].semantics,
                "B": batch,
                "sample_count": len(group),
                "metrics": metrics,
                "cohort_prefill": {
                    "comparable_sample_count": sum(
                        sample.cohort_prefill_comparable for sample in group
                    ),
                    "method": group[0].cohort_prefill_method,
                    "ratio_eligible": all(
                        sample.cohort_prefill_comparable for sample in group
                    ),
                },
                "decode": {
                    "comparable_sample_count": decode_included,
                    "excluded_sample_count": len(group) - decode_included,
                    "exclusion_reasons": decode_reasons,
                    "ratio_eligible": decode_included == len(group),
                },
                "artifact_paths": [sample.path.as_posix() for sample in group],
            }
        )
    comparisons: list[dict[str, Any]] = []
    groups_by_batch: dict[int, list[dict[str, Any]]] = {}
    for group in result_groups:
        groups_by_batch.setdefault(group["B"], []).append(group)
    for batch, batch_groups in sorted(groups_by_batch.items()):
        by_engine = {group["engine"]: group for group in batch_groups}
        numerator = by_engine["wkvm-native"]
        for incumbent_engine in ("vllm", "sglang"):
            denominator = by_engine[incumbent_engine]
            e2e_ratio = (
                numerator["metrics"]["e2e_output_tok_s"]["median"]
                / denominator["metrics"]["e2e_output_tok_s"]["median"]
            )
            conservative_e2e_ratio = (
                numerator["metrics"]["e2e_output_tok_s"]["min"]
                / denominator["metrics"]["e2e_output_tok_s"]["max"]
            )
            input_ratio = None
            input_exclusion = None
            if (
                denominator["cohort_prefill"]["ratio_eligible"]
                and numerator["cohort_prefill"]["ratio_eligible"]
            ):
                input_ratio = (
                    numerator["metrics"]["cohort_input_tok_s"]["median"]
                    / denominator["metrics"]["cohort_input_tok_s"]["median"]
                )
            else:
                methods = sorted(
                    {
                        denominator["cohort_prefill"]["method"],
                        numerator["cohort_prefill"]["method"],
                    }
                )
                input_exclusion = "incomparable_methods:" + ",".join(methods)
            decode_ratio = None
            decode_exclusion = None
            if denominator["decode"]["ratio_eligible"] and numerator["decode"]["ratio_eligible"]:
                decode_ratio = (
                    numerator["metrics"]["comparable_decode_tok_s"]["median"]
                    / denominator["metrics"]["comparable_decode_tok_s"]["median"]
                )
            else:
                reasons = sorted(
                    set(denominator["decode"]["exclusion_reasons"])
                    | set(numerator["decode"]["exclusion_reasons"])
                )
                decode_exclusion = ",".join(reasons) or "incomplete_same_run_decode_evidence"
            comparisons.append(
                {
                    "B": batch,
                    "ratio_definition": "numerator_over_denominator",
                    "numerator_engine": numerator["engine"],
                    "denominator_engine": denominator["engine"],
                    "semantics_warning": (
                        numerator["semantics"] != denominator["semantics"]
                    ),
                    "median_cohort_input_ratio": input_ratio,
                    "cohort_input_ratio_exclusion": input_exclusion,
                    "median_e2e_output_ratio": e2e_ratio,
                    "conservative_e2e_output_ratio": conservative_e2e_ratio,
                    "ten_x_e2e_claim_pass": (
                        conservative_e2e_ratio >= PUBLIC_E2E_CLAIM_RATIO
                    ),
                    "median_comparable_decode_ratio": decode_ratio,
                    "decode_ratio_exclusion": decode_exclusion,
                }
            )
    claim_batches = []
    for batch in sorted(groups_by_batch):
        batch_comparisons = [
            comparison
            for comparison in comparisons
            if comparison["B"] == batch
        ]
        ratios = {
            comparison["denominator_engine"]: comparison[
                "conservative_e2e_output_ratio"
            ]
            for comparison in batch_comparisons
        }
        claim_batches.append(
            {
                "B": batch,
                "incumbent_conservative_ratios": ratios,
                "passes_all_incumbents": (
                    set(ratios) == {"vllm", "sglang"}
                    and all(
                        ratio >= PUBLIC_E2E_CLAIM_RATIO
                        for ratio in ratios.values()
                    )
                ),
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "pass",
        "minimum_samples_per_engine_batch": min_samples,
        "contract": contract,
        "groups": result_groups,
        "comparisons": comparisons,
        "ten_x_e2e_claim_gate": {
            "threshold_ratio": PUBLIC_E2E_CLAIM_RATIO,
            "metric": "e2e_output_tok_s",
            "method": "minimum_wkvm_over_maximum_incumbent",
            "observed_repeat_envelope_not_confidence_interval": True,
            "batches": claim_batches,
            "any_batch_passes_all_incumbents": any(
                batch["passes_all_incumbents"] for batch in claim_batches
            ),
        },
        "caveats": [
            "routed_span_approximate and full_kv are different model-state semantics",
            "whole-device memory includes every process on the selected GPU",
            "SGLang separate max_tokens=1 prefill is excluded from cohort-input ratios",
            "SGLang separate-run subtraction is excluded from decode ratios",
            "the 10x gate applies to E2E output throughput and uses the worst observed repeated-run envelope",
        ],
    }


def _triplet(metric: dict[str, Any] | None, unit: str = "") -> str:
    if metric is None:
        return "excluded"
    suffix = f" {unit}" if unit else ""
    return (
        f"{metric['median']:.3f} "
        f"[{metric['min']:.3f}, {metric['max']:.3f}]{suffix}"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("status") != "pass":
        _fail("cannot render an unvalidated reliable report summary")
    contract = summary["contract"]
    shape = contract["shape"]
    gpu = contract["gpu"]
    lines = [
        "# Reliable Gemma Repeated-Run Report",
        "",
        "Evidence gate: **PASS**.",
        "",
        f"- Shape: ctx={shape['context_tokens_per_session']}, "
        f"out={shape['decode_tokens_per_session']}, "
        f"prompt={shape['prompt_lengths_mode']}, dtype={shape['dtype']}",
        f"- Model: `{contract['model_path']}`",
        f"- Model manifest SHA-256: `{contract['model_identity_sha256']}`",
        "- Model identity exclusions: "
        + ", ".join(
            f"`{pattern}`"
            for pattern in contract["model_identity_excluded_path_patterns"]
        ),
        f"- GPU cohort: {gpu['model']} `{gpu['uuid']}` with driver {gpu['driver_version']}",
        f"- Warmup policy: `{contract['policy']['warmup']}` (cold one-shot requires `False`)",
        f"- Pre-load GPU baseline ceiling: {contract['policy']['max_baseline_gpu_used_gib']:.3f} GiB",
        f"- GPU memory sample interval: {contract['policy']['gpu_memory_sample_interval_s']:.3f} s",
        f"- Minimum repeats: {summary['minimum_samples_per_engine_batch']} per engine/B",
        f"- Benchmark commit: `{contract['git_commit']}`",
        f"- Exact source/worktree SHA-256: `{contract['source_identity_sha256']}`",
        "- Exact greedy output fingerprints: "
        + ", ".join(
            f"B={batch} `{digest}`"
            for batch, digest in sorted(
                contract["output_fingerprints_by_batch"].items(),
                key=lambda item: int(item[0]),
            )
        ),
        "- Source identity exclusions: "
        + ", ".join(
            f"`{pattern}`"
            for pattern in contract["source_identity_excluded_path_patterns"]
        ),
        "- Semantics: `routed_span_approximate` and `full_kv` are reported separately and are not equivalent.",
        "",
        "## Aggregates",
        "",
        "Cells are median [min, max] across validated repeated runs.",
        "",
        "| B | Engine | Semantics | n | Cohort input tok/s | Cohort prefill wall | E2E output tok/s | TTFT p50 | TTFT p95 | Batch wall | GPU peak used | Decode interval | Comparable decode tok/s |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        cohort_input = _triplet(metrics["cohort_input_tok_s"], "tok/s")
        if not group["cohort_prefill"]["ratio_eligible"]:
            cohort_input = f"reported only; ratio excluded ({cohort_input})"
        decode_interval = _triplet(metrics["comparable_decode_interval_s"], "s")
        decode_throughput = _triplet(metrics["comparable_decode_tok_s"], "tok/s")
        if not group["decode"]["ratio_eligible"]:
            reason = ",".join(group["decode"]["exclusion_reasons"])
            decode_interval = f"excluded ({reason})"
            decode_throughput = f"excluded ({reason})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(group["B"]),
                    f"{group['engine']} {group['engine_version']}",
                    f"`{group['semantics']}`",
                    str(group["sample_count"]),
                    cohort_input,
                    _triplet(metrics["cohort_prefill_wall_s"], "s"),
                    _triplet(metrics["e2e_output_tok_s"], "tok/s"),
                    _triplet(metrics["p50_ttft_s"], "s"),
                    _triplet(metrics["p95_ttft_s"], "s"),
                    _triplet(metrics["batch_wall_s"], "s"),
                    _triplet(metrics["peak_gpu_used_gib"], "GiB"),
                    decode_interval,
                    decode_throughput,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## WKVM / Incumbent Median Ratios",
            "",
            "Every ratio is `wkvm-native / incumbent` (numerator / denominator). Cross-semantics ratios describe measured workload performance, not semantic equivalence.",
            "",
            "| B | Ratio | Cohort input | E2E output | Comparable decode |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for comparison in summary["comparisons"]:
        input_ratio = comparison["median_cohort_input_ratio"]
        input_text = (
            f"{input_ratio:.3f}x"
            if input_ratio is not None
            else f"excluded ({comparison['cohort_input_ratio_exclusion']})"
        )
        decode = comparison["median_comparable_decode_ratio"]
        decode_text = (
            f"{decode:.3f}x"
            if decode is not None
            else f"excluded ({comparison['decode_ratio_exclusion']})"
        )
        lines.append(
            f"| {comparison['B']} | {comparison['numerator_engine']} / "
            f"{comparison['denominator_engine']} | "
            f"{input_text} | "
            f"{comparison['median_e2e_output_ratio']:.3f}x | {decode_text} |"
        )
    claim_gate = summary["ten_x_e2e_claim_gate"]
    lines.extend(
        [
            "",
            "## 10x E2E Claim Gate",
            "",
            "This observed-run gate passes only when minimum WKVM E2E output throughput divided by maximum incumbent throughput is at least 10.000x for both vLLM and SGLang at the same B. It is deliberately stricter than a median ratio and is not a statistical confidence interval.",
            "",
            "| B | Conservative WKVM / vLLM | Conservative WKVM / SGLang | All incumbents |",
            "|---:|---:|---:|---|",
        ]
    )
    for batch in claim_gate["batches"]:
        ratios = batch["incumbent_conservative_ratios"]
        status = "PASS" if batch["passes_all_incumbents"] else "FAIL"
        lines.append(
            f"| {batch['B']} | {ratios['vllm']:.3f}x | "
            f"{ratios['sglang']:.3f}x | **{status}** |"
        )
    lines.extend(["", "## Artifacts", ""])
    for artifact in contract["artifacts"]:
        provenance = artifact["provenance"]
        gpu_provenance = provenance["gpu"]
        gpu_name = gpu_provenance.get("gpu_name", gpu_provenance.get("name"))
        lines.extend(
            [
                f"- `{artifact['path']}`",
                f"  - Engine: `{artifact['engine']} {artifact['engine_version']}`; semantics: `{artifact['semantics']}`; B={','.join(str(value) for value in artifact['batch_sizes'])}",
                f"  - GPU provenance: {gpu_name}, driver {gpu_provenance.get('driver_version')}",
                f"  - Launch: `{artifact['launch_command']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in summary["caveats"])
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def write_report(
    paths: Iterable[Path],
    *,
    markdown_path: Path,
    summary_json_path: Path,
    min_samples: int = 3,
) -> dict[str, Any]:
    summary = build_summary(paths, min_samples=min_samples)
    markdown = render_markdown(summary)
    atomic_write(markdown_path, markdown)
    atomic_write(
        summary_json_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=SEQUENTIAL_RUNBOOK,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MINIMUM_PUBLIC_REPEATS,
        help="Required repeats per engine/B; must be at least 3.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        write_report(
            args.artifacts,
            markdown_path=args.markdown,
            summary_json_path=args.summary_json,
            min_samples=args.min_samples,
        )
    except ReliableReportError as exc:
        raise SystemExit(f"reliable report evidence validation failed: {exc}") from exc
    print(f"WROTE {args.markdown}")
    print(f"WROTE {args.summary_json}")


if __name__ == "__main__":
    main()
