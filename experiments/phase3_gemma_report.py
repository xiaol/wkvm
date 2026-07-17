#!/usr/bin/env python
"""Validate and summarize isolated WKVM Phase 3 benchmark profiles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


SUMMARY_SCHEMA = "wkvm.phase3_gemma_report.v1"
ARTIFACT_SCHEMA = "wkvm.native_gemma_bench.v1"
SOURCE_IDENTITY_SCHEMA = "wkvm.git_worktree_identity.sha256.v1"
MODEL_IDENTITY_SCHEMA = "wkvm.model_checkpoint_identity.sha256.v1"
MEMORY_SCHEMA = "wkvm.whole_gpu_memory.v1"
PROMPT_FINGERPRINT_SCHEMA = "wkvm.prompt_token_ids.sha256.v1"
OUTPUT_FINGERPRINT_SCHEMA = "wkvm.generated_output_token_ids.sha256.v1"
MINIMUM_REPEATS = 3
MAXIMUM_IDLE_BASELINE_GIB = 1.0
MINIMUM_CANDIDATE_PREFILL_RATIO = 1.05
PROFILE_PATTERN = re.compile(r"^(?P<profile>.+)-r(?P<repeat>[1-9][0-9]*)\.json$")


@dataclass(frozen=True)
class ProfileSpec:
    family: str
    batch: int
    context_tokens: int
    output_tokens: int
    projection_backend: str
    attention_backend: str
    batched_routed_packets: bool = False
    completion_prefill_lane_size: int = 0
    isolated_candidate: bool = False


PROFILE_SPECS = {
    "prefill-baseline": ProfileSpec(
        family="prefill",
        batch=8,
        context_tokens=16_384,
        output_tokens=1,
        projection_backend="separate",
        attention_backend="sdpa_single_gqa",
    ),
    "prefill-packed": ProfileSpec(
        family="prefill",
        batch=8,
        context_tokens=16_384,
        output_tokens=1,
        projection_backend="qkv_gate_up_packed",
        attention_backend="sdpa_single_gqa",
        isolated_candidate=True,
    ),
    "prefill-routed-packets": ProfileSpec(
        family="prefill",
        batch=8,
        context_tokens=16_384,
        output_tokens=1,
        projection_backend="separate",
        attention_backend="sdpa_single_gqa",
        batched_routed_packets=True,
        isolated_candidate=True,
    ),
    "prefill-native-gqa": ProfileSpec(
        family="prefill",
        batch=8,
        context_tokens=16_384,
        output_tokens=1,
        projection_backend="separate",
        attention_backend="triton_dense_gqa",
        isolated_candidate=True,
    ),
    "prefill-combined": ProfileSpec(
        family="prefill",
        batch=8,
        context_tokens=16_384,
        output_tokens=1,
        projection_backend="qkv_gate_up_packed",
        attention_backend="triton_dense_gqa",
        batched_routed_packets=True,
    ),
    "schedule-baseline": ProfileSpec(
        family="schedule",
        batch=16,
        context_tokens=16_384,
        output_tokens=32,
        projection_backend="separate",
        attention_backend="sdpa_single_gqa",
    ),
    "schedule-lane8": ProfileSpec(
        family="schedule",
        batch=16,
        context_tokens=16_384,
        output_tokens=32,
        projection_backend="separate",
        attention_backend="sdpa_single_gqa",
        completion_prefill_lane_size=8,
    ),
}


class Phase3ReportError(ValueError):
    """Raised when artifacts cannot support a Phase 3 comparison."""


@dataclass(frozen=True)
class ValidatedArtifact:
    path: Path
    profile: str
    repeat: int
    family: str
    source_identity_sha256: str
    git_commit: str
    model_identity_sha256: str
    gpu_identity: tuple[str, str, str]
    environment_signature: str
    prompt_fingerprint: str
    output_fingerprint: str
    config_signature: str
    metrics: dict[str, float]


def _fail(message: str) -> None:
    raise Phase3ReportError(message)


def _required_number(value: Any, *, source: str, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        _fail(f"{source}: {field} must be a positive finite number")
    return float(value)


def _required_string(value: Any, *, source: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{source}: {field} must be a non-empty string")
    return value


def _digest(value: Any, *, source: str, field: str) -> str:
    digest = _required_string(value, source=source, field=field).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        _fail(f"{source}: {field} must be a 64-character SHA-256 digest")
    return digest


def _close(actual: float, expected: float) -> bool:
    tolerance = max(0.02, abs(expected) * 0.01)
    return abs(actual - expected) <= tolerance


def _profile_from_path(path: Path) -> tuple[str, int]:
    match = PROFILE_PATTERN.fullmatch(path.name)
    if match is None:
        _fail(
            f"{path}: filename must be <profile>-rN.json for a known Phase 3 profile"
        )
    profile = match.group("profile")
    if profile not in PROFILE_SPECS:
        _fail(f"{path}: unknown Phase 3 profile {profile!r}")
    return profile, int(match.group("repeat"))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{path}: cannot load JSON: {exc}")
    if not isinstance(data, dict):
        _fail(f"{path}: top-level JSON must be an object")
    return data


def _validate_profile_config(
    data: dict[str, Any],
    row: dict[str, Any],
    *,
    spec: ProfileSpec,
    source: str,
) -> str:
    config = data.get("config")
    if not isinstance(config, dict):
        _fail(f"{source}: missing config object")
    expected = {
        "slots": spec.batch,
        "route_chunk": 512,
        "chunk": 2048,
        "prefill_microbatch_rows": 8,
        "decode_microbatch_rows": spec.batch,
        "decode_microbatch_bytes": None,
        "decode_batch_planner": "scheduler",
        "decode_workspace_bytes": None,
        "decode_workspace_width_bucket": 16,
        "cuda_phase_metrics": False,
        "persistent_exact_decode": True,
        "persistent_padded_decode": True,
        "persistent_padded_decode_steps": spec.output_tokens,
        "persistent_padded_full_attention_rows": None,
        "persistent_padded_sliding_metadata_padding": True,
        "persistent_padded_decode_cuda_graph": True,
        "persistent_padded_decode_graph_warmup_iters": 0,
        "native_gemma_projection_backend": spec.projection_backend,
        "native_gemma_attention_backend": spec.attention_backend,
        "native_gemma_weight_backend": "hf_live",
        "native_gemma_release_hf_decoder_layers": False,
        "use_native_gemma_forward": True,
        "uses_hf_config": False,
        "native_gemma_config_loader": True,
        "batched_routed_packets": spec.batched_routed_packets,
        "routed_packet_workspace_bytes": 67_108_864,
        "completion_prefill_lane_size": spec.completion_prefill_lane_size,
        "synthetic_prompts": True,
        "synthetic_vocab_size": 262_144,
        "enable_token_pool_attention": True,
        "enable_token_pool_metadata": None,
        "enable_token_pool_triton": True,
        "enable_token_pool_paged_triton": True,
        "enable_token_pool_paged_split_triton": True,
        "token_pool_max_context_len": 16_640,
        "token_pool_paged_block_size": 16,
        "token_pool_triton_strict": True,
        "token_pool_sliding_paged_metadata_only": True,
        "token_pool_triton_env": {
            "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON": "1",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON": "1",
            "WKVM_ENABLE_TOKEN_POOL_TRITON": "1",
            "WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY": "1",
            "WKVM_TOKEN_POOL_TRITON_STRICT": "1",
        },
        "sink": 16,
        "window": 1024,
        "m_slots": 32,
        "token_budget": None,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            _fail(
                f"{source}: config.{key}={config.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    expected_capacity = 65_536
    if config.get("token_pool_capacity") != expected_capacity:
        _fail(
            f"{source}: config.token_pool_capacity must be {expected_capacity}"
        )
    if data.get("native_gemma_checkpoint_loader") is not True:
        _fail(f"{source}: native checkpoint loader must be enabled")
    if data.get("uses_hf_transformer_forward") is not False:
        _fail(f"{source}: HF transformer forward must be disabled")
    if data.get("uses_hf_model_construction") is not False:
        _fail(f"{source}: HF model construction must be disabled")
    top_level_expected = {
        "native_gemma_attention_backend": spec.attention_backend,
        "native_gemma_projection_backend": spec.projection_backend,
        "native_gemma_weight_backend": "hf_live",
        "native_gemma_release_hf_decoder_layers": False,
        "native_gemma_config_loader": True,
        "uses_hf_config": False,
        "uses_hf_tokenizer": False,
        "use_native_gemma_forward": True,
        "token_pool_attention_enabled": True,
        "cuda_phase_metrics_enabled": False,
    }
    for key, expected_value in top_level_expected.items():
        if data.get(key) != expected_value:
            _fail(
                f"{source}: top-level {key}={data.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    expected_triton_env = config.get("token_pool_triton_env")
    if not isinstance(expected_triton_env, dict):
        _fail(f"{source}: missing token-pool Triton environment provenance")
    if data.get("token_pool_triton_env") != expected_triton_env:
        _fail(f"{source}: top-level Triton environment disagrees with config")
    if data.get("batched_routed_packets") is not spec.batched_routed_packets:
        _fail(f"{source}: top-level routed packet flag disagrees with profile")
    if data.get("completion_prefill_lane_size") != spec.completion_prefill_lane_size:
        _fail(f"{source}: top-level completion lane size disagrees with profile")
    native_requirement = data.get("native_no_hf_requirement")
    if not isinstance(native_requirement, dict) or native_requirement.get("passed") is not True:
        _fail(f"{source}: native no-HF requirement did not pass")

    routed_stats = row.get("routed_packets")
    if not isinstance(routed_stats, dict):
        _fail(f"{source}: missing routed packet stats")
    if spec.batched_routed_packets:
        if routed_stats.get("enabled") is not True:
            _fail(f"{source}: routed packet workspace was not enabled")
        if int(routed_stats.get("packet_batches", 0) or 0) < 1:
            _fail(f"{source}: routed packet profile produced zero packet batches")
        if int(routed_stats.get("capacity_fallback_batches", 0) or 0) != 0:
            _fail(f"{source}: routed packet profile used capacity fallback")
        packet_batches = int(routed_stats.get("packet_batches", 0) or 0)
        if int(routed_stats.get("d2h_copies", 0) or 0) != packet_batches:
            _fail(f"{source}: routed packet profile did not use one D2H copy per packet")
        if int(routed_stats.get("packet_folds", 0) or 0) <= packet_batches:
            _fail(f"{source}: routed packets did not batch multiple route folds")
        if int(routed_stats.get("packet_request_rows", 0) or 0) < 2:
            _fail(f"{source}: routed packets did not batch multiple request rows")
        if int(routed_stats.get("workspace_pinned_host_buffer_bytes", 0) or 0) < 1:
            _fail(f"{source}: routed packet host workspace was not pinned")
        if row.get("routed_packet_evidence_passed") is not True:
            _fail(f"{source}: routed packet evidence gate did not pass")
    elif routed_stats.get("enabled") is not False:
        _fail(f"{source}: non-packet profile unexpectedly enabled packet workspace")
    routed_evidence = data.get("routed_packet_evidence")
    if not isinstance(routed_evidence, dict):
        _fail(f"{source}: missing top-level routed packet evidence")
    if routed_evidence.get("required") is not spec.batched_routed_packets:
        _fail(f"{source}: routed packet evidence requirement disagrees with profile")
    if routed_evidence.get("passed") is not True:
        _fail(f"{source}: top-level routed packet evidence did not pass")

    native_forward_timing = row.get("native_forward_timing")
    if not isinstance(native_forward_timing, dict) or native_forward_timing.get("available") is not True:
        _fail(f"{source}: missing native forward dispatch telemetry")
    gqa_prefill_calls = int(native_forward_timing.get("dense_gqa_prefill_calls", 0) or 0)
    gqa_prefill_fallbacks = int(
        native_forward_timing.get("dense_gqa_prefill_fallbacks", 0) or 0
    )
    if spec.attention_backend == "triton_dense_gqa":
        if gqa_prefill_calls < 1 or gqa_prefill_fallbacks != 0:
            _fail(f"{source}: native dense GQA prefill dispatch was not exclusive")
    elif gqa_prefill_calls != 0 or gqa_prefill_fallbacks != 0:
        _fail(f"{source}: non-GQA profile reported dense GQA prefill activity")

    lane_starts = int(row.get("completion_prefill_lane_starts", 0) or 0)
    lane_completions = int(row.get("completion_prefill_lane_completions", 0) or 0)
    lane_cancellations = int(row.get("completion_prefill_lane_cancellations", 0) or 0)
    if spec.completion_prefill_lane_size:
        if row.get("completion_prefill_lane_size") != spec.completion_prefill_lane_size:
            _fail(f"{source}: completion-prefill lane size telemetry disagrees")
        if lane_starts < 1 or lane_completions != lane_starts:
            _fail(f"{source}: completion-prefill lanes did not all complete")
        if lane_cancellations != 0:
            _fail(f"{source}: completion-prefill lane cancellation observed")
    elif any((lane_starts, lane_completions, lane_cancellations)):
        _fail(f"{source}: disabled lane profile reported lane activity")

    signature_fields = dict(config)
    return json.dumps(signature_fields, sort_keys=True, separators=(",", ":"))


def validate_artifact(path: Path) -> ValidatedArtifact:
    profile, repeat = _profile_from_path(path)
    spec = PROFILE_SPECS[profile]
    data = _load_json(path)
    source = f"{path} profile={profile} repeat={repeat}"
    if data.get("schema") != ARTIFACT_SCHEMA or data.get("engine") != "wkvm-native":
        _fail(f"{source}: unsupported benchmark schema or engine")
    if data.get("context_tokens_per_session") != spec.context_tokens:
        _fail(f"{source}: context length does not match profile")
    if data.get("decode_tokens_per_session") != spec.output_tokens:
        _fail(f"{source}: output length does not match profile")
    if data.get("concurrency") != [spec.batch]:
        _fail(f"{source}: concurrency must be [{spec.batch}]")
    if data.get("prompt_lengths_mode") != "uniform":
        _fail(f"{source}: prompts must use uniform lengths")
    if data.get("prompt_token_source") != "synthetic":
        _fail(f"{source}: prompts must be synthetic and deterministic")
    if data.get("dtype") != "bfloat16":
        _fail(f"{source}: dtype must be bfloat16")
    if str(data.get("device", "")).lower() not in {"cuda", "cuda:0"}:
        _fail(f"{source}: compute device must be cuda or cuda:0")
    if data.get("attn") != "sdpa":
        _fail(f"{source}: model load attention policy must be sdpa")
    if data.get("warmup") is not False:
        _fail(f"{source}: Phase 3 evidence must use cold runs")
    if data.get("mem_cap_gib") != 24.0 or data.get("headroom_gib") != 4.0:
        _fail(f"{source}: memory policy must be 24 GiB with 4 GiB headroom")
    if data.get("max_baseline_gpu_used_gib") != 1.0:
        _fail(f"{source}: idle GPU ceiling must be 1 GiB")
    if data.get("fatal_error") is not None:
        _fail(f"{source}: artifact contains a fatal error")

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        _fail(f"{source}: artifact must contain exactly one benchmark row")
    row = rows[0]
    if row.get("B") != spec.batch:
        _fail(f"{source}: row B does not match profile")
    if (
        row.get("success_count") != spec.batch
        or row.get("error_count") != 0
        or row.get("error") is not None
    ):
        _fail(f"{source}: benchmark row is not fully successful")
    if row.get("green") is not True:
        _fail(f"{source}: benchmark row failed the memory gate")
    if row.get("torch_reserved_green") is not True:
        _fail(f"{source}: benchmark row failed the torch-reserved memory gate")
    if (
        row.get("uses_hf_transformer_forward") is not False
        or row.get("uses_hf_model_construction") is not False
        or row.get("native_gemma_checkpoint_loader") is not True
    ):
        _fail(f"{source}: row-level native no-HF evidence did not pass")

    coverage = data.get("generated_output_fingerprint_coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or coverage.get("successful_rows") != 1
        or coverage.get("fingerprinted_successful_rows") != 1
    ):
        _fail(f"{source}: generated output fingerprint coverage is incomplete")
    prompt_total = spec.batch * spec.context_tokens
    prompt_fingerprint = row.get("prompt_fingerprint")
    if (
        not isinstance(prompt_fingerprint, dict)
        or prompt_fingerprint.get("schema") != PROMPT_FINGERPRINT_SCHEMA
        or prompt_fingerprint.get("prompt_token_source") != "synthetic"
        or prompt_fingerprint.get("prompt_count") != spec.batch
        or prompt_fingerprint.get("prompt_lengths")
        != [spec.context_tokens] * spec.batch
        or prompt_fingerprint.get("prompt_total_tokens") != prompt_total
        or row.get("prompt_total_tokens") != prompt_total
        or row.get("cohort_input_token_count") != prompt_total
        or row.get("prompt_token_source") != "synthetic"
        or row.get("prompt_lengths") != [spec.context_tokens] * spec.batch
    ):
        _fail(f"{source}: prompt fingerprint or token accounting is invalid")
    prompt_digest = _digest(
        prompt_fingerprint.get("prompt_token_ids_sha256"),
        source=source,
        field="prompt fingerprint",
    )
    if row.get("prompt_token_ids_sha256") != prompt_digest:
        _fail(f"{source}: row prompt digest disagrees with fingerprint")

    output_fingerprint = row.get("generated_output_fingerprint")
    output_counts = [spec.output_tokens] * spec.batch
    output_total = spec.output_tokens * spec.batch
    if (
        not isinstance(output_fingerprint, dict)
        or output_fingerprint.get("schema") != OUTPUT_FINGERPRINT_SCHEMA
        or output_fingerprint.get("request_count") != spec.batch
        or output_fingerprint.get("request_ids")
        != sorted(
            [f"bench-{spec.batch}-{index}" for index in range(spec.batch)],
            key=lambda value: value.encode("utf-8"),
        )
        or output_fingerprint.get("output_token_counts") != output_counts
        or output_fingerprint.get("output_token_count") != output_total
        or row.get("generated_output_request_count") != spec.batch
        or row.get("generated_output_token_counts") != output_counts
        or row.get("generated_output_token_count") != output_total
        or row.get("generated_output_fingerprint_schema")
        != OUTPUT_FINGERPRINT_SCHEMA
        or row.get("generated_output_request_ids")
        != sorted(
            [f"bench-{spec.batch}-{index}" for index in range(spec.batch)],
            key=lambda value: value.encode("utf-8"),
        )
    ):
        _fail(f"{source}: generated output fingerprint or token accounting is invalid")
    output_digest = _digest(
        output_fingerprint.get("request_output_token_ids_sha256"),
        source=source,
        field="output fingerprint",
    )
    if row.get("request_output_token_ids_sha256") != output_digest:
        _fail(f"{source}: row output digest disagrees with fingerprint")

    provenance = data.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != "wkvm.native_gemma_bench.provenance.v1"
    ):
        _fail(f"{source}: missing provenance")
    benchmark = provenance.get("benchmark")
    if not isinstance(benchmark, dict):
        _fail(f"{source}: missing benchmark provenance")
    source_identity = benchmark.get("source_identity")
    if (
        not isinstance(source_identity, dict)
        or source_identity.get("schema") != SOURCE_IDENTITY_SCHEMA
        or source_identity.get("error") is not None
        or source_identity.get("excluded_paths") not in (None, [])
        or source_identity.get("git_worktree_dirty") is not False
        or benchmark.get("git_worktree_dirty") is not False
        or benchmark.get("source_identity_unchanged_during_run") is not True
    ):
        _fail(f"{source}: source identity is dirty, invalid, or changed during run")
    source_identity_sha256 = _digest(
        source_identity.get("identity_sha256"),
        source=source,
        field="source identity SHA-256",
    )
    git_commit = _required_string(
        benchmark.get("git_commit"), source=source, field="git commit"
    )
    if len(git_commit) != 40 or any(char not in "0123456789abcdef" for char in git_commit):
        _fail(f"{source}: git commit must be a 40-character hexadecimal digest")
    if data.get("git_commit") != git_commit:
        _fail(f"{source}: top-level git commit disagrees with provenance")
    if source_identity.get("git_commit") != git_commit:
        _fail(f"{source}: source identity git commit disagrees with benchmark")
    if benchmark.get("pre_run_source_identity_sha256") != source_identity_sha256:
        _fail(f"{source}: pre-run source identity disagrees with final identity")

    model_identity = data.get("model_identity")
    if (
        not isinstance(model_identity, dict)
        or model_identity.get("schema") != MODEL_IDENTITY_SCHEMA
        or model_identity.get("error") is not None
    ):
        _fail(f"{source}: invalid model identity")
    model_identity_sha256 = _digest(
        model_identity.get("manifest_sha256"),
        source=source,
        field="model manifest SHA-256",
    )

    gpu = provenance.get("gpu")
    if not isinstance(gpu, dict):
        _fail(f"{source}: missing GPU provenance")
    gpu_identity = (
        _required_string(gpu.get("device_uuid"), source=source, field="GPU UUID"),
        _required_string(gpu.get("gpu_name"), source=source, field="GPU name"),
        _required_string(
            gpu.get("driver_version"), source=source, field="GPU driver"
        ),
    )
    if gpu_identity[1] != "NVIDIA GeForce RTX 4090":
        _fail(f"{source}: Phase 3 4090 report requires an RTX 4090")
    memory_total_mib = gpu.get("memory_total_mib")
    if (
        not isinstance(memory_total_mib, int)
        or isinstance(memory_total_mib, bool)
        or not 23_000 <= memory_total_mib <= 26_000
    ):
        _fail(f"{source}: GPU memory total is not a 24 GiB-class RTX 4090")

    environment = provenance.get("environment")
    if not isinstance(environment, dict) or not isinstance(environment.get("packages"), dict):
        _fail(f"{source}: missing environment/package provenance")
    environment_signature = json.dumps(
        {
            "python_version": environment.get("python_version"),
            "python_implementation": environment.get("python_implementation"),
            "packages": environment.get("packages"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    memory = data.get("gpu_memory")
    if (
        not isinstance(memory, dict)
        or memory.get("schema") != MEMORY_SCHEMA
        or memory.get("scope") != "whole_device"
        or memory.get("source") != "nvidia-smi"
        or memory.get("error") is not None
        or memory.get("query_error_count") != 0
    ):
        _fail(f"{source}: invalid whole-GPU memory telemetry")
    sample_count = memory.get("sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 2:
        _fail(f"{source}: whole-GPU memory telemetry requires at least two samples")
    baseline_gib = _required_number(
        memory.get("baseline_used_mib"), source=source, field="GPU baseline MiB"
    ) / 1024.0 if memory.get("baseline_used_mib") != 0 else 0.0
    if baseline_gib > MAXIMUM_IDLE_BASELINE_GIB:
        _fail(
            f"{source}: pre-load GPU baseline {baseline_gib:.3f} GiB exceeds "
            f"{MAXIMUM_IDLE_BASELINE_GIB:.3f} GiB"
        )
    peak_used_mib = _required_number(
        memory.get("peak_used_mib"), source=source, field="GPU peak MiB"
    )
    peak_delta_mib = _required_number(
        memory.get("peak_delta_mib"), source=source, field="GPU peak delta MiB"
    )
    baseline_mib = float(memory.get("baseline_used_mib"))
    if (
        peak_used_mib < baseline_mib
        or peak_delta_mib < 0
        or abs(peak_delta_mib - (peak_used_mib - baseline_mib)) > 1
        or memory.get("device_uuid") != gpu_identity[0]
        or memory.get("gpu_name") != gpu_identity[1]
        or memory.get("driver_version") != gpu_identity[2]
        or memory.get("memory_total_mib") != memory_total_mib
    ):
        _fail(f"{source}: inconsistent whole-GPU memory telemetry")

    config_signature = _validate_profile_config(
        data, row, spec=spec, source=source
    )
    metrics = {
        key: _required_number(row.get(key), source=source, field=key)
        for key in (
            "cohort_input_tok_s",
            "cohort_prefill_wall_s",
            "prefill_time_p50_s",
            "prefill_time_p95_s",
            "p50_ttft_s",
            "p95_ttft_s",
            "max_ttft_s",
            "batch_wall_s",
            "e2e_output_tok_s",
            "peak_reserved_gib",
            "peak_engine_delta_gib",
        )
    }
    if not _close(metrics["cohort_input_tok_s"], prompt_total / metrics["cohort_prefill_wall_s"]):
        _fail(f"{source}: cohort input throughput disagrees with token accounting")
    if not _close(metrics["e2e_output_tok_s"], output_total / metrics["batch_wall_s"]):
        _fail(f"{source}: E2E output throughput disagrees with token accounting")
    if not _close(metrics["peak_engine_delta_gib"], peak_delta_mib / 1024.0):
        _fail(f"{source}: engine memory delta disagrees with whole-GPU telemetry")
    if metrics["peak_engine_delta_gib"] > 20.0:
        _fail(f"{source}: engine memory delta exceeds the 20 GiB Phase 3 gate")
    if metrics["peak_reserved_gib"] > 20.0:
        _fail(f"{source}: torch reserved memory exceeds the 20 GiB Phase 3 gate")
    if not (
        metrics["prefill_time_p50_s"] <= metrics["prefill_time_p95_s"]
        and metrics["p50_ttft_s"] <= metrics["p95_ttft_s"] <= metrics["max_ttft_s"]
    ):
        _fail(f"{source}: prefill or TTFT percentile ordering is invalid")
    if not _close(metrics["max_ttft_s"], metrics["cohort_prefill_wall_s"]):
        _fail(f"{source}: max TTFT disagrees with cohort prefill wall")
    return ValidatedArtifact(
        path=path,
        profile=profile,
        repeat=repeat,
        family=spec.family,
        source_identity_sha256=source_identity_sha256,
        git_commit=git_commit,
        model_identity_sha256=model_identity_sha256,
        gpu_identity=gpu_identity,
        environment_signature=environment_signature,
        prompt_fingerprint=prompt_digest,
        output_fingerprint=output_digest,
        config_signature=config_signature,
        metrics=metrics,
    )


def _aggregate(values: Iterable[float]) -> dict[str, float | int]:
    present = [float(value) for value in values]
    return {
        "count": len(present),
        "median": statistics.median(present),
        "min": min(present),
        "max": max(present),
    }


def build_summary(
    paths: Iterable[Path],
    *,
    min_samples: int = MINIMUM_REPEATS,
) -> dict[str, Any]:
    if (
        not isinstance(min_samples, int)
        or isinstance(min_samples, bool)
        or min_samples < MINIMUM_REPEATS
    ):
        _fail(f"min_samples must be an integer >= {MINIMUM_REPEATS}")
    artifacts = [validate_artifact(Path(path)) for path in paths]
    grouped: dict[str, list[ValidatedArtifact]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.profile, []).append(artifact)
    missing = sorted(set(PROFILE_SPECS) - set(grouped))
    if missing:
        _fail(f"missing required Phase 3 profiles: {missing}")
    extra = sorted(set(grouped) - set(PROFILE_SPECS))
    if extra:
        _fail(f"unexpected Phase 3 profiles: {extra}")
    for profile, group in grouped.items():
        if len(group) < min_samples:
            _fail(
                f"profile {profile} has {len(group)} repeats; require {min_samples}"
            )
        if len({artifact.repeat for artifact in group}) != len(group):
            _fail(f"profile {profile} has duplicate repeat numbers")
        if len({artifact.config_signature for artifact in group}) != 1:
            _fail(f"profile {profile} changed configuration across repeats")

    def require_one(label: str, values: set[Any]) -> Any:
        if len(values) != 1:
            _fail(f"artifacts disagree on {label}")
        return next(iter(values))

    source_identity = require_one(
        "source identity",
        {artifact.source_identity_sha256 for artifact in artifacts},
    )
    git_commit = require_one(
        "git commit", {artifact.git_commit for artifact in artifacts}
    )
    model_identity = require_one(
        "model identity",
        {artifact.model_identity_sha256 for artifact in artifacts},
    )
    gpu_identity = require_one(
        "GPU identity", {artifact.gpu_identity for artifact in artifacts}
    )
    environment_signature = require_one(
        "Python/package environment",
        {artifact.environment_signature for artifact in artifacts},
    )
    family_fingerprints: dict[str, dict[str, str]] = {}
    for family in ("prefill", "schedule"):
        family_artifacts = [artifact for artifact in artifacts if artifact.family == family]
        family_fingerprints[family] = {
            "prompt_token_ids_sha256": require_one(
                f"{family} prompt fingerprint",
                {artifact.prompt_fingerprint for artifact in family_artifacts},
            ),
            "request_output_token_ids_sha256": require_one(
                f"{family} output fingerprint",
                {artifact.output_fingerprint for artifact in family_artifacts},
            ),
        }

    metric_names = tuple(next(iter(artifacts)).metrics)
    groups = []
    for profile in PROFILE_SPECS:
        group = sorted(grouped[profile], key=lambda artifact: artifact.repeat)
        groups.append(
            {
                "profile": profile,
                "family": PROFILE_SPECS[profile].family,
                "sample_count": len(group),
                "metrics": {
                    metric: _aggregate(artifact.metrics[metric] for artifact in group)
                    for metric in metric_names
                },
                "artifacts": [artifact.path.as_posix() for artifact in group],
            }
        )
    groups_by_name = {group["profile"]: group for group in groups}
    baseline = groups_by_name["prefill-baseline"]
    prefill_comparisons = []
    for profile in (
        "prefill-packed",
        "prefill-routed-packets",
        "prefill-native-gqa",
        "prefill-combined",
    ):
        group = groups_by_name[profile]
        median_ratio = (
            group["metrics"]["cohort_input_tok_s"]["median"]
            / baseline["metrics"]["cohort_input_tok_s"]["median"]
        )
        conservative_ratio = (
            group["metrics"]["cohort_input_tok_s"]["min"]
            / baseline["metrics"]["cohort_input_tok_s"]["max"]
        )
        isolated = PROFILE_SPECS[profile].isolated_candidate
        prefill_comparisons.append(
            {
                "profile": profile,
                "median_input_throughput_ratio": median_ratio,
                "conservative_input_throughput_ratio": conservative_ratio,
                "median_peak_reserved_delta_gib": (
                    group["metrics"]["peak_reserved_gib"]["median"]
                    - baseline["metrics"]["peak_reserved_gib"]["median"]
                ),
                "isolated_candidate": isolated,
                "passes_candidate_gate": (
                    isolated and median_ratio >= MINIMUM_CANDIDATE_PREFILL_RATIO
                ),
            }
        )

    schedule_baseline = groups_by_name["schedule-baseline"]
    schedule_lane = groups_by_name["schedule-lane8"]
    schedule_comparison = {
        "profile": "schedule-lane8",
        "median_batch_wall_ratio": (
            schedule_lane["metrics"]["batch_wall_s"]["median"]
            / schedule_baseline["metrics"]["batch_wall_s"]["median"]
        ),
        "median_p50_ttft_ratio": (
            schedule_lane["metrics"]["p50_ttft_s"]["median"]
            / schedule_baseline["metrics"]["p50_ttft_s"]["median"]
        ),
        "median_p95_ttft_ratio": (
            schedule_lane["metrics"]["p95_ttft_s"]["median"]
            / schedule_baseline["metrics"]["p95_ttft_s"]["median"]
        ),
        "median_max_ttft_ratio": (
            schedule_lane["metrics"]["max_ttft_s"]["median"]
            / schedule_baseline["metrics"]["max_ttft_s"]["median"]
        ),
    }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "pass",
        "minimum_samples_per_profile": min_samples,
        "contract": {
            "git_commit": git_commit,
            "source_identity_sha256": source_identity,
            "model_identity_sha256": model_identity,
            "environment_signature": environment_signature,
            "gpu": {
                "uuid": gpu_identity[0],
                "name": gpu_identity[1],
                "driver_version": gpu_identity[2],
            },
            "family_fingerprints": family_fingerprints,
            "idle_baseline_ceiling_gib": MAXIMUM_IDLE_BASELINE_GIB,
            "green_engine_delta_ceiling_gib": 20.0,
        },
        "groups": groups,
        "prefill_comparisons": prefill_comparisons,
        "selected_candidates": [
            comparison["profile"]
            for comparison in prefill_comparisons
            if comparison["passes_candidate_gate"]
        ],
        "schedule_comparison": schedule_comparison,
        "caveats": [
            "Phase 3 profiles compare routed-span approximate semantics, not full-KV equivalence",
            "candidate selection uses the median of at least three cold runs",
            "completion-biased scheduling must report p95 and max TTFT alongside batch wall",
            "a passing Phase 3 profile still requires a separate incumbent comparison for any public speed claim",
        ],
    }


def _triplet(metric: dict[str, Any], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"{metric['median']:.3f} "
        f"[{metric['min']:.3f}, {metric['max']:.3f}]{suffix}"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("status") != "pass":
        _fail("cannot render an invalid Phase 3 summary")
    contract = summary["contract"]
    lines = [
        "# WKVM Phase 3 Repeated-Run Report",
        "",
        "Evidence gate: **PASS**.",
        "",
        f"- Minimum repeats: {summary['minimum_samples_per_profile']} per profile",
        f"- Commit: `{contract['git_commit']}`",
        f"- Source identity: `{contract['source_identity_sha256']}`",
        f"- Model identity: `{contract['model_identity_sha256']}`",
        f"- GPU: {contract['gpu']['name']} `{contract['gpu']['uuid']}`; driver {contract['gpu']['driver_version']}",
        "- All profiles passed clean-source, idle-GPU, completion, memory, feature-activation, and exact-output gates.",
        "",
        "## Profiles",
        "",
        "Cells are median [min, max] across validated cold runs.",
        "",
        "| Profile | n | Input tok/s | Prefill p50 | TTFT p50 | TTFT p95 | Batch wall | E2E tok/s | Peak reserved | Engine delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in summary["groups"]:
        metrics = group["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    group["profile"],
                    str(group["sample_count"]),
                    _triplet(metrics["cohort_input_tok_s"], "tok/s"),
                    _triplet(metrics["prefill_time_p50_s"], "s"),
                    _triplet(metrics["p50_ttft_s"], "s"),
                    _triplet(metrics["p95_ttft_s"], "s"),
                    _triplet(metrics["batch_wall_s"], "s"),
                    _triplet(metrics["e2e_output_tok_s"], "tok/s"),
                    _triplet(metrics["peak_reserved_gib"], "GiB"),
                    _triplet(metrics["peak_engine_delta_gib"], "GiB"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prefill Comparisons",
            "",
            "Ratios are profile / prefill-baseline. The 5% gate applies only to isolated opt-ins.",
            "",
            "| Profile | Median ratio | Conservative ratio | Reserved delta | Isolated | Candidate gate |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for comparison in summary["prefill_comparisons"]:
        lines.append(
            f"| {comparison['profile']} | "
            f"{comparison['median_input_throughput_ratio']:.3f}x | "
            f"{comparison['conservative_input_throughput_ratio']:.3f}x | "
            f"{comparison['median_peak_reserved_delta_gib']:+.3f} GiB | "
            f"{'yes' if comparison['isolated_candidate'] else 'no'} | "
            f"{'PASS' if comparison['passes_candidate_gate'] else 'FAIL'} |"
        )
    schedule = summary["schedule_comparison"]
    lines.extend(
        [
            "",
            "## Scheduling Tradeoff",
            "",
            "Ratios are schedule-lane8 / schedule-baseline; lower is better for latency and wall time.",
            "",
            "| Batch wall | TTFT p50 | TTFT p95 | TTFT max |",
            "|---:|---:|---:|---:|",
            f"| {schedule['median_batch_wall_ratio']:.3f}x | "
            f"{schedule['median_p50_ttft_ratio']:.3f}x | "
            f"{schedule['median_p95_ttft_ratio']:.3f}x | "
            f"{schedule['median_max_ttft_ratio']:.3f}x |",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in summary["caveats"])
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def write_report(
    paths: Iterable[Path],
    *,
    markdown_path: Path,
    summary_json_path: Path,
    min_samples: int = MINIMUM_REPEATS,
) -> dict[str, Any]:
    summary = build_summary(paths, min_samples=min_samples)
    _atomic_write(markdown_path, render_markdown(summary))
    _atomic_write(
        summary_json_path,
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument(
        "--min-samples",
        type=int,
        default=MINIMUM_REPEATS,
        help=f"Required repeats per profile; must be at least {MINIMUM_REPEATS}.",
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
    except Phase3ReportError as exc:
        raise SystemExit(f"Phase 3 evidence validation failed: {exc}") from exc
    print(f"WROTE {args.markdown}")
    print(f"WROTE {args.summary_json}")


if __name__ == "__main__":
    main()
