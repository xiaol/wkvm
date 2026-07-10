#!/usr/bin/env python
"""Concurrency benchmark for the native Gemma routed-span wkvm engine."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_engine_smoke import build_prompt, chunked_scheduler_config, prompt_lengths
from native_gemma_smoke import break_mask_for, load_model, resolve_model_path

from wkvm.core.request import Request
from wkvm.gemma_engine import GemmaNativeEngine
from wkvm.models.gemma import gemma4_e4b_routed_span_config


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_concurrency(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val < 1:
            raise argparse.ArgumentTypeError("concurrency values must be >= 1")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("--concurrency must contain at least one value")
    return vals


def bench_prompt_lengths(ctx: int, concurrency: int, mode: str) -> list[int]:
    if mode == "staggered":
        return prompt_lengths(ctx, concurrency)
    if mode == "uniform":
        return [ctx] * concurrency
    raise ValueError(f"unknown prompt length mode: {mode}")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def round_or_none(x: float | None, ndigits: int = 3) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(x, ndigits)


def hf_boundary_summary(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    evidence_rows = [
        row
        for row in rows
        if "uses_hf_transformer_forward" in row
        and "uses_hf_model_construction" in row
        and "native_gemma_checkpoint_loader" in row
    ]
    if evidence_rows:
        uses_hf_forward = any(
            row.get("uses_hf_transformer_forward") is True
            for row in evidence_rows
        )
        uses_hf_construction = any(
            row.get("uses_hf_model_construction") is True
            for row in evidence_rows
        )
        checkpoint_loader = all(
            row.get("native_gemma_checkpoint_loader") is True
            for row in evidence_rows
        )
        row_backends = [
            row.get("model_forward_backend")
            for row in evidence_rows
            if isinstance(row.get("model_forward_backend"), str)
        ]
        if row_backends and all(backend == row_backends[0] for backend in row_backends):
            model_forward_backend = row_backends[0]
        else:
            model_forward_backend = "mixed_or_unavailable"
    else:
        uses_hf_forward = not bool(args.use_native_gemma_forward)
        uses_hf_construction = not bool(args.native_gemma_checkpoint_loader)
        checkpoint_loader = bool(args.native_gemma_checkpoint_loader)
        model_forward_backend = (
            "wkvm_native_gemma_forward_bridge"
            if args.use_native_gemma_forward
            else "hf_transformers_gemma4_forward"
        )
    return {
        "evidence_rows": len(evidence_rows),
        "model_forward_backend": model_forward_backend,
        "uses_hf_transformer_forward": uses_hf_forward,
        "uses_hf_model_construction": uses_hf_construction,
        "native_gemma_checkpoint_loader": checkpoint_loader,
        "intent": {
            "use_native_gemma_forward": bool(args.use_native_gemma_forward),
            "native_gemma_checkpoint_loader": bool(
                args.native_gemma_checkpoint_loader
            ),
        },
    }


def native_no_hf_requirement_report(
    rows: list[dict[str, Any]],
    *,
    required: bool,
) -> dict[str, Any]:
    checked_rows = 0
    violations: list[dict[str, Any]] = []
    for row in rows:
        if row.get("success_count") != row.get("B"):
            continue
        checked_rows += 1
        problems = []
        if row.get("uses_hf_transformer_forward") is not False:
            problems.append("uses_hf_transformer_forward_not_false")
        if row.get("uses_hf_model_construction") is not False:
            problems.append("uses_hf_model_construction_not_false")
        if row.get("native_gemma_checkpoint_loader") is not True:
            problems.append("native_gemma_checkpoint_loader_not_true")
        if problems:
            violations.append({"B": row.get("B"), "problems": problems})
    if required and checked_rows == 0:
        violations.append({"B": None, "problems": ["no_successful_rows_to_check"]})
    return {
        "required": bool(required),
        "checked_successful_rows": checked_rows,
        "passed": not violations,
        "violations": violations,
    }


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def token_pool_triton_stats_snapshot() -> dict[str, Any]:
    try:
        from wkvm.runner.gemma_native_forward import token_pool_triton_stats

        stats = token_pool_triton_stats()
        stats["available"] = True
        return stats
    except Exception as exc:
        return {"available": False, "error": str(exc).splitlines()[0]}


def token_pool_triton_stats_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return after
    delta: dict[str, Any] = {"available": True}
    for key, value in after.items():
        if key in {
            "available",
            "env_enabled",
            "env_disabled",
            "split_size",
            "split_min_splits",
        }:
            delta[key] = value
        elif key == "disabled_shape_count":
            prev = before.get(key, 0)
            delta[key] = value
            if type(value) is int and type(prev) is int:
                delta["disabled_shape_delta"] = value - prev
        elif key == "fallback_reasons":
            reasons = {}
            prev_reasons = before.get(key, {})
            if isinstance(value, dict) and isinstance(prev_reasons, dict):
                for reason, count in value.items():
                    prev_count = prev_reasons.get(reason, 0)
                    if type(count) is int and type(prev_count) is int:
                        diff = count - prev_count
                        if diff:
                            reasons[reason] = diff
                    else:
                        reasons[reason] = count
            delta[key] = reasons
        else:
            prev = before.get(key, 0)
            if type(value) is int and type(prev) is int:
                delta[key] = value - prev
            else:
                delta[key] = value
    return delta


def native_forward_timing_stats_snapshot() -> dict[str, Any]:
    try:
        from wkvm.runner.gemma_native_forward import native_forward_timing_stats

        stats = native_forward_timing_stats()
        stats["available"] = True
        return stats
    except Exception as exc:
        return {"available": False, "error": str(exc).splitlines()[0]}


def native_forward_timing_stats_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return after
    delta: dict[str, Any] = {"available": True}
    for key, value in after.items():
        if key in {"available", "enabled"}:
            delta[key] = value
            continue
        prev = before.get(key)
        if type(value) in {int, float} and type(prev) in {int, float}:
            delta[key] = value - prev
        else:
            delta[key] = value
    return delta


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


def build_native_config(model, args):
    return gemma4_e4b_routed_span_config(
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


def cuda_peak_reserved_gib() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 2**30


def torch_usable_gib(mem_cap_gib: float) -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    usable = free + torch.cuda.memory_reserved()
    usable = min(usable, int(mem_cap_gib * 2**30))
    return usable / 2**30


def make_engine(model, cfg, prompts: list[list[int]], args) -> GemmaNativeEngine:
    slots = args.slots or len(prompts)
    sched_cfg = chunked_scheduler_config(
        prompts,
        slots=slots,
        token_budget=args.token_budget,
        chunk=args.chunk,
    )
    return GemmaNativeEngine(
        model,
        cfg,
        num_slots=slots,
        scheduler_config=sched_cfg,
        prefill_chunk=args.chunk,
        decode_microbatch_rows=args.decode_microbatch_rows,
        decode_microbatch_bytes=args.decode_microbatch_bytes,
        decode_batch_planner=args.decode_batch_planner,
        decode_workspace_bytes=args.decode_workspace_bytes,
        decode_workspace_width_bucket=args.decode_workspace_width_bucket,
        persistent_exact_decode=not args.disable_persistent_exact_decode,
        persistent_padded_decode=not args.disable_persistent_padded_decode,
        persistent_padded_decode_steps=args.persistent_padded_decode_steps,
        persistent_padded_full_attention_rows=(
            getattr(args, "persistent_padded_full_attention_rows", None)
        ),
        persistent_padded_sliding_metadata_padding=getattr(
            args,
            "persistent_padded_sliding_metadata_padding",
            False,
        ),
        persistent_padded_decode_cuda_graph=args.persistent_padded_decode_cuda_graph,
        persistent_padded_decode_graph_warmup_iters=(
            args.persistent_padded_decode_graph_warmup_iters
        ),
        use_native_gemma_forward=args.use_native_gemma_forward,
        native_gemma_attention_backend=args.native_gemma_attention_backend,
        native_gemma_projection_backend=getattr(
            args,
            "native_gemma_projection_backend",
            "separate",
        ),
        native_gemma_weight_backend=getattr(
            args,
            "native_gemma_weight_backend",
            "hf_live",
        ),
        native_gemma_release_hf_decoder_layers=getattr(
            args,
            "native_gemma_release_hf_decoder_layers",
            False,
        ),
        enable_token_pool_metadata=getattr(args, "enable_token_pool_metadata", None),
        enable_token_pool_attention=getattr(args, "enable_token_pool_attention", False),
        token_pool_max_context_len=getattr(args, "token_pool_max_context_len", None),
        token_pool_capacity=getattr(args, "token_pool_capacity", None),
        token_pool_paged_block_size=getattr(args, "token_pool_paged_block_size", None),
        collect_cuda_memory_phase_metrics=getattr(args, "cuda_phase_metrics", False),
    )


def run_row(model, tok, cfg, B: int, args, usable_gib: float | None) -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    row: dict[str, Any] = {
        "B": B,
        "success_count": 0,
        "error_count": 0,
        "p50_latency_s": None,
        "p95_latency_s": None,
        "agg_decode_tok_s": None,
        "e2e_output_tok_s": None,
        "peak_reserved_gib": None,
        "green": False,
        "elapsed_s": None,
        "error": None,
    }
    triton_stats_before = token_pool_triton_stats_snapshot()
    native_timing_before = native_forward_timing_stats_snapshot()
    native_decode_timing_before: dict[str, Any] | None = None
    started = time.perf_counter()
    engine: GemmaNativeEngine | None = None
    try:
        lengths = bench_prompt_lengths(args.ctx, B, args.prompt_lengths)
        prompts = [build_prompt(tok, n, i) for i, n in enumerate(lengths)]
        engine = make_engine(model, cfg, prompts, args)
        reqs = [
            Request(prompt_token_ids=prompt, max_new_tokens=args.out, req_id=f"bench-{B}-{i}")
            for i, prompt in enumerate(prompts)
        ]
        for req, prompt in zip(reqs, prompts):
            engine.add_request(req, break_mask=break_mask_for(tok, prompt))

        while engine.has_unfinished:
            if native_decode_timing_before is None and all(
                req.num_computed_tokens >= req.num_prompt_tokens for req in reqs
            ):
                native_decode_timing_before = native_forward_timing_stats_snapshot()
            engine.step()
            if engine.metrics.steps > args.max_steps:
                raise RuntimeError("native Gemma benchmark row did not converge")

        elapsed = time.perf_counter() - started
        traces = engine.finished_traces
        successes = [
            req
            for req in reqs
            if req.status.is_finished and len(req.output_token_ids) == req.max_new_tokens
        ]
        latencies = [
            traces[req.req_id].as_dict()["total_latency_s"]
            for req in successes
            if req.req_id in traces
        ]
        output_tokens = sum(len(req.output_token_ids) for req in successes)
        decode_tokens = sum(max(0, len(req.output_token_ids) - 1) for req in successes)
        decode_starts = [
            traces[req.req_id].first_token_time
            for req in successes
            if req.req_id in traces and traces[req.req_id].first_token_time is not None
        ]
        finish_times = [
            traces[req.req_id].finish_time
            for req in successes
            if req.req_id in traces and traces[req.req_id].finish_time is not None
        ]
        decode_s = None
        if decode_starts and finish_times:
            decode_s = max(finish_times) - min(decode_starts)
        peak_reserved = cuda_peak_reserved_gib()
        engine_stats = engine.stats()
        row.update(
            {
                "success_count": len(successes),
                "error_count": len(reqs) - len(successes) + engine.metrics.error_count,
                "p50_latency_s": round_or_none(statistics.median(latencies) if latencies else None),
                "p95_latency_s": round_or_none(percentile(latencies, 0.95)),
                "agg_decode_tok_s": round_or_none(
                    decode_tokens / decode_s if decode_s and decode_s > 0 else None
                ),
                "e2e_output_tok_s": round_or_none(output_tokens / elapsed if elapsed > 0 else None),
                "peak_reserved_gib": round_or_none(peak_reserved),
                "green": bool(
                    peak_reserved is not None
                    and usable_gib is not None
                    and peak_reserved <= usable_gib - args.headroom_gib
                ),
                "elapsed_s": round(elapsed, 3),
                "prompt_lengths": [len(p) for p in prompts],
                "model_forward_backend": engine_stats["model_forward_backend"],
                "uses_hf_transformer_forward": engine_stats["uses_hf_transformer_forward"],
                "uses_hf_model_construction": engine_stats[
                    "uses_hf_model_construction"
                ],
                "native_gemma_checkpoint_loader": engine_stats[
                    "native_gemma_checkpoint_loader"
                ],
                "native_gemma_attention_backend": engine_stats[
                    "native_gemma_attention_backend"
                ],
                "native_gemma_projection_backend": engine_stats[
                    "native_gemma_projection_backend"
                ],
                "native_gemma_weight_backend": engine_stats[
                    "native_gemma_weight_backend"
                ],
                "native_gemma_release_hf_decoder_layers": engine_stats[
                    "native_gemma_release_hf_decoder_layers"
                ],
                "native_gemma_released_hf_decoder_layers": engine_stats[
                    "native_gemma_released_hf_decoder_layers"
                ],
                "persistent_padded_full_attention_rows": engine_stats[
                    "persistent_padded_full_attention_rows"
                ],
                "token_pool_metadata_enabled": engine_stats["token_pool_metadata_enabled"],
                "token_pool_attention_enabled": engine_stats["token_pool_attention_enabled"],
                "token_pool": engine_stats["token_pool"],
                "token_pool_decode_metadata_batches": (
                    engine.metrics.token_pool_decode_metadata_batches
                ),
                "token_pool_decode_metadata_rows": (
                    engine.metrics.token_pool_decode_metadata_rows
                ),
                "token_pool_decode_covered_layer_type_batches": dict(
                    engine.metrics.token_pool_decode_covered_layer_type_batches
                ),
                "token_pool_decode_covered_layer_type_rows": dict(
                    engine.metrics.token_pool_decode_covered_layer_type_rows
                ),
                "token_pool_decode_graph_signature_recording": engine_stats.get(
                    "token_pool_decode_graph_signature_recording",
                    False,
                ),
                "token_pool_decode_graph_candidate_batches": (
                    engine.metrics.token_pool_decode_graph_candidate_batches
                ),
                "token_pool_decode_graph_static_shape_starts": (
                    engine.metrics.token_pool_decode_graph_static_shape_starts
                ),
                "token_pool_decode_graph_static_shape_reuses": (
                    engine.metrics.token_pool_decode_graph_static_shape_reuses
                ),
                "token_pool_decode_graph_shape_mismatches": (
                    engine.metrics.token_pool_decode_graph_shape_mismatches
                ),
                "token_pool_decode_graph_shape_mismatch_reasons": dict(
                    engine.metrics.token_pool_decode_graph_shape_mismatch_reasons
                ),
                "token_pool_full_attention_row_rebuilds": (
                    engine.metrics.token_pool_full_attention_row_rebuilds
                ),
                "token_pool_full_attention_row_reuses": (
                    engine.metrics.token_pool_full_attention_row_reuses
                ),
                "token_pool_full_attention_row_appends": (
                    engine.metrics.token_pool_full_attention_row_appends
                ),
                "token_pool_full_attention_row_invalidations": (
                    engine.metrics.token_pool_full_attention_row_invalidations
                ),
                "token_pool_slot_high_watermark": (
                    engine.metrics.token_pool_slot_high_watermark
                ),
                "steps": engine.metrics.steps,
                "max_decode_batch_rows": engine.metrics.max_decode_batch_rows,
                "max_decode_model_batch_rows": engine.metrics.max_decode_model_batch_rows,
                "max_decode_model_batch_bytes": engine.metrics.max_decode_model_batch_bytes,
                "decode_microbatch_splits": engine.metrics.decode_microbatch_splits,
                "decode_microbatch_byte_splits": engine.metrics.decode_microbatch_byte_splits,
                "decode_length_bucketed_batches": engine.metrics.decode_length_bucketed_batches,
                "decode_model_calls": engine.metrics.decode_model_calls,
                "batched_decode_model_calls": engine.metrics.batched_decode_model_calls,
                "fallback_decode_model_calls": engine.metrics.fallback_decode_model_calls,
                "decode_timing_merge_s": round_or_none(
                    engine.metrics.decode_timing_merge_s,
                    6,
                ),
                "decode_timing_model_forward_s": round_or_none(
                    engine.metrics.decode_timing_model_forward_s,
                    6,
                ),
                "decode_timing_commit_s": round_or_none(
                    engine.metrics.decode_timing_commit_s,
                    6,
                ),
                "decode_timing_split_s": round_or_none(
                    engine.metrics.decode_timing_split_s,
                    6,
                ),
                "decode_timing_mask_s": round_or_none(
                    engine.metrics.decode_timing_mask_s,
                    6,
                ),
                "decode_timing_total_s": round_or_none(
                    engine.metrics.decode_timing_total_s,
                    6,
                ),
                "decode_timing_graph_input_copy_s": round_or_none(
                    engine.metrics.decode_timing_graph_input_copy_s,
                    6,
                ),
                "decode_timing_graph_metadata_copy_s": round_or_none(
                    engine.metrics.decode_timing_graph_metadata_copy_s,
                    6,
                ),
                "decode_timing_graph_replay_s": round_or_none(
                    engine.metrics.decode_timing_graph_replay_s,
                    6,
                ),
                "token_pool_decode_graph_metadata_tensor_copies": (
                    engine.metrics.token_pool_decode_graph_metadata_tensor_copies
                ),
                "token_pool_decode_graph_metadata_tensor_copy_skips": (
                    engine.metrics.token_pool_decode_graph_metadata_tensor_copy_skips
                ),
                "token_pool_decode_graph_metadata_alias_fastpath_metadata_skips": (
                    engine.metrics.token_pool_decode_graph_metadata_alias_fastpath_metadata_skips
                ),
                "exact_decode_batch_rows": engine.metrics.exact_decode_batch_rows,
                "padded_decode_batch_rows": engine.metrics.padded_decode_batch_rows,
                "padded_decode_temp_bytes": engine.metrics.padded_decode_temp_bytes,
                "padded_decode_temp_mask_bytes": engine.metrics.padded_decode_temp_mask_bytes,
                "padded_decode_copied_kv_bytes": engine.metrics.padded_decode_copied_kv_bytes,
                "padded_decode_pad_kv_bytes": engine.metrics.padded_decode_pad_kv_bytes,
                "padded_decode_source_pad_kv_bytes": engine.metrics.padded_decode_source_pad_kv_bytes,
                "padded_decode_workspace_extra_pad_kv_bytes": engine.metrics.padded_decode_workspace_extra_pad_kv_bytes,
                "padded_decode_reserved_kv_bytes": engine.metrics.padded_decode_reserved_kv_bytes,
                "padded_decode_workspace_allocations": engine.metrics.padded_decode_workspace_allocations,
                "padded_decode_workspace_reuses": engine.metrics.padded_decode_workspace_reuses,
                "padded_decode_workspace_bypasses": engine.metrics.padded_decode_workspace_bypasses,
                "max_padded_decode_temp_bytes": engine.metrics.max_padded_decode_temp_bytes,
                "max_padded_decode_pad_slots": engine.metrics.max_padded_decode_pad_slots,
                "max_padded_decode_workspace_extra_pad_slots": engine.metrics.max_padded_decode_workspace_extra_pad_slots,
                "persistent_exact_decode_starts": engine.metrics.persistent_exact_decode_starts,
                "persistent_exact_decode_reuses": engine.metrics.persistent_exact_decode_reuses,
                "persistent_exact_decode_splits": engine.metrics.persistent_exact_decode_splits,
                "persistent_exact_decode_rows": engine.metrics.persistent_exact_decode_rows,
                "persistent_padded_decode_starts": engine.metrics.persistent_padded_decode_starts,
                "persistent_padded_decode_reuses": engine.metrics.persistent_padded_decode_reuses,
                "persistent_padded_decode_splits": engine.metrics.persistent_padded_decode_splits,
                "persistent_padded_decode_rows": engine.metrics.persistent_padded_decode_rows,
                "persistent_padded_decode_cuda_graph_captures": (
                    engine.metrics.persistent_padded_decode_cuda_graph_captures
                ),
                "persistent_padded_decode_cuda_graph_replays": (
                    engine.metrics.persistent_padded_decode_cuda_graph_replays
                ),
                "persistent_padded_decode_cuda_graph_skips": (
                    engine.metrics.persistent_padded_decode_cuda_graph_skips
                ),
                "persistent_padded_decode_cuda_graph_skip_reasons": dict(
                    engine.metrics.persistent_padded_decode_cuda_graph_skip_reasons
                ),
                "cuda_phase_metrics_enabled": engine_stats[
                    "cuda_phase_metrics_enabled"
                ],
                "decode_batch_fallback_reasons": dict(
                    engine.metrics.decode_batch_fallback_reasons
                ),
                "max_resident_state_slots": engine.metrics.max_resident_state_slots,
                "max_active_cache_bytes": engine.metrics.max_active_cache_bytes,
                "max_cuda_allocated_bytes": engine.metrics.max_cuda_allocated_bytes,
                "max_cuda_reserved_bytes": engine.metrics.max_cuda_reserved_bytes,
                "max_cuda_allocated_phase": engine.metrics.max_cuda_allocated_phase,
                "max_cuda_reserved_phase": engine.metrics.max_cuda_reserved_phase,
                "cuda_current_allocated_by_phase": dict(
                    engine.metrics.cuda_current_allocated_by_phase
                ),
                "cuda_current_reserved_by_phase": dict(
                    engine.metrics.cuda_current_reserved_by_phase
                ),
                "cuda_peak_allocated_advances_by_phase": dict(
                    engine.metrics.cuda_peak_allocated_advances_by_phase
                ),
                "cuda_peak_reserved_advances_by_phase": dict(
                    engine.metrics.cuda_peak_reserved_advances_by_phase
                ),
                "decode_cuda_current_allocated_by_phase": dict(
                    engine.metrics.decode_cuda_current_allocated_by_phase
                ),
                "decode_cuda_current_reserved_by_phase": dict(
                    engine.metrics.decode_cuda_current_reserved_by_phase
                ),
                "decode_cuda_peak_allocated_advances_by_phase": dict(
                    engine.metrics.decode_cuda_peak_allocated_advances_by_phase
                ),
                "decode_cuda_peak_reserved_advances_by_phase": dict(
                    engine.metrics.decode_cuda_peak_reserved_advances_by_phase
                ),
                "max_decode_cuda_allocated_bytes": (
                    engine.metrics.max_decode_cuda_allocated_bytes
                ),
                "max_decode_cuda_reserved_bytes": (
                    engine.metrics.max_decode_cuda_reserved_bytes
                ),
                "max_decode_cuda_allocated_phase": (
                    engine.metrics.max_decode_cuda_allocated_phase
                ),
                "max_decode_cuda_reserved_phase": (
                    engine.metrics.max_decode_cuda_reserved_phase
                ),
            }
        )
    except Exception as exc:
        row["error_count"] = max(B - row["success_count"], 1)
        row["error"] = str(exc).splitlines()[0]
        row["elapsed_s"] = round(time.perf_counter() - started, 3)
        row["peak_reserved_gib"] = round_or_none(cuda_peak_reserved_gib())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    row["token_pool_triton"] = token_pool_triton_stats_delta(
        triton_stats_before,
        token_pool_triton_stats_snapshot(),
    )
    native_timing_after = native_forward_timing_stats_snapshot()
    row["native_forward_timing"] = native_forward_timing_stats_delta(
        native_timing_before,
        native_timing_after,
    )
    if native_decode_timing_before is not None:
        row["native_forward_decode_timing"] = native_forward_timing_stats_delta(
            native_decode_timing_before,
            native_timing_after,
        )
    return row


def run(args) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    tok = AutoTokenizer.from_pretrained(path)
    if args.native_gemma_checkpoint_loader:
        args.use_native_gemma_forward = True
        if args.native_gemma_weight_backend != "hf_live":
            raise ValueError(
                "--native-gemma-checkpoint-loader owns checkpoint tensors directly "
                "and requires --native-gemma-weight-backend hf_live"
            )
        if args.native_gemma_release_hf_decoder_layers:
            raise ValueError(
                "--native-gemma-checkpoint-loader does not construct HF decoder "
                "layers, so --native-gemma-release-hf-decoder-layers is invalid"
            )
    release_per_row = bool(args.native_gemma_release_hf_decoder_layers)

    model = None
    cfg = None
    if not release_per_row:
        model = load_model(
            path,
            args.device,
            args.attn,
            native_checkpoint_loader=args.native_gemma_checkpoint_loader,
            native_gemma_attention_backend=args.native_gemma_attention_backend,
            native_gemma_projection_backend=args.native_gemma_projection_backend,
        )
        cfg = build_native_config(model, args)
    usable_gib = torch_usable_gib(args.mem_cap_gib)

    rows = []
    for B in args.concurrency:
        row_model = model
        row_cfg = cfg
        if release_per_row:
            row_model = load_model(
                path,
                args.device,
                args.attn,
                native_checkpoint_loader=args.native_gemma_checkpoint_loader,
                native_gemma_attention_backend=args.native_gemma_attention_backend,
                native_gemma_projection_backend=args.native_gemma_projection_backend,
            )
            row_cfg = build_native_config(row_model, args)
        try:
            row = run_row(row_model, tok, row_cfg, B, args, usable_gib)
        finally:
            if release_per_row:
                del row_model
                del row_cfg
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        rows.append(row)
        print(
            f"[wkvm-native ctx={args.ctx} out={args.out} B={B}] "
            f"success={row['success_count']}/{B} "
            f"p50={row['p50_latency_s']}s p95={row['p95_latency_s']}s "
            f"agg={row['agg_decode_tok_s']}tok/s "
            f"reserved={row['peak_reserved_gib']}GiB green={row['green']}"
        )
        if row.get("error") and args.stop_on_failure:
            break

    hf_boundary = hf_boundary_summary(rows, args)
    native_no_hf_requirement = native_no_hf_requirement_report(
        rows,
        required=args.require_native_no_hf,
    )
    payload: dict[str, Any] = {
        "schema": "wkvm.native_gemma_bench.v1",
        "engine": "wkvm-native",
        "context_tokens_per_session": args.ctx,
        "prompt_lengths_mode": args.prompt_lengths,
        "decode_tokens_per_session": args.out,
        "mem_cap_gib": args.mem_cap_gib,
        "headroom_gib": args.headroom_gib,
        "torch_usable_gib": round_or_none(usable_gib),
        "model_path": path,
        "dtype": "bfloat16",
        "device": args.device,
        "attn": args.attn,
        "model_forward_backend": hf_boundary["model_forward_backend"],
        "uses_hf_transformer_forward": hf_boundary[
            "uses_hf_transformer_forward"
        ],
        "uses_hf_model_construction": hf_boundary["uses_hf_model_construction"],
        "native_gemma_checkpoint_loader": hf_boundary[
            "native_gemma_checkpoint_loader"
        ],
        "hf_boundary": hf_boundary,
        "native_no_hf_requirement": native_no_hf_requirement,
        "native_gemma_attention_backend": args.native_gemma_attention_backend,
        "native_gemma_projection_backend": args.native_gemma_projection_backend,
        "native_gemma_weight_backend": args.native_gemma_weight_backend,
        "native_gemma_release_hf_decoder_layers": (
            args.native_gemma_release_hf_decoder_layers
        ),
        "token_pool_attention_enabled": args.enable_token_pool_attention,
        "cuda_phase_metrics_enabled": args.cuda_phase_metrics,
        "git_commit": git_commit(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "concurrency": args.concurrency,
        "config": {
            "sink": args.sink,
            "window": args.window,
            "m_slots": args.m_slots,
            "route_chunk": args.route_chunk,
            "chunk": args.chunk,
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
            "persistent_padded_decode_cuda_graph": args.persistent_padded_decode_cuda_graph,
            "persistent_padded_decode_graph_warmup_iters": (
                args.persistent_padded_decode_graph_warmup_iters
            ),
            "cuda_phase_metrics": args.cuda_phase_metrics,
            "use_native_gemma_forward": args.use_native_gemma_forward,
            "native_gemma_attention_backend": args.native_gemma_attention_backend,
            "native_gemma_projection_backend": args.native_gemma_projection_backend,
            "native_gemma_weight_backend": args.native_gemma_weight_backend,
            "native_gemma_release_hf_decoder_layers": (
                args.native_gemma_release_hf_decoder_layers
            ),
            "enable_token_pool_metadata": args.enable_token_pool_metadata,
            "enable_token_pool_attention": args.enable_token_pool_attention,
            "token_pool_max_context_len": args.token_pool_max_context_len,
            "token_pool_capacity": args.token_pool_capacity,
            "token_pool_paged_block_size": args.token_pool_paged_block_size,
            "slots": args.slots,
            "token_budget": args.token_budget,
        },
        "summary": {
            "bmax_green": max((r["B"] for r in rows if r["green"]), default=0),
            "max_success_B": max(
                (r["B"] for r in rows if r["success_count"] == r["B"]),
                default=0,
            ),
            "best_green_agg_decode_tok_s": max(
                (r["agg_decode_tok_s"] or 0.0 for r in rows if r["green"]),
                default=0.0,
            ),
        },
        "rows": rows,
    }
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.require_native_no_hf and not native_no_hf_requirement["passed"]:
        raise RuntimeError(
            "native no-HF requirement failed: "
            f"{native_no_hf_requirement['violations']}"
        )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency("1,8,16,32"))
    ap.add_argument("--prompt-lengths", choices=["staggered", "uniform"], default="staggered")
    ap.add_argument("--mem-cap-gib", type=float, default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)))
    ap.add_argument("--headroom-gib", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--decode-microbatch-rows", type=int, default=16)
    ap.add_argument("--decode-microbatch-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-width-bucket", type=int, default=16)
    ap.add_argument("--disable-persistent-exact-decode", action="store_true")
    ap.add_argument("--disable-persistent-padded-decode", action="store_true")
    ap.add_argument("--persistent-padded-decode-steps", type=int, default=8)
    full_attention_rows_group = ap.add_mutually_exclusive_group()
    full_attention_rows_group.add_argument(
        "--persistent-padded-full-attention-rows",
        dest="persistent_padded_full_attention_rows",
        action="store_true",
        default=None,
    )
    full_attention_rows_group.add_argument(
        "--disable-persistent-padded-full-attention-rows",
        dest="persistent_padded_full_attention_rows",
        action="store_false",
    )
    ap.add_argument("--persistent-padded-sliding-metadata-padding", action="store_true")
    ap.add_argument("--persistent-padded-decode-cuda-graph", action="store_true")
    ap.add_argument("--persistent-padded-decode-graph-warmup-iters", type=int, default=3)
    ap.add_argument(
        "--cuda-phase-metrics",
        action="store_true",
        default=env_flag("WKVM_CUDA_PHASE_METRICS"),
        help=(
            "Collect engine and padded-decode CUDA memory phase snapshots. "
            "Disabled by default because torch.cuda.memory_* calls perturb "
            "throughput measurements."
        ),
    )
    ap.add_argument(
        "--use-native-gemma-forward",
        action="store_true",
        help=(
            "Run model calls through wkvm's NativeGemma4ForCausalLM bridge instead "
            "of transformers.Gemma4ForCausalLM.forward. Still uses loaded HF weights."
        ),
    )
    ap.add_argument(
        "--native-gemma-checkpoint-loader",
        action="store_true",
        help=(
            "Load Gemma4 text tensors directly from safetensors into wkvm's native "
            "forward bridge instead of constructing transformers.Gemma4ForCausalLM. "
            "Still uses Transformers for config/tokenizer metadata."
        ),
    )
    ap.add_argument(
        "--native-gemma-attention-backend",
        choices=["manual", "manual_gqa", "sdpa", "sdpa_single_gqa", "triton_dense_gqa"],
        default="manual",
        help="Attention primitive used inside --use-native-gemma-forward.",
    )
    ap.add_argument(
        "--native-gemma-projection-backend",
        choices=["separate", "qkv_packed", "gate_up_packed", "qkv_gate_up_packed"],
        default="separate",
        help="Projection primitive used inside --use-native-gemma-forward.",
    )
    ap.add_argument(
        "--native-gemma-weight-backend",
        choices=["hf_live", "owned", "owned_cpu"],
        default="hf_live",
        help=(
            "Weight source used inside --use-native-gemma-forward. 'owned' copies "
            "decoder-layer weights into native tensors at bridge construction; "
            "'owned_cpu' keeps those snapshots on CPU and stages per operation."
        ),
    )
    ap.add_argument(
        "--native-gemma-release-hf-decoder-layers",
        action="store_true",
        help=(
            "After constructing the native owned-weight bridge, replace HF decoder "
            "layers with empty modules so benchmark memory reflects the owned "
            "native stack instead of duplicate HF decoder weights. Requires "
            "--native-gemma-weight-backend owned or owned_cpu."
        ),
    )
    ap.add_argument(
        "--enable-token-pool-metadata",
        action="store_true",
        default=None,
        help=(
            "Build token-pool request/token metadata without enabling token-pool "
            "attention. By default this is only enabled when token-pool attention "
            "or explicit token-pool sizing options are used."
        ),
    )
    ap.add_argument(
        "--enable-token-pool-attention",
        action="store_true",
        help=(
            "Experimental: backfill a token-granularity KV pool for native "
            "sliding-attention layers and pass ragged token metadata into native decode."
        ),
    )
    ap.add_argument("--token-pool-max-context-len", type=int, default=None)
    ap.add_argument("--token-pool-capacity", type=int, default=None)
    ap.add_argument(
        "--token-pool-paged-block-size",
        type=int,
        default=None,
    )
    ap.add_argument(
        "--decode-batch-planner",
        choices=["scheduler", "length_bucketed"],
        default="scheduler",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument(
        "--require-native-no-hf",
        action="store_true",
        help=(
            "After writing the benchmark payload, fail unless every successful "
            "row proves no HF model construction, no HF transformer forward, "
            "and use of the native Gemma checkpoint loader."
        ),
    )
    ap.add_argument("--stop-on-failure", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
