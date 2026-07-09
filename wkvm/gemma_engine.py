"""Native Gemma engine harness over the shared wkvm scheduler.

This is the N3 transition boundary: requests, admission, slot allocation, and
finish semantics are the same no-phases scheduler used by the RWKV engine, while
Gemma keeps one wkvm-owned routed-span cache object per live request. Fused
heterogeneous-cache decode is intentionally left for the graph/static-buffer
milestone; this harness makes that gap explicit in its metrics.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Literal

from wkvm.core.arena import StateArena
from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request
from wkvm.core.scheduler import Scheduler, SchedulerOutput
from wkvm.models.gemma import GemmaRoutedSpanConfig
from wkvm.runner.gemma_runner import (
    DistinctCacheBatchError,
    GemmaRoutedSpanRunner,
    NativeGemmaRoutedCache,
    PaddedDecodeWorkspace,
)
from wkvm.runner.gemma_state import GemmaRoutedStateBank
from wkvm.runner.gemma_token_pool import (
    DecodeBatchMetadata,
    PagedDecodeBatchMetadata,
    ReqToTokenTable,
    TokenKVLayerSpec,
    TokenKVPool,
    TokenPoolDecodeContext,
    TokenSlotRowChunks,
    TokenSlotAllocator,
    build_decode_metadata_from_token_slot_rows,
)


_DEFAULT_TOKEN_POOL_PAGED_BLOCK_SIZE = 16
_DEFAULT_TOKEN_POOL_PAGE_TABLE_METADATA_MAX_ROWS = 2


def _default_token_pool_paged_block_size() -> int:
    value = os.environ.get("WKVM_TOKEN_POOL_PAGED_BLOCK_SIZE")
    if value is None:
        return _DEFAULT_TOKEN_POOL_PAGED_BLOCK_SIZE
    return int(value)


def _default_token_pool_page_table_metadata_max_rows() -> int:
    value = os.environ.get("WKVM_TOKEN_POOL_PAGE_TABLE_METADATA_MAX_ROWS")
    if value is None:
        return _DEFAULT_TOKEN_POOL_PAGE_TABLE_METADATA_MAX_ROWS
    return int(value)


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _token_pool_paged_metadata_requested() -> bool:
    return any(
        _env_flag(name)
        for name in (
            "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON",
            "WKVM_TOKEN_POOL_BUILD_PAGED_METADATA",
        )
    )


def _token_pool_graph_signature_recording_requested() -> bool:
    return _env_flag("WKVM_TOKEN_POOL_RECORD_GRAPH_SIGNATURES")


def _sample_argmax_token_ids(logits: Any, *, rows: int | None = None) -> list[int]:
    """Greedy sample a batch with one host transfer for real tensor logits."""

    batch_logits = logits
    try:
        shape = tuple(int(dim) for dim in logits.shape)
    except Exception:
        shape = None
    if shape is not None:
        if len(shape) == 3:
            batch_logits = logits[:, -1, :]
        elif len(shape) == 1 and hasattr(logits, "reshape"):
            batch_logits = logits.reshape(1, -1)

    argmax = getattr(batch_logits, "argmax", None)
    if argmax is not None:
        try:
            token_ids = argmax(dim=-1)
        except TypeError:
            token_ids = None
        if token_ids is not None:
            if rows is not None:
                token_ids = token_ids[: int(rows)]
            detach = getattr(token_ids, "detach", None)
            if detach is not None:
                token_ids = detach()
            to_device = getattr(token_ids, "to", None)
            if to_device is not None:
                token_ids = to_device("cpu")
            tolist = getattr(token_ids, "tolist", None)
            if tolist is not None:
                values = tolist()
                if isinstance(values, int):
                    return [int(values)]
                return [int(value) for value in values]

    if rows is None:
        if shape is not None and shape:
            rows = 1 if len(shape) == 1 else int(shape[0])
        else:
            rows = len(logits)
    return [int(logits[row].argmax().item()) for row in range(int(rows))]


@dataclass
class GemmaEngineMetrics:
    steps: int = 0
    scheduled_tokens: int = 0
    admitted_requests: int = 0
    finished_requests: int = 0
    error_count: int = 0
    prefill_calls: int = 0
    decode_batches: int = 0
    decode_rows: int = 0
    decode_model_calls: int = 0
    batched_decode_model_calls: int = 0
    fallback_decode_model_calls: int = 0
    max_decode_batch_rows: int = 0
    max_decode_model_batch_rows: int = 0
    max_decode_model_batch_bytes: int = 0
    decode_microbatch_splits: int = 0
    decode_microbatch_byte_splits: int = 0
    decode_length_bucketed_batches: int = 0
    distinct_history_decode_batches: int = 0
    exact_decode_batch_rows: int = 0
    padded_decode_batch_rows: int = 0
    padded_decode_temp_bytes: int = 0
    padded_decode_temp_mask_bytes: int = 0
    padded_decode_copied_kv_bytes: int = 0
    padded_decode_pad_kv_bytes: int = 0
    padded_decode_source_pad_kv_bytes: int = 0
    padded_decode_workspace_extra_pad_kv_bytes: int = 0
    padded_decode_reserved_kv_bytes: int = 0
    padded_decode_workspace_allocations: int = 0
    padded_decode_workspace_reuses: int = 0
    padded_decode_workspace_bypasses: int = 0
    max_padded_decode_temp_bytes: int = 0
    max_padded_decode_pad_slots: int = 0
    max_padded_decode_workspace_extra_pad_slots: int = 0
    decode_timing_merge_s: float = 0.0
    decode_timing_model_forward_s: float = 0.0
    decode_timing_commit_s: float = 0.0
    decode_timing_split_s: float = 0.0
    decode_timing_mask_s: float = 0.0
    decode_timing_total_s: float = 0.0
    decode_timing_graph_input_copy_s: float = 0.0
    decode_timing_graph_metadata_copy_s: float = 0.0
    decode_timing_graph_replay_s: float = 0.0
    persistent_exact_decode_starts: int = 0
    persistent_exact_decode_reuses: int = 0
    persistent_exact_decode_splits: int = 0
    persistent_exact_decode_rows: int = 0
    persistent_padded_decode_starts: int = 0
    persistent_padded_decode_reuses: int = 0
    persistent_padded_decode_splits: int = 0
    persistent_padded_decode_rows: int = 0
    persistent_padded_decode_cuda_graph_captures: int = 0
    persistent_padded_decode_cuda_graph_replays: int = 0
    persistent_padded_decode_cuda_graph_skips: int = 0
    persistent_padded_decode_cuda_graph_skip_reasons: dict[str, int] = field(
        default_factory=dict
    )
    token_pool_decode_metadata_batches: int = 0
    token_pool_decode_metadata_rows: int = 0
    token_pool_decode_covered_layer_type_batches: dict[str, int] = field(
        default_factory=dict
    )
    token_pool_decode_covered_layer_type_rows: dict[str, int] = field(
        default_factory=dict
    )
    token_pool_decode_graph_candidate_batches: int = 0
    token_pool_decode_graph_static_shape_starts: int = 0
    token_pool_decode_graph_static_shape_reuses: int = 0
    token_pool_decode_graph_shape_mismatches: int = 0
    token_pool_decode_graph_shape_mismatch_reasons: dict[str, int] = field(
        default_factory=dict
    )
    token_pool_decode_graph_metadata_tensor_copies: int = 0
    token_pool_decode_graph_metadata_tensor_copy_skips: int = 0
    token_pool_full_attention_row_rebuilds: int = 0
    token_pool_full_attention_row_reuses: int = 0
    token_pool_full_attention_row_appends: int = 0
    token_pool_full_attention_row_invalidations: int = 0
    token_pool_slot_high_watermark: int = 0
    max_waiting: int = 0
    max_running: int = 0
    max_runnable_rows: int = 0
    max_resident_state_slots: int = 0
    max_active_cache_bytes: int = 0
    backpressure_events: int = 0
    retraction_events: int = 0
    max_cuda_allocated_bytes: int = 0
    max_cuda_reserved_bytes: int = 0
    max_cuda_allocated_phase: str = ""
    max_cuda_reserved_phase: str = ""
    cuda_current_allocated_by_phase: dict[str, int] = field(default_factory=dict)
    cuda_current_reserved_by_phase: dict[str, int] = field(default_factory=dict)
    cuda_peak_allocated_advances_by_phase: dict[str, int] = field(default_factory=dict)
    cuda_peak_reserved_advances_by_phase: dict[str, int] = field(default_factory=dict)
    decode_cuda_current_allocated_by_phase: dict[str, int] = field(default_factory=dict)
    decode_cuda_current_reserved_by_phase: dict[str, int] = field(default_factory=dict)
    decode_cuda_peak_allocated_advances_by_phase: dict[str, int] = field(default_factory=dict)
    decode_cuda_peak_reserved_advances_by_phase: dict[str, int] = field(default_factory=dict)
    max_decode_cuda_allocated_bytes: int = 0
    max_decode_cuda_reserved_bytes: int = 0
    max_decode_cuda_allocated_phase: str = ""
    max_decode_cuda_reserved_phase: str = ""
    backpressure_reasons: dict[str, int] = field(default_factory=dict)
    decode_batch_fallback_reasons: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "scheduled_tokens": self.scheduled_tokens,
            "admitted_requests": self.admitted_requests,
            "finished_requests": self.finished_requests,
            "error_count": self.error_count,
            "prefill_calls": self.prefill_calls,
            "decode_batches": self.decode_batches,
            "decode_rows": self.decode_rows,
            "decode_model_calls": self.decode_model_calls,
            "batched_decode_model_calls": self.batched_decode_model_calls,
            "fallback_decode_model_calls": self.fallback_decode_model_calls,
            "max_decode_batch_rows": self.max_decode_batch_rows,
            "max_decode_model_batch_rows": self.max_decode_model_batch_rows,
            "max_decode_model_batch_bytes": self.max_decode_model_batch_bytes,
            "decode_microbatch_splits": self.decode_microbatch_splits,
            "decode_microbatch_byte_splits": self.decode_microbatch_byte_splits,
            "decode_length_bucketed_batches": self.decode_length_bucketed_batches,
            "distinct_history_decode_batches": self.distinct_history_decode_batches,
            "exact_decode_batch_rows": self.exact_decode_batch_rows,
            "padded_decode_batch_rows": self.padded_decode_batch_rows,
            "padded_decode_temp_bytes": self.padded_decode_temp_bytes,
            "padded_decode_temp_mask_bytes": self.padded_decode_temp_mask_bytes,
            "padded_decode_copied_kv_bytes": self.padded_decode_copied_kv_bytes,
            "padded_decode_pad_kv_bytes": self.padded_decode_pad_kv_bytes,
            "padded_decode_source_pad_kv_bytes": self.padded_decode_source_pad_kv_bytes,
            "padded_decode_workspace_extra_pad_kv_bytes": self.padded_decode_workspace_extra_pad_kv_bytes,
            "padded_decode_reserved_kv_bytes": self.padded_decode_reserved_kv_bytes,
            "padded_decode_workspace_allocations": self.padded_decode_workspace_allocations,
            "padded_decode_workspace_reuses": self.padded_decode_workspace_reuses,
            "padded_decode_workspace_bypasses": self.padded_decode_workspace_bypasses,
            "max_padded_decode_temp_bytes": self.max_padded_decode_temp_bytes,
            "max_padded_decode_pad_slots": self.max_padded_decode_pad_slots,
            "max_padded_decode_workspace_extra_pad_slots": self.max_padded_decode_workspace_extra_pad_slots,
            "decode_timing_merge_s": self.decode_timing_merge_s,
            "decode_timing_model_forward_s": self.decode_timing_model_forward_s,
            "decode_timing_commit_s": self.decode_timing_commit_s,
            "decode_timing_split_s": self.decode_timing_split_s,
            "decode_timing_mask_s": self.decode_timing_mask_s,
            "decode_timing_total_s": self.decode_timing_total_s,
            "decode_timing_graph_input_copy_s": self.decode_timing_graph_input_copy_s,
            "decode_timing_graph_metadata_copy_s": (
                self.decode_timing_graph_metadata_copy_s
            ),
            "decode_timing_graph_replay_s": self.decode_timing_graph_replay_s,
            "persistent_exact_decode_starts": self.persistent_exact_decode_starts,
            "persistent_exact_decode_reuses": self.persistent_exact_decode_reuses,
            "persistent_exact_decode_splits": self.persistent_exact_decode_splits,
            "persistent_exact_decode_rows": self.persistent_exact_decode_rows,
            "persistent_padded_decode_starts": self.persistent_padded_decode_starts,
            "persistent_padded_decode_reuses": self.persistent_padded_decode_reuses,
            "persistent_padded_decode_splits": self.persistent_padded_decode_splits,
            "persistent_padded_decode_rows": self.persistent_padded_decode_rows,
            "persistent_padded_decode_cuda_graph_captures": self.persistent_padded_decode_cuda_graph_captures,
            "persistent_padded_decode_cuda_graph_replays": self.persistent_padded_decode_cuda_graph_replays,
            "persistent_padded_decode_cuda_graph_skips": self.persistent_padded_decode_cuda_graph_skips,
            "persistent_padded_decode_cuda_graph_skip_reasons": dict(
                self.persistent_padded_decode_cuda_graph_skip_reasons
            ),
            "token_pool_decode_metadata_batches": self.token_pool_decode_metadata_batches,
            "token_pool_decode_metadata_rows": self.token_pool_decode_metadata_rows,
            "token_pool_decode_covered_layer_type_batches": dict(
                self.token_pool_decode_covered_layer_type_batches
            ),
            "token_pool_decode_covered_layer_type_rows": dict(
                self.token_pool_decode_covered_layer_type_rows
            ),
            "token_pool_decode_graph_candidate_batches": (
                self.token_pool_decode_graph_candidate_batches
            ),
            "token_pool_decode_graph_static_shape_starts": (
                self.token_pool_decode_graph_static_shape_starts
            ),
            "token_pool_decode_graph_static_shape_reuses": (
                self.token_pool_decode_graph_static_shape_reuses
            ),
            "token_pool_decode_graph_shape_mismatches": (
                self.token_pool_decode_graph_shape_mismatches
            ),
            "token_pool_decode_graph_shape_mismatch_reasons": dict(
                self.token_pool_decode_graph_shape_mismatch_reasons
            ),
            "token_pool_decode_graph_metadata_tensor_copies": (
                self.token_pool_decode_graph_metadata_tensor_copies
            ),
            "token_pool_decode_graph_metadata_tensor_copy_skips": (
                self.token_pool_decode_graph_metadata_tensor_copy_skips
            ),
            "token_pool_full_attention_row_rebuilds": (
                self.token_pool_full_attention_row_rebuilds
            ),
            "token_pool_full_attention_row_reuses": (
                self.token_pool_full_attention_row_reuses
            ),
            "token_pool_full_attention_row_appends": (
                self.token_pool_full_attention_row_appends
            ),
            "token_pool_full_attention_row_invalidations": (
                self.token_pool_full_attention_row_invalidations
            ),
            "token_pool_slot_high_watermark": self.token_pool_slot_high_watermark,
            "max_waiting": self.max_waiting,
            "max_running": self.max_running,
            "max_runnable_rows": self.max_runnable_rows,
            "max_resident_state_slots": self.max_resident_state_slots,
            "max_active_cache_bytes": self.max_active_cache_bytes,
            "max_cuda_allocated_bytes": self.max_cuda_allocated_bytes,
            "max_cuda_reserved_bytes": self.max_cuda_reserved_bytes,
            "max_cuda_allocated_phase": self.max_cuda_allocated_phase,
            "max_cuda_reserved_phase": self.max_cuda_reserved_phase,
            "cuda_current_allocated_by_phase": dict(self.cuda_current_allocated_by_phase),
            "cuda_current_reserved_by_phase": dict(self.cuda_current_reserved_by_phase),
            "cuda_peak_allocated_advances_by_phase": dict(
                self.cuda_peak_allocated_advances_by_phase
            ),
            "cuda_peak_reserved_advances_by_phase": dict(
                self.cuda_peak_reserved_advances_by_phase
            ),
            "decode_cuda_current_allocated_by_phase": dict(
                self.decode_cuda_current_allocated_by_phase
            ),
            "decode_cuda_current_reserved_by_phase": dict(
                self.decode_cuda_current_reserved_by_phase
            ),
            "decode_cuda_peak_allocated_advances_by_phase": dict(
                self.decode_cuda_peak_allocated_advances_by_phase
            ),
            "decode_cuda_peak_reserved_advances_by_phase": dict(
                self.decode_cuda_peak_reserved_advances_by_phase
            ),
            "max_decode_cuda_allocated_bytes": self.max_decode_cuda_allocated_bytes,
            "max_decode_cuda_reserved_bytes": self.max_decode_cuda_reserved_bytes,
            "max_decode_cuda_allocated_phase": self.max_decode_cuda_allocated_phase,
            "max_decode_cuda_reserved_phase": self.max_decode_cuda_reserved_phase,
            "backpressure_events": self.backpressure_events,
            "retraction_events": self.retraction_events,
            "backpressure_reasons": dict(self.backpressure_reasons),
            "decode_batch_fallback_reasons": dict(self.decode_batch_fallback_reasons),
        }


@dataclass
class GemmaRequestTrace:
    req_id: str
    enqueue_time: float
    prompt_tokens: int
    target_output_tokens: int
    prefill_start: float | None = None
    prefill_end: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    output_tokens: int = 0
    finish_reason: str | None = None
    error: str | None = None

    def as_dict(self, *, now: float | None = None) -> dict[str, Any]:
        t = self.finish_time if self.finish_time is not None else (now or time.perf_counter())
        prefill_time = None
        if self.prefill_start is not None and self.prefill_end is not None:
            prefill_time = self.prefill_end - self.prefill_start
        first_token_latency = None
        if self.first_token_time is not None:
            first_token_latency = self.first_token_time - self.enqueue_time
        return {
            "req_id": self.req_id,
            "prompt_tokens": self.prompt_tokens,
            "target_output_tokens": self.target_output_tokens,
            "output_tokens": self.output_tokens,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "queue_time_s": (
                None
                if self.prefill_start is None
                else round(self.prefill_start - self.enqueue_time, 6)
            ),
            "prefill_time_s": None if prefill_time is None else round(prefill_time, 6),
            "decode_time_s": (
                None
                if self.first_token_time is None
                else round(max(0.0, t - self.first_token_time), 6)
            ),
            "first_token_latency_s": (
                None if first_token_latency is None else round(first_token_latency, 6)
            ),
            "total_latency_s": round(t - self.enqueue_time, 6),
        }


@dataclass
class _TokenPoolDecodeReservation:
    req_id: str
    req_slot: int
    token_slot: int
    token_slot_tensor: Any
    previous_length: int
    full_attention_token_slot: int | None = None
    persistent_full_attention_row: bool = False


@dataclass
class _TokenPoolFullAttentionRow:
    row_slots: list[int]
    owned_slots: list[int]
    append_slots: list[int] = field(default_factory=list)


class GemmaNativeEngine:
    """Scheduler-owned native Gemma routed-span engine.

    The current Gemma runner supports scheduler-driven chunked prompt prefill
    and one-cache decode. Batched decode is available when independently
    prefetched caches can be merged exactly or via the padded decode path.
    """

    def __init__(
        self,
        model: Any,
        config: GemmaRoutedSpanConfig,
        num_slots: int,
        scheduler_config: SchedulerConfig | None = None,
        stop_token_ids: frozenset[int] = frozenset(),
        prefill_chunk: int = 2048,
        decode_microbatch_rows: int | None = 16,
        decode_microbatch_bytes: int | None = None,
        decode_batch_planner: Literal["scheduler", "length_bucketed"] = "scheduler",
        decode_workspace_bytes: int | None = None,
        decode_workspace_width_bucket: int = 16,
        persistent_exact_decode: bool = True,
        persistent_padded_decode: bool = True,
        persistent_padded_decode_steps: int = 8,
        persistent_padded_full_attention_rows: bool | None = None,
        persistent_padded_sliding_metadata_padding: bool = False,
        persistent_padded_decode_cuda_graph: bool = False,
        persistent_padded_decode_graph_warmup_iters: int = 3,
        use_native_gemma_forward: bool = False,
        native_gemma_attention_backend: Literal[
            "manual",
            "manual_gqa",
            "sdpa",
            "sdpa_single_gqa",
            "triton_dense_gqa",
        ] = "manual",
        native_gemma_projection_backend: Literal[
            "separate",
            "qkv_packed",
            "gate_up_packed",
            "qkv_gate_up_packed",
        ] = "separate",
        native_gemma_weight_backend: Literal["hf_live", "owned", "owned_cpu"] = "hf_live",
        native_gemma_release_hf_decoder_layers: bool = False,
        enable_token_pool_metadata: bool | None = None,
        enable_token_pool_attention: bool = False,
        token_pool_max_context_len: int | None = None,
        token_pool_capacity: int | None = None,
        token_pool_paged_block_size: int | None = None,
        collect_cuda_memory_phase_metrics: bool = False,
        finished_trace_limit: int | None = 4096,
    ) -> None:
        if decode_microbatch_rows == 0:
            decode_microbatch_rows = None
        if decode_microbatch_rows is not None and decode_microbatch_rows < 1:
            raise ValueError("decode_microbatch_rows must be >= 1, 0, or None")
        if decode_microbatch_bytes == 0:
            decode_microbatch_bytes = None
        if decode_microbatch_bytes is not None and decode_microbatch_bytes < 1:
            raise ValueError("decode_microbatch_bytes must be >= 1, 0, or None")
        if decode_batch_planner not in {"scheduler", "length_bucketed"}:
            raise ValueError("decode_batch_planner must be 'scheduler' or 'length_bucketed'")
        if decode_workspace_bytes == 0:
            decode_workspace_bytes = None
        if decode_workspace_bytes is not None and decode_workspace_bytes < 1:
            raise ValueError("decode_workspace_bytes must be >= 1, 0, or None")
        if decode_workspace_width_bucket < 1:
            raise ValueError("decode_workspace_width_bucket must be >= 1")
        if finished_trace_limit is not None and finished_trace_limit < 1:
            raise ValueError("finished_trace_limit must be >= 1 or None")
        if persistent_padded_decode_steps < 1:
            raise ValueError("persistent_padded_decode_steps must be >= 1")
        if persistent_padded_decode_graph_warmup_iters < 0:
            raise ValueError("persistent_padded_decode_graph_warmup_iters must be >= 0")
        if native_gemma_attention_backend not in {
            "manual",
            "manual_gqa",
            "sdpa",
            "sdpa_single_gqa",
            "triton_dense_gqa",
        }:
            raise ValueError(
                "native_gemma_attention_backend must be 'manual', 'manual_gqa', "
                "'sdpa', 'sdpa_single_gqa', or 'triton_dense_gqa'"
            )
        if native_gemma_projection_backend not in {
            "separate",
            "qkv_packed",
            "gate_up_packed",
            "qkv_gate_up_packed",
        }:
            raise ValueError(
                "native_gemma_projection_backend must be 'separate', 'qkv_packed', "
                "'gate_up_packed', or 'qkv_gate_up_packed'"
            )
        if native_gemma_weight_backend not in {"hf_live", "owned", "owned_cpu"}:
            raise ValueError(
                "native_gemma_weight_backend must be 'hf_live', 'owned', or 'owned_cpu'"
            )
        if (
            native_gemma_release_hf_decoder_layers
            and native_gemma_weight_backend not in {"owned", "owned_cpu"}
        ):
            raise ValueError(
                "native_gemma_release_hf_decoder_layers requires "
                "native_gemma_weight_backend='owned' or 'owned_cpu'"
            )
        token_pool_config_requested = (
            token_pool_max_context_len is not None
            or token_pool_capacity is not None
            or token_pool_paged_block_size is not None
        )
        if token_pool_max_context_len is not None and token_pool_max_context_len < 1:
            raise ValueError("token_pool_max_context_len must be >= 1 or None")
        if token_pool_capacity is not None and token_pool_capacity < 1:
            raise ValueError("token_pool_capacity must be >= 1 or None")
        if token_pool_paged_block_size is None:
            token_pool_paged_block_size = _default_token_pool_paged_block_size()
        token_pool_paged_block_size = int(token_pool_paged_block_size)
        if token_pool_paged_block_size < 1:
            raise ValueError("token_pool_paged_block_size must be >= 1")
        if enable_token_pool_metadata is None:
            enable_token_pool_metadata = bool(
                enable_token_pool_attention or token_pool_config_requested
            )
        if enable_token_pool_attention and not enable_token_pool_metadata:
            raise ValueError("enable_token_pool_attention requires token-pool metadata")
        self.config = config
        self.bank = GemmaRoutedStateBank(config, num_slots=num_slots)
        self.arena = StateArena(config.state_spec(), num_slots=num_slots)
        self.scheduler = Scheduler(
            scheduler_config
            or SchedulerConfig(
                max_running_requests=num_slots,
                max_tokens_per_request_per_step=max(1, prefill_chunk),
            ),
            self.arena,
        )
        decode_workspace = (
            None
            if decode_workspace_bytes is None
            else PaddedDecodeWorkspace(
                width_bucket=decode_workspace_width_bucket,
                max_buffer_bytes=decode_workspace_bytes,
            )
        )
        self.runner = GemmaRoutedSpanRunner(
            model,
            self.bank,
            prefill_chunk=prefill_chunk,
            decode_workspace=decode_workspace,
            persistent_padded_decode_cuda_graph=persistent_padded_decode_cuda_graph,
            persistent_padded_decode_graph_warmup_iters=(
                persistent_padded_decode_graph_warmup_iters
            ),
            use_native_gemma_forward=use_native_gemma_forward,
            native_gemma_attention_backend=native_gemma_attention_backend,
            native_gemma_projection_backend=native_gemma_projection_backend,
            native_gemma_weight_backend=native_gemma_weight_backend,
            native_gemma_release_hf_decoder_layers=(
                native_gemma_release_hf_decoder_layers
            ),
            collect_cuda_memory_phase_metrics=collect_cuda_memory_phase_metrics,
        )
        self.stop_token_ids = stop_token_ids
        self.decode_microbatch_rows = decode_microbatch_rows
        self.decode_microbatch_bytes = decode_microbatch_bytes
        self.decode_batch_planner = decode_batch_planner
        self.decode_workspace_bytes = decode_workspace_bytes
        self.decode_workspace_width_bucket = decode_workspace_width_bucket
        self.persistent_exact_decode = bool(persistent_exact_decode)
        self.persistent_padded_decode = bool(persistent_padded_decode)
        self.persistent_padded_decode_steps = int(persistent_padded_decode_steps)
        if persistent_padded_full_attention_rows is None:
            persistent_padded_full_attention_rows = bool(
                persistent_padded_decode and enable_token_pool_attention
            )
        self.persistent_padded_full_attention_rows = bool(
            persistent_padded_full_attention_rows
        )
        self.persistent_padded_sliding_metadata_padding = bool(
            persistent_padded_sliding_metadata_padding
        )
        self.persistent_padded_decode_cuda_graph = bool(persistent_padded_decode_cuda_graph)
        self.persistent_padded_decode_graph_warmup_iters = int(
            persistent_padded_decode_graph_warmup_iters
        )
        self.record_token_pool_decode_graph_signatures = bool(
            self.persistent_padded_decode_cuda_graph
            or _token_pool_graph_signature_recording_requested()
        )
        self.use_native_gemma_forward = bool(use_native_gemma_forward)
        self.native_gemma_attention_backend = native_gemma_attention_backend
        self.native_gemma_projection_backend = native_gemma_projection_backend
        self.native_gemma_weight_backend = native_gemma_weight_backend
        self.native_gemma_release_hf_decoder_layers = bool(
            native_gemma_release_hf_decoder_layers
        )
        self.collect_cuda_memory_phase_metrics = bool(
            collect_cuda_memory_phase_metrics
        )
        self.native_gemma_released_hf_decoder_layers = int(
            getattr(self.runner.model, "released_hf_decoder_layers", 0)
        )
        self.model_forward_backend = getattr(
            self.runner.model,
            "wkvm_forward_backend",
            "hf_transformers_gemma4_forward",
        )
        self.uses_hf_transformer_forward = not bool(
            getattr(self.runner.model, "wkvm_no_hf_transformer_forward", False)
        )
        self.uses_hf_model_construction = bool(
            getattr(self.runner.model, "wkvm_uses_hf_model_construction", True)
        )
        self.native_gemma_checkpoint_loader = bool(
            getattr(self.runner.model, "wkvm_checkpoint_native_loader", False)
        )
        if enable_token_pool_attention and self.uses_hf_transformer_forward:
            raise ValueError("enable_token_pool_attention requires native Gemma forward")
        self.finished_trace_limit = finished_trace_limit
        self.metrics = GemmaEngineMetrics()
        self._break_masks: dict[str, list[bool] | None] = {}
        self._caches: dict[str, NativeGemmaRoutedCache] = {}
        self._persistent_exact_decode_groups: dict[tuple[str, ...], NativeGemmaRoutedCache] = {}
        self._persistent_padded_decode_groups: dict[tuple[str, ...], NativeGemmaRoutedCache] = {}
        self._persistent_padded_token_pool_decode_signatures: dict[
            tuple[str, ...], dict[str, Any]
        ] = {}
        self._traces: dict[str, GemmaRequestTrace] = {}
        self.finished_traces: OrderedDict[str, GemmaRequestTrace] = OrderedDict()
        self.enable_token_pool_metadata = bool(enable_token_pool_metadata)
        self.enable_token_pool_attention = bool(enable_token_pool_attention)
        self.token_pool_capacity = token_pool_capacity
        self.token_pool_paged_block_size = token_pool_paged_block_size
        self.token_pool_page_table_metadata_max_rows = max(
            0,
            int(_default_token_pool_page_table_metadata_max_rows()),
        )
        self._token_table: ReqToTokenTable | None = None
        self._token_slot_allocator: TokenSlotAllocator | TokenKVPool | None = None
        self._token_kv_pool: TokenKVPool | None = None
        self._token_pool_req_slots: dict[str, int] = {}
        self._token_pool_token_slots: dict[str, list[int]] = {}
        self._token_pool_page_tables: dict[str, dict[int, int]] = {}
        self._token_pool_page_owned_slots: dict[str, set[int]] = {}
        self._token_pool_page_table_tensor: Any | None = None
        self._token_pool_full_attention_slots: dict[str, list[int]] = {}
        self._token_pool_full_attention_rows: dict[str, _TokenPoolFullAttentionRow] = {}
        self._token_pool_full_attention_decode_metadata_workspace: dict[str, Any] = {}
        self.last_token_pool_decode_metadata: dict[str, DecodeBatchMetadata] | None = None
        self.last_token_pool_decode_metadata_by_layer_id: dict[int, DecodeBatchMetadata] | None = None
        self.last_token_pool_paged_decode_metadata: (
            dict[str, PagedDecodeBatchMetadata] | None
        ) = None
        self.last_token_pool_paged_decode_metadata_by_layer_id: (
            dict[int, PagedDecodeBatchMetadata] | None
        ) = None
        self.last_token_pool_decode_covered_layer_types: frozenset[str] = frozenset()
        if self.enable_token_pool_metadata:
            model_cfg = getattr(self.runner.model, "config", getattr(model, "config", None))
            initial_context_len = token_pool_max_context_len
            if initial_context_len is None:
                initial_context_len = int(
                    getattr(model_cfg, "max_position_embeddings", 0)
                    or getattr(model_cfg, "max_seq_len", 0)
                    or 16384
                )
            table_device = self.runner.device if self.enable_token_pool_attention else "cpu"
            self._token_table = ReqToTokenTable(
                max_requests=num_slots,
                max_context_len=max(1, int(initial_context_len)),
                device=table_device,
            )
            if self.enable_token_pool_attention:
                capacity = token_pool_capacity
                if capacity is None:
                    capacity = num_slots * max(1, int(self.config.sliding_window) + 1)
                self._token_kv_pool = self._build_token_kv_pool(
                    capacity=int(capacity),
                    defer_buffer_allocation=True,
                )
                self._token_slot_allocator = self._token_kv_pool
                self._token_pool_page_table_tensor = (
                    self._new_token_pool_page_table_tensor()
                )
            else:
                self._token_slot_allocator = TokenSlotAllocator(capacity=token_pool_capacity)
        self._record_cuda_memory_phase("engine_init")

    def add_request(self, request: Request, *, break_mask: list[bool] | None = None) -> None:
        self.scheduler.add_request(request)
        self._break_masks[request.req_id] = break_mask
        self._traces[request.req_id] = GemmaRequestTrace(
            req_id=request.req_id,
            enqueue_time=time.perf_counter(),
            prompt_tokens=request.num_prompt_tokens,
            target_output_tokens=request.max_new_tokens,
        )
        self._record_queue_state()

    def abort_request(self, req_id: str) -> None:
        self._flush_exact_decode_groups_touching({req_id})
        self._flush_padded_decode_groups_touching({req_id})
        self.scheduler.abort_request(req_id)
        trace = self._traces.pop(req_id, None)
        if trace is not None:
            trace.finish_time = time.perf_counter()
            req = self.scheduler.requests.get(req_id)
            trace.output_tokens = 0 if req is None else len(req.output_token_ids)
            trace.finish_reason = "aborted"
            self._store_finished_trace(req_id, trace)
        self._break_masks.pop(req_id, None)
        self._caches.pop(req_id, None)
        self._token_pool_release_request(req_id)
        self._record_queue_state()

    def fail_unfinished(self, error: str) -> list[Request]:
        self._persistent_exact_decode_groups.clear()
        self._persistent_padded_decode_groups.clear()
        self._persistent_padded_token_pool_decode_signatures.clear()
        self._token_pool_clear_full_attention_rows(list(self._token_pool_full_attention_rows))
        failed: list[Request] = []
        for req_id in list(self.scheduler.requests):
            req = self.scheduler.requests[req_id]
            if req.status.is_finished:
                continue
            failed_req = self.scheduler.fail_request(req_id)
            if failed_req is None:
                continue
            self._finish_trace(failed_req, error=error)
            self._break_masks.pop(req_id, None)
            self._caches.pop(req_id, None)
            self._token_pool_release_request(req_id)
            failed.append(failed_req)
        self.metrics.error_count += len(failed)
        self._record_cache_bytes()
        self._record_queue_state()
        return failed

    @property
    def has_unfinished(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def step(self) -> list[Request]:
        out = self.scheduler.schedule()
        self.metrics.steps += 1
        self._record_queue_state()
        self._record_backpressure(out)
        if out.is_empty:
            return []
        for req in out.admitted:
            self.bank.zero_slots(req.slots)
        self.metrics.admitted_requests += len(out.admitted)
        self.metrics.scheduled_tokens += out.total_tokens
        try:
            sampled = self._execute(out)
            finished = self.scheduler.update_from_output(
                out, sampled, stop_token_ids=self.stop_token_ids
            )
        except Exception as exc:
            self.metrics.error_count += 1
            self._fail_scheduled(out, str(exc).splitlines()[0])
            raise
        for req in finished:
            self._flush_exact_decode_groups_touching({req.req_id})
            self._flush_padded_decode_groups_touching({req.req_id})
            self._finish_trace(req)
            self._break_masks.pop(req.req_id, None)
            self._caches.pop(req.req_id, None)
            self._token_pool_release_request(req.req_id)
        self.metrics.finished_requests += len(finished)
        self._record_cache_bytes()
        self._record_queue_state()
        self._record_cuda_memory_phase("step_end")
        return finished

    def stats(self) -> dict[str, Any]:
        gpu_memory = self._gpu_memory_stats()
        return {
            **self.metrics.as_dict(),
            "queue_depth": len(self.scheduler.waiting),
            "runnable_rows": len(self.scheduler.running),
            "resident_state_slots": self.arena.num_slots - self.arena.num_free_slots(),
            "free_state_slots": self.arena.num_free_slots(),
            "active_cache_bytes": self._active_cache_bytes(),
            "state_bytes_per_request": self.config.state_spec().bytes_per_request,
            "decode_microbatch_rows": self.decode_microbatch_rows,
            "decode_microbatch_bytes": self.decode_microbatch_bytes,
            "decode_batch_planner": self.decode_batch_planner,
            "decode_workspace_bytes": self.decode_workspace_bytes,
            "decode_workspace_width_bucket": self.decode_workspace_width_bucket,
            "persistent_exact_decode": self.persistent_exact_decode,
            "persistent_padded_decode": self.persistent_padded_decode,
            "persistent_padded_decode_steps": self.persistent_padded_decode_steps,
            "persistent_padded_full_attention_rows": (
                self.persistent_padded_full_attention_rows
            ),
            "persistent_padded_sliding_metadata_padding": (
                self.persistent_padded_sliding_metadata_padding
            ),
            "persistent_padded_decode_cuda_graph": self.persistent_padded_decode_cuda_graph,
            "persistent_padded_decode_graph_warmup_iters": self.persistent_padded_decode_graph_warmup_iters,
            "token_pool_decode_graph_signature_recording": (
                self.record_token_pool_decode_graph_signatures
            ),
            "cuda_phase_metrics_enabled": self.collect_cuda_memory_phase_metrics,
            "use_native_gemma_forward": self.use_native_gemma_forward,
            "native_gemma_attention_backend": self.native_gemma_attention_backend,
            "native_gemma_projection_backend": self.native_gemma_projection_backend,
            "native_gemma_weight_backend": self.native_gemma_weight_backend,
            "native_gemma_release_hf_decoder_layers": (
                self.native_gemma_release_hf_decoder_layers
            ),
            "native_gemma_released_hf_decoder_layers": (
                self.native_gemma_released_hf_decoder_layers
            ),
            "model_forward_backend": self.model_forward_backend,
            "uses_hf_transformer_forward": self.uses_hf_transformer_forward,
            "uses_hf_model_construction": self.uses_hf_model_construction,
            "native_gemma_checkpoint_loader": self.native_gemma_checkpoint_loader,
            "native_config": {
                "sink_tokens": self.config.sink_tokens,
                "ring_tokens": self.config.ring_tokens,
                "routed_slots": self.config.routed_slots,
                "pending_tokens": self.config.pending_tokens,
                "sliding_window": self.config.sliding_window,
            },
            "token_pool_metadata_enabled": self.enable_token_pool_metadata,
            "token_pool_attention_enabled": self.enable_token_pool_attention,
            "token_pool": self._token_pool_stats(),
            "gpu_memory": gpu_memory,
            "state": self._state_stats(),
            "requests": {
                req_id: trace.as_dict()
                for req_id, trace in {**self.finished_traces, **self._traces}.items()
            },
        }

    def _execute(self, out: SchedulerOutput) -> dict[str, list[int]]:
        decode_reqs: list[Request] = []
        sampled: dict[str, list[int]] = {}

        for req_id, n in out.num_scheduled_tokens.items():
            req = self.scheduler.requests[req_id]
            if req.req_id not in self._caches:
                sampled.update(self._execute_prefill_chunk(req, n, initial=True))
            elif req.num_computed_tokens < req.num_prompt_tokens:
                sampled.update(self._execute_prefill_chunk(req, n, initial=False))
            elif n == 1:
                decode_reqs.append(req)
            else:
                raise NotImplementedError(
                    "GemmaNativeEngine does not support multi-token decode steps"
                )

        if decode_reqs:
            sampled.update(self._execute_decode_batch(decode_reqs))
        return sampled

    def _execute_prefill_chunk(
        self,
        req: Request,
        n: int,
        *,
        initial: bool,
    ) -> dict[str, list[int]]:
        if n < 1:
            raise ValueError("prefill chunk must contain at least one token")
        if req.num_computed_tokens >= req.num_prompt_tokens:
            raise AssertionError(f"{req.req_id}: prefill scheduled after prompt")
        if req.num_computed_tokens + n > req.num_prompt_tokens:
            raise AssertionError(f"{req.req_id}: prefill chunk crosses prompt boundary")
        trace = self._traces.get(req.req_id)
        if initial:
            if req.num_computed_tokens != 0:
                raise AssertionError(f"{req.req_id}: initial prefill from nonzero offset")
            cache = self.runner.build_cache(req.slots)
            self._caches[req.req_id] = cache
            self._record_cuda_memory_phase("prefill_cache_build")
        else:
            cache = self._caches[req.req_id]
        if trace is not None and trace.prefill_start is None:
            trace.prefill_start = time.perf_counter()
        logits = self.runner.prefill_chunk_step(
            cache,
            self._feed_tokens(req, n),
            req.slots,
            start_pos=req.num_computed_tokens,
            break_mask=self._break_masks.get(req.req_id),
        )
        self._record_cuda_memory_phase("prefill_forward")
        final_prefill = req.num_computed_tokens + n >= req.num_prompt_tokens
        self._token_pool_commit_prefill_tokens(
            req,
            n,
            cache=cache,
            final_prefill=final_prefill,
        )
        self._record_cuda_memory_phase("prefill_token_pool_commit")
        if final_prefill:
            self._token_pool_release_prefill_sliding_storage(cache)
            self._record_cuda_memory_phase("prefill_final_release")
        if trace is not None and final_prefill:
            now = time.perf_counter()
            trace.prefill_end = now
            trace.first_token_time = now
        self.metrics.prefill_calls += 1
        self._record_cache_bytes()
        self._record_cuda_memory_phase("prefill_chunk")
        if self._closes_gap(req, n):
            return {req.req_id: _sample_argmax_token_ids(logits, rows=1)}
        return {}

    def _execute_decode_batch(self, reqs: list[Request]) -> dict[str, list[int]]:
        self.metrics.decode_batches += 1
        self.metrics.decode_rows += len(reqs)
        self.metrics.max_decode_batch_rows = max(self.metrics.max_decode_batch_rows, len(reqs))
        if len({req.num_tokens for req in reqs}) > 1:
            self.metrics.distinct_history_decode_batches += 1

        batches, byte_split = self._plan_decode_model_batches(reqs)
        allowed_group_keys = {
            tuple(req.req_id for req in batch)
            for batch in batches
            if len(batch) > 1 or self._single_row_persistent_padded_enabled()
        }
        self._flush_exact_decode_groups_except(allowed_group_keys)
        self._flush_padded_decode_groups_except(allowed_group_keys)
        if len(batches) > 1:
            self.metrics.decode_microbatch_splits += 1
        if byte_split:
            self.metrics.decode_microbatch_byte_splits += 1

        sampled: dict[str, list[int]] = {}
        for batch in batches:
            sampled.update(self._execute_decode_model_batch(batch))
        self._record_cache_bytes()
        self._record_cuda_memory_phase("decode_batch")
        return sampled

    def _execute_decode_model_batch(self, reqs: list[Request]) -> dict[str, list[int]]:
        sampled: dict[str, list[int]] = {}
        self.metrics.max_decode_model_batch_bytes = max(
            self.metrics.max_decode_model_batch_bytes,
            self._estimate_decode_model_batch_bytes(reqs),
        )
        self.metrics.max_decode_model_batch_rows = max(
            self.metrics.max_decode_model_batch_rows, len(reqs)
        )
        batched = len(reqs) > 1
        use_persistent_single = (
            len(reqs) == 1 and self._single_row_persistent_padded_enabled()
        )
        if batched or use_persistent_single:
            key = tuple(req.req_id for req in reqs)
            if key in self._persistent_padded_decode_groups:
                persistent = self._try_execute_persistent_padded_decode_batch(reqs)
                if persistent is not None:
                    return persistent
            else:
                if batched:
                    persistent = self._try_execute_persistent_exact_decode_batch(reqs)
                    if persistent is not None:
                        return persistent
                persistent = self._try_execute_persistent_padded_decode_batch(reqs)
                if persistent is not None:
                    return persistent
        if batched:
            reservations = self._token_pool_prepare_decode_batch(reqs)
            token_pool_decode = self._token_pool_decode_context(reservations)
            try:
                logits = self.runner.decode_batch(
                    [self._caches[req.req_id] for req in reqs],
                    [self._feed_tokens(req, 1)[0] for req in reqs],
                    position_ids=[req.num_computed_tokens for req in reqs],
                    token_pool_decode=token_pool_decode,
                )
                self._token_pool_commit_decode_reservations(reservations)
                self.metrics.decode_model_calls += 1
                self.metrics.batched_decode_model_calls += 1
                info = getattr(self.runner, "last_decode_batch_info", {})
                self._record_decode_timing_info(info)
                if info.get("merge") == "padded_valid_mask_concat":
                    self.metrics.padded_decode_batch_rows += len(reqs)
                    self._record_padded_decode_temp_info(info)
                else:
                    self.metrics.exact_decode_batch_rows += len(reqs)
                self._record_cuda_memory_phase("decode_model_batch")
                token_ids: list[int] | None = None
                for row, req in enumerate(reqs):
                    if self._closes_gap(req, 1):
                        if token_ids is None:
                            token_ids = _sample_argmax_token_ids(
                                logits,
                                rows=len(reqs),
                            )
                        sampled[req.req_id] = [token_ids[row]]
                return sampled
            except DistinctCacheBatchError as exc:
                self._token_pool_discard_decode_reservations(reservations)
                self._record_decode_batch_fallback(exc)
            except Exception:
                self._token_pool_discard_decode_reservations(reservations)
                raise

        for req in reqs:
            cache = self._caches[req.req_id]
            last_token = self._feed_tokens(req, 1)[0]
            reservations = self._token_pool_prepare_decode_batch([req])
            token_pool_decode = self._token_pool_decode_context(reservations)
            try:
                logits = self.runner.decode_step(
                    cache,
                    [last_token],
                    position_ids=[req.num_computed_tokens],
                    token_pool_decode=token_pool_decode,
                )
                self._token_pool_commit_decode_reservations(reservations)
            except Exception:
                self._token_pool_discard_decode_reservations(reservations)
                raise
            self.metrics.decode_model_calls += 1
            self.metrics.fallback_decode_model_calls += 1
            self._record_decode_timing_info(
                getattr(self.runner, "last_decode_batch_info", {})
            )
            self._record_cuda_memory_phase("decode_model_single")
            if self._closes_gap(req, 1):
                sampled[req.req_id] = _sample_argmax_token_ids(logits, rows=1)
        return sampled

    def _single_row_persistent_padded_enabled(self) -> bool:
        if not (self.persistent_padded_decode and self.persistent_padded_decode_cuda_graph):
            return False
        can_graph = getattr(self.runner, "_can_cuda_graph_decode", None)
        if can_graph is None:
            return True
        try:
            return bool(can_graph())
        except Exception:
            return False

    def _try_execute_persistent_exact_decode_batch(
        self,
        reqs: list[Request],
    ) -> dict[str, list[int]] | None:
        if not self.persistent_exact_decode:
            return None
        if self._token_kv_pool is not None:
            return None
        begin = getattr(self.runner, "decode_batch_exact_persistent", None)
        reuse = getattr(self.runner, "decode_persistent_exact_batch", None)
        if begin is None or reuse is None:
            return None
        key = tuple(req.req_id for req in reqs)
        last_tokens = [self._feed_tokens(req, 1)[0] for req in reqs]
        position_ids = [req.num_computed_tokens for req in reqs]
        merged_cache = self._persistent_exact_decode_groups.get(key)
        reservations = self._token_pool_prepare_decode_batch(reqs)
        token_pool_decode = self._token_pool_decode_context(reservations)
        started_new = merged_cache is None
        try:
            if started_new:
                logits, merged_cache = begin(
                    [self._caches[req.req_id] for req in reqs],
                    last_tokens,
                    position_ids=position_ids,
                    token_pool_decode=token_pool_decode,
                )
                self._persistent_exact_decode_groups[key] = merged_cache
                self._release_stale_exact_group_rows(key, merged_cache)
                self.metrics.persistent_exact_decode_starts += 1
            else:
                logits = reuse(
                    merged_cache,
                    last_tokens,
                    position_ids=position_ids,
                    token_pool_decode=token_pool_decode,
                )
                self.metrics.persistent_exact_decode_reuses += 1
            self._token_pool_commit_decode_reservations(reservations)
        except DistinctCacheBatchError as exc:
            self._token_pool_discard_decode_reservations(reservations)
            self._record_decode_batch_fallback(exc)
            return None
        except Exception:
            self._token_pool_discard_decode_reservations(reservations)
            raise

        self.metrics.decode_model_calls += 1
        if len(reqs) > 1:
            self.metrics.batched_decode_model_calls += 1
        self._record_decode_timing_info(
            getattr(self.runner, "last_decode_batch_info", {})
        )
        self.metrics.exact_decode_batch_rows += len(reqs)
        self.metrics.persistent_exact_decode_rows += len(reqs)
        self._record_cuda_memory_phase(
            "persistent_exact_decode_start"
            if started_new
            else "persistent_exact_decode_reuse"
        )

        sampled: dict[str, list[int]] = {}
        should_flush = False
        token_ids: list[int] | None = None
        for row, req in enumerate(reqs):
            if not self._closes_gap(req, 1):
                continue
            if token_ids is None:
                token_ids = _sample_argmax_token_ids(logits, rows=len(reqs))
            tok = token_ids[row]
            sampled[req.req_id] = [tok]
            if tok in self.stop_token_ids or len(req.output_token_ids) + 1 >= req.max_new_tokens:
                should_flush = True
        if should_flush:
            self._flush_exact_decode_group(key)
        return sampled

    def _try_execute_persistent_padded_decode_batch(
        self,
        reqs: list[Request],
    ) -> dict[str, list[int]] | None:
        if not self.persistent_padded_decode:
            return None
        begin = getattr(self.runner, "decode_batch_padded_persistent", None)
        reuse = getattr(self.runner, "decode_persistent_padded_batch", None)
        if begin is None or reuse is None:
            return None
        key = tuple(req.req_id for req in reqs)
        last_tokens = [self._feed_tokens(req, 1)[0] for req in reqs]
        position_ids = [req.num_computed_tokens for req in reqs]
        merged_cache = self._persistent_padded_decode_groups.get(key)
        reservations: list[_TokenPoolDecodeReservation] = []
        started_new = merged_cache is None
        post_step_remaining_capacity: int | None = None
        try:
            if started_new:
                reserve_steps = self._persistent_padded_reserve_steps(reqs)
                if reserve_steps < 1:
                    return None
                reservations = self._token_pool_prepare_decode_batch(
                    reqs,
                    full_attention_kv_indices_padding_steps=max(
                        0,
                        reserve_steps - 1,
                    ),
                    sliding_attention_kv_indices_padding_steps=max(
                        0,
                        reserve_steps - 1,
                    )
                    if self.persistent_padded_sliding_metadata_padding
                    else 0,
                    persistent_full_attention_rows=(
                        self.persistent_padded_full_attention_rows
                    ),
                )
                token_pool_decode = self._token_pool_decode_context(reservations)
                logits, merged_cache = begin(
                    [self._caches[req.req_id] for req in reqs],
                    last_tokens,
                    position_ids=position_ids,
                    reserve_steps=reserve_steps,
                    token_pool_decode=token_pool_decode,
                )
                self._persistent_padded_decode_groups[key] = merged_cache
                self.metrics.persistent_padded_decode_starts += 1
                if self.record_token_pool_decode_graph_signatures:
                    self._record_persistent_padded_token_pool_decode_signature(
                        key,
                        token_pool_decode,
                        started_new=True,
                    )
                info = getattr(self.runner, "last_decode_batch_info", {})
                if info.get("merge") == "padded_valid_mask_concat":
                    self._record_padded_decode_temp_info(info)
                post_step_remaining_capacity = int(reserve_steps) - 1
            else:
                remaining_capacity = getattr(
                    merged_cache,
                    "padded_decode_remaining_capacity",
                    lambda: 0,
                )()
                if remaining_capacity < 1:
                    self._flush_padded_decode_group(key)
                    return self._try_execute_persistent_padded_decode_batch(reqs)
                reservations = self._token_pool_prepare_decode_batch(
                    reqs,
                    full_attention_kv_indices_padding_steps=max(
                        0,
                        int(remaining_capacity) - 1,
                    ),
                    sliding_attention_kv_indices_padding_steps=max(
                        0,
                        int(remaining_capacity) - 1,
                    )
                    if self.persistent_padded_sliding_metadata_padding
                    else 0,
                    persistent_full_attention_rows=(
                        self.persistent_padded_full_attention_rows
                    ),
                )
                token_pool_decode = self._token_pool_decode_context(reservations)
                logits = reuse(
                    merged_cache,
                    last_tokens,
                    position_ids=position_ids,
                    token_pool_decode=token_pool_decode,
                )
                self.metrics.persistent_padded_decode_reuses += 1
                if self.record_token_pool_decode_graph_signatures:
                    self._record_persistent_padded_token_pool_decode_signature(
                        key,
                        token_pool_decode,
                        started_new=False,
                    )
                post_step_remaining_capacity = int(remaining_capacity) - 1
            self._token_pool_commit_decode_reservations(reservations)
        except DistinctCacheBatchError as exc:
            self._token_pool_discard_decode_reservations(reservations)
            self._record_decode_batch_fallback(exc)
            if key in self._persistent_padded_decode_groups:
                self._flush_padded_decode_group(key)
            return None
        except Exception:
            self._token_pool_discard_decode_reservations(reservations)
            raise

        self.metrics.decode_model_calls += 1
        if len(reqs) > 1:
            self.metrics.batched_decode_model_calls += 1
        self._record_decode_timing_info(
            getattr(self.runner, "last_decode_batch_info", {})
        )
        self.metrics.padded_decode_batch_rows += len(reqs)
        self.metrics.persistent_padded_decode_rows += len(reqs)
        self._record_cuda_memory_phase(
            "persistent_padded_decode_start"
            if started_new
            else "persistent_padded_decode_reuse"
        )

        sampled: dict[str, list[int]] = {}
        should_flush = False
        token_ids: list[int] | None = None
        for row, req in enumerate(reqs):
            if not self._closes_gap(req, 1):
                continue
            if token_ids is None:
                token_ids = _sample_argmax_token_ids(logits, rows=len(reqs))
            tok = token_ids[row]
            sampled[req.req_id] = [tok]
            if tok in self.stop_token_ids or len(req.output_token_ids) + 1 >= req.max_new_tokens:
                should_flush = True
        remaining_capacity = (
            0
            if post_step_remaining_capacity is None
            else max(0, int(post_step_remaining_capacity))
        )
        if should_flush or remaining_capacity < 1:
            self._flush_padded_decode_group(key)
        return sampled

    def _persistent_padded_reserve_steps(self, reqs: list[Request]) -> int:
        if not reqs:
            return 0
        remaining_decode = min(
            max(req.max_new_tokens - len(req.output_token_ids), 0)
            for req in reqs
        )
        reserve = min(self.persistent_padded_decode_steps, remaining_decode)
        if reserve < 1:
            return 0
        route_margin = self._persistent_padded_route_margin(reqs)
        if route_margin is not None:
            reserve = min(reserve, route_margin)
        return max(0, reserve)

    def _persistent_padded_route_margin(self, reqs: list[Request]) -> int | None:
        margin: int | None = None
        for req in reqs:
            cache = self._caches.get(req.req_id)
            if cache is None:
                return 0
            for layer in getattr(cache, "layers", ()):
                route_chunk = getattr(layer, "route_chunk", None)
                pending = getattr(layer, "_pend_k", None)
                if route_chunk is None or pending is None:
                    continue
                layer_margin = int(route_chunk) - int(pending.shape[2]) - 1
                margin = layer_margin if margin is None else min(margin, layer_margin)
        return margin

    def _plan_decode_model_batches(
        self,
        reqs: list[Request],
    ) -> tuple[list[list[Request]], bool]:
        row_cap = self.decode_microbatch_rows or len(reqs)
        byte_cap = self.decode_microbatch_bytes
        if len(reqs) <= row_cap and (
            byte_cap is None or self._estimate_decode_model_batch_bytes(reqs) <= byte_cap
        ):
            return [reqs], False

        batches: list[list[Request]] = []
        ordered_reqs = list(reqs)
        if self.decode_batch_planner == "length_bucketed":
            ordered_reqs = sorted(
                reqs,
                key=lambda req: (
                    self._estimate_decode_request_width(req),
                    req.num_tokens,
                    req.req_id,
                ),
            )
            if [req.req_id for req in ordered_reqs] != [req.req_id for req in reqs]:
                self.metrics.decode_length_bucketed_batches += 1
        current: list[Request] = []
        split_for_bytes = False
        for req in ordered_reqs:
            candidate = [*current, req]
            too_many_rows = len(candidate) > row_cap
            too_many_bytes = (
                byte_cap is not None
                and current
                and self._estimate_decode_model_batch_bytes(candidate) > byte_cap
            )
            if too_many_rows or too_many_bytes:
                split_for_bytes = split_for_bytes or too_many_bytes
                batches.append(current)
                current = [req]
            else:
                current = candidate
        if current:
            batches.append(current)
        return batches, split_for_bytes

    def _estimate_decode_request_width(self, req: Request) -> int:
        cache = self._caches.get(req.req_id)
        if cache is None or not hasattr(cache, "layers"):
            return int(req.num_tokens)
        width = 0
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            if keys is not None:
                width = max(width, int(keys.shape[2]))
        return width

    def _estimate_decode_model_batch_bytes(self, reqs: list[Request]) -> int:
        if not reqs:
            return 0
        caches = [self._caches.get(req.req_id) for req in reqs]
        if any(cache is None for cache in caches):
            return 0
        if any(not hasattr(cache, "layers") for cache in caches):
            return 0
        layer_count = min(len(cache.layers) for cache in caches if cache is not None)
        total = 0
        for layer_idx in range(layer_count):
            lengths: list[int] = []
            key_ref = None
            value_ref = None
            for cache in caches:
                layer = cache.layers[layer_idx]  # type: ignore[union-attr]
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is None or values is None:
                    continue
                if key_ref is None:
                    key_ref = keys
                    value_ref = values
                lengths.append(int(keys.shape[2]))
            if key_ref is None or value_ref is None or not lengths:
                continue
            batch = len(reqs)
            width = max(lengths) + 1
            heads = int(key_ref.shape[1])
            dim = int(key_ref.shape[3])
            key_elem = int(key_ref.element_size())
            value_elem = int(value_ref.element_size())
            total += batch * heads * width * dim * (key_elem + value_elem)
        return total

    def _record_padded_decode_temp_info(self, info: dict[str, Any]) -> None:
        layers = info.get("layers")
        if not isinstance(layers, list):
            return
        call_temp = 0
        call_pad_slots = 0
        call_workspace_extra_pad_slots = 0
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            temp = int(layer.get("temporary_total_bytes", 0) or 0)
            mask = int(layer.get("temporary_mask_bytes", 0) or 0)
            copied = int(layer.get("copied_kv_bytes", 0) or 0)
            padded = int(layer.get("padded_kv_bytes", 0) or 0)
            source_padded = int(layer.get("source_padded_kv_bytes", 0) or 0)
            workspace_extra_padded = int(
                layer.get("workspace_extra_padded_kv_bytes", 0) or 0
            )
            reserved = int(layer.get("reserved_decode_kv_bytes", 0) or 0)
            pad_slots = int(layer.get("pad_slots_total", 0) or 0)
            workspace_extra_pad_slots = int(
                layer.get("workspace_extra_pad_slots_total", 0) or 0
            )
            workspace_allocated = int(layer.get("workspace_allocated", 0) or 0)
            workspace_reused = int(layer.get("workspace_reused", 0) or 0)
            workspace_bypassed = int(layer.get("workspace_bypassed", 0) or 0)
            self.metrics.padded_decode_temp_bytes += temp
            self.metrics.padded_decode_temp_mask_bytes += mask
            self.metrics.padded_decode_copied_kv_bytes += copied
            self.metrics.padded_decode_pad_kv_bytes += padded
            self.metrics.padded_decode_source_pad_kv_bytes += source_padded
            self.metrics.padded_decode_workspace_extra_pad_kv_bytes += workspace_extra_padded
            self.metrics.padded_decode_reserved_kv_bytes += reserved
            self.metrics.padded_decode_workspace_allocations += workspace_allocated
            self.metrics.padded_decode_workspace_reuses += workspace_reused
            self.metrics.padded_decode_workspace_bypasses += workspace_bypassed
            call_temp += temp
            call_pad_slots += pad_slots
            call_workspace_extra_pad_slots += workspace_extra_pad_slots
        self.metrics.max_padded_decode_temp_bytes = max(
            self.metrics.max_padded_decode_temp_bytes,
            call_temp,
        )
        self.metrics.max_padded_decode_pad_slots = max(
            self.metrics.max_padded_decode_pad_slots,
            call_pad_slots,
        )
        self.metrics.max_padded_decode_workspace_extra_pad_slots = max(
            self.metrics.max_padded_decode_workspace_extra_pad_slots,
            call_workspace_extra_pad_slots,
        )

    def _record_decode_timing_value(self, metric_name: str, value: float) -> None:
        if value <= 0:
            return
        setattr(self.metrics, metric_name, getattr(self.metrics, metric_name) + value)

    def _record_decode_timing_info(self, info: dict[str, Any]) -> None:
        if not isinstance(info, dict):
            return
        timing = info.get("timing")
        if not isinstance(timing, dict):
            timing = info
        mapping = (
            ("decode_timing_merge_s", ("merge_s", "merge_wall_s")),
            ("decode_timing_model_forward_s", ("model_forward_s", "model_forward_wall_s")),
            ("decode_timing_commit_s", ("commit_s", "commit_wall_s")),
            ("decode_timing_split_s", ("split_s", "split_wall_s")),
            ("decode_timing_mask_s", ("mask_s", "padded_mask_wall_s")),
            ("decode_timing_total_s", ("total_s", "decode_wall_s_total")),
            (
                "decode_timing_graph_input_copy_s",
                ("cuda_graph_input_copy_s", "cuda_graph_input_copy_wall_s"),
            ),
            (
                "decode_timing_graph_metadata_copy_s",
                ("cuda_graph_metadata_copy_s", "cuda_graph_metadata_copy_wall_s"),
            ),
            (
                "decode_timing_graph_replay_s",
                ("cuda_graph_replay_s", "cuda_graph_replay_wall_s"),
            ),
        )
        for metric_name, keys in mapping:
            value: float | None = None
            for key in keys:
                if key not in timing:
                    continue
                try:
                    value = float(timing.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    value = None
                break
            if value is not None:
                self._record_decode_timing_value(metric_name, value)
        for info_key, metric_name in (
            (
                "cuda_graph_metadata_tensor_copies",
                "token_pool_decode_graph_metadata_tensor_copies",
            ),
            (
                "cuda_graph_metadata_tensor_copy_skips",
                "token_pool_decode_graph_metadata_tensor_copy_skips",
            ),
        ):
            try:
                value = int(info.get(info_key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                setattr(self.metrics, metric_name, getattr(self.metrics, metric_name) + value)
        if int(info.get("persistent_padded_decode_cuda_graph_captured", 0) or 0):
            self.metrics.persistent_padded_decode_cuda_graph_captures += 1
        if int(info.get("cuda_graph_replay", 0) or 0):
            self.metrics.persistent_padded_decode_cuda_graph_replays += 1
        if info.get("persistent_padded_decode_cuda_graph_skip"):
            self.metrics.persistent_padded_decode_cuda_graph_skips += 1
            reason = str(info.get("persistent_padded_decode_cuda_graph_skip"))
            reasons = self.metrics.persistent_padded_decode_cuda_graph_skip_reasons
            reasons[reason] = reasons.get(reason, 0) + 1
        self._record_decode_cuda_memory_info(info)

    def _record_decode_cuda_memory_info(self, info: dict[str, Any]) -> None:
        snapshots = info.get("cuda_memory")
        if not isinstance(snapshots, dict):
            return
        for phase, raw in snapshots.items():
            if not isinstance(raw, dict):
                continue
            phase_name = str(phase)
            allocated = raw.get("allocated_bytes")
            reserved = raw.get("reserved_bytes")
            max_allocated = raw.get("max_allocated_bytes")
            max_reserved = raw.get("max_reserved_bytes")
            if allocated is not None:
                allocated = int(allocated)
                self.metrics.decode_cuda_current_allocated_by_phase[phase_name] = max(
                    self.metrics.decode_cuda_current_allocated_by_phase.get(
                        phase_name,
                        0,
                    ),
                    allocated,
                )
            if reserved is not None:
                reserved = int(reserved)
                self.metrics.decode_cuda_current_reserved_by_phase[phase_name] = max(
                    self.metrics.decode_cuda_current_reserved_by_phase.get(
                        phase_name,
                        0,
                    ),
                    reserved,
                )
            if max_allocated is not None:
                max_allocated = int(max_allocated)
                if max_allocated > self.metrics.max_decode_cuda_allocated_bytes:
                    self.metrics.max_decode_cuda_allocated_bytes = max_allocated
                    self.metrics.max_decode_cuda_allocated_phase = phase_name
                    self.metrics.decode_cuda_peak_allocated_advances_by_phase[
                        phase_name
                    ] = max(
                        self.metrics.decode_cuda_peak_allocated_advances_by_phase.get(
                            phase_name,
                            0,
                        ),
                        max_allocated,
                    )
            if max_reserved is not None:
                max_reserved = int(max_reserved)
                if max_reserved > self.metrics.max_decode_cuda_reserved_bytes:
                    self.metrics.max_decode_cuda_reserved_bytes = max_reserved
                    self.metrics.max_decode_cuda_reserved_phase = phase_name
                    self.metrics.decode_cuda_peak_reserved_advances_by_phase[
                        phase_name
                    ] = max(
                        self.metrics.decode_cuda_peak_reserved_advances_by_phase.get(
                            phase_name,
                            0,
                        ),
                        max_reserved,
                    )

    def _record_decode_batch_fallback(self, exc: Exception) -> None:
        reason = str(exc).split(":", 1)[-1].strip() or type(exc).__name__
        self.metrics.decode_batch_fallback_reasons[reason] = (
            self.metrics.decode_batch_fallback_reasons.get(reason, 0) + 1
        )

    def _fail_scheduled(self, out: SchedulerOutput, error: str) -> None:
        self._discard_exact_decode_groups_touching(set(out.num_scheduled_tokens))
        self._discard_padded_decode_groups_touching(set(out.num_scheduled_tokens))
        for req_id in list(out.num_scheduled_tokens):
            failed = self.scheduler.fail_request(req_id)
            if failed is None:
                continue
            self._finish_trace(failed, error=error)
            self._break_masks.pop(req_id, None)
            self._caches.pop(req_id, None)
            self._token_pool_release_request(req_id)
            self.metrics.finished_requests += 1
        self._record_cache_bytes()
        self._record_queue_state()

    def _flush_exact_decode_groups_except(self, allowed_keys: set[tuple[str, ...]]) -> None:
        for key in list(self._persistent_exact_decode_groups):
            if key not in allowed_keys:
                self._flush_exact_decode_group(key)

    def _flush_padded_decode_groups_except(self, allowed_keys: set[tuple[str, ...]]) -> None:
        for key in list(self._persistent_padded_decode_groups):
            if key not in allowed_keys:
                self._flush_padded_decode_group(key)

    def _flush_exact_decode_groups_touching(self, req_ids: set[str]) -> None:
        for key in list(self._persistent_exact_decode_groups):
            if any(req_id in req_ids for req_id in key):
                self._flush_exact_decode_group(key)

    def _flush_padded_decode_groups_touching(self, req_ids: set[str]) -> None:
        for key in list(self._persistent_padded_decode_groups):
            if any(req_id in req_ids for req_id in key):
                self._flush_padded_decode_group(key)

    def _discard_exact_decode_groups_touching(self, req_ids: set[str]) -> None:
        for key in list(self._persistent_exact_decode_groups):
            if any(req_id in req_ids for req_id in key):
                self._persistent_exact_decode_groups.pop(key, None)

    def _discard_padded_decode_groups_touching(self, req_ids: set[str]) -> None:
        for key in list(self._persistent_padded_decode_groups):
            if any(req_id in req_ids for req_id in key):
                self._persistent_padded_decode_groups.pop(key, None)
                self._persistent_padded_token_pool_decode_signatures.pop(key, None)
        self._token_pool_invalidate_full_attention_rows(req_ids)

    def _flush_exact_decode_group(self, key: tuple[str, ...]) -> None:
        merged_cache = self._persistent_exact_decode_groups.pop(key, None)
        if merged_cache is None:
            return
        caches = [self._caches.get(req_id) for req_id in key]
        if any(cache is None for cache in caches):
            return
        split_start = time.perf_counter()
        merged_cache.split_exact_decode_into(caches)  # type: ignore[arg-type]
        split_wall = time.perf_counter() - split_start
        self._record_decode_timing_value("decode_timing_split_s", split_wall)
        self._record_decode_timing_value("decode_timing_total_s", split_wall)
        self.metrics.persistent_exact_decode_splits += 1
        self._record_cache_bytes()
        self._record_cuda_memory_phase("persistent_exact_decode_split")

    def _flush_padded_decode_group(self, key: tuple[str, ...]) -> None:
        merged_cache = self._persistent_padded_decode_groups.pop(key, None)
        self._persistent_padded_token_pool_decode_signatures.pop(key, None)
        if merged_cache is None:
            self._token_pool_clear_full_attention_rows(key)
            return
        caches = [self._caches.get(req_id) for req_id in key]
        if any(cache is None for cache in caches):
            self._token_pool_clear_full_attention_rows(key)
            return
        commit_start = time.perf_counter()
        merged_cache.commit_padded_decode_into(caches)  # type: ignore[arg-type]
        commit_wall = time.perf_counter() - commit_start
        self._record_decode_timing_value("decode_timing_commit_s", commit_wall)
        self._record_decode_timing_value("decode_timing_total_s", commit_wall)
        self.metrics.persistent_padded_decode_splits += 1
        self._token_pool_clear_full_attention_rows(key)
        self._record_cache_bytes()
        self._record_cuda_memory_phase("persistent_padded_decode_commit")

    def _release_stale_exact_group_rows(
        self,
        key: tuple[str, ...],
        merged_cache: NativeGemmaRoutedCache,
    ) -> None:
        for req_id in key:
            cache = self._caches.get(req_id)
            if cache is None or cache is merged_cache:
                continue
            release = getattr(cache, "release_tensor_storage", None)
            if release is not None:
                release()

    def _finish_trace(self, req: Request, *, error: str | None = None) -> None:
        trace = self._traces.pop(req.req_id, None)
        if trace is None:
            return
        trace.finish_time = time.perf_counter()
        trace.output_tokens = len(req.output_token_ids)
        trace.finish_reason = (
            "error" if error is not None else req.status.name.removeprefix("FINISHED_").lower()
        )
        trace.error = error
        self._store_finished_trace(req.req_id, trace)

    def _store_finished_trace(self, req_id: str, trace: GemmaRequestTrace) -> None:
        self.finished_traces[req_id] = trace
        self.finished_traces.move_to_end(req_id)
        if self.finished_trace_limit is None:
            return
        while len(self.finished_traces) > self.finished_trace_limit:
            self.finished_traces.popitem(last=False)

    def _record_queue_state(self) -> None:
        self.metrics.max_waiting = max(self.metrics.max_waiting, len(self.scheduler.waiting))
        self.metrics.max_running = max(self.metrics.max_running, len(self.scheduler.running))
        self.metrics.max_runnable_rows = max(self.metrics.max_runnable_rows, len(self.scheduler.running))
        resident = self.arena.num_slots - self.arena.num_free_slots()
        self.metrics.max_resident_state_slots = max(self.metrics.max_resident_state_slots, resident)

    def _record_cache_bytes(self) -> None:
        active = self._active_cache_bytes()
        self.metrics.max_active_cache_bytes = max(self.metrics.max_active_cache_bytes, active)

    def _record_cuda_memory_phase(self, phase: str) -> None:
        if not getattr(self, "collect_cuda_memory_phase_metrics", False):
            return
        stats = self._gpu_memory_stats()
        allocated = stats.get("allocated_bytes")
        reserved = stats.get("reserved_bytes")
        max_allocated = stats.get("max_allocated_bytes")
        max_reserved = stats.get("max_reserved_bytes")
        if allocated is None or reserved is None:
            return
        phase = str(phase)
        allocated = int(allocated)
        reserved = int(reserved)
        self.metrics.cuda_current_allocated_by_phase[phase] = max(
            self.metrics.cuda_current_allocated_by_phase.get(phase, 0),
            allocated,
        )
        self.metrics.cuda_current_reserved_by_phase[phase] = max(
            self.metrics.cuda_current_reserved_by_phase.get(phase, 0),
            reserved,
        )
        if max_allocated is not None:
            max_allocated = int(max_allocated)
            if max_allocated > self.metrics.max_cuda_allocated_bytes:
                self.metrics.max_cuda_allocated_bytes = max_allocated
                self.metrics.max_cuda_allocated_phase = phase
                self.metrics.cuda_peak_allocated_advances_by_phase[phase] = max(
                    self.metrics.cuda_peak_allocated_advances_by_phase.get(phase, 0),
                    max_allocated,
                )
        if max_reserved is not None:
            max_reserved = int(max_reserved)
            if max_reserved > self.metrics.max_cuda_reserved_bytes:
                self.metrics.max_cuda_reserved_bytes = max_reserved
                self.metrics.max_cuda_reserved_phase = phase
                self.metrics.cuda_peak_reserved_advances_by_phase[phase] = max(
                    self.metrics.cuda_peak_reserved_advances_by_phase.get(phase, 0),
                    max_reserved,
                )

    def _active_cache_bytes(self) -> int:
        total = 0
        seen: set[int] = set()
        for cache in (
            *self._caches.values(),
            *self._persistent_exact_decode_groups.values(),
            *self._persistent_padded_decode_groups.values(),
        ):
            ident = id(cache)
            if ident in seen:
                continue
            seen.add(ident)
            state_bytes = getattr(cache, "state_bytes", None)
            if state_bytes is not None:
                total += int(state_bytes())
        return total

    def _token_pool_page_table_width_for_context(self, context_len: int) -> int:
        block_size = max(1, int(self.token_pool_paged_block_size))
        return max(1, (max(1, int(context_len)) + block_size - 1) // block_size)

    def _new_token_pool_page_table_tensor(self):
        table = self._token_table
        if table is None:
            return None
        import torch

        width = self._token_pool_page_table_width_for_context(table.max_context_len)
        return torch.full(
            (table.max_requests, width),
            -1,
            dtype=torch.int32,
            device=table.req_to_token.device,
        )

    def _token_pool_ensure_page_table_width(self, context_len: int) -> None:
        page_table = self._token_pool_page_table_tensor
        if page_table is None:
            return
        width = self._token_pool_page_table_width_for_context(context_len)
        if width <= int(page_table.shape[1]):
            return
        import torch

        grown = torch.full(
            (int(page_table.shape[0]), width),
            -1,
            dtype=page_table.dtype,
            device=page_table.device,
        )
        grown[:, : int(page_table.shape[1])].copy_(page_table)
        self._token_pool_page_table_tensor = grown

    def _token_pool_reset_page_table_row(self, req_slot: int) -> None:
        page_table = self._token_pool_page_table_tensor
        if page_table is None:
            return
        page_table[int(req_slot)].fill_(-1)

    def _build_token_kv_pool(
        self,
        *,
        capacity: int,
        defer_buffer_allocation: bool = False,
    ) -> TokenKVPool:
        dtype = self._infer_model_dtype()
        specs = self._token_kv_layer_specs(dtype=dtype)
        if not specs:
            raise ValueError("token-pool attention found no supported native Gemma KV layers")
        return TokenKVPool(
            capacity=capacity,
            layer_specs=specs,
            dtype=dtype,
            device=self.runner.device,
            defer_buffer_allocation=defer_buffer_allocation,
            validate_slot_writes=False,
        )

    def _infer_model_dtype(self):
        try:
            for param in self.runner.model.parameters():
                return param.dtype
        except Exception:
            pass
        import torch

        return torch.float32

    def _token_kv_layer_specs(self, *, dtype) -> list[TokenKVLayerSpec]:
        text_prefix = getattr(self.runner.model, "text_prefix", None)
        layers = list(getattr(text_prefix, "layers", ()) or ())
        if not layers:
            return []

        num_attention_heads = int(getattr(self.runner.model.config, "num_attention_heads", 0))
        if num_attention_heads < 1:
            return []

        supported_layer_types = {"full_attention", "sliding_attention"}
        owner_by_type: dict[str, int] = {}
        for layer in layers:
            attn = getattr(layer, "attn_meta", None)
            layer_type = getattr(attn, "layer_type", None)
            if layer_type not in supported_layer_types:
                continue
            if bool(getattr(attn, "is_kv_shared_layer", False)):
                continue
            owner_by_type[layer_type] = int(layer.layer_idx)

        specs: list[TokenKVLayerSpec] = []
        for layer in layers:
            attn = getattr(layer, "attn_meta", None)
            layer_type = getattr(attn, "layer_type", None)
            if layer_type not in supported_layer_types:
                continue
            groups = int(getattr(attn, "num_key_value_groups", 0) or 0)
            if groups < 1 or num_attention_heads % groups:
                continue
            layer_idx = int(layer.layer_idx)
            share_target = None
            if bool(getattr(attn, "is_kv_shared_layer", False)):
                share_target = owner_by_type.get(layer_type)
                if share_target is None:
                    continue
            specs.append(
                TokenKVLayerSpec(
                    layer_id=layer_idx,
                    num_kv_heads=num_attention_heads // groups,
                    head_dim=int(attn.head_dim),
                    dtype=dtype,
                    kv_share_target_layer=share_target,
                )
            )
        return specs

    def _token_pool_stats(self) -> dict[str, Any]:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return {"enabled": False}
        table_bytes = table.req_to_token.numel() * table.req_to_token.element_size()
        kv_pool = self._token_kv_pool
        stats = {
            "enabled": True,
            "attention_enabled": self.enable_token_pool_attention,
            "active_request_slots": len(self._token_pool_req_slots),
            "allocated_token_slots": allocator.allocated_count,
            "free_token_slots": allocator.free_count,
            "next_token_slot": allocator.next_slot,
            "token_slot_high_watermark": allocator.high_watermark,
            "token_slot_capacity": allocator.capacity,
            "paged_block_size": self.token_pool_paged_block_size,
            "page_table_metadata_max_rows": self.token_pool_page_table_metadata_max_rows,
            "max_context_len": table.max_context_len,
            "metadata_bytes": int(table_bytes),
            "kv_pool_bytes": 0 if kv_pool is None else kv_pool.state_bytes(),
            "kv_pool_layers": 0 if kv_pool is None else len(kv_pool.layer_specs),
        }
        if kv_pool is not None:
            stats.update(
                {
                    "kv_set_calls": int(getattr(kv_pool, "kv_set_calls", 0)),
                    "kv_set_tokens": int(getattr(kv_pool, "kv_set_tokens", 0)),
                    "kv_set_index_copy_calls": int(
                        getattr(kv_pool, "kv_set_index_copy_calls", 0)
                    ),
                    "kv_set_slice_copy_calls": int(
                        getattr(kv_pool, "kv_set_slice_copy_calls", 0)
                    ),
                    "kv_set_triton_copy_calls": int(
                        getattr(kv_pool, "kv_set_triton_copy_calls", 0)
                    ),
                    "kv_set_triton_fallback_calls": int(
                        getattr(kv_pool, "kv_set_triton_fallback_calls", 0)
                    ),
                    "kv_set_wall_s": float(getattr(kv_pool, "kv_set_wall_s", 0.0)),
                    "kv_set_index_copy_wall_s": float(
                        getattr(kv_pool, "kv_set_index_copy_wall_s", 0.0)
                    ),
                    "kv_set_slice_copy_wall_s": float(
                        getattr(kv_pool, "kv_set_slice_copy_wall_s", 0.0)
                    ),
                    "kv_set_triton_copy_wall_s": float(
                        getattr(kv_pool, "kv_set_triton_copy_wall_s", 0.0)
                    ),
                }
            )
        page_table = self._token_pool_page_table_tensor
        if page_table is not None:
            stats["page_table_tensor_shape"] = tuple(
                int(dim) for dim in page_table.shape
            )
        return stats

    def _token_pool_admit_request(self, req: Request) -> None:
        table = self._token_table
        if table is None:
            return
        if req.req_id in self._token_pool_req_slots:
            return
        req_slot = table.allocate(req.req_id)
        self._token_pool_req_slots[req.req_id] = req_slot
        self._token_pool_token_slots[req.req_id] = []
        self._token_pool_page_tables[req.req_id] = {}
        self._token_pool_page_owned_slots[req.req_id] = set()
        self._token_pool_reset_page_table_row(req_slot)

    def _token_pool_release_request(self, req_id: str) -> None:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        self._token_pool_clear_full_attention_rows([req_id])
        req_slot = self._token_pool_req_slots.get(req_id)
        if req_slot is not None:
            self._token_pool_reset_page_table_row(req_slot)
        page_slots = self._token_pool_page_owned_slots.pop(req_id, set())
        token_slots = self._token_pool_token_slots.pop(req_id, [])
        if page_slots:
            token_slots = [slot for slot in token_slots if slot not in page_slots]
            allocator.free_slots(sorted(page_slots))
        if token_slots:
            allocator.free_slots(token_slots)
        self._token_pool_page_tables.pop(req_id, None)
        if req_id in self._token_pool_req_slots:
            table.free(req_id)
            self._token_pool_req_slots.pop(req_id, None)

    def _token_pool_commit_prefill_tokens(
        self,
        req: Request,
        n: int,
        *,
        cache=None,
        final_prefill: bool = False,
    ) -> None:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        self._token_pool_admit_request(req)
        req_slot = self._token_pool_req_slots[req.req_id]
        current = table.length(req_slot)
        if current != req.num_computed_tokens:
            raise RuntimeError(
                f"{req.req_id}: token table length {current} does not match "
                f"computed tokens {req.num_computed_tokens}"
            )
        new_length = current + n
        table.ensure_context_len(new_length)
        self._token_pool_ensure_page_table_width(new_length)
        sliding_window = self._token_pool_attention_window()
        keep_start = 0 if sliding_window is None else max(new_length - sliding_window, 0)
        keep_new_start = current if sliding_window is None else max(current, keep_start)
        keep_new = n - (keep_new_start - current)
        if self._token_kv_pool is not None:
            keep_new = min(keep_new, self._token_pool_available_prefill_tail(cache, n))
        keep_new = max(0, int(keep_new))
        pad_new = n - keep_new
        token_slots = None
        page_table_snapshot = dict(self._token_pool_page_tables.get(req.req_id, {}))
        page_owned_snapshot = set(self._token_pool_page_owned_slots.get(req.req_id, set()))
        page_table_tensor_snapshot = None
        if self._token_pool_page_table_tensor is not None:
            page_table_tensor_snapshot = (
                self._token_pool_page_table_tensor[req_slot].clone()
            )
        self._token_pool_clear_prefix(
            req.req_id,
            req_slot,
            min(current, keep_start),
        )
        try:
            if keep_new:
                if self._token_kv_pool is not None:
                    token_slots, token_slot_ids = self._token_pool_alloc_page_aligned_slots(
                        req.req_id,
                        current + pad_new,
                        keep_new,
                    )
                else:
                    token_slots, token_slot_ids = allocator.alloc_slots_with_ids(keep_new)
                self._token_pool_backfill_prefill_tokens(
                    cache,
                    token_slots,
                    keep_new,
                    token_slot_ids=token_slot_ids,
                    release_covered=final_prefill,
                )
            append_values = []
            if pad_new:
                import torch

                append_values.append(
                    torch.full(
                        (pad_new,),
                        table.padding_token,
                        dtype=table.dtype,
                        device=table.req_to_token.device,
                    )
                )
            if token_slots is not None:
                append_values.append(token_slots)
            if append_values:
                import torch

                table.append_slots(req_slot, torch.cat(append_values))
        except Exception:
            if token_slots is not None:
                if self._token_kv_pool is None:
                    allocator.free_slots(token_slot_ids)
                else:
                    current_owned = self._token_pool_page_owned_slots.get(
                        req.req_id,
                        set(),
                    )
                    added = sorted(current_owned - page_owned_snapshot)
                    if added:
                        allocator.free_slots(added)
                    self._token_pool_page_tables[req.req_id] = page_table_snapshot
                    self._token_pool_page_owned_slots[req.req_id] = page_owned_snapshot
                    page_table_tensor = self._token_pool_page_table_tensor
                    if (
                        page_table_tensor is not None
                        and page_table_tensor_snapshot is not None
                    ):
                        page_table_tensor[req_slot].fill_(-1)
                        width = min(
                            int(page_table_tensor.shape[1]),
                            int(page_table_tensor_snapshot.numel()),
                        )
                        page_table_tensor[req_slot, :width].copy_(
                            page_table_tensor_snapshot[:width]
                        )
            raise
        if token_slots is not None:
            if self._token_kv_pool is None:
                self._token_pool_token_slots[req.req_id].extend(token_slot_ids)
        self.metrics.token_pool_slot_high_watermark = max(
            self.metrics.token_pool_slot_high_watermark,
            allocator.high_watermark,
        )

    def _token_pool_attention_window(self) -> int | None:
        if self._token_kv_pool is None:
            return None
        return max(1, int(self.config.sliding_window))

    def _token_pool_alloc_page_aligned_slots(
        self,
        req_id: str,
        start_position: int,
        n: int,
    ):
        allocator = self._token_slot_allocator
        if allocator is None:
            raise RuntimeError("token-pool allocator is not initialized")
        alloc_page = getattr(allocator, "alloc_page_block_with_ids", None)
        if alloc_page is None:
            return allocator.alloc_slots_with_ids(n)
        import torch

        block_size = self.token_pool_paged_block_size
        start_position = int(start_position)
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        page_table = self._token_pool_page_tables.setdefault(req_id, {})
        owned_slots = self._token_pool_page_owned_slots.setdefault(req_id, set())
        req_slot = self._token_pool_req_slots.get(req_id)
        self._token_pool_ensure_page_table_width(start_position + n)
        page_table_tensor = self._token_pool_page_table_tensor
        slots: list[int] = []
        for logical_pos in range(start_position, start_position + n):
            logical_block = logical_pos // block_size
            physical_block = page_table.get(logical_block)
            if physical_block is None:
                physical_block, block_slots = alloc_page(block_size)
                page_table[logical_block] = int(physical_block)
                owned_slots.update(int(slot) for slot in block_slots)
            if page_table_tensor is not None and req_slot is not None:
                page_table_tensor[int(req_slot), int(logical_block)] = int(
                    physical_block
                )
            slot = int(physical_block) * block_size + (logical_pos % block_size)
            if slot not in owned_slots:
                raise RuntimeError("page-aligned token slot is not owned by request")
            slots.append(slot)
        return torch.as_tensor(slots, dtype=torch.int32, device=allocator.device), slots

    def _token_pool_release_prefill_sliding_storage(self, cache) -> None:
        if self._token_kv_pool is None or cache is None:
            return
        release = getattr(cache, "release_token_pool_covered_sliding_storage", None)
        if release is None:
            return
        release({"sliding_attention"})

    def _token_pool_available_prefill_tail(self, cache, n: int) -> int:
        pool = self._token_kv_pool
        if pool is None:
            return int(n)
        if cache is None:
            raise RuntimeError("token-pool attention requires a cache for prefill backfill")
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise RuntimeError("token-pool attention requires native Gemma cache layers")
        available = int(n)
        saw_layer = False
        for layer_id in sorted(pool.layer_specs):
            if pool.target_layer(layer_id) != layer_id:
                continue
            if layer_id >= len(layers):
                continue
            layer = layers[layer_id]
            if not bool(getattr(layer, "is_sliding", False)):
                continue
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                raise RuntimeError(f"layer {layer_id} has no prefill KV to backfill")
            if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
                raise RuntimeError("token-pool prefill backfill expects one cache row")
            available = min(available, int(keys.shape[2]), int(values.shape[2]))
            saw_layer = True
        if not saw_layer:
            raise RuntimeError("token-pool attention requires at least one sliding KV layer")
        return max(0, int(available))

    def _token_pool_backfill_prefill_tokens(
        self,
        cache,
        token_slots,
        n: int,
        *,
        token_slot_ids: list[int] | None = None,
        release_covered: bool = False,
    ) -> None:
        pool = self._token_kv_pool
        if pool is None:
            return
        if cache is None:
            raise RuntimeError("token-pool attention requires a cache for prefill backfill")
        layers = getattr(cache, "layers", None)
        if layers is None:
            raise RuntimeError("token-pool attention requires native Gemma cache layers")
        for layer_id in sorted(pool.layer_specs):
            if pool.target_layer(layer_id) != layer_id:
                continue
            if layer_id >= len(layers):
                continue
            layer = layers[layer_id]
            if not bool(getattr(layer, "is_sliding", False)):
                continue
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                raise RuntimeError(f"layer {layer_id} has no prefill KV to backfill")
            if int(keys.shape[0]) != 1 or int(values.shape[0]) != 1:
                raise RuntimeError("token-pool prefill backfill expects one cache row")
            tail_len = min(int(n), int(keys.shape[2]), int(values.shape[2]))
            if tail_len != int(n):
                raise RuntimeError(
                    f"layer {layer_id} cache does not contain the requested prefill KV tail"
                )
            key_tail = keys[0, :, -tail_len:, :].permute(1, 0, 2).contiguous()
            value_tail = values[0, :, -tail_len:, :].permute(1, 0, 2).contiguous()
            write_slots = (
                token_slot_ids[-tail_len:]
                if token_slot_ids is not None
                else token_slots[-tail_len:]
            )
            pool.set_kv(layer_id, write_slots, key_tail, value_tail)
            if release_covered:
                layer.keys = None
                layer.values = None
                if hasattr(layer, "_dense_storage_released"):
                    layer._dense_storage_released = True
        self._record_cuda_memory_phase("token_pool_prefill_backfill")

    def _token_pool_prepare_decode_batch(
        self,
        reqs: list[Request],
        *,
        full_attention_kv_indices_padding_steps: int = 0,
        sliding_attention_kv_indices_padding_steps: int = 0,
        persistent_full_attention_rows: bool = False,
    ) -> list[_TokenPoolDecodeReservation]:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return []
        reservations: list[_TokenPoolDecodeReservation] = []
        req_slots: list[int] = []
        out_cache_loc: list[int] = []
        self.last_token_pool_decode_metadata_by_layer_id = None
        self.last_token_pool_paged_decode_metadata = None
        self.last_token_pool_paged_decode_metadata_by_layer_id = None
        self.last_token_pool_decode_covered_layer_types = frozenset()
        try:
            for req in reqs:
                self._token_pool_admit_request(req)
                req_slot = self._token_pool_req_slots[req.req_id]
                previous_length = table.length(req_slot)
                if previous_length != req.num_computed_tokens:
                    raise RuntimeError(
                        f"{req.req_id}: token table length {previous_length} "
                        f"does not match computed tokens {req.num_computed_tokens}"
                    )
                table.ensure_context_len(previous_length + 1)
                self._token_pool_ensure_page_table_width(previous_length + 1)
                if self._token_kv_pool is not None:
                    token_slot_tensor, token_slot_ids = (
                        self._token_pool_alloc_page_aligned_slots(
                            req.req_id,
                            previous_length,
                            1,
                        )
                    )
                else:
                    token_slot_tensor, token_slot_ids = allocator.alloc_slots_with_ids(1)
                token_slot = token_slot_ids[0]
                reservation = _TokenPoolDecodeReservation(
                    req_id=req.req_id,
                    req_slot=req_slot,
                    token_slot=token_slot,
                    token_slot_tensor=token_slot_tensor[:1],
                    previous_length=previous_length,
                    persistent_full_attention_row=bool(
                        persistent_full_attention_rows
                    ),
                )
                reservations.append(reservation)
                table.append_slots(req_slot, token_slot_tensor)
                if self._token_kv_pool is None:
                    self._token_pool_token_slots[req.req_id].append(token_slot)
                req_slots.append(req_slot)
                out_cache_loc.append(token_slot)
            if self._token_kv_pool is None:
                self.last_token_pool_decode_metadata = {
                    "full_attention": table.build_decode_metadata(
                        req_slots,
                        out_cache_loc=out_cache_loc,
                    ),
                    "sliding_attention": table.build_decode_metadata(
                        req_slots,
                        out_cache_loc=out_cache_loc,
                        sliding_window=self.config.sliding_window,
                    ),
                }
            else:
                sliding_metadata = table.build_decode_metadata(
                    req_slots,
                    out_cache_loc=out_cache_loc,
                    sliding_window=self.config.sliding_window,
                    allow_padding=True,
                    workspace_key="sliding_attention",
                )
                sliding_paged_metadata = None
                if self._should_build_sliding_paged_decode_metadata():
                    sliding_paged_metadata = self._build_sliding_paged_decode_metadata(
                        reservations,
                    )
                sliding_metadata = self._pad_sliding_decode_metadata_kv_indices(
                    sliding_metadata,
                    extra_steps=sliding_attention_kv_indices_padding_steps,
                    current_seq_lens=[
                        min(
                            max(1, int(self.config.sliding_window)),
                            reservation.previous_length + 1,
                        )
                        for reservation in reservations
                    ],
                )
                layer_metadata, full_metadata = self._token_pool_prepare_layer_decode_metadata(
                    reqs,
                    reservations,
                    sliding_metadata,
                    full_attention_kv_indices_padding_steps=(
                        full_attention_kv_indices_padding_steps
                    ),
                    persistent_full_attention_rows=(
                        persistent_full_attention_rows
                    ),
                )
                self.last_token_pool_decode_metadata = {
                    "sliding_attention": sliding_metadata,
                }
                if sliding_paged_metadata is not None:
                    self.last_token_pool_paged_decode_metadata = {
                        "sliding_attention": sliding_paged_metadata,
                    }
                if full_metadata is not None:
                    self.last_token_pool_decode_metadata["full_attention"] = full_metadata
                covered_layer_types = {"sliding_attention"}
                if full_metadata is not None:
                    covered_layer_types.add("full_attention")
                self.last_token_pool_decode_covered_layer_types = frozenset(
                    covered_layer_types
                )
                self.last_token_pool_decode_metadata_by_layer_id = (
                    layer_metadata if layer_metadata else None
                )
                if sliding_paged_metadata is not None:
                    self.last_token_pool_paged_decode_metadata_by_layer_id = {
                        int(layer_id): sliding_paged_metadata
                        for layer_id in sorted(self._token_kv_pool.layer_specs)
                        if self._token_pool_layer_type(int(layer_id))
                        == "sliding_attention"
                    }
                    if not self.last_token_pool_paged_decode_metadata_by_layer_id:
                        self.last_token_pool_paged_decode_metadata_by_layer_id = None
            self.metrics.token_pool_decode_metadata_batches += 1
            self.metrics.token_pool_decode_metadata_rows += len(reqs)
            for layer_type in self.last_token_pool_decode_covered_layer_types:
                key = str(layer_type)
                self.metrics.token_pool_decode_covered_layer_type_batches[key] = (
                    self.metrics.token_pool_decode_covered_layer_type_batches.get(
                        key, 0
                    )
                    + 1
                )
                self.metrics.token_pool_decode_covered_layer_type_rows[key] = (
                    self.metrics.token_pool_decode_covered_layer_type_rows.get(
                        key, 0
                    )
                    + len(reqs)
                )
            self.metrics.token_pool_slot_high_watermark = max(
                self.metrics.token_pool_slot_high_watermark,
                allocator.high_watermark,
            )
            return reservations
        except Exception:
            self._token_pool_discard_decode_reservations(reservations)
            raise

    def _build_sliding_paged_decode_metadata(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> PagedDecodeBatchMetadata | None:
        table = self._token_table
        pool = self._token_kv_pool
        if table is None or pool is None:
            return None
        req_slots = [reservation.req_slot for reservation in reservations]
        out_cache_loc = [reservation.token_slot for reservation in reservations]
        logical_lens = [
            int(reservation.previous_length) + 1 for reservation in reservations
        ]
        block_size = self.token_pool_paged_block_size
        sliding_window = max(1, int(self.config.sliding_window))
        # A full sliding window can start at the last token of a page, so it may
        # span one more physical page than ceil(window / block_size).
        block_table_width = (
            sliding_window + block_size - 1 + block_size - 1
        ) // block_size
        page_table_tensor = self._token_pool_page_table_tensor
        if page_table_tensor is not None:
            try:
                return table.build_paged_decode_metadata_from_page_table_tensor(
                    req_slots,
                    page_table_tensor,
                    block_size=block_size,
                    block_table_width=block_table_width,
                    seq_lens=logical_lens,
                    out_cache_loc=out_cache_loc,
                    sliding_window=sliding_window,
                    token_pool_capacity=pool.capacity,
                    workspace_key="sliding_attention_paged",
                    validate=False,
                )
            except (RuntimeError, ValueError, KeyError):
                pass
        page_tables = [
            self._token_pool_page_tables.get(reservation.req_id, {})
            for reservation in reservations
        ]
        if len(reservations) <= self.token_pool_page_table_metadata_max_rows:
            try:
                return table.build_paged_decode_metadata_from_page_tables(
                    req_slots,
                    page_tables,
                    block_size=block_size,
                    block_table_width=block_table_width,
                    seq_lens=logical_lens,
                    out_cache_loc=out_cache_loc,
                    sliding_window=sliding_window,
                    token_pool_capacity=pool.capacity,
                    workspace_key="sliding_attention_paged_from_dict",
                )
            except (RuntimeError, ValueError, KeyError):
                pass
        try:
            return table.build_paged_decode_metadata(
                req_slots,
                block_size=block_size,
                block_table_width=block_table_width,
                seq_lens=logical_lens,
                out_cache_loc=out_cache_loc,
                sliding_window=sliding_window,
                token_pool_capacity=pool.capacity,
                workspace_key="sliding_attention_paged_from_table",
            )
        except (RuntimeError, ValueError, KeyError):
            return None

    def _should_build_sliding_paged_decode_metadata(self) -> bool:
        if _token_pool_paged_metadata_requested():
            return True
        table = self._token_table
        if table is None:
            return False
        req_to_token = getattr(table, "req_to_token", None)
        if req_to_token is None:
            return False
        return not bool(getattr(req_to_token, "is_cuda", False))

    def _token_pool_decode_context(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> TokenPoolDecodeContext | None:
        if (
            not reservations
            or self.last_token_pool_decode_metadata is None
            or self._token_kv_pool is None
        ):
            return None
        return TokenPoolDecodeContext(
            metadata_by_layer_type=self.last_token_pool_decode_metadata,
            kv_pool=self._token_kv_pool,
            metadata_by_layer_id=self.last_token_pool_decode_metadata_by_layer_id,
            paged_metadata_by_layer_type=self.last_token_pool_paged_decode_metadata,
            paged_metadata_by_layer_id=(
                self.last_token_pool_paged_decode_metadata_by_layer_id
            ),
            covered_layer_types=self.last_token_pool_decode_covered_layer_types,
            layer_id_metadata_only_types=frozenset({"full_attention"}),
        )

    def _record_persistent_padded_token_pool_decode_signature(
        self,
        key: tuple[str, ...],
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        started_new: bool,
    ) -> None:
        if token_pool_decode is None:
            self._persistent_padded_token_pool_decode_signatures.pop(key, None)
            return

        signature = self._token_pool_decode_shape_signature(token_pool_decode)
        self.metrics.token_pool_decode_graph_candidate_batches += 1
        previous = self._persistent_padded_token_pool_decode_signatures.get(key)
        if started_new:
            self._persistent_padded_token_pool_decode_signatures[key] = signature
            self.metrics.token_pool_decode_graph_static_shape_starts += 1
            return

        if previous is None:
            self._persistent_padded_token_pool_decode_signatures[key] = signature
            self.metrics.token_pool_decode_graph_shape_mismatches += 1
            reasons = self.metrics.token_pool_decode_graph_shape_mismatch_reasons
            reasons["missing_start_signature"] = reasons.get("missing_start_signature", 0) + 1
            return

        if signature == previous:
            self.metrics.token_pool_decode_graph_static_shape_reuses += 1
            return

        self.metrics.token_pool_decode_graph_shape_mismatches += 1
        reasons = self.metrics.token_pool_decode_graph_shape_mismatch_reasons
        for reason in self._token_pool_decode_shape_mismatch_reasons(
            previous,
            signature,
        ):
            reasons[reason] = reasons.get(reason, 0) + 1

    @classmethod
    def _token_pool_decode_shape_signature(
        cls,
        token_pool_decode: TokenPoolDecodeContext,
    ) -> dict[str, Any]:
        return {
            "kv_pool_present": token_pool_decode.kv_pool is not None,
            "covered_layer_types": tuple(
                sorted(str(value) for value in (token_pool_decode.covered_layer_types or ()))
            ),
            "layer_id_metadata_only_types": tuple(
                sorted(str(value) for value in token_pool_decode.layer_id_metadata_only_types)
            ),
            "metadata_by_layer_type": {
                str(layer_type): cls._decode_metadata_shape_signature(metadata)
                for layer_type, metadata in sorted(
                    token_pool_decode.metadata_by_layer_type.items(),
                    key=lambda item: str(item[0]),
                )
            },
            "metadata_by_layer_id": {
                int(layer_id): cls._decode_metadata_shape_signature(metadata)
                for layer_id, metadata in sorted(
                    (token_pool_decode.metadata_by_layer_id or {}).items(),
                    key=lambda item: int(item[0]),
                )
            },
            "paged_metadata_by_layer_type": {
                str(layer_type): cls._paged_decode_metadata_shape_signature(metadata)
                for layer_type, metadata in sorted(
                    (token_pool_decode.paged_metadata_by_layer_type or {}).items(),
                    key=lambda item: str(item[0]),
                )
            },
            "paged_metadata_by_layer_id": {
                int(layer_id): cls._paged_decode_metadata_shape_signature(metadata)
                for layer_id, metadata in sorted(
                    (token_pool_decode.paged_metadata_by_layer_id or {}).items(),
                    key=lambda item: int(item[0]),
                )
            },
        }

    @classmethod
    def _decode_metadata_shape_signature(
        cls,
        metadata: DecodeBatchMetadata,
    ) -> dict[str, Any]:
        return {
            "req_pool_indices": cls._tensor_shape_signature(metadata.req_pool_indices),
            "seq_lens": cls._tensor_shape_signature(metadata.seq_lens),
            "logical_seq_lens": cls._tensor_shape_signature(metadata.logical_seq_lens),
            "out_cache_loc": cls._tensor_shape_signature(metadata.out_cache_loc),
            "kv_indptr": cls._tensor_shape_signature(metadata.kv_indptr),
            "kv_indices": cls._tensor_shape_signature(metadata.kv_indices),
            "out_cache_loc_long": cls._tensor_shape_signature(
                getattr(metadata, "out_cache_loc_long", None)
            ),
            "max_seq_len": getattr(metadata, "max_seq_len", None),
            "triton_decode_plan": cls._triton_decode_plan_signature(
                getattr(metadata, "triton_decode_plan", None)
            ),
        }

    @classmethod
    def _paged_decode_metadata_shape_signature(
        cls,
        metadata: PagedDecodeBatchMetadata,
    ) -> dict[str, Any]:
        return {
            "req_pool_indices": cls._tensor_shape_signature(metadata.req_pool_indices),
            "seq_lens": cls._tensor_shape_signature(metadata.seq_lens),
            "logical_seq_lens": cls._tensor_shape_signature(metadata.logical_seq_lens),
            "out_cache_loc": cls._tensor_shape_signature(metadata.out_cache_loc),
            "block_tables": cls._tensor_shape_signature(metadata.block_tables),
            "block_table_lens": cls._tensor_shape_signature(metadata.block_table_lens),
            "selected_start_positions": cls._tensor_shape_signature(
                metadata.selected_start_positions
            ),
            "slot_mapping": cls._tensor_shape_signature(
                getattr(metadata, "slot_mapping", None)
            ),
            "out_cache_loc_long": cls._tensor_shape_signature(
                getattr(metadata, "out_cache_loc_long", None)
            ),
            "block_size": int(metadata.block_size),
            "max_seq_len": getattr(metadata, "max_seq_len", None),
            "triton_decode_plan": cls._triton_decode_plan_signature(
                getattr(metadata, "triton_decode_plan", None)
            ),
        }

    @staticmethod
    def _triton_decode_plan_signature(plan: Any) -> dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "should_split": bool(getattr(plan, "should_split")),
            "split_size": int(getattr(plan, "split_size")),
            "min_splits": int(getattr(plan, "min_splits")),
            "max_splits": (
                None
                if getattr(plan, "max_splits", None) is None
                else int(getattr(plan, "max_splits"))
            ),
        }

    @staticmethod
    def _tensor_shape_signature(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is None:
            size = getattr(value, "size", None)
            if callable(size):
                try:
                    shape = size()
                except TypeError:
                    shape = None
        try:
            shape_tuple = tuple(int(dim) for dim in (shape or ()))
        except TypeError:
            shape_tuple = ()

        numel = None
        numel_fn = getattr(value, "numel", None)
        if callable(numel_fn):
            try:
                numel = int(numel_fn())
            except TypeError:
                numel = None
        if numel is None:
            try:
                numel = int(len(value))
            except TypeError:
                numel = None

        return {
            "shape": shape_tuple,
            "numel": numel,
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "")),
        }

    @classmethod
    def _token_pool_decode_shape_mismatch_reasons(
        cls,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> list[str]:
        reasons: list[str] = []
        for field in (
            "kv_pool_present",
            "covered_layer_types",
            "layer_id_metadata_only_types",
        ):
            if expected.get(field) != actual.get(field):
                reasons.append(field)
        reasons.extend(
            cls._metadata_shape_mismatch_reasons(
                "metadata_by_layer_type",
                expected.get("metadata_by_layer_type", {}),
                actual.get("metadata_by_layer_type", {}),
            )
        )
        reasons.extend(
            cls._metadata_shape_mismatch_reasons(
                "metadata_by_layer_id",
                expected.get("metadata_by_layer_id", {}),
                actual.get("metadata_by_layer_id", {}),
            )
        )
        reasons.extend(
            cls._metadata_shape_mismatch_reasons(
                "paged_metadata_by_layer_type",
                expected.get("paged_metadata_by_layer_type", {}),
                actual.get("paged_metadata_by_layer_type", {}),
            )
        )
        reasons.extend(
            cls._metadata_shape_mismatch_reasons(
                "paged_metadata_by_layer_id",
                expected.get("paged_metadata_by_layer_id", {}),
                actual.get("paged_metadata_by_layer_id", {}),
            )
        )
        return reasons or ["unknown"]

    @staticmethod
    def _metadata_shape_mismatch_reasons(
        prefix: str,
        expected: dict[Any, Any],
        actual: dict[Any, Any],
    ) -> list[str]:
        reasons: list[str] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            reasons.append(f"{prefix}.keys")
        for key in sorted(expected_keys & actual_keys, key=str):
            expected_metadata = expected[key]
            actual_metadata = actual[key]
            for field in (
                "req_pool_indices",
                "seq_lens",
                "logical_seq_lens",
                "out_cache_loc",
                "kv_indptr",
                "kv_indices",
                "block_tables",
                "block_table_lens",
                "selected_start_positions",
                "slot_mapping",
                "out_cache_loc_long",
                "block_size",
                "max_seq_len",
                "triton_decode_plan",
            ):
                if expected_metadata.get(field) != actual_metadata.get(field):
                    reasons.append(f"{prefix}.{key}.{field}")
        return reasons

    def _token_pool_commit_decode_reservations(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> None:
        allocator = self._token_slot_allocator
        if allocator is None or not reservations:
            return
        try:
            self._token_pool_commit_decode_to_full_attention_caches(reservations)
        finally:
            self._token_pool_clear_full_attention_rows(
                [
                    reservation.req_id
                    for reservation in reservations
                    if not reservation.persistent_full_attention_row
                ]
            )
        window = self._token_pool_attention_window()
        if window is not None:
            for reservation in reservations:
                self._token_pool_clear_prefix(
                    reservation.req_id,
                    reservation.req_slot,
                    max(reservation.previous_length + 1 - window, 0),
                )
        self.metrics.token_pool_slot_high_watermark = max(
            self.metrics.token_pool_slot_high_watermark,
            allocator.high_watermark,
        )

    def _token_pool_clear_prefix(self, req_id: str, req_slot: int, length: int) -> None:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        dropped = table.clear_before(req_slot, int(length))
        if dropped:
            self._token_pool_invalidate_full_attention_rows_containing(dropped)
            page_owned = self._token_pool_page_owned_slots.get(req_id, set())
            releasable = [slot for slot in dropped if slot not in page_owned]
            if releasable:
                allocator.free_slots(releasable)
            active = self._token_pool_token_slots.get(req_id)
            if active is not None:
                dropped_set = set(dropped)
                self._token_pool_token_slots[req_id] = [
                    slot for slot in active if slot not in dropped_set
                ]
        self._token_pool_release_expired_page_blocks(req_id, req_slot, length)

    def _token_pool_release_expired_page_blocks(
        self,
        req_id: str,
        req_slot: int,
        clear_before_len: int,
    ) -> None:
        allocator = self._token_slot_allocator
        if allocator is None or self._token_kv_pool is None:
            return
        block_size = max(1, int(self.token_pool_paged_block_size))
        clear_before_len = max(0, int(clear_before_len))
        page_table = self._token_pool_page_tables.get(req_id)
        owned_slots = self._token_pool_page_owned_slots.get(req_id)
        if not page_table or owned_slots is None:
            return
        expired_logical_blocks = [
            int(logical_block)
            for logical_block in page_table
            if (int(logical_block) + 1) * block_size <= clear_before_len
        ]
        if not expired_logical_blocks:
            return
        page_table_tensor = self._token_pool_page_table_tensor
        slots_to_free: list[int] = []
        for logical_block in sorted(expired_logical_blocks):
            physical_block = page_table.pop(logical_block, None)
            if physical_block is None:
                continue
            if (
                page_table_tensor is not None
                and 0 <= int(req_slot) < int(page_table_tensor.shape[0])
                and 0 <= logical_block < int(page_table_tensor.shape[1])
            ):
                page_table_tensor[int(req_slot), logical_block] = -1
            start_slot = int(physical_block) * block_size
            for slot in range(start_slot, start_slot + block_size):
                if slot in owned_slots:
                    owned_slots.remove(slot)
                    slots_to_free.append(slot)
        if slots_to_free:
            allocator.free_slots(slots_to_free)

    def _token_pool_prepare_layer_decode_metadata(
        self,
        reqs: list[Request],
        reservations: list[_TokenPoolDecodeReservation],
        sliding_metadata: DecodeBatchMetadata,
        *,
        full_attention_kv_indices_padding_steps: int = 0,
        persistent_full_attention_rows: bool = False,
    ) -> tuple[dict[int, DecodeBatchMetadata], DecodeBatchMetadata | None]:
        pool = self._token_kv_pool
        if pool is None:
            return {}, None
        metadata_by_layer_id: dict[int, DecodeBatchMetadata] = {}
        for layer_id in sorted(pool.layer_specs):
            if self._token_pool_layer_type(layer_id) == "sliding_attention":
                metadata_by_layer_id[int(layer_id)] = sliding_metadata

        full_metadata = self._token_pool_prepare_full_attention_decode_metadata(
            reqs,
            reservations,
            kv_indices_padding_steps=full_attention_kv_indices_padding_steps,
            persistent_rows=persistent_full_attention_rows,
        )
        if full_metadata is not None:
            for layer_id in sorted(pool.layer_specs):
                if self._token_pool_layer_type(layer_id) == "full_attention":
                    metadata_by_layer_id[int(layer_id)] = full_metadata
        return metadata_by_layer_id, full_metadata

    def _token_pool_prepare_full_attention_decode_metadata(
        self,
        reqs: list[Request],
        reservations: list[_TokenPoolDecodeReservation],
        *,
        kv_indices_padding_steps: int = 0,
        persistent_rows: bool = False,
    ) -> DecodeBatchMetadata | None:
        pool = self._token_kv_pool
        allocator = self._token_slot_allocator
        if pool is None or allocator is None:
            return None
        import torch

        owner_layer_ids = self._token_pool_full_attention_owner_layer_ids()
        if not owner_layer_ids:
            return None
        full_layer_ids = set(self._token_pool_full_attention_layer_ids())
        pool_full_layer_ids = {
            int(layer_id)
            for layer_id in pool.layer_specs
            if self._token_pool_layer_type(int(layer_id)) == "full_attention"
        }
        if not full_layer_ids or not full_layer_ids.issubset(pool_full_layer_ids):
            return None
        expected_owner_layer_ids = set(self.config.full_kv_layers)
        if not expected_owner_layer_ids.issubset(set(owner_layer_ids)):
            return None
        req_ids = [req.req_id for req in reqs]
        if not persistent_rows:
            self._token_pool_clear_full_attention_rows(req_ids)
        rows = []
        logical_lens: list[int] = []
        req_slots: list[int] = []
        out_cache_loc: list[int] = []
        try:
            for req, reservation in zip(reqs, reservations):
                cache = self._caches.get(req.req_id)
                if cache is None:
                    raise DistinctCacheBatchError(
                        f"{req.req_id}: missing cache for full-attention token-pool metadata"
                    )
                cache_layers = getattr(cache, "layers", None)
                if cache_layers is None:
                    raise DistinctCacheBatchError(
                        f"{req.req_id}: missing native cache layers"
                    )
                materialized_width: int | None = None
                routed_layers = []
                for layer_id in owner_layer_ids:
                    if layer_id >= len(cache_layers):
                        raise DistinctCacheBatchError(
                            f"{req.req_id}: missing full-attention layer {layer_id}"
                        )
                    layer = cache_layers[layer_id]
                    writer = getattr(layer, "write_materialized_readout_to_token_pool", None)
                    if writer is None:
                        raise DistinctCacheBatchError(
                            f"{req.req_id}: layer {layer_id} cannot backfill materialized KV"
                        )
                    width = int(layer.materialized_tokens())
                    if materialized_width is None:
                        materialized_width = width
                    elif width != materialized_width:
                        raise DistinctCacheBatchError(
                            f"{req.req_id}: full-attention materialized widths differ"
                        )
                    routed_layers.append((layer_id, layer, writer))
                materialized_width = int(materialized_width or 0)
                existing_row = (
                    self._token_pool_full_attention_rows.get(req.req_id)
                    if persistent_rows
                    else None
                )
                append_reserve_slots = max(1, int(kv_indices_padding_steps) + 1)
                if existing_row is not None:
                    if len(existing_row.row_slots) == materialized_width:
                        if existing_row.append_slots:
                            full_token_slot = int(existing_row.append_slots.pop(0))
                        else:
                            _, append_slot_list = allocator.alloc_slots_with_ids(
                                append_reserve_slots
                            )
                            full_token_slot = int(append_slot_list[0])
                            existing_row.owned_slots.extend(append_slot_list)
                            existing_row.append_slots.extend(append_slot_list[1:])
                        reservation.full_attention_token_slot = full_token_slot
                        existing_row.row_slots.append(full_token_slot)

                        rows.append(
                            TokenSlotRowChunks(
                                (
                                    torch.as_tensor(
                                        existing_row.row_slots,
                                        dtype=torch.int32,
                                        device=pool.device,
                                    ),
                                ),
                                trusted=True,
                            )
                        )
                        req_slots.append(reservation.req_slot)
                        out_cache_loc.append(full_token_slot)
                        first_layer = routed_layers[0][1]
                        logical_lens.append(int(first_layer.cumulative_length) + 1)
                        self.metrics.token_pool_full_attention_row_reuses += 1
                        self.metrics.token_pool_full_attention_row_appends += 1
                        continue
                    self._token_pool_invalidate_full_attention_rows([req.req_id])
                if materialized_width:
                    (
                        materialized_slots,
                        materialized_slot_list,
                    ) = allocator.alloc_slots_with_ids(materialized_width)
                    materialized_slots_long = materialized_slots.to(dtype=torch.long)
                else:
                    materialized_slots = torch.empty(
                        0,
                        dtype=torch.int32,
                        device=pool.device,
                    )
                    materialized_slot_list = []
                    materialized_slots_long = materialized_slots.to(dtype=torch.long)
                persistent_row: _TokenPoolFullAttentionRow | None = None
                if persistent_rows:
                    persistent_row = _TokenPoolFullAttentionRow(
                        row_slots=list(materialized_slot_list),
                        owned_slots=list(materialized_slot_list),
                    )
                    self._token_pool_full_attention_rows[req.req_id] = persistent_row
                    append_slots_tensor, append_slot_list = allocator.alloc_slots_with_ids(
                        append_reserve_slots
                    )
                    full_token_slot = int(append_slot_list[0])
                    reservation.full_attention_token_slot = full_token_slot
                    persistent_row.row_slots.append(full_token_slot)
                    persistent_row.owned_slots.extend(append_slot_list)
                    persistent_row.append_slots.extend(append_slot_list[1:])
                    decode_slot = append_slots_tensor[:1]
                else:
                    self._token_pool_full_attention_slots[req.req_id] = list(
                        materialized_slot_list
                    )
                    full_token_slot = reservation.token_slot
                    decode_slot = reservation.token_slot_tensor
                    decode_device = getattr(decode_slot, "device", None)
                    decode_device_matches = (
                        decode_device is not None
                        and torch.device(decode_device) == torch.device(pool.device)
                    )
                    if (
                        not hasattr(decode_slot, "numel")
                        or int(decode_slot.numel()) != 1
                        or getattr(decode_slot, "dtype", None) != torch.int32
                        or not decode_device_matches
                    ):
                        decode_slot = torch.as_tensor(
                            [full_token_slot],
                            dtype=torch.int32,
                            device=pool.device,
                        )
                if materialized_width:
                    for layer_id, _layer, writer in routed_layers:
                        writer(
                            pool,
                            materialized_slots,
                            layer_id=int(layer_id),
                            token_slots_long=materialized_slots_long,
                            token_slot_ids=materialized_slot_list,
                        )
                rows.append(
                    TokenSlotRowChunks(
                        (materialized_slots, decode_slot),
                        trusted=True,
                    )
                )
                req_slots.append(reservation.req_slot)
                out_cache_loc.append(full_token_slot)
                first_layer = routed_layers[0][1]
                logical_lens.append(int(first_layer.cumulative_length) + 1)
                if persistent_rows:
                    self.metrics.token_pool_full_attention_row_rebuilds += 1
            metadata = build_decode_metadata_from_token_slot_rows(
                rows,
                req_slots=req_slots,
                logical_seq_lens=logical_lens,
                out_cache_loc=out_cache_loc,
                device=pool.device,
                token_pool_capacity=pool.capacity,
                workspace=self._token_pool_full_attention_decode_metadata_workspace,
                kv_indices_padding_slots=int(kv_indices_padding_steps) * len(rows),
                trusted_aux_metadata=True,
            )
            return metadata
        except (DistinctCacheBatchError, RuntimeError, ValueError, KeyError):
            self._token_pool_clear_full_attention_rows(req_ids)
            return None

    @staticmethod
    def _pad_decode_metadata_kv_indices(
        metadata: DecodeBatchMetadata,
        *,
        extra_slots: int,
        max_seq_len: int | None = None,
    ) -> DecodeBatchMetadata:
        extra_slots = max(0, int(extra_slots))
        if extra_slots < 1:
            return metadata
        kv_indices = metadata.kv_indices
        current = int(kv_indices.numel())
        if current > 0:
            padding = kv_indices[-1:].expand(extra_slots)
        else:
            import torch

            padding = torch.zeros(
                extra_slots,
                dtype=kv_indices.dtype,
                device=kv_indices.device,
            )
        import torch

        return DecodeBatchMetadata(
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=metadata.seq_lens,
            logical_seq_lens=metadata.logical_seq_lens,
            out_cache_loc=metadata.out_cache_loc,
            kv_indptr=metadata.kv_indptr,
            kv_indices=torch.cat((kv_indices, padding), dim=0).contiguous(),
            out_cache_loc_long=metadata.out_cache_loc_long,
            max_seq_len=(
                getattr(metadata, "max_seq_len", None)
                if max_seq_len is None
                else int(max_seq_len)
            ),
        )

    def _pad_sliding_decode_metadata_kv_indices(
        self,
        metadata: DecodeBatchMetadata,
        *,
        extra_steps: int,
        current_seq_lens: list[int] | None = None,
    ) -> DecodeBatchMetadata:
        extra_steps = max(0, int(extra_steps))
        if extra_steps < 1:
            return metadata
        if current_seq_lens is None:
            try:
                seq_lens = [
                    int(value)
                    for value in metadata.seq_lens.detach().cpu().reshape(-1).tolist()
                ]
            except AttributeError:
                return metadata
        else:
            seq_lens = [int(value) for value in current_seq_lens]
        if not seq_lens:
            return metadata
        window = max(1, int(self.config.sliding_window))
        target_total = sum(min(window, seq_len + extra_steps) for seq_len in seq_lens)
        target_max_seq_len = max(
            min(window, seq_len + extra_steps) for seq_len in seq_lens
        )
        return self._pad_decode_metadata_kv_indices(
            metadata,
            extra_slots=max(0, int(target_total) - int(metadata.kv_indices.numel())),
            max_seq_len=target_max_seq_len,
        )

    def _token_pool_full_attention_owner_layer_ids(self) -> list[int]:
        pool = self._token_kv_pool
        if pool is None:
            return []
        return [
            int(layer_id)
            for layer_id in sorted(pool.layer_specs)
            if pool.target_layer(int(layer_id)) == int(layer_id)
            and self._token_pool_layer_type(int(layer_id)) == "full_attention"
        ]

    def _token_pool_full_attention_layer_ids(self) -> list[int]:
        text_prefix = getattr(self.runner.model, "text_prefix", None)
        layers = list(getattr(text_prefix, "layers", ()) or ())
        if layers:
            return [
                int(layer.layer_idx)
                for layer in layers
                if self._token_pool_layer_type(int(layer.layer_idx)) == "full_attention"
            ]
        return [int(layer_id) for layer_id in self.config.full_kv_layers]

    def _token_pool_layer_type(self, layer_id: int) -> str | None:
        layer_id = int(layer_id)
        model_cfg = getattr(self.runner.model, "config", None)
        layer_types = getattr(model_cfg, "layer_types", None)
        if layer_types is not None and layer_id < len(layer_types):
            return str(layer_types[layer_id])
        effective = getattr(self.config, "_effective_layer_types", ())
        if layer_id < len(effective):
            return str(effective[layer_id])
        return None

    def _token_pool_commit_decode_to_full_attention_caches(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> None:
        pool = self._token_kv_pool
        if pool is None:
            return
        metadata_by_type = self.last_token_pool_decode_metadata or {}
        if "full_attention" not in metadata_by_type:
            return
        owner_layer_ids = self._token_pool_full_attention_owner_layer_ids()
        if not owner_layer_ids:
            return
        invalidate_req_ids: set[str] = set()
        for reservation in reservations:
            cache = self._caches.get(reservation.req_id)
            if cache is None:
                continue
            cache_layers = getattr(cache, "layers", None)
            if cache_layers is None:
                continue
            decode_token_slot = (
                reservation.full_attention_token_slot
                if reservation.full_attention_token_slot is not None
                else reservation.token_slot
            )
            for layer_id in owner_layer_ids:
                if layer_id >= len(cache_layers):
                    continue
                layer = cache_layers[layer_id]
                key_rows, value_rows = pool.gather_kv(
                    layer_id,
                    [decode_token_slot],
                )
                key_states = key_rows.permute(1, 0, 2).unsqueeze(0).contiguous()
                value_states = value_rows.permute(1, 0, 2).unsqueeze(0).contiguous()
                commit_decode_token = getattr(layer, "commit_decode_token", None)
                if commit_decode_token is not None and commit_decode_token(
                    key_states,
                    value_states,
                ):
                    continue
                layer.update(key_states, value_states)
                if reservation.persistent_full_attention_row:
                    invalidate_req_ids.add(reservation.req_id)
        if invalidate_req_ids:
            self._token_pool_invalidate_full_attention_rows(invalidate_req_ids)

    def _token_pool_clear_full_attention_rows(self, req_ids) -> None:
        allocator = self._token_slot_allocator
        if allocator is None:
            return
        slots: list[int] = []
        req_id_list = [req_ids] if isinstance(req_ids, str) else list(req_ids)
        for req_id in req_id_list:
            req_key = str(req_id)
            slots.extend(self._token_pool_full_attention_slots.pop(req_key, []))
            persistent_row = self._token_pool_full_attention_rows.pop(req_key, None)
            if persistent_row is not None:
                slots.extend(persistent_row.owned_slots)
        if slots:
            allocator.free_slots(slots)

    def _token_pool_invalidate_full_attention_rows(self, req_ids) -> None:
        req_id_list = [req_ids] if isinstance(req_ids, str) else list(req_ids)
        invalidated = sum(
            1
            for req_id in req_id_list
            if str(req_id) in self._token_pool_full_attention_rows
        )
        self._token_pool_clear_full_attention_rows(req_id_list)
        if invalidated:
            self.metrics.token_pool_full_attention_row_invalidations += invalidated

    def _token_pool_invalidate_full_attention_rows_containing(self, slots) -> None:
        slot_set = {int(slot) for slot in slots}
        if not slot_set:
            return
        req_ids = [
            req_id
            for req_id, row in self._token_pool_full_attention_rows.items()
            if any(int(slot) in slot_set for slot in row.row_slots)
        ]
        if req_ids:
            self._token_pool_invalidate_full_attention_rows(req_ids)

    def _token_pool_discard_decode_reservations(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> None:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        self._token_pool_clear_full_attention_rows(
            [reservation.req_id for reservation in reservations]
        )
        for reservation in reversed(reservations):
            if reservation.req_id in self._token_pool_req_slots:
                table.truncate(reservation.req_slot, reservation.previous_length)
            token_slots = self._token_pool_token_slots.get(reservation.req_id)
            if token_slots is not None and reservation.token_slot in token_slots:
                token_slots.remove(reservation.token_slot)
            page_owned = self._token_pool_page_owned_slots.get(reservation.req_id, set())
            if reservation.token_slot not in page_owned:
                allocator.free_slots([reservation.token_slot])

    def _state_stats(self) -> dict[str, int]:
        resident = self.arena.num_slots - self.arena.num_free_slots()
        active_slot_ids = {
            slots["gemma_routed_span"]
            for req in self.scheduler.running
            for slots in (req.slots,)
            if "gemma_routed_span" in slots
        }
        active_states = [self.bank.slots[i] for i in active_slot_ids]
        return {
            "resident_slots": resident,
            "ring_tokens": sum(state.ring_tokens for state in active_states),
            "span_bank_tokens": sum(state.span_bank_tokens for state in active_states),
            "pending_span_tokens": sum(state.pending_tokens for state in active_states),
        }

    @staticmethod
    def _gpu_memory_stats() -> dict[str, int | None]:
        try:
            import torch
        except Exception:
            return {
                "allocated_bytes": None,
                "reserved_bytes": None,
                "max_allocated_bytes": None,
                "max_reserved_bytes": None,
            }
        if not torch.cuda.is_available():
            return {
                "allocated_bytes": None,
                "reserved_bytes": None,
                "max_allocated_bytes": None,
                "max_reserved_bytes": None,
            }
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def _record_backpressure(self, out: SchedulerOutput) -> None:
        if not self.scheduler.waiting:
            return
        reasons: list[str] = []
        if not self.arena.can_admit():
            reasons.append("no_free_slots")
        if len(self.scheduler.running) >= self.scheduler.config.max_running_requests:
            reasons.append("max_running_requests")
        if out.total_tokens >= self.scheduler.config.max_tokens_per_step:
            reasons.append("token_budget")
        if not reasons:
            reasons.append("waiting_queue")
        self.metrics.backpressure_events += 1
        for reason in reasons:
            self.metrics.backpressure_reasons[reason] = (
                self.metrics.backpressure_reasons.get(reason, 0) + 1
            )

    def _feed_tokens(self, req: Request, n: int) -> list[int]:
        start = req.num_computed_tokens
        if start < req.num_prompt_tokens:
            tokens = (req.prompt_token_ids + req.output_token_ids)[start : start + n]
        else:
            tokens = req.output_token_ids[start - req.num_prompt_tokens :][:n]
        if len(tokens) != n:
            raise AssertionError(f"{req.req_id}: scheduled past known tokens")
        return tokens

    @staticmethod
    def _closes_gap(req: Request, n: int) -> bool:
        return req.num_computed_tokens + n == req.num_tokens
