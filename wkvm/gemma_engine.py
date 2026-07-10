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
    TokenPoolBlockTables,
    TokenPoolDecodeBackendState,
    TokenPoolDecodeBatchState,
    TokenPoolDecodeContext,
    TokenPoolDecodeReservation,
    TokenPoolDecodeGraphSignatureUpdate,
    TokenPoolDecodeGraphSignatureTracker,
    TokenPoolFullAttentionRow,
    TokenSlotAllocator,
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


def _token_pool_full_attention_paged_metadata_requested() -> bool:
    return _env_flag("WKVM_TOKEN_POOL_BUILD_PAGED_METADATA")


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
    token_pool_decode_graph_metadata_alias_fastpath_metadata_skips: int = 0
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
            "token_pool_decode_graph_metadata_alias_fastpath_metadata_skips": (
                self.token_pool_decode_graph_metadata_alias_fastpath_metadata_skips
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


_TokenPoolDecodeReservation = TokenPoolDecodeReservation


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
        self._token_pool_decode_graph_signature_fallback = (
            TokenPoolDecodeGraphSignatureTracker()
        )
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
        self._token_pool_block_tables: TokenPoolBlockTables | None = None
        self._token_pool_decode_backend: TokenPoolDecodeBackendState | None = None
        self._token_pool_full_attention_slots: dict[str, list[int]] = {}
        self._token_pool_full_attention_rows: dict[str, TokenPoolFullAttentionRow] = {}
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
                self._token_pool_block_tables = self._new_token_pool_block_tables()
            else:
                self._token_slot_allocator = TokenSlotAllocator(capacity=token_pool_capacity)
            self._token_pool_decode_backend = self._new_token_pool_decode_backend()
            backend = self._token_pool_decode_backend
            row_slots = None if backend is None else backend.full_attention_transient_slots
            row_records = None if backend is None else backend.full_attention_row_records
            request_slots = None if backend is None else backend.request_slots
            request_token_slots = (
                None if backend is None else backend.request_token_slots
            )
            page_tables = None if backend is None else backend.request_page_tables
            page_owned_slots = (
                None if backend is None else backend.request_page_owned_slots
            )
            if request_slots is not None:
                self._token_pool_req_slots = request_slots
            if request_token_slots is not None:
                self._token_pool_token_slots = request_token_slots
            if page_tables is not None:
                self._token_pool_page_tables = page_tables
            if page_owned_slots is not None:
                self._token_pool_page_owned_slots = page_owned_slots
            if row_slots is not None:
                self._token_pool_full_attention_slots = row_slots
            if row_records is not None:
                self._token_pool_full_attention_rows = row_records
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
        self._token_pool_clear_graph_decode_signatures()
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
        token_pool_decode: TokenPoolDecodeContext | None = None
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
            if (
                self.record_token_pool_decode_graph_signatures
                and token_pool_decode is not None
                and "token-pool cuda graph metadata incompatible" in str(exc)
            ):
                self._record_persistent_padded_token_pool_decode_signature(
                    key,
                    token_pool_decode,
                    started_new=started_new,
                )
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
            (
                "cuda_graph_metadata_alias_fastpath_metadata_skips",
                "token_pool_decode_graph_metadata_alias_fastpath_metadata_skips",
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
                self._token_pool_discard_graph_decode_signature(key)
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
        self._token_pool_discard_graph_decode_signature(key)
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

    @property
    def _token_pool_page_table_tensor(self):
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.page_table_tensor
        block_tables = self._token_pool_block_tables
        return None if block_tables is None else block_tables.tensor

    def _new_token_pool_block_tables(self) -> TokenPoolBlockTables | None:
        table = self._token_table
        if table is None:
            return None
        return TokenPoolBlockTables(
            max_requests=table.max_requests,
            max_context_len=table.max_context_len,
            block_size=self.token_pool_paged_block_size,
            device=table.req_to_token.device,
        )

    def _new_token_pool_decode_backend(self) -> TokenPoolDecodeBackendState | None:
        table = self._token_table
        if table is None:
            return None
        return TokenPoolDecodeBackendState(
            table=table,
            allocator=self._token_slot_allocator,
            kv_pool=self._token_kv_pool,
            block_tables=self._token_pool_block_tables,
            block_size=self.token_pool_paged_block_size,
            page_table_metadata_max_rows=(
                self.token_pool_page_table_metadata_max_rows
            ),
            token_pool_capacity=(
                None if self._token_kv_pool is None else self._token_kv_pool.capacity
            ),
        )

    def _token_pool_current_decode_batch_state(
        self,
    ) -> TokenPoolDecodeBatchState | None:
        backend = self._token_pool_decode_backend
        return None if backend is None else backend.current_decode_batch_state

    @property
    def last_token_pool_decode_metadata(
        self,
    ) -> dict[str, DecodeBatchMetadata] | None:
        state = self._token_pool_current_decode_batch_state()
        if state is None:
            return None
        return state.metadata_by_layer_type

    @property
    def last_token_pool_decode_metadata_by_layer_id(
        self,
    ) -> dict[int, DecodeBatchMetadata] | None:
        state = self._token_pool_current_decode_batch_state()
        if state is None:
            return None
        return state.metadata_by_layer_id

    @property
    def last_token_pool_paged_decode_metadata(
        self,
    ) -> dict[str, PagedDecodeBatchMetadata] | None:
        state = self._token_pool_current_decode_batch_state()
        if state is None:
            return None
        return state.paged_metadata_by_layer_type

    @property
    def last_token_pool_paged_decode_metadata_by_layer_id(
        self,
    ) -> dict[int, PagedDecodeBatchMetadata] | None:
        state = self._token_pool_current_decode_batch_state()
        if state is None:
            return None
        return state.paged_metadata_by_layer_id

    @property
    def last_token_pool_decode_covered_layer_types(self) -> frozenset[str]:
        backend = self._token_pool_decode_backend
        if backend is None:
            return frozenset()
        return backend.current_covered_layer_types

    def _token_pool_clear_graph_decode_signatures(self) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.clear_graph_decode_signatures()
            return
        self._token_pool_decode_graph_signature_fallback.clear()

    def _token_pool_discard_graph_decode_signature(
        self,
        key: tuple[str, ...],
    ) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.discard_graph_decode_signature(key)
            return
        self._token_pool_decode_graph_signature_fallback.discard(key)

    def _token_pool_record_graph_decode_signature(
        self,
        key: tuple[str, ...],
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        started_new: bool,
    ) -> TokenPoolDecodeGraphSignatureUpdate:
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.record_graph_decode_signature(
                key,
                token_pool_decode,
                started_new=started_new,
            )
        return self._token_pool_decode_graph_signature_fallback.record(
            key,
            token_pool_decode,
            started_new=started_new,
        )

    @property
    def _persistent_padded_token_pool_decode_signatures(
        self,
    ) -> dict[tuple[str, ...], dict[str, Any]]:
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.graph_decode_signatures
        return self._token_pool_decode_graph_signature_fallback.signatures

    def _token_pool_ensure_page_table_width(self, context_len: int) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.ensure_page_table_width(context_len)
            return
        block_tables = self._token_pool_block_tables
        if block_tables is None:
            return
        block_tables.ensure_context_len(context_len)

    def _token_pool_reset_page_table_row(self, req_slot: int) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.reset_page_table_row(req_slot)
            return
        block_tables = self._token_pool_block_tables
        if block_tables is None:
            return
        block_tables.reset_row(req_slot)

    def _token_pool_snapshot_page_table_row(self, req_slot: int):
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.snapshot_page_table_row(req_slot)
        block_tables = self._token_pool_block_tables
        if block_tables is None:
            return None
        return block_tables.snapshot_row(req_slot)

    def _token_pool_restore_page_table_row(self, req_slot: int, snapshot) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.restore_page_table_row(req_slot, snapshot)
            return
        block_tables = self._token_pool_block_tables
        if block_tables is None:
            return
        block_tables.restore_row(req_slot, snapshot)

    def _token_pool_set_page_table_block(
        self,
        req_slot: int,
        logical_block: int,
        physical_block: int,
    ) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.set_page_table_block(req_slot, logical_block, physical_block)
            return
        block_tables = self._token_pool_block_tables
        if block_tables is None:
            return
        block_tables.set_block(req_slot, logical_block, physical_block)

    def _token_pool_clear_page_table_block(
        self,
        req_slot: int,
        logical_block: int,
    ) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.clear_page_table_block(req_slot, logical_block)
            return
        block_tables = self._token_pool_block_tables
        if block_tables is None:
            return
        block_tables.clear_block(req_slot, logical_block)

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
                share_target = getattr(attn, "kv_shared_layer_index", None)
                if share_target is None:
                    share_target = owner_by_type.get(layer_type)
                if share_target is None:
                    continue
                share_target = int(share_target)
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
        backend = self._token_pool_decode_backend
        if backend is None:
            return {"enabled": False}
        return backend.stats(
            attention_enabled=self.enable_token_pool_attention,
            paged_block_size=self.token_pool_paged_block_size,
        )

    def _token_pool_admit_request(self, req: Request) -> None:
        table = self._token_table
        if table is None:
            return
        backend = self._token_pool_decode_backend
        if backend is not None:
            if backend.has_request(req.req_id):
                return
            backend.admit_request(req.req_id)
            return
        if req.req_id in self._token_pool_req_slots:
            return
        req_slot = table.allocate(req.req_id)
        self._token_pool_req_slots[req.req_id] = req_slot
        self._token_pool_token_slots[req.req_id] = []
        self._token_pool_page_tables[req.req_id] = {}
        self._token_pool_page_owned_slots[req.req_id] = set()
        self._token_pool_reset_page_table_row(req_slot)

    def _token_pool_request_slot(self, req_id: str) -> int:
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.request_slot_for(req_id)
        return self._token_pool_req_slots[req_id]

    def _token_pool_request_length(self, req_id_or_slot: str | int) -> int:
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.request_length(req_id_or_slot)
        table = self._token_table
        if table is None:
            raise RuntimeError("token-pool table is not initialized")
        return table.length(req_id_or_slot)

    def _token_pool_ensure_context_len(self, context_len: int) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.ensure_context_len(context_len)
            return
        table = self._token_table
        if table is not None:
            table.ensure_context_len(context_len)
        self._token_pool_ensure_page_table_width(context_len)

    def _token_pool_append_table_slots(self, req_id_or_slot: str | int, slots) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.append_table_slots(req_id_or_slot, slots)
            return
        table = self._token_table
        if table is None:
            raise RuntimeError("token-pool table is not initialized")
        table.append_slots(req_id_or_slot, slots)

    def _token_pool_truncate_table_row(
        self,
        req_id_or_slot: str | int,
        length: int,
    ) -> None:
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.truncate_table_row(req_id_or_slot, length)
            return
        table = self._token_table
        if table is None:
            return
        table.truncate(req_id_or_slot, length)

    def _token_pool_clear_table_before(
        self,
        req_id_or_slot: str | int,
        length: int,
    ) -> list[int]:
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.clear_table_before(req_id_or_slot, length)
        table = self._token_table
        if table is None:
            return []
        return table.clear_before(req_id_or_slot, length)

    def _token_pool_release_request(self, req_id: str) -> None:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.release_request(req_id)
        else:
            self._token_pool_clear_full_attention_rows([req_id])
            req_slot = self._token_pool_req_slots.get(req_id)
            if req_slot is not None:
                self._token_pool_reset_page_table_row(req_slot)
            page_slots = self._token_pool_page_owned_slots.pop(req_id, set())
            self._token_pool_page_tables.pop(req_id, None)
            token_slots = self._token_pool_token_slots.pop(req_id, [])
            if page_slots:
                token_slots = [slot for slot in token_slots if slot not in page_slots]
            if token_slots:
                allocator.free_slots(token_slots)
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
        req_slot = self._token_pool_request_slot(req.req_id)
        current = self._token_pool_request_length(req_slot)
        if current != req.num_computed_tokens:
            raise RuntimeError(
                f"{req.req_id}: token table length {current} does not match "
                f"computed tokens {req.num_computed_tokens}"
            )
        new_length = current + n
        self._token_pool_ensure_context_len(new_length)
        sliding_window = self._token_pool_attention_window()
        keep_start = 0 if sliding_window is None else max(new_length - sliding_window, 0)
        keep_new_start = current if sliding_window is None else max(current, keep_start)
        keep_new = n - (keep_new_start - current)
        if self._token_kv_pool is not None:
            keep_new = min(keep_new, self._token_pool_available_prefill_tail(cache, n))
        keep_new = max(0, int(keep_new))
        pad_new = n - keep_new
        token_slots = None
        page_state_snapshot = None
        backend = self._token_pool_decode_backend
        if self._token_kv_pool is not None and backend is not None:
            page_state_snapshot = backend.snapshot_request_page_state(
                req.req_id,
                req_slot,
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

                self._token_pool_append_table_slots(req_slot, torch.cat(append_values))
        except Exception:
            if token_slots is not None:
                if self._token_kv_pool is None:
                    allocator.free_slots(token_slot_ids)
                else:
                    backend = self._token_pool_decode_backend
                    if backend is not None:
                        backend.restore_request_page_state(page_state_snapshot)
            raise
        if token_slots is not None:
            if self._token_kv_pool is None:
                backend = self._token_pool_decode_backend
                if backend is not None:
                    backend.append_request_token_slots(req.req_id, token_slot_ids)
                else:
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
        backend = self._token_pool_decode_backend
        if backend is not None:
            return backend.allocate_page_aligned_slots(
                req_id,
                start_position,
                n,
                req_slot=(
                    backend.request_slot_for(req_id)
                    if backend.has_request(req_id)
                    else None
                ),
            )
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
        slots: list[int] = []
        for logical_pos in range(start_position, start_position + n):
            logical_block = logical_pos // block_size
            physical_block = page_table.get(logical_block)
            if physical_block is None:
                physical_block, block_slots = alloc_page(block_size)
                page_table[logical_block] = int(physical_block)
                owned_slots.update(int(slot) for slot in block_slots)
            if req_slot is not None:
                self._token_pool_set_page_table_block(
                    req_slot,
                    logical_block,
                    physical_block,
                )
            slot = int(physical_block) * block_size + (logical_pos % block_size)
            if slot not in owned_slots:
                raise RuntimeError("page-aligned token slot is not owned by request")
            slots.append(slot)
        return torch.as_tensor(slots, dtype=torch.int32, device=allocator.device), slots

    def _token_pool_alloc_page_aligned_full_attention_row_slots(
        self,
        start_position: int,
        min_slots: int,
    ):
        backend = self._token_pool_decode_backend
        if backend is None:
            raise RuntimeError("token-pool full-attention row manager is not initialized")
        return backend.allocate_page_aligned_full_attention_row_slots(
            start_position,
            min_slots,
        )

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
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.clear_decode_batch_state()
        if table is None or allocator is None:
            return []
        reservations: list[_TokenPoolDecodeReservation] = []
        req_slots: list[int] = []
        out_cache_loc: list[int] = []
        try:
            for req in reqs:
                self._token_pool_admit_request(req)
                req_slot = self._token_pool_request_slot(req.req_id)
                previous_length = self._token_pool_request_length(req_slot)
                if previous_length != req.num_computed_tokens:
                    raise RuntimeError(
                        f"{req.req_id}: token table length {previous_length} "
                        f"does not match computed tokens {req.num_computed_tokens}"
                    )
                self._token_pool_ensure_context_len(previous_length + 1)
                page_state_snapshot = None
                if self._token_kv_pool is not None:
                    if (
                        getattr(allocator, "alloc_page_block_with_ids", None)
                        is not None
                    ):
                        backend = self._token_pool_decode_backend
                        if backend is not None:
                            page_state_snapshot = backend.snapshot_request_page_state(
                                req.req_id,
                                req_slot,
                            )
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
                    page_state_snapshot=page_state_snapshot,
                )
                reservations.append(reservation)
                self._token_pool_append_table_slots(req_slot, token_slot_tensor)
                if self._token_kv_pool is None:
                    backend = self._token_pool_decode_backend
                    if backend is not None:
                        backend.append_request_token_slot(req.req_id, token_slot)
                    else:
                        self._token_pool_token_slots[req.req_id].append(token_slot)
                req_slots.append(req_slot)
                out_cache_loc.append(token_slot)
            if self._token_kv_pool is None:
                if backend is None:
                    raise RuntimeError("token-pool decode backend is not initialized")
                metadata_by_type = backend.build_decode_metadata_by_layer_type(
                    req_slots=req_slots,
                    out_cache_loc=out_cache_loc,
                    sliding_window=self.config.sliding_window,
                )
                backend.set_decode_batch_state(
                    metadata_by_layer_type=metadata_by_type,
                    covered_layer_types=frozenset(),
                )
            else:
                if backend is None:
                    raise RuntimeError("token-pool decode backend is not initialized")
                logical_lens = [
                    int(reservation.previous_length) + 1
                    for reservation in reservations
                ]
                sliding_metadata, sliding_paged_metadata = (
                    backend.build_sliding_decode_metadata(
                        req_slots=req_slots,
                        logical_seq_lens=logical_lens,
                        out_cache_loc=out_cache_loc,
                        sliding_window=self.config.sliding_window,
                        page_tables=backend.page_tables_for_requests(
                            reservation.req_id for reservation in reservations
                        ),
                        kv_indices_padding_steps=(
                            sliding_attention_kv_indices_padding_steps
                        ),
                    )
                )
                (
                    full_metadata,
                    full_paged_metadata,
                ) = self._token_pool_prepare_layer_decode_metadata(
                    reqs,
                    reservations,
                    full_attention_kv_indices_padding_steps=(
                        full_attention_kv_indices_padding_steps
                    ),
                    persistent_full_attention_rows=(
                        persistent_full_attention_rows
                    ),
                )
                metadata_by_type = {
                    "sliding_attention": sliding_metadata,
                }
                paged_metadata_by_type = None
                if sliding_paged_metadata is not None:
                    paged_metadata_by_type = {
                        "sliding_attention": sliding_paged_metadata,
                    }
                if full_metadata is not None:
                    metadata_by_type["full_attention"] = full_metadata
                if full_paged_metadata is not None:
                    if paged_metadata_by_type is None:
                        paged_metadata_by_type = {}
                    paged_metadata_by_type["full_attention"] = full_paged_metadata
                backend.set_decode_batch_state_by_layer_type(
                    metadata_by_layer_type=metadata_by_type,
                    paged_metadata_by_layer_type=paged_metadata_by_type,
                    layer_type_by_layer_id=self._token_pool_layer_type_by_layer_id(),
                )
            self.metrics.token_pool_decode_metadata_batches += 1
            self.metrics.token_pool_decode_metadata_rows += len(reqs)
            prepared_batch = (
                backend.prepared_decode_batch(reservations)
                if backend is not None
                else None
            )
            covered_layer_types = (
                prepared_batch.covered_layer_types
                if prepared_batch is not None
                else frozenset()
            )
            for layer_type in covered_layer_types:
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
            if backend is not None:
                backend.clear_decode_batch_state()
            raise

    def _token_pool_decode_context(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> TokenPoolDecodeContext | None:
        if not reservations or self._token_kv_pool is None:
            return None
        backend = self._token_pool_decode_backend
        if backend is None:
            raise RuntimeError("token-pool decode backend is not initialized")
        return backend.build_decode_context_for_batch(
            reservations,
            layer_id_metadata_only_types=frozenset({"full_attention"}),
        )

    def _record_persistent_padded_token_pool_decode_signature(
        self,
        key: tuple[str, ...],
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        started_new: bool,
    ) -> None:
        update = self._token_pool_record_graph_decode_signature(
            key,
            token_pool_decode,
            started_new=started_new,
        )
        self.metrics.token_pool_decode_graph_candidate_batches += (
            update.candidate_batches
        )
        self.metrics.token_pool_decode_graph_static_shape_starts += (
            update.static_shape_starts
        )
        self.metrics.token_pool_decode_graph_static_shape_reuses += (
            update.static_shape_reuses
        )
        self.metrics.token_pool_decode_graph_shape_mismatches += (
            update.shape_mismatches
        )
        reasons = self.metrics.token_pool_decode_graph_shape_mismatch_reasons
        for reason, count in update.shape_mismatch_reasons.items():
            reasons[reason] = reasons.get(reason, 0) + int(count)

    @classmethod
    def _token_pool_decode_shape_signature(
        cls,
        token_pool_decode: TokenPoolDecodeContext,
    ) -> dict[str, Any]:
        return TokenPoolDecodeBackendState.graph_decode_shape_signature(
            token_pool_decode
        )

    @classmethod
    def _decode_metadata_shape_signature(
        cls,
        metadata: DecodeBatchMetadata,
    ) -> dict[str, Any]:
        return TokenPoolDecodeBackendState.graph_decode_metadata_shape_signature(
            metadata
        )

    @classmethod
    def _paged_decode_metadata_shape_signature(
        cls,
        metadata: PagedDecodeBatchMetadata,
    ) -> dict[str, Any]:
        return TokenPoolDecodeBackendState.graph_paged_decode_metadata_shape_signature(
            metadata
        )

    @staticmethod
    def _triton_decode_plan_signature(plan: Any) -> dict[str, Any] | None:
        return TokenPoolDecodeBackendState.graph_triton_decode_plan_signature(plan)

    @staticmethod
    def _tensor_shape_signature(value: Any) -> dict[str, Any] | None:
        return TokenPoolDecodeBackendState.graph_tensor_shape_signature(value)

    @classmethod
    def _token_pool_decode_shape_mismatch_reasons(
        cls,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> list[str]:
        return TokenPoolDecodeBackendState.graph_decode_shape_mismatch_reasons(
            expected,
            actual,
        )

    @staticmethod
    def _metadata_shape_mismatch_reasons(
        prefix: str,
        expected: dict[Any, Any],
        actual: dict[Any, Any],
    ) -> list[str]:
        return TokenPoolDecodeBackendState.graph_metadata_shape_mismatch_reasons(
            prefix,
            expected,
            actual,
        )

    def _token_pool_commit_decode_reservations(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> None:
        allocator = self._token_slot_allocator
        if allocator is None or not reservations:
            return
        backend = self._token_pool_decode_backend
        if backend is not None:
            result = backend.commit_decode_batch(
                reservations,
                caches_by_req_id=self._caches,
                owner_layer_ids=self._token_pool_full_attention_owner_layer_ids(),
                attention_window=self._token_pool_attention_window(),
            )
            if result.invalidated_full_attention_rows:
                self.metrics.token_pool_full_attention_row_invalidations += (
                    result.invalidated_full_attention_rows
                )
            self.metrics.token_pool_slot_high_watermark = max(
                self.metrics.token_pool_slot_high_watermark,
                allocator.high_watermark,
            )
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
        backend = self._token_pool_decode_backend
        if backend is not None:
            result = backend.clear_request_prefix(req_id, req_slot, int(length))
            if result.invalidated_full_attention_rows:
                self.metrics.token_pool_full_attention_row_invalidations += (
                    result.invalidated_full_attention_rows
                )
            return
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        dropped = self._token_pool_clear_table_before(req_slot, int(length))
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
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.release_expired_page_blocks(req_id, req_slot, clear_before_len)
            return
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
        slots_to_free: list[int] = []
        for logical_block in sorted(expired_logical_blocks):
            physical_block = page_table.pop(logical_block, None)
            if physical_block is None:
                continue
            self._token_pool_clear_page_table_block(req_slot, logical_block)
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
        *,
        full_attention_kv_indices_padding_steps: int = 0,
        persistent_full_attention_rows: bool = False,
    ) -> tuple[DecodeBatchMetadata | None, PagedDecodeBatchMetadata | None]:
        pool = self._token_kv_pool
        if pool is None:
            return None, None

        (
            full_metadata,
            full_paged_metadata,
        ) = self._token_pool_prepare_full_attention_decode_metadata(
            reqs,
            reservations,
            kv_indices_padding_steps=full_attention_kv_indices_padding_steps,
            persistent_rows=persistent_full_attention_rows,
        )
        return full_metadata, full_paged_metadata

    def _token_pool_prepare_full_attention_decode_metadata(
        self,
        reqs: list[Request],
        reservations: list[_TokenPoolDecodeReservation],
        *,
        kv_indices_padding_steps: int = 0,
        persistent_rows: bool = False,
    ) -> tuple[DecodeBatchMetadata | None, PagedDecodeBatchMetadata | None]:
        pool = self._token_kv_pool
        backend = self._token_pool_decode_backend
        if pool is None or backend is None:
            return None, None
        if not backend.has_full_attention_rows():
            return None, None
        owner_layer_ids = self._token_pool_full_attention_owner_layer_ids()
        if not owner_layer_ids:
            return None, None
        full_layer_ids = set(self._token_pool_full_attention_layer_ids())
        pool_full_layer_ids = {
            int(layer_id)
            for layer_id in pool.layer_specs
            if self._token_pool_layer_type(int(layer_id)) == "full_attention"
        }
        if not full_layer_ids or not full_layer_ids.issubset(pool_full_layer_ids):
            return None, None
        expected_owner_layer_ids = set(self.config.full_kv_layers)
        if not expected_owner_layer_ids.issubset(set(owner_layer_ids)):
            return None, None
        req_ids = [req.req_id for req in reqs]
        if not persistent_rows:
            self._token_pool_clear_full_attention_rows(req_ids)
        build_paged_rows = bool(
            persistent_rows and _token_pool_full_attention_paged_metadata_requested()
        )
        try:
            prepared_batch = backend.prepare_full_attention_decode_batch(
                requests=reqs,
                reservations=reservations,
                caches_by_req_id=self._caches,
                owner_layer_ids=owner_layer_ids,
                kv_indices_padding_steps=kv_indices_padding_steps,
                persistent_rows=persistent_rows,
                build_paged_rows=build_paged_rows,
            )
            self.metrics.token_pool_full_attention_row_invalidations += (
                prepared_batch.invalidated_existing_rows
            )
            self.metrics.token_pool_full_attention_row_reuses += (
                prepared_batch.reused_existing_rows
            )
            self.metrics.token_pool_full_attention_row_appends += (
                prepared_batch.appended_existing_rows
            )
            self.metrics.token_pool_full_attention_row_rebuilds += (
                prepared_batch.rebuilt_persistent_rows
            )
            return prepared_batch.metadata, prepared_batch.paged_metadata
        except (DistinctCacheBatchError, RuntimeError, ValueError, KeyError):
            self._token_pool_clear_full_attention_rows(req_ids)
            return None, None

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

    def _token_pool_layer_type_by_layer_id(self) -> dict[int, str]:
        pool = self._token_kv_pool
        if pool is None:
            return {}
        result: dict[int, str] = {}
        for layer_id in sorted(pool.layer_specs):
            layer_type = self._token_pool_layer_type(int(layer_id))
            if layer_type is not None:
                result[int(layer_id)] = str(layer_type)
        return result

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
        backend = self._token_pool_decode_backend
        if backend is None:
            return
        owner_layer_ids = self._token_pool_full_attention_owner_layer_ids()
        if not owner_layer_ids:
            return
        invalidate_req_ids = backend.commit_full_attention_decode_to_caches(
            reservations=reservations,
            caches_by_req_id=self._caches,
            owner_layer_ids=owner_layer_ids,
        )
        if invalidate_req_ids:
            self._token_pool_invalidate_full_attention_rows(invalidate_req_ids)

    def _token_pool_clear_full_attention_rows(self, req_ids) -> None:
        backend = self._token_pool_decode_backend
        if backend is None:
            return
        backend.clear_full_attention_rows(req_ids)

    def _token_pool_invalidate_full_attention_rows(self, req_ids) -> None:
        backend = self._token_pool_decode_backend
        if backend is None:
            return
        invalidated = backend.invalidate_full_attention_rows(req_ids)
        if invalidated:
            self.metrics.token_pool_full_attention_row_invalidations += invalidated

    def _token_pool_invalidate_full_attention_rows_containing(self, slots) -> None:
        backend = self._token_pool_decode_backend
        if backend is None:
            return
        invalidated = backend.invalidate_full_attention_rows_containing(slots)
        if invalidated:
            self.metrics.token_pool_full_attention_row_invalidations += invalidated

    def _token_pool_discard_decode_reservations(
        self,
        reservations: list[_TokenPoolDecodeReservation],
    ) -> None:
        table = self._token_table
        allocator = self._token_slot_allocator
        if table is None or allocator is None:
            return
        backend = self._token_pool_decode_backend
        if backend is not None:
            backend.discard_decode_batch(reservations)
            return
        self._token_pool_clear_full_attention_rows(
            [reservation.req_id for reservation in reservations]
        )
        for reservation in reversed(reservations):
            if reservation.req_id in self._token_pool_req_slots:
                self._token_pool_truncate_table_row(
                    reservation.req_slot,
                    reservation.previous_length,
                )
            token_slots = self._token_pool_token_slots.get(reservation.req_id)
            if token_slots is not None and reservation.token_slot in token_slots:
                token_slots.remove(reservation.token_slot)
            if reservation.page_state_snapshot is not None:
                continue
            page_owned = (
                self._token_pool_page_owned_slots.get(reservation.req_id, set())
            )
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
