"""Token-pool attention backend launch helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenPoolTritonDecodeResult:
    output: Any | None = None
    attempted: bool = False
    succeeded: bool = False


@dataclass(frozen=True)
class TokenPoolTritonAttentionBackendHooks:
    decode_fn: Callable[[], Callable[..., Any]]
    split_decode_fn: Callable[[], Callable[..., Any]]
    paged_decode_fn: Callable[[], Callable[..., Any]]
    paged_split_decode_fn: Callable[[], Callable[..., Any]]
    block_groups: Callable[[int, Any], int]
    record_fallback: Callable[[str], None]
    is_recoverable_runtime_error: Callable[[RuntimeError], bool]


class TokenPoolTritonAttentionBackend:
    """Owns token-pool Triton decode launch and fallback accounting."""

    def __init__(
        self,
        *,
        stats: dict[str, int],
        disabled_shapes: set[tuple[Any, ...]],
        hooks: TokenPoolTritonAttentionBackendHooks,
    ) -> None:
        self._stats = stats
        self._disabled_shapes = disabled_shapes
        self._hooks = hooks

    def decode(
        self,
        attn: Any,
        query_states: Any,
        *,
        dispatch_context: Any,
        dispatch_plan: Any,
    ) -> TokenPoolTritonDecodeResult:
        if not query_states.is_cuda or not dispatch_plan.effective_enabled:
            return TokenPoolTritonDecodeResult()

        shape_key = self._shape_key(attn, query_states, dispatch_plan)
        if shape_key in self._disabled_shapes:
            self._stats["disabled_shape_skips"] += 1
            self._hooks.record_fallback("disabled_shape")
            return TokenPoolTritonDecodeResult()

        self._stats["attempts"] += 1
        try:
            output = self._launch(attn, query_states, dispatch_context, dispatch_plan)
        except ImportError:
            self._stats["import_error_fallbacks"] += 1
            self._hooks.record_fallback("import_error")
            return TokenPoolTritonDecodeResult(attempted=True)
        except RuntimeError as exc:
            self._stats["runtime_errors"] += 1
            if dispatch_plan.strict:
                self._stats["nonrecoverable_runtime_errors"] += 1
                raise
            if not dispatch_context.has_flat_metadata:
                self._stats["nonrecoverable_runtime_errors"] += 1
                raise
            if not self._hooks.is_recoverable_runtime_error(exc):
                self._stats["nonrecoverable_runtime_errors"] += 1
                raise
            self._stats["recoverable_runtime_fallbacks"] += 1
            self._hooks.record_fallback("recoverable_runtime_error")
            self._disabled_shapes.add(shape_key)
            return TokenPoolTritonDecodeResult(attempted=True)

        self._stats["successes"] += 1
        return TokenPoolTritonDecodeResult(
            output=output,
            attempted=True,
            succeeded=True,
        )

    def _shape_key(
        self,
        attn: Any,
        query_states: Any,
        dispatch_plan: Any,
    ) -> tuple[Any, ...]:
        return (
            int(query_states.shape[1]),
            int(query_states.shape[3]),
            int(attn.num_key_value_groups),
            query_states.dtype,
            query_states.device,
            dispatch_plan.input_precision_policy,
            dispatch_plan.dot_dtype_policy,
        )

    def _launch(
        self,
        attn: Any,
        query_states: Any,
        dispatch_context: Any,
        dispatch_plan: Any,
    ) -> Any:
        kv_buffers = dispatch_context.kv_buffers_for_attention()
        if kv_buffers is None:
            raise RuntimeError("token-pool KV buffers are required for Triton attention")
        key_buffer, value_buffer = kv_buffers
        output_buffer = dispatch_context.attention_output_buffer(
            batch=int(query_states.shape[0]),
            query_heads=int(query_states.shape[1]),
            head_dim=int(query_states.shape[3]),
            dtype=query_states.dtype,
            device=query_states.device,
        )
        kernel_dispatch = dispatch_context.select_triton_dispatch(
            paged_enabled=dispatch_plan.paged_enabled,
            split_enabled=dispatch_plan.split_enabled,
            paged_split_enabled=dispatch_plan.paged_split_enabled,
        )
        if kernel_dispatch.is_paged:
            return self._launch_paged(
                attn,
                query_states,
                key_buffer,
                value_buffer,
                output_buffer,
                dispatch_context,
                kernel_dispatch,
            )
        return self._launch_flat(
            attn,
            query_states,
            key_buffer,
            value_buffer,
            output_buffer,
            dispatch_context,
            kernel_dispatch,
        )

    def _split_workspace(
        self,
        *,
        dispatch_context: Any,
        query_states: Any,
        key_buffer: Any,
        max_splits: int | None,
        num_key_value_groups: int,
    ) -> Any:
        if max_splits is None:
            return None
        return dispatch_context.attention_split_workspace(
            batch=int(query_states.shape[0]),
            kv_heads=int(key_buffer.shape[1]),
            max_splits=max_splits,
            block_groups=self._hooks.block_groups(
                int(num_key_value_groups),
                query_states.dtype,
            ),
            head_dim=int(query_states.shape[3]),
            device=query_states.device,
        )

    def _launch_paged(
        self,
        attn: Any,
        query_states: Any,
        key_buffer: Any,
        value_buffer: Any,
        output_buffer: Any,
        dispatch_context: Any,
        kernel_dispatch: Any,
    ) -> Any:
        metadata = kernel_dispatch.metadata
        self._stats["paged_attempts"] += 1
        if kernel_dispatch.is_split:
            self._stats["paged_split_attempts"] += 1
            self._stats["split_attempts"] += 1
            split_workspace = self._split_workspace(
                dispatch_context=dispatch_context,
                query_states=query_states,
                key_buffer=key_buffer,
                max_splits=kernel_dispatch.max_splits,
                num_key_value_groups=attn.num_key_value_groups,
            )
            output = self._hooks.paged_split_decode_fn()(
                query_states,
                key_buffer,
                value_buffer,
                metadata.block_tables,
                metadata.block_table_lens,
                metadata.selected_start_positions,
                metadata.seq_lens,
                block_size=metadata.block_size,
                num_key_value_groups=attn.num_key_value_groups,
                scaling=attn.scaling,
                max_seq_len=kernel_dispatch.max_seq_len,
                split_size=kernel_dispatch.split_size,
                min_splits=kernel_dispatch.min_splits,
                workspace=split_workspace,
                output=output_buffer,
            )
            self._stats["paged_split_successes"] += 1
            self._stats["split_successes"] += 1
        else:
            if kernel_dispatch.split_skipped_by_min_splits:
                self._stats["paged_split_skips_by_min_splits"] += 1
                self._stats["split_skips_by_min_splits"] += 1
            output = self._hooks.paged_decode_fn()(
                query_states,
                key_buffer,
                value_buffer,
                metadata.block_tables,
                metadata.block_table_lens,
                metadata.selected_start_positions,
                metadata.seq_lens,
                block_size=metadata.block_size,
                num_key_value_groups=attn.num_key_value_groups,
                scaling=attn.scaling,
                output=output_buffer,
            )
        self._stats["paged_successes"] += 1
        return output

    def _launch_flat(
        self,
        attn: Any,
        query_states: Any,
        key_buffer: Any,
        value_buffer: Any,
        output_buffer: Any,
        dispatch_context: Any,
        kernel_dispatch: Any,
    ) -> Any:
        metadata = kernel_dispatch.metadata
        if kernel_dispatch.is_split:
            self._stats["split_attempts"] += 1
            split_workspace = self._split_workspace(
                dispatch_context=dispatch_context,
                query_states=query_states,
                key_buffer=key_buffer,
                max_splits=kernel_dispatch.max_splits,
                num_key_value_groups=attn.num_key_value_groups,
            )
            output = self._hooks.split_decode_fn()(
                query_states,
                key_buffer,
                value_buffer,
                metadata.kv_indptr,
                metadata.kv_indices,
                num_key_value_groups=attn.num_key_value_groups,
                scaling=attn.scaling,
                max_seq_len=kernel_dispatch.max_seq_len,
                split_size=kernel_dispatch.split_size,
                min_splits=kernel_dispatch.min_splits,
                seq_lens=metadata.seq_lens,
                workspace=split_workspace,
                output=output_buffer,
            )
            self._stats["split_successes"] += 1
            return output

        if kernel_dispatch.split_skipped_by_min_splits:
            self._stats["split_skips_by_min_splits"] += 1
        return self._hooks.decode_fn()(
            query_states,
            key_buffer,
            value_buffer,
            metadata.kv_indptr,
            metadata.kv_indices,
            num_key_value_groups=attn.num_key_value_groups,
            scaling=attn.scaling,
            output=output_buffer,
        )
