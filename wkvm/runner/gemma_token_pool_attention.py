"""Token-pool attention backend launch helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
import time
from typing import Any


TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES = (
    "WKVM_ENABLE_TOKEN_POOL_TRITON",
    "WKVM_DISABLE_TOKEN_POOL_TRITON",
    "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON",
    "WKVM_ENABLE_TOKEN_POOL_SPLIT_TRITON",
    "WKVM_TOKEN_POOL_TRITON_SPLIT_KV",
    "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON",
    "WKVM_TOKEN_POOL_TRITON_PAGED_SPLIT_KV",
    "WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION",
    "WKVM_TOKEN_POOL_TRITON_DOT_DTYPE",
    "WKVM_TOKEN_POOL_TRITON_STRICT",
)


@dataclass(frozen=True)
class TokenPoolTritonDispatchPlan:
    env_enabled: bool
    env_forced_off: bool
    env_disabled: bool
    effective_enabled: bool
    auto_default_enabled: bool
    paged_enabled: bool
    split_enabled: bool
    paged_split_enabled: bool
    input_precision_policy: str
    dot_dtype_policy: str
    strict: bool


_TOKEN_POOL_TRITON_DISPATCH_ENV_KEY: tuple[str | None, ...] | None = None
_TOKEN_POOL_TRITON_DISPATCH_PLAN: TokenPoolTritonDispatchPlan | None = None
_TOKEN_POOL_TRITON_DECODE_FN = None
_TOKEN_POOL_TRITON_SPLIT_DECODE_FN = None
_TOKEN_POOL_TRITON_PAGED_DECODE_FN = None
_TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN = None
_TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN = None
_TOKEN_POOL_TRITON_DISABLED_SHAPES: set[tuple[Any, ...]] = set()
_TOKEN_POOL_TRITON_FALLBACK_REASONS: dict[str, int] = {}
_TOKEN_POOL_TRITON_STATS: dict[str, int] = {
    "calls": 0,
    "env_enabled_calls": 0,
    "env_disabled_calls": 0,
    "effective_enabled_calls": 0,
    "effective_disabled_calls": 0,
    "auto_enabled_calls": 0,
    "disabled_shape_skips": 0,
    "attempts": 0,
    "successes": 0,
    "import_error_fallbacks": 0,
    "runtime_errors": 0,
    "recoverable_runtime_fallbacks": 0,
    "nonrecoverable_runtime_errors": 0,
    "paged_enabled_calls": 0,
    "paged_attempts": 0,
    "paged_successes": 0,
    "paged_request_table_attempts": 0,
    "paged_request_table_successes": 0,
    "paged_split_enabled_calls": 0,
    "paged_split_attempts": 0,
    "paged_split_successes": 0,
    "paged_split_skips_by_min_splits": 0,
    "split_enabled_calls": 0,
    "split_attempts": 0,
    "split_successes": 0,
    "split_skips_by_min_splits": 0,
}


@dataclass(frozen=True)
class TokenPoolTritonDecodeResult:
    output: Any | None = None
    attempted: bool = False
    succeeded: bool = False


@dataclass(frozen=True)
class TokenPoolAttentionDecodeResult:
    output: Any
    weights: Any | None = None
    kind: str = "reference"


@dataclass(frozen=True)
class TokenPoolTritonAttentionBackendHooks:
    decode_fn: Callable[[], Callable[..., Any]]
    split_decode_fn: Callable[[], Callable[..., Any]]
    paged_decode_fn: Callable[[], Callable[..., Any]]
    paged_split_decode_fn: Callable[[], Callable[..., Any]]
    block_groups: Callable[[int, Any], int]
    record_fallback: Callable[[str], None]
    is_recoverable_runtime_error: Callable[[RuntimeError], bool]
    paged_request_table_decode_fn: Callable[[], Callable[..., Any]] | None = None


@dataclass(frozen=True)
class TokenPoolAttentionBackendHooks:
    triton: TokenPoolTritonAttentionBackendHooks
    reference_decode: Callable[..., tuple[Any, Any | None]]
    slot_count: Callable[[Any], int]
    record_kv_write_timing: Callable[..., None]
    record_triton_attempt_timing: Callable[[float], None]
    record_attention_timing: Callable[[str, int, float], None]
    now: Callable[[], float] = time.perf_counter


def _coerce_env_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return False


def token_pool_triton_input_precision_policy() -> str:
    return token_pool_triton_input_precision_policy_from_raw(
        os.environ.get("WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION")
    )


def token_pool_triton_dot_dtype_policy() -> str:
    return token_pool_triton_dot_dtype_policy_from_raw(
        os.environ.get("WKVM_TOKEN_POOL_TRITON_DOT_DTYPE")
    )


def token_pool_triton_input_precision_policy_from_raw(raw: str | None) -> str:
    return raw if raw is not None else "auto_float32_ieee_low_precision_tf32"


def token_pool_triton_dot_dtype_policy_from_raw(raw: str | None) -> str:
    return raw if raw is not None else "auto_float32_fp32_low_precision_native"


def token_pool_triton_block_groups(groups: int, dtype: Any) -> int:
    groups = int(groups)
    fallback = 1 << (groups - 1).bit_length()
    try:
        from wkvm.runner.gemma_token_pool_triton import _block_g, _resolve_native_dot

        return int(_block_g(groups, _resolve_native_dot(dtype)))
    except Exception:
        return fallback


def is_recoverable_token_pool_triton_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "out of resource",
            "shared memory",
            "ptxas",
            "triton",
            "cuda error",
            "invalid argument",
            "illegal memory access",
            "no kernel image",
        )
    )


def token_pool_triton_effective_enabled() -> tuple[bool, bool]:
    plan = token_pool_triton_dispatch_plan()
    return plan.effective_enabled, plan.auto_default_enabled


def token_pool_triton_effective_enabled_from_values(
    enabled: bool | None,
    disabled: bool | None,
) -> tuple[bool, bool]:
    if disabled is True or enabled is False:
        return False, False
    if enabled is True:
        return True, False
    return True, True


def token_pool_triton_dispatch_plan() -> TokenPoolTritonDispatchPlan:
    global _TOKEN_POOL_TRITON_DISPATCH_ENV_KEY, _TOKEN_POOL_TRITON_DISPATCH_PLAN

    env_key = tuple(
        os.environ.get(name) for name in TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES
    )
    if (
        _TOKEN_POOL_TRITON_DISPATCH_PLAN is not None
        and env_key == _TOKEN_POOL_TRITON_DISPATCH_ENV_KEY
    ):
        return _TOKEN_POOL_TRITON_DISPATCH_PLAN

    enabled = _coerce_env_bool(env_key[0])
    disabled = _coerce_env_bool(env_key[1])
    effective_enabled, auto_default_enabled = (
        token_pool_triton_effective_enabled_from_values(enabled, disabled)
    )
    plan = TokenPoolTritonDispatchPlan(
        env_enabled=enabled is True,
        env_forced_off=enabled is False,
        env_disabled=disabled is True,
        effective_enabled=effective_enabled,
        auto_default_enabled=auto_default_enabled,
        paged_enabled=_coerce_env_bool(env_key[2]) is True,
        split_enabled=(
            _coerce_env_bool(env_key[3]) is True
            or _coerce_env_bool(env_key[4]) is True
        ),
        paged_split_enabled=(
            _coerce_env_bool(env_key[5]) is True
            or _coerce_env_bool(env_key[6]) is True
        ),
        input_precision_policy=token_pool_triton_input_precision_policy_from_raw(
            env_key[7]
        ),
        dot_dtype_policy=token_pool_triton_dot_dtype_policy_from_raw(env_key[8]),
        strict=str(env_key[9] or "").lower() in {"1", "true", "yes"},
    )
    _TOKEN_POOL_TRITON_DISPATCH_ENV_KEY = env_key
    _TOKEN_POOL_TRITON_DISPATCH_PLAN = plan
    return plan


def reset_token_pool_triton_dispatch_plan_cache() -> None:
    global _TOKEN_POOL_TRITON_DISPATCH_ENV_KEY, _TOKEN_POOL_TRITON_DISPATCH_PLAN
    _TOKEN_POOL_TRITON_DISPATCH_ENV_KEY = None
    _TOKEN_POOL_TRITON_DISPATCH_PLAN = None


def token_pool_triton_decode_fn():
    global _TOKEN_POOL_TRITON_DECODE_FN
    if _TOKEN_POOL_TRITON_DECODE_FN is None:
        from wkvm.runner.gemma_token_pool_triton import token_pool_gqa_decode

        _TOKEN_POOL_TRITON_DECODE_FN = token_pool_gqa_decode
    return _TOKEN_POOL_TRITON_DECODE_FN


def token_pool_triton_split_decode_fn():
    global _TOKEN_POOL_TRITON_SPLIT_DECODE_FN
    if _TOKEN_POOL_TRITON_SPLIT_DECODE_FN is None:
        from wkvm.runner.gemma_token_pool_triton import token_pool_gqa_decode_split_kv

        _TOKEN_POOL_TRITON_SPLIT_DECODE_FN = token_pool_gqa_decode_split_kv
    return _TOKEN_POOL_TRITON_SPLIT_DECODE_FN


def token_pool_triton_paged_decode_fn():
    global _TOKEN_POOL_TRITON_PAGED_DECODE_FN
    if _TOKEN_POOL_TRITON_PAGED_DECODE_FN is None:
        from wkvm.runner.gemma_token_pool_triton import token_pool_paged_gqa_decode

        _TOKEN_POOL_TRITON_PAGED_DECODE_FN = token_pool_paged_gqa_decode
    return _TOKEN_POOL_TRITON_PAGED_DECODE_FN


def token_pool_triton_paged_request_table_decode_fn():
    global _TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN
    if _TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN is None:
        from wkvm.runner.gemma_token_pool_triton import (
            token_pool_paged_request_table_gqa_decode,
        )

        _TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN = (
            token_pool_paged_request_table_gqa_decode
        )
    return _TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN


def token_pool_triton_paged_split_decode_fn():
    global _TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN
    if _TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN is None:
        from wkvm.runner.gemma_token_pool_triton import (
            token_pool_paged_gqa_decode_split_kv,
        )

        _TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN = (
            token_pool_paged_gqa_decode_split_kv
        )
    return _TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN


def reset_token_pool_triton_decode_fn_cache() -> None:
    global _TOKEN_POOL_TRITON_DECODE_FN, _TOKEN_POOL_TRITON_SPLIT_DECODE_FN
    global _TOKEN_POOL_TRITON_PAGED_DECODE_FN, _TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN
    global _TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN
    _TOKEN_POOL_TRITON_DECODE_FN = None
    _TOKEN_POOL_TRITON_SPLIT_DECODE_FN = None
    _TOKEN_POOL_TRITON_PAGED_DECODE_FN = None
    _TOKEN_POOL_TRITON_PAGED_REQUEST_TABLE_DECODE_FN = None
    _TOKEN_POOL_TRITON_PAGED_SPLIT_DECODE_FN = None


def token_pool_triton_disabled_shapes() -> set[tuple[Any, ...]]:
    return _TOKEN_POOL_TRITON_DISABLED_SHAPES


def token_pool_triton_disabled_shape_count() -> int:
    return len(_TOKEN_POOL_TRITON_DISABLED_SHAPES)


def clear_token_pool_triton_disabled_shapes() -> None:
    _TOKEN_POOL_TRITON_DISABLED_SHAPES.clear()


def record_token_pool_triton_fallback(reason: str) -> None:
    _TOKEN_POOL_TRITON_FALLBACK_REASONS[reason] = (
        _TOKEN_POOL_TRITON_FALLBACK_REASONS.get(reason, 0) + 1
    )


def token_pool_triton_fallback_reasons() -> dict[str, int]:
    return dict(_TOKEN_POOL_TRITON_FALLBACK_REASONS)


def reset_token_pool_triton_fallback_reasons() -> None:
    _TOKEN_POOL_TRITON_FALLBACK_REASONS.clear()


def token_pool_triton_stats_storage() -> dict[str, int]:
    return _TOKEN_POOL_TRITON_STATS


def token_pool_triton_stats_snapshot() -> dict[str, int]:
    return dict(_TOKEN_POOL_TRITON_STATS)


def token_pool_triton_stats_report(
    *,
    split_plan: tuple[bool, int, int, int | None],
) -> dict[str, Any]:
    stats: dict[str, Any] = token_pool_triton_stats_snapshot()
    plan = token_pool_triton_dispatch_plan()
    stats["fallback_reasons"] = token_pool_triton_fallback_reasons()
    stats["disabled_shape_count"] = token_pool_triton_disabled_shape_count()
    stats["env_enabled"] = plan.env_enabled
    stats["env_disabled"] = plan.env_disabled
    stats["split_enabled"] = plan.split_enabled
    stats["paged_split_enabled"] = plan.paged_split_enabled
    stats["split_size"] = int(split_plan[1])
    stats["split_min_splits"] = int(split_plan[2])
    stats["input_precision_policy"] = plan.input_precision_policy
    stats["dot_dtype_policy"] = plan.dot_dtype_policy
    stats["effective_enabled"] = plan.effective_enabled
    stats["auto_default_enabled"] = plan.auto_default_enabled
    return stats


def reset_token_pool_triton_stats_counts() -> None:
    for key in _TOKEN_POOL_TRITON_STATS:
        _TOKEN_POOL_TRITON_STATS[key] = 0


def reset_token_pool_triton_runtime_state(
    *,
    clear_disabled_shapes: bool = False,
) -> None:
    reset_token_pool_triton_stats_counts()
    reset_token_pool_triton_fallback_reasons()
    reset_token_pool_triton_decode_fn_cache()
    reset_token_pool_triton_dispatch_plan_cache()
    if clear_disabled_shapes:
        clear_token_pool_triton_disabled_shapes()


class TokenPoolAttentionBackend:
    """Owns decode-side token-pool attention ordering and fallback."""

    def __init__(
        self,
        *,
        stats: dict[str, int],
        disabled_shapes: set[tuple[Any, ...]],
        hooks: TokenPoolAttentionBackendHooks,
    ) -> None:
        self._stats = stats
        self._hooks = hooks
        self._triton = TokenPoolTritonAttentionBackend(
            stats=stats,
            disabled_shapes=disabled_shapes,
            hooks=hooks.triton,
        )

    def decode(
        self,
        attn: Any,
        query_states: Any,
        *,
        dispatch_context: Any,
        dispatch_plan: Any,
        current_key_states: Any | None = None,
        current_value_states: Any | None = None,
        timing_enabled: bool = False,
    ) -> TokenPoolAttentionDecodeResult:
        attention_start = self._hooks.now() if timing_enabled else 0.0
        attention_rows = int(query_states.shape[0])
        self._record_dispatch_plan_call(dispatch_plan)

        if current_key_states is not None and current_value_states is not None:
            kv_write_start = self._hooks.now() if timing_enabled else 0.0
            out_cache_loc = dispatch_context.store_current_kv(
                current_key_states,
                current_value_states,
            )
            if timing_enabled and out_cache_loc is not None:
                self._hooks.record_kv_write_timing(
                    tokens=self._hooks.slot_count(out_cache_loc),
                    elapsed=self._hooks.now() - kv_write_start,
                )

        triton_attempt_start = self._hooks.now() if timing_enabled else 0.0
        triton_result = self._triton.decode(
            attn,
            query_states,
            dispatch_context=dispatch_context,
            dispatch_plan=dispatch_plan,
        )
        if triton_result.attempted and timing_enabled:
            self._hooks.record_triton_attempt_timing(
                self._hooks.now() - triton_attempt_start
            )
        if triton_result.succeeded:
            if timing_enabled:
                self._hooks.record_attention_timing(
                    "triton",
                    attention_rows,
                    self._hooks.now() - attention_start,
                )
            return TokenPoolAttentionDecodeResult(
                output=triton_result.output,
                weights=None,
                kind="triton",
            )

        reference_metadata, reference_kv_pool, reference_layer_idx = (
            dispatch_context.reference_decode_inputs()
        )
        output, weights = self._hooks.reference_decode(
            attn,
            query_states,
            decode_metadata=reference_metadata,
            token_kv_pool=reference_kv_pool,
            layer_idx=reference_layer_idx,
        )
        if timing_enabled:
            self._hooks.record_attention_timing(
                "reference",
                attention_rows,
                self._hooks.now() - attention_start,
            )
        return TokenPoolAttentionDecodeResult(
            output=output,
            weights=weights,
            kind="reference",
        )

    def decode_call(
        self,
        attn: Any,
        query_states: Any,
        *,
        attention_call: Any,
        dispatch_plan: Any | None = None,
        timing_enabled: bool = False,
    ) -> TokenPoolAttentionDecodeResult:
        if dispatch_plan is None:
            dispatch_plan = token_pool_triton_dispatch_plan()
        dispatch_context = attention_call.backend_dispatch_context()
        current_key_states, current_value_states = (
            attention_call.current_kv_for_backend()
        )
        return self.decode(
            attn,
            query_states,
            dispatch_context=dispatch_context,
            dispatch_plan=dispatch_plan,
            current_key_states=current_key_states,
            current_value_states=current_value_states,
            timing_enabled=timing_enabled,
        )

    def try_decode_call(
        self,
        attn: Any,
        query_states: Any,
        *,
        attention_call: Any,
        dispatch_plan: Any | None = None,
        timing_enabled: bool = False,
    ) -> TokenPoolAttentionDecodeResult | None:
        if not bool(attention_call.decode_attention_enabled):
            return None
        return self.decode_call(
            attn,
            query_states,
            attention_call=attention_call,
            dispatch_plan=dispatch_plan,
            timing_enabled=timing_enabled,
        )

    def _record_dispatch_plan_call(self, dispatch_plan: Any) -> None:
        self._stats["calls"] += 1
        if dispatch_plan.env_enabled:
            self._stats["env_enabled_calls"] += 1
        if dispatch_plan.env_forced_off or dispatch_plan.env_disabled:
            self._stats["env_disabled_calls"] += 1
        if dispatch_plan.effective_enabled:
            self._stats["effective_enabled_calls"] += 1
        else:
            self._stats["effective_disabled_calls"] += 1
        if dispatch_plan.auto_default_enabled:
            self._stats["auto_enabled_calls"] += 1
        if dispatch_plan.paged_enabled:
            self._stats["paged_enabled_calls"] += 1
        if dispatch_plan.split_enabled:
            self._stats["split_enabled_calls"] += 1
        if dispatch_plan.paged_split_enabled:
            self._stats["paged_split_enabled_calls"] += 1


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
            if not bool(getattr(metadata, "block_tables_materialized", True)):
                raise RuntimeError("paged split decode requires compact block tables")
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
            request_block_tables = getattr(metadata, "request_block_tables", None)
            if request_block_tables is not None:
                self._stats["paged_request_table_attempts"] = (
                    self._stats.get("paged_request_table_attempts", 0) + 1
                )
                paged_request_table_decode_fn = (
                    self._hooks.paged_request_table_decode_fn
                    or token_pool_triton_paged_request_table_decode_fn
                )
                output = paged_request_table_decode_fn()(
                    query_states,
                    key_buffer,
                    value_buffer,
                    metadata.req_pool_indices,
                    request_block_tables,
                    metadata.block_table_lens,
                    metadata.selected_start_positions,
                    metadata.seq_lens,
                    block_size=metadata.block_size,
                    num_key_value_groups=attn.num_key_value_groups,
                    scaling=attn.scaling,
                    output=output_buffer,
                )
                self._stats["paged_request_table_successes"] = (
                    self._stats.get("paged_request_table_successes", 0) + 1
                )
            else:
                if not bool(getattr(metadata, "block_tables_materialized", True)):
                    raise RuntimeError("paged decode requires materialized block tables")
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


def build_token_pool_attention_backend(
    *,
    reference_decode: Callable[..., tuple[Any, Any | None]],
    slot_count: Callable[[Any], int],
    record_kv_write_timing: Callable[..., None],
    record_triton_attempt_timing: Callable[[float], None],
    record_attention_timing: Callable[[str, int, float], None],
    block_groups: Callable[[int, Any], int] | None = None,
    is_recoverable_runtime_error: Callable[[RuntimeError], bool] | None = None,
    now: Callable[[], float] = time.perf_counter,
    stats: dict[str, int] | None = None,
    disabled_shapes: set[tuple[Any, ...]] | None = None,
) -> TokenPoolAttentionBackend:
    block_groups = block_groups or token_pool_triton_block_groups
    is_recoverable_runtime_error = (
        is_recoverable_runtime_error or is_recoverable_token_pool_triton_error
    )
    triton_hooks = TokenPoolTritonAttentionBackendHooks(
        decode_fn=token_pool_triton_decode_fn,
        split_decode_fn=token_pool_triton_split_decode_fn,
        paged_decode_fn=token_pool_triton_paged_decode_fn,
        paged_split_decode_fn=token_pool_triton_paged_split_decode_fn,
        block_groups=block_groups,
        record_fallback=record_token_pool_triton_fallback,
        is_recoverable_runtime_error=is_recoverable_runtime_error,
        paged_request_table_decode_fn=token_pool_triton_paged_request_table_decode_fn,
    )
    hooks = TokenPoolAttentionBackendHooks(
        triton=triton_hooks,
        reference_decode=reference_decode,
        slot_count=slot_count,
        record_kv_write_timing=record_kv_write_timing,
        record_triton_attempt_timing=record_triton_attempt_timing,
        record_attention_timing=record_attention_timing,
        now=now,
    )
    return TokenPoolAttentionBackend(
        stats=token_pool_triton_stats_storage() if stats is None else stats,
        disabled_shapes=(
            token_pool_triton_disabled_shapes()
            if disabled_shapes is None
            else disabled_shapes
        ),
        hooks=hooks,
    )
