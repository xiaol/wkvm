"""Decode-side token KV pool primitives for native Gemma serving.

These classes model the minimal vLLM/SGLang-style substrate WKVM needs before
the dense padded-KV decode path can be replaced by paged/token-pool attention.
They do not run attention by themselves; they own request-to-token mappings,
per-layer KV buffers, and flattened decode metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class TokenKVLayerSpec:
    layer_id: int
    num_kv_heads: int
    head_dim: int
    dtype: Any | None = None
    kv_share_target_layer: int | None = None


@dataclass(frozen=True)
class TokenPoolTritonDecodePlan:
    should_split: bool
    split_size: int
    min_splits: int
    max_splits: int | None = None


@dataclass(frozen=True)
class DecodeBatchMetadata:
    req_pool_indices: Any
    seq_lens: Any
    logical_seq_lens: Any
    out_cache_loc: Any | None
    kv_indptr: Any
    kv_indices: Any
    out_cache_loc_long: Any | None = None
    max_seq_len: int | None = None
    triton_decode_plan: TokenPoolTritonDecodePlan | None = None

    def __post_init__(self) -> None:
        if self.triton_decode_plan is None:
            object.__setattr__(
                self,
                "triton_decode_plan",
                build_token_pool_triton_decode_plan(self.max_seq_len),
            )


@dataclass(frozen=True)
class TokenSlotRowChunks:
    chunks: tuple[Any, ...]
    trusted: bool = False


@dataclass(frozen=True)
class PagedDecodeBatchMetadata:
    req_pool_indices: Any
    seq_lens: Any
    logical_seq_lens: Any
    out_cache_loc: Any | None
    block_tables: Any
    block_table_lens: Any
    selected_start_positions: Any
    block_size: int
    slot_mapping: Any | None = None
    out_cache_loc_long: Any | None = None
    max_seq_len: int | None = None
    triton_decode_plan: TokenPoolTritonDecodePlan | None = None

    def __post_init__(self) -> None:
        if self.triton_decode_plan is None:
            object.__setattr__(
                self,
                "triton_decode_plan",
                build_token_pool_triton_decode_plan(self.max_seq_len),
            )


@dataclass(frozen=True)
class TokenPoolLayerDecodeBinding:
    metadata: DecodeBatchMetadata | None
    paged_metadata: PagedDecodeBatchMetadata | None


@dataclass(frozen=True)
class TokenPoolAttentionBinding:
    layer_idx: int | None
    metadata: DecodeBatchMetadata | None
    paged_metadata: PagedDecodeBatchMetadata | None
    kv_pool: Any | None

    def out_cache_loc_for_write(self) -> Any | None:
        return _metadata_out_cache_loc_for_write(self.metadata)

    def store_current_kv(self, key_states: Any, value_states: Any) -> Any | None:
        out_cache_loc = self.out_cache_loc_for_write()
        if self.layer_idx is None or self.kv_pool is None or out_cache_loc is None:
            return None
        self.kv_pool.set_kv(
            int(self.layer_idx),
            out_cache_loc,
            key_states,
            value_states,
        )
        return out_cache_loc


@dataclass(frozen=True)
class TokenPoolAttentionPlan:
    layer_idx: int | None
    metadata: DecodeBatchMetadata | None
    paged_metadata: PagedDecodeBatchMetadata | None
    kv_pool: Any | None
    binding: Any | None = None
    use_decode_attention: bool = False

    @classmethod
    def from_binding(
        cls,
        binding: Any | None,
        *,
        layer_idx: int | None,
        attention_mask_present: bool = False,
        query_seq_len: int | None = None,
    ) -> "TokenPoolAttentionPlan":
        if binding is None:
            return cls(
                layer_idx=None,
                metadata=None,
                paged_metadata=None,
                kv_pool=None,
                binding=None,
                use_decode_attention=False,
            )
        metadata = getattr(binding, "metadata", None)
        paged_metadata = getattr(binding, "paged_metadata", None)
        kv_pool = getattr(binding, "kv_pool", None)
        bound_layer_idx = getattr(binding, "layer_idx", layer_idx)
        use_decode_attention = _token_pool_decode_attention_enabled(
            layer_idx=bound_layer_idx,
            metadata=metadata,
            kv_pool=kv_pool,
            attention_mask_present=attention_mask_present,
            query_seq_len=query_seq_len,
        )
        return cls(
            layer_idx=bound_layer_idx,
            metadata=metadata,
            paged_metadata=paged_metadata,
            kv_pool=kv_pool,
            binding=binding,
            use_decode_attention=use_decode_attention,
        )

    def attention_kwargs(self) -> dict[str, Any]:
        return {
            "decode_metadata": self.metadata,
            "paged_decode_metadata": self.paged_metadata,
            "token_kv_pool": self.kv_pool,
            "layer_idx": self.layer_idx,
        }

    def store_current_kv(self, key_states: Any, value_states: Any) -> Any | None:
        if key_states is None or value_states is None:
            return None
        out_cache_loc = _binding_out_cache_loc_for_write(self.binding, self.metadata)
        if self.layer_idx is None or self.kv_pool is None or out_cache_loc is None:
            return None
        store_current_kv = getattr(self.binding, "store_current_kv", None)
        if store_current_kv is not None:
            return store_current_kv(key_states, value_states)
        self.kv_pool.set_kv(
            int(self.layer_idx),
            out_cache_loc,
            key_states,
            value_states,
        )
        return out_cache_loc


def _metadata_out_cache_loc_for_write(metadata: Any | None) -> Any | None:
    if metadata is None:
        return None
    out_cache_loc = getattr(metadata, "out_cache_loc_long", None)
    if out_cache_loc is None:
        out_cache_loc = getattr(metadata, "out_cache_loc", None)
    return out_cache_loc


def _binding_out_cache_loc_for_write(binding: Any | None, metadata: Any | None) -> Any | None:
    out_cache_loc_for_write = getattr(binding, "out_cache_loc_for_write", None)
    if out_cache_loc_for_write is not None:
        return out_cache_loc_for_write()
    return _metadata_out_cache_loc_for_write(metadata)


def _token_pool_decode_attention_enabled(
    *,
    layer_idx: int | None,
    metadata: Any | None,
    kv_pool: Any | None,
    attention_mask_present: bool,
    query_seq_len: int | None,
) -> bool:
    if metadata is None or kv_pool is None or layer_idx is None:
        return False
    if bool(attention_mask_present):
        return False
    if getattr(metadata, "out_cache_loc", None) is None:
        return False
    try:
        return int(query_seq_len) == 1
    except (TypeError, ValueError):
        return False


def _null_attention_binding() -> TokenPoolAttentionBinding:
    return TokenPoolAttentionBinding(
        layer_idx=None,
        metadata=None,
        paged_metadata=None,
        kv_pool=None,
    )


def resolve_token_pool_attention_binding(
    token_pool_decode: Any | None,
    layer_idx: int | None,
    layer_type: str | None,
    *,
    attention_mask_present: bool = False,
) -> Any:
    if token_pool_decode is None or layer_idx is None or layer_type is None:
        return _null_attention_binding()

    attention_binding_for_layer = getattr(
        token_pool_decode,
        "attention_binding_for_layer",
        None,
    )
    if attention_binding_for_layer is not None:
        binding = attention_binding_for_layer(
            layer_idx,
            layer_type,
            attention_mask_present=attention_mask_present,
        )
        return binding if binding is not None else _null_attention_binding()

    attention_metadata_for_layer = getattr(
        token_pool_decode,
        "attention_metadata_for_layer",
        None,
    )
    if attention_metadata_for_layer is not None:
        metadata, paged_metadata, token_kv_pool = attention_metadata_for_layer(
            layer_idx,
            layer_type,
            attention_mask_present=attention_mask_present,
        )
        return TokenPoolAttentionBinding(
            layer_idx=int(layer_idx),
            metadata=metadata,
            paged_metadata=paged_metadata,
            kv_pool=token_kv_pool,
        )

    metadata_for_layer = getattr(token_pool_decode, "metadata_for_layer", None)
    if metadata_for_layer is not None:
        metadata = metadata_for_layer(layer_idx, layer_type)
    else:
        metadata_by_layer_type = getattr(token_pool_decode, "metadata_by_layer_type", {})
        metadata = metadata_by_layer_type.get(str(layer_type))

    paged_metadata = None
    paged_metadata_for_layer = getattr(token_pool_decode, "paged_metadata_for_layer", None)
    if paged_metadata_for_layer is not None:
        paged_metadata = paged_metadata_for_layer(layer_idx, layer_type)
    token_kv_pool = getattr(token_pool_decode, "kv_pool", None)
    layer_specs = getattr(token_kv_pool, "layer_specs", {}) if token_kv_pool is not None else {}
    if token_kv_pool is not None and int(layer_idx) not in layer_specs:
        if metadata is not None and not bool(attention_mask_present):
            raise RuntimeError(
                f"token-pool metadata was provided for layer {int(layer_idx)}, "
                "but the KV pool has no spec for that layer"
            )
        metadata = None
        paged_metadata = None
        token_kv_pool = None
    return TokenPoolAttentionBinding(
        layer_idx=int(layer_idx),
        metadata=metadata,
        paged_metadata=paged_metadata,
        kv_pool=token_kv_pool,
    )


def resolve_token_pool_attention_plan(
    token_pool_decode: Any | None,
    layer_idx: int | None,
    layer_type: str | None,
    *,
    attention_mask_present: bool = False,
    query_seq_len: int | None = None,
) -> TokenPoolAttentionPlan:
    if token_pool_decode is not None:
        attention_plan_for_layer = getattr(
            token_pool_decode,
            "attention_plan_for_layer",
            None,
        )
        if attention_plan_for_layer is not None:
            plan = attention_plan_for_layer(
                layer_idx,
                layer_type,
                attention_mask_present=attention_mask_present,
                query_seq_len=query_seq_len,
            )
            if plan is not None:
                return plan
    binding = resolve_token_pool_attention_binding(
        token_pool_decode,
        layer_idx,
        layer_type,
        attention_mask_present=attention_mask_present,
    )
    return TokenPoolAttentionPlan.from_binding(
        binding,
        layer_idx=layer_idx,
        attention_mask_present=attention_mask_present,
        query_seq_len=query_seq_len,
    )


@dataclass(frozen=True)
class TokenPoolDecodeContext:
    metadata_by_layer_type: dict[str, DecodeBatchMetadata]
    kv_pool: Any | None = None
    metadata_by_layer_id: dict[int, DecodeBatchMetadata] | None = None
    paged_metadata_by_layer_type: dict[str, PagedDecodeBatchMetadata] | None = None
    paged_metadata_by_layer_id: dict[int, PagedDecodeBatchMetadata] | None = None
    covered_layer_types: frozenset[str] | None = None
    layer_id_metadata_only_types: frozenset[str] = frozenset()
    _layer_bindings_by_id: dict[int, TokenPoolLayerDecodeBinding] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        layer_ids: set[int] = set()
        if self.metadata_by_layer_id:
            layer_ids.update(int(layer_id) for layer_id in self.metadata_by_layer_id)
        if self.paged_metadata_by_layer_id:
            layer_ids.update(int(layer_id) for layer_id in self.paged_metadata_by_layer_id)
        layer_specs = getattr(self.kv_pool, "layer_specs", None)
        if layer_specs is not None:
            layer_ids.update(int(layer_id) for layer_id in layer_specs)
        bindings: dict[int, TokenPoolLayerDecodeBinding] = {}
        for layer_id in sorted(layer_ids):
            metadata = (
                self.metadata_by_layer_id.get(layer_id)
                if self.metadata_by_layer_id is not None
                else None
            )
            paged_metadata = (
                self.paged_metadata_by_layer_id.get(layer_id)
                if self.paged_metadata_by_layer_id is not None
                else None
            )
            if metadata is not None or paged_metadata is not None:
                bindings[layer_id] = TokenPoolLayerDecodeBinding(
                    metadata=metadata,
                    paged_metadata=paged_metadata,
                )
        object.__setattr__(self, "_layer_bindings_by_id", bindings)

    def metadata_for_layer(
        self,
        layer_idx: int | None,
        layer_type: str | None,
    ) -> DecodeBatchMetadata | None:
        if layer_idx is not None:
            binding = self._layer_bindings_by_id.get(int(layer_idx))
            if binding is not None and binding.metadata is not None:
                return binding.metadata
        if layer_idx is not None and self.metadata_by_layer_id:
            metadata = self.metadata_by_layer_id.get(int(layer_idx))
            if metadata is not None:
                return metadata
            if (
                layer_type is not None
                and str(layer_type) in self.layer_id_metadata_only_types
            ):
                return None
        if layer_type is None:
            return None
        return self.metadata_by_layer_type.get(str(layer_type))

    def paged_metadata_for_layer(
        self,
        layer_idx: int | None,
        layer_type: str | None,
    ) -> PagedDecodeBatchMetadata | None:
        if layer_idx is not None:
            binding = self._layer_bindings_by_id.get(int(layer_idx))
            if binding is not None and binding.paged_metadata is not None:
                return binding.paged_metadata
        if layer_idx is not None and self.paged_metadata_by_layer_id:
            metadata = self.paged_metadata_by_layer_id.get(int(layer_idx))
            if metadata is not None:
                return metadata
            if (
                layer_type is not None
                and str(layer_type) in self.layer_id_metadata_only_types
            ):
                return None
        if layer_type is None or self.paged_metadata_by_layer_type is None:
            return None
        return self.paged_metadata_by_layer_type.get(str(layer_type))

    def attention_metadata_for_layer(
        self,
        layer_idx: int | None,
        layer_type: str | None,
        *,
        attention_mask_present: bool = False,
    ) -> tuple[DecodeBatchMetadata | None, PagedDecodeBatchMetadata | None, Any | None]:
        binding = self.attention_binding_for_layer(
            layer_idx,
            layer_type,
            attention_mask_present=attention_mask_present,
        )
        return binding.metadata, binding.paged_metadata, binding.kv_pool

    def attention_plan_for_layer(
        self,
        layer_idx: int | None,
        layer_type: str | None,
        *,
        attention_mask_present: bool = False,
        query_seq_len: int | None = None,
    ) -> TokenPoolAttentionPlan:
        binding = self.attention_binding_for_layer(
            layer_idx,
            layer_type,
            attention_mask_present=attention_mask_present,
        )
        return TokenPoolAttentionPlan.from_binding(
            binding,
            layer_idx=layer_idx,
            attention_mask_present=attention_mask_present,
            query_seq_len=query_seq_len,
        )

    def attention_binding_for_layer(
        self,
        layer_idx: int | None,
        layer_type: str | None,
        *,
        attention_mask_present: bool = False,
    ) -> TokenPoolAttentionBinding:
        if layer_idx is None or layer_type is None:
            return TokenPoolAttentionBinding(
                layer_idx=None,
                metadata=None,
                paged_metadata=None,
                kv_pool=None,
            )
        layer_idx = int(layer_idx)
        metadata = self.metadata_for_layer(layer_idx, layer_type)
        paged_metadata = self.paged_metadata_for_layer(layer_idx, layer_type)
        token_kv_pool = self.kv_pool
        if token_kv_pool is None:
            return TokenPoolAttentionBinding(
                layer_idx=layer_idx,
                metadata=metadata,
                paged_metadata=paged_metadata,
                kv_pool=None,
            )
        layer_specs = getattr(token_kv_pool, "layer_specs", {})
        if layer_idx not in layer_specs:
            if metadata is not None and not bool(attention_mask_present):
                raise RuntimeError(
                    f"token-pool metadata was provided for layer {layer_idx}, "
                    "but the KV pool has no spec for that layer"
                )
            return TokenPoolAttentionBinding(
                layer_idx=layer_idx,
                metadata=None,
                paged_metadata=None,
                kv_pool=None,
            )
        return TokenPoolAttentionBinding(
            layer_idx=layer_idx,
            metadata=metadata,
            paged_metadata=paged_metadata,
            kv_pool=token_kv_pool,
        )


def _env_flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    return int(raw.strip())


def _token_pool_triton_split_size() -> int:
    split_size = _env_int("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE", 512)
    if split_size < 1:
        raise ValueError("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE must be >= 1")
    return split_size


def _token_pool_triton_min_splits() -> int:
    min_splits = _env_int("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS", 4)
    if min_splits < 2:
        raise ValueError("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS must be >= 2")
    return min_splits


def build_token_pool_triton_decode_plan(
    max_seq_len: Any,
) -> TokenPoolTritonDecodePlan:
    split_size = _token_pool_triton_split_size()
    min_splits = _token_pool_triton_min_splits()
    if max_seq_len is None:
        return TokenPoolTritonDecodePlan(
            should_split=True,
            split_size=split_size,
            min_splits=min_splits,
            max_splits=None,
        )
    max_splits = (int(max_seq_len) + split_size - 1) // split_size
    return TokenPoolTritonDecodePlan(
        should_split=max_splits >= min_splits,
        split_size=split_size,
        min_splits=min_splits,
        max_splits=max_splits,
    )


def _token_pool_timing_enabled() -> bool:
    return _env_flag("WKVM_TOKEN_POOL_TIMING") or _env_flag("WKVM_NATIVE_FORWARD_TIMING")


def _token_pool_kv_store_triton_enabled() -> bool:
    if _env_bool("WKVM_DISABLE_TOKEN_POOL_TRITON") is True:
        return False
    if _env_bool("WKVM_DISABLE_TOKEN_POOL_KV_STORE_TRITON") is True:
        return False
    explicit = _env_bool("WKVM_ENABLE_TOKEN_POOL_KV_STORE_TRITON")
    if explicit is not None:
        return bool(explicit)
    if _env_bool("WKVM_ENABLE_TOKEN_POOL_TRITON") is False:
        return False
    return True


class TokenPoolDecodeGraphBuffer:
    """Graph-stable token-pool decode metadata tensors.

    The graph path captures one context whose tensor addresses must stay stable.
    When metadata builders already return persistent workspace slices, the buffer
    can alias those tensors and replay only validates the incoming context. If a
    later builder returns different tensors with the same shape, replay copies
    values into the captured tensors.
    """

    def __init__(self, context: TokenPoolDecodeContext | None) -> None:
        self.context = context

    @classmethod
    def capture(
        cls,
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        clone_tensors: bool = False,
    ) -> "TokenPoolDecodeGraphBuffer":
        if token_pool_decode is None:
            return cls(None)
        return cls(
            cls._copy_context_structure(
                token_pool_decode,
                clone_tensors=bool(clone_tensors),
            )
        )

    @classmethod
    def _copy_context_structure(
        cls,
        token_pool_decode: TokenPoolDecodeContext,
        *,
        clone_tensors: bool,
    ) -> TokenPoolDecodeContext:
        memo: dict[int, Any] = {}

        def copy_metadata(metadata):
            ident = id(metadata)
            copied = memo.get(ident)
            if copied is None:
                copied = (
                    cls._clone_decode_metadata(metadata)
                    if clone_tensors
                    else metadata
                )
                memo[ident] = copied
            return copied

        return TokenPoolDecodeContext(
            metadata_by_layer_type={
                str(layer_type): copy_metadata(metadata)
                for layer_type, metadata in token_pool_decode.metadata_by_layer_type.items()
            },
            kv_pool=token_pool_decode.kv_pool,
            metadata_by_layer_id={
                int(layer_id): copy_metadata(metadata)
                for layer_id, metadata in (
                    token_pool_decode.metadata_by_layer_id or {}
                ).items()
            }
            or None,
            paged_metadata_by_layer_type={
                str(layer_type): copy_metadata(metadata)
                for layer_type, metadata in (
                    token_pool_decode.paged_metadata_by_layer_type or {}
                ).items()
            }
            or None,
            paged_metadata_by_layer_id={
                int(layer_id): copy_metadata(metadata)
                for layer_id, metadata in (
                    token_pool_decode.paged_metadata_by_layer_id or {}
                ).items()
            }
            or None,
            covered_layer_types=token_pool_decode.covered_layer_types,
            layer_id_metadata_only_types=token_pool_decode.layer_id_metadata_only_types,
        )

    def copy_from(self, token_pool_decode: TokenPoolDecodeContext | None) -> dict[str, int]:
        stats = {
            "cuda_graph_metadata_tensor_copies": 0,
            "cuda_graph_metadata_tensor_copy_skips": 0,
        }
        if self.context is None:
            if token_pool_decode is not None:
                raise ValueError("graph was captured without token-pool metadata")
            return stats
        if token_pool_decode is None:
            raise ValueError("graph token-pool metadata is required for replay")
        if self.context.kv_pool is not token_pool_decode.kv_pool:
            raise ValueError("token-pool graph kv_pool changed")
        if self.context.covered_layer_types != token_pool_decode.covered_layer_types:
            raise ValueError("token-pool graph covered layer types changed")
        if (
            self.context.layer_id_metadata_only_types
            != token_pool_decode.layer_id_metadata_only_types
        ):
            raise ValueError("token-pool graph metadata-only layer types changed")
        copied: set[tuple[int, int]] = set()
        copied_metadata: set[tuple[int, int]] = set()
        self._copy_decode_metadata_group(
            self.context.metadata_by_layer_type,
            token_pool_decode.metadata_by_layer_type,
            "metadata_by_layer_type",
            copied=copied,
            copied_metadata=copied_metadata,
            stats=stats,
        )
        self._copy_decode_metadata_group(
            self.context.metadata_by_layer_id or {},
            token_pool_decode.metadata_by_layer_id or {},
            "metadata_by_layer_id",
            copied=copied,
            copied_metadata=copied_metadata,
            stats=stats,
        )
        self._copy_decode_metadata_group(
            self.context.paged_metadata_by_layer_type or {},
            token_pool_decode.paged_metadata_by_layer_type or {},
            "paged_metadata_by_layer_type",
            copied=copied,
            copied_metadata=copied_metadata,
            stats=stats,
        )
        self._copy_decode_metadata_group(
            self.context.paged_metadata_by_layer_id or {},
            token_pool_decode.paged_metadata_by_layer_id or {},
            "paged_metadata_by_layer_id",
            copied=copied,
            copied_metadata=copied_metadata,
            stats=stats,
        )
        return stats

    @staticmethod
    def _clone_decode_metadata(metadata):
        if getattr(metadata, "block_tables", None) is not None:
            return PagedDecodeBatchMetadata(
                req_pool_indices=metadata.req_pool_indices.clone(),
                seq_lens=metadata.seq_lens.clone(),
                logical_seq_lens=metadata.logical_seq_lens.clone(),
                out_cache_loc=(
                    None
                    if metadata.out_cache_loc is None
                    else metadata.out_cache_loc.clone()
                ),
                block_tables=metadata.block_tables.clone(),
                block_table_lens=metadata.block_table_lens.clone(),
                selected_start_positions=metadata.selected_start_positions.clone(),
                block_size=int(metadata.block_size),
                slot_mapping=(
                    None
                    if getattr(metadata, "slot_mapping", None) is None
                    else metadata.slot_mapping.clone()
                ),
                out_cache_loc_long=(
                    None
                    if getattr(metadata, "out_cache_loc_long", None) is None
                    else metadata.out_cache_loc_long.clone()
                ),
                max_seq_len=getattr(metadata, "max_seq_len", None),
                triton_decode_plan=getattr(metadata, "triton_decode_plan", None),
            )
        return DecodeBatchMetadata(
            req_pool_indices=metadata.req_pool_indices.clone(),
            seq_lens=metadata.seq_lens.clone(),
            logical_seq_lens=metadata.logical_seq_lens.clone(),
            out_cache_loc=(
                None
                if metadata.out_cache_loc is None
                else metadata.out_cache_loc.clone()
            ),
            kv_indptr=metadata.kv_indptr.clone(),
            kv_indices=metadata.kv_indices.clone(),
            out_cache_loc_long=(
                None
                if getattr(metadata, "out_cache_loc_long", None) is None
                else metadata.out_cache_loc_long.clone()
            ),
            max_seq_len=getattr(metadata, "max_seq_len", None),
            triton_decode_plan=getattr(metadata, "triton_decode_plan", None),
        )

    @classmethod
    def _copy_decode_metadata_group(
        cls,
        dst_group: dict[Any, Any],
        src_group: dict[Any, Any],
        name: str,
        *,
        copied: set[tuple[int, int]] | None = None,
        copied_metadata: set[tuple[int, int]] | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        if set(dst_group) != set(src_group):
            raise ValueError(f"token-pool graph {name} keys changed")
        for key in dst_group:
            cls._copy_decode_metadata(
                dst_group[key],
                src_group[key],
                f"{name}.{key}",
                copied=copied,
                copied_metadata=copied_metadata,
                stats=stats,
            )

    @classmethod
    def _copy_decode_metadata(
        cls,
        dst,
        src,
        prefix: str,
        *,
        copied: set[tuple[int, int]] | None = None,
        copied_metadata: set[tuple[int, int]] | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        metadata_pair = None
        if copied_metadata is not None:
            metadata_pair = (id(dst), id(src))
            if metadata_pair in copied_metadata:
                if stats is not None:
                    stats["cuda_graph_metadata_tensor_copy_skips"] = (
                        stats.get("cuda_graph_metadata_tensor_copy_skips", 0)
                        + cls._decode_metadata_tensor_pair_count(dst, src)
                    )
                return
        if int(getattr(dst, "block_size", -1)) != int(getattr(src, "block_size", -1)):
            raise ValueError(f"token-pool graph {prefix}.block_size changed")
        if getattr(dst, "max_seq_len", None) != getattr(src, "max_seq_len", None):
            raise ValueError(f"token-pool graph {prefix}.max_seq_len changed")
        if getattr(dst, "triton_decode_plan", None) != getattr(
            src,
            "triton_decode_plan",
            None,
        ):
            raise ValueError(f"token-pool graph {prefix}.triton_decode_plan changed")
        for name in (
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
        ):
            cls._copy_decode_metadata_tensor(
                getattr(dst, name, None),
                getattr(src, name, None),
                f"{prefix}.{name}",
                copied=copied,
                stats=stats,
            )
        if copied_metadata is not None and metadata_pair is not None:
            copied_metadata.add(metadata_pair)

    @staticmethod
    def _decode_metadata_tensor_pair_count(dst, src) -> int:
        count = 0
        for name in (
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
        ):
            if getattr(dst, name, None) is not None and getattr(src, name, None) is not None:
                count += 1
        return count

    @staticmethod
    def _copy_decode_metadata_tensor(
        dst,
        src,
        name: str,
        *,
        copied: set[tuple[int, int]] | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        if dst is None or src is None:
            if dst is not src:
                raise ValueError(f"token-pool graph {name} presence changed")
            return
        copy_key = None
        if copied is not None:
            copy_key = (id(dst), id(src))
            if copy_key in copied:
                if stats is not None:
                    stats["cuda_graph_metadata_tensor_copy_skips"] = (
                        stats.get("cuda_graph_metadata_tensor_copy_skips", 0) + 1
                    )
                return
        if tuple(dst.shape) != tuple(src.shape):
            raise ValueError(f"token-pool graph {name} shape changed")
        if dst.dtype != src.dtype:
            raise ValueError(f"token-pool graph {name} dtype changed")
        try:
            same_storage = (
                dst is src
                or (
                    dst.device == src.device
                    and int(dst.data_ptr()) == int(src.data_ptr())
                )
            )
        except Exception:
            same_storage = dst is src
        if same_storage:
            if copied is not None and copy_key is not None:
                copied.add(copy_key)
            if stats is not None:
                stats["cuda_graph_metadata_tensor_copy_skips"] = (
                    stats.get("cuda_graph_metadata_tensor_copy_skips", 0) + 1
                )
            return
        if copied is not None:
            copied.add(copy_key)
        if stats is not None:
            stats["cuda_graph_metadata_tensor_copies"] = (
                stats.get("cuda_graph_metadata_tensor_copies", 0) + 1
            )
        src_tensor = src if src.device == dst.device else src.to(device=dst.device)
        dst.copy_(src_tensor, non_blocking=True)


def _ensure_decode_metadata_workspace(
    workspace: dict[str, Any] | None,
    *,
    device: Any,
    row_count: int,
    kv_capacity: int,
) -> dict[str, Any]:
    import torch

    row_count = int(row_count)
    kv_capacity = int(kv_capacity)
    if row_count < 1:
        raise ValueError("row_count must be >= 1")
    if kv_capacity < 0:
        raise ValueError("kv_capacity must be >= 0")
    target = workspace if workspace is not None else {}
    row_buffer_size = row_count
    indptr_size = row_count + 1
    kv_buffer_size = max(1, kv_capacity)
    target_device = torch.empty(0, device=device).device

    def needs(name: str, size: int, dtype: Any) -> bool:
        tensor = target.get(name)
        return (
            tensor is None
            or int(tensor.numel()) < int(size)
            or tensor.dtype != dtype
            or tensor.device != target_device
        )

    if (
        needs("req_pool_indices", row_buffer_size, torch.int32)
        or needs("seq_lens", row_buffer_size, torch.int32)
        or needs("logical_seq_lens", row_buffer_size, torch.int32)
        or needs("out_cache_loc", row_buffer_size, torch.int32)
        or needs("out_cache_loc_long", row_buffer_size, torch.long)
        or needs("kv_indptr", indptr_size, torch.int32)
        or needs("kv_indices", kv_buffer_size, torch.int32)
    ):
        old = target
        new_workspace = {
            "req_pool_indices": old.get("req_pool_indices")
            if old.get("req_pool_indices") is not None
            and int(old["req_pool_indices"].numel()) >= row_buffer_size
            and old["req_pool_indices"].dtype == torch.int32
            and old["req_pool_indices"].device == target_device
            else torch.empty(row_buffer_size, dtype=torch.int32, device=device),
            "seq_lens": old.get("seq_lens")
            if old.get("seq_lens") is not None
            and int(old["seq_lens"].numel()) >= row_buffer_size
            and old["seq_lens"].dtype == torch.int32
            and old["seq_lens"].device == target_device
            else torch.empty(row_buffer_size, dtype=torch.int32, device=device),
            "logical_seq_lens": old.get("logical_seq_lens")
            if old.get("logical_seq_lens") is not None
            and int(old["logical_seq_lens"].numel()) >= row_buffer_size
            and old["logical_seq_lens"].dtype == torch.int32
            and old["logical_seq_lens"].device == target_device
            else torch.empty(row_buffer_size, dtype=torch.int32, device=device),
            "out_cache_loc": old.get("out_cache_loc")
            if old.get("out_cache_loc") is not None
            and int(old["out_cache_loc"].numel()) >= row_buffer_size
            and old["out_cache_loc"].dtype == torch.int32
            and old["out_cache_loc"].device == target_device
            else torch.empty(row_buffer_size, dtype=torch.int32, device=device),
            "out_cache_loc_long": old.get("out_cache_loc_long")
            if old.get("out_cache_loc_long") is not None
            and int(old["out_cache_loc_long"].numel()) >= row_buffer_size
            and old["out_cache_loc_long"].dtype == torch.long
            and old["out_cache_loc_long"].device == target_device
            else torch.empty(row_buffer_size, dtype=torch.long, device=device),
            "kv_indptr": old.get("kv_indptr")
            if old.get("kv_indptr") is not None
            and int(old["kv_indptr"].numel()) >= indptr_size
            and old["kv_indptr"].dtype == torch.int32
            and old["kv_indptr"].device == target_device
            else torch.empty(indptr_size, dtype=torch.int32, device=device),
            "kv_indices": old.get("kv_indices")
            if old.get("kv_indices") is not None
            and int(old["kv_indices"].numel()) >= kv_buffer_size
            and old["kv_indices"].dtype == torch.int32
            and old["kv_indices"].device == target_device
            else torch.empty(kv_buffer_size, dtype=torch.int32, device=device),
        }
        target.clear()
        target.update(new_workspace)
    return target


def _ensure_paged_decode_metadata_workspace(
    workspace: dict[str, Any] | None,
    *,
    device: Any,
    row_count: int,
    block_table_width: int,
) -> dict[str, Any]:
    import torch

    row_count = int(row_count)
    block_table_width = int(block_table_width)
    if row_count < 1:
        raise ValueError("row_count must be >= 1")
    if block_table_width < 1:
        raise ValueError("block_table_width must be >= 1")
    target = workspace if workspace is not None else {}
    target_device = torch.empty(0, device=device).device

    def valid_1d(name: str, dtype: Any) -> bool:
        tensor = target.get(name)
        return (
            tensor is not None
            and int(tensor.numel()) >= row_count
            and tensor.dtype == dtype
            and tensor.device == target_device
        )

    def valid_block_tables() -> bool:
        tensor = target.get("block_tables")
        return (
            tensor is not None
            and len(tuple(tensor.shape)) == 2
            and int(tensor.shape[0]) >= row_count
            and int(tensor.shape[1]) >= block_table_width
            and tensor.dtype == torch.int32
            and tensor.device == target_device
        )

    if (
        not valid_1d("req_pool_indices", torch.int32)
        or not valid_1d("seq_lens", torch.int32)
        or not valid_1d("logical_seq_lens", torch.int32)
        or not valid_1d("out_cache_loc", torch.int32)
        or not valid_1d("out_cache_loc_long", torch.long)
        or not valid_1d("block_table_lens", torch.int32)
        or not valid_1d("selected_start_positions", torch.int32)
        or not valid_block_tables()
    ):
        old = target

        def reuse_1d(name: str, dtype: Any):
            tensor = old.get(name)
            if (
                tensor is not None
                and int(tensor.numel()) >= row_count
                and tensor.dtype == dtype
                and tensor.device == target_device
            ):
                return tensor
            return torch.empty(row_count, dtype=dtype, device=device)

        block_tables = old.get("block_tables")
        if not (
            block_tables is not None
            and len(tuple(block_tables.shape)) == 2
            and int(block_tables.shape[0]) >= row_count
            and int(block_tables.shape[1]) >= block_table_width
            and block_tables.dtype == torch.int32
            and block_tables.device == target_device
        ):
            block_tables = torch.empty(
                (row_count, block_table_width),
                dtype=torch.int32,
                device=device,
            )
        new_workspace = {
            "req_pool_indices": reuse_1d("req_pool_indices", torch.int32),
            "seq_lens": reuse_1d("seq_lens", torch.int32),
            "logical_seq_lens": reuse_1d("logical_seq_lens", torch.int32),
            "out_cache_loc": reuse_1d("out_cache_loc", torch.int32),
            "out_cache_loc_long": reuse_1d("out_cache_loc_long", torch.long),
            "block_table_lens": reuse_1d("block_table_lens", torch.int32),
            "selected_start_positions": reuse_1d(
                "selected_start_positions",
                torch.int32,
            ),
            "block_tables": block_tables,
        }
        target.clear()
        target.update(new_workspace)
    return target


class TokenPoolDecodeMetadataWorkspace:
    """Typed owner for reusable decode metadata workspaces."""

    def __init__(self) -> None:
        self.flat_workspaces: dict[str, dict[str, Any]] = {}
        self.paged_workspaces: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(key: str | None) -> str:
        return "__default__" if key is None else str(key)

    def flat_workspace(self, key: str | None = None) -> dict[str, Any]:
        return self.flat_workspaces.setdefault(self._key(key), {})

    def paged_workspace(self, key: str | None = None) -> dict[str, Any]:
        return self.paged_workspaces.setdefault(self._key(key), {})

    def ensure_flat(
        self,
        key: str | None = None,
        *,
        device: Any,
        row_count: int,
        kv_capacity: int,
    ) -> dict[str, Any]:
        return _ensure_decode_metadata_workspace(
            self.flat_workspace(key),
            device=device,
            row_count=row_count,
            kv_capacity=kv_capacity,
        )

    def ensure_paged(
        self,
        key: str | None = None,
        *,
        device: Any,
        row_count: int,
        block_table_width: int,
    ) -> dict[str, Any]:
        return _ensure_paged_decode_metadata_workspace(
            self.paged_workspace(key),
            device=device,
            row_count=row_count,
            block_table_width=block_table_width,
        )


def build_decode_metadata_from_token_slot_rows(
    token_slot_rows: Iterable[Iterable[int] | Any],
    *,
    req_slots: Iterable[int] | Any | None = None,
    logical_seq_lens: Iterable[int] | Any | None = None,
    out_cache_loc: Iterable[int] | Any | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
    token_pool_capacity: int | None = None,
    workspace: dict[str, Any] | TokenPoolDecodeMetadataWorkspace | None = None,
    workspace_key: str | None = None,
    kv_indices_padding_slots: int = 0,
    trusted_aux_metadata: bool = False,
) -> DecodeBatchMetadata:
    """Build decode metadata from already-ordered per-row token-slot lists.

    ``trusted_aux_metadata`` is for engine-owned hot paths where request slots,
    logical lengths, and output slots come directly from WKVM allocators. Public
    callers should leave it disabled so duplicate/range checks remain strict.
    """

    import torch

    rows = list(token_slot_rows)
    if not rows:
        raise ValueError("decode metadata requires at least one token-slot row")
    if token_pool_capacity is not None and int(token_pool_capacity) < 1:
        raise ValueError("token_pool_capacity must be >= 1 or None")
    if device is None:
        for value in (*rows, req_slots, logical_seq_lens, out_cache_loc):
            if isinstance(value, TokenSlotRowChunks):
                for chunk in value.chunks:
                    if hasattr(chunk, "device"):
                        device = chunk.device
                        break
                if device is not None:
                    break
            if hasattr(value, "device"):
                device = value.device
                break
    if device is None:
        device = "cpu"
    dtype = dtype if dtype is not None else torch.int32
    kv_indices_padding_slots = max(0, int(kv_indices_padding_slots))

    chunks_by_row = []
    trusted_rows: list[bool] = []
    selected_lens: list[int] = []
    indptr = [0]
    for row in rows:
        if isinstance(row, TokenSlotRowChunks):
            row_sources = row.chunks
            trusted_row = bool(row.trusted)
        else:
            row_sources = (row,)
            trusted_row = False
        row_chunks = []
        row_len = 0
        for source in row_sources:
            slots = torch.as_tensor(source, dtype=dtype, device=device).reshape(-1)
            width = int(slots.numel())
            if width < 1:
                continue
            if not trusted_row:
                if bool((slots < 0).any().item()):
                    raise ValueError("token-slot rows must not contain negative slots")
                if (
                    token_pool_capacity is not None
                    and bool((slots >= int(token_pool_capacity)).any().item())
                ):
                    raise ValueError("token-slot row exceeds token_pool_capacity")
            row_chunks.append(slots)
            row_len += width
        if row_len < 1:
            raise ValueError("token-slot rows must be non-empty")
        if not trusted_row:
            if len(row_chunks) == 1:
                validation_slots = row_chunks[0]
            else:
                validation_slots = torch.cat(row_chunks)
            if int(torch.unique(validation_slots).numel()) != row_len:
                raise ValueError("token-slot rows must not contain duplicate slots")
        chunks_by_row.append(tuple(row_chunks))
        trusted_rows.append(trusted_row)
        selected_lens.append(row_len)
        indptr.append(indptr[-1] + row_len)

    row_count = len(rows)
    if req_slots is None:
        req_pool_source = torch.arange(row_count, dtype=torch.int32, device=device)
    else:
        req_pool_source = torch.as_tensor(
            req_slots,
            dtype=torch.int32,
            device=device,
        ).reshape(-1)
        if int(req_pool_source.numel()) != row_count:
            raise ValueError("req_slots length must match slot_sequences")
        if not trusted_aux_metadata:
            if bool((req_pool_source < 0).any().item()):
                raise ValueError("req_slots must be non-negative")
            if int(torch.unique(req_pool_source).numel()) != int(req_pool_source.numel()):
                raise ValueError("req_slots must be unique")

    if logical_seq_lens is None:
        logical_lens_source = torch.as_tensor(
            selected_lens,
            dtype=torch.int32,
            device=device,
        )
    else:
        logical_lens_source = torch.as_tensor(
            logical_seq_lens,
            dtype=torch.int32,
            device=device,
        ).reshape(-1)
        if int(logical_lens_source.numel()) != row_count:
            raise ValueError("logical_seq_lens length must match slot_sequences")
        if not trusted_aux_metadata and bool((logical_lens_source < 0).any().item()):
            raise ValueError("logical_seq_lens must be non-negative")

    out_source = None
    if out_cache_loc is not None:
        out_source = torch.as_tensor(
            out_cache_loc,
            dtype=torch.int32,
            device=device,
        ).reshape(-1)
        if int(out_source.numel()) != row_count:
            raise ValueError("out_cache_loc length must match slot_sequences")
        if not trusted_aux_metadata:
            if bool((out_source < 0).any().item()):
                raise ValueError("out_cache_loc must be non-negative")
            if (
                token_pool_capacity is not None
                and bool((out_source >= int(token_pool_capacity)).any().item())
            ):
                raise ValueError("out_cache_loc exceeds token_pool_capacity")
            if int(torch.unique(out_source).numel()) != int(out_source.numel()):
                raise ValueError("out_cache_loc must be unique")
            for row_chunks, trusted_row, out_slot in zip(
                chunks_by_row,
                trusted_rows,
                out_source,
            ):
                if trusted_row:
                    continue
                matches = 0
                for row_chunk in row_chunks:
                    matches += int(
                        (row_chunk.to(dtype=torch.int32) == out_slot).sum().item()
                    )
                if matches != 1:
                    raise ValueError(
                        "each out_cache_loc slot must appear exactly once in its row"
                    )

    total_kv = int(indptr[-1])
    visible_kv = total_kv + kv_indices_padding_slots
    padded_max_seq_len = max(selected_lens) + (
        (kv_indices_padding_slots + row_count - 1) // row_count
    )
    seq_lens_source = torch.as_tensor(selected_lens, dtype=torch.int32, device=device)
    indptr_source = torch.as_tensor(indptr, dtype=torch.int32, device=device)

    if workspace is not None:
        if isinstance(workspace, TokenPoolDecodeMetadataWorkspace):
            metadata_workspace = workspace.ensure_flat(
                workspace_key,
                device=device,
                row_count=row_count,
                kv_capacity=visible_kv,
            )
        else:
            metadata_workspace = _ensure_decode_metadata_workspace(
                workspace,
                device=device,
                row_count=row_count,
                kv_capacity=visible_kv,
            )
        workspace = metadata_workspace
        req_pool_indices = workspace["req_pool_indices"][:row_count]
        seq_lens = workspace["seq_lens"][:row_count]
        logical_lens = workspace["logical_seq_lens"][:row_count]
        kv_indptr = workspace["kv_indptr"][: row_count + 1]
        kv_indices = workspace["kv_indices"][:visible_kv]
        req_pool_indices.copy_(req_pool_source)
        seq_lens.copy_(seq_lens_source)
        logical_lens.copy_(logical_lens_source)
        kv_indptr.copy_(indptr_source)
        offset = 0
        for row_chunks in chunks_by_row:
            for slots in row_chunks:
                width = int(slots.numel())
                kv_indices[offset : offset + width].copy_(slots.to(dtype=torch.int32))
                offset += width
        if kv_indices_padding_slots:
            if total_kv > 0:
                kv_indices[total_kv:visible_kv].copy_(
                    kv_indices[total_kv - 1 : total_kv].expand(
                        kv_indices_padding_slots
                    )
                )
            else:
                kv_indices[total_kv:visible_kv].zero_()
        out = None
        out_long = None
        if out_source is not None:
            out = workspace["out_cache_loc"][:row_count]
            out.copy_(out_source)
            out_long = workspace["out_cache_loc_long"][:row_count]
            out_long.copy_(out_source.to(dtype=torch.long))
        return DecodeBatchMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            logical_seq_lens=logical_lens,
            out_cache_loc=out,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            out_cache_loc_long=out_long,
            max_seq_len=padded_max_seq_len,
        )

    if total_kv:
        kv_indices = torch.cat(
            [slots for row_chunks in chunks_by_row for slots in row_chunks]
        ).to(dtype=torch.int32)
    else:
        kv_indices = torch.empty(0, dtype=torch.int32, device=device)
    if kv_indices_padding_slots:
        if total_kv > 0:
            padding = kv_indices[-1:].expand(kv_indices_padding_slots)
        else:
            padding = torch.zeros(
                kv_indices_padding_slots,
                dtype=kv_indices.dtype,
                device=kv_indices.device,
            )
        kv_indices = torch.cat((kv_indices, padding), dim=0).contiguous()
    out = out_source
    out_long = None if out_source is None else out_source.to(dtype=torch.long)

    return DecodeBatchMetadata(
        req_pool_indices=req_pool_source,
        seq_lens=seq_lens_source,
        logical_seq_lens=logical_lens_source,
        out_cache_loc=out,
        kv_indptr=indptr_source,
        kv_indices=kv_indices,
        out_cache_loc_long=out_long,
        max_seq_len=padded_max_seq_len,
    )


def build_decode_metadata_from_slot_sequences(
    slot_sequences: Iterable[Iterable[int] | Any],
    **kwargs,
) -> DecodeBatchMetadata:
    return build_decode_metadata_from_token_slot_rows(slot_sequences, **kwargs)


def build_paged_decode_metadata_from_token_slot_rows(
    token_slot_rows: Iterable[Iterable[int] | Any],
    *,
    block_size: int,
    block_table_width: int | None = None,
    req_slots: Iterable[int] | Any | None = None,
    logical_seq_lens: Iterable[int] | Any | None = None,
    out_cache_loc: Iterable[int] | Any | None = None,
    selected_start_positions: Iterable[int] | Any | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
    token_pool_capacity: int | None = None,
    padding_block: int = -1,
    allow_selected_len_gt_logical_len: bool = False,
    max_seq_len: int | None = None,
) -> PagedDecodeBatchMetadata:
    """Build selected-window-relative block tables from page-aligned token rows.

    The returned block table starts at ``selected_start_positions // block_size``
    for each row, not necessarily at logical block zero. A future attention
    backend must use ``selected_start_positions`` when translating logical
    positions into block-table columns.
    """

    import torch

    rows = list(token_slot_rows)
    if not rows:
        raise ValueError("paged decode metadata requires at least one token-slot row")
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if block_table_width is not None:
        block_table_width = int(block_table_width)
        if block_table_width < 1:
            raise ValueError("block_table_width must be >= 1 or None")
    if token_pool_capacity is not None and int(token_pool_capacity) < 1:
        raise ValueError("token_pool_capacity must be >= 1 or None")
    if device is None:
        for value in (
            *rows,
            req_slots,
            logical_seq_lens,
            out_cache_loc,
            selected_start_positions,
        ):
            if hasattr(value, "device"):
                device = value.device
                break
    if device is None:
        device = "cpu"
    dtype = dtype if dtype is not None else torch.int32

    row_slot_values: list[list[int]] = []
    selected_lens: list[int] = []
    for row in rows:
        slots = torch.as_tensor(row, dtype=dtype, device=device).reshape(-1)
        selected_len = int(slots.numel())
        if selected_len < 1:
            raise ValueError("token-slot rows must be non-empty")
        values = [int(value) for value in slots.detach().cpu().tolist()]
        if any(value < 0 for value in values):
            raise ValueError("token-slot rows must not contain negative slots")
        if token_pool_capacity is not None and any(
            value >= int(token_pool_capacity) for value in values
        ):
            raise ValueError("token-slot row exceeds token_pool_capacity")
        if len(set(values)) != len(values):
            raise ValueError("token-slot rows must not contain duplicate slots")
        row_slot_values.append(values)
        selected_lens.append(selected_len)

    row_count = len(row_slot_values)
    if req_slots is None:
        req_pool_indices = torch.arange(row_count, dtype=torch.int32, device=device)
    else:
        req_pool_indices = torch.as_tensor(
            req_slots,
            dtype=torch.int32,
            device=device,
        ).reshape(-1)
        if int(req_pool_indices.numel()) != row_count:
            raise ValueError("req_slots length must match token-slot rows")
        if bool((req_pool_indices < 0).any().item()):
            raise ValueError("req_slots must be non-negative")
        if int(torch.unique(req_pool_indices).numel()) != row_count:
            raise ValueError("req_slots must be unique")

    if logical_seq_lens is None:
        logical_lens = torch.as_tensor(selected_lens, dtype=torch.int32, device=device)
    else:
        logical_lens = torch.as_tensor(
            logical_seq_lens,
            dtype=torch.int32,
            device=device,
        ).reshape(-1)
        if int(logical_lens.numel()) != row_count:
            raise ValueError("logical_seq_lens length must match token-slot rows")
        if bool((logical_lens < 0).any().item()):
            raise ValueError("logical_seq_lens must be non-negative")

    if selected_start_positions is None:
        start_positions = [
            int(logical_lens[row].item()) - int(selected_lens[row])
            for row in range(row_count)
        ]
    else:
        start_positions = [
            int(value)
            for value in torch.as_tensor(
                selected_start_positions,
                dtype=torch.int32,
                device=device,
            )
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        ]
        if len(start_positions) != row_count:
            raise ValueError("selected_start_positions length must match token-slot rows")
    if any(value < 0 for value in start_positions):
        raise ValueError("selected_start_positions must be non-negative")

    out = None
    out_long = None
    if out_cache_loc is not None:
        out = torch.as_tensor(
            out_cache_loc,
            dtype=torch.int32,
            device=device,
        ).reshape(-1)
        if int(out.numel()) != row_count:
            raise ValueError("out_cache_loc length must match token-slot rows")
        if bool((out < 0).any().item()):
            raise ValueError("out_cache_loc must be non-negative")
        if token_pool_capacity is not None and bool(
            (out >= int(token_pool_capacity)).any().item()
        ):
            raise ValueError("out_cache_loc exceeds token_pool_capacity")
        if int(torch.unique(out).numel()) != row_count:
            raise ValueError("out_cache_loc must be unique")
        out_long = out.to(dtype=torch.long)

    block_rows: list[list[int]] = []
    for row_idx, values in enumerate(row_slot_values):
        selected_len = int(selected_lens[row_idx])
        logical_len = int(logical_lens[row_idx].item())
        start = int(start_positions[row_idx])
        if (
            start + selected_len > logical_len
            and not bool(allow_selected_len_gt_logical_len)
        ):
            raise ValueError("selected_start_positions plus row length exceeds logical length")
        if out is not None:
            out_slot = int(out[row_idx].item())
            if values.count(out_slot) != 1:
                raise ValueError("each out_cache_loc slot must appear exactly once in its row")
            if allow_selected_len_gt_logical_len and start + selected_len > logical_len:
                decode_offset = selected_len - 1
            else:
                decode_offset = logical_len - 1 - start
            if decode_offset < 0 or decode_offset >= selected_len:
                raise ValueError("out_cache_loc must be inside the selected decode row")
            if int(values[decode_offset]) != out_slot:
                raise ValueError("out_cache_loc must identify the final logical token")

        logical_to_physical_block: dict[int, int] = {}
        row_blocks: list[int] = []
        for offset, slot in enumerate(values):
            logical_pos = start + offset
            if slot % block_size != logical_pos % block_size:
                raise ValueError("token-slot row is not page-aligned for block metadata")
            logical_block = logical_pos // block_size
            physical_block = slot // block_size
            existing = logical_to_physical_block.get(logical_block)
            if existing is None:
                logical_to_physical_block[logical_block] = physical_block
                row_blocks.append(physical_block)
            elif existing != physical_block:
                raise ValueError("one logical block maps to multiple physical blocks")
        block_rows.append(row_blocks)

    max_blocks = max(len(row) for row in block_rows)
    if block_table_width is not None:
        if max_blocks > block_table_width:
            raise ValueError("block_table_width is smaller than the live block table")
        max_blocks = block_table_width
    metadata_max_seq_len = max(selected_lens)
    if max_seq_len is not None:
        max_seq_len = int(max_seq_len)
        if max_seq_len < metadata_max_seq_len:
            raise ValueError("max_seq_len is smaller than the live selected length")
        metadata_max_seq_len = max_seq_len
    block_tables = torch.full(
        (row_count, max_blocks),
        int(padding_block),
        dtype=torch.int32,
        device=device,
    )
    for row_idx, blocks in enumerate(block_rows):
        block_tables[row_idx, : len(blocks)] = torch.as_tensor(
            blocks,
            dtype=torch.int32,
            device=device,
        )

    return PagedDecodeBatchMetadata(
        req_pool_indices=req_pool_indices,
        seq_lens=torch.as_tensor(selected_lens, dtype=torch.int32, device=device),
        logical_seq_lens=logical_lens,
        out_cache_loc=out,
        block_tables=block_tables,
        block_table_lens=torch.as_tensor(
            [len(row) for row in block_rows],
            dtype=torch.int32,
            device=device,
        ),
        selected_start_positions=torch.as_tensor(
            start_positions,
            dtype=torch.int32,
            device=device,
        ),
        block_size=block_size,
        slot_mapping=out_long,
        out_cache_loc_long=out_long,
        max_seq_len=metadata_max_seq_len,
    )


def _slot_values_to_list(slots: Iterable[int] | Any) -> list[int]:
    if isinstance(slots, int):
        return [int(slots)]
    if isinstance(slots, (list, tuple, range)):
        return [int(value) for value in slots]
    try:
        import torch

        if torch.is_tensor(slots):
            return [int(value) for value in slots.detach().cpu().reshape(-1).tolist()]
    except ImportError:
        pass
    try:
        return [int(value) for value in slots]
    except TypeError:
        return [int(slots)]


class ReqToTokenTable:
    """Request-slot to token-slot table for decode metadata construction."""

    def __init__(
        self,
        *,
        max_requests: int,
        max_context_len: int,
        device: Any = "cpu",
        dtype: Any | None = None,
        padding_token: int = -1,
    ) -> None:
        import torch

        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if max_context_len < 1:
            raise ValueError("max_context_len must be >= 1")
        self.max_requests = int(max_requests)
        self.max_context_len = int(max_context_len)
        self.padding_token = int(padding_token)
        self.padding_req_slot = self.max_requests
        self.dtype = dtype if dtype is not None else torch.int32
        self.device = device
        self.req_to_token = torch.full(
            (self.max_requests + 1, self.max_context_len),
            self.padding_token,
            dtype=self.dtype,
            device=device,
        )
        self._lengths = torch.zeros(
            self.max_requests + 1,
            dtype=torch.int32,
            device=device,
        )
        self._free_req_slots = list(range(self.max_requests))
        self._req_to_slot: dict[str, int] = {}
        self._slot_to_req: dict[int, str] = {}
        self._cleared_prefix_lengths = [0 for _ in range(self.max_requests)]
        self.decode_metadata_workspace = TokenPoolDecodeMetadataWorkspace()
        self._decode_metadata_workspaces = (
            self.decode_metadata_workspace.flat_workspaces
        )
        self._paged_decode_metadata_workspaces = (
            self.decode_metadata_workspace.paged_workspaces
        )

    def allocate(self, req_id: str) -> int:
        req_id = str(req_id)
        if req_id in self._req_to_slot:
            raise ValueError(f"request {req_id!r} already has a token-table slot")
        if not self._free_req_slots:
            raise RuntimeError("no free request token-table slots")
        slot = self._free_req_slots.pop(0)
        self._req_to_slot[req_id] = slot
        self._slot_to_req[slot] = req_id
        self.req_to_token[slot].fill_(self.padding_token)
        self._lengths[slot] = 0
        self._cleared_prefix_lengths[slot] = 0
        return slot

    def free(self, req_id_or_slot: str | int) -> None:
        slot = self._resolve_req_slot(req_id_or_slot)
        req_id = self._slot_to_req.pop(slot)
        self._req_to_slot.pop(req_id, None)
        self.req_to_token[slot].fill_(self.padding_token)
        self._lengths[slot] = 0
        self._cleared_prefix_lengths[slot] = 0
        self._free_req_slots.append(slot)
        self._free_req_slots.sort()

    def slot_for(self, req_id: str) -> int:
        return self._req_to_slot[str(req_id)]

    def length(self, req_id_or_slot: str | int) -> int:
        slot = self._resolve_req_slot(req_id_or_slot)
        return int(self._lengths[slot].item())

    def append_slots(self, req_id_or_slot: str | int, token_slots: Iterable[int] | Any) -> tuple[int, int]:
        import torch

        slot = self._resolve_req_slot(req_id_or_slot)
        values = torch.as_tensor(token_slots, dtype=self.dtype, device=self.req_to_token.device)
        values = values.reshape(-1)
        if values.numel() < 1:
            raise ValueError("token_slots must contain at least one slot")
        start = int(self._lengths[slot].item())
        end = start + int(values.numel())
        if end > self.max_context_len:
            raise RuntimeError("request token table context capacity exceeded")
        self.req_to_token[slot, start:end].copy_(values)
        self._lengths[slot] = end
        return start, end

    def truncate(self, req_id_or_slot: str | int, length: int) -> None:
        slot = self._resolve_req_slot(req_id_or_slot)
        length = int(length)
        current = int(self._lengths[slot].item())
        if length < 0 or length > current:
            raise ValueError("truncate length must be within the current request length")
        if length < current:
            self.req_to_token[slot, length:current].fill_(self.padding_token)
            self._lengths[slot] = length
            self._cleared_prefix_lengths[slot] = min(
                self._cleared_prefix_lengths[slot],
                length,
            )

    def clear_before(self, req_id_or_slot: str | int, length: int) -> list[int]:
        slot = self._resolve_req_slot(req_id_or_slot)
        length = int(length)
        current = int(self._lengths[slot].item())
        if length < 0 or length > current:
            raise ValueError("clear length must be within the current request length")
        start = int(self._cleared_prefix_lengths[slot])
        if length <= start:
            return []
        prefix = self.req_to_token[slot, start:length]
        active = prefix[prefix != self.padding_token].detach().cpu().reshape(-1).tolist()
        prefix.fill_(self.padding_token)
        self._cleared_prefix_lengths[slot] = length
        return [int(value) for value in active]

    def ensure_context_len(self, min_context_len: int) -> None:
        import torch

        min_context_len = int(min_context_len)
        if min_context_len <= self.max_context_len:
            return
        new_context_len = max(min_context_len, self.max_context_len * 2)
        extra = torch.full(
            (self.max_requests + 1, new_context_len - self.max_context_len),
            self.padding_token,
            dtype=self.dtype,
            device=self.req_to_token.device,
        )
        self.req_to_token = torch.cat((self.req_to_token, extra), dim=1)
        self.max_context_len = int(new_context_len)

    def slots_for(
        self,
        req_id_or_slot: str | int,
        *,
        start: int = 0,
        end: int | None = None,
    ):
        slot = self._resolve_req_slot(req_id_or_slot)
        length = int(self._lengths[slot].item())
        if end is None:
            end = length
        start = int(start)
        end = int(end)
        if start < 0 or end < start or end > length:
            raise ValueError("invalid request token slice")
        return self.req_to_token[slot, start:end]

    def build_decode_metadata(
        self,
        req_slots: Iterable[int],
        *,
        seq_lens: Iterable[int] | None = None,
        out_cache_loc: Iterable[int] | Any | None = None,
        sliding_window: int | None = None,
        allow_padding: bool = False,
        workspace_key: str | None = None,
    ) -> DecodeBatchMetadata:
        import torch

        req_slots_list = [int(slot) for slot in req_slots]
        if not req_slots_list:
            raise ValueError("decode metadata requires at least one request slot")
        if seq_lens is None:
            logical_lens = [int(self._lengths[slot].item()) for slot in req_slots_list]
        else:
            logical_lens = [int(length) for length in seq_lens]
        if len(logical_lens) != len(req_slots_list):
            raise ValueError("seq_lens length must match req_slots")
        if sliding_window is not None and int(sliding_window) < 1:
            raise ValueError("sliding_window must be >= 1 or None")

        chunks = []
        selected_lens: list[int] = []
        indptr = [0]
        for req_slot, seq_len in zip(req_slots_list, logical_lens):
            self._validate_allocated_slot(req_slot)
            table_len = int(self._lengths[req_slot].item())
            if seq_len < 0 or seq_len > table_len:
                raise ValueError("seq_lens cannot exceed request table length")
            start = 0 if sliding_window is None else max(seq_len - int(sliding_window), 0)
            slots = self.req_to_token[req_slot, start:seq_len]
            if slots.numel():
                padding_mask = slots == self.padding_token
                if bool(padding_mask.any().item()):
                    if not allow_padding:
                        raise RuntimeError(
                            "request token table contains padding inside metadata slice"
                        )
                    slots = slots[~padding_mask]
            chunks.append(slots)
            selected_lens.append(int(slots.numel()))
            indptr.append(indptr[-1] + int(slots.numel()))

        out = None
        out_long = None
        if workspace_key is not None:
            total_kv = int(indptr[-1])
            workspace = self._ensure_decode_metadata_workspace(
                str(workspace_key),
                row_count=len(req_slots_list),
                kv_capacity=total_kv,
            )
            req_pool_indices = workspace["req_pool_indices"][: len(req_slots_list)]
            seq_lens_tensor = workspace["seq_lens"][: len(req_slots_list)]
            logical_seq_lens_tensor = workspace["logical_seq_lens"][: len(req_slots_list)]
            kv_indptr = workspace["kv_indptr"][: len(req_slots_list) + 1]
            kv_indices = workspace["kv_indices"][:total_kv]
            req_pool_indices.copy_(
                torch.as_tensor(
                    req_slots_list,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            seq_lens_tensor.copy_(
                torch.as_tensor(
                    selected_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            logical_seq_lens_tensor.copy_(
                torch.as_tensor(
                    logical_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            kv_indptr.copy_(
                torch.as_tensor(
                    indptr,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            if total_kv:
                if self.dtype == torch.int32:
                    torch.cat(chunks, dim=0, out=kv_indices)
                else:
                    torch.cat(
                        [slots.to(dtype=torch.int32) for slots in chunks],
                        dim=0,
                        out=kv_indices,
                    )
            if out_cache_loc is not None:
                out_src = torch.as_tensor(
                    out_cache_loc,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                ).reshape(-1)
                if int(out_src.numel()) != len(req_slots_list):
                    raise ValueError("out_cache_loc length must match req_slots")
                out = workspace["out_cache_loc"][: len(req_slots_list)]
                out.copy_(out_src)
                out_long = workspace["out_cache_loc_long"][: len(req_slots_list)]
                out_long.copy_(out_src.to(dtype=torch.long))
            return DecodeBatchMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_tensor,
                logical_seq_lens=logical_seq_lens_tensor,
                out_cache_loc=out,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                out_cache_loc_long=out_long,
                max_seq_len=max(selected_lens),
            )

        if chunks:
            kv_indices = torch.cat(chunks).to(dtype=torch.int32)
        else:
            kv_indices = torch.empty(0, dtype=torch.int32, device=self.req_to_token.device)
        if out_cache_loc is not None:
            out = torch.as_tensor(
                out_cache_loc,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ).reshape(-1)
            if int(out.numel()) != len(req_slots_list):
                raise ValueError("out_cache_loc length must match req_slots")
            out_long = out.to(dtype=torch.long)
        return DecodeBatchMetadata(
            req_pool_indices=torch.as_tensor(
                req_slots_list,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            seq_lens=torch.as_tensor(
                selected_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            logical_seq_lens=torch.as_tensor(
                logical_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            out_cache_loc=out,
            kv_indptr=torch.as_tensor(
                indptr,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            kv_indices=kv_indices,
            out_cache_loc_long=out_long,
            max_seq_len=max(selected_lens),
        )

    def _ensure_decode_metadata_workspace(
        self,
        key: str,
        *,
        row_count: int,
        kv_capacity: int,
    ) -> dict[str, Any]:
        return self.decode_metadata_workspace.ensure_flat(
            key,
            device=self.req_to_token.device,
            row_count=row_count,
            kv_capacity=kv_capacity,
        )

    def build_paged_decode_metadata(
        self,
        req_slots: Iterable[int],
        *,
        block_size: int,
        block_table_width: int | None = None,
        seq_lens: Iterable[int] | None = None,
        out_cache_loc: Iterable[int] | Any | None = None,
        sliding_window: int | None = None,
        token_pool_capacity: int | None = None,
        workspace_key: str | None = None,
    ) -> PagedDecodeBatchMetadata:
        import torch

        req_slots_list = [int(slot) for slot in req_slots]
        if not req_slots_list:
            raise ValueError("paged decode metadata requires at least one request slot")
        if seq_lens is None:
            logical_lens = [int(self._lengths[slot].item()) for slot in req_slots_list]
        else:
            logical_lens = [int(length) for length in seq_lens]
        if len(logical_lens) != len(req_slots_list):
            raise ValueError("seq_lens length must match req_slots")
        if sliding_window is not None and int(sliding_window) < 1:
            raise ValueError("sliding_window must be >= 1 or None")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if block_table_width is not None:
            block_table_width = int(block_table_width)
            if block_table_width < 1:
                raise ValueError("block_table_width must be >= 1 or None")
        if token_pool_capacity is not None and int(token_pool_capacity) < 1:
            raise ValueError("token_pool_capacity must be >= 1 or None")
        if len(set(req_slots_list)) != len(req_slots_list):
            raise ValueError("req_slots must be unique")

        out = None
        out_long = None
        if out_cache_loc is not None:
            out = torch.as_tensor(
                out_cache_loc,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ).reshape(-1)
            if int(out.numel()) != len(req_slots_list):
                raise ValueError("out_cache_loc length must match req_slots")
            if bool((out < 0).any().item()):
                raise ValueError("out_cache_loc must be non-negative")
            if token_pool_capacity is not None and bool(
                (out >= int(token_pool_capacity)).any().item()
            ):
                raise ValueError("out_cache_loc exceeds token_pool_capacity")
            if int(torch.unique(out).numel()) != int(out.numel()):
                raise ValueError("out_cache_loc must be unique")
            out_long = out.to(dtype=torch.long)

        block_rows = []
        block_lens: list[int] = []
        selected_lens: list[int] = []
        start_positions: list[int] = []
        for row_idx, (req_slot, seq_len) in enumerate(zip(req_slots_list, logical_lens)):
            self._validate_allocated_slot(req_slot)
            table_len = int(self._lengths[req_slot].item())
            if seq_len < 0 or seq_len > table_len:
                raise ValueError("seq_lens cannot exceed request table length")
            start = 0 if sliding_window is None else max(seq_len - int(sliding_window), 0)
            slots = self.req_to_token[req_slot, start:seq_len]
            if slots.numel() < 1:
                raise ValueError("paged token-slot rows must be non-empty")
            if bool((slots == self.padding_token).any().item()):
                raise RuntimeError("request token table contains padding inside paged metadata slice")
            if bool((slots < 0).any().item()):
                raise ValueError("token-slot rows must not contain negative slots")
            if token_pool_capacity is not None and bool(
                (slots >= int(token_pool_capacity)).any().item()
            ):
                raise ValueError("token-slot row exceeds token_pool_capacity")
            if int(torch.unique(slots).numel()) != int(slots.numel()):
                raise ValueError("token-slot rows must not contain duplicate slots")
            selected_len = int(slots.numel())
            if out is not None:
                out_slot = out[row_idx]
                if int((slots.to(dtype=torch.int32) == out_slot).sum().item()) != 1:
                    raise ValueError("each out_cache_loc slot must appear exactly once in its row")
                decode_offset = int(seq_len) - 1 - int(start)
                if decode_offset < 0 or decode_offset >= selected_len:
                    raise ValueError("out_cache_loc must be inside the selected decode row")
                if int(slots[decode_offset].item()) != int(out_slot.item()):
                    raise ValueError("out_cache_loc must identify the final logical token")

            slots_long = slots.to(dtype=torch.long)
            logical_positions = torch.arange(
                int(start),
                int(start) + selected_len,
                dtype=torch.long,
                device=self.req_to_token.device,
            )
            if bool(((slots_long % block_size) != (logical_positions % block_size)).any().item()):
                raise ValueError("token-slot row is not page-aligned for block metadata")
            logical_blocks = logical_positions // block_size
            physical_blocks = slots_long // block_size
            first_block = int(start) // block_size
            last_block = (int(start) + selected_len - 1) // block_size
            block_count = last_block - first_block + 1
            block_offsets = torch.arange(
                block_count,
                dtype=torch.long,
                device=self.req_to_token.device,
            )
            first_offsets = torch.clamp(
                (first_block + block_offsets) * block_size - int(start),
                min=0,
                max=selected_len - 1,
            )
            row_blocks = physical_blocks[first_offsets].to(dtype=torch.int32)
            expected_physical_blocks = row_blocks[(logical_blocks - first_block).to(dtype=torch.long)]
            if bool((expected_physical_blocks != physical_blocks).any().item()):
                raise ValueError("one logical block maps to multiple physical blocks")
            block_rows.append(row_blocks)
            block_lens.append(block_count)
            selected_lens.append(selected_len)
            start_positions.append(int(start))

        max_blocks = max(block_lens)
        if block_table_width is not None:
            if max_blocks > block_table_width:
                raise ValueError("block_table_width is smaller than the live block table")
            max_blocks = block_table_width
        if workspace_key is not None:
            workspace = self._ensure_paged_decode_metadata_workspace(
                str(workspace_key),
                row_count=len(req_slots_list),
                block_table_width=max_blocks,
            )
            req_pool_indices = workspace["req_pool_indices"][: len(req_slots_list)]
            seq_lens_tensor = workspace["seq_lens"][: len(req_slots_list)]
            logical_seq_lens_tensor = workspace["logical_seq_lens"][: len(req_slots_list)]
            block_table_lens_tensor = workspace["block_table_lens"][: len(req_slots_list)]
            selected_start_positions_tensor = workspace["selected_start_positions"][
                : len(req_slots_list)
            ]
            block_tables = workspace["block_tables"][
                : len(req_slots_list),
                :max_blocks,
            ]
            block_tables.fill_(-1)
            req_pool_indices.copy_(
                torch.as_tensor(
                    req_slots_list,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            seq_lens_tensor.copy_(
                torch.as_tensor(
                    selected_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            logical_seq_lens_tensor.copy_(
                torch.as_tensor(
                    logical_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            block_table_lens_tensor.copy_(
                torch.as_tensor(
                    block_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            selected_start_positions_tensor.copy_(
                torch.as_tensor(
                    start_positions,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            for row_idx, row_blocks in enumerate(block_rows):
                block_tables[row_idx, : int(row_blocks.numel())].copy_(row_blocks)
            if out is not None:
                out_ws = workspace["out_cache_loc"][: len(req_slots_list)]
                out_ws.copy_(out)
                out_long = workspace["out_cache_loc_long"][: len(req_slots_list)]
                out_long.copy_(out.to(dtype=torch.long))
                out = out_ws
            return PagedDecodeBatchMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_tensor,
                logical_seq_lens=logical_seq_lens_tensor,
                out_cache_loc=out,
                block_tables=block_tables,
                block_table_lens=block_table_lens_tensor,
                selected_start_positions=selected_start_positions_tensor,
                block_size=block_size,
                slot_mapping=out_long,
                out_cache_loc_long=out_long,
                max_seq_len=max(selected_lens),
            )

        block_tables = torch.full(
            (len(req_slots_list), max_blocks),
            -1,
            dtype=torch.int32,
            device=self.req_to_token.device,
        )
        for row_idx, row_blocks in enumerate(block_rows):
            block_tables[row_idx, : int(row_blocks.numel())].copy_(row_blocks)

        return PagedDecodeBatchMetadata(
            req_pool_indices=torch.as_tensor(
                req_slots_list,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            seq_lens=torch.as_tensor(
                selected_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            logical_seq_lens=torch.as_tensor(
                logical_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            out_cache_loc=out,
            block_tables=block_tables,
            block_table_lens=torch.as_tensor(
                block_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            selected_start_positions=torch.as_tensor(
                start_positions,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            block_size=block_size,
            slot_mapping=out_long,
            out_cache_loc_long=out_long,
            max_seq_len=max(selected_lens),
        )

    def build_paged_decode_metadata_from_page_tables(
        self,
        req_slots: Iterable[int],
        page_tables: Iterable[dict[int, int]],
        *,
        block_size: int,
        block_table_width: int | None = None,
        seq_lens: Iterable[int] | None = None,
        out_cache_loc: Iterable[int] | Any | None = None,
        sliding_window: int | None = None,
        token_pool_capacity: int | None = None,
        workspace_key: str | None = None,
    ) -> PagedDecodeBatchMetadata:
        import torch

        req_slots_list = [int(slot) for slot in req_slots]
        page_table_list = [dict(table) for table in page_tables]
        if not req_slots_list:
            raise ValueError("paged decode metadata requires at least one request slot")
        if len(page_table_list) != len(req_slots_list):
            raise ValueError("page_tables length must match req_slots")
        if seq_lens is None:
            logical_lens = [int(self._lengths[slot].item()) for slot in req_slots_list]
        else:
            logical_lens = [int(length) for length in seq_lens]
        if len(logical_lens) != len(req_slots_list):
            raise ValueError("seq_lens length must match req_slots")
        if sliding_window is not None and int(sliding_window) < 1:
            raise ValueError("sliding_window must be >= 1 or None")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if block_table_width is not None:
            block_table_width = int(block_table_width)
            if block_table_width < 1:
                raise ValueError("block_table_width must be >= 1 or None")
        if token_pool_capacity is not None and int(token_pool_capacity) < 1:
            raise ValueError("token_pool_capacity must be >= 1 or None")
        if len(set(req_slots_list)) != len(req_slots_list):
            raise ValueError("req_slots must be unique")

        out = None
        out_long = None
        if out_cache_loc is not None:
            out = torch.as_tensor(
                out_cache_loc,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ).reshape(-1)
            if int(out.numel()) != len(req_slots_list):
                raise ValueError("out_cache_loc length must match req_slots")
            if bool((out < 0).any().item()):
                raise ValueError("out_cache_loc must be non-negative")
            if token_pool_capacity is not None and bool(
                (out >= int(token_pool_capacity)).any().item()
            ):
                raise ValueError("out_cache_loc exceeds token_pool_capacity")
            if int(torch.unique(out).numel()) != int(out.numel()):
                raise ValueError("out_cache_loc must be unique")
            out_long = out.to(dtype=torch.long)

        block_rows: list[list[int]] = []
        block_lens: list[int] = []
        selected_lens: list[int] = []
        start_positions: list[int] = []
        for row_idx, (req_slot, seq_len, page_table) in enumerate(
            zip(req_slots_list, logical_lens, page_table_list)
        ):
            self._validate_allocated_slot(req_slot)
            table_len = int(self._lengths[req_slot].item())
            if seq_len < 0 or seq_len > table_len:
                raise ValueError("seq_lens cannot exceed request table length")
            start = 0 if sliding_window is None else max(seq_len - int(sliding_window), 0)
            selected_len = int(seq_len) - int(start)
            if selected_len < 1:
                raise ValueError("paged token-slot rows must be non-empty")
            first_block = int(start) // block_size
            last_block = (int(start) + selected_len - 1) // block_size
            blocks: list[int] = []
            for logical_block in range(first_block, last_block + 1):
                if logical_block not in page_table:
                    raise ValueError("page table is missing a selected logical block")
                physical_block = int(page_table[logical_block])
                if physical_block < 0:
                    raise ValueError("physical page blocks must be non-negative")
                blocks.append(physical_block)
            if out is not None:
                out_slot = int(out[row_idx].item())
                final_logical_pos = int(seq_len) - 1
                final_block = final_logical_pos // block_size
                physical_block = int(page_table[final_block])
                expected_out = physical_block * block_size + (
                    final_logical_pos % block_size
                )
                if out_slot != expected_out:
                    raise ValueError("out_cache_loc must identify the final logical token")
            block_rows.append(blocks)
            block_lens.append(len(blocks))
            selected_lens.append(selected_len)
            start_positions.append(int(start))

        max_blocks = max(block_lens)
        if block_table_width is not None:
            if max_blocks > block_table_width:
                raise ValueError("block_table_width is smaller than the live block table")
            max_blocks = block_table_width
        if workspace_key is not None:
            workspace = self._ensure_paged_decode_metadata_workspace(
                str(workspace_key),
                row_count=len(req_slots_list),
                block_table_width=max_blocks,
            )
            req_pool_indices = workspace["req_pool_indices"][: len(req_slots_list)]
            seq_lens_tensor = workspace["seq_lens"][: len(req_slots_list)]
            logical_seq_lens_tensor = workspace["logical_seq_lens"][: len(req_slots_list)]
            block_table_lens_tensor = workspace["block_table_lens"][: len(req_slots_list)]
            selected_start_positions_tensor = workspace["selected_start_positions"][
                : len(req_slots_list)
            ]
            block_tables = workspace["block_tables"][
                : len(req_slots_list),
                :max_blocks,
            ]
            block_tables.fill_(-1)
            req_pool_indices.copy_(
                torch.as_tensor(
                    req_slots_list,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            seq_lens_tensor.copy_(
                torch.as_tensor(
                    selected_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            logical_seq_lens_tensor.copy_(
                torch.as_tensor(
                    logical_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            block_table_lens_tensor.copy_(
                torch.as_tensor(
                    block_lens,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            selected_start_positions_tensor.copy_(
                torch.as_tensor(
                    start_positions,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )
            )
            for row_idx, row_blocks in enumerate(block_rows):
                block_tables[row_idx, : len(row_blocks)].copy_(
                    torch.as_tensor(
                        row_blocks,
                        dtype=torch.int32,
                        device=self.req_to_token.device,
                    )
                )
            if out is not None:
                out_ws = workspace["out_cache_loc"][: len(req_slots_list)]
                out_ws.copy_(out)
                out_long = workspace["out_cache_loc_long"][: len(req_slots_list)]
                out_long.copy_(out.to(dtype=torch.long))
                out = out_ws
            return PagedDecodeBatchMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_tensor,
                logical_seq_lens=logical_seq_lens_tensor,
                out_cache_loc=out,
                block_tables=block_tables,
                block_table_lens=block_table_lens_tensor,
                selected_start_positions=selected_start_positions_tensor,
                block_size=block_size,
                slot_mapping=out_long,
                out_cache_loc_long=out_long,
                max_seq_len=max(selected_lens),
            )

        block_tables = torch.full(
            (len(req_slots_list), max_blocks),
            -1,
            dtype=torch.int32,
            device=self.req_to_token.device,
        )
        for row_idx, row_blocks in enumerate(block_rows):
            block_tables[row_idx, : len(row_blocks)] = torch.as_tensor(
                row_blocks,
                dtype=torch.int32,
                device=self.req_to_token.device,
            )
        return PagedDecodeBatchMetadata(
            req_pool_indices=torch.as_tensor(
                req_slots_list,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            seq_lens=torch.as_tensor(
                selected_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            logical_seq_lens=torch.as_tensor(
                logical_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            out_cache_loc=out,
            block_tables=block_tables,
            block_table_lens=torch.as_tensor(
                block_lens,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            selected_start_positions=torch.as_tensor(
                start_positions,
                dtype=torch.int32,
                device=self.req_to_token.device,
            ),
            block_size=block_size,
            slot_mapping=out_long,
            out_cache_loc_long=out_long,
            max_seq_len=max(selected_lens),
        )

    def build_paged_decode_metadata_from_page_table_tensor(
        self,
        req_slots: Iterable[int],
        page_table,
        *,
        block_size: int,
        block_table_width: int | None = None,
        seq_lens: Iterable[int] | None = None,
        out_cache_loc: Iterable[int] | Any | None = None,
        sliding_window: int | None = None,
        token_pool_capacity: int | None = None,
        workspace_key: str | None = None,
        validate: bool = True,
    ) -> PagedDecodeBatchMetadata:
        import torch

        req_slots_list = [int(slot) for slot in req_slots]
        if not req_slots_list:
            raise ValueError("paged decode metadata requires at least one request slot")
        if seq_lens is None:
            logical_lens = [int(self._lengths[slot].item()) for slot in req_slots_list]
        else:
            logical_lens = [int(length) for length in seq_lens]
        if len(logical_lens) != len(req_slots_list):
            raise ValueError("seq_lens length must match req_slots")
        if sliding_window is not None and int(sliding_window) < 1:
            raise ValueError("sliding_window must be >= 1 or None")
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if block_table_width is not None:
            block_table_width = int(block_table_width)
            if block_table_width < 1:
                raise ValueError("block_table_width must be >= 1 or None")
        if token_pool_capacity is not None and int(token_pool_capacity) < 1:
            raise ValueError("token_pool_capacity must be >= 1 or None")
        if len(set(req_slots_list)) != len(req_slots_list):
            raise ValueError("req_slots must be unique")
        if getattr(page_table, "ndim", None) != 2:
            raise ValueError("page_table must have shape [max_requests, max_pages]")
        if int(page_table.shape[0]) < max(req_slots_list) + 1:
            raise ValueError("page_table has fewer request rows than req_slots")

        starts: list[int] = []
        selected_lens: list[int] = []
        block_lens: list[int] = []
        first_blocks: list[int] = []
        for req_slot, seq_len in zip(req_slots_list, logical_lens):
            self._validate_allocated_slot(req_slot)
            table_len = int(self._lengths[req_slot].item())
            if seq_len < 0 or seq_len > table_len:
                raise ValueError("seq_lens cannot exceed request table length")
            start = 0 if sliding_window is None else max(seq_len - int(sliding_window), 0)
            selected_len = int(seq_len) - int(start)
            if selected_len < 1:
                raise ValueError("paged token-slot rows must be non-empty")
            first_block = int(start) // block_size
            last_block = (int(start) + selected_len - 1) // block_size
            block_count = last_block - first_block + 1
            starts.append(int(start))
            selected_lens.append(selected_len)
            first_blocks.append(first_block)
            block_lens.append(block_count)

        max_blocks = max(block_lens)
        if block_table_width is not None:
            if max_blocks > block_table_width:
                raise ValueError("block_table_width is smaller than the live block table")
            max_blocks = block_table_width
        max_required_page = max(
            first_block + block_count - 1
            for first_block, block_count in zip(first_blocks, block_lens)
        )
        if max_required_page >= int(page_table.shape[1]):
            raise ValueError("page_table has fewer logical pages than selected rows")

        device = self.req_to_token.device
        out = None
        out_long = None
        if out_cache_loc is not None:
            out = torch.as_tensor(out_cache_loc, dtype=torch.int32, device=device).reshape(-1)
            if int(out.numel()) != len(req_slots_list):
                raise ValueError("out_cache_loc length must match req_slots")
            if validate:
                if bool((out < 0).any().item()):
                    raise ValueError("out_cache_loc must be non-negative")
                if token_pool_capacity is not None and bool(
                    (out >= int(token_pool_capacity)).any().item()
                ):
                    raise ValueError("out_cache_loc exceeds token_pool_capacity")
                if int(torch.unique(out).numel()) != int(out.numel()):
                    raise ValueError("out_cache_loc must be unique")
            out_long = out.to(dtype=torch.long)

        req_slots_tensor = torch.as_tensor(req_slots_list, dtype=torch.long, device=device)
        first_blocks_tensor = torch.as_tensor(first_blocks, dtype=torch.long, device=device)
        offsets = torch.arange(max_blocks, dtype=torch.long, device=device)
        logical_blocks = first_blocks_tensor[:, None] + offsets[None, :]
        valid = offsets[None, :] < torch.as_tensor(
            block_lens,
            dtype=torch.long,
            device=device,
        )[:, None]
        gathered = page_table[
            req_slots_tensor[:, None],
            logical_blocks.clamp(min=0, max=int(page_table.shape[1]) - 1),
        ].to(dtype=torch.int32)
        if validate and bool((gathered[valid] < 0).any().item()):
            raise ValueError("page table is missing a selected logical block")
        filled_block_tables = torch.where(
            valid,
            gathered,
            torch.full_like(gathered, -1),
        )

        if out is not None and validate:
            final_positions = torch.as_tensor(logical_lens, dtype=torch.long, device=device) - 1
            final_blocks = final_positions // block_size
            final_offsets = final_positions % block_size
            final_physical_blocks = page_table[
                req_slots_tensor,
                final_blocks.clamp(min=0, max=int(page_table.shape[1]) - 1),
            ].to(dtype=torch.long)
            expected_out = final_physical_blocks * block_size + final_offsets
            if bool((out.to(dtype=torch.long) != expected_out).any().item()):
                raise ValueError("out_cache_loc must identify the final logical token")

        if workspace_key is not None:
            workspace = self._ensure_paged_decode_metadata_workspace(
                str(workspace_key),
                row_count=len(req_slots_list),
                block_table_width=max_blocks,
            )
            req_pool_indices = workspace["req_pool_indices"][: len(req_slots_list)]
            seq_lens_tensor = workspace["seq_lens"][: len(req_slots_list)]
            logical_seq_lens_tensor = workspace["logical_seq_lens"][: len(req_slots_list)]
            block_table_lens_tensor = workspace["block_table_lens"][: len(req_slots_list)]
            selected_start_positions_tensor = workspace["selected_start_positions"][
                : len(req_slots_list)
            ]
            block_tables = workspace["block_tables"][
                : len(req_slots_list),
                :max_blocks,
            ]
            req_pool_indices.copy_(
                torch.as_tensor(req_slots_list, dtype=torch.int32, device=device)
            )
            seq_lens_tensor.copy_(
                torch.as_tensor(selected_lens, dtype=torch.int32, device=device)
            )
            logical_seq_lens_tensor.copy_(
                torch.as_tensor(logical_lens, dtype=torch.int32, device=device)
            )
            block_table_lens_tensor.copy_(
                torch.as_tensor(block_lens, dtype=torch.int32, device=device)
            )
            selected_start_positions_tensor.copy_(
                torch.as_tensor(starts, dtype=torch.int32, device=device)
            )
            block_tables.copy_(filled_block_tables)
            if out is not None:
                out_ws = workspace["out_cache_loc"][: len(req_slots_list)]
                out_ws.copy_(out)
                out_long = workspace["out_cache_loc_long"][: len(req_slots_list)]
                out_long.copy_(out.to(dtype=torch.long))
                out = out_ws
            return PagedDecodeBatchMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_tensor,
                logical_seq_lens=logical_seq_lens_tensor,
                out_cache_loc=out,
                block_tables=block_tables,
                block_table_lens=block_table_lens_tensor,
                selected_start_positions=selected_start_positions_tensor,
                block_size=block_size,
                slot_mapping=out_long,
                out_cache_loc_long=out_long,
                max_seq_len=max(selected_lens),
            )

        return PagedDecodeBatchMetadata(
            req_pool_indices=torch.as_tensor(req_slots_list, dtype=torch.int32, device=device),
            seq_lens=torch.as_tensor(selected_lens, dtype=torch.int32, device=device),
            logical_seq_lens=torch.as_tensor(logical_lens, dtype=torch.int32, device=device),
            out_cache_loc=out,
            block_tables=filled_block_tables,
            block_table_lens=torch.as_tensor(block_lens, dtype=torch.int32, device=device),
            selected_start_positions=torch.as_tensor(starts, dtype=torch.int32, device=device),
            block_size=block_size,
            slot_mapping=out_long,
            out_cache_loc_long=out_long,
            max_seq_len=max(selected_lens),
        )

    def _ensure_paged_decode_metadata_workspace(
        self,
        key: str,
        *,
        row_count: int,
        block_table_width: int,
    ) -> dict[str, Any]:
        return self.decode_metadata_workspace.ensure_paged(
            key,
            device=self.req_to_token.device,
            row_count=row_count,
            block_table_width=block_table_width,
        )

    def _resolve_req_slot(self, req_id_or_slot: str | int) -> int:
        if isinstance(req_id_or_slot, str):
            return self._req_to_slot[req_id_or_slot]
        slot = int(req_id_or_slot)
        self._validate_allocated_slot(slot)
        return slot

    def _validate_allocated_slot(self, slot: int) -> None:
        if slot < 0 or slot >= self.max_requests or slot not in self._slot_to_req:
            raise KeyError(f"request slot {slot} is not allocated")


class TokenSlotAllocator:
    """Token-slot allocator for decode metadata before KV buffers are wired in."""

    def __init__(
        self,
        *,
        capacity: int | None = None,
        device: Any = "cpu",
        dtype: Any | None = None,
    ) -> None:
        import torch

        if capacity is not None and int(capacity) < 1:
            raise ValueError("capacity must be >= 1 or None")
        self.capacity = None if capacity is None else int(capacity)
        self.device = device
        self.dtype = dtype if dtype is not None else torch.int32
        self._next_slot = 0
        self._free_slots: list[int] = []
        self._allocated_slots: set[int] = set()
        self.high_watermark = 0

    def _alloc_slot_list(self, n: int) -> list[int]:
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        slots: list[int] = []
        while self._free_slots and len(slots) < n:
            slots.append(self._free_slots.pop(0))
        needed = n - len(slots)
        if needed:
            if self.capacity is not None and self._next_slot + needed > self.capacity:
                self._free_slots = sorted(slots + self._free_slots)
                raise RuntimeError("token slot allocator capacity exceeded")
            start = self._next_slot
            slots.extend(range(start, start + needed))
            self._next_slot += needed
        self._allocated_slots.update(slots)
        self.high_watermark = max(self.high_watermark, len(self._allocated_slots))
        return slots

    def alloc_slots(self, n: int):
        import torch

        slots = self._alloc_slot_list(n)
        return torch.as_tensor(slots, dtype=self.dtype, device=self.device)

    def alloc_slots_with_ids(self, n: int):
        import torch

        slots = self._alloc_slot_list(n)
        return torch.as_tensor(slots, dtype=self.dtype, device=self.device), slots

    def alloc_page_block_with_ids(self, block_size: int) -> tuple[int, list[int]]:
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.capacity is None:
            start = self._next_slot
            offset = start % block_size
            if offset:
                padding = block_size - offset
                self._free_slots.extend(range(start, start + padding))
                self._free_slots.sort()
                start += padding
                self._next_slot = start
            slots = list(range(start, start + block_size))
            self._next_slot += block_size
        else:
            start = 0
            free = set(self._free_slots)
            while start + block_size <= self.capacity:
                slots = list(range(start, start + block_size))
                if all(
                    slot not in self._allocated_slots
                    and (slot >= self._next_slot or slot in free)
                    for slot in slots
                ):
                    break
                start += block_size
            else:
                raise RuntimeError("token slot allocator page capacity exceeded")
            if start > self._next_slot:
                self._free_slots.extend(range(self._next_slot, start))
            self._next_slot = max(self._next_slot, start + block_size)
            slot_set = set(slots)
            self._free_slots = [slot for slot in self._free_slots if slot not in slot_set]
            self._free_slots.sort()
        self._allocated_slots.update(slots)
        self.high_watermark = max(self.high_watermark, len(self._allocated_slots))
        return start // block_size, slots

    def free_slots(self, slots: Iterable[int] | Any) -> None:
        values = _slot_values_to_list(slots)
        freed: list[int] = []
        for value in values:
            slot = int(value)
            if slot not in self._allocated_slots:
                raise KeyError(f"token slot {slot} is not allocated")
            self._allocated_slots.remove(slot)
            freed.append(slot)
        self._free_slots.extend(freed)
        self._free_slots.sort()

    @property
    def allocated_count(self) -> int:
        return len(self._allocated_slots)

    @property
    def free_count(self) -> int:
        return len(self._free_slots)

    @property
    def next_slot(self) -> int:
        return int(self._next_slot)


class TokenKVPool:
    """Token-granularity KV storage with optional layer-level KV sharing."""

    def __init__(
        self,
        *,
        capacity: int,
        layer_specs: Iterable[TokenKVLayerSpec],
        dtype: Any | None = None,
        device: Any = "cpu",
        defer_buffer_allocation: bool = False,
        validate_slot_writes: bool = True,
    ) -> None:
        import torch

        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.device = device
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.layer_specs = {int(spec.layer_id): spec for spec in layer_specs}
        if not self.layer_specs:
            raise ValueError("TokenKVPool requires at least one layer spec")
        self._target_layers = {
            layer_id: self._resolve_target_layer(layer_id)
            for layer_id in self.layer_specs
        }
        self._validate_shared_layer_specs()
        self._buffers: dict[int, tuple[Any, Any]] = {}
        self._defer_buffer_allocation = bool(defer_buffer_allocation)
        self.validate_slot_writes = bool(validate_slot_writes)
        self._attention_output_buffers: dict[tuple[Any, ...], Any] = {}
        self._attention_split_workspaces: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self.kv_set_calls = 0
        self.kv_set_tokens = 0
        self.kv_set_index_copy_calls = 0
        self.kv_set_slice_copy_calls = 0
        self.kv_set_triton_copy_calls = 0
        self.kv_set_triton_fallback_calls = 0
        self.kv_set_wall_s = 0.0
        self.kv_set_index_copy_wall_s = 0.0
        self.kv_set_slice_copy_wall_s = 0.0
        self.kv_set_triton_copy_wall_s = 0.0
        if not self._defer_buffer_allocation:
            for layer_id in self.layer_specs:
                if self._target_layers[layer_id] == layer_id:
                    self._allocate_layer_buffer(layer_id)
        self._free_slots = list(range(self.capacity))
        self._allocated_slots: set[int] = set()
        self.high_watermark = 0

    def _alloc_slot_list(self, n: int) -> list[int]:
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > len(self._free_slots):
            raise RuntimeError("token KV pool capacity exceeded")
        slots = self._free_slots[:n]
        del self._free_slots[:n]
        self._allocated_slots.update(slots)
        self.high_watermark = max(self.high_watermark, len(self._allocated_slots))
        return slots

    def alloc_slots(self, n: int):
        import torch

        slots = self._alloc_slot_list(n)
        return torch.as_tensor(slots, dtype=torch.int32, device=self.device)

    def alloc_slots_with_ids(self, n: int):
        import torch

        slots = self._alloc_slot_list(n)
        return torch.as_tensor(slots, dtype=torch.int32, device=self.device), slots

    def alloc_page_block_with_ids(self, block_size: int) -> tuple[int, list[int]]:
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        free = set(self._free_slots)
        start = 0
        while start + block_size <= self.capacity:
            slots = list(range(start, start + block_size))
            if all(slot in free for slot in slots):
                break
            start += block_size
        else:
            raise RuntimeError("token KV pool page capacity exceeded")
        slot_set = set(slots)
        self._free_slots = [slot for slot in self._free_slots if slot not in slot_set]
        self._allocated_slots.update(slots)
        self.high_watermark = max(self.high_watermark, len(self._allocated_slots))
        return start // block_size, slots

    def free_slots(self, slots: Iterable[int] | Any) -> None:
        values = _slot_values_to_list(slots)
        freed: list[int] = []
        for value in values:
            slot = int(value)
            if slot not in self._allocated_slots:
                raise KeyError(f"token slot {slot} is not allocated")
            self._allocated_slots.remove(slot)
            freed.append(slot)
        self._free_slots.extend(freed)
        self._free_slots.sort()

    def set_kv(self, layer_id: int, slot_ids, key_states, value_states) -> None:
        import torch

        timing_enabled = _token_pool_timing_enabled()
        set_start = time.perf_counter() if timing_enabled else 0.0
        layer_id = int(layer_id)
        target = self._target_layers[layer_id]
        if target != layer_id:
            raise ValueError(f"layer {layer_id} shares KV with layer {target} and cannot write KV")
        self.ensure_layer_allocated(layer_id)
        keys, values = self._buffers[target]
        key_states = self._normalize_kv_input(key_states, keys, "key_states")
        value_states = self._normalize_kv_input(value_states, values, "value_states")
        slot_span = self._host_contiguous_slot_span(slot_ids)
        if slot_span is None:
            slots = torch.as_tensor(
                slot_ids,
                dtype=torch.long,
                device=keys.device,
            ).reshape(-1)
            slot_count = int(slots.numel())
        else:
            start_slot, slot_count = slot_span
            slots = None
        if slot_count != int(key_states.shape[0]) or slot_count != int(value_states.shape[0]):
            raise ValueError("slot_ids length must match key/value batch")
        if slot_span is not None and (
            int(start_slot) < 0 or int(start_slot) + int(slot_count) > self.capacity
        ):
            raise IndexError("token slot span exceeds token KV pool capacity")
        if self.validate_slot_writes:
            if slot_span is None:
                self._validate_allocated_token_slots(slots)
            else:
                self._validate_allocated_token_slot_span(start_slot, slot_count)
        self.kv_set_calls += 1
        self.kv_set_tokens += int(slot_count)
        if slot_span is not None:
            end_slot = int(start_slot) + int(slot_count)
            copy_start = time.perf_counter() if timing_enabled else 0.0
            keys[start_slot:end_slot].copy_(key_states)
            values[start_slot:end_slot].copy_(value_states)
            self.kv_set_slice_copy_calls += 1
            if timing_enabled:
                now = time.perf_counter()
                self.kv_set_wall_s += now - set_start
                self.kv_set_slice_copy_wall_s += now - copy_start
            return
        copy_start = time.perf_counter() if timing_enabled else 0.0
        if (
            bool(getattr(keys, "is_cuda", False))
            and bool(getattr(slots, "is_cuda", False))
            and _token_pool_kv_store_triton_enabled()
        ):
            try:
                from wkvm.runner.gemma_token_pool_triton import token_pool_store_kv

                token_pool_store_kv(
                    key_states,
                    value_states,
                    keys,
                    values,
                    slots,
                )
                self.kv_set_triton_copy_calls += 1
                if timing_enabled:
                    now = time.perf_counter()
                    self.kv_set_wall_s += now - set_start
                    self.kv_set_triton_copy_wall_s += now - copy_start
                return
            except Exception:
                self.kv_set_triton_fallback_calls += 1
        keys.index_copy_(0, slots, key_states)
        values.index_copy_(0, slots, value_states)
        self.kv_set_index_copy_calls += 1
        if timing_enabled:
            now = time.perf_counter()
            self.kv_set_wall_s += now - set_start
            self.kv_set_index_copy_wall_s += now - copy_start

    def get_kv_buffer(self, layer_id: int):
        target = self._target_layers[int(layer_id)]
        if target not in self._buffers:
            raise RuntimeError(f"token KV buffer for layer {target} has not been allocated")
        return self._buffers[target]

    def gather_kv(self, layer_id: int, kv_indices):
        import torch

        keys, values = self.get_kv_buffer(layer_id)
        indices = torch.as_tensor(kv_indices, dtype=torch.long, device=keys.device).reshape(-1)
        return keys.index_select(0, indices), values.index_select(0, indices)

    def attention_output_buffer(
        self,
        *,
        batch: int,
        query_heads: int,
        head_dim: int,
        dtype,
        device,
    ):
        import torch

        shape = (int(batch), 1, int(query_heads), int(head_dim))
        key = (str(device), dtype, shape)
        output = self._attention_output_buffers.get(key)
        if output is None:
            output = torch.empty(shape, dtype=dtype, device=device)
            self._attention_output_buffers[key] = output
        return output

    def attention_split_workspace(
        self,
        *,
        batch: int,
        kv_heads: int,
        max_splits: int,
        block_groups: int,
        head_dim: int,
        device,
    ):
        import torch

        batch = int(batch)
        kv_heads = int(kv_heads)
        max_splits = int(max_splits)
        block_groups = int(block_groups)
        head_dim = int(head_dim)
        if min(batch, kv_heads, max_splits, block_groups, head_dim) < 1:
            raise ValueError("attention split workspace dimensions must be >= 1")
        stats_shape = (batch, kv_heads, max_splits, block_groups)
        acc_shape = (batch, kv_heads, max_splits, block_groups, head_dim)
        key = (str(device), torch.float32, stats_shape, acc_shape)
        workspace = self._attention_split_workspaces.get(key)
        if workspace is None:
            workspace = (
                torch.empty(stats_shape, dtype=torch.float32, device=device),
                torch.empty(stats_shape, dtype=torch.float32, device=device),
                torch.empty(acc_shape, dtype=torch.float32, device=device),
            )
            self._attention_split_workspaces[key] = workspace
        return workspace

    def target_layer(self, layer_id: int) -> int:
        return self._target_layers[int(layer_id)]

    def ensure_layer_allocated(self, layer_id: int) -> None:
        target = self._target_layers[int(layer_id)]
        if target not in self._buffers:
            self._allocate_layer_buffer(target)

    @property
    def allocated_count(self) -> int:
        return len(self._allocated_slots)

    @property
    def free_count(self) -> int:
        return len(self._free_slots)

    @property
    def next_slot(self) -> int:
        return self.capacity - len(self._free_slots)

    def state_bytes(self) -> int:
        total = 0
        for keys, values in self._buffers.values():
            total += keys.numel() * keys.element_size()
            total += values.numel() * values.element_size()
        return total

    @property
    def allocated_layer_count(self) -> int:
        return len(self._buffers)

    def _allocate_layer_buffer(self, layer_id: int) -> None:
        import torch

        layer_id = int(layer_id)
        spec = self.layer_specs[layer_id]
        layer_dtype = spec.dtype if spec.dtype is not None else self.dtype
        shape = (self.capacity, int(spec.num_kv_heads), int(spec.head_dim))
        self._buffers[layer_id] = (
            torch.empty(shape, dtype=layer_dtype, device=self.device),
            torch.empty(shape, dtype=layer_dtype, device=self.device),
        )

    def _resolve_target_layer(self, layer_id: int) -> int:
        seen: set[int] = set()
        current = int(layer_id)
        while True:
            if current in seen:
                raise ValueError(f"KV sharing cycle includes layer {current}")
            seen.add(current)
            spec = self.layer_specs.get(current)
            if spec is None:
                raise KeyError(f"missing TokenKVLayerSpec for layer {current}")
            target = spec.kv_share_target_layer
            if target is None:
                return current
            current = int(target)

    def _validate_shared_layer_specs(self) -> None:
        for layer_id, target in self._target_layers.items():
            if layer_id == target:
                continue
            spec = self.layer_specs[layer_id]
            target_spec = self.layer_specs[target]
            spec_dtype = spec.dtype if spec.dtype is not None else self.dtype
            target_dtype = (
                target_spec.dtype if target_spec.dtype is not None else self.dtype
            )
            if (
                int(spec.num_kv_heads) != int(target_spec.num_kv_heads)
                or int(spec.head_dim) != int(target_spec.head_dim)
                or spec_dtype != target_dtype
            ):
                raise ValueError(
                    f"layer {layer_id} shared KV shape does not match target layer {target}"
                )

    def _validate_allocated_token_slots(self, slots) -> None:
        try:
            import torch

            if (
                bool(getattr(slots, "is_cuda", False))
                and torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            ):
                return
        except Exception:
            pass
        for value in slots.detach().cpu().reshape(-1).tolist():
            slot = int(value)
            if slot < 0 or slot >= self.capacity or slot not in self._allocated_slots:
                raise KeyError(f"token slot {slot} is not allocated")

    def _validate_allocated_token_slot_span(self, start_slot: int, count: int) -> None:
        start_slot = int(start_slot)
        count = int(count)
        for slot in range(start_slot, start_slot + count):
            if slot < 0 or slot >= self.capacity or slot not in self._allocated_slots:
                raise KeyError(f"token slot {slot} is not allocated")

    @staticmethod
    def _host_contiguous_slot_span(slot_ids) -> tuple[int, int] | None:
        if isinstance(slot_ids, int):
            return int(slot_ids), 1
        if isinstance(slot_ids, range):
            length = len(slot_ids)
            if length < 1:
                return None
            if slot_ids.step != 1:
                return None
            return int(slot_ids.start), int(length)
        if isinstance(slot_ids, (list, tuple)):
            values = [int(value) for value in slot_ids]
        else:
            try:
                import torch

                if torch.is_tensor(slot_ids):
                    if bool(getattr(slot_ids, "is_cuda", False)):
                        return None
                    values = [
                        int(value)
                        for value in slot_ids.detach().cpu().reshape(-1).tolist()
                    ]
                else:
                    return None
            except ImportError:
                return None
        if not values:
            return None
        first = int(values[0])
        for offset, value in enumerate(values):
            if int(value) != first + int(offset):
                return None
        return first, len(values)

    @staticmethod
    def _normalize_kv_input(tensor, buffer, name: str):
        if tensor.ndim == 4 and int(tensor.shape[2]) == 1:
            tensor = tensor[:, :, 0, :]
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [N, H, D] or [N, H, 1, D]")
        if tuple(tensor.shape[1:]) != tuple(buffer.shape[1:]):
            raise ValueError(f"{name} shape {tuple(tensor.shape)} does not match KV buffer")
        if tensor.dtype != buffer.dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} does not match KV buffer {buffer.dtype}")
        if tensor.device != buffer.device:
            raise ValueError(f"{name} device {tensor.device} does not match KV buffer {buffer.device}")
        return tensor.contiguous()
