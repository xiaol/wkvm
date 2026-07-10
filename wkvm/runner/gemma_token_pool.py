"""Decode-side token KV pool primitives for native Gemma serving.

These classes model the minimal vLLM/SGLang-style substrate WKVM needs before
the dense padded-KV decode path can be replaced by paged/token-pool attention.
They do not run attention by themselves; they own request-to-token mappings,
per-layer KV buffers, and flattened decode metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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


def token_pool_triton_decode_plan_from_metadata(
    metadata: Any | None,
    max_seq_len: Any | None = None,
) -> TokenPoolTritonDecodePlan:
    plan = getattr(metadata, "triton_decode_plan", None)
    if plan is not None:
        return plan
    if max_seq_len is None:
        max_seq_len = getattr(metadata, "max_seq_len", None)
    return build_token_pool_triton_decode_plan(max_seq_len)


@dataclass(frozen=True)
class TokenPoolLayerDecodeBinding:
    metadata: DecodeBatchMetadata | None
    paged_metadata: PagedDecodeBatchMetadata | None


@dataclass(frozen=True)
class TokenPoolRequestPageStateSnapshot:
    req_id: str
    req_slot: int | None
    page_table: dict[int, int]
    owned_slots: frozenset[int]
    block_table_snapshot: Any | None = None


@dataclass
class TokenPoolDecodeReservation:
    req_id: str
    req_slot: int
    token_slot: int
    token_slot_tensor: Any
    previous_length: int
    full_attention_token_slot: int | None = None
    persistent_full_attention_row: bool = False
    page_state_snapshot: TokenPoolRequestPageStateSnapshot | None = None


@dataclass(frozen=True)
class TokenPoolRequestPrefixClearResult:
    dropped_slots: tuple[int, ...] = ()
    released_slots: tuple[int, ...] = ()
    expired_page_slots: tuple[int, ...] = ()
    invalidated_full_attention_rows: int = 0


class TokenPoolAttentionWorkspace:
    """Backend-owned reusable scratch buffers for token-pool attention."""

    def __init__(self) -> None:
        self._output_buffers: dict[tuple[Any, ...], Any] = {}
        self._split_workspaces: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}

    def attention_output_buffer(
        self,
        *,
        batch: int,
        query_heads: int,
        head_dim: int,
        dtype: Any,
        device: Any,
    ) -> Any:
        import torch

        shape = (int(batch), 1, int(query_heads), int(head_dim))
        key = (str(device), dtype, shape)
        output = self._output_buffers.get(key)
        if output is None:
            output = torch.empty(shape, dtype=dtype, device=device)
            self._output_buffers[key] = output
        return output

    def attention_split_workspace(
        self,
        *,
        batch: int,
        kv_heads: int,
        max_splits: int,
        block_groups: int,
        head_dim: int,
        device: Any,
    ) -> tuple[Any, Any, Any]:
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
        workspace = self._split_workspaces.get(key)
        if workspace is None:
            workspace = (
                torch.empty(stats_shape, dtype=torch.float32, device=device),
                torch.empty(stats_shape, dtype=torch.float32, device=device),
                torch.empty(acc_shape, dtype=torch.float32, device=device),
            )
            self._split_workspaces[key] = workspace
        return workspace


@dataclass(frozen=True)
class TokenPoolAttentionKernelDispatch:
    kind: str
    metadata: Any
    max_seq_len: Any | None = None
    split_size: int | None = None
    min_splits: int | None = None
    max_splits: int | None = None
    split_skipped_by_min_splits: bool = False

    @property
    def is_paged(self) -> bool:
        return self.kind.startswith("paged")

    @property
    def is_split(self) -> bool:
        return self.kind.endswith("split")


@dataclass(frozen=True)
class TokenPoolAttentionDispatchContext:
    layer_idx: int | None
    flat_metadata: DecodeBatchMetadata | None
    paged_metadata: Any | None
    token_kv_pool: Any | None
    kv_buffer_owner: Any | None = None
    workspace_owner: Any | None = None
    kv_write_owner: Any | None = None

    @property
    def has_flat_metadata(self) -> bool:
        return (
            getattr(self.flat_metadata, "kv_indptr", None) is not None
            and getattr(self.flat_metadata, "kv_indices", None) is not None
        )

    @property
    def has_paged_metadata(self) -> bool:
        return getattr(self.paged_metadata, "block_tables", None) is not None

    def kv_buffers_for_attention(self) -> tuple[Any, Any] | None:
        kv_buffers_for_attention = getattr(
            self.kv_buffer_owner,
            "kv_buffers_for_attention",
            None,
        )
        if kv_buffers_for_attention is not None:
            return kv_buffers_for_attention()
        if self.layer_idx is None or self.token_kv_pool is None:
            return None
        get_kv_buffer = getattr(self.token_kv_pool, "get_kv_buffer", None)
        if get_kv_buffer is None:
            return None
        return get_kv_buffer(int(self.layer_idx))

    def attention_output_buffer(
        self,
        *,
        batch: int,
        query_heads: int,
        head_dim: int,
        dtype: Any,
        device: Any,
    ) -> Any | None:
        output_buffer = getattr(self.workspace_owner, "attention_output_buffer", None)
        if output_buffer is None:
            return None
        return output_buffer(
            batch=batch,
            query_heads=query_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def attention_split_workspace(
        self,
        *,
        batch: int,
        kv_heads: int,
        max_splits: int,
        block_groups: int,
        head_dim: int,
        device: Any,
    ) -> Any | None:
        split_workspace = getattr(self.workspace_owner, "attention_split_workspace", None)
        if split_workspace is None:
            return None
        return split_workspace(
            batch=batch,
            kv_heads=kv_heads,
            max_splits=max_splits,
            block_groups=block_groups,
            head_dim=head_dim,
            device=device,
        )

    def store_current_kv(self, key_states: Any, value_states: Any) -> Any | None:
        if key_states is None or value_states is None:
            return None
        store_current_kv = getattr(self.kv_write_owner, "store_current_kv", None)
        if store_current_kv is not None:
            return store_current_kv(key_states, value_states)
        out_cache_loc = _metadata_out_cache_loc_for_write(
            self.flat_metadata if self.flat_metadata is not None else self.paged_metadata
        )
        if self.layer_idx is None or self.token_kv_pool is None or out_cache_loc is None:
            return None
        set_kv = getattr(self.token_kv_pool, "set_kv", None)
        if set_kv is None:
            return None
        set_kv(
            int(self.layer_idx),
            out_cache_loc,
            key_states,
            value_states,
        )
        return out_cache_loc

    def reference_decode_inputs(self) -> tuple[Any, Any, int]:
        if not self.has_flat_metadata:
            raise RuntimeError(
                "token-pool flat decode metadata is required for reference fallback"
            )
        if self.token_kv_pool is None or self.layer_idx is None:
            raise RuntimeError(
                "token-pool KV pool and layer index are required for reference fallback"
            )
        return self.flat_metadata, self.token_kv_pool, int(self.layer_idx)

    def triton_split_plan_for_metadata(
        self,
        metadata: Any | None,
        max_seq_len: Any | None = None,
    ) -> tuple[bool, int, int, int | None]:
        plan = token_pool_triton_decode_plan_from_metadata(metadata, max_seq_len)
        return (
            bool(plan.should_split),
            int(plan.split_size),
            int(plan.min_splits),
            None if plan.max_splits is None else int(plan.max_splits),
        )

    def select_triton_dispatch(
        self,
        *,
        paged_enabled: bool,
        split_enabled: bool,
        paged_split_enabled: bool,
    ) -> TokenPoolAttentionKernelDispatch:
        if paged_enabled and self.has_paged_metadata:
            metadata = self.paged_metadata
            max_seq_len = getattr(metadata, "max_seq_len", None)
            if paged_split_enabled:
                should_split, split_size, min_splits, max_splits = (
                    self.triton_split_plan_for_metadata(metadata, max_seq_len)
                )
                if should_split:
                    return TokenPoolAttentionKernelDispatch(
                        kind="paged_split",
                        metadata=metadata,
                        max_seq_len=max_seq_len,
                        split_size=split_size,
                        min_splits=min_splits,
                        max_splits=max_splits,
                    )
                return TokenPoolAttentionKernelDispatch(
                    kind="paged",
                    metadata=metadata,
                    split_skipped_by_min_splits=True,
                )
            return TokenPoolAttentionKernelDispatch(kind="paged", metadata=metadata)

        if not self.has_flat_metadata:
            raise RuntimeError(
                "token-pool flat decode metadata is required when paged "
                "Triton decode is unavailable"
            )

        metadata = self.flat_metadata
        max_seq_len = getattr(metadata, "max_seq_len", None)
        if split_enabled:
            should_split, split_size, min_splits, max_splits = (
                self.triton_split_plan_for_metadata(metadata, max_seq_len)
            )
            if should_split:
                return TokenPoolAttentionKernelDispatch(
                    kind="flat_split",
                    metadata=metadata,
                    max_seq_len=max_seq_len,
                    split_size=split_size,
                    min_splits=min_splits,
                    max_splits=max_splits,
                )
            return TokenPoolAttentionKernelDispatch(
                kind="flat",
                metadata=metadata,
                split_skipped_by_min_splits=True,
            )
        return TokenPoolAttentionKernelDispatch(kind="flat", metadata=metadata)


@dataclass
class TokenPoolFullAttentionRow:
    row_slots: list[int]
    owned_slots: list[int]
    append_slots: list[int] = field(default_factory=list)
    page_aligned: bool = False


@dataclass(frozen=True)
class TokenPoolFullAttentionRowAppend:
    row: TokenPoolFullAttentionRow
    full_token_slot: int
    reused_existing_row: bool


@dataclass(frozen=True)
class TokenPoolFullAttentionPreparedDecodeRow:
    row_chunks: TokenSlotRowChunks
    full_token_slot: int
    materialized_slots: Any
    materialized_slot_ids: list[int]
    materialized_slots_long: Any
    persistent_row: TokenPoolFullAttentionRow | None = None
    paged_row: list[int] | None = None
    reused_existing_row: bool = False
    rebuilt_persistent_row: bool = False
    appended_existing_row: bool = False
    invalidated_existing_rows: int = 0


@dataclass(frozen=True)
class TokenPoolFullAttentionPreparedBatch:
    metadata: DecodeBatchMetadata
    paged_metadata: PagedDecodeBatchMetadata | None
    prepared_rows: tuple[TokenPoolFullAttentionPreparedDecodeRow, ...]
    req_ids: tuple[str, ...]
    req_slots: tuple[int, ...]
    out_cache_loc: tuple[int, ...]
    logical_seq_lens: tuple[int, ...]

    @property
    def invalidated_existing_rows(self) -> int:
        return sum(int(row.invalidated_existing_rows) for row in self.prepared_rows)

    @property
    def reused_existing_rows(self) -> int:
        return sum(1 for row in self.prepared_rows if row.reused_existing_row)

    @property
    def appended_existing_rows(self) -> int:
        return sum(1 for row in self.prepared_rows if row.appended_existing_row)

    @property
    def rebuilt_persistent_rows(self) -> int:
        return sum(1 for row in self.prepared_rows if row.rebuilt_persistent_row)


class TokenPoolFullAttentionRowManager:
    """Backend-owned lifecycle for materialized full-attention token rows."""

    def __init__(
        self,
        *,
        allocator: Any,
        block_size: int = 16,
    ) -> None:
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.allocator = allocator
        self.block_size = block_size
        self.transient_slots: dict[str, list[int]] = {}
        self.rows: dict[str, TokenPoolFullAttentionRow] = {}

    def allocate_page_aligned_row_slots(
        self,
        start_position: int,
        min_slots: int,
    ) -> tuple[Any, list[int], list[int]]:
        allocator = self.allocator
        alloc_page = getattr(allocator, "alloc_page_block_with_ids", None)
        if alloc_page is None:
            slots_tensor, slot_ids = allocator.alloc_slots_with_ids(min_slots)
            return slots_tensor, slot_ids, list(slot_ids)
        import torch

        block_size = self.block_size
        start_position = int(start_position)
        min_slots = int(min_slots)
        if start_position < 0:
            raise ValueError("start_position must be non-negative")
        if min_slots < 1:
            raise ValueError("min_slots must be >= 1")
        if start_position % block_size:
            raise RuntimeError(
                "page-aligned full-attention row append must start at a page boundary"
            )
        end_position = start_position + min_slots
        rounded_end = ((end_position + block_size - 1) // block_size) * block_size
        slots: list[int] = []
        owned_slots: list[int] = []
        first_block = start_position // block_size
        last_block = (rounded_end - 1) // block_size
        for logical_block in range(first_block, last_block + 1):
            _physical_block, block_slots = alloc_page(block_size)
            owned_slots.extend(int(slot) for slot in block_slots)
            block_start = max(start_position, logical_block * block_size)
            block_end = min(rounded_end, (logical_block + 1) * block_size)
            for logical_pos in range(block_start, block_end):
                slots.append(int(block_slots[logical_pos % block_size]))
        return (
            torch.as_tensor(slots, dtype=torch.int32, device=allocator.device),
            slots,
            owned_slots,
        )

    def clear(self, req_ids: str | Iterable[Any]) -> None:
        req_id_list = [req_ids] if isinstance(req_ids, str) else list(req_ids)
        slots: list[int] = []
        for req_id in req_id_list:
            req_key = str(req_id)
            slots.extend(self.transient_slots.pop(req_key, []))
            persistent_row = self.rows.pop(req_key, None)
            if persistent_row is not None:
                slots.extend(persistent_row.owned_slots)
        if slots:
            self.allocator.free_slots(slots)

    def invalidate(self, req_ids: str | Iterable[Any]) -> int:
        req_id_list = [req_ids] if isinstance(req_ids, str) else list(req_ids)
        invalidated = sum(
            1 for req_id in req_id_list if str(req_id) in self.rows
        )
        self.clear(req_id_list)
        return invalidated

    def invalidate_containing(self, slots: Iterable[int] | Any) -> int:
        slot_set = {int(slot) for slot in _slot_values_to_list(slots)}
        if not slot_set:
            return 0
        req_ids = [
            req_id
            for req_id, row in self.rows.items()
            if any(int(slot) in slot_set for slot in row.row_slots)
        ]
        if not req_ids:
            return 0
        return self.invalidate(req_ids)

    def append_existing_row(
        self,
        req_id: str,
        *,
        append_reserve_slots: int,
    ) -> TokenPoolFullAttentionRowAppend | None:
        row = self.rows.get(str(req_id))
        if row is None:
            return None
        append_reserve_slots = max(1, int(append_reserve_slots))
        if row.append_slots:
            full_token_slot = int(row.append_slots.pop(0))
        elif row.page_aligned:
            _, append_slot_list, append_owned_slots = (
                self.allocate_page_aligned_row_slots(
                    len(row.row_slots),
                    append_reserve_slots,
                )
            )
            full_token_slot = int(append_slot_list[0])
            row.owned_slots.extend(append_owned_slots)
            row.append_slots.extend(append_slot_list[1:])
        else:
            _, append_slot_list = self.allocator.alloc_slots_with_ids(
                append_reserve_slots
            )
            full_token_slot = int(append_slot_list[0])
            row.owned_slots.extend(append_slot_list)
            row.append_slots.extend(append_slot_list[1:])
        row.row_slots.append(full_token_slot)
        return TokenPoolFullAttentionRowAppend(
            row=row,
            full_token_slot=full_token_slot,
            reused_existing_row=True,
        )

    def start_persistent_row(
        self,
        req_id: str,
        *,
        materialized_slots: Iterable[int] | Any,
        append_reserve_slots: int,
        page_aligned: bool,
    ) -> TokenPoolFullAttentionRowAppend:
        req_key = str(req_id)
        materialized_slot_list = _slot_values_to_list(materialized_slots)
        append_reserve_slots = max(1, int(append_reserve_slots))
        row = TokenPoolFullAttentionRow(
            row_slots=list(materialized_slot_list),
            owned_slots=list(materialized_slot_list),
            page_aligned=bool(page_aligned),
        )
        if row.page_aligned:
            _, append_slot_list, append_owned_slots = (
                self.allocate_page_aligned_row_slots(
                    len(row.row_slots),
                    append_reserve_slots,
                )
            )
            row.owned_slots.extend(append_owned_slots)
        else:
            _, append_slot_list = self.allocator.alloc_slots_with_ids(
                append_reserve_slots
            )
            row.owned_slots.extend(append_slot_list)
        full_token_slot = int(append_slot_list[0])
        row.row_slots.append(full_token_slot)
        row.append_slots.extend(append_slot_list[1:])
        self.rows[req_key] = row
        return TokenPoolFullAttentionRowAppend(
            row=row,
            full_token_slot=full_token_slot,
            reused_existing_row=False,
        )

    def start_page_aligned_persistent_row(
        self,
        req_id: str,
        *,
        materialized_width: int,
        append_reserve_slots: int,
    ) -> tuple[Any, list[int], TokenPoolFullAttentionRowAppend]:
        materialized_width = max(0, int(materialized_width))
        row_slots_tensor, row_slot_list, row_owned_slots = (
            self.allocate_page_aligned_row_slots(
                0,
                materialized_width + max(1, int(append_reserve_slots)),
            )
        )
        materialized_slot_list = row_slot_list[:materialized_width]
        row = TokenPoolFullAttentionRow(
            row_slots=list(materialized_slot_list),
            owned_slots=list(row_owned_slots),
            page_aligned=True,
        )
        full_token_slot = int(row_slot_list[materialized_width])
        row.row_slots.append(full_token_slot)
        row.append_slots.extend(row_slot_list[materialized_width + 1:])
        self.rows[str(req_id)] = row
        return (
            row_slots_tensor[:materialized_width],
            materialized_slot_list,
            TokenPoolFullAttentionRowAppend(
                row=row,
                full_token_slot=full_token_slot,
                reused_existing_row=False,
            ),
        )

    def record_transient_materialized_slots(
        self,
        req_id: str,
        materialized_slots: Iterable[int] | Any,
    ) -> None:
        self.transient_slots[str(req_id)] = _slot_values_to_list(materialized_slots)

    def prepare_decode_row(
        self,
        req_id: str,
        *,
        materialized_width: int,
        decode_token_slot: int,
        decode_token_slot_tensor: Any,
        persistent_rows: bool,
        build_paged_rows: bool,
        append_reserve_slots: int,
        device: Any,
    ) -> TokenPoolFullAttentionPreparedDecodeRow:
        import torch

        req_key = str(req_id)
        materialized_width = max(0, int(materialized_width))
        append_reserve_slots = max(1, int(append_reserve_slots))
        invalidated = 0
        existing_row = self.rows.get(req_key) if persistent_rows else None
        if existing_row is not None and build_paged_rows and not existing_row.page_aligned:
            invalidated += self.invalidate([req_key])
            existing_row = None

        if existing_row is not None:
            if len(existing_row.row_slots) == materialized_width:
                append = self.append_existing_row(
                    req_key,
                    append_reserve_slots=append_reserve_slots,
                )
                if append is None:
                    raise RuntimeError("existing full-attention row disappeared")
                empty_slots = torch.empty(0, dtype=torch.int32, device=device)
                paged_row = (
                    list(append.row.row_slots)
                    if build_paged_rows and append.row.page_aligned
                    else None
                )
                return TokenPoolFullAttentionPreparedDecodeRow(
                    row_chunks=TokenSlotRowChunks(
                        (
                            torch.as_tensor(
                                append.row.row_slots,
                                dtype=torch.int32,
                                device=device,
                            ),
                        ),
                        trusted=True,
                    ),
                    full_token_slot=append.full_token_slot,
                    materialized_slots=empty_slots,
                    materialized_slot_ids=[],
                    materialized_slots_long=empty_slots.to(dtype=torch.long),
                    persistent_row=append.row,
                    paged_row=paged_row,
                    reused_existing_row=True,
                    appended_existing_row=True,
                    invalidated_existing_rows=invalidated,
                )
            invalidated += self.invalidate([req_key])

        persistent_row: TokenPoolFullAttentionRow | None = None
        if persistent_rows and build_paged_rows:
            materialized_slots, materialized_slot_ids, append = (
                self.start_page_aligned_persistent_row(
                    req_key,
                    materialized_width=materialized_width,
                    append_reserve_slots=append_reserve_slots,
                )
            )
            persistent_row = append.row
            full_token_slot = append.full_token_slot
            decode_slot = torch.as_tensor(
                [full_token_slot],
                dtype=torch.int32,
                device=device,
            )
        else:
            if materialized_width:
                materialized_slots, materialized_slot_ids = (
                    self.allocator.alloc_slots_with_ids(materialized_width)
                )
            else:
                materialized_slots = torch.empty(0, dtype=torch.int32, device=device)
                materialized_slot_ids = []
            if persistent_rows:
                append = self.start_persistent_row(
                    req_key,
                    materialized_slots=materialized_slot_ids,
                    append_reserve_slots=append_reserve_slots,
                    page_aligned=False,
                )
                persistent_row = append.row
                full_token_slot = append.full_token_slot
                decode_slot = torch.as_tensor(
                    [full_token_slot],
                    dtype=torch.int32,
                    device=device,
                )
            else:
                self.record_transient_materialized_slots(req_key, materialized_slot_ids)
                full_token_slot = int(decode_token_slot)
                decode_slot = self._normalized_decode_slot_tensor(
                    decode_token_slot_tensor,
                    full_token_slot=full_token_slot,
                    device=device,
                )

        materialized_slots_long = materialized_slots.to(dtype=torch.long)
        paged_row = (
            list(persistent_row.row_slots)
            if build_paged_rows
            and persistent_row is not None
            and persistent_row.page_aligned
            else None
        )
        return TokenPoolFullAttentionPreparedDecodeRow(
            row_chunks=TokenSlotRowChunks(
                (materialized_slots, decode_slot),
                trusted=True,
            ),
            full_token_slot=full_token_slot,
            materialized_slots=materialized_slots,
            materialized_slot_ids=list(materialized_slot_ids),
            materialized_slots_long=materialized_slots_long,
            persistent_row=persistent_row,
            paged_row=paged_row,
            rebuilt_persistent_row=bool(persistent_rows),
            invalidated_existing_rows=invalidated,
        )

    @staticmethod
    def _normalized_decode_slot_tensor(
        decode_slot: Any,
        *,
        full_token_slot: int,
        device: Any,
    ) -> Any:
        import torch

        decode_device = getattr(decode_slot, "device", None)
        decode_device_matches = (
            decode_device is not None
            and torch.device(decode_device) == torch.device(device)
        )
        if (
            hasattr(decode_slot, "numel")
            and int(decode_slot.numel()) == 1
            and getattr(decode_slot, "dtype", None) == torch.int32
            and decode_device_matches
        ):
            return decode_slot
        return torch.as_tensor(
            [int(full_token_slot)],
            dtype=torch.int32,
            device=device,
        )


@dataclass(frozen=True)
class TokenPoolAttentionBinding:
    layer_idx: int | None
    metadata: DecodeBatchMetadata | None
    paged_metadata: PagedDecodeBatchMetadata | None
    kv_pool: Any | None
    attention_workspace: Any | None = None

    @property
    def has_kv_pool(self) -> bool:
        return self.kv_pool is not None

    def out_cache_loc_for_write(self) -> Any | None:
        return _metadata_out_cache_loc_for_write(self.metadata)

    def has_write_location(self) -> bool:
        return self.out_cache_loc_for_write() is not None

    def flat_metadata_for_attention(self) -> DecodeBatchMetadata | None:
        if getattr(self.metadata, "block_tables", None) is not None:
            return None
        return self.metadata

    def paged_metadata_for_attention(self) -> Any | None:
        if self.paged_metadata is not None:
            return self.paged_metadata
        if getattr(self.metadata, "block_tables", None) is not None:
            return self.metadata
        return None

    def attention_metadata_for_dispatch(
        self,
    ) -> tuple[DecodeBatchMetadata | None, Any | None]:
        return self.flat_metadata_for_attention(), self.paged_metadata_for_attention()

    def should_use_decode_attention(
        self,
        *,
        attention_mask_present: bool = False,
        query_seq_len: int | None = None,
    ) -> bool:
        return _token_pool_decode_attention_enabled(
            layer_idx=self.layer_idx,
            metadata=self.metadata,
            kv_pool=self.kv_pool,
            attention_mask_present=attention_mask_present,
            query_seq_len=query_seq_len,
            out_cache_loc=self.out_cache_loc_for_write(),
        )

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

    def attention_output_buffer(
        self,
        *,
        batch: int,
        query_heads: int,
        head_dim: int,
        dtype: Any,
        device: Any,
    ) -> Any | None:
        if self.attention_workspace is not None:
            return self.attention_workspace.attention_output_buffer(
                batch=batch,
                query_heads=query_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
        if self.kv_pool is None:
            return None
        output_buffer = getattr(self.kv_pool, "attention_output_buffer", None)
        if output_buffer is None:
            return None
        return output_buffer(
            batch=batch,
            query_heads=query_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def attention_split_workspace(
        self,
        *,
        batch: int,
        kv_heads: int,
        max_splits: int,
        block_groups: int,
        head_dim: int,
        device: Any,
    ) -> Any | None:
        if self.attention_workspace is not None:
            return self.attention_workspace.attention_split_workspace(
                batch=batch,
                kv_heads=kv_heads,
                max_splits=max_splits,
                block_groups=block_groups,
                head_dim=head_dim,
                device=device,
            )
        if self.kv_pool is None:
            return None
        workspace = getattr(self.kv_pool, "attention_split_workspace", None)
        if workspace is None:
            return None
        return workspace(
            batch=batch,
            kv_heads=kv_heads,
            max_splits=max_splits,
            block_groups=block_groups,
            head_dim=head_dim,
            device=device,
        )

    def kv_buffers_for_attention(self) -> tuple[Any, Any] | None:
        if self.layer_idx is None or self.kv_pool is None:
            return None
        get_kv_buffer = getattr(self.kv_pool, "get_kv_buffer", None)
        if get_kv_buffer is None:
            return None
        return get_kv_buffer(int(self.layer_idx))


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
        should_use_decode_attention = getattr(
            binding,
            "should_use_decode_attention",
            None,
        )
        if should_use_decode_attention is not None:
            use_decode_attention = bool(
                should_use_decode_attention(
                    attention_mask_present=attention_mask_present,
                    query_seq_len=query_seq_len,
                )
            )
        else:
            use_decode_attention = _token_pool_decode_attention_enabled(
                layer_idx=bound_layer_idx,
                metadata=metadata,
                kv_pool=kv_pool,
                attention_mask_present=attention_mask_present,
                query_seq_len=query_seq_len,
                out_cache_loc=_binding_out_cache_loc_for_write(binding, metadata),
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

    def decode_attention_enabled(self) -> bool:
        return bool(self.use_decode_attention)

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

    def attention_output_buffer(
        self,
        *,
        batch: int,
        query_heads: int,
        head_dim: int,
        dtype: Any,
        device: Any,
    ) -> Any | None:
        output_buffer = getattr(self.binding, "attention_output_buffer", None)
        if output_buffer is not None:
            return output_buffer(
                batch=batch,
                query_heads=query_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
        if self.kv_pool is None:
            return None
        output_buffer = getattr(self.kv_pool, "attention_output_buffer", None)
        if output_buffer is None:
            return None
        return output_buffer(
            batch=batch,
            query_heads=query_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def kv_buffers_for_attention(self) -> tuple[Any, Any] | None:
        kv_buffers_for_attention = getattr(
            self.binding,
            "kv_buffers_for_attention",
            None,
        )
        if kv_buffers_for_attention is not None:
            return kv_buffers_for_attention()
        if self.layer_idx is None or self.kv_pool is None:
            return None
        get_kv_buffer = getattr(self.kv_pool, "get_kv_buffer", None)
        if get_kv_buffer is None:
            return None
        return get_kv_buffer(int(self.layer_idx))

    def flat_metadata_for_attention(self) -> DecodeBatchMetadata | None:
        flat_metadata_for_attention = getattr(
            self.binding,
            "flat_metadata_for_attention",
            None,
        )
        if flat_metadata_for_attention is not None:
            return flat_metadata_for_attention()
        if getattr(self.metadata, "block_tables", None) is not None:
            return None
        return self.metadata

    def paged_metadata_for_attention(self) -> Any | None:
        paged_metadata_for_attention = getattr(
            self.binding,
            "paged_metadata_for_attention",
            None,
        )
        if paged_metadata_for_attention is not None:
            return paged_metadata_for_attention()
        if self.paged_metadata is not None:
            return self.paged_metadata
        if getattr(self.metadata, "block_tables", None) is not None:
            return self.metadata
        return None

    def attention_metadata_for_dispatch(
        self,
    ) -> tuple[DecodeBatchMetadata | None, Any | None]:
        metadata_for_dispatch = getattr(
            self.binding,
            "attention_metadata_for_dispatch",
            None,
        )
        if metadata_for_dispatch is not None:
            return metadata_for_dispatch()
        return self.flat_metadata_for_attention(), self.paged_metadata_for_attention()

    def attention_dispatch_context(
        self,
        *,
        decode_metadata: Any | None = None,
        paged_decode_metadata: Any | None = None,
        token_kv_pool: Any | None = None,
        layer_idx: int | None = None,
    ) -> TokenPoolAttentionDispatchContext:
        flat_metadata, paged_metadata = self.attention_metadata_for_dispatch()
        resolved_layer_idx = self.layer_idx if self.layer_idx is not None else layer_idx
        resolved_kv_pool = self.kv_pool if self.kv_pool is not None else token_kv_pool
        return TokenPoolAttentionDispatchContext(
            layer_idx=resolved_layer_idx,
            flat_metadata=flat_metadata,
            paged_metadata=paged_metadata,
            token_kv_pool=resolved_kv_pool,
            kv_buffer_owner=self,
            workspace_owner=self,
            kv_write_owner=self,
        )

    def attention_split_workspace(
        self,
        *,
        batch: int,
        kv_heads: int,
        max_splits: int,
        block_groups: int,
        head_dim: int,
        device: Any,
    ) -> Any | None:
        workspace = getattr(self.binding, "attention_split_workspace", None)
        if workspace is not None:
            return workspace(
                batch=batch,
                kv_heads=kv_heads,
                max_splits=max_splits,
                block_groups=block_groups,
                head_dim=head_dim,
                device=device,
            )
        if self.kv_pool is None:
            return None
        workspace = getattr(self.kv_pool, "attention_split_workspace", None)
        if workspace is None:
            return None
        return workspace(
            batch=batch,
            kv_heads=kv_heads,
            max_splits=max_splits,
            block_groups=block_groups,
            head_dim=head_dim,
            device=device,
        )


@dataclass(frozen=True)
class TokenPoolAttentionCall:
    plan: Any | None = None
    attention_kwargs: dict[str, Any] = field(default_factory=dict)
    decode_attention_enabled: bool = False
    key_states_for_write: Any | None = None
    value_states_for_write: Any | None = None

    def current_key_states(
        self,
        key_states: Any,
        *,
        is_kv_shared_layer: bool = False,
    ) -> Any | None:
        if self.decode_attention_enabled and not bool(is_kv_shared_layer):
            return key_states
        return None

    def current_value_states(
        self,
        value_states: Any,
        *,
        is_kv_shared_layer: bool = False,
    ) -> Any | None:
        if self.decode_attention_enabled and not bool(is_kv_shared_layer):
            return value_states
        return None

    def should_update_dense_cache(
        self,
        *,
        has_past_key_values: bool,
        is_kv_shared_layer: bool = False,
    ) -> bool:
        return (
            bool(has_past_key_values)
            and not bool(is_kv_shared_layer)
            and not self.decode_attention_enabled
        )

    def bind_layer_kv(
        self,
        key_states: Any,
        value_states: Any,
        *,
        has_past_key_values: bool,
        is_kv_shared_layer: bool = False,
    ) -> "TokenPoolAttentionLayerKVBinding":
        attention_call = self.with_current_kv(
            key_states,
            value_states,
            is_kv_shared_layer=is_kv_shared_layer,
        )
        return TokenPoolAttentionLayerKVBinding(
            attention_call=attention_call,
            should_update_dense_cache=attention_call.should_update_dense_cache(
                has_past_key_values=has_past_key_values,
                is_kv_shared_layer=is_kv_shared_layer,
            ),
        )

    def with_current_kv(
        self,
        key_states: Any,
        value_states: Any,
        *,
        is_kv_shared_layer: bool = False,
    ) -> "TokenPoolAttentionCall":
        return replace(
            self,
            key_states_for_write=self.current_key_states(
                key_states,
                is_kv_shared_layer=is_kv_shared_layer,
            ),
            value_states_for_write=self.current_value_states(
                value_states,
                is_kv_shared_layer=is_kv_shared_layer,
            ),
        )

    def backend_decode_kwargs(self) -> dict[str, Any]:
        layer_idx = self.attention_kwargs.get("layer_idx")
        return {
            "decode_metadata": self.attention_kwargs.get("decode_metadata"),
            "paged_decode_metadata": self.attention_kwargs.get(
                "paged_decode_metadata"
            ),
            "token_kv_pool": self.attention_kwargs.get("token_kv_pool"),
            "layer_idx": None if layer_idx is None else int(layer_idx),
            "token_pool_plan": self.plan,
            "current_key_states": self.key_states_for_write,
            "current_value_states": self.value_states_for_write,
        }

    def backend_dispatch_context(self) -> TokenPoolAttentionDispatchContext:
        call_kwargs = self.backend_decode_kwargs()
        return build_token_pool_attention_dispatch_context(
            token_pool_plan=call_kwargs.get("token_pool_plan"),
            decode_metadata=call_kwargs.get("decode_metadata"),
            paged_decode_metadata=call_kwargs.get("paged_decode_metadata"),
            token_kv_pool=call_kwargs.get("token_kv_pool"),
            layer_idx=call_kwargs.get("layer_idx"),
        )

    def current_kv_for_backend(self) -> tuple[Any | None, Any | None]:
        return self.key_states_for_write, self.value_states_for_write


@dataclass(frozen=True)
class TokenPoolAttentionLayerKVBinding:
    attention_call: TokenPoolAttentionCall
    should_update_dense_cache: bool


def token_pool_attention_plan_kwargs(token_pool_plan: Any) -> dict[str, Any]:
    attention_kwargs = getattr(token_pool_plan, "attention_kwargs", None)
    if attention_kwargs is not None:
        return dict(attention_kwargs())
    return {
        "decode_metadata": getattr(token_pool_plan, "metadata", None),
        "paged_decode_metadata": getattr(token_pool_plan, "paged_metadata", None),
        "token_kv_pool": getattr(token_pool_plan, "kv_pool", None),
        "layer_idx": getattr(token_pool_plan, "layer_idx", None),
    }


def token_pool_attention_plan_decode_enabled(token_pool_plan: Any) -> bool:
    decode_attention_enabled = getattr(
        token_pool_plan,
        "decode_attention_enabled",
        None,
    )
    if decode_attention_enabled is not None:
        return bool(decode_attention_enabled())
    return bool(getattr(token_pool_plan, "use_decode_attention", False))


def build_token_pool_attention_call(
    *,
    token_pool_plan: Any | None = None,
    decode_metadata: Any | None = None,
    paged_decode_metadata: Any | None = None,
    token_kv_pool: Any | None = None,
    layer_idx: int | None = None,
    attention_mask_present: bool = False,
    query_seq_len: int | None = None,
) -> TokenPoolAttentionCall:
    if token_pool_plan is not None:
        return TokenPoolAttentionCall(
            plan=token_pool_plan,
            attention_kwargs=token_pool_attention_plan_kwargs(token_pool_plan),
            decode_attention_enabled=token_pool_attention_plan_decode_enabled(
                token_pool_plan
            ),
        )
    return TokenPoolAttentionCall(
        plan=None,
        attention_kwargs={
            "decode_metadata": decode_metadata,
            "paged_decode_metadata": paged_decode_metadata,
            "token_kv_pool": token_kv_pool,
            "layer_idx": layer_idx,
        },
        decode_attention_enabled=(
            decode_metadata is not None
            and token_kv_pool is not None
            and layer_idx is not None
            and not bool(attention_mask_present)
            and _query_seq_len_is_one(query_seq_len)
        ),
    )


def build_token_pool_attention_dispatch_context(
    *,
    token_pool_plan: Any | None = None,
    decode_metadata: Any | None = None,
    paged_decode_metadata: Any | None = None,
    token_kv_pool: Any | None = None,
    layer_idx: int | None = None,
) -> TokenPoolAttentionDispatchContext:
    attention_dispatch_context = getattr(
        token_pool_plan,
        "attention_dispatch_context",
        None,
    )
    if attention_dispatch_context is not None:
        context = attention_dispatch_context(
            decode_metadata=decode_metadata,
            paged_decode_metadata=paged_decode_metadata,
            token_kv_pool=token_kv_pool,
            layer_idx=layer_idx,
        )
        if context is not None:
            return context

    if token_pool_plan is not None:
        metadata_for_dispatch = getattr(
            token_pool_plan,
            "attention_metadata_for_dispatch",
            None,
        )
        if metadata_for_dispatch is not None:
            flat_metadata, paged_metadata = metadata_for_dispatch()
        else:
            plan_metadata = getattr(token_pool_plan, "metadata", None)
            plan_paged_metadata = getattr(token_pool_plan, "paged_metadata", None)
            flat_metadata, paged_metadata = _normalize_attention_dispatch_metadata(
                plan_metadata,
                plan_paged_metadata,
            )
        return TokenPoolAttentionDispatchContext(
            layer_idx=getattr(token_pool_plan, "layer_idx", layer_idx),
            flat_metadata=flat_metadata,
            paged_metadata=paged_metadata,
            token_kv_pool=getattr(token_pool_plan, "kv_pool", token_kv_pool),
            kv_buffer_owner=token_pool_plan,
            workspace_owner=token_pool_plan,
            kv_write_owner=token_pool_plan,
        )

    flat_metadata, paged_metadata = _normalize_attention_dispatch_metadata(
        decode_metadata,
        paged_decode_metadata,
    )
    return TokenPoolAttentionDispatchContext(
        layer_idx=layer_idx,
        flat_metadata=flat_metadata,
        paged_metadata=paged_metadata,
        token_kv_pool=token_kv_pool,
        kv_buffer_owner=token_kv_pool,
        workspace_owner=token_kv_pool,
        kv_write_owner=token_kv_pool,
    )


def _normalize_attention_dispatch_metadata(
    decode_metadata: Any | None,
    paged_decode_metadata: Any | None,
) -> tuple[DecodeBatchMetadata | None, Any | None]:
    paged_metadata = paged_decode_metadata
    if paged_metadata is None and getattr(decode_metadata, "block_tables", None) is not None:
        paged_metadata = decode_metadata
    flat_metadata = (
        None
        if getattr(decode_metadata, "block_tables", None) is not None
        else decode_metadata
    )
    return flat_metadata, paged_metadata


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


def _query_seq_len_is_one(query_seq_len: int | None) -> bool:
    try:
        return int(query_seq_len) == 1
    except (TypeError, ValueError):
        return False


def _token_pool_decode_attention_enabled(
    *,
    layer_idx: int | None,
    metadata: Any | None,
    kv_pool: Any | None,
    attention_mask_present: bool,
    query_seq_len: int | None,
    out_cache_loc: Any | None = None,
) -> bool:
    if metadata is None or kv_pool is None or layer_idx is None:
        return False
    if bool(attention_mask_present):
        return False
    if out_cache_loc is None:
        out_cache_loc = getattr(metadata, "out_cache_loc", None)
    if out_cache_loc is None:
        return False
    return _query_seq_len_is_one(query_seq_len)


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
    attention_workspace = getattr(token_pool_decode, "attention_workspace", None)
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
            attention_workspace=attention_workspace,
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
        attention_workspace=attention_workspace,
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


def resolve_token_pool_attention_call(
    token_pool_decode: Any | None,
    layer_idx: int | None,
    layer_type: str | None,
    *,
    attention_mask_present: bool = False,
    query_seq_len: int | None = None,
) -> TokenPoolAttentionCall:
    if token_pool_decode is None or layer_idx is None or layer_type is None:
        return TokenPoolAttentionCall()
    plan = resolve_token_pool_attention_plan(
        token_pool_decode,
        layer_idx,
        layer_type,
        attention_mask_present=attention_mask_present,
        query_seq_len=query_seq_len,
    )
    return build_token_pool_attention_call(token_pool_plan=plan)


@dataclass(frozen=True)
class TokenPoolDecodeContext:
    metadata_by_layer_type: dict[str, DecodeBatchMetadata]
    kv_pool: Any | None = None
    attention_workspace: Any | None = None
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

    def covered_decode_layer_types(self) -> frozenset[str]:
        return token_pool_decode_covered_layer_types(self)

    def attention_mask_for_decode(self, attention_mask: Any) -> Any:
        return token_pool_attention_mask_for_decode(attention_mask, self)

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
                attention_workspace=self.attention_workspace,
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
                attention_workspace=self.attention_workspace,
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
                attention_workspace=self.attention_workspace,
            )
        return TokenPoolAttentionBinding(
            layer_idx=layer_idx,
            metadata=metadata,
            paged_metadata=paged_metadata,
            kv_pool=token_kv_pool,
            attention_workspace=self.attention_workspace,
        )


@dataclass(frozen=True)
class TokenPoolDecodeBatchState:
    """Backend-owned metadata for the current decode batch."""

    metadata_by_layer_type: dict[str, DecodeBatchMetadata]
    metadata_by_layer_id: dict[int, DecodeBatchMetadata] | None = None
    paged_metadata_by_layer_type: dict[str, PagedDecodeBatchMetadata] | None = None
    paged_metadata_by_layer_id: dict[int, PagedDecodeBatchMetadata] | None = None
    covered_layer_types: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata_by_layer_type",
            {str(key): value for key, value in self.metadata_by_layer_type.items()},
        )
        if self.metadata_by_layer_id is not None:
            object.__setattr__(
                self,
                "metadata_by_layer_id",
                {int(key): value for key, value in self.metadata_by_layer_id.items()},
            )
        if self.paged_metadata_by_layer_type is not None:
            object.__setattr__(
                self,
                "paged_metadata_by_layer_type",
                {
                    str(key): value
                    for key, value in self.paged_metadata_by_layer_type.items()
                },
            )
        if self.paged_metadata_by_layer_id is not None:
            object.__setattr__(
                self,
                "paged_metadata_by_layer_id",
                {
                    int(key): value
                    for key, value in self.paged_metadata_by_layer_id.items()
                },
            )
        object.__setattr__(
            self,
            "covered_layer_types",
            frozenset(str(layer_type) for layer_type in self.covered_layer_types),
        )

    def build_context(
        self,
        *,
        kv_pool: Any | None,
        attention_workspace: Any | None,
        layer_id_metadata_only_types: frozenset[str] = frozenset(),
    ) -> TokenPoolDecodeContext:
        return TokenPoolDecodeContext(
            metadata_by_layer_type=self.metadata_by_layer_type,
            kv_pool=kv_pool,
            attention_workspace=attention_workspace,
            metadata_by_layer_id=self.metadata_by_layer_id,
            paged_metadata_by_layer_type=self.paged_metadata_by_layer_type,
            paged_metadata_by_layer_id=self.paged_metadata_by_layer_id,
            covered_layer_types=self.covered_layer_types,
            layer_id_metadata_only_types=layer_id_metadata_only_types,
        )


@dataclass(frozen=True)
class TokenPoolPreparedDecodeBatch:
    """Backend-owned decode transaction state for one prepared model step."""

    reservations: tuple[TokenPoolDecodeReservation, ...]
    state: TokenPoolDecodeBatchState | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reservations", tuple(self.reservations))

    @property
    def covered_layer_types(self) -> frozenset[str]:
        if self.state is None:
            return frozenset()
        return self.state.covered_layer_types

    def build_context(
        self,
        *,
        kv_pool: Any | None,
        attention_workspace: Any | None,
        layer_id_metadata_only_types: frozenset[str] = frozenset(),
    ) -> TokenPoolDecodeContext | None:
        if not self.reservations or self.state is None or kv_pool is None:
            return None
        return self.state.build_context(
            kv_pool=kv_pool,
            attention_workspace=attention_workspace,
            layer_id_metadata_only_types=layer_id_metadata_only_types,
        )


@dataclass(frozen=True)
class TokenPoolDecodeCommitResult:
    invalidated_full_attention_rows: int = 0
    cleared_prefix_slots: tuple[int, ...] = ()
    released_prefix_slots: tuple[int, ...] = ()
    expired_page_slots: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenPoolDecodeDiscardResult:
    freed_token_slots: tuple[int, ...] = ()
    restored_page_slots: tuple[int, ...] = ()


def token_pool_decode_covered_layer_types(
    token_pool_decode: Any | None,
) -> frozenset[str]:
    if token_pool_decode is None:
        return frozenset()
    if getattr(token_pool_decode, "kv_pool", None) is None:
        return frozenset()
    explicit_covered = getattr(token_pool_decode, "covered_layer_types", None)
    if explicit_covered is not None:
        return frozenset(str(layer_type) for layer_type in explicit_covered)
    metadata_by_layer_type = getattr(token_pool_decode, "metadata_by_layer_type", None)
    if not metadata_by_layer_type:
        return frozenset()
    covered = {
        str(layer_type)
        for layer_type, metadata in metadata_by_layer_type.items()
        if metadata is not None and getattr(metadata, "out_cache_loc", None) is not None
    }
    return frozenset(covered)


def token_pool_attention_mask_for_decode(
    attention_mask: Any,
    token_pool_decode: Any | None,
) -> Any:
    if token_pool_decode is None or not isinstance(attention_mask, dict):
        return attention_mask
    explicit_covered = getattr(token_pool_decode, "covered_layer_types", None)
    if explicit_covered is not None:
        covered = {str(layer_type) for layer_type in explicit_covered}
        adjusted = dict(attention_mask)
        changed = False
        for layer_type in ("full_attention", "sliding_attention"):
            if layer_type not in covered:
                continue
            if layer_type in adjusted:
                adjusted[layer_type] = None
                changed = True
        return adjusted if changed else attention_mask
    metadata_by_layer_type = getattr(token_pool_decode, "metadata_by_layer_type", None)
    if not metadata_by_layer_type:
        return attention_mask
    adjusted = None
    for layer_type in ("full_attention", "sliding_attention"):
        metadata = metadata_by_layer_type.get(layer_type)
        if metadata is None or getattr(metadata, "out_cache_loc", None) is None:
            continue
        if attention_mask.get(layer_type) is None:
            continue
        if adjusted is None:
            adjusted = dict(attention_mask)
        adjusted[layer_type] = None
    if adjusted is None:
        return attention_mask
    return adjusted


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


def _token_pool_paged_metadata_requested() -> bool:
    return any(
        _env_flag(name)
        for name in (
            "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON",
            "WKVM_TOKEN_POOL_BUILD_PAGED_METADATA",
        )
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

    def copy_compatible_from(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> dict[str, int]:
        alias_stats = self._copy_aliasing_workspace_compatible(token_pool_decode)
        if alias_stats is not None:
            return alias_stats
        compatibility_error = self.replay_compatibility_error(token_pool_decode)
        if compatibility_error is not None:
            raise ValueError(
                "token-pool cuda graph metadata incompatible: "
                f"{compatibility_error}"
            )
        return self.copy_from(token_pool_decode)

    def _copy_aliasing_workspace_compatible(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> dict[str, int] | None:
        if self.context is None or token_pool_decode is None:
            return None
        if self.context.kv_pool is not token_pool_decode.kv_pool:
            return None
        if self.context.covered_layer_types != token_pool_decode.covered_layer_types:
            return None
        if (
            self.context.layer_id_metadata_only_types
            != token_pool_decode.layer_id_metadata_only_types
        ):
            return None
        stats = {
            "cuda_graph_metadata_tensor_copies": 0,
            "cuda_graph_metadata_tensor_copy_skips": 0,
            "cuda_graph_metadata_alias_fastpath_metadata_skips": 0,
        }
        checked_metadata: set[tuple[int, int]] = set()
        for dst_group, src_group in (
            (
                self.context.metadata_by_layer_type,
                token_pool_decode.metadata_by_layer_type,
            ),
            (
                self.context.metadata_by_layer_id or {},
                token_pool_decode.metadata_by_layer_id or {},
            ),
            (
                self.context.paged_metadata_by_layer_type or {},
                token_pool_decode.paged_metadata_by_layer_type or {},
            ),
            (
                self.context.paged_metadata_by_layer_id or {},
                token_pool_decode.paged_metadata_by_layer_id or {},
            ),
        ):
            if not self._metadata_group_aliases_same_workspace(
                dst_group,
                src_group,
                checked_metadata=checked_metadata,
                stats=stats,
            ):
                return None
        return stats

    def replay_compatibility_error(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> str | None:
        if self.context is None:
            if token_pool_decode is not None:
                return "graph was captured without token-pool metadata"
            return None
        if token_pool_decode is None:
            return "graph token-pool metadata is required for replay"
        if self.context.kv_pool is not token_pool_decode.kv_pool:
            return "kv_pool changed"
        if self.context.covered_layer_types != token_pool_decode.covered_layer_types:
            return "covered_layer_types changed"
        if (
            self.context.layer_id_metadata_only_types
            != token_pool_decode.layer_id_metadata_only_types
        ):
            return "layer_id_metadata_only_types changed"

        expected = TokenPoolDecodeGraphSignatureTracker.shape_signature(self.context)
        actual = TokenPoolDecodeGraphSignatureTracker.shape_signature(token_pool_decode)
        if expected == actual:
            return None
        reasons = TokenPoolDecodeGraphSignatureTracker.shape_mismatch_reasons(
            expected,
            actual,
        )
        return ", ".join(str(reason) for reason in reasons)

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

    @classmethod
    def _metadata_group_aliases_same_workspace(
        cls,
        dst_group: dict[Any, Any],
        src_group: dict[Any, Any],
        *,
        checked_metadata: set[tuple[int, int]],
        stats: dict[str, int],
    ) -> bool:
        if set(dst_group) != set(src_group):
            return False
        for key in dst_group:
            metadata_pair = (id(dst_group[key]), id(src_group[key]))
            if metadata_pair in checked_metadata:
                stats["cuda_graph_metadata_alias_fastpath_metadata_skips"] = (
                    stats.get("cuda_graph_metadata_alias_fastpath_metadata_skips", 0)
                    + 1
                )
                continue
            if not cls._decode_metadata_aliases_same_workspace(
                dst_group[key],
                src_group[key],
                stats=stats,
            ):
                return False
            checked_metadata.add(metadata_pair)
        return True

    @classmethod
    def _decode_metadata_aliases_same_workspace(
        cls,
        dst,
        src,
        *,
        stats: dict[str, int],
    ) -> bool:
        if int(getattr(dst, "block_size", -1)) != int(getattr(src, "block_size", -1)):
            return False
        if getattr(dst, "max_seq_len", None) != getattr(src, "max_seq_len", None):
            return False
        if getattr(dst, "triton_decode_plan", None) != getattr(
            src,
            "triton_decode_plan",
            None,
        ):
            return False
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
            if not cls._decode_metadata_tensor_aliases_same_workspace(
                getattr(dst, name, None),
                getattr(src, name, None),
                stats=stats,
            ):
                return False
        return True

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
    def _decode_metadata_tensor_aliases_same_workspace(
        dst,
        src,
        *,
        stats: dict[str, int],
    ) -> bool:
        if dst is None or src is None:
            return dst is src
        if tuple(dst.shape) != tuple(src.shape):
            return False
        if dst.dtype != src.dtype:
            return False
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
        if not same_storage:
            return False
        stats["cuda_graph_metadata_tensor_copy_skips"] = (
            stats.get("cuda_graph_metadata_tensor_copy_skips", 0) + 1
        )
        return True

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


class TokenPoolDecodeGraphMetadata:
    """Backend-facing handle for graph-stable token-pool decode metadata."""

    def __init__(self, buffer: TokenPoolDecodeGraphBuffer) -> None:
        self._buffer = buffer

    @classmethod
    def capture(
        cls,
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        clone_tensors: bool = False,
    ) -> "TokenPoolDecodeGraphMetadata":
        return cls(
            TokenPoolDecodeGraphBuffer.capture(
                token_pool_decode,
                clone_tensors=clone_tensors,
            )
        )

    @property
    def context(self) -> TokenPoolDecodeContext | None:
        return self._buffer.context

    def copy_from(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> dict[str, int]:
        return self._buffer.copy_from(token_pool_decode)

    def copy_compatible_from(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> dict[str, int]:
        return self._buffer.copy_compatible_from(token_pool_decode)

    def replay_compatibility_error(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> str | None:
        return self._buffer.replay_compatibility_error(token_pool_decode)


@dataclass(frozen=True)
class TokenPoolDecodeGraphSignatureUpdate:
    candidate_batches: int = 0
    static_shape_starts: int = 0
    static_shape_reuses: int = 0
    shape_mismatches: int = 0
    shape_mismatch_reasons: dict[str, int] = field(default_factory=dict)


class TokenPoolDecodeGraphSignatureTracker:
    """Backend-owned static-shape signatures for token-pool CUDA graphs."""

    def __init__(self) -> None:
        self.signatures: dict[tuple[str, ...], dict[str, Any]] = {}

    def clear(self) -> None:
        self.signatures.clear()

    def discard(self, key: tuple[str, ...]) -> None:
        self.signatures.pop(tuple(str(part) for part in key), None)

    def discard_touching(self, req_ids: Iterable[Any]) -> int:
        req_id_set = {str(req_id) for req_id in req_ids}
        discarded = 0
        for key in list(self.signatures):
            if any(req_id in req_id_set for req_id in key):
                self.signatures.pop(key, None)
                discarded += 1
        return discarded

    def record(
        self,
        key: tuple[str, ...],
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        started_new: bool,
    ) -> TokenPoolDecodeGraphSignatureUpdate:
        key = tuple(str(part) for part in key)
        if token_pool_decode is None:
            self.signatures.pop(key, None)
            return TokenPoolDecodeGraphSignatureUpdate()

        signature = self.shape_signature(token_pool_decode)
        previous = self.signatures.get(key)
        if started_new:
            self.signatures[key] = signature
            return TokenPoolDecodeGraphSignatureUpdate(
                candidate_batches=1,
                static_shape_starts=1,
            )

        if previous is None:
            self.signatures[key] = signature
            return TokenPoolDecodeGraphSignatureUpdate(
                candidate_batches=1,
                shape_mismatches=1,
                shape_mismatch_reasons={"missing_start_signature": 1},
            )

        if signature == previous:
            return TokenPoolDecodeGraphSignatureUpdate(
                candidate_batches=1,
                static_shape_reuses=1,
            )

        reasons: dict[str, int] = {}
        for reason in self.shape_mismatch_reasons(previous, signature):
            reasons[reason] = reasons.get(reason, 0) + 1
        return TokenPoolDecodeGraphSignatureUpdate(
            candidate_batches=1,
            shape_mismatches=1,
            shape_mismatch_reasons=reasons,
        )

    @classmethod
    def shape_signature(
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
                str(layer_type): cls.decode_metadata_shape_signature(metadata)
                for layer_type, metadata in sorted(
                    token_pool_decode.metadata_by_layer_type.items(),
                    key=lambda item: str(item[0]),
                )
            },
            "metadata_by_layer_id": {
                int(layer_id): cls.decode_metadata_shape_signature(metadata)
                for layer_id, metadata in sorted(
                    (token_pool_decode.metadata_by_layer_id or {}).items(),
                    key=lambda item: int(item[0]),
                )
            },
            "paged_metadata_by_layer_type": {
                str(layer_type): cls.paged_decode_metadata_shape_signature(metadata)
                for layer_type, metadata in sorted(
                    (token_pool_decode.paged_metadata_by_layer_type or {}).items(),
                    key=lambda item: str(item[0]),
                )
            },
            "paged_metadata_by_layer_id": {
                int(layer_id): cls.paged_decode_metadata_shape_signature(metadata)
                for layer_id, metadata in sorted(
                    (token_pool_decode.paged_metadata_by_layer_id or {}).items(),
                    key=lambda item: int(item[0]),
                )
            },
        }

    @classmethod
    def decode_metadata_shape_signature(
        cls,
        metadata: DecodeBatchMetadata,
    ) -> dict[str, Any]:
        return {
            "req_pool_indices": cls.tensor_shape_signature(metadata.req_pool_indices),
            "seq_lens": cls.tensor_shape_signature(metadata.seq_lens),
            "logical_seq_lens": cls.tensor_shape_signature(metadata.logical_seq_lens),
            "out_cache_loc": cls.tensor_shape_signature(metadata.out_cache_loc),
            "kv_indptr": cls.tensor_shape_signature(metadata.kv_indptr),
            "kv_indices": cls.tensor_shape_signature(metadata.kv_indices),
            "out_cache_loc_long": cls.tensor_shape_signature(
                getattr(metadata, "out_cache_loc_long", None)
            ),
            "max_seq_len": getattr(metadata, "max_seq_len", None),
            "triton_decode_plan": cls.triton_decode_plan_signature(
                getattr(metadata, "triton_decode_plan", None)
            ),
        }

    @classmethod
    def paged_decode_metadata_shape_signature(
        cls,
        metadata: PagedDecodeBatchMetadata,
    ) -> dict[str, Any]:
        return {
            "req_pool_indices": cls.tensor_shape_signature(metadata.req_pool_indices),
            "seq_lens": cls.tensor_shape_signature(metadata.seq_lens),
            "logical_seq_lens": cls.tensor_shape_signature(metadata.logical_seq_lens),
            "out_cache_loc": cls.tensor_shape_signature(metadata.out_cache_loc),
            "block_tables": cls.tensor_shape_signature(metadata.block_tables),
            "block_table_lens": cls.tensor_shape_signature(metadata.block_table_lens),
            "selected_start_positions": cls.tensor_shape_signature(
                metadata.selected_start_positions
            ),
            "slot_mapping": cls.tensor_shape_signature(
                getattr(metadata, "slot_mapping", None)
            ),
            "out_cache_loc_long": cls.tensor_shape_signature(
                getattr(metadata, "out_cache_loc_long", None)
            ),
            "block_size": int(metadata.block_size),
            "max_seq_len": getattr(metadata, "max_seq_len", None),
            "triton_decode_plan": cls.triton_decode_plan_signature(
                getattr(metadata, "triton_decode_plan", None)
            ),
        }

    @staticmethod
    def triton_decode_plan_signature(plan: Any) -> dict[str, Any] | None:
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
    def tensor_shape_signature(value: Any) -> dict[str, Any] | None:
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
    def shape_mismatch_reasons(
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
            cls.metadata_shape_mismatch_reasons(
                "metadata_by_layer_type",
                expected.get("metadata_by_layer_type", {}),
                actual.get("metadata_by_layer_type", {}),
            )
        )
        reasons.extend(
            cls.metadata_shape_mismatch_reasons(
                "metadata_by_layer_id",
                expected.get("metadata_by_layer_id", {}),
                actual.get("metadata_by_layer_id", {}),
            )
        )
        reasons.extend(
            cls.metadata_shape_mismatch_reasons(
                "paged_metadata_by_layer_type",
                expected.get("paged_metadata_by_layer_type", {}),
                actual.get("paged_metadata_by_layer_type", {}),
            )
        )
        reasons.extend(
            cls.metadata_shape_mismatch_reasons(
                "paged_metadata_by_layer_id",
                expected.get("paged_metadata_by_layer_id", {}),
                actual.get("paged_metadata_by_layer_id", {}),
            )
        )
        return reasons or ["unknown"]

    @staticmethod
    def metadata_shape_mismatch_reasons(
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


class TokenPoolBlockTables:
    """Reusable request-to-physical-block tables for paged token-pool decode."""

    def __init__(
        self,
        *,
        max_requests: int,
        max_context_len: int,
        block_size: int,
        device: Any = "cpu",
        padding_block: int = -1,
    ) -> None:
        import torch

        self.max_requests = int(max_requests)
        self.block_size = int(block_size)
        self.padding_block = int(padding_block)
        if self.max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if int(max_context_len) < 1:
            raise ValueError("max_context_len must be >= 1")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.device = device
        self.request_block_tables = torch.full(
            (self.max_requests, self.width_for_context(max_context_len)),
            self.padding_block,
            dtype=torch.int32,
            device=device,
        )
        self._staged_writes: list[tuple[int, int, int]] = []
        self._gather_workspaces: dict[str, dict[str, Any]] = {}
        self._slot_mapping_workspaces: dict[str, Any] = {}

    @property
    def tensor(self):
        return self.request_block_tables

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(dim) for dim in self.request_block_tables.shape)

    def width_for_context(self, context_len: int) -> int:
        return max(
            1,
            (max(1, int(context_len)) + self.block_size - 1) // self.block_size,
        )

    def ensure_context_len(self, context_len: int):
        return self.ensure_width(self.width_for_context(context_len))

    def ensure_width(self, width: int):
        import torch

        width = int(width)
        if width < 1:
            raise ValueError("width must be >= 1")
        if width <= int(self.request_block_tables.shape[1]):
            return self.request_block_tables
        grown = torch.full(
            (self.max_requests, width),
            self.padding_block,
            dtype=self.request_block_tables.dtype,
            device=self.request_block_tables.device,
        )
        grown[:, : int(self.request_block_tables.shape[1])].copy_(
            self.request_block_tables
        )
        self.request_block_tables = grown
        return self.request_block_tables

    def reset_row(self, req_slot: int) -> None:
        self._validate_req_slot(req_slot)
        self.request_block_tables[int(req_slot)].fill_(self.padding_block)

    def snapshot_row(self, req_slot: int):
        self._validate_req_slot(req_slot)
        return self.request_block_tables[int(req_slot)].clone()

    def restore_row(self, req_slot: int, snapshot: Any) -> None:
        self._validate_req_slot(req_slot)
        if snapshot is None:
            self.reset_row(req_slot)
            return
        self.ensure_width(int(snapshot.numel()))
        row = self.request_block_tables[int(req_slot)]
        row.fill_(self.padding_block)
        width = min(int(row.numel()), int(snapshot.numel()))
        row[:width].copy_(snapshot[:width].to(device=row.device, dtype=row.dtype))

    def stage_block(self, req_slot: int, logical_block: int, physical_block: int) -> None:
        req_slot = self._validate_req_slot(req_slot)
        logical_block = self._validate_logical_block(logical_block)
        physical_block = self._validate_physical_block(physical_block)
        self.ensure_width(logical_block + 1)
        self._staged_writes.append((req_slot, logical_block, physical_block))

    def set_block(
        self,
        req_slot: int,
        logical_block: int,
        physical_block: int,
        *,
        staged: bool = False,
    ) -> None:
        if staged:
            self.stage_block(req_slot, logical_block, physical_block)
            return
        req_slot = self._validate_req_slot(req_slot)
        logical_block = self._validate_logical_block(logical_block)
        physical_block = self._validate_physical_block(physical_block)
        self.ensure_width(logical_block + 1)
        self.request_block_tables[req_slot, logical_block] = physical_block

    def apply_staged_writes(self) -> int:
        count = len(self._staged_writes)
        for req_slot, logical_block, physical_block in self._staged_writes:
            self.request_block_tables[req_slot, logical_block] = physical_block
        self._staged_writes.clear()
        return count

    def clear_block(self, req_slot: int, logical_block: int) -> None:
        req_slot = self._validate_req_slot(req_slot)
        logical_block = self._validate_logical_block(logical_block)
        if logical_block < int(self.request_block_tables.shape[1]):
            self.request_block_tables[req_slot, logical_block] = self.padding_block

    def block_for(self, req_slot: int, logical_block: int) -> int:
        req_slot = self._validate_req_slot(req_slot)
        logical_block = self._validate_logical_block(logical_block)
        if logical_block >= int(self.request_block_tables.shape[1]):
            return self.padding_block
        return int(self.request_block_tables[req_slot, logical_block].item())

    def gather_block_tables(
        self,
        req_slots: Iterable[int],
        first_blocks: Iterable[int],
        block_lens: Iterable[int],
        *,
        block_table_width: int | None = None,
        workspace_key: str | None = None,
    ):
        import torch

        req_slots_list = [self._validate_req_slot(slot) for slot in req_slots]
        first_block_list = [
            self._validate_logical_block(block) for block in first_blocks
        ]
        block_lens_list = [int(length) for length in block_lens]
        if not req_slots_list:
            raise ValueError("block-table gather requires at least one request slot")
        if (
            len(first_block_list) != len(req_slots_list)
            or len(block_lens_list) != len(req_slots_list)
        ):
            raise ValueError("first_blocks and block_lens must match req_slots")
        if any(length < 1 for length in block_lens_list):
            raise ValueError("block_lens must be >= 1")

        max_blocks = max(block_lens_list)
        if block_table_width is not None:
            block_table_width = int(block_table_width)
            if block_table_width < max_blocks:
                raise ValueError("block_table_width is smaller than live block table")
            max_blocks = block_table_width
        max_required = max(
            first_block + length - 1
            for first_block, length in zip(first_block_list, block_lens_list)
        )
        self.ensure_width(max_required + 1)

        device = self.request_block_tables.device
        req_slots_tensor = torch.as_tensor(req_slots_list, dtype=torch.long, device=device)
        first_blocks_tensor = torch.as_tensor(
            first_block_list,
            dtype=torch.long,
            device=device,
        )
        offsets = torch.arange(max_blocks, dtype=torch.long, device=device)
        logical_blocks = first_blocks_tensor[:, None] + offsets[None, :]
        valid = offsets[None, :] < torch.as_tensor(
            block_lens_list,
            dtype=torch.long,
            device=device,
        )[:, None]
        gathered = self.request_block_tables[
            req_slots_tensor[:, None],
            logical_blocks.clamp(
                min=0,
                max=int(self.request_block_tables.shape[1]) - 1,
            ),
        ]
        filled = torch.where(
            valid,
            gathered,
            torch.full_like(gathered, self.padding_block),
        )
        if workspace_key is None:
            return filled
        workspace = self._gather_workspaces.setdefault(str(workspace_key), {})
        block_tables = workspace.get("block_tables")
        if (
            block_tables is None
            or int(block_tables.shape[0]) < len(req_slots_list)
            or int(block_tables.shape[1]) < max_blocks
            or block_tables.dtype != filled.dtype
            or block_tables.device != device
        ):
            block_tables = torch.empty(
                (len(req_slots_list), max_blocks),
                dtype=filled.dtype,
                device=device,
            )
            workspace["block_tables"] = block_tables
        block_tables[: len(req_slots_list), :max_blocks].copy_(filled)
        return block_tables[: len(req_slots_list), :max_blocks]

    def compute_slot_mapping(
        self,
        req_slots: Iterable[int],
        logical_positions: Iterable[int],
        *,
        pad_slot_id: int = -1,
        workspace_key: str | None = None,
    ):
        import torch

        req_slots_list = [self._validate_req_slot(slot) for slot in req_slots]
        logical_positions_list = [int(pos) for pos in logical_positions]
        if not req_slots_list:
            raise ValueError("slot mapping requires at least one request slot")
        if len(logical_positions_list) != len(req_slots_list):
            raise ValueError("logical_positions must match req_slots")
        if any(pos < 0 for pos in logical_positions_list):
            raise ValueError("logical_positions must be non-negative")
        self.ensure_context_len(max(logical_positions_list) + 1)

        device = self.request_block_tables.device
        req_slots_tensor = torch.as_tensor(req_slots_list, dtype=torch.long, device=device)
        logical_positions_tensor = torch.as_tensor(
            logical_positions_list,
            dtype=torch.long,
            device=device,
        )
        logical_blocks = logical_positions_tensor // self.block_size
        physical_blocks = self.request_block_tables[
            req_slots_tensor,
            logical_blocks,
        ].to(dtype=torch.long)
        slots = physical_blocks * self.block_size + (
            logical_positions_tensor % self.block_size
        )
        slots = torch.where(
            physical_blocks >= 0,
            slots,
            torch.full_like(slots, int(pad_slot_id)),
        ).to(dtype=torch.long)
        if workspace_key is None:
            return slots
        key = str(workspace_key)
        out = self._slot_mapping_workspaces.get(key)
        if (
            out is None
            or int(out.numel()) < len(req_slots_list)
            or out.dtype != slots.dtype
            or out.device != device
        ):
            out = torch.empty(len(req_slots_list), dtype=slots.dtype, device=device)
            self._slot_mapping_workspaces[key] = out
        out[: len(req_slots_list)].copy_(slots)
        return out[: len(req_slots_list)]

    def state_bytes(self) -> int:
        total = (
            self.request_block_tables.numel()
            * self.request_block_tables.element_size()
        )
        for workspace in self._gather_workspaces.values():
            tensor = workspace.get("block_tables")
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        for tensor in self._slot_mapping_workspaces.values():
            total += tensor.numel() * tensor.element_size()
        return int(total)

    def _validate_req_slot(self, req_slot: int) -> int:
        req_slot = int(req_slot)
        if req_slot < 0 or req_slot >= self.max_requests:
            raise ValueError("req_slot is outside block-table capacity")
        return req_slot

    @staticmethod
    def _validate_logical_block(logical_block: int) -> int:
        logical_block = int(logical_block)
        if logical_block < 0:
            raise ValueError("logical_block must be non-negative")
        return logical_block

    @staticmethod
    def _validate_physical_block(physical_block: int) -> int:
        physical_block = int(physical_block)
        if physical_block < 0:
            raise ValueError("physical_block must be non-negative")
        return physical_block


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
    workspace: dict[str, Any] | TokenPoolDecodeMetadataWorkspace | None = None,
    workspace_key: str | None = None,
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
    if workspace is not None:
        if isinstance(workspace, TokenPoolDecodeMetadataWorkspace):
            metadata_workspace = workspace.ensure_paged(
                workspace_key,
                device=device,
                row_count=row_count,
                block_table_width=max_blocks,
            )
        else:
            metadata_workspace = _ensure_paged_decode_metadata_workspace(
                workspace,
                device=device,
                row_count=row_count,
                block_table_width=max_blocks,
            )
        req_pool_ws = metadata_workspace["req_pool_indices"][:row_count]
        seq_lens_ws = metadata_workspace["seq_lens"][:row_count]
        logical_lens_ws = metadata_workspace["logical_seq_lens"][:row_count]
        out_ws = metadata_workspace["out_cache_loc"][:row_count]
        out_long_ws = metadata_workspace["out_cache_loc_long"][:row_count]
        block_lens_ws = metadata_workspace["block_table_lens"][:row_count]
        starts_ws = metadata_workspace["selected_start_positions"][:row_count]
        block_tables_ws = metadata_workspace["block_tables"][
            :row_count,
            :max_blocks,
        ]
        req_pool_ws.copy_(req_pool_indices)
        seq_lens_ws.copy_(
            torch.as_tensor(selected_lens, dtype=torch.int32, device=device)
        )
        logical_lens_ws.copy_(logical_lens)
        block_lens_ws.copy_(
            torch.as_tensor(
                [len(row) for row in block_rows],
                dtype=torch.int32,
                device=device,
            )
        )
        starts_ws.copy_(
            torch.as_tensor(start_positions, dtype=torch.int32, device=device)
        )
        block_tables_ws.fill_(int(padding_block))
        for row_idx, blocks in enumerate(block_rows):
            block_tables_ws[row_idx, : len(blocks)].copy_(
                torch.as_tensor(blocks, dtype=torch.int32, device=device)
            )
        out_result = None
        out_long_result = None
        if out is not None:
            out_ws.copy_(out)
            out_long_ws.copy_(out.to(dtype=torch.long))
            out_result = out_ws
            out_long_result = out_long_ws
        return PagedDecodeBatchMetadata(
            req_pool_indices=req_pool_ws,
            seq_lens=seq_lens_ws,
            logical_seq_lens=logical_lens_ws,
            out_cache_loc=out_result,
            block_tables=block_tables_ws,
            block_table_lens=block_lens_ws,
            selected_start_positions=starts_ws,
            block_size=block_size,
            slot_mapping=out_long_result,
            out_cache_loc_long=out_long_result,
            max_seq_len=metadata_max_seq_len,
        )
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


def _slot_values_source_and_count(slots: Iterable[int] | Any) -> tuple[Any, int]:
    if isinstance(slots, int):
        return [int(slots)], 1
    numel = getattr(slots, "numel", None)
    if numel is not None:
        return slots, int(numel())
    if isinstance(slots, range):
        return slots, len(slots)
    if isinstance(slots, (list, tuple)):
        return slots, len(slots)
    try:
        return slots, len(slots)
    except TypeError:
        values = _slot_values_to_list(slots)
        return values, len(values)


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


def pad_decode_metadata_kv_indices(
    metadata: DecodeBatchMetadata,
    *,
    extra_slots: int,
    max_seq_len: int | None = None,
) -> DecodeBatchMetadata:
    import torch

    extra_slots = max(0, int(extra_slots))
    if extra_slots < 1:
        return metadata
    kv_indices = metadata.kv_indices
    current = int(kv_indices.numel())
    if current > 0:
        padding = kv_indices[-1:].expand(extra_slots)
    else:
        padding = torch.zeros(
            extra_slots,
            dtype=kv_indices.dtype,
            device=kv_indices.device,
        )

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


def pad_sliding_decode_metadata_kv_indices(
    metadata: DecodeBatchMetadata,
    *,
    sliding_window: int,
    extra_steps: int,
    current_seq_lens: Iterable[int] | None = None,
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
    window = max(1, int(sliding_window))
    target_total = sum(min(window, seq_len + extra_steps) for seq_len in seq_lens)
    target_max_seq_len = max(min(window, seq_len + extra_steps) for seq_len in seq_lens)
    return pad_decode_metadata_kv_indices(
        metadata,
        extra_slots=max(0, int(target_total) - int(metadata.kv_indices.numel())),
        max_seq_len=target_max_seq_len,
    )


class TokenPoolDecodeBackendState:
    """Backend-owned decode metadata builder for token-pool attention."""

    def __init__(
        self,
        *,
        table: ReqToTokenTable,
        allocator: Any | None = None,
        kv_pool: Any | None = None,
        block_tables: TokenPoolBlockTables | None = None,
        block_size: int = 16,
        page_table_metadata_max_rows: int = 2,
        token_pool_capacity: int | None = None,
        graph_signature_tracker: TokenPoolDecodeGraphSignatureTracker | None = None,
    ) -> None:
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if token_pool_capacity is not None and int(token_pool_capacity) < 1:
            raise ValueError("token_pool_capacity must be >= 1 or None")
        self.table = table
        self.kv_pool = kv_pool
        self.allocator = allocator if allocator is not None else kv_pool
        self.block_tables = block_tables
        self.block_size = block_size
        self.page_table_metadata_max_rows = max(0, int(page_table_metadata_max_rows))
        if token_pool_capacity is None and kv_pool is not None:
            token_pool_capacity = getattr(kv_pool, "capacity", None)
        self.token_pool_capacity = (
            None if token_pool_capacity is None else int(token_pool_capacity)
        )
        self.decode_metadata_workspace = TokenPoolDecodeMetadataWorkspace()
        self.full_attention_decode_metadata_workspace = (
            self.decode_metadata_workspace.flat_workspace("full_attention")
        )
        self.attention_workspace = TokenPoolAttentionWorkspace()
        self.graph_signature_tracker = (
            graph_signature_tracker
            if graph_signature_tracker is not None
            else TokenPoolDecodeGraphSignatureTracker()
        )
        self.full_attention_rows = (
            None
            if self.allocator is None
            else TokenPoolFullAttentionRowManager(
                allocator=self.allocator,
                block_size=block_size,
            )
        )
        self.request_slots: dict[str, int] = {}
        self.request_token_slots: dict[str, list[int]] = {}
        self.request_page_tables: dict[str, dict[int, int]] = {}
        self.request_page_owned_slots: dict[str, set[int]] = {}
        self.current_decode_batch_state: TokenPoolDecodeBatchState | None = None

    def clear_decode_batch_state(self) -> None:
        self.current_decode_batch_state = None

    def set_decode_batch_state(
        self,
        *,
        metadata_by_layer_type: dict[str, DecodeBatchMetadata],
        metadata_by_layer_id: dict[int, DecodeBatchMetadata] | None = None,
        paged_metadata_by_layer_type: (
            dict[str, PagedDecodeBatchMetadata] | None
        ) = None,
        paged_metadata_by_layer_id: dict[int, PagedDecodeBatchMetadata] | None = None,
        covered_layer_types: Iterable[str] | None = None,
    ) -> TokenPoolDecodeBatchState:
        if covered_layer_types is None:
            covered_layer_types = (
                ()
                if self.kv_pool is None
                else (
                    layer_type
                    for layer_type, metadata in metadata_by_layer_type.items()
                    if metadata is not None
                    and getattr(metadata, "out_cache_loc", None) is not None
                )
            )
        state = TokenPoolDecodeBatchState(
            metadata_by_layer_type=metadata_by_layer_type,
            metadata_by_layer_id=metadata_by_layer_id,
            paged_metadata_by_layer_type=paged_metadata_by_layer_type,
            paged_metadata_by_layer_id=paged_metadata_by_layer_id,
            covered_layer_types=frozenset(covered_layer_types),
        )
        self.current_decode_batch_state = state
        return state

    def set_decode_batch_state_by_layer_type(
        self,
        *,
        metadata_by_layer_type: dict[str, DecodeBatchMetadata],
        layer_type_by_layer_id: dict[int, str],
        paged_metadata_by_layer_type: (
            dict[str, PagedDecodeBatchMetadata] | None
        ) = None,
        covered_layer_types: Iterable[str] | None = None,
    ) -> TokenPoolDecodeBatchState:
        metadata_by_layer_id: dict[int, DecodeBatchMetadata] = {}
        paged_metadata_by_layer_id: dict[int, PagedDecodeBatchMetadata] = {}
        typed_metadata = {
            str(layer_type): metadata
            for layer_type, metadata in metadata_by_layer_type.items()
            if metadata is not None
        }
        typed_paged_metadata = {
            str(layer_type): metadata
            for layer_type, metadata in (paged_metadata_by_layer_type or {}).items()
            if metadata is not None
        }
        for layer_id, layer_type in sorted(layer_type_by_layer_id.items()):
            layer_id = int(layer_id)
            layer_type = str(layer_type)
            metadata = typed_metadata.get(layer_type)
            if metadata is not None:
                metadata_by_layer_id[layer_id] = metadata
            paged_metadata = typed_paged_metadata.get(layer_type)
            if paged_metadata is not None:
                paged_metadata_by_layer_id[layer_id] = paged_metadata
        return self.set_decode_batch_state(
            metadata_by_layer_type=typed_metadata,
            metadata_by_layer_id=(
                metadata_by_layer_id if metadata_by_layer_id else None
            ),
            paged_metadata_by_layer_type=(
                typed_paged_metadata if typed_paged_metadata else None
            ),
            paged_metadata_by_layer_id=(
                paged_metadata_by_layer_id if paged_metadata_by_layer_id else None
            ),
            covered_layer_types=covered_layer_types,
        )

    @property
    def current_covered_layer_types(self) -> frozenset[str]:
        state = self.current_decode_batch_state
        if state is None:
            return frozenset()
        return state.covered_layer_types

    def build_current_decode_context(
        self,
        *,
        layer_id_metadata_only_types: frozenset[str] = frozenset(),
    ) -> TokenPoolDecodeContext | None:
        state = self.current_decode_batch_state
        if state is None or self.kv_pool is None:
            return None
        return state.build_context(
            kv_pool=self.kv_pool,
            attention_workspace=self.attention_workspace,
            layer_id_metadata_only_types=layer_id_metadata_only_types,
        )

    @staticmethod
    def _decode_batch_reservations(
        batch_or_reservations: Any,
    ) -> tuple[Any, ...]:
        reservations = getattr(batch_or_reservations, "reservations", None)
        if reservations is not None:
            return tuple(reservations)
        if batch_or_reservations is None:
            return ()
        return tuple(batch_or_reservations)

    def prepared_decode_batch(
        self,
        reservations: Iterable[Any],
    ) -> TokenPoolPreparedDecodeBatch:
        return TokenPoolPreparedDecodeBatch(
            reservations=tuple(reservations),
            state=self.current_decode_batch_state,
        )

    def build_decode_context_for_batch(
        self,
        batch_or_reservations: Any,
        *,
        layer_id_metadata_only_types: frozenset[str] = frozenset(),
    ) -> TokenPoolDecodeContext | None:
        prepared = (
            batch_or_reservations
            if isinstance(batch_or_reservations, TokenPoolPreparedDecodeBatch)
            else self.prepared_decode_batch(
                self._decode_batch_reservations(batch_or_reservations)
            )
        )
        return prepared.build_context(
            kv_pool=self.kv_pool,
            attention_workspace=self.attention_workspace,
            layer_id_metadata_only_types=layer_id_metadata_only_types,
        )

    def commit_decode_batch(
        self,
        batch_or_reservations: Any,
        *,
        caches_by_req_id: Any | None = None,
        owner_layer_ids: Iterable[int] = (),
        attention_window: int | None = None,
        clear_nonpersistent_full_attention_rows: bool = True,
    ) -> TokenPoolDecodeCommitResult:
        reservations = self._decode_batch_reservations(batch_or_reservations)
        if not reservations:
            return TokenPoolDecodeCommitResult()

        invalidated_full_attention_rows = 0
        try:
            if caches_by_req_id is not None:
                invalidate_req_ids = self.commit_full_attention_decode_to_caches(
                    reservations=reservations,
                    caches_by_req_id=caches_by_req_id,
                    owner_layer_ids=owner_layer_ids,
                )
                if invalidate_req_ids:
                    invalidated_full_attention_rows += (
                        self.invalidate_full_attention_rows(invalidate_req_ids)
                    )
        finally:
            if clear_nonpersistent_full_attention_rows:
                self.clear_full_attention_rows(
                    [
                        getattr(reservation, "req_id")
                        for reservation in reservations
                        if not bool(
                            getattr(
                                reservation,
                                "persistent_full_attention_row",
                                False,
                            )
                        )
                    ]
                )

        cleared_prefix_slots: list[int] = []
        released_prefix_slots: list[int] = []
        expired_page_slots: list[int] = []
        if attention_window is not None:
            window = int(attention_window)
            for reservation in reservations:
                previous_length = int(getattr(reservation, "previous_length"))
                clear_before = max(previous_length + 1 - window, 0)
                result = self.clear_request_prefix(
                    str(getattr(reservation, "req_id")),
                    int(getattr(reservation, "req_slot")),
                    clear_before,
                )
                invalidated_full_attention_rows += (
                    result.invalidated_full_attention_rows
                )
                cleared_prefix_slots.extend(result.dropped_slots)
                released_prefix_slots.extend(result.released_slots)
                expired_page_slots.extend(result.expired_page_slots)

        return TokenPoolDecodeCommitResult(
            invalidated_full_attention_rows=int(invalidated_full_attention_rows),
            cleared_prefix_slots=tuple(cleared_prefix_slots),
            released_prefix_slots=tuple(released_prefix_slots),
            expired_page_slots=tuple(expired_page_slots),
        )

    def discard_decode_batch(
        self,
        batch_or_reservations: Any,
    ) -> TokenPoolDecodeDiscardResult:
        reservations = self._decode_batch_reservations(batch_or_reservations)
        if not reservations:
            return TokenPoolDecodeDiscardResult()

        self.clear_full_attention_rows(
            [getattr(reservation, "req_id") for reservation in reservations]
        )
        freed_token_slots: list[int] = []
        restored_page_slots: list[int] = []
        for reservation in reversed(reservations):
            req_id = str(getattr(reservation, "req_id"))
            req_slot = int(getattr(reservation, "req_slot"))
            token_slot = int(getattr(reservation, "token_slot"))
            if req_id in self.request_slots:
                self.truncate_table_row(
                    req_slot,
                    int(getattr(reservation, "previous_length")),
                )
            self.remove_request_token_slot(req_id, token_slot)
            page_state_snapshot = getattr(reservation, "page_state_snapshot", None)
            if page_state_snapshot is not None:
                restored_page_slots.extend(
                    self.restore_request_page_state(page_state_snapshot)
                )
                continue
            if token_slot not in self.page_owned_slots_for_request(req_id):
                allocator = self.allocator
                if allocator is None:
                    raise RuntimeError("token-pool allocator is not initialized")
                allocator.free_slots([token_slot])
                freed_token_slots.append(token_slot)

        return TokenPoolDecodeDiscardResult(
            freed_token_slots=tuple(freed_token_slots),
            restored_page_slots=tuple(restored_page_slots),
        )

    @property
    def full_attention_transient_slots(self) -> dict[str, list[int]] | None:
        if self.full_attention_rows is None:
            return None
        return self.full_attention_rows.transient_slots

    @property
    def full_attention_row_records(
        self,
    ) -> dict[str, TokenPoolFullAttentionRow] | None:
        if self.full_attention_rows is None:
            return None
        return self.full_attention_rows.rows

    def has_full_attention_rows(self) -> bool:
        return self.full_attention_rows is not None

    def allocate_page_aligned_full_attention_row_slots(
        self,
        start_position: int,
        min_slots: int,
    ) -> tuple[Any, list[int], list[int]]:
        if self.full_attention_rows is None:
            raise RuntimeError("token-pool full-attention row manager is not initialized")
        return self.full_attention_rows.allocate_page_aligned_row_slots(
            start_position,
            min_slots,
        )

    def clear_full_attention_rows(self, req_ids: str | Iterable[Any]) -> None:
        if self.full_attention_rows is None:
            return
        self.full_attention_rows.clear(req_ids)

    def invalidate_full_attention_rows(self, req_ids: str | Iterable[Any]) -> int:
        if self.full_attention_rows is None:
            return 0
        return self.full_attention_rows.invalidate(req_ids)

    def invalidate_full_attention_rows_containing(self, slots: Iterable[int] | Any) -> int:
        if self.full_attention_rows is None:
            return 0
        return self.full_attention_rows.invalidate_containing(slots)

    @property
    def graph_decode_signatures(self) -> dict[tuple[str, ...], dict[str, Any]]:
        return self.graph_signature_tracker.signatures

    def clear_graph_decode_signatures(self) -> None:
        self.graph_signature_tracker.clear()

    def discard_graph_decode_signature(self, key: tuple[str, ...]) -> None:
        self.graph_signature_tracker.discard(key)

    def discard_graph_decode_signatures_touching(self, req_ids: Iterable[Any]) -> int:
        return self.graph_signature_tracker.discard_touching(req_ids)

    def record_graph_decode_signature(
        self,
        key: tuple[str, ...],
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        started_new: bool,
    ) -> TokenPoolDecodeGraphSignatureUpdate:
        return self.graph_signature_tracker.record(
            key,
            token_pool_decode,
            started_new=started_new,
        )

    @staticmethod
    def graph_decode_shape_signature(
        token_pool_decode: TokenPoolDecodeContext,
    ) -> dict[str, Any]:
        return TokenPoolDecodeGraphSignatureTracker.shape_signature(token_pool_decode)

    @staticmethod
    def graph_decode_metadata_shape_signature(
        metadata: DecodeBatchMetadata,
    ) -> dict[str, Any]:
        return TokenPoolDecodeGraphSignatureTracker.decode_metadata_shape_signature(
            metadata
        )

    @staticmethod
    def graph_paged_decode_metadata_shape_signature(
        metadata: PagedDecodeBatchMetadata,
    ) -> dict[str, Any]:
        return TokenPoolDecodeGraphSignatureTracker.paged_decode_metadata_shape_signature(
            metadata
        )

    @staticmethod
    def graph_triton_decode_plan_signature(plan: Any) -> dict[str, Any] | None:
        return TokenPoolDecodeGraphSignatureTracker.triton_decode_plan_signature(plan)

    @staticmethod
    def graph_tensor_shape_signature(value: Any) -> dict[str, Any] | None:
        return TokenPoolDecodeGraphSignatureTracker.tensor_shape_signature(value)

    @staticmethod
    def graph_decode_shape_mismatch_reasons(
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> list[str]:
        return TokenPoolDecodeGraphSignatureTracker.shape_mismatch_reasons(
            expected,
            actual,
        )

    @staticmethod
    def graph_metadata_shape_mismatch_reasons(
        prefix: str,
        expected: dict[Any, Any],
        actual: dict[Any, Any],
    ) -> list[str]:
        return TokenPoolDecodeGraphSignatureTracker.metadata_shape_mismatch_reasons(
            prefix,
            expected,
            actual,
        )

    def capture_graph_metadata(
        self,
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        clone_tensors: bool = False,
    ) -> TokenPoolDecodeGraphMetadata:
        return self.capture_graph_decode_metadata(
            token_pool_decode,
            clone_tensors=clone_tensors,
        )

    @staticmethod
    def capture_graph_decode_metadata(
        token_pool_decode: TokenPoolDecodeContext | None,
        *,
        clone_tensors: bool = False,
    ) -> TokenPoolDecodeGraphMetadata:
        return TokenPoolDecodeGraphMetadata.capture(
            token_pool_decode,
            clone_tensors=clone_tensors,
        )

    @staticmethod
    def clone_graph_decode_context(
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> TokenPoolDecodeContext | None:
        return TokenPoolDecodeGraphMetadata.capture(
            token_pool_decode,
            clone_tensors=True,
        ).context

    @staticmethod
    def clone_graph_decode_metadata(metadata: Any) -> Any:
        return TokenPoolDecodeGraphBuffer._clone_decode_metadata(metadata)

    @staticmethod
    def copy_graph_decode_metadata_group(
        dst_group: dict[Any, Any],
        src_group: dict[Any, Any],
        name: str,
        *,
        copied: set[tuple[int, int]] | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        TokenPoolDecodeGraphBuffer._copy_decode_metadata_group(
            dst_group,
            src_group,
            name,
            copied=copied,
            stats=stats,
        )

    @staticmethod
    def copy_graph_decode_metadata(
        dst: Any,
        src: Any,
        prefix: str,
        *,
        copied: set[tuple[int, int]] | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        TokenPoolDecodeGraphBuffer._copy_decode_metadata(
            dst,
            src,
            prefix,
            copied=copied,
            stats=stats,
        )

    @staticmethod
    def copy_graph_decode_metadata_tensor(
        dst: Any,
        src: Any,
        name: str,
        *,
        copied: set[tuple[int, int]] | None = None,
        stats: dict[str, int] | None = None,
    ) -> None:
        TokenPoolDecodeGraphBuffer._copy_decode_metadata_tensor(
            dst,
            src,
            name,
            copied=copied,
            stats=stats,
        )

    @staticmethod
    def graph_decode_context_is_graphable(
        token_pool_decode: TokenPoolDecodeContext | None,
    ) -> bool:
        if token_pool_decode is None:
            return False
        if getattr(token_pool_decode, "kv_pool", None) is None:
            return False
        metadata_groups = [getattr(token_pool_decode, "metadata_by_layer_type", {})]
        for group_name in (
            "metadata_by_layer_id",
            "paged_metadata_by_layer_type",
            "paged_metadata_by_layer_id",
        ):
            group = getattr(token_pool_decode, group_name, None)
            if group:
                metadata_groups.append(group)
        for group in metadata_groups:
            for metadata in group.values():
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
                    tensor = getattr(metadata, name, None)
                    if tensor is not None and not bool(getattr(tensor, "is_cuda", False)):
                        return False
        return True

    @staticmethod
    def covered_decode_layer_types(
        token_pool_decode: Any | None,
    ) -> frozenset[str]:
        return token_pool_decode_covered_layer_types(token_pool_decode)

    @staticmethod
    def attention_mask_for_decode(
        attention_mask: Any,
        token_pool_decode: Any | None,
    ) -> Any:
        return token_pool_attention_mask_for_decode(attention_mask, token_pool_decode)

    def stats(
        self,
        *,
        active_request_slots: int | None = None,
        attention_enabled: bool = False,
        paged_block_size: int | None = None,
    ) -> dict[str, Any]:
        active_request_slot_count = (
            self.active_request_slots
            if active_request_slots is None
            else int(active_request_slots)
        )
        table_bytes = (
            self.table.req_to_token.numel() * self.table.req_to_token.element_size()
        )
        allocator = self.allocator
        kv_pool = self.kv_pool
        stats = {
            "enabled": True,
            "attention_enabled": bool(attention_enabled),
            "active_request_slots": max(0, active_request_slot_count),
            "allocated_token_slots": int(getattr(allocator, "allocated_count", 0) or 0),
            "free_token_slots": int(getattr(allocator, "free_count", 0) or 0),
            "next_token_slot": int(getattr(allocator, "next_slot", 0) or 0),
            "token_slot_high_watermark": int(
                getattr(allocator, "high_watermark", 0) or 0
            ),
            "token_slot_capacity": getattr(allocator, "capacity", None),
            "paged_block_size": paged_block_size,
            "page_table_metadata_max_rows": self.page_table_metadata_max_rows,
            "max_context_len": self.table.max_context_len,
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
        page_table = self.page_table_tensor
        if page_table is not None:
            stats["page_table_tensor_shape"] = tuple(int(dim) for dim in page_table.shape)
        block_tables = self.block_tables
        if block_tables is not None:
            stats["block_table_bytes"] = int(block_tables.state_bytes())
        return stats

    @property
    def page_table_tensor(self):
        block_tables = self.block_tables
        return None if block_tables is None else block_tables.tensor

    def ensure_page_table_width(self, context_len: int) -> None:
        block_tables = self.block_tables
        if block_tables is None:
            return
        block_tables.ensure_context_len(context_len)

    def reset_page_table_row(self, req_slot: int) -> None:
        block_tables = self.block_tables
        if block_tables is None:
            return
        block_tables.reset_row(req_slot)

    def snapshot_page_table_row(self, req_slot: int) -> Any | None:
        block_tables = self.block_tables
        if block_tables is None:
            return None
        return block_tables.snapshot_row(req_slot)

    def restore_page_table_row(self, req_slot: int, snapshot: Any | None) -> None:
        block_tables = self.block_tables
        if block_tables is None:
            return
        block_tables.restore_row(req_slot, snapshot)

    def set_page_table_block(
        self,
        req_slot: int,
        logical_block: int,
        physical_block: int,
    ) -> None:
        block_tables = self.block_tables
        if block_tables is None:
            return
        block_tables.set_block(req_slot, logical_block, physical_block)

    def clear_page_table_block(self, req_slot: int, logical_block: int) -> None:
        block_tables = self.block_tables
        if block_tables is None:
            return
        block_tables.clear_block(req_slot, logical_block)

    @property
    def active_request_slots(self) -> int:
        return len(self.request_slots)

    def has_request(self, req_id: str) -> bool:
        return str(req_id) in self.request_slots

    def request_slot_for(self, req_id: str) -> int:
        return self.request_slots[str(req_id)]

    def request_length(self, req_id_or_slot: str | int) -> int:
        return self.table.length(req_id_or_slot)

    def ensure_context_len(self, context_len: int) -> None:
        self.table.ensure_context_len(context_len)
        self.ensure_page_table_width(context_len)

    def append_table_slots(
        self,
        req_id_or_slot: str | int,
        token_slots: Iterable[int] | Any,
    ) -> tuple[int, int]:
        return self.table.append_slots(req_id_or_slot, token_slots)

    def truncate_table_row(self, req_id_or_slot: str | int, length: int) -> None:
        self.table.truncate(req_id_or_slot, length)

    def clear_table_before(self, req_id_or_slot: str | int, length: int) -> list[int]:
        return self.table.clear_before(req_id_or_slot, length)

    def admit_request(self, req_id: str) -> int:
        req_id = str(req_id)
        existing = self.request_slots.get(req_id)
        if existing is not None:
            return int(existing)
        req_slot = int(self.table.allocate(req_id))
        self.request_slots[req_id] = req_slot
        self.request_token_slots[req_id] = []
        self.admit_request_page_state(req_id, req_slot)
        return req_slot

    def release_request(self, req_id: str) -> tuple[int | None, set[int], list[int]]:
        req_id = str(req_id)
        req_slot = self.request_slots.get(req_id)
        self.clear_full_attention_rows([req_id])
        page_slots = self.release_request_page_state(req_id, req_slot)
        token_slots = self.request_token_slots.pop(req_id, [])
        if page_slots:
            token_slots = [slot for slot in token_slots if slot not in page_slots]
        if token_slots:
            allocator = self.allocator
            if allocator is None:
                raise RuntimeError("token-pool allocator is not initialized")
            allocator.free_slots(token_slots)
        if req_slot is not None:
            self.table.free(req_id)
            self.request_slots.pop(req_id, None)
        return (None if req_slot is None else int(req_slot)), page_slots, token_slots

    def append_request_token_slots(self, req_id: str, slots: Iterable[int] | Any) -> None:
        req_id = str(req_id)
        self.request_token_slots.setdefault(req_id, []).extend(
            int(slot) for slot in _slot_values_to_list(slots)
        )

    def append_request_token_slot(self, req_id: str, slot: int) -> None:
        self.request_token_slots.setdefault(str(req_id), []).append(int(slot))

    def remove_request_token_slot(self, req_id: str, slot: int) -> bool:
        token_slots = self.request_token_slots.get(str(req_id))
        if token_slots is None:
            return False
        slot = int(slot)
        if slot not in token_slots:
            return False
        token_slots.remove(slot)
        return True

    def prune_request_token_slots(
        self,
        req_id: str,
        dropped_slots: Iterable[int] | Any,
    ) -> None:
        token_slots = self.request_token_slots.get(str(req_id))
        if token_slots is None:
            return
        dropped = {int(slot) for slot in _slot_values_to_list(dropped_slots)}
        self.request_token_slots[str(req_id)] = [
            slot for slot in token_slots if slot not in dropped
        ]

    def release_dropped_table_slots(
        self,
        req_id: str,
        dropped_slots: Iterable[int] | Any,
    ) -> list[int]:
        dropped = [int(slot) for slot in _slot_values_to_list(dropped_slots)]
        if not dropped:
            return []
        page_owned = self.page_owned_slots_for_request(req_id)
        releasable = [slot for slot in dropped if slot not in page_owned]
        if releasable:
            allocator = self.allocator
            if allocator is None:
                raise RuntimeError("token-pool allocator is not initialized")
            allocator.free_slots(releasable)
        self.prune_request_token_slots(req_id, dropped)
        return releasable

    def clear_request_prefix(
        self,
        req_id: str,
        req_slot: int,
        length: int,
    ) -> TokenPoolRequestPrefixClearResult:
        dropped = self.clear_table_before(req_slot, int(length))
        invalidated = (
            self.invalidate_full_attention_rows_containing(dropped) if dropped else 0
        )
        released = self.release_dropped_table_slots(req_id, dropped) if dropped else []
        expired = self.release_expired_page_blocks(req_id, req_slot, int(length))
        return TokenPoolRequestPrefixClearResult(
            dropped_slots=tuple(int(slot) for slot in dropped),
            released_slots=tuple(int(slot) for slot in released),
            expired_page_slots=tuple(int(slot) for slot in expired),
            invalidated_full_attention_rows=int(invalidated),
        )

    def admit_request_page_state(self, req_id: str, req_slot: int | None = None) -> None:
        req_id = str(req_id)
        self.request_page_tables[req_id] = {}
        self.request_page_owned_slots[req_id] = set()
        if req_slot is not None:
            self.reset_page_table_row(int(req_slot))

    def snapshot_request_page_state(
        self,
        req_id: str,
        req_slot: int | None = None,
    ) -> TokenPoolRequestPageStateSnapshot:
        req_id = str(req_id)
        block_table_snapshot = (
            None
            if req_slot is None
            else self.snapshot_page_table_row(int(req_slot))
        )
        return TokenPoolRequestPageStateSnapshot(
            req_id=req_id,
            req_slot=None if req_slot is None else int(req_slot),
            page_table={
                int(logical_block): int(physical_block)
                for logical_block, physical_block in self.request_page_tables.get(
                    req_id,
                    {},
                ).items()
            },
            owned_slots=frozenset(
                int(slot) for slot in self.request_page_owned_slots.get(req_id, set())
            ),
            block_table_snapshot=block_table_snapshot,
        )

    def restore_request_page_state(
        self,
        snapshot: TokenPoolRequestPageStateSnapshot | None,
        *,
        free_added: bool = True,
    ) -> list[int]:
        if snapshot is None:
            return []
        req_id = str(snapshot.req_id)
        snapshot_owned = {int(slot) for slot in snapshot.owned_slots}
        current_owned = self.request_page_owned_slots.get(req_id, set())
        added_slots = sorted(int(slot) for slot in current_owned - snapshot_owned)
        if added_slots and free_added:
            allocator = self.allocator
            if allocator is None:
                raise RuntimeError("token-pool allocator is not initialized")
            allocator.free_slots(added_slots)
        self.request_page_tables[req_id] = {
            int(logical_block): int(physical_block)
            for logical_block, physical_block in snapshot.page_table.items()
        }
        self.request_page_owned_slots[req_id] = snapshot_owned
        if snapshot.req_slot is not None:
            self.restore_page_table_row(
                int(snapshot.req_slot),
                snapshot.block_table_snapshot,
            )
        return added_slots

    def release_request_page_state(
        self,
        req_id: str,
        req_slot: int | None = None,
        *,
        free_owned: bool = True,
    ) -> set[int]:
        req_id = str(req_id)
        if req_slot is not None:
            self.reset_page_table_row(int(req_slot))
        owned_slots = {
            int(slot) for slot in self.request_page_owned_slots.pop(req_id, set())
        }
        self.request_page_tables.pop(req_id, None)
        if owned_slots and free_owned:
            allocator = self.allocator
            if allocator is None:
                raise RuntimeError("token-pool allocator is not initialized")
            allocator.free_slots(sorted(owned_slots))
        return owned_slots

    def page_table_for_request(self, req_id: str) -> dict[int, int]:
        return self.request_page_tables.get(str(req_id), {})

    def page_tables_for_requests(self, req_ids: Iterable[Any]) -> list[dict[int, int]]:
        return [self.page_table_for_request(str(req_id)) for req_id in req_ids]

    def page_owned_slots_for_request(self, req_id: str) -> set[int]:
        return self.request_page_owned_slots.get(str(req_id), set())

    def allocate_page_aligned_slots(
        self,
        req_id: str,
        start_position: int,
        n: int,
        *,
        req_slot: int | None = None,
    ) -> tuple[Any, list[int]]:
        allocator = self.allocator
        if allocator is None:
            raise RuntimeError("token-pool allocator is not initialized")
        alloc_page = getattr(allocator, "alloc_page_block_with_ids", None)
        if alloc_page is None:
            return allocator.alloc_slots_with_ids(n)

        import torch

        req_id = str(req_id)
        block_size = self.block_size
        start_position = int(start_position)
        n = int(n)
        if n < 1:
            raise ValueError("n must be >= 1")
        page_table = self.request_page_tables.setdefault(req_id, {})
        owned_slots = self.request_page_owned_slots.setdefault(req_id, set())
        if req_slot is not None:
            self.ensure_page_table_width(start_position + n)
        slots: list[int] = []
        for logical_pos in range(start_position, start_position + n):
            logical_block = logical_pos // block_size
            physical_block = page_table.get(logical_block)
            if physical_block is None:
                physical_block, block_slots = alloc_page(block_size)
                physical_block = int(physical_block)
                page_table[logical_block] = physical_block
                owned_slots.update(int(slot) for slot in block_slots)
            if req_slot is not None:
                self.set_page_table_block(
                    int(req_slot),
                    logical_block,
                    int(physical_block),
                )
            slot = int(physical_block) * block_size + (logical_pos % block_size)
            if slot not in owned_slots:
                raise RuntimeError("page-aligned token slot is not owned by request")
            slots.append(slot)
        return torch.as_tensor(slots, dtype=torch.int32, device=self.device), slots

    def release_expired_page_blocks(
        self,
        req_id: str,
        req_slot: int,
        clear_before_len: int,
    ) -> list[int]:
        allocator = self.allocator
        if allocator is None or self.kv_pool is None:
            return []
        req_id = str(req_id)
        block_size = self.block_size
        clear_before_len = max(0, int(clear_before_len))
        page_table = self.request_page_tables.get(req_id)
        owned_slots = self.request_page_owned_slots.get(req_id)
        if not page_table or owned_slots is None:
            return []
        expired_logical_blocks = [
            int(logical_block)
            for logical_block in page_table
            if (int(logical_block) + 1) * block_size <= clear_before_len
        ]
        if not expired_logical_blocks:
            return []
        slots_to_free: list[int] = []
        for logical_block in sorted(expired_logical_blocks):
            physical_block = page_table.pop(logical_block, None)
            if physical_block is None:
                continue
            self.clear_page_table_block(int(req_slot), logical_block)
            start_slot = int(physical_block) * block_size
            for slot in range(start_slot, start_slot + block_size):
                if slot in owned_slots:
                    owned_slots.remove(slot)
                    slots_to_free.append(slot)
        if slots_to_free:
            allocator.free_slots(slots_to_free)
        return slots_to_free

    def should_build_sliding_paged_metadata(self) -> bool:
        if _token_pool_paged_metadata_requested():
            return True
        req_to_token = getattr(self.table, "req_to_token", None)
        if req_to_token is None:
            return False
        return not bool(getattr(req_to_token, "is_cuda", False))

    def sliding_block_table_width(self, sliding_window: int) -> int:
        sliding_window = max(1, int(sliding_window))
        # A full sliding window can start at the last token of a page, so it may
        # span one more physical page than ceil(window / block_size).
        return (
            sliding_window + self.block_size - 1 + self.block_size - 1
        ) // self.block_size

    def build_decode_metadata_by_layer_type(
        self,
        *,
        req_slots: Iterable[int],
        out_cache_loc: Iterable[int] | Any,
        sliding_window: int,
    ) -> dict[str, DecodeBatchMetadata]:
        req_slots_list = [int(slot) for slot in req_slots]
        return {
            "full_attention": self.table.build_decode_metadata(
                req_slots_list,
                out_cache_loc=out_cache_loc,
            ),
            "sliding_attention": self.table.build_decode_metadata(
                req_slots_list,
                out_cache_loc=out_cache_loc,
                sliding_window=sliding_window,
            ),
        }

    def build_sliding_decode_metadata(
        self,
        *,
        req_slots: Iterable[int],
        logical_seq_lens: Iterable[int],
        out_cache_loc: Iterable[int] | Any,
        sliding_window: int,
        build_paged_metadata: bool | None = None,
        page_tables: Iterable[dict[int, int]] | None = None,
        kv_indices_padding_steps: int = 0,
    ) -> tuple[DecodeBatchMetadata, PagedDecodeBatchMetadata | None]:
        req_slots_list = [int(slot) for slot in req_slots]
        logical_lens = [int(length) for length in logical_seq_lens]
        if not req_slots_list:
            raise ValueError("sliding decode metadata requires at least one request slot")
        if len(logical_lens) != len(req_slots_list):
            raise ValueError("logical_seq_lens length must match req_slots")
        sliding_window = max(1, int(sliding_window))
        out_cache_loc_source, out_cache_loc_count = _slot_values_source_and_count(
            out_cache_loc
        )
        if out_cache_loc_count != len(req_slots_list):
            raise ValueError("out_cache_loc length must match req_slots")

        metadata = self.table.build_decode_metadata(
            req_slots_list,
            seq_lens=logical_lens,
            out_cache_loc=out_cache_loc_source,
            sliding_window=sliding_window,
            allow_padding=True,
            workspace_key="sliding_attention",
        )
        current_seq_lens = [
            min(sliding_window, max(0, length)) for length in logical_lens
        ]
        metadata = pad_sliding_decode_metadata_kv_indices(
            metadata,
            sliding_window=sliding_window,
            extra_steps=kv_indices_padding_steps,
            current_seq_lens=current_seq_lens,
        )

        should_build_paged = (
            self.should_build_sliding_paged_metadata()
            if build_paged_metadata is None
            else bool(build_paged_metadata)
        )
        paged_metadata = None
        if should_build_paged:
            paged_metadata = self.build_sliding_paged_decode_metadata(
                req_slots=req_slots_list,
                logical_seq_lens=logical_lens,
                out_cache_loc=out_cache_loc_source,
                sliding_window=sliding_window,
                page_tables=page_tables,
            )
        return metadata, paged_metadata

    def build_sliding_paged_decode_metadata(
        self,
        *,
        req_slots: Iterable[int],
        logical_seq_lens: Iterable[int],
        out_cache_loc: Iterable[int] | Any,
        sliding_window: int,
        page_tables: Iterable[dict[int, int]] | None = None,
    ) -> PagedDecodeBatchMetadata | None:
        req_slots_list = [int(slot) for slot in req_slots]
        logical_lens = [int(length) for length in logical_seq_lens]
        if not req_slots_list:
            raise ValueError("sliding paged metadata requires at least one request slot")
        if len(logical_lens) != len(req_slots_list):
            raise ValueError("logical_seq_lens length must match req_slots")
        out_cache_loc_source, out_cache_loc_count = _slot_values_source_and_count(
            out_cache_loc
        )
        if out_cache_loc_count != len(req_slots_list):
            raise ValueError("out_cache_loc length must match req_slots")

        block_table_width = self.sliding_block_table_width(sliding_window)
        page_table_tensor = self.page_table_tensor
        if page_table_tensor is not None:
            try:
                return self.table.build_paged_decode_metadata_from_page_table_tensor(
                    req_slots_list,
                    page_table_tensor,
                    block_size=self.block_size,
                    block_table_width=block_table_width,
                    seq_lens=logical_lens,
                    out_cache_loc=out_cache_loc_source,
                    sliding_window=max(1, int(sliding_window)),
                    token_pool_capacity=self.token_pool_capacity,
                    workspace_key="sliding_attention_paged",
                    validate=False,
                )
            except (RuntimeError, ValueError, KeyError):
                pass

        page_table_list = (
            None if page_tables is None else [dict(table) for table in page_tables]
        )
        if (
            page_table_list is not None
            and len(page_table_list) == len(req_slots_list)
            and len(req_slots_list) <= self.page_table_metadata_max_rows
        ):
            try:
                return self.table.build_paged_decode_metadata_from_page_tables(
                    req_slots_list,
                    page_table_list,
                    block_size=self.block_size,
                    block_table_width=block_table_width,
                    seq_lens=logical_lens,
                    out_cache_loc=out_cache_loc_source,
                    sliding_window=max(1, int(sliding_window)),
                    token_pool_capacity=self.token_pool_capacity,
                    workspace_key="sliding_attention_paged_from_dict",
                )
            except (RuntimeError, ValueError, KeyError):
                pass

        try:
            return self.table.build_paged_decode_metadata(
                req_slots_list,
                block_size=self.block_size,
                block_table_width=block_table_width,
                seq_lens=logical_lens,
                out_cache_loc=out_cache_loc_source,
                sliding_window=max(1, int(sliding_window)),
                token_pool_capacity=self.token_pool_capacity,
                workspace_key="sliding_attention_paged_from_table",
            )
        except (RuntimeError, ValueError, KeyError):
            return None

    @property
    def device(self):
        if self.kv_pool is not None:
            return getattr(self.kv_pool, "device", self.table.req_to_token.device)
        return self.table.req_to_token.device

    def build_full_attention_decode_metadata(
        self,
        *,
        rows: Iterable[Iterable[int] | Any],
        req_slots: Iterable[int],
        logical_seq_lens: Iterable[int],
        out_cache_loc: Iterable[int] | Any,
        kv_indices_padding_steps: int = 0,
        trusted_aux_metadata: bool = True,
    ) -> DecodeBatchMetadata:
        rows_list = list(rows)
        if not rows_list:
            raise ValueError("full-attention decode metadata requires rows")
        return build_decode_metadata_from_token_slot_rows(
            rows_list,
            req_slots=req_slots,
            logical_seq_lens=logical_seq_lens,
            out_cache_loc=out_cache_loc,
            device=self.device,
            token_pool_capacity=self.token_pool_capacity,
            workspace=self.decode_metadata_workspace,
            workspace_key="full_attention",
            kv_indices_padding_slots=int(kv_indices_padding_steps) * len(rows_list),
            trusted_aux_metadata=trusted_aux_metadata,
        )

    def build_full_attention_paged_decode_metadata(
        self,
        *,
        paged_rows: Iterable[Iterable[int] | Any],
        req_slots: Iterable[int],
        logical_seq_lens: Iterable[int],
        out_cache_loc: Iterable[int] | Any,
        kv_indices_padding_steps: int = 0,
    ) -> PagedDecodeBatchMetadata:
        paged_rows_list = [list(row) for row in paged_rows]
        if not paged_rows_list:
            raise ValueError("full-attention paged metadata requires rows")
        max_paged_row_len = max(len(row) for row in paged_rows_list)
        padded_paged_row_len = max_paged_row_len + max(
            0,
            int(kv_indices_padding_steps),
        )
        block_table_width = max(
            1,
            (padded_paged_row_len + self.block_size - 1) // self.block_size,
        )
        return build_paged_decode_metadata_from_token_slot_rows(
            paged_rows_list,
            block_size=self.block_size,
            block_table_width=block_table_width,
            req_slots=req_slots,
            logical_seq_lens=logical_seq_lens,
            out_cache_loc=out_cache_loc,
            selected_start_positions=[0 for _ in paged_rows_list],
            device=self.device,
            token_pool_capacity=self.token_pool_capacity,
            allow_selected_len_gt_logical_len=True,
            max_seq_len=padded_paged_row_len,
            workspace=self.decode_metadata_workspace,
            workspace_key="full_attention_paged",
        )

    def prepare_full_attention_decode_batch(
        self,
        *,
        requests: Iterable[Any],
        reservations: Iterable[Any],
        caches_by_req_id: Any,
        owner_layer_ids: Iterable[int],
        kv_indices_padding_steps: int = 0,
        persistent_rows: bool = False,
        build_paged_rows: bool = False,
    ) -> TokenPoolFullAttentionPreparedBatch:
        pool = self.kv_pool
        row_manager = self.full_attention_rows
        if pool is None or row_manager is None:
            raise RuntimeError("full-attention token-pool backend is not initialized")

        request_list = list(requests)
        reservation_list = list(reservations)
        if len(request_list) != len(reservation_list):
            raise ValueError("requests length must match reservations")
        if not request_list:
            raise ValueError("full-attention decode batch requires requests")
        owner_layer_id_list = [int(layer_id) for layer_id in owner_layer_ids]
        if not owner_layer_id_list:
            raise ValueError("full-attention decode batch requires owner layers")

        rows: list[TokenSlotRowChunks] = []
        paged_rows: list[list[int]] = []
        logical_lens: list[int] = []
        req_slots: list[int] = []
        out_cache_loc: list[int] = []
        req_ids: list[str] = []
        prepared_rows: list[TokenPoolFullAttentionPreparedDecodeRow] = []
        append_reserve_slots = max(1, int(kv_indices_padding_steps) + 1)

        for request, reservation in zip(request_list, reservation_list):
            req_id = str(getattr(request, "req_id", request))
            req_ids.append(req_id)
            if hasattr(caches_by_req_id, "get"):
                cache = caches_by_req_id.get(req_id)
            else:
                cache = caches_by_req_id[req_id]
            if cache is None:
                raise RuntimeError(
                    f"{req_id}: missing cache for full-attention token-pool metadata"
                )
            cache_layers = getattr(cache, "layers", None)
            if cache_layers is None:
                raise RuntimeError(f"{req_id}: missing native cache layers")

            materialized_width: int | None = None
            routed_layers: list[tuple[int, Any, Any]] = []
            for layer_id in owner_layer_id_list:
                if layer_id >= len(cache_layers):
                    raise RuntimeError(
                        f"{req_id}: missing full-attention layer {layer_id}"
                    )
                layer = cache_layers[layer_id]
                writer = getattr(layer, "write_materialized_readout_to_token_pool", None)
                if writer is None:
                    raise RuntimeError(
                        f"{req_id}: layer {layer_id} cannot backfill materialized KV"
                    )
                materialized_tokens = getattr(layer, "materialized_tokens", None)
                if materialized_tokens is None:
                    raise RuntimeError(
                        f"{req_id}: layer {layer_id} has no materialized width"
                    )
                width = int(materialized_tokens())
                if materialized_width is None:
                    materialized_width = width
                elif width != materialized_width:
                    raise RuntimeError(
                        f"{req_id}: full-attention materialized widths differ"
                    )
                routed_layers.append((layer_id, layer, writer))
            materialized_width = int(materialized_width or 0)

            prepared_row = row_manager.prepare_decode_row(
                req_id,
                materialized_width=materialized_width,
                decode_token_slot=int(getattr(reservation, "token_slot")),
                decode_token_slot_tensor=getattr(reservation, "token_slot_tensor"),
                persistent_rows=persistent_rows,
                build_paged_rows=build_paged_rows,
                append_reserve_slots=append_reserve_slots,
                device=pool.device,
            )
            full_token_slot = int(prepared_row.full_token_slot)
            if persistent_rows:
                setattr(reservation, "full_attention_token_slot", full_token_slot)

            if materialized_width and not prepared_row.reused_existing_row:
                for layer_id, _layer, writer in routed_layers:
                    writer(
                        pool,
                        prepared_row.materialized_slots,
                        layer_id=int(layer_id),
                        token_slots_long=prepared_row.materialized_slots_long,
                        token_slot_ids=prepared_row.materialized_slot_ids,
                    )

            rows.append(prepared_row.row_chunks)
            req_slots.append(int(getattr(reservation, "req_slot")))
            out_cache_loc.append(full_token_slot)
            first_layer = routed_layers[0][1]
            logical_lens.append(int(getattr(first_layer, "cumulative_length")) + 1)
            if prepared_row.paged_row is not None:
                paged_rows.append(prepared_row.paged_row)
            prepared_rows.append(prepared_row)

        metadata = self.build_full_attention_decode_metadata(
            rows=rows,
            req_slots=req_slots,
            logical_seq_lens=logical_lens,
            out_cache_loc=out_cache_loc,
            kv_indices_padding_steps=kv_indices_padding_steps,
            trusted_aux_metadata=True,
        )
        paged_metadata = None
        if build_paged_rows and paged_rows:
            try:
                paged_metadata = self.build_full_attention_paged_decode_metadata(
                    paged_rows=paged_rows,
                    req_slots=req_slots,
                    logical_seq_lens=logical_lens,
                    out_cache_loc=out_cache_loc,
                    kv_indices_padding_steps=kv_indices_padding_steps,
                )
            except (RuntimeError, ValueError, KeyError):
                paged_metadata = None
        return TokenPoolFullAttentionPreparedBatch(
            metadata=metadata,
            paged_metadata=paged_metadata,
            prepared_rows=tuple(prepared_rows),
            req_ids=tuple(req_ids),
            req_slots=tuple(req_slots),
            out_cache_loc=tuple(out_cache_loc),
            logical_seq_lens=tuple(logical_lens),
        )

    def commit_full_attention_decode_to_caches(
        self,
        *,
        reservations: Iterable[Any],
        caches_by_req_id: Any,
        owner_layer_ids: Iterable[int],
    ) -> tuple[str, ...]:
        pool = self.kv_pool
        if pool is None or self.current_decode_batch_state is None:
            return ()
        if "full_attention" not in self.current_covered_layer_types:
            return ()
        owner_layer_id_list = [int(layer_id) for layer_id in owner_layer_ids]
        if not owner_layer_id_list:
            return ()

        invalidate_req_ids: set[str] = set()
        for reservation in reservations:
            req_id = str(getattr(reservation, "req_id"))
            if hasattr(caches_by_req_id, "get"):
                cache = caches_by_req_id.get(req_id)
            else:
                cache = caches_by_req_id[req_id]
            if cache is None:
                continue
            cache_layers = getattr(cache, "layers", None)
            if cache_layers is None:
                continue
            decode_token_slot = getattr(
                reservation,
                "full_attention_token_slot",
                None,
            )
            if decode_token_slot is None:
                decode_token_slot = getattr(reservation, "token_slot")
            decode_token_slot = int(decode_token_slot)
            for layer_id in owner_layer_id_list:
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
                if bool(getattr(reservation, "persistent_full_attention_row", False)):
                    invalidate_req_ids.add(req_id)
        return tuple(sorted(invalidate_req_ids))

    def build_decode_context(
        self,
        *,
        metadata_by_layer_type: dict[str, DecodeBatchMetadata],
        metadata_by_layer_id: dict[int, DecodeBatchMetadata] | None = None,
        paged_metadata_by_layer_type: (
            dict[str, PagedDecodeBatchMetadata] | None
        ) = None,
        paged_metadata_by_layer_id: dict[int, PagedDecodeBatchMetadata] | None = None,
        covered_layer_types: frozenset[str] | None = None,
        layer_id_metadata_only_types: frozenset[str] = frozenset(),
    ) -> TokenPoolDecodeContext:
        return TokenPoolDecodeContext(
            metadata_by_layer_type=metadata_by_layer_type,
            kv_pool=self.kv_pool,
            attention_workspace=self.attention_workspace,
            metadata_by_layer_id=metadata_by_layer_id,
            paged_metadata_by_layer_type=paged_metadata_by_layer_type,
            paged_metadata_by_layer_id=paged_metadata_by_layer_id,
            covered_layer_types=covered_layer_types,
            layer_id_metadata_only_types=layer_id_metadata_only_types,
        )


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
        self._attention_workspace = TokenPoolAttentionWorkspace()
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
        return self._attention_workspace.attention_output_buffer(
            batch=batch,
            query_heads=query_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

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
        return self._attention_workspace.attention_split_workspace(
            batch=batch,
            kv_heads=kv_heads,
            max_splits=max_splits,
            block_groups=block_groups,
            head_dim=head_dim,
            device=device,
        )

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
