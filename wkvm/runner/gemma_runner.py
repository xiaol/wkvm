"""Native Gemma routed-span runner boundary.

This is the N2 transition runner: wkvm owns the cache/state objects and exposes
the small cache protocol Gemma's HF module math calls (`update`,
`get_seq_length`, `get_mask_sizes`, and `is_sliding`). It intentionally avoids
HF `DynamicCache` replacement classes in the hot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from wkvm.models.gemma import GemmaRoutedSpanConfig, gemma4_e4b_routed_span_config
from wkvm.runner.gemma_state import GemmaRoutedStateBank


class DistinctCacheBatchError(RuntimeError):
    """Raised when independently-prefilled caches cannot share one decode call."""


def _cuda_memory_snapshot(enabled: bool = True) -> dict[str, int]:
    if not enabled:
        return {}
    try:
        import torch
    except Exception:
        return {}
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


class PaddedDecodeWorkspace:
    """Per-runner reusable buffers for HF-compatible padded decode batches."""

    def __init__(
        self,
        *,
        width_bucket: int = 16,
        max_buffer_bytes: int | None = None,
    ) -> None:
        if width_bucket < 1:
            raise ValueError("width_bucket must be >= 1")
        self.width_bucket = int(width_bucket)
        if max_buffer_bytes is not None and max_buffer_bytes < 1:
            raise ValueError("max_buffer_bytes must be >= 1 or None")
        self.max_buffer_bytes = max_buffer_bytes
        self._buffers: dict[tuple[Any, ...], tuple[int, Any, Any, Any]] = {}
        self.allocations = 0
        self.reuses = 0
        self.bypasses = 0
        self.allocated_bytes = 0
        self.max_observed_buffer_bytes = 0

    def borrow(
        self,
        *,
        layer_idx: int,
        batch: int,
        heads: int,
        width: int,
        head_dim: int,
        dtype,
        device,
    ):
        import torch

        capacity_width = self._round_width(width)
        buffer_bytes = self._buffer_bytes(
            batch=batch,
            heads=heads,
            width=capacity_width,
            head_dim=head_dim,
            dtype=dtype,
            mask_dtype=torch.bool,
        )
        if self.max_buffer_bytes is not None and buffer_bytes > self.max_buffer_bytes:
            self.bypasses += 1
            keys = torch.empty(batch, heads, width, head_dim, dtype=dtype, device=device)
            values = torch.empty(batch, heads, width, head_dim, dtype=dtype, device=device)
            mask = torch.empty(batch, width - 1, dtype=torch.bool, device=device)
            exact_bytes = self._buffer_bytes(
                batch=batch,
                heads=heads,
                width=width,
                head_dim=head_dim,
                dtype=dtype,
                mask_dtype=torch.bool,
            )
            self.max_observed_buffer_bytes = max(
                self.max_observed_buffer_bytes,
                exact_bytes,
            )
            return keys, values, mask, {
                "workspace_reused": 0,
                "workspace_allocated": 0,
                "workspace_bypassed": 1,
                "workspace_capacity_width": int(width),
                "workspace_width_bucket": self.width_bucket,
            }

        key = (
            int(layer_idx),
            int(batch),
            int(heads),
            int(head_dim),
            dtype,
            str(device),
        )
        existing = self._buffers.get(key)
        reused = existing is not None and int(existing[0]) >= capacity_width
        if reused:
            self.reuses += 1
            _capacity, keys, values, mask = existing
        else:
            self.allocations += 1
            keys = torch.empty(batch, heads, capacity_width, head_dim, dtype=dtype, device=device)
            values = torch.empty(batch, heads, capacity_width, head_dim, dtype=dtype, device=device)
            mask = torch.empty(batch, capacity_width - 1, dtype=torch.bool, device=device)
            self.allocated_bytes += buffer_bytes
            self.max_observed_buffer_bytes = max(
                self.max_observed_buffer_bytes,
                buffer_bytes,
            )
            self._buffers[key] = (capacity_width, keys, values, mask)
        return keys, values, mask, {
            "workspace_reused": int(reused),
            "workspace_allocated": int(not reused),
            "workspace_bypassed": 0,
            "workspace_capacity_width": int(capacity_width),
            "workspace_width_bucket": self.width_bucket,
        }

    def _round_width(self, width: int) -> int:
        return ((int(width) + self.width_bucket - 1) // self.width_bucket) * self.width_bucket

    @staticmethod
    def _buffer_bytes(*, batch: int, heads: int, width: int, head_dim: int, dtype, mask_dtype) -> int:
        import torch

        elem_size = torch.empty((), dtype=dtype).element_size()
        mask_elem_size = torch.empty((), dtype=mask_dtype).element_size()
        return (
            2 * batch * heads * width * head_dim * elem_size
            + batch * (width - 1) * mask_elem_size
        )


class _NativeGemmaLayer:
    is_sliding = False

    def __init__(self) -> None:
        self.keys = None
        self.values = None
        self.is_initialized = False
        self.cumulative_length = 0
        self.dtype = None
        self.device = None

    def lazy_initialization(self, key_states, value_states) -> None:
        import torch

        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.empty(
            (*key_states.shape[:-2], 0, key_states.shape[-1]),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.empty(
            (*value_states.shape[:-2], 0, value_states.shape[-1]),
            dtype=value_states.dtype,
            device=value_states.device,
        )
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        raise NotImplementedError

    def get_seq_length(self) -> int:
        return int(self.cumulative_length)

    def get_max_cache_shape(self) -> int:
        return -1

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.is_initialized and self.keys is not None:
            self.keys = self.keys.repeat_interleave(repeats, dim=0).contiguous()
            self.values = self.values.repeat_interleave(repeats, dim=0).contiguous()


class NativeSlidingWindowLayer(_NativeGemmaLayer):
    is_sliding = True

    def __init__(self, sliding_window: int) -> None:
        super().__init__()
        self.sliding_window = int(sliding_window)
        self._dense_storage_released = False

    def update(self, key_states, value_states, *args, **kwargs):
        if self._dense_storage_released:
            raise DistinctCacheBatchError(
                "sliding KV storage was released for token-pool decode"
            )
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]
        full_keys = _cat_time(self.keys, key_states)
        full_values = _cat_time(self.values, value_states)
        self.keys = full_keys[:, :, -self.sliding_window + 1 :, :].contiguous()
        self.values = full_values[:, :, -self.sliding_window + 1 :, :].contiguous()
        return full_keys, full_values

    def commit_decode_token(self, key_states, value_states) -> bool:
        if (
            self._dense_storage_released
            or
            not self.is_initialized
            or key_states.shape[0] != 1
            or value_states.shape[0] != 1
            or key_states.shape[-2] != 1
            or value_states.shape[-2] != 1
            or key_states.shape[1] != self.keys.shape[1]
            or value_states.shape[1] != self.values.shape[1]
            or key_states.shape[-1] != self.keys.shape[-1]
            or value_states.shape[-1] != self.values.shape[-1]
            or key_states.dtype != self.keys.dtype
            or value_states.dtype != self.values.dtype
            or key_states.device != self.keys.device
            or value_states.device != self.values.device
        ):
            return False
        keep = self.sliding_window - 1
        if keep < 1:
            return False
        self.cumulative_length += 1
        if self.keys.shape[2] >= keep:
            key_tail = self.keys[:, :, -(keep - 1) :, :] if keep > 1 else self.keys[:, :, :0, :]
            value_tail = self.values[:, :, -(keep - 1) :, :] if keep > 1 else self.values[:, :, :0, :]
            self.keys = _cat_time(key_tail, key_states)
            self.values = _cat_time(value_tail, value_states)
        else:
            self.keys = _cat_time(self.keys, key_states)
            self.values = _cat_time(self.values, value_states)
        return True

    def commit_decode_tokens(self, key_states, value_states) -> bool:
        if (
            self._dense_storage_released
            or
            not self.is_initialized
            or key_states.shape[0] != 1
            or value_states.shape[0] != 1
            or key_states.shape[-2] < 1
            or value_states.shape[-2] != key_states.shape[-2]
            or key_states.shape[1] != self.keys.shape[1]
            or value_states.shape[1] != self.values.shape[1]
            or key_states.shape[-1] != self.keys.shape[-1]
            or value_states.shape[-1] != self.values.shape[-1]
            or key_states.dtype != self.keys.dtype
            or value_states.dtype != self.values.dtype
            or key_states.device != self.keys.device
            or value_states.device != self.values.device
        ):
            return False
        keep = self.sliding_window - 1
        if keep < 1:
            return False
        self.cumulative_length += int(key_states.shape[-2])
        self.keys = _cat_time(self.keys, key_states)[:, :, -keep:, :].contiguous()
        self.values = _cat_time(self.values, value_states)[:, :, -keep:, :].contiguous()
        return True

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        kv_length = self.sliding_window - 1 + query_length if is_full else self.cumulative_length + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        return self.sliding_window


class NativeRoutedSpanLayer(_NativeGemmaLayer):
    """wkvm-owned sink/ring/routed-span layer.

    Tensor routing uses value vectors; metadata routing is mirrored into the
    `GemmaRoutedStateBank` slot so arena/metrics can inspect resident state.
    """

    is_sliding = False

    def __init__(
        self,
        layer_id: int,
        config: GemmaRoutedSpanConfig,
        slot_state=None,
        *,
        is_leader: bool = False,
        coord: list | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.sink = config.sink_tokens
        self.window = config.ring_tokens
        self.m_slots = config.routed_slots
        self.reps = config.reps_per_slot
        self.route_chunk = config.pending_tokens
        self.span_budget = config.span_budget_tokens
        self.max_span = config.max_span_tokens
        self.novelty_thresh = 0.85
        self.dup_floor = 0.10
        self.slot_state = slot_state
        self.is_leader = is_leader
        self.coord = coord
        self.break_mask: list[bool] | None = None
        self._op_cursor = 0
        self._evicted = 0
        self._n_active = 0
        self._cent = None
        self._gmean = None
        self._gcnt = 0

    def lazy_initialization(self, key_states, value_states) -> None:
        import torch

        super().lazy_initialization(key_states, value_states)
        bsz, heads, _, dim = key_states.shape
        empty = lambda: torch.empty(bsz, heads, 0, dim, dtype=key_states.dtype, device=key_states.device)
        self._sink_k, self._sink_v = empty(), empty()
        self._ring_k, self._ring_v = empty(), empty()
        self._pend_k, self._pend_v = empty(), empty()
        self._slot_mk = torch.zeros(bsz, heads, self.m_slots, dim, dtype=torch.float32, device=key_states.device)
        self._slot_mv = torch.zeros_like(self._slot_mk)
        self._slot_cnt = [0] * self.m_slots
        self._slot_spans: list[list[dict[str, Any]]] = [[] for _ in range(self.m_slots)]

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        batched_decode = key_states.shape[0] > 1
        if batched_decode and key_states.shape[-2] != 1:
            raise NotImplementedError("native routed-span batched path is decode-only")
        ret_k = _cat_time(self.keys, key_states) if self.keys.numel() else key_states
        ret_v = _cat_time(self.values, value_states) if self.values.numel() else value_states
        self.cumulative_length += key_states.shape[-2]

        rk = _cat_time(self._ring_k, key_states)
        rv = _cat_time(self._ring_v, value_states)
        deficit = self.sink - self._sink_k.shape[2]
        if deficit > 0:
            take = min(deficit, rk.shape[2])
            self._sink_k = _cat_time(self._sink_k, rk[:, :, :take])
            self._sink_v = _cat_time(self._sink_v, rv[:, :, :take])
            rk, rv = rk[:, :, take:], rv[:, :, take:]
        if rk.shape[2] > self.window:
            cut = rk.shape[2] - self.window
            self._pend_k = _cat_time(self._pend_k, rk[:, :, :cut])
            self._pend_v = _cat_time(self._pend_v, rv[:, :, :cut])
            rk, rv = rk[:, :, cut:], rv[:, :, cut:]
        self._ring_k, self._ring_v = rk.contiguous(), rv.contiguous()
        if self._pend_k.shape[2] >= self.route_chunk:
            if batched_decode:
                raise NotImplementedError(
                    "batched routed-span decode cannot route new overflow spans"
                )
            n = self._route_fold(self._pend_k, self._pend_v)
            self._pend_k = self._pend_k[:, :, n:].contiguous()
            self._pend_v = self._pend_v[:, :, n:].contiguous()
        self._materialize()
        return ret_k, ret_v

    def commit_decode_token(self, key_states, value_states) -> bool:
        if (
            not self.is_initialized
            or self.keys is None
            or self.values is None
            or key_states.shape[0] != 1
            or value_states.shape[0] != 1
            or key_states.shape[-2] != 1
            or value_states.shape[-2] != 1
            or key_states.shape[1] != self.keys.shape[1]
            or value_states.shape[1] != self.values.shape[1]
            or key_states.shape[-1] != self.keys.shape[-1]
            or value_states.shape[-1] != self.values.shape[-1]
            or key_states.dtype != self.keys.dtype
            or value_states.dtype != self.values.dtype
            or key_states.device != self.keys.device
            or value_states.device != self.values.device
            or self.keys.shape[2] != self._materialized_width()
            or self.values.shape[2] != self._materialized_width()
        ):
            return False

        ring_after_append = int(self._ring_k.shape[2]) + 1
        sink_deficit = max(self.sink - int(self._sink_k.shape[2]), 0)
        ring_after_sink = max(ring_after_append - sink_deficit, 0)
        pending_growth = max(ring_after_sink - self.window, 0)
        if int(self._pend_k.shape[2]) + pending_growth >= self.route_chunk:
            return False

        self.keys = _cat_time(self.keys, key_states)
        self.values = _cat_time(self.values, value_states)
        self.cumulative_length += 1

        rk = _cat_time(self._ring_k, key_states)
        rv = _cat_time(self._ring_v, value_states)
        if sink_deficit > 0:
            take = min(sink_deficit, rk.shape[2])
            self._sink_k = _cat_time(self._sink_k, rk[:, :, :take])
            self._sink_v = _cat_time(self._sink_v, rv[:, :, :take])
            rk, rv = rk[:, :, take:], rv[:, :, take:]
        if rk.shape[2] > self.window:
            cut = rk.shape[2] - self.window
            self._pend_k = _cat_time(self._pend_k, rk[:, :, :cut])
            self._pend_v = _cat_time(self._pend_v, rv[:, :, :cut])
            rk, rv = rk[:, :, cut:], rv[:, :, cut:]
        self._ring_k, self._ring_v = rk.contiguous(), rv.contiguous()
        return True

    def commit_decode_tokens(self, key_states, value_states) -> bool:
        if (
            not self.is_initialized
            or self.keys is None
            or self.values is None
            or key_states.shape[0] != 1
            or value_states.shape[0] != 1
            or key_states.shape[-2] < 1
            or value_states.shape[-2] != key_states.shape[-2]
            or key_states.shape[1] != self.keys.shape[1]
            or value_states.shape[1] != self.values.shape[1]
            or key_states.shape[-1] != self.keys.shape[-1]
            or value_states.shape[-1] != self.values.shape[-1]
            or key_states.dtype != self.keys.dtype
            or value_states.dtype != self.values.dtype
            or key_states.device != self.keys.device
            or value_states.device != self.values.device
            or self.keys.shape[2] != self._materialized_width()
            or self.values.shape[2] != self._materialized_width()
        ):
            return False

        steps = int(key_states.shape[-2])
        ring_after_append = int(self._ring_k.shape[2]) + steps
        sink_deficit = max(self.sink - int(self._sink_k.shape[2]), 0)
        ring_after_sink = max(ring_after_append - sink_deficit, 0)
        pending_growth = max(ring_after_sink - self.window, 0)
        if int(self._pend_k.shape[2]) + pending_growth >= self.route_chunk:
            return False

        self.keys = _cat_time(self.keys, key_states)
        self.values = _cat_time(self.values, value_states)
        self.cumulative_length += steps

        rk = _cat_time(self._ring_k, key_states)
        rv = _cat_time(self._ring_v, value_states)
        if sink_deficit > 0:
            take = min(sink_deficit, rk.shape[2])
            self._sink_k = _cat_time(self._sink_k, rk[:, :, :take])
            self._sink_v = _cat_time(self._sink_v, rv[:, :, :take])
            rk, rv = rk[:, :, take:], rv[:, :, take:]
        if rk.shape[2] > self.window:
            cut = rk.shape[2] - self.window
            self._pend_k = _cat_time(self._pend_k, rk[:, :, :cut])
            self._pend_v = _cat_time(self._pend_v, rv[:, :, :cut])
            rk, rv = rk[:, :, cut:], rv[:, :, cut:]
        self._ring_k, self._ring_v = rk.contiguous(), rv.contiguous()
        return True

    def _materialized_width(self) -> int:
        width = int(self._sink_k.shape[2])
        for spans in self._slot_spans:
            if spans:
                width += 1
                width += sum(int(span["k"].shape[2]) for span in spans)
        width += int(self._pend_k.shape[2])
        width += int(self._ring_k.shape[2])
        return width

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        stored = 0 if self.keys is None else self.keys.shape[-2]
        return stored + query_length, self.cumulative_length - stored

    def n_bank_slots(self) -> int:
        return sum(1 + sum(sp["k"].shape[2] for sp in spans) for spans in self._slot_spans if spans)

    def materialized_tokens(self) -> int:
        return 0 if self.keys is None else int(self.keys.shape[2])

    def write_materialized_readout_to_token_pool(
        self,
        token_kv_pool,
        token_slots,
        *,
        layer_id: int | None = None,
        token_slots_long=None,
        token_slot_ids: list[int] | None = None,
    ) -> None:
        """Mirror the current routed readout into token-pool KV slots."""

        import torch

        if not self.is_initialized or self.keys is None or self.values is None:
            raise DistinctCacheBatchError("routed-span layer has no materialized KV")
        if int(self.keys.shape[0]) != 1 or int(self.values.shape[0]) != 1:
            raise DistinctCacheBatchError("token-pool routed backfill expects one cache row")
        width = int(self.keys.shape[2])
        if token_slot_ids is not None:
            write_slots = [int(slot) for slot in token_slot_ids]
            slot_count = len(write_slots)
        else:
            slots = torch.as_tensor(
                token_slots,
                dtype=torch.int32,
                device=self.keys.device,
            ).reshape(-1)
            write_slots = slots
            slot_count = int(slots.numel())
        if int(slot_count) != width:
            raise ValueError(
                f"token slot count {int(slot_count)} does not match "
                f"materialized routed width {width}"
            )
        key_rows = self.keys[0].permute(1, 0, 2).contiguous()
        value_rows = self.values[0].permute(1, 0, 2).contiguous()
        if token_slots_long is not None and token_slot_ids is None:
            write_slots = torch.as_tensor(
                token_slots_long,
                dtype=torch.long,
                device=self.keys.device,
            ).reshape(-1)
            if int(write_slots.numel()) != width:
                raise ValueError(
                    f"token slot count {int(write_slots.numel())} does not match "
                    f"materialized routed width {width}"
                )
        token_kv_pool.set_kv(
            self.layer_id if layer_id is None else int(layer_id),
            write_slots,
            key_rows,
            value_rows,
        )

    def _decide(self, kind: str, compute):
        if self.coord is None or self.is_leader:
            op = (kind, *compute())
            if self.coord is not None:
                self.coord.append(op)
        else:
            op = self.coord[self._op_cursor]
            if op[0] != kind:
                raise RuntimeError(f"routed-span op log desync: {op[0]} != {kind}")
        self._op_cursor += 1
        return op

    def _route_fold(self, cut_k, cut_v) -> int:
        op = self._decide("route_span", lambda: self._route_decisions_span(cut_v))
        _, spans, assign, keeps, n_routed = op
        abs_start = self.sink + self._evicted
        self._evicted += n_routed
        for route_slot in keeps:
            new_spans = []
            for j, (a, b) in enumerate(spans):
                if assign[j] != route_slot:
                    continue
                new_spans.append(
                    {
                        "k": cut_k[:, :, a:b].contiguous(),
                        "v": cut_v[:, :, a:b].contiguous(),
                        "pos": tuple(range(abs_start + a, abs_start + b)),
                    }
                )
                n_new = b - a
                cnt = self._slot_cnt[route_slot]
                self._slot_mk[:, :, route_slot] = (
                    self._slot_mk[:, :, route_slot] * cnt + cut_k[:, :, a:b].float().sum(2)
                ) / (cnt + n_new)
                self._slot_mv[:, :, route_slot] = (
                    self._slot_mv[:, :, route_slot] * cnt + cut_v[:, :, a:b].float().sum(2)
                ) / (cnt + n_new)
                self._slot_cnt[route_slot] = cnt + n_new
                if self.slot_state is not None:
                    self.slot_state.full_layers[self.layer_id].add_span(
                        tuple(range(abs_start + a, abs_start + b)),
                        route_slot,
                        feature_kind="value",
                    )
            cand = self._slot_spans[route_slot] + new_spans
            self._slot_spans[route_slot] = [cand[i] for i in keeps[route_slot]]
        return n_routed

    def _route_decisions_span(self, cut_v):
        import torch
        from torch.nn import functional as F

        abs_start = self.sink + self._evicted
        spans, n_routed = self._split_spans(abs_start, cut_v.shape[2])
        feats = self._span_feats(cut_v, spans)
        if self._cent is None:
            n0 = min(self.m_slots, feats.shape[0])
            chosen = [feats[0]]
            sims = feats @ feats[0]
            for _ in range(n0 - 1):
                nxt = int(sims.argmin())
                chosen.append(feats[nxt])
                sims = torch.maximum(sims, feats @ feats[nxt])
            self._cent = torch.zeros(self.m_slots, feats.shape[1], device=feats.device)
            self._cent[:n0] = torch.stack(chosen)
            self._n_active = n0
        assign: list[int] = []
        for j in range(feats.shape[0]):
            sims = feats[j] @ F.normalize(self._cent[: self._n_active], dim=-1).T
            best = int(sims.argmax())
            if float(sims[best]) < self.novelty_thresh and self._n_active < self.m_slots:
                best = self._n_active
                self._cent[best] = feats[j]
                self._n_active += 1
            else:
                self._cent[best] = self._cent[best] * 0.9 + feats[j] * 0.1
            assign.append(best)
        keeps = self._span_keeps(feats, spans, assign)
        return spans, assign, keeps, n_routed

    def _split_spans(self, abs_start: int, n_tokens: int) -> tuple[list[tuple[int, int]], int]:
        breaks: list[int] = []
        for i in range(n_tokens):
            p = abs_start + i
            if self.break_mask is not None and p < len(self.break_mask):
                if self.break_mask[p]:
                    breaks.append(i)
            elif (i + 1) % 24 == 0:
                breaks.append(i)
        if not breaks:
            return [(a, min(a + 24, n_tokens)) for a in range(0, n_tokens, 24)], n_tokens
        n_routed = breaks[-1] + 1
        spans: list[tuple[int, int]] = []
        start = 0
        for b in breaks:
            end = b + 1
            while end - start > self.max_span:
                spans.append((start, start + self.max_span))
                start += self.max_span
            spans.append((start, end))
            start = end
        return spans, n_routed

    def _span_feats(self, cut_v, spans: list[tuple[int, int]]):
        import torch
        from torch.nn import functional as F

        vf = cut_v[0].permute(1, 0, 2).reshape(cut_v.shape[2], -1).float()
        mean = vf.mean(0)
        total = self._gcnt + vf.shape[0]
        self._gmean = mean if self._gmean is None else (self._gmean * self._gcnt + mean * vf.shape[0]) / total
        self._gcnt = total
        novelty = 1.0 - F.cosine_similarity(vf, self._gmean.unsqueeze(0), dim=-1)
        feats = torch.stack([vf[a + int(novelty[a:b].argmax())] for a, b in spans])
        return F.normalize(feats, dim=-1)

    def _span_keeps(self, feats, spans: list[tuple[int, int]], assign: list[int]) -> dict[int, list[int]]:
        import torch
        from torch.nn import functional as F

        keeps: dict[int, list[int]] = {}
        for route_slot in set(assign):
            old = self._slot_spans[route_slot]
            old_feats = [F.normalize(sp["v"][0].permute(1, 0, 2).reshape(sp["v"].shape[2], -1).float().mean(0), dim=-1) for sp in old]
            new_feats = [feats[j] for j in range(len(spans)) if assign[j] == route_slot]
            cand_feats = old_feats + new_feats
            cand_len = [sp["k"].shape[2] for sp in old] + [b - a for j, (a, b) in enumerate(spans) if assign[j] == route_slot]
            if not cand_feats:
                continue
            x = torch.stack(cand_feats)
            first = int((1.0 - x @ F.normalize(x.mean(0), dim=-1)).argmax())
            selected = [first]
            used = cand_len[first]
            min_dist = 1.0 - x @ x[first]
            while used < self.span_budget:
                min_dist[torch.tensor(selected, device=min_dist.device)] = -1
                nxt = int(min_dist.argmax())
                if float(min_dist[nxt]) < self.dup_floor or used + cand_len[nxt] > self.span_budget:
                    break
                selected.append(nxt)
                used += cand_len[nxt]
                min_dist = torch.minimum(min_dist, 1.0 - x @ x[nxt])
            keeps[route_slot] = sorted(selected)
        return keeps

    def _materialize(self) -> None:
        parts_k = [self._sink_k]
        parts_v = [self._sink_v]
        for slot_id, spans in enumerate(self._slot_spans):
            if spans:
                parts_k.append(self._slot_mk[:, :, slot_id : slot_id + 1].to(self.dtype))
                parts_v.append(self._slot_mv[:, :, slot_id : slot_id + 1].to(self.dtype))
                for span in spans:
                    parts_k.append(span["k"])
                    parts_v.append(span["v"])
        parts_k.extend([self._pend_k, self._ring_k])
        parts_v.extend([self._pend_v, self._ring_v])
        self.keys = _cat_nonempty(parts_k)
        self.values = _cat_nonempty(parts_v)

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        if not self.is_initialized:
            return
        for attr in ("_sink_k", "_sink_v", "_ring_k", "_ring_v", "_pend_k", "_pend_v", "_slot_mk", "_slot_mv"):
            setattr(self, attr, getattr(self, attr).repeat_interleave(repeats, dim=0).contiguous())
        for spans in self._slot_spans:
            for span in spans:
                span["k"] = span["k"].repeat_interleave(repeats, dim=0).contiguous()
                span["v"] = span["v"].repeat_interleave(repeats, dim=0).contiguous()
        self._materialize()


class NativeGemmaRoutedCache:
    """Small cache protocol object consumed by HF Gemma math."""

    is_compileable = False

    def __init__(
        self,
        hf_config,
        native_config: GemmaRoutedSpanConfig | None = None,
        slot_state=None,
    ) -> None:
        decoder = hf_config.get_text_config(decoder=True) if hasattr(hf_config, "get_text_config") else hf_config
        self.hf_config = decoder
        self.native_config = native_config or config_from_hf(decoder)
        self.slot_state = slot_state
        self.static_padded_decode = False
        self._static_padded_decode_layers = None
        self._init_shared_kv_store()
        n_owned = decoder.num_hidden_layers - getattr(decoder, "num_kv_shared_layers", 0)
        layer_types = list(decoder.layer_types[:n_owned])
        coord: list | None = []
        full_layers = set(self.native_config.full_kv_layers)
        first_full = min(full_layers) if full_layers else -1
        self.layers = []
        for layer_id, layer_type in enumerate(layer_types):
            if layer_type == "full_attention" and layer_id in full_layers:
                self.layers.append(
                    NativeRoutedSpanLayer(
                        layer_id,
                        self.native_config,
                        slot_state,
                        is_leader=(layer_id == first_full),
                        coord=coord,
                    )
                )
            else:
                self.layers.append(
                    NativeSlidingWindowLayer(
                        getattr(decoder, "sliding_window", None)
                        or getattr(decoder, "attention_chunk_size", None)
                        or self.native_config.sliding_window
                    )
                )

    def _init_shared_kv_store(self) -> None:
        self._shared_kv_by_layer: dict[int, tuple[Any, Any]] = {}
        self._shared_kv_by_type: dict[str, tuple[Any, Any]] = {}

    def _ensure_shared_kv_store(self) -> None:
        if not hasattr(self, "_shared_kv_by_layer"):
            self._shared_kv_by_layer = {}
        if not hasattr(self, "_shared_kv_by_type"):
            self._shared_kv_by_type = {}

    @property
    def is_sliding(self) -> list[bool]:
        return [layer.is_sliding for layer in self.layers]

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def store_shared_kv(
        self,
        *,
        layer_idx: int,
        layer_type: str | None,
        key_states,
        value_states,
    ) -> None:
        self._ensure_shared_kv_store()
        shared_kv = (key_states, value_states)
        self._shared_kv_by_layer[int(layer_idx)] = shared_kv
        if layer_type is not None:
            self._shared_kv_by_type[str(layer_type)] = shared_kv

    def get_shared_kv(
        self,
        *,
        layer_idx: int | None = None,
        layer_type: str | None = None,
    ):
        self._ensure_shared_kv_store()
        if layer_idx is not None:
            shared_kv = self._shared_kv_by_layer.get(int(layer_idx))
            if shared_kv is not None:
                return shared_kv
        if layer_type is not None:
            return self._shared_kv_by_type.get(str(layer_type))
        return None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if not self.layers:
            return 0
        if layer_idx >= len(self.layers):
            layer_idx = 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            layer.batch_repeat_interleave(repeats)
        self._ensure_shared_kv_store()
        repeated: dict[int, tuple[Any, Any]] = {}

        def repeat_shared(kv: tuple[Any, Any]) -> tuple[Any, Any]:
            key = id(kv)
            if key not in repeated:
                repeated[key] = (
                    kv[0].repeat_interleave(repeats, dim=0).contiguous(),
                    kv[1].repeat_interleave(repeats, dim=0).contiguous(),
                )
            return repeated[key]

        self._shared_kv_by_layer = {
            layer_idx: repeat_shared(kv)
            for layer_idx, kv in self._shared_kv_by_layer.items()
        }
        self._shared_kv_by_type = {
            layer_type: repeat_shared(kv)
            for layer_type, kv in self._shared_kv_by_type.items()
        }

    def state_bytes(self) -> int:
        self._ensure_shared_kv_store()
        total = 0
        seen: set[int] = set()

        def add_tensor(tensor) -> None:
            nonlocal total
            if tensor is None:
                return
            ptr = int(tensor.data_ptr())
            if ptr in seen:
                return
            seen.add(ptr)
            total += tensor.numel() * tensor.element_size()

        for layer in self.layers:
            for tensor in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                add_tensor(tensor)
        for store in (self._shared_kv_by_layer, self._shared_kv_by_type):
            for key_states, value_states in store.values():
                add_tensor(key_states)
                add_tensor(value_states)
        return total

    def release_tensor_storage(self) -> None:
        self._ensure_shared_kv_store()
        self._shared_kv_by_layer.clear()
        self._shared_kv_by_type.clear()
        for layer in self.layers:
            layer.keys = None
            layer.values = None
            if isinstance(layer, NativeRoutedSpanLayer) and layer.is_initialized:
                for attr in (
                    "_sink_k",
                    "_sink_v",
                    "_ring_k",
                    "_ring_v",
                    "_pend_k",
                    "_pend_v",
                    "_slot_mk",
                    "_slot_mv",
                ):
                    setattr(layer, attr, None)
                layer._slot_spans = [[] for _ in layer._slot_spans]

    def release_token_pool_covered_sliding_storage(
        self,
        layer_types: set[str] | frozenset[str],
    ) -> int:
        released = 0
        covered = set(layer_types)
        if not covered:
            return 0
        for layer_idx, layer in enumerate(self.layers):
            if not isinstance(layer, NativeSlidingWindowLayer):
                continue
            if _cache_layer_type(self.hf_config, layer_idx) not in covered:
                continue
            if layer.keys is None and layer.values is None:
                continue
            layer.keys = None
            layer.values = None
            layer._dense_storage_released = True
            released += 1
        return released

    def set_span_break_mask(self, mask: list[bool]) -> None:
        for layer in self.layers:
            if isinstance(layer, NativeRoutedSpanLayer):
                layer.break_mask = mask

    @classmethod
    def merge_exact_decode(
        cls,
        caches: list["NativeGemmaRoutedCache"],
        *,
        decode_steps: int = 1,
    ) -> tuple["NativeGemmaRoutedCache", dict[str, Any]]:
        """Merge compatible B=1 request caches into one B-row decode cache.

        This is the first production hot-path step beyond scheduler-only
        batching. It intentionally accepts only exact structural matches; rows
        with ragged materialized state need the later padded/static-buffer path.
        """

        if not caches:
            raise ValueError("no caches to merge")
        if len(caches) == 1:
            return caches[0], {"merge": "single_row"}
        base = caches[0]
        if any(cache.hf_config is not base.hf_config for cache in caches[1:]):
            raise DistinctCacheBatchError("cache hf_config objects differ")
        if any(len(cache.layers) != len(base.layers) for cache in caches[1:]):
            raise DistinctCacheBatchError("cache layer counts differ")

        layer_infos: list[dict[str, Any]] = []
        layer_groups: list[tuple[str, int, list[Any]]] = []
        for layer_idx, base_layer in enumerate(base.layers):
            layers = [cache.layers[layer_idx] for cache in caches]
            if any(type(layer) is not type(base_layer) for layer in layers[1:]):
                raise DistinctCacheBatchError(f"layer {layer_idx}: layer types differ")
            if isinstance(base_layer, NativeRoutedSpanLayer):
                info = _check_routed_span_layer_exact(layers, decode_steps, layer_idx)
                layer_groups.append(("routed", layer_idx, layers))
            elif isinstance(base_layer, NativeSlidingWindowLayer):
                info = _check_sliding_layer_exact(layers, layer_idx)
                layer_groups.append(("sliding", layer_idx, layers))
            else:
                raise DistinctCacheBatchError(
                    f"layer {layer_idx}: unsupported layer {type(base_layer).__name__}"
                )
            info["layer"] = layer_idx
            info["type"] = type(base_layer).__name__
            layer_infos.append(info)

        for kind, layer_idx, layers in layer_groups:
            if kind == "routed":
                _merge_routed_span_layer_exact(layers, layer_idx)
            else:
                _merge_sliding_layer_exact(layers, layer_idx)
        return base, {"merge": "exact_structural_concat", "layers": layer_infos}

    def split_exact_decode_into(self, caches: list["NativeGemmaRoutedCache"]) -> None:
        """Split a merged exact decode cache back into the original cache rows."""

        if not caches:
            return
        batch = len(caches)
        if any(len(cache.layers) != len(self.layers) for cache in caches):
            raise DistinctCacheBatchError("cannot split into caches with different layer counts")
        for layer_idx, merged_layer in enumerate(self.layers):
            layers = [cache.layers[layer_idx] for cache in caches]
            if isinstance(merged_layer, NativeRoutedSpanLayer):
                _split_routed_span_layer_exact(merged_layer, layers, batch, layer_idx)
            elif isinstance(merged_layer, NativeSlidingWindowLayer):
                _split_sliding_layer_exact(merged_layer, layers, batch, layer_idx)
            else:
                raise DistinctCacheBatchError(
                    f"layer {layer_idx}: unsupported split layer {type(merged_layer).__name__}"
                )

    @classmethod
    def merge_padded_decode(
        cls,
        caches: list["NativeGemmaRoutedCache"],
        *,
        decode_steps: int = 1,
        workspace: PaddedDecodeWorkspace | None = None,
        persistent: bool = False,
        graph_static: bool = False,
        token_pool_covered_layer_types: set[str] | frozenset[str] | None = None,
    ) -> tuple["NativeGemmaRoutedCache", dict[str, Any]]:
        """Build a temporary padded cache for ragged decode-only batches."""

        if not caches:
            raise ValueError("no caches to merge")
        if graph_static and not persistent:
            raise ValueError("graph_static padded decode requires persistent=True")
        if len(caches) == 1 and not persistent and not graph_static:
            return caches[0], {"merge": "single_row"}
        base = caches[0]
        if any(cache.hf_config is not base.hf_config for cache in caches[1:]):
            raise DistinctCacheBatchError("cache hf_config objects differ")
        if any(len(cache.layers) != len(base.layers) for cache in caches[1:]):
            raise DistinctCacheBatchError("cache layer counts differ")

        merged = object.__new__(cls)
        merged.hf_config = base.hf_config
        merged.native_config = base.native_config
        merged.slot_state = None
        merged.static_padded_decode = bool(graph_static)
        merged._static_padded_decode_layers = None
        merged._init_shared_kv_store()
        merged.layers = []
        layer_infos: list[dict[str, Any]] = []
        covered_layer_types = set(token_pool_covered_layer_types or ())
        for layer_idx, base_layer in enumerate(base.layers):
            layers = [cache.layers[layer_idx] for cache in caches]
            if any(type(layer) is not type(base_layer) for layer in layers[1:]):
                raise DistinctCacheBatchError(f"layer {layer_idx}: layer types differ")
            if isinstance(base_layer, NativeRoutedSpanLayer):
                layer_type = _cache_layer_type(base.hf_config, layer_idx)
                if layer_type in covered_layer_types:
                    layer, info = _merge_token_pool_covered_routed_layer(
                        layers,
                        layer_idx,
                        decode_steps=decode_steps,
                        layer_type=layer_type,
                    )
                else:
                    layer, info = _merge_routed_span_layer_padded(
                        layers,
                        decode_steps,
                        layer_idx,
                        workspace=workspace,
                        persistent=persistent,
                        graph_static=graph_static,
                    )
            elif isinstance(base_layer, NativeSlidingWindowLayer):
                layer_type = _cache_layer_type(base.hf_config, layer_idx)
                if layer_type in covered_layer_types:
                    layer, info = _merge_token_pool_covered_sliding_layer(
                        layers,
                        layer_idx,
                        decode_steps=decode_steps,
                        layer_type=layer_type,
                    )
                else:
                    layer, info = _merge_sliding_layer_padded(
                        layers,
                        layer_idx,
                        workspace=workspace,
                        reserve_steps=decode_steps,
                        persistent=persistent,
                        graph_static=graph_static,
                    )
            else:
                raise DistinctCacheBatchError(
                    f"layer {layer_idx}: unsupported layer {type(base_layer).__name__}"
                )
            info["layer"] = layer_idx
            info["type"] = type(layer).__name__
            merged.layers.append(layer)
            layer_infos.append(info)

        _validate_shared_attention_masks(merged.layers)
        if graph_static:
            _attach_static_padded_attention_masks(merged.layers)
            merged._static_padded_decode_layers = [
                layer
                for layer in merged.layers
                if isinstance(layer, _PersistentPaddedDecodeLayer) and layer._static_width
            ]
        return merged, {"merge": "padded_valid_mask_concat", "layers": layer_infos}

    def padded_attention_mask(self, *, static: bool | None = None) -> dict[str, Any]:
        use_static = self.static_padded_decode if static is None else bool(static)
        return {
            "full_attention": _mask_from_padded_layers(
                self.layers,
                is_sliding=False,
                static=use_static,
            ),
            "sliding_attention": _mask_from_padded_layers(
                self.layers,
                is_sliding=True,
                static=use_static,
            ),
        }

    def graph_padded_attention_mask(self) -> dict[str, Any]:
        return self.padded_attention_mask(static=True)

    def snapshot_static_padded_decode_state(self, *, include_kv: bool = False):
        return [
            layer.snapshot_static_state(include_kv=include_kv)
            for layer in self._static_padded_layers()
        ]

    def restore_static_padded_decode_state(self, snapshots) -> None:
        static_layers = self._static_padded_layers()
        if len(static_layers) != len(snapshots):
            raise DistinctCacheBatchError("static padded snapshot layer count mismatch")
        for layer, snapshot in zip(static_layers, snapshots):
            layer.restore_static_state(snapshot)

    def set_static_valid_mask_updates_enabled(self, enabled: bool) -> None:
        for layer in self._static_padded_layers():
            layer.set_static_valid_mask_updates_enabled(enabled)

    def record_static_padded_decode_replay(self, steps: int = 1) -> None:
        for layer in self._static_padded_layers():
            layer.record_static_replay(steps)

    def record_token_pool_covered_decode_step(self, steps: int = 1) -> None:
        for layer in self._token_pool_covered_decode_layers():
            layer.record_decode_step(steps)

    def _static_padded_layers(self) -> list["_PersistentPaddedDecodeLayer"]:
        layers = getattr(self, "_static_padded_decode_layers", None)
        if layers is None:
            layers = [
                layer
                for layer in self.layers
                if isinstance(layer, _PersistentPaddedDecodeLayer) and layer._static_width
            ]
            self._static_padded_decode_layers = layers
        return layers

    def _token_pool_covered_decode_layers(self) -> list["_TokenPoolCoveredDecodeLayer"]:
        layers = getattr(self, "_token_pool_covered_decode_layers_cache", None)
        if layers is None:
            layers = [
                layer
                for layer in self.layers
                if isinstance(layer, _TokenPoolCoveredDecodeLayer)
            ]
            self._token_pool_covered_decode_layers_cache = layers
        return layers

    def _padded_decode_capacity_layers(
        self,
    ) -> list["_PersistentPaddedDecodeLayer | _TokenPoolCoveredDecodeLayer"]:
        layers = getattr(self, "_padded_decode_capacity_layers_cache", None)
        if layers is None:
            layers = [
                layer
                for layer in self.layers
                if isinstance(
                    layer,
                    (_PersistentPaddedDecodeLayer, _TokenPoolCoveredDecodeLayer),
                )
            ]
            self._padded_decode_capacity_layers_cache = layers
        return layers

    def commit_padded_decode_into(self, caches: list["NativeGemmaRoutedCache"]) -> None:
        if not caches:
            return
        if any(len(cache.layers) != len(self.layers) for cache in caches):
            raise DistinctCacheBatchError("cannot commit into caches with different layer counts")
        for layer_idx, merged_layer in enumerate(self.layers):
            if not isinstance(merged_layer, _PaddedDecodeLayer):
                continue
            merged_layer.commit_into([cache.layers[layer_idx] for cache in caches])

    def padded_decode_remaining_capacity(self) -> int:
        layers = self._padded_decode_capacity_layers()
        if not layers:
            return 0
        return min(layer.remaining_capacity() for layer in layers)


@dataclass
class GemmaRoutedSpanRunner:
    model: Any
    bank: GemmaRoutedStateBank
    prefill_chunk: int = 2048
    decode_workspace: PaddedDecodeWorkspace | None = None
    persistent_padded_decode_cuda_graph: bool = False
    persistent_padded_decode_graph_warmup_iters: int = 3
    use_native_gemma_forward: bool = False
    native_gemma_attention_backend: str = "manual"
    native_gemma_projection_backend: str = "separate"
    native_gemma_weight_backend: str = "hf_live"
    native_gemma_release_hf_decoder_layers: bool = False
    collect_cuda_memory_phase_metrics: bool = False

    def __post_init__(self) -> None:
        if self.prefill_chunk < 1:
            raise ValueError("prefill_chunk must be >= 1")
        if self.persistent_padded_decode_graph_warmup_iters < 0:
            raise ValueError("persistent_padded_decode_graph_warmup_iters must be >= 0")
        if self.native_gemma_attention_backend not in {
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
        if self.native_gemma_projection_backend not in {
            "separate",
            "qkv_packed",
            "gate_up_packed",
            "qkv_gate_up_packed",
        }:
            raise ValueError(
                "native_gemma_projection_backend must be 'separate', 'qkv_packed', "
                "'gate_up_packed', or 'qkv_gate_up_packed'"
            )
        if self.native_gemma_weight_backend not in {"hf_live", "owned", "owned_cpu"}:
            raise ValueError(
                "native_gemma_weight_backend must be 'hf_live', 'owned', or 'owned_cpu'"
            )
        if (
            self.native_gemma_release_hf_decoder_layers
            and self.native_gemma_weight_backend not in {"owned", "owned_cpu"}
        ):
            raise ValueError(
                "native_gemma_release_hf_decoder_layers requires "
                "native_gemma_weight_backend='owned' or 'owned_cpu'"
            )
        if self.use_native_gemma_forward:
            self.model = _wrap_native_gemma_forward(
                self.model,
                native_attention_backend=self.native_gemma_attention_backend,
                native_projection_backend=self.native_gemma_projection_backend,
                native_weight_backend=self.native_gemma_weight_backend,
                native_release_hf_decoder_layers=(
                    self.native_gemma_release_hf_decoder_layers
                ),
            )
        self.device = self._model_device()

    def _record_cuda_memory_snapshot(
        self,
        snapshots: dict[str, dict[str, int]],
        phase: str,
    ) -> None:
        snapshot = _cuda_memory_snapshot(self.collect_cuda_memory_phase_metrics)
        if snapshot:
            snapshots[phase] = snapshot

    def build_cache(self, slots: dict[str, int]) -> NativeGemmaRoutedCache:
        return NativeGemmaRoutedCache(self.model.config, self.bank.config, self.bank.slot_state(slots))

    def prefill(
        self,
        token_ids: list[int],
        slots: dict[str, int],
        *,
        break_mask: list[bool] | None = None,
    ):
        if not token_ids:
            raise ValueError("empty prompt")
        cache = self.build_cache(slots)
        if break_mask is not None:
            cache.set_span_break_mask(break_mask)
        logits = None
        for start in range(0, len(token_ids), self.prefill_chunk):
            logits = self.prefill_chunk_step(
                cache,
                token_ids[start : start + self.prefill_chunk],
                slots,
                start_pos=start,
                break_mask=break_mask,
            )
        if logits is None:
            raise AssertionError("prefill produced no logits")
        return logits[0, -1], cache

    def prefill_chunk_step(
        self,
        cache: NativeGemmaRoutedCache,
        token_ids: list[int],
        slots: dict[str, int],
        *,
        start_pos: int,
        break_mask: list[bool] | None = None,
    ):
        import torch

        if not token_ids:
            raise ValueError("empty prefill chunk")
        if start_pos < 0:
            raise ValueError("start_pos must be >= 0")
        if break_mask is not None:
            cache.set_span_break_mask(break_mask)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        pos = torch.arange(
            start_pos,
            start_pos + len(token_ids),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            out = self.model(
                input_ids=ids.unsqueeze(0),
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        self.bank.ingest_positions(
            slots,
            list(range(start_pos, start_pos + len(token_ids))),
            break_mask,
        )
        return out.logits

    def decode_step(
        self,
        cache: NativeGemmaRoutedCache,
        last_tokens,
        *,
        position_ids=None,
        token_pool_decode=None,
    ):
        import torch

        total_start = time.perf_counter()
        ids = torch.as_tensor(last_tokens, dtype=torch.long, device=self.device).reshape(-1, 1)
        pos = None
        if position_ids is not None:
            pos = torch.as_tensor(position_ids, dtype=torch.long, device=self.device).reshape(-1, 1)
        model_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(
                input_ids=ids,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                attention_mask={"full_attention": None, "sliding_attention": None},
                **self._native_forward_extra_kwargs(
                    token_pool_decode=token_pool_decode,
                ),
            )
        model_wall = time.perf_counter() - model_start
        self.last_decode_batch_info = {
            "merge": "single_row",
            "model_forward_wall_s": model_wall,
            "decode_wall_s_total": time.perf_counter() - total_start,
        }
        return out.logits[:, -1]

    def decode_batch(
        self,
        caches: list[NativeGemmaRoutedCache],
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        token_pool_decode=None,
    ):
        import torch

        if len(caches) != len(last_tokens):
            raise ValueError("caches and last_tokens length mismatch")
        if not caches:
            raise ValueError("empty decode batch")
        total_start = time.perf_counter()
        attention_mask = {"full_attention": None, "sliding_attention": None}
        commit_padded = False
        merge_start = time.perf_counter()
        padded_mask_wall = 0.0
        native_forward_kwargs = self._native_forward_extra_kwargs(
            token_pool_decode=token_pool_decode,
        )
        token_pool_covered_layer_types = _token_pool_covered_layer_types(
            native_forward_kwargs.get("wkvm_token_pool_decode"),
        )
        exact_exc = None
        if token_pool_covered_layer_types:
            exact_exc = DistinctCacheBatchError(
                "token-pool covered layers require placeholder padded merge"
            )
        else:
            try:
                merged_cache, info = NativeGemmaRoutedCache.merge_exact_decode(
                    caches, decode_steps=1
                )
            except DistinctCacheBatchError as exc:
                exact_exc = exc
        if exact_exc is not None:
            try:
                merged_cache, info = NativeGemmaRoutedCache.merge_padded_decode(
                    caches,
                    decode_steps=1,
                    workspace=self.decode_workspace,
                    token_pool_covered_layer_types=token_pool_covered_layer_types,
                )
                commit_padded = True
            except DistinctCacheBatchError as padded_exc:
                raise DistinctCacheBatchError(
                    f"exact={exact_exc}; padded={padded_exc}"
                ) from padded_exc
        merge_wall = time.perf_counter() - merge_start
        if commit_padded:
            mask_start = time.perf_counter()
            attention_mask = merged_cache.padded_attention_mask()
            padded_mask_wall = time.perf_counter() - mask_start
        info["merge_wall_s"] = merge_wall
        info["padded_mask_wall_s"] = padded_mask_wall
        self.last_decode_batch_info = info
        ids = torch.as_tensor(last_tokens, dtype=torch.long, device=self.device).reshape(-1, 1)
        pos = None
        if position_ids is not None:
            pos = torch.as_tensor(position_ids, dtype=torch.long, device=self.device).reshape(-1, 1)
        model_wall = 0.0
        commit_wall = 0.0
        split_wall = 0.0
        if commit_padded:
            attention_mask = _attention_mask_for_token_pool_decode(
                attention_mask,
                native_forward_kwargs.get("wkvm_token_pool_decode"),
            )
            model_start = time.perf_counter()
            with torch.inference_mode():
                out = self.model(
                    input_ids=ids,
                    position_ids=pos,
                    past_key_values=merged_cache,
                    use_cache=True,
                    logits_to_keep=1,
                    attention_mask=attention_mask,
                    **native_forward_kwargs,
                )
            model_wall = time.perf_counter() - model_start
            commit_start = time.perf_counter()
            merged_cache.commit_padded_decode_into(caches)
            commit_wall = time.perf_counter() - commit_start
        else:
            try:
                model_start = time.perf_counter()
                with torch.inference_mode():
                    out = self.model(
                        input_ids=ids,
                        position_ids=pos,
                        past_key_values=merged_cache,
                        use_cache=True,
                        logits_to_keep=1,
                        attention_mask=attention_mask,
                        **native_forward_kwargs,
                    )
                model_wall = time.perf_counter() - model_start
            finally:
                split_start = time.perf_counter()
                merged_cache.split_exact_decode_into(caches)
                split_wall = time.perf_counter() - split_start
        info["model_forward_wall_s"] = model_wall
        info["commit_wall_s"] = commit_wall
        info["split_wall_s"] = split_wall
        info["decode_wall_s_total"] = time.perf_counter() - total_start
        return out.logits[:, -1]

    def decode_batch_exact_persistent(
        self,
        caches: list[NativeGemmaRoutedCache],
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        token_pool_decode=None,
    ):
        native_forward_kwargs = self._native_forward_extra_kwargs(
            token_pool_decode=token_pool_decode,
        )
        if _token_pool_covered_layer_types(
            native_forward_kwargs.get("wkvm_token_pool_decode")
        ):
            raise DistinctCacheBatchError(
                "token-pool covered layers require placeholder padded merge"
            )
        total_start = time.perf_counter()
        merge_start = time.perf_counter()
        merged_cache, info = NativeGemmaRoutedCache.merge_exact_decode(
            caches, decode_steps=1
        )
        info["merge_wall_s"] = time.perf_counter() - merge_start
        if info.get("merge") != "exact_structural_concat":
            raise DistinctCacheBatchError(f"persistent exact decode needs exact merge, got {info.get('merge')}")
        logits = self.decode_persistent_exact_batch(
            merged_cache,
            last_tokens,
            position_ids=position_ids,
            token_pool_decode=token_pool_decode,
        )
        decode_info = getattr(self, "last_decode_batch_info", {})
        for key in ("model_forward_wall_s", "decode_wall_s_total"):
            if key in decode_info:
                info[key] = decode_info[key]
        info["persistent_exact_decode"] = "start"
        info["decode_wall_s_total"] = time.perf_counter() - total_start
        self.last_decode_batch_info = info
        return logits, merged_cache

    def decode_persistent_exact_batch(
        self,
        merged_cache: NativeGemmaRoutedCache,
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        token_pool_decode=None,
    ):
        total_start = time.perf_counter()
        logits = self.decode_step(
            merged_cache,
            last_tokens,
            position_ids=position_ids,
            token_pool_decode=token_pool_decode,
        )
        decode_info = getattr(self, "last_decode_batch_info", {})
        self.last_decode_batch_info = {
            "merge": "exact_structural_concat",
            "persistent_exact_decode": "reuse",
            "model_forward_wall_s": float(decode_info.get("model_forward_wall_s", 0.0) or 0.0),
            "decode_wall_s_total": time.perf_counter() - total_start,
        }
        return logits

    def decode_batch_padded_persistent(
        self,
        caches: list[NativeGemmaRoutedCache],
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        reserve_steps: int = 1,
        token_pool_decode=None,
    ):
        if reserve_steps < 1:
            raise ValueError("reserve_steps must be >= 1")
        total_start = time.perf_counter()
        merge_start = time.perf_counter()
        cuda_memory: dict[str, dict[str, int]] = {}
        self._record_cuda_memory_snapshot(cuda_memory, "before_padded_merge")
        native_forward_kwargs = self._native_forward_extra_kwargs(
            token_pool_decode=token_pool_decode,
        )
        token_pool_covered_layer_types = _token_pool_covered_layer_types(
            native_forward_kwargs.get("wkvm_token_pool_decode"),
        )
        from wkvm.runner.gemma_token_pool import TokenPoolDecodeBackendState

        token_pool_decode = native_forward_kwargs.get("wkvm_token_pool_decode")
        use_cuda_graph = self._can_cuda_graph_decode() and (
            not token_pool_covered_layer_types
            or TokenPoolDecodeBackendState.graph_decode_context_is_graphable(
                token_pool_decode,
            )
        )
        merged_cache, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=reserve_steps,
            workspace=None,
            persistent=True,
            graph_static=use_cuda_graph,
            token_pool_covered_layer_types=token_pool_covered_layer_types,
        )
        self._record_cuda_memory_snapshot(cuda_memory, "after_padded_merge")
        info["merge_wall_s"] = time.perf_counter() - merge_start
        if cuda_memory:
            info["cuda_memory"] = cuda_memory
        if info.get("merge") != "padded_valid_mask_concat":
            raise DistinctCacheBatchError(
                f"persistent padded decode needs padded merge, got {info.get('merge')}"
            )
        if self.persistent_padded_decode_cuda_graph:
            info["persistent_padded_decode_cuda_graph_requested"] = 1
            info["persistent_padded_decode_cuda_graph"] = int(use_cuda_graph)
            if not use_cuda_graph:
                info["persistent_padded_decode_cuda_graph_skip"] = "unavailable"
        if use_cuda_graph:
            capture_start = time.perf_counter()
            try:
                merged_cache._padded_decode_graph = _GraphedPaddedDecodeStep(  # type: ignore[attr-defined]
                    self.model,
                    merged_cache,
                    len(last_tokens),
                    device=self.device,
                    warmup_iters=self.persistent_padded_decode_graph_warmup_iters,
                    token_pool_decode=native_forward_kwargs.get(
                        "wkvm_token_pool_decode"
                    ),
                )
            except Exception as exc:
                if hasattr(merged_cache, "_padded_decode_graph"):
                    delattr(merged_cache, "_padded_decode_graph")
                import traceback

                trace = traceback.extract_tb(exc.__traceback__)
                site = ""
                if trace:
                    frame = trace[-1]
                    site = f"{frame.filename}:{frame.lineno}:{frame.name}: "
                info["persistent_padded_decode_cuda_graph"] = 0
                info["persistent_padded_decode_cuda_graph_captured"] = 0
                info["persistent_padded_decode_cuda_graph_skip"] = (
                    f"capture_failed:{type(exc).__name__}: {site}"
                    f"{str(exc).splitlines()[0]}"
                )
            else:
                info["persistent_padded_decode_cuda_graph_captured"] = 1
            finally:
                info["cuda_graph_capture_wall_s"] = time.perf_counter() - capture_start
                self._record_cuda_memory_snapshot(
                    cuda_memory,
                    "after_padded_cuda_graph_capture",
                )
                if cuda_memory:
                    info["cuda_memory"] = cuda_memory
        logits = self._decode_padded_cache(
            merged_cache,
            last_tokens,
            position_ids=position_ids,
            token_pool_decode=token_pool_decode,
        )
        decode_info = getattr(self, "last_decode_batch_info", {})
        for key in (
            "padded_mask_wall_s",
            "model_forward_wall_s",
            "cuda_graph_replay",
            "cuda_graph_input_copy_wall_s",
            "cuda_graph_metadata_copy_wall_s",
            "cuda_graph_replay_wall_s",
            "cuda_graph_decode_wall_s_total",
            "cuda_graph_metadata_tensor_copies",
            "cuda_graph_metadata_tensor_copy_skips",
            "cuda_graph_metadata_alias_fastpath_metadata_skips",
        ):
            if key in decode_info:
                info[key] = decode_info[key]
        decode_cuda_memory = decode_info.get("cuda_memory")
        if isinstance(decode_cuda_memory, dict):
            cuda_memory.update(decode_cuda_memory)
        info["persistent_padded_decode"] = "start"
        info["persistent_padded_decode_reserve_steps"] = int(reserve_steps)
        info["decode_wall_s_total"] = time.perf_counter() - total_start
        self.last_decode_batch_info = info
        return logits, merged_cache

    def decode_persistent_padded_batch(
        self,
        merged_cache: NativeGemmaRoutedCache,
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        token_pool_decode=None,
    ):
        total_start = time.perf_counter()
        logits = self._decode_padded_cache(
            merged_cache,
            last_tokens,
            position_ids=position_ids,
            token_pool_decode=token_pool_decode,
        )
        decode_info = getattr(self, "last_decode_batch_info", {})
        self.last_decode_batch_info = {
            "merge": "padded_valid_mask_concat",
            "persistent_padded_decode": "reuse",
            "padded_mask_wall_s": float(decode_info.get("padded_mask_wall_s", 0.0) or 0.0),
            "model_forward_wall_s": float(decode_info.get("model_forward_wall_s", 0.0) or 0.0),
            "decode_wall_s_total": time.perf_counter() - total_start,
        }
        for key in (
            "cuda_graph_input_copy_wall_s",
            "cuda_graph_metadata_copy_wall_s",
            "cuda_graph_replay_wall_s",
            "cuda_graph_decode_wall_s_total",
            "cuda_graph_metadata_tensor_copies",
            "cuda_graph_metadata_tensor_copy_skips",
            "cuda_graph_metadata_alias_fastpath_metadata_skips",
        ):
            if key in decode_info:
                self.last_decode_batch_info[key] = decode_info[key]
        decode_cuda_memory = decode_info.get("cuda_memory")
        if isinstance(decode_cuda_memory, dict):
            self.last_decode_batch_info["cuda_memory"] = decode_cuda_memory
        if "cuda_graph_replay" in decode_info:
            self.last_decode_batch_info["cuda_graph_replay"] = decode_info["cuda_graph_replay"]
        return logits

    def _decode_padded_cache(
        self,
        cache: NativeGemmaRoutedCache,
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        token_pool_decode=None,
    ):
        import torch

        graph = getattr(cache, "_padded_decode_graph", None)
        cuda_memory: dict[str, dict[str, int]] = {}
        self._record_cuda_memory_snapshot(cuda_memory, "before_padded_decode")
        if graph is not None:
            total_start = time.perf_counter()
            model_start = time.perf_counter()
            logits = graph.decode(
                last_tokens,
                position_ids=position_ids,
                token_pool_decode=token_pool_decode,
            )
            self._record_cuda_memory_snapshot(
                cuda_memory,
                "after_padded_cuda_graph_replay",
            )
            info = {
                "merge": "padded_valid_mask_concat",
                "padded_mask_wall_s": 0.0,
                "model_forward_wall_s": time.perf_counter() - model_start,
                "decode_wall_s_total": time.perf_counter() - total_start,
                "cuda_graph_replay": 1,
            }
            graph_info = getattr(graph, "last_decode_info", None)
            if isinstance(graph_info, dict):
                info.update(graph_info)
            if cuda_memory:
                info["cuda_memory"] = cuda_memory
            self.last_decode_batch_info = info
            return logits

        ids = torch.as_tensor(last_tokens, dtype=torch.long, device=self.device).reshape(-1, 1)
        pos = None
        if position_ids is not None:
            pos = torch.as_tensor(position_ids, dtype=torch.long, device=self.device).reshape(-1, 1)
        self._record_cuda_memory_snapshot(cuda_memory, "after_padded_input_staging")
        total_start = time.perf_counter()
        mask_start = time.perf_counter()
        attention_mask = cache.padded_attention_mask()
        native_forward_kwargs = self._native_forward_extra_kwargs(
            token_pool_decode=token_pool_decode,
        )
        attention_mask = _attention_mask_for_token_pool_decode(
            attention_mask,
            native_forward_kwargs.get("wkvm_token_pool_decode"),
        )
        self._record_cuda_memory_snapshot(cuda_memory, "after_padded_attention_mask")
        padded_mask_wall = time.perf_counter() - mask_start
        model_start = time.perf_counter()
        with torch.inference_mode():
            out = self.model(
                input_ids=ids,
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                attention_mask=attention_mask,
                **native_forward_kwargs,
            )
        if token_pool_decode is not None:
            record_token_pool_step = getattr(
                cache,
                "record_token_pool_covered_decode_step",
                None,
            )
            if record_token_pool_step is not None:
                record_token_pool_step(int(ids.shape[1]))
        self._record_cuda_memory_snapshot(cuda_memory, "after_padded_model_forward")
        info = {
            "merge": "padded_valid_mask_concat",
            "padded_mask_wall_s": padded_mask_wall,
            "model_forward_wall_s": time.perf_counter() - model_start,
            "decode_wall_s_total": time.perf_counter() - total_start,
        }
        if cuda_memory:
            info["cuda_memory"] = cuda_memory
        self.last_decode_batch_info = info
        if graph is not None:
            self.last_decode_batch_info["cuda_graph_replay"] = 0
            self.last_decode_batch_info["cuda_graph_skip"] = "token_pool_decode_context"
        return out.logits[:, -1]

    def _native_forward_extra_kwargs(self, *, token_pool_decode=None) -> dict[str, Any]:
        if token_pool_decode is None:
            return {}
        if not getattr(self.model, "wkvm_no_hf_transformer_forward", False):
            return {}
        return {"wkvm_token_pool_decode": token_pool_decode}

    def generate_greedy(
        self,
        token_ids: list[int],
        slots: dict[str, int],
        *,
        max_new_tokens: int,
        break_mask: list[bool] | None = None,
    ) -> list[int]:
        logits, cache = self.prefill(token_ids, slots, break_mask=break_mask)
        out: list[int] = []
        tok = int(logits.argmax().item())
        out.append(tok)
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(cache, [tok])
            tok = int(logits[0].argmax().item())
            out.append(tok)
        return out

    def _model_device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def _can_cuda_graph_decode(self) -> bool:
        if not self.persistent_padded_decode_cuda_graph:
            return False
        try:
            import torch

            return torch.cuda.is_available() and torch.device(self.device).type == "cuda"
        except Exception:
            return False


def _attention_mask_for_token_pool_decode(attention_mask, token_pool_decode):
    from wkvm.runner.gemma_token_pool import TokenPoolDecodeBackendState

    return TokenPoolDecodeBackendState.attention_mask_for_decode(
        attention_mask,
        token_pool_decode,
    )


def _token_pool_covered_layer_types(token_pool_decode) -> frozenset[str]:
    from wkvm.runner.gemma_token_pool import TokenPoolDecodeBackendState

    return TokenPoolDecodeBackendState.covered_decode_layer_types(token_pool_decode)


class _GraphedPaddedDecodeStep:
    """One-token CUDA graph replay for a graph-static persistent padded cache."""

    def __init__(
        self,
        model,
        cache: NativeGemmaRoutedCache,
        batch_size: int,
        *,
        device,
        warmup_iters: int = 3,
        token_pool_decode=None,
    ) -> None:
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise DistinctCacheBatchError("CUDA graph decode requires a CUDA model device")
        self.model = model
        self.cache = cache
        self.batch_size = int(batch_size)
        self.ids = torch.zeros(self.batch_size, 1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(self.batch_size, 1, dtype=torch.long, device=device)
        self._ids_flat = self.ids.reshape(self.batch_size)
        self._position_ids_flat = self.position_ids.reshape(self.batch_size)
        self._ids_cpu, self._ids_cpu_np = self._new_decode_staging_buffer(torch)
        self._position_ids_cpu, self._position_ids_cpu_np = self._new_decode_staging_buffer(torch)
        from wkvm.runner.gemma_token_pool import TokenPoolDecodeBackendState

        self._token_pool_metadata = (
            TokenPoolDecodeBackendState.capture_graph_decode_metadata(
                token_pool_decode,
                clone_tensors=False,
            )
        )
        self.token_pool_decode = self._token_pool_metadata.context
        self._records_token_pool_decode_steps = bool(
            cache._token_pool_covered_decode_layers()
        )
        self.attention_mask = _attention_mask_for_token_pool_decode(
            cache.graph_padded_attention_mask(),
            self.token_pool_decode,
        )
        self.logits = None
        self.last_decode_info: dict[str, Any] = {}

        cache.set_static_valid_mask_updates_enabled(False)
        try:
            with torch.inference_mode():
                snap = cache.snapshot_static_padded_decode_state()
                try:
                    logits = self._forward()
                    self.logits = torch.empty_like(logits)
                finally:
                    cache.restore_static_padded_decode_state(snap)
                    self.ids.zero_()
                    self.position_ids.zero_()
                torch.cuda.synchronize()

                snap = cache.snapshot_static_padded_decode_state()
                try:
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(warmup_iters):
                            self._step()
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                finally:
                    cache.restore_static_padded_decode_state(snap)
                    self.ids.zero_()
                    self.position_ids.zero_()

                snap = cache.snapshot_static_padded_decode_state()
                try:
                    self.graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(self.graph):
                        self._step()
                    torch.cuda.synchronize()
                finally:
                    cache.restore_static_padded_decode_state(snap)
                    self.ids.zero_()
                    self.position_ids.zero_()
        finally:
            cache.set_static_valid_mask_updates_enabled(True)

    def _forward(self):
        native_forward_kwargs = {}
        if self.token_pool_decode is not None:
            native_forward_kwargs["wkvm_token_pool_decode"] = self.token_pool_decode
        out = self.model(
            input_ids=self.ids,
            position_ids=self.position_ids,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
            attention_mask=self.attention_mask,
            **native_forward_kwargs,
        )
        return out.logits[:, -1]

    def _step(self) -> None:
        logits = self._forward()
        if self.logits is None:
            raise AssertionError("graph logits buffer was not initialized")
        self.logits.copy_(logits)

    def _new_decode_staging_buffer(self, torch):
        try:
            cpu = torch.empty(self.batch_size, dtype=torch.long, pin_memory=True)
        except Exception:
            cpu = torch.empty(self.batch_size, dtype=torch.long)
        try:
            cpu_np = cpu.numpy()
        except Exception:
            cpu_np = None
        return cpu, cpu_np

    def _copy_list_to_static_input(self, values: list[int], cpu, cpu_np, dst) -> None:
        if cpu_np is not None:
            cpu_np[:] = values
        else:
            for idx, value in enumerate(values):
                cpu[idx] = int(value)
        dst.copy_(cpu, non_blocking=bool(getattr(cpu, "is_pinned", lambda: False)()))

    def decode(
        self,
        last_tokens: list[int],
        *,
        position_ids: list[int] | None = None,
        token_pool_decode=None,
    ):
        total_start = time.perf_counter()
        if len(last_tokens) != self.batch_size:
            raise ValueError("last_tokens length does not match graph batch size")
        input_start = time.perf_counter()
        self._copy_list_to_static_input(
            last_tokens,
            self._ids_cpu,
            self._ids_cpu_np,
            self._ids_flat,
        )
        if position_ids is None:
            self.position_ids.fill_(int(self.cache.get_seq_length()))
        else:
            if len(position_ids) != self.batch_size:
                raise ValueError("position_ids length does not match graph batch size")
            self._copy_list_to_static_input(
                position_ids,
                self._position_ids_cpu,
                self._position_ids_cpu_np,
                self._position_ids_flat,
            )
        input_wall = time.perf_counter() - input_start
        metadata_start = time.perf_counter()
        metadata_stats = self._copy_token_pool_decode_context(token_pool_decode)
        metadata_wall = time.perf_counter() - metadata_start
        replay_start = time.perf_counter()
        self.graph.replay()
        replay_wall = time.perf_counter() - replay_start
        self.cache.record_static_padded_decode_replay()
        if self._records_token_pool_decode_steps:
            self.cache.record_token_pool_covered_decode_step()
        self.last_decode_info = {
            "cuda_graph_input_copy_wall_s": input_wall,
            "cuda_graph_metadata_copy_wall_s": metadata_wall,
            "cuda_graph_replay_wall_s": replay_wall,
            "cuda_graph_decode_wall_s_total": time.perf_counter() - total_start,
            **metadata_stats,
        }
        if self.logits is None:
            raise AssertionError("graph logits buffer was not initialized")
        return self.logits

    def _copy_token_pool_decode_context(self, token_pool_decode) -> dict[str, int]:
        try:
            return self._token_pool_metadata.copy_compatible_from(token_pool_decode)
        except ValueError as exc:
            raise DistinctCacheBatchError(str(exc)) from exc


def config_from_hf(hf_config) -> GemmaRoutedSpanConfig:
    decoder = hf_config.get_text_config(decoder=True) if hasattr(hf_config, "get_text_config") else hf_config
    return gemma4_e4b_routed_span_config(
        num_hidden_layers=decoder.num_hidden_layers,
        num_kv_shared_layers=getattr(decoder, "num_kv_shared_layers", 0),
        layer_types=tuple(decoder.layer_types),
        num_kv_heads=getattr(decoder, "num_global_key_value_heads", None)
        or getattr(decoder, "num_key_value_heads", 2),
        head_dim=getattr(decoder, "global_head_dim", None) or getattr(decoder, "head_dim", 512),
        sliding_window=getattr(decoder, "sliding_window", None) or 1024,
    )


def _wrap_native_gemma_forward(
    model,
    *,
    native_attention_backend: str = "manual",
    native_projection_backend: str = "separate",
    native_weight_backend: str = "hf_live",
    native_release_hf_decoder_layers: bool = False,
):
    if getattr(model, "wkvm_no_hf_transformer_forward", False):
        current_backend = getattr(model, "native_attention_backend", native_attention_backend)
        current_projection_backend = getattr(
            model,
            "native_projection_backend",
            native_projection_backend,
        )
        current_weight_backend = getattr(
            model,
            "native_weight_backend",
            native_weight_backend,
        )
        current_release_hf_decoder_layers = bool(
            getattr(
                model,
                "release_hf_decoder_layers",
                native_release_hf_decoder_layers,
            )
        )
        if current_backend != native_attention_backend:
            raise ValueError(
                "model is already wrapped with native Gemma attention backend "
                f"{current_backend!r}, requested {native_attention_backend!r}"
            )
        if current_projection_backend != native_projection_backend:
            raise ValueError(
                "model is already wrapped with native Gemma projection backend "
                f"{current_projection_backend!r}, requested {native_projection_backend!r}"
            )
        if current_weight_backend != native_weight_backend:
            raise ValueError(
                "model is already wrapped with native Gemma weight backend "
                f"{current_weight_backend!r}, requested {native_weight_backend!r}"
            )
        if current_release_hf_decoder_layers != bool(native_release_hf_decoder_layers):
            raise ValueError(
                "model is already wrapped with native Gemma HF decoder release "
                f"{current_release_hf_decoder_layers!r}, requested "
                f"{bool(native_release_hf_decoder_layers)!r}"
            )
        return model
    from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM

    return NativeGemma4ForCausalLM(
        model,
        native_attention_backend=native_attention_backend,
        native_projection_backend=native_projection_backend,
        native_weight_backend=native_weight_backend,
        release_hf_decoder_layers=native_release_hf_decoder_layers,
    )


def _cat_time(a, b):
    import torch

    if a is None or a.numel() == 0:
        return b.contiguous()
    return torch.cat([a, b], dim=2).contiguous()


def _cat_nonempty(parts):
    import torch

    nonempty = [p for p in parts if p is not None and p.numel()]
    if not nonempty:
        ref = next(p for p in parts if p is not None)
        return ref
    return torch.cat(nonempty, dim=2).contiguous()


def _cat_batch(tensors: list[Any], what: str):
    import torch

    _check_batch_shapes(tensors, what)
    return torch.cat(tensors, dim=0).contiguous()


def _check_batch_shapes(tensors: list[Any], what: str) -> None:
    if any(t is None for t in tensors):
        raise DistinctCacheBatchError(f"{what}: missing tensor")
    tail = tensors[0].shape[1:]
    if any(t.shape[0] != 1 or t.shape[1:] != tail for t in tensors):
        raise DistinctCacheBatchError(
            f"{what}: incompatible shapes {[tuple(t.shape) for t in tensors[:4]]}"
        )


def _row_slices(tensor, batch: int, what: str) -> list[Any]:
    if tensor is None:
        raise DistinctCacheBatchError(f"{what}: missing merged tensor")
    if tensor.shape[0] != batch:
        raise DistinctCacheBatchError(
            f"{what}: merged batch {tensor.shape[0]} does not match {batch}"
        )
    return [tensor[i : i + 1].contiguous() for i in range(batch)]


def _routed_span_signature(layer: NativeRoutedSpanLayer) -> tuple[Any, ...]:
    slots = []
    for spans in layer._slot_spans:
        slots.append(tuple(tuple(span["pos"]) for span in spans))
    return (
        layer.cumulative_length,
        layer._evicted,
        layer._n_active,
        tuple(layer._slot_cnt),
        layer._sink_k.shape[2],
        layer._pend_k.shape[2],
        layer._ring_k.shape[2],
        tuple(slots),
    )


def _cache_layer_type(hf_config, layer_idx: int) -> str | None:
    layer_types = getattr(hf_config, "layer_types", None)
    if layer_types is None or int(layer_idx) >= len(layer_types):
        return None
    return layer_types[int(layer_idx)]


class _TokenPoolCoveredDecodeLayer(_NativeGemmaLayer):
    """Placeholder for padded layers served from TokenKVPool."""

    def __init__(
        self,
        *,
        cumulative_length: int,
        is_sliding: bool,
        sliding_window: int | None,
        layer_type: str,
        dtype,
        device,
        reserved_decode_steps: int = 1,
    ) -> None:
        super().__init__()
        self.cumulative_length = int(cumulative_length)
        self.is_sliding = bool(is_sliding)
        self.sliding_window = None if sliding_window is None else int(sliding_window)
        self.layer_type = str(layer_type)
        self.dtype = dtype
        self.device = device
        self.is_initialized = True
        self._reserved_decode_steps = max(1, int(reserved_decode_steps))
        self._consumed_decode_steps = 0

    def update(self, key_states, value_states, *args, **kwargs):
        raise DistinctCacheBatchError(
            f"{self.layer_type} is covered by token-pool decode and cannot use dense cache update"
        )

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return query_length, 0

    def get_max_cache_shape(self) -> int:
        if self.sliding_window is not None:
            return self.sliding_window
        return self.cumulative_length

    def remaining_capacity(self) -> int:
        return max(0, self._reserved_decode_steps - self._consumed_decode_steps)

    def record_decode_step(self, steps: int = 1) -> None:
        steps = int(steps)
        if steps < 1:
            return
        self._consumed_decode_steps += steps
        self.cumulative_length += steps


class _PaddedDecodeLayer(_NativeGemmaLayer):
    """Temporary padded layer for one ragged decode forward."""

    def __init__(
        self,
        keys,
        values,
        valid_mask,
        *,
        cumulative_length: int,
        is_sliding: bool,
        pending_tail: int | None = None,
        route_chunk: int | None = None,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.values = values
        self.valid_mask = valid_mask.contiguous()
        self._past_width = int(valid_mask.shape[1])
        self._write_width = self._past_width
        self.cumulative_length = int(cumulative_length)
        self.is_sliding = bool(is_sliding)
        self.pending_tail = None if pending_tail is None else int(pending_tail)
        self.route_chunk = None if route_chunk is None else int(route_chunk)
        self.dtype = keys.dtype
        self.device = keys.device
        self.is_initialized = True
        self._last_key_states = None
        self._last_value_states = None

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.shape[-2] != 1:
            raise NotImplementedError("padded native Gemma cache is decode-only")
        if (
            self.pending_tail is not None
            and self.route_chunk is not None
            and self.pending_tail + key_states.shape[-2] >= self.route_chunk
        ):
            raise NotImplementedError("padded decode would cross routed-span fold boundary")
        self._last_key_states = key_states.detach()
        self._last_value_states = value_states.detach()
        end = self._write_width + key_states.shape[-2]
        if end > self.keys.shape[2]:
            raise DistinctCacheBatchError("padded decode layer reserved width exhausted")
        self.keys[:, :, self._write_width : end, :].copy_(key_states)
        self.values[:, :, self._write_width : end, :].copy_(value_states)
        self._write_width = end
        self.cumulative_length += key_states.shape[-2]
        if self.pending_tail is not None:
            self.pending_tail += key_states.shape[-2]
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self._past_width + query_length, 0

    def get_max_cache_shape(self) -> int:
        return int(self.keys.shape[2])

    def commit_into(self, layers: list[_NativeGemmaLayer]) -> None:
        if self._last_key_states is None or self._last_value_states is None:
            raise DistinctCacheBatchError("padded decode layer has no update to commit")
        if len(layers) != self._last_key_states.shape[0]:
            raise DistinctCacheBatchError("padded commit batch size mismatch")
        for row, layer in enumerate(layers):
            key_states = self._last_key_states[row : row + 1]
            value_states = self._last_value_states[row : row + 1]
            commit_decode_token = getattr(layer, "commit_decode_token", None)
            if commit_decode_token is not None and commit_decode_token(key_states, value_states):
                continue
            layer.update(key_states, value_states)


class _PersistentPaddedDecodeLayer(_PaddedDecodeLayer):
    """Temporary padded layer kept alive across several decode forwards."""

    def __init__(
        self,
        keys,
        values,
        valid_mask,
        *,
        initial_write_width: int,
        cumulative_length: int,
        is_sliding: bool,
        pending_tail: int | None = None,
        route_chunk: int | None = None,
        static_width: bool = False,
    ) -> None:
        super().__init__(
            keys,
            values,
            valid_mask,
            cumulative_length=cumulative_length,
            is_sliding=is_sliding,
            pending_tail=pending_tail,
            route_chunk=route_chunk,
        )
        self._initial_write_width = int(initial_write_width)
        self._write_width = int(initial_write_width)
        self._past_width = int(initial_write_width)
        if self._write_width < 0 or self._write_width >= int(keys.shape[2]):
            raise DistinctCacheBatchError("persistent padded decode has invalid write width")
        self._static_width = bool(static_width)
        self._static_write_index = None
        self._static_attention_mask = None
        self._static_valid_mask_updates_enabled = True
        if self._static_width:
            import torch

            self._static_write_index = torch.tensor(
                [self._write_width],
                dtype=torch.long,
                device=keys.device,
            )

    def update(self, key_states, value_states, *args, **kwargs):
        if key_states.shape[-2] != 1:
            raise NotImplementedError("persistent padded Gemma cache is decode-only")
        if (
            self.pending_tail is not None
            and self.route_chunk is not None
            and self.pending_tail + key_states.shape[-2] >= self.route_chunk
        ):
            raise NotImplementedError("persistent padded decode would cross routed-span fold boundary")
        self._last_key_states = key_states.detach()
        self._last_value_states = value_states.detach()
        end = self._write_width + key_states.shape[-2]
        if end > self.keys.shape[2]:
            raise DistinctCacheBatchError("persistent padded decode capacity exhausted")
        if self._static_width:
            if self._static_write_index is None:
                raise DistinctCacheBatchError("static persistent padded decode has no write index")
            self.keys.index_copy_(2, self._static_write_index, key_states)
            self.values.index_copy_(2, self._static_write_index, value_states)
            if self._static_valid_mask_updates_enabled:
                self.valid_mask.index_fill_(1, self._static_write_index, True)
            if self._static_attention_mask is not None:
                self._static_attention_mask.index_fill_(3, self._static_write_index, 0.0)
            self._static_write_index.copy_(self._static_write_index + key_states.shape[-2])
        else:
            self.keys[:, :, self._write_width : end, :].copy_(key_states)
            self.values[:, :, self._write_width : end, :].copy_(value_states)
            self.valid_mask[:, self._write_width : end] = True
        self._write_width = end
        self._past_width = end
        self.cumulative_length += key_states.shape[-2]
        if self.pending_tail is not None:
            self.pending_tail += key_states.shape[-2]
        if self._static_width:
            return self.keys, self.values
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if self._static_width:
            return int(self.keys.shape[2]), 0
        return self._write_width + query_length, 0

    def get_max_cache_shape(self) -> int:
        return int(self.keys.shape[2])

    def remaining_capacity(self) -> int:
        return max(0, int(self.keys.shape[2]) - int(self._write_width))

    def commit_into(self, layers: list[_NativeGemmaLayer]) -> None:
        appended = self._write_width - self._initial_write_width
        if appended < 1:
            return
        if len(layers) != self.keys.shape[0]:
            raise DistinctCacheBatchError("persistent padded commit batch size mismatch")
        for row, layer in enumerate(layers):
            key_states = self.keys[
                row : row + 1,
                :,
                self._initial_write_width : self._write_width,
                :,
            ]
            value_states = self.values[
                row : row + 1,
                :,
                self._initial_write_width : self._write_width,
                :,
            ]
            commit_decode_tokens = getattr(layer, "commit_decode_tokens", None)
            if commit_decode_tokens is not None and commit_decode_tokens(key_states, value_states):
                continue
            if appended == 1:
                commit_decode_token = getattr(layer, "commit_decode_token", None)
                if commit_decode_token is not None and commit_decode_token(key_states, value_states):
                    continue
            layer.update(key_states.contiguous(), value_states.contiguous())

    def set_static_attention_mask(self, mask) -> None:
        if not self._static_width:
            raise DistinctCacheBatchError("cannot attach static mask to dynamic padded layer")
        self._static_attention_mask = mask

    def set_static_valid_mask_updates_enabled(self, enabled: bool) -> None:
        if not self._static_width:
            raise DistinctCacheBatchError("dynamic padded layer has no static valid mask mode")
        self._static_valid_mask_updates_enabled = bool(enabled)

    def static_attention_mask(self):
        return self._static_attention_mask

    def snapshot_static_state(self, *, include_kv: bool = False):
        if not self._static_width:
            raise DistinctCacheBatchError("dynamic persistent padded layer has no static snapshot")
        return (
            self.keys.clone() if include_kv else None,
            self.values.clone() if include_kv else None,
            self.valid_mask.clone(),
            None if self._static_write_index is None else self._static_write_index.clone(),
            None if self._static_attention_mask is None else self._static_attention_mask.clone(),
            self._write_width,
            self._past_width,
            self.cumulative_length,
            self.pending_tail,
        )

    def restore_static_state(self, snap) -> None:
        (
            keys,
            values,
            valid_mask,
            write_index,
            attention_mask,
            write_width,
            past_width,
            cumulative_length,
            pending_tail,
        ) = snap
        if keys is not None:
            self.keys.copy_(keys)
        if values is not None:
            self.values.copy_(values)
        self.valid_mask.copy_(valid_mask)
        if self._static_write_index is not None and write_index is not None:
            self._static_write_index.copy_(write_index)
        if self._static_attention_mask is not None and attention_mask is not None:
            self._static_attention_mask.copy_(attention_mask)
        self._write_width = int(write_width)
        self._past_width = int(past_width)
        self.cumulative_length = int(cumulative_length)
        self.pending_tail = None if pending_tail is None else int(pending_tail)

    def record_static_replay(self, steps: int = 1) -> None:
        if not self._static_width:
            raise DistinctCacheBatchError("dynamic persistent padded layer has no static replay bookkeeping")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if (
            self.pending_tail is not None
            and self.route_chunk is not None
            and self.pending_tail + steps >= self.route_chunk
        ):
            raise NotImplementedError("static padded decode replay would cross routed-span fold boundary")
        end = self._write_width + int(steps)
        if end > self.keys.shape[2]:
            raise DistinctCacheBatchError("static padded decode replay capacity exhausted")
        self._write_width = end
        self._past_width = end
        self.cumulative_length += int(steps)
        if self.pending_tail is not None:
            self.pending_tail += int(steps)


def _pad_kv_and_stack(
    key_tensors: list[Any],
    value_tensors: list[Any],
    what: str,
    *,
    reserve_steps: int,
    workspace: PaddedDecodeWorkspace | None = None,
    layer_idx: int | None = None,
):
    import torch

    if reserve_steps < 0:
        raise ValueError("reserve_steps must be >= 0")
    if workspace is not None and reserve_steps != 1:
        raise ValueError("workspace padded decode currently requires reserve_steps=1")
    if len(key_tensors) != len(value_tensors):
        raise DistinctCacheBatchError(f"{what}: key/value batch size mismatch")
    if any(t is None for t in key_tensors) or any(t is None for t in value_tensors):
        raise DistinctCacheBatchError(f"{what}: missing tensor")
    batch = len(key_tensors)
    heads = key_tensors[0].shape[1]
    dim = key_tensors[0].shape[3]
    dtype = key_tensors[0].dtype
    device = key_tensors[0].device
    lengths = [int(t.shape[2]) for t in key_tensors]
    max_len = max(lengths)
    width = max_len + reserve_steps
    workspace_info: dict[str, int] = {
        "workspace_reused": 0,
        "workspace_allocated": 0,
        "workspace_capacity_width": width,
        "workspace_width_bucket": 0,
    }
    if workspace is None:
        keys = torch.empty(batch, heads, width, dim, dtype=dtype, device=device)
        values = torch.empty(batch, heads, width, dim, dtype=dtype, device=device)
        mask = torch.empty(batch, max_len, dtype=torch.bool, device=device)
    else:
        keys, values, mask, workspace_info = workspace.borrow(
            layer_idx=-1 if layer_idx is None else layer_idx,
            batch=batch,
            heads=heads,
            width=width,
            head_dim=dim,
            dtype=dtype,
            device=device,
        )
        width = int(keys.shape[2])
    past_width = int(mask.shape[1])
    for row, (key_tensor, value_tensor) in enumerate(zip(key_tensors, value_tensors)):
        expected = (1, heads, lengths[row], dim)
        if tuple(key_tensor.shape) != expected or tuple(value_tensor.shape) != expected:
            raise DistinctCacheBatchError(
                f"{what}: incompatible shapes "
                f"{[(tuple(k.shape), tuple(v.shape)) for k, v in zip(key_tensors[:4], value_tensors[:4])]}"
            )
        if key_tensor.dtype != dtype or value_tensor.dtype != dtype:
            raise DistinctCacheBatchError(f"{what}: key/value dtypes differ")
        if key_tensor.device != device or value_tensor.device != device:
            raise DistinctCacheBatchError(f"{what}: key/value devices differ")
        length = lengths[row]
        keys[row, :, :length].copy_(key_tensor[0])
        values[row, :, :length].copy_(value_tensor[0])
        if length < past_width:
            keys[row, :, length:past_width].zero_()
            values[row, :, length:past_width].zero_()
        mask[row, :length] = True
        if length < mask.shape[1]:
            mask[row, length:].zero_()
    return keys, values, mask, workspace_info


def _pad_kv_and_stack_persistent(
    key_tensors: list[Any],
    value_tensors: list[Any],
    what: str,
    *,
    reserve_steps: int,
):
    import torch

    if reserve_steps < 1:
        raise ValueError("reserve_steps must be >= 1")
    if len(key_tensors) != len(value_tensors):
        raise DistinctCacheBatchError(f"{what}: key/value batch size mismatch")
    if any(t is None for t in key_tensors) or any(t is None for t in value_tensors):
        raise DistinctCacheBatchError(f"{what}: missing tensor")
    batch = len(key_tensors)
    heads = key_tensors[0].shape[1]
    dim = key_tensors[0].shape[3]
    dtype = key_tensors[0].dtype
    device = key_tensors[0].device
    lengths = [int(t.shape[2]) for t in key_tensors]
    max_len = max(lengths)
    width = max_len + reserve_steps
    keys = torch.empty(batch, heads, width, dim, dtype=dtype, device=device)
    values = torch.empty(batch, heads, width, dim, dtype=dtype, device=device)
    mask = torch.zeros(batch, width, dtype=torch.bool, device=device)
    for row, (key_tensor, value_tensor) in enumerate(zip(key_tensors, value_tensors)):
        expected = (1, heads, lengths[row], dim)
        if tuple(key_tensor.shape) != expected or tuple(value_tensor.shape) != expected:
            raise DistinctCacheBatchError(
                f"{what}: incompatible shapes "
                f"{[(tuple(k.shape), tuple(v.shape)) for k, v in zip(key_tensors[:4], value_tensors[:4])]}"
            )
        if key_tensor.dtype != dtype or value_tensor.dtype != dtype:
            raise DistinctCacheBatchError(f"{what}: key/value dtypes differ")
        if key_tensor.device != device or value_tensor.device != device:
            raise DistinctCacheBatchError(f"{what}: key/value devices differ")
        length = lengths[row]
        keys[row, :, :length].copy_(key_tensor[0])
        values[row, :, :length].copy_(value_tensor[0])
        if length < max_len:
            keys[row, :, length:max_len].zero_()
            values[row, :, length:max_len].zero_()
        mask[row, :length] = True
    return keys, values, mask, {
        "workspace_reused": 0,
        "workspace_allocated": 0,
        "workspace_bypassed": 0,
        "workspace_capacity_width": int(width),
        "workspace_width_bucket": 0,
    }


def _padded_decode_temp_stats(
    keys,
    values,
    valid_mask,
    lengths: list[int],
    *,
    workspace_info: dict[str, int] | None = None,
) -> dict[str, int]:
    batch = int(keys.shape[0])
    heads = int(keys.shape[1])
    width = int(keys.shape[2])
    head_dim = int(keys.shape[3])
    source_materialized_width = max(int(length) for length in lengths)
    temporary_past_width = int(valid_mask.shape[1])
    key_elem = int(keys.element_size())
    value_elem = int(values.element_size())
    mask_elem = int(valid_mask.element_size())
    kv_bytes_per_slot = heads * head_dim * (key_elem + value_elem)
    pad_slots_total = int(sum(temporary_past_width - length for length in lengths))
    source_pad_slots_total = int(sum(source_materialized_width - length for length in lengths))
    workspace_extra_pad_slots_total = pad_slots_total - source_pad_slots_total
    copied_slots_total = int(sum(lengths))
    reserved_decode_slots_total = batch * max(0, width - temporary_past_width)
    kv_bytes = batch * width * kv_bytes_per_slot
    mask_bytes = int(valid_mask.numel()) * mask_elem
    stats = {
        "temporary_kv_bytes": kv_bytes,
        "temporary_mask_bytes": mask_bytes,
        "temporary_total_bytes": kv_bytes + mask_bytes,
        "copied_kv_bytes": copied_slots_total * kv_bytes_per_slot,
        "padded_kv_bytes": pad_slots_total * kv_bytes_per_slot,
        "source_padded_kv_bytes": source_pad_slots_total * kv_bytes_per_slot,
        "workspace_extra_padded_kv_bytes": workspace_extra_pad_slots_total * kv_bytes_per_slot,
        "reserved_decode_kv_bytes": reserved_decode_slots_total * kv_bytes_per_slot,
        "copied_slots_total": copied_slots_total,
        "pad_slots_total": pad_slots_total,
        "source_pad_slots_total": source_pad_slots_total,
        "workspace_extra_pad_slots_total": workspace_extra_pad_slots_total,
        "reserved_decode_slots_total": reserved_decode_slots_total,
        "batch_rows": batch,
        "heads": heads,
        "head_dim": head_dim,
        "source_materialized_slots_max": source_materialized_width,
        "temporary_past_slots": temporary_past_width,
        "materialized_slots_max": temporary_past_width,
        "materialized_slots_min": int(min(lengths)),
        "reserved_decode_slots": max(0, width - temporary_past_width),
    }
    if workspace_info:
        stats.update(workspace_info)
    return stats


def _validate_shared_attention_masks(layers: list[_NativeGemmaLayer]) -> None:
    for is_sliding in (False, True):
        masks = [
            layer.valid_mask
            for layer in layers
            if isinstance(layer, _PaddedDecodeLayer) and layer.is_sliding is is_sliding
        ]
        if not masks:
            continue
        first = masks[0]
        for mask in masks[1:]:
            if mask.shape != first.shape:
                kind = "sliding" if is_sliding else "full"
                raise DistinctCacheBatchError(
                    f"padded {kind} layers need one shared mask shape"
                )
            import torch

            if not torch.equal(mask, first):
                kind = "sliding" if is_sliding else "full"
                raise DistinctCacheBatchError(
                    f"padded {kind} layers need one shared mask"
                )
        persistent_layers = [
            layer
            for layer in layers
            if isinstance(layer, _PersistentPaddedDecodeLayer) and layer.is_sliding is is_sliding
        ]
        if persistent_layers:
            write_width = persistent_layers[0]._write_width
            if any(layer._write_width != write_width for layer in persistent_layers[1:]):
                kind = "sliding" if is_sliding else "full"
                raise DistinctCacheBatchError(
                    f"persistent padded {kind} layers need one shared write width"
                )


def _attach_static_padded_attention_masks(layers: list[_NativeGemmaLayer]) -> None:
    import torch

    for is_sliding in (False, True):
        static_layers = [
            layer
            for layer in layers
            if (
                isinstance(layer, _PersistentPaddedDecodeLayer)
                and layer.is_sliding is is_sliding
                and layer._static_width
            )
        ]
        if not static_layers:
            continue
        first = static_layers[0]
        visible = first.valid_mask[:, None, None, :]
        mask = torch.where(
            visible,
            torch.tensor(0.0, dtype=first.keys.dtype, device=first.keys.device),
            torch.tensor(torch.finfo(first.keys.dtype).min, dtype=first.keys.dtype, device=first.keys.device),
        ).contiguous()
        static_layers[0].set_static_attention_mask(mask)


def _mask_from_padded_layers(
    layers: list[_NativeGemmaLayer],
    *,
    is_sliding: bool,
    static: bool = False,
):
    import torch

    padded_layers = [
        layer
        for layer in layers
        if isinstance(layer, _PaddedDecodeLayer) and layer.is_sliding is is_sliding
    ]
    if not padded_layers:
        return None
    first_layer = padded_layers[0]
    if static and isinstance(first_layer, _PersistentPaddedDecodeLayer):
        mask = first_layer.static_attention_mask()
        if mask is not None:
            return mask
    first = first_layer.valid_mask
    if isinstance(first_layer, _PersistentPaddedDecodeLayer):
        write_width = first_layer._write_width
        current = torch.ones(first.shape[0], 1, dtype=torch.bool, device=first.device)
        visible = torch.cat([first[:, :write_width], current], dim=1)[:, None, None, :]
    else:
        current = torch.ones(first.shape[0], 1, dtype=torch.bool, device=first.device)
        visible = torch.cat([first, current], dim=1)[:, None, None, :]
    dtype = next(
        layer.keys.dtype
        for layer in padded_layers
    )
    return torch.where(
        visible,
        torch.tensor(0.0, dtype=dtype, device=first.device),
        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=first.device),
    )


def _check_sliding_layer_exact(
    layers: list[NativeSlidingWindowLayer],
    layer_idx: int,
) -> dict[str, Any]:
    base = layers[0]
    if not base.is_initialized:
        return {"merge": "uninitialized_skip"}
    if any(not layer.is_initialized for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: mixed initialization state")
    if any(layer.cumulative_length != base.cumulative_length for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: cumulative lengths differ")
    if any(layer.sliding_window != base.sliding_window for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: sliding windows differ")
    _check_batch_shapes([layer.keys for layer in layers], f"layer{layer_idx}.keys")
    _check_batch_shapes([layer.values for layer in layers], f"layer{layer_idx}.values")
    return {
        "merge": "tensor_concat",
        "materialized_slots": int(base.keys.shape[2]),
    }


def _merge_sliding_layer_exact(
    layers: list[NativeSlidingWindowLayer],
    layer_idx: int,
) -> None:
    base = layers[0]
    if not base.is_initialized:
        return
    base.keys = _cat_batch([layer.keys for layer in layers], f"layer{layer_idx}.keys")
    base.values = _cat_batch([layer.values for layer in layers], f"layer{layer_idx}.values")


def _merge_sliding_layer_padded(
    layers: list[NativeSlidingWindowLayer],
    layer_idx: int,
    *,
    workspace: PaddedDecodeWorkspace | None = None,
    reserve_steps: int = 1,
    persistent: bool = False,
    graph_static: bool = False,
) -> tuple[_PaddedDecodeLayer, dict[str, Any]]:
    base = layers[0]
    if not base.is_initialized:
        raise DistinctCacheBatchError(f"layer {layer_idx}: uninitialized sliding layer")
    if any(not layer.is_initialized for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: mixed initialization state")
    if any(layer.sliding_window != base.sliding_window for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: sliding windows differ")
    lengths = [int(layer.keys.shape[2]) for layer in layers]
    if persistent:
        keys, values, valid, workspace_info = _pad_kv_and_stack_persistent(
            [layer.keys for layer in layers],
            [layer.values for layer in layers],
            f"layer{layer_idx}",
            reserve_steps=reserve_steps,
        )
        layer = _PersistentPaddedDecodeLayer(
            keys,
            values,
            valid,
            initial_write_width=max(lengths),
            cumulative_length=max(layer.cumulative_length for layer in layers),
            is_sliding=True,
            static_width=graph_static,
        )
    else:
        keys, values, valid, workspace_info = _pad_kv_and_stack(
            [layer.keys for layer in layers],
            [layer.values for layer in layers],
            f"layer{layer_idx}",
            reserve_steps=1,
            workspace=workspace,
            layer_idx=layer_idx,
        )
        layer = _PaddedDecodeLayer(
            keys,
            values,
            valid,
            cumulative_length=max(layer.cumulative_length for layer in layers),
            is_sliding=True,
        )
    return (
        layer,
        {
            "merge": "padded_valid_mask_concat",
            **_padded_decode_temp_stats(
                keys,
                values,
                valid,
                lengths,
                workspace_info=workspace_info,
            ),
        },
    )


def _merge_token_pool_covered_sliding_layer(
    layers: list[NativeSlidingWindowLayer],
    layer_idx: int,
    *,
    decode_steps: int = 1,
    layer_type: str,
) -> tuple[_TokenPoolCoveredDecodeLayer, dict[str, Any]]:
    base = layers[0]
    if not base.is_initialized:
        raise DistinctCacheBatchError(f"layer {layer_idx}: uninitialized sliding layer")
    if any(not layer.is_initialized for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: mixed initialization state")
    if any(layer.sliding_window != base.sliding_window for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: sliding windows differ")
    if any((layer.keys is None) != (layer.values is None) for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: partial sliding KV")
    dtype = base.dtype
    device = base.device
    sample_keys = None
    for layer in layers:
        if layer.keys is not None:
            sample_keys = layer.keys
            dtype = layer.keys.dtype
            device = layer.keys.device
            break
    if dtype is None or device is None:
        raise DistinctCacheBatchError(f"layer {layer_idx}: missing sliding KV metadata")
    lengths = [0 if layer.keys is None else int(layer.keys.shape[2]) for layer in layers]
    layer = _TokenPoolCoveredDecodeLayer(
        cumulative_length=max(layer.cumulative_length for layer in layers),
        is_sliding=True,
        sliding_window=base.sliding_window,
        layer_type=layer_type,
        dtype=dtype,
        device=device,
        reserved_decode_steps=decode_steps,
    )
    return (
        layer,
        {
            "merge": "token_pool_covered_skip",
            "temporary_kv_bytes": 0,
            "temporary_mask_bytes": 0,
            "temporary_total_bytes": 0,
            "copied_kv_bytes": 0,
            "padded_kv_bytes": 0,
            "source_padded_kv_bytes": 0,
            "workspace_extra_padded_kv_bytes": 0,
            "reserved_decode_kv_bytes": 0,
            "copied_slots_total": 0,
            "pad_slots_total": 0,
            "source_pad_slots_total": 0,
            "workspace_extra_pad_slots_total": 0,
            "reserved_decode_slots_total": 0,
            "batch_rows": len(layers),
            "heads": 0 if sample_keys is None else int(sample_keys.shape[1]),
            "head_dim": 0 if sample_keys is None else int(sample_keys.shape[3]),
            "source_materialized_slots_max": max(lengths),
            "temporary_past_slots": 0,
            "materialized_slots_max": 0,
            "materialized_slots_min": 0,
            "reserved_decode_slots": 0,
            "workspace_reused": 0,
            "workspace_allocated": 0,
            "workspace_bypassed": 0,
            "workspace_capacity_width": 0,
            "workspace_width_bucket": 0,
            "token_pool_covered_layer_type": layer_type,
        },
    )


def _merge_token_pool_covered_routed_layer(
    layers: list[NativeRoutedSpanLayer],
    layer_idx: int,
    *,
    decode_steps: int = 1,
    layer_type: str,
) -> tuple[_TokenPoolCoveredDecodeLayer, dict[str, Any]]:
    base = layers[0]
    if not base.is_initialized:
        raise DistinctCacheBatchError(f"layer {layer_idx}: uninitialized routed-span layer")
    if any(not layer.is_initialized for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: mixed initialization state")
    if any(layer.route_chunk != base.route_chunk for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: route_chunk differs")
    dtype = base.dtype
    device = base.device
    sample_keys = None
    for layer in layers:
        if layer.keys is not None:
            sample_keys = layer.keys
            dtype = layer.keys.dtype
            device = layer.keys.device
            break
    if dtype is None or device is None:
        raise DistinctCacheBatchError(f"layer {layer_idx}: missing routed KV metadata")
    lengths = [0 if layer.keys is None else int(layer.keys.shape[2]) for layer in layers]
    pending_tail = max(int(layer._pend_k.shape[2]) for layer in layers)
    route_chunk = int(base.route_chunk)
    if pending_tail + int(decode_steps) >= route_chunk:
        raise DistinctCacheBatchError(
            f"layer {layer_idx}: pending tail {pending_tail} + "
            f"decode {int(decode_steps)} reaches route_chunk {route_chunk}"
        )
    layer = _TokenPoolCoveredDecodeLayer(
        cumulative_length=max(layer.cumulative_length for layer in layers),
        is_sliding=False,
        sliding_window=None,
        layer_type=layer_type,
        dtype=dtype,
        device=device,
        reserved_decode_steps=decode_steps,
    )
    return (
        layer,
        {
            "merge": "token_pool_covered_skip",
            "temporary_kv_bytes": 0,
            "temporary_mask_bytes": 0,
            "temporary_total_bytes": 0,
            "copied_kv_bytes": 0,
            "padded_kv_bytes": 0,
            "source_padded_kv_bytes": 0,
            "workspace_extra_padded_kv_bytes": 0,
            "reserved_decode_kv_bytes": 0,
            "copied_slots_total": 0,
            "pad_slots_total": 0,
            "source_pad_slots_total": 0,
            "workspace_extra_pad_slots_total": 0,
            "reserved_decode_slots_total": 0,
            "batch_rows": len(layers),
            "heads": 0 if sample_keys is None else int(sample_keys.shape[1]),
            "head_dim": 0 if sample_keys is None else int(sample_keys.shape[3]),
            "source_materialized_slots_max": max(lengths),
            "temporary_past_slots": 0,
            "materialized_slots_max": 0,
            "materialized_slots_min": 0,
            "reserved_decode_slots": 0,
            "workspace_reused": 0,
            "workspace_allocated": 0,
            "workspace_bypassed": 0,
            "workspace_capacity_width": 0,
            "workspace_width_bucket": 0,
            "pending_tail": pending_tail,
            "route_chunk": route_chunk,
            "token_pool_covered_layer_type": layer_type,
        },
    )


def _check_routed_span_layer_exact(
    layers: list[NativeRoutedSpanLayer],
    decode_steps: int,
    layer_idx: int,
) -> dict[str, Any]:
    base = layers[0]
    if not base.is_initialized:
        return {"merge": "uninitialized_skip"}
    if any(not layer.is_initialized for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: mixed initialization state")
    signature = _routed_span_signature(base)
    for row, layer in enumerate(layers[1:], start=1):
        if _routed_span_signature(layer) != signature:
            raise DistinctCacheBatchError(
                f"layer {layer_idx}: routed-span layout differs at row {row}"
            )
    if base._pend_k.shape[2] + decode_steps >= base.route_chunk:
        raise DistinctCacheBatchError(
            f"layer {layer_idx}: pending tail {base._pend_k.shape[2]} + "
            f"decode {decode_steps} reaches route_chunk {base.route_chunk}"
        )

    for attr in (
        "keys",
        "values",
        "_sink_k",
        "_sink_v",
        "_ring_k",
        "_ring_v",
        "_pend_k",
        "_pend_v",
        "_slot_mk",
        "_slot_mv",
    ):
        _check_batch_shapes([getattr(layer, attr) for layer in layers], attr)
    for slot_id, spans in enumerate(base._slot_spans):
        for span_idx, span in enumerate(spans):
            _check_batch_shapes(
                [layer._slot_spans[slot_id][span_idx]["k"] for layer in layers],
                f"layer{layer_idx}.slot{slot_id}.span{span_idx}.k",
            )
            _check_batch_shapes(
                [layer._slot_spans[slot_id][span_idx]["v"] for layer in layers],
                f"layer{layer_idx}.slot{slot_id}.span{span_idx}.v",
            )
    return {
        "merge": "exact_structural_concat",
        "materialized_slots": int(base.keys.shape[2]),
        "pending_tail": int(base._pend_k.shape[2]),
        "bank_slots": int(base.n_bank_slots()),
    }


def _merge_routed_span_layer_exact(
    layers: list[NativeRoutedSpanLayer],
    layer_idx: int,
) -> None:
    base = layers[0]
    if not base.is_initialized:
        return
    base.keys = base.values = None
    for attr in (
        "_sink_k",
        "_sink_v",
        "_ring_k",
        "_ring_v",
        "_pend_k",
        "_pend_v",
        "_slot_mk",
        "_slot_mv",
    ):
        setattr(base, attr, _cat_batch([getattr(layer, attr) for layer in layers], attr))
    for slot_id, spans in enumerate(base._slot_spans):
        for span_idx, span in enumerate(spans):
            span["k"] = _cat_batch(
                [layer._slot_spans[slot_id][span_idx]["k"] for layer in layers],
                f"layer{layer_idx}.slot{slot_id}.span{span_idx}.k",
            )
            span["v"] = _cat_batch(
                [layer._slot_spans[slot_id][span_idx]["v"] for layer in layers],
                f"layer{layer_idx}.slot{slot_id}.span{span_idx}.v",
            )
    base._materialize()


def _merge_routed_span_layer_padded(
    layers: list[NativeRoutedSpanLayer],
    decode_steps: int,
    layer_idx: int,
    *,
    workspace: PaddedDecodeWorkspace | None = None,
    persistent: bool = False,
    graph_static: bool = False,
) -> tuple[_PaddedDecodeLayer, dict[str, Any]]:
    base = layers[0]
    if not base.is_initialized:
        raise DistinctCacheBatchError(f"layer {layer_idx}: uninitialized routed-span layer")
    if any(not layer.is_initialized for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: mixed initialization state")
    route_chunk = base.route_chunk
    if any(layer.route_chunk != route_chunk for layer in layers):
        raise DistinctCacheBatchError(f"layer {layer_idx}: route_chunk differs")
    pending_tail = max(int(layer._pend_k.shape[2]) for layer in layers)
    if pending_tail + decode_steps >= route_chunk:
        raise DistinctCacheBatchError(
            f"layer {layer_idx}: pending tail {pending_tail} + "
            f"decode {decode_steps} reaches route_chunk {route_chunk}"
        )
    lengths = [int(layer.keys.shape[2]) for layer in layers]
    if persistent:
        keys, values, valid, workspace_info = _pad_kv_and_stack_persistent(
            [layer.keys for layer in layers],
            [layer.values for layer in layers],
            f"layer{layer_idx}",
            reserve_steps=decode_steps,
        )
        layer = _PersistentPaddedDecodeLayer(
            keys,
            values,
            valid,
            initial_write_width=max(lengths),
            cumulative_length=max(layer.cumulative_length for layer in layers),
            is_sliding=False,
            pending_tail=pending_tail,
            route_chunk=route_chunk,
            static_width=graph_static,
        )
    else:
        keys, values, valid, workspace_info = _pad_kv_and_stack(
            [layer.keys for layer in layers],
            [layer.values for layer in layers],
            f"layer{layer_idx}",
            reserve_steps=1,
            workspace=workspace,
            layer_idx=layer_idx,
        )
        layer = _PaddedDecodeLayer(
            keys,
            values,
            valid,
            cumulative_length=max(layer.cumulative_length for layer in layers),
            is_sliding=False,
            pending_tail=pending_tail,
            route_chunk=route_chunk,
        )
    return (
        layer,
        {
            "merge": "padded_valid_mask_concat",
            **_padded_decode_temp_stats(
                keys,
                values,
                valid,
                lengths,
                workspace_info=workspace_info,
            ),
            "pending_tail": int(pending_tail),
        },
    )


def _split_sliding_layer_exact(
    merged: NativeSlidingWindowLayer,
    layers: list[NativeSlidingWindowLayer],
    batch: int,
    layer_idx: int,
) -> None:
    key_rows = _row_slices(merged.keys, batch, f"layer{layer_idx}.keys")
    value_rows = _row_slices(merged.values, batch, f"layer{layer_idx}.values")
    for row, layer in enumerate(layers):
        layer.keys = key_rows[row]
        layer.values = value_rows[row]
        layer.cumulative_length = merged.cumulative_length
        layer.dtype = merged.dtype
        layer.device = merged.device
        layer.is_initialized = True


def _split_routed_span_layer_exact(
    merged: NativeRoutedSpanLayer,
    layers: list[NativeRoutedSpanLayer],
    batch: int,
    layer_idx: int,
) -> None:
    rows_by_attr = {
        attr: _row_slices(getattr(merged, attr), batch, f"layer{layer_idx}.{attr}")
        for attr in (
            "keys",
            "values",
            "_sink_k",
            "_sink_v",
            "_ring_k",
            "_ring_v",
            "_pend_k",
            "_pend_v",
            "_slot_mk",
            "_slot_mv",
        )
    }
    span_rows: list[list[list[dict[str, Any]]]] = []
    for spans in merged._slot_spans:
        slot_rows: list[list[dict[str, Any]]] = [[] for _ in range(batch)]
        for span in spans:
            key_rows = _row_slices(span["k"], batch, f"layer{layer_idx}.span.k")
            value_rows = _row_slices(span["v"], batch, f"layer{layer_idx}.span.v")
            for row in range(batch):
                slot_rows[row].append(
                    {
                        **span,
                        "k": key_rows[row],
                        "v": value_rows[row],
                    }
                )
        span_rows.append(slot_rows)

    for row, layer in enumerate(layers):
        for attr, values in rows_by_attr.items():
            setattr(layer, attr, values[row])
        layer._slot_spans = [slot[row] for slot in span_rows]
        layer._slot_cnt = list(merged._slot_cnt)
        layer.cumulative_length = merged.cumulative_length
        layer.dtype = merged.dtype
        layer.device = merged.device
        layer.is_initialized = True
