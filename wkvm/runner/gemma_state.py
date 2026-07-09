"""wkvm-owned Gemma routed-span cache/state objects.

These classes are the native state boundary for the Gemma routed-span engine:
request rows point to arena slot ids, and the slot owns ring/pending/span-bank
metadata. The optional tensor payloads are deliberately duck-typed so this
module stays importable without torch; the GPU runner materialises real tensors
behind the same structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from wkvm.models.gemma import GemmaRoutedSpanConfig


@dataclass
class SpanRecord:
    """One retained routed span in a state slot."""

    positions: tuple[int, ...]
    route_slot: int
    feature_kind: str = "value"
    key: Any | None = None
    value: Any | None = None

    def __post_init__(self) -> None:
        if self.feature_kind != "value":
            raise ValueError("routed-span records must route on value features")
        if not self.positions:
            raise ValueError("span must contain at least one token position")
        if self.route_slot < 0:
            raise ValueError("route_slot must be >= 0")

    @property
    def n_tokens(self) -> int:
        return len(self.positions)


@dataclass
class RoutedSpanLayerState:
    """Per-request state for one full-attention KV-owning Gemma layer."""

    layer_id: int
    config: GemmaRoutedSpanConfig
    sink_positions: list[int] = field(default_factory=list)
    ring_positions: list[int] = field(default_factory=list)
    pending_positions: list[int] = field(default_factory=list)
    span_slots: list[list[SpanRecord]] = field(init=False)
    cumulative_length: int = 0

    def __post_init__(self) -> None:
        self.span_slots = [[] for _ in range(self.config.routed_slots)]

    def reset(self) -> None:
        self.sink_positions.clear()
        self.ring_positions.clear()
        self.pending_positions.clear()
        for spans in self.span_slots:
            spans.clear()
        self.cumulative_length = 0

    def ingest_positions(self, positions: list[int], break_mask: list[bool] | None = None) -> None:
        """Bookkeep token positions through sink, ring, pending, and span bank.

        The tensor runner supplies actual K/V payloads; this method mirrors the
        native ownership and routing metadata using positions only, which is
        enough to test capacity and mask invariants without torch.
        """

        for pos in positions:
            if len(self.sink_positions) < self.config.sink_tokens:
                self.sink_positions.append(pos)
            else:
                self.ring_positions.append(pos)
                if len(self.ring_positions) > self.config.ring_tokens:
                    self.pending_positions.append(self.ring_positions.pop(0))
            self.cumulative_length = max(self.cumulative_length, pos + 1)
        self._route_complete_pending_spans(break_mask)

    def add_span(self, positions: tuple[int, ...], route_slot: int, *, feature_kind: str = "value") -> None:
        if route_slot >= self.config.routed_slots:
            raise ValueError("route_slot exceeds routed slot count")
        span = SpanRecord(positions=positions, route_slot=route_slot, feature_kind=feature_kind)
        slot = self.span_slots[route_slot]
        slot.append(span)
        self._enforce_span_budget(slot)

    def materialized_positions(self) -> list[int | None]:
        out: list[int | None] = list(self.sink_positions)
        for slot_id, spans in enumerate(self.span_slots):
            if spans:
                out.append(None)  # slot mean pseudo-token
                for span in spans:
                    out.extend(span.positions)
        out.extend(self.pending_positions)
        out.extend(self.ring_positions)
        return out

    def valid_mask(self, padded_to: int | None = None) -> list[bool]:
        n = len(self.materialized_positions())
        width = n if padded_to is None else padded_to
        if width < n:
            raise ValueError("padded_to shorter than materialized state")
        return [True] * n + [False] * (width - n)

    @property
    def ring_capacity(self) -> int:
        return self.config.ring_tokens

    @property
    def bank_occupancy_tokens(self) -> int:
        return sum(span.n_tokens for spans in self.span_slots for span in spans)

    @property
    def pending_occupancy_tokens(self) -> int:
        return len(self.pending_positions)

    def _route_complete_pending_spans(self, break_mask: list[bool] | None) -> None:
        while len(self.pending_positions) >= self.config.pending_tokens:
            chunk = self.pending_positions[: self.config.pending_tokens]
            route_n = self._routable_prefix_len(chunk, break_mask)
            if route_n <= 0:
                break
            for span in self._split_spans(chunk[:route_n], break_mask):
                slot = self._value_route_slot(span)
                self.add_span(tuple(span), slot, feature_kind="value")
            del self.pending_positions[:route_n]

    def _routable_prefix_len(self, chunk: list[int], break_mask: list[bool] | None) -> int:
        if not break_mask:
            return len(chunk)
        last = -1
        for i, pos in enumerate(chunk):
            if pos < len(break_mask) and break_mask[pos]:
                last = i
        return last + 1

    def _split_spans(self, positions: list[int], break_mask: list[bool] | None) -> list[list[int]]:
        spans: list[list[int]] = []
        current: list[int] = []
        for pos in positions:
            current.append(pos)
            is_break = bool(break_mask and pos < len(break_mask) and break_mask[pos])
            if is_break or len(current) >= self.config.max_span_tokens:
                spans.append(current)
                current = []
        if current:
            spans.append(current)
        return spans

    def _value_route_slot(self, positions: list[int]) -> int:
        # Metadata-only deterministic stand-in for the tensor path's value-vector
        # routing. This keeps tests stable while preserving the contract that the
        # routing feature is value based rather than RoPE-key based.
        return (sum((p + 1) * 1315423911 for p in positions) ^ len(positions)) % self.config.routed_slots

    def _enforce_span_budget(self, spans: list[SpanRecord]) -> None:
        budget = self.config.span_budget_tokens
        used = sum(span.n_tokens for span in spans)
        while spans and used > budget:
            dropped = spans.pop(0)
            used -= dropped.n_tokens


@dataclass
class GemmaRoutedStateSlot:
    """All wkvm-owned routed-span state for one request slot."""

    slot_id: int
    config: GemmaRoutedSpanConfig
    full_layers: dict[int, RoutedSpanLayerState] = field(init=False)
    valid_mask_width: int | None = None

    def __post_init__(self) -> None:
        self.full_layers = {
            layer_id: RoutedSpanLayerState(layer_id, self.config)
            for layer_id in self.config.full_kv_layers
        }

    def reset(self) -> None:
        for layer in self.full_layers.values():
            layer.reset()
        self.valid_mask_width = None

    def ingest_positions(self, positions: list[int], break_mask: list[bool] | None = None) -> None:
        for layer in self.full_layers.values():
            layer.ingest_positions(positions, break_mask)
        self.valid_mask_width = max(
            (len(layer.materialized_positions()) for layer in self.full_layers.values()),
            default=0,
        )

    @property
    def resident_tokens(self) -> int:
        return sum(len(layer.materialized_positions()) for layer in self.full_layers.values())

    @property
    def ring_tokens(self) -> int:
        return sum(len(layer.ring_positions) for layer in self.full_layers.values())

    @property
    def pending_tokens(self) -> int:
        return sum(layer.pending_occupancy_tokens for layer in self.full_layers.values())

    @property
    def span_bank_tokens(self) -> int:
        return sum(layer.bank_occupancy_tokens for layer in self.full_layers.values())

    def valid_masks(self, padded_to: int | None = None) -> dict[int, list[bool]]:
        width = padded_to if padded_to is not None else self.valid_mask_width
        return {layer_id: layer.valid_mask(width) for layer_id, layer in self.full_layers.items()}


class GemmaRoutedStateBank:
    """Slot-indexed native state bank for Gemma routed-span metadata."""

    def __init__(self, config: GemmaRoutedSpanConfig, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.config = config
        self.num_slots = num_slots
        self.slots = [GemmaRoutedStateSlot(i, config) for i in range(num_slots + 1)]

    def zero_slots(self, slots: dict[str, int]) -> None:
        slot = self._slot_from_mapping(slots)
        self.slots[slot].reset()

    def slot_state(self, slots: dict[str, int]) -> GemmaRoutedStateSlot:
        return self.slots[self._slot_from_mapping(slots)]

    def ingest_positions(
        self,
        slots: dict[str, int],
        positions: list[int],
        break_mask: list[bool] | None = None,
    ) -> None:
        self.slot_state(slots).ingest_positions(positions, break_mask)

    def memory_accounting(self, resident_sessions: int | None = None) -> dict[str, int]:
        sessions = self.num_slots if resident_sessions is None else resident_sessions
        if sessions < 0:
            raise ValueError("resident_sessions must be >= 0")
        spec = self.config.state_spec()
        return {
            "resident_sessions": sessions,
            "bytes_per_slot": spec.bytes_per_request,
            "estimated_bytes": sessions * spec.bytes_per_request,
            "routed_materialized_tokens": self.config.routed_materialized_tokens,
            "full_kv_layers": len(self.config.full_kv_layers),
            "sliding_kv_layers": len(self.config.sliding_kv_layers),
        }

    @staticmethod
    def _slot_from_mapping(slots: dict[str, int]) -> int:
        if "gemma_routed_span" in slots:
            return slots["gemma_routed_span"]
        if "gemma_sliding_kv" in slots:
            return slots["gemma_sliding_kv"]
        raise KeyError("Gemma slots must include gemma_routed_span or gemma_sliding_kv")


def pad_valid_masks(slots: list[GemmaRoutedStateSlot]) -> dict[int, list[list[bool]]]:
    """Build layer -> batch valid masks for distinct-cache batching."""

    if not slots:
        return {}
    layer_ids = sorted(slots[0].full_layers)
    widths = {
        layer_id: max(len(slot.full_layers[layer_id].materialized_positions()) for slot in slots)
        for layer_id in layer_ids
    }
    return {
        layer_id: [slot.full_layers[layer_id].valid_mask(widths[layer_id]) for slot in slots]
        for layer_id in layer_ids
    }
