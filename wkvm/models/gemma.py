"""Gemma routed-span state metadata.

The native Gemma milestone is deliberately narrower than a general transformer
engine: Gemma-4-E4B-it, one GPU, bf16, routed-span recurrent mode. This module
keeps the model/cache shape description dependency-light so the scheduler and
arena can reason about admission before torch or transformers are imported by
the runner.
"""

from __future__ import annotations

from dataclasses import dataclass

from wkvm.core.config import ModelStateSpec, StateFamilySpec


@dataclass(frozen=True)
class GemmaRoutedSpanConfig:
    """Static state layout for the first native Gemma routed-span runner."""

    num_hidden_layers: int = 42
    num_kv_shared_layers: int = 18
    layer_types: tuple[str, ...] = ()
    num_kv_heads: int = 2
    head_dim: int = 512
    bytes_per_elem: int = 2
    sink_tokens: int = 16
    ring_tokens: int = 1024
    routed_slots: int = 64
    reps_per_slot: int = 8
    span_budget_tokens: int = 144
    pending_tokens: int = 512
    sliding_window: int = 1024
    max_span_tokens: int = 48

    def __post_init__(self) -> None:
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must match num_hidden_layers")
        for name in (
            "num_hidden_layers",
            "num_kv_heads",
            "head_dim",
            "bytes_per_elem",
            "sink_tokens",
            "ring_tokens",
            "routed_slots",
            "reps_per_slot",
            "span_budget_tokens",
            "pending_tokens",
            "sliding_window",
            "max_span_tokens",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.num_kv_shared_layers < 0:
            raise ValueError("num_kv_shared_layers must be >= 0")
        if self.num_kv_shared_layers >= self.num_hidden_layers:
            raise ValueError("num_kv_shared_layers must be < num_hidden_layers")

    @property
    def n_owned_layers(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    @property
    def kv_shape_per_token(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def full_kv_layers(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, layer_type in enumerate(self._effective_layer_types[: self.n_owned_layers])
            if layer_type == "full_attention"
        )

    @property
    def sliding_kv_layers(self) -> tuple[int, ...]:
        return tuple(
            i
            for i, layer_type in enumerate(self._effective_layer_types[: self.n_owned_layers])
            if layer_type == "sliding_attention"
        )

    @property
    def _effective_layer_types(self) -> tuple[str, ...]:
        if self.layer_types:
            return self.layer_types
        # Gemma-4-E4B-it PoC-observed owner pattern: 42 layers, last 18 share
        # KV; among the 24 owners, full layers are 5/11/17/23 and the rest are
        # sliding. Shared tail layers are outside n_owned.
        return tuple(
            "full_attention" if i in {5, 11, 17, 23} else "sliding_attention"
            for i in range(self.num_hidden_layers)
        )

    @property
    def routed_materialized_tokens(self) -> int:
        """Maximum readout tokens per routed full-attention layer."""

        return self.routed_authoritative_tokens + self.routed_slots

    @property
    def routed_retained_tokens_per_slot(self) -> int:
        return max(
            self.span_budget_tokens,
            self.max_span_tokens,
        )

    @property
    def routed_authoritative_tokens(self) -> int:
        """Maximum real KV tokens retained behind one routed layer."""

        return (
            self.sink_tokens
            + self.ring_tokens
            + self.pending_tokens
            - 1
            + self.routed_slots * self.routed_retained_tokens_per_slot
        )

    @property
    def routed_readout_layer_bytes(self) -> int:
        return 2 * self.routed_materialized_tokens * self.kv_shape_per_token * self.bytes_per_elem

    @property
    def routed_authoritative_layer_bytes(self) -> int:
        return 2 * self.routed_authoritative_tokens * self.kv_shape_per_token * self.bytes_per_elem

    @property
    def routed_slot_summary_layer_bytes(self) -> int:
        return 2 * self.routed_slots * self.kv_shape_per_token * 4

    @property
    def routed_leader_state_bytes(self) -> int:
        return (self.routed_slots + 1) * self.kv_shape_per_token * 4

    @property
    def routed_layer_bytes(self) -> int:
        return (
            self.routed_readout_layer_bytes
            + self.routed_authoritative_layer_bytes
            + self.routed_slot_summary_layer_bytes
        )

    @property
    def sliding_layer_bytes(self) -> int:
        # key + value; sliding_window is the static native buffer target.
        return 2 * self.sliding_window * self.kv_shape_per_token * self.bytes_per_elem

    @property
    def metadata_bytes_per_slot(self) -> int:
        # Conservative slot-owned metadata: valid masks, per-span positions,
        # counters, route assignments, and graph padding scratch. Kept separate
        # from KV tensors so observability can report it explicitly.
        routed = len(self.full_kv_layers)
        return routed * (
            self.routed_materialized_tokens
            + self.routed_slots * self.max_span_tokens
            + self.routed_slots * 8
        )

    def state_spec(self) -> ModelStateSpec:
        families: list[StateFamilySpec] = []
        if self.sliding_kv_layers:
            families.append(
                StateFamilySpec(
                    name="gemma_sliding_kv",
                    bytes_per_slot=len(self.sliding_kv_layers) * self.sliding_layer_bytes,
                    layer_ids=self.sliding_kv_layers,
                )
            )
        if self.full_kv_layers:
            families.append(
                StateFamilySpec(
                    name="gemma_routed_span",
                    bytes_per_slot=(
                        len(self.full_kv_layers) * self.routed_layer_bytes
                        + self.routed_leader_state_bytes
                    ),
                    layer_ids=self.full_kv_layers,
                )
            )
            families.append(
                StateFamilySpec(
                    name="gemma_routed_meta",
                    bytes_per_slot=max(1, self.metadata_bytes_per_slot),
                    layer_ids=self.full_kv_layers,
                )
            )
        if not families:
            raise ValueError("Gemma routed-span config produced no state families")
        return ModelStateSpec(families=tuple(families))


def gemma4_e4b_routed_span_config(**overrides) -> GemmaRoutedSpanConfig:
    """Default Gemma-4-E4B-it routed-span layout from the PoC contract."""

    return GemmaRoutedSpanConfig(**overrides)
