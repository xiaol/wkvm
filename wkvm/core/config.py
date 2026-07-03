"""Typed, frozen configuration objects.

Design rule (docs/ANGLE.md §3): no god-config. Each subsystem takes the one
frozen config it needs; nothing threads a 400-field object through every
constructor.

M0 note: shapes/dtypes are described in bytes and abstract dims only — core
never imports torch. The GPU runner (M1) materialises tensors from these specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StateFamilySpec:
    """One family of fixed-size per-request state.

    A model declares one family per distinct state kind it carries, e.g.:
      - "wkv":  RWKV-7 matrix state, layers x heads x d x d
      - "shift": token-shift / conv window state
      - "ring":  sink+window KV ring for guest/recurrent-mode attention layers
      - "bank":  segmented state bank (K states per slot, docs/RECURRENT_MODE.md)

    ``bytes_per_slot`` is the full per-request footprint of this family across
    all layers that use it. The arena allocates exactly ``num_slots`` of these
    at startup; admission is counting free slots — that exactness is the point.
    """

    name: str
    bytes_per_slot: int
    # Layer indices using this family (heterogeneous per-layer layouts are
    # first-class: shallow-band memory, KV-shared tails, etc.).
    layer_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.bytes_per_slot <= 0:
            raise ValueError(f"family {self.name!r}: bytes_per_slot must be > 0")


@dataclass(frozen=True)
class ModelStateSpec:
    """Everything the allocator needs to know about a model's state footprint."""

    families: tuple[StateFamilySpec, ...]

    def __post_init__(self) -> None:
        names = [f.name for f in self.families]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate family names: {names}")

    @property
    def bytes_per_request(self) -> int:
        return sum(f.bytes_per_slot for f in self.families)


@dataclass(frozen=True)
class SchedulerConfig:
    """Limits for the no-phases scheduling loop."""

    # Global per-step token budget (chunked prefill falls out of this: a long
    # prompt is simply scheduled ``budget`` tokens at a time).
    max_tokens_per_step: int = 8192
    # Hard cap on concurrently RUNNING requests (usually == arena slots; kept
    # separate so decode-batch bucketing for CUDA graphs can cap lower).
    max_running_requests: int = 1024
    # Cap a single request's tokens within one step so one long prefill cannot
    # starve running decodes (vLLM's long_prefill_token_threshold, simplified).
    max_tokens_per_request_per_step: int = 4096

    def __post_init__(self) -> None:
        if self.max_tokens_per_step < 1:
            raise ValueError("max_tokens_per_step must be >= 1")
        if self.max_running_requests < 1:
            raise ValueError("max_running_requests must be >= 1")
        if self.max_tokens_per_request_per_step < 1:
            raise ValueError("max_tokens_per_request_per_step must be >= 1")


def rwkv7_state_spec(
    *,
    n_layer: int,
    d_model: int,
    head_dim: int = 64,
    bytes_per_elem: int = 2,
) -> ModelStateSpec:
    """Convenience constructor for a pure RWKV-7 model.

    WKV state per layer: heads x head_dim x head_dim; shift states are one
    d_model vector per mixing site (2 per layer: time-mix + channel-mix).
    """
    if d_model % head_dim != 0:
        raise ValueError("d_model must be divisible by head_dim")
    n_head = d_model // head_dim
    wkv = StateFamilySpec(
        name="wkv",
        bytes_per_slot=n_layer * n_head * head_dim * head_dim * bytes_per_elem,
        layer_ids=tuple(range(n_layer)),
    )
    shift = StateFamilySpec(
        name="shift",
        bytes_per_slot=n_layer * 2 * d_model * bytes_per_elem,
        layer_ids=tuple(range(n_layer)),
    )
    return ModelStateSpec(families=(wkv, shift))
