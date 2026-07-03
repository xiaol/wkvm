"""StateArena: the allocator that makes this engine state-native.

The primary allocation object is a fixed-size per-request slot per state
family. Consequences (docs/ANGLE.md §5):

- Admission is exact: a request is admissible iff every family has a free
  slot. No fragmentation, no watermark heuristics, no admit-then-deadlock.
- Preemption is releasing slot ids (state contents are swapped or discarded
  by the owner of the tensors — never by this class).
- Fork is a refcount question, not a copy question, at this layer: slot
  contents are copied by the runner; the arena only hands out the new ids.

This class is pure bookkeeping over integers. It owns no tensors and never
imports torch; the GPU runner materialises one dense ``[num_slots, ...]``
tensor per family and indexes it with the slot ids allocated here. That split
keeps the whole admission/scheduling layer unit-testable without a GPU.
"""

from __future__ import annotations

from collections import deque

from wkvm.core.config import ModelStateSpec


class StateArena:
    def __init__(self, spec: ModelStateSpec, num_slots: int) -> None:
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.spec = spec
        self.num_slots = num_slots
        # Per family: free slot ids. Slot 0 is reserved in every family as the
        # dummy write target for CUDA-graph padded batch rows (docs/ANGLE.md §2,
        # a convention both incumbents converged on), so usable slots are 1..N.
        self._free: dict[str, deque[int]] = {
            f.name: deque(range(1, num_slots + 1)) for f in spec.families
        }
        self._allocated: dict[str, set[int]] = {f.name: set() for f in spec.families}

    # -- queries ----------------------------------------------------------

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(self._free.keys())

    def num_free_slots(self) -> int:
        """Free capacity in requests (the min across families)."""
        return min(len(q) for q in self._free.values())

    def can_admit(self, n_requests: int = 1) -> bool:
        return self.num_free_slots() >= n_requests

    # -- allocation -------------------------------------------------------

    def allocate(self) -> dict[str, int]:
        """Allocate one slot in every family. All-or-nothing."""
        if not self.can_admit():
            raise NoFreeSlots(
                f"no free slots (capacity {self.num_slots}, "
                f"free per family: { {k: len(v) for k, v in self._free.items()} })"
            )
        slots: dict[str, int] = {}
        for name, q in self._free.items():
            slot = q.popleft()
            self._allocated[name].add(slot)
            slots[name] = slot
        return slots

    def free(self, slots: dict[str, int]) -> None:
        """Return slots to the free lists. Freed slots go to the *back* so
        recently-freed state contents survive longest for opportunistic reuse
        (the lazy-eviction idea from vLLM's block pool, at slot granularity)."""
        for name, slot in slots.items():
            allocated = self._allocated.get(name)
            if allocated is None:
                raise KeyError(f"unknown family {name!r}")
            if slot not in allocated:
                raise ValueError(f"double free: family {name!r} slot {slot}")
            allocated.remove(slot)
            self._free[name].append(slot)

    def fork(self, parent_slots: dict[str, int]) -> dict[str, int]:
        """Allocate a child slot set next to a live parent.

        The runner is responsible for the O(MB) state copy parent -> child;
        the arena only guarantees the child ids are valid and distinct. COW
        refcounting arrives with the StateStore (M3); at M0 fork is eager.
        """
        for name, slot in parent_slots.items():
            if slot not in self._allocated.get(name, ()):
                raise ValueError(f"fork of unallocated parent: {name!r} slot {slot}")
        return self.allocate()


class NoFreeSlots(RuntimeError):
    pass
