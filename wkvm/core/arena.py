"""StateArena: the allocator that makes this engine state-native.

The primary allocation object is a fixed-size per-request slot per state
family. Consequences (docs/ANGLE.md §5):

- Admission is exact: a request is admissible iff every family has a free
  slot. No fragmentation, no watermark heuristics, no admit-then-deadlock.
- Preemption is releasing slot ids (state contents are swapped or discarded
  by the owner of the tensors — never by this class).
- Fork is a refcount question, not a copy question, at this layer: slot
  contents are copied by the runner; the arena only hands out the new ids.

M4 adds the *guest* pool for hybrids: a paged family (``page_tokens`` set on
its spec) is allocated in pages instead of one slot. A request reserves the
pages for its whole lifetime (``spec.pages_for(num_tokens + max_new_tokens)``)
at admission, so admission stays a count — free slots AND free pages — and
nothing needs preempting or retracting mid-flight. One allocator serves both
kinds; there is no second block pool to keep in step with the arena.

This class is pure bookkeeping over integers. It owns no tensors and never
imports torch; the GPU runner materialises one dense ``[num_slots, ...]``
tensor per slot family and one ``[num_pages, ...]`` tensor per paged family,
indexed with the ids allocated here. That split keeps the whole
admission/scheduling layer unit-testable without a GPU.
"""

from __future__ import annotations

from collections import deque

from wkvm.core.config import ModelStateSpec

SlotMap = dict[str, "int | tuple[int, ...]"]


class StateArena:
    def __init__(self, spec: ModelStateSpec, num_slots: int, num_pages: int = 0) -> None:
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        self.spec = spec
        self.num_slots = num_slots
        # Per slot family: free slot ids. Slot 0 is reserved in every family
        # as the dummy write target for CUDA-graph padded batch rows
        # (docs/ANGLE.md §2, a convention both incumbents converged on), so
        # usable slots are 1..N.
        self._free: dict[str, deque[int]] = {
            f.name: deque(range(1, num_slots + 1)) for f in spec.slot_families
        }
        self._allocated: dict[str, set[int]] = {f.name: set() for f in spec.slot_families}
        # Paged families share one page pool (one page size per spec). Page 0
        # is likewise reserved as the read target of masked-out positions.
        self._paged = tuple(f.name for f in spec.paged_families)
        if self._paged and num_pages < 1:
            raise ValueError("num_pages must be >= 1 for a spec with paged families")
        self.num_pages = num_pages if self._paged else 0
        self._free_pages: deque[int] = deque(range(1, self.num_pages + 1))
        self._allocated_pages: set[int] = set()

    # -- queries ----------------------------------------------------------

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(self._free.keys()) + self._paged

    @property
    def page_tokens(self) -> int | None:
        return self.spec.page_tokens

    @property
    def max_tokens_per_request(self) -> int | None:
        """Largest reservation a single request could ever get (None when
        the model has no paged family)."""
        if not self._paged:
            return None
        return self.num_pages * self.spec.page_tokens

    def pages_for(self, num_tokens: int) -> int:
        return self.spec.pages_for(num_tokens)

    def num_free_slots(self) -> int:
        """Free capacity in requests (the min across slot families)."""
        return min(len(q) for q in self._free.values())

    def num_free_pages(self) -> int:
        return len(self._free_pages)

    def can_admit(self, n_requests: int = 1, pages: int = 0) -> bool:
        """``pages`` is the reservation of ONE request (per paged family)."""
        if self.num_free_slots() < n_requests:
            return False
        if pages and (not self._paged or self.num_free_pages() < pages * n_requests * len(self._paged)):
            return False
        return True

    # -- allocation -------------------------------------------------------

    def allocate(self, pages: int = 0) -> SlotMap:
        """Allocate one slot in every slot family and ``pages`` pages in
        every paged family. All-or-nothing."""
        if not self.can_admit(pages=pages):
            raise NoFreeSlots(
                f"no free capacity (slots {self.num_slots}, free per family: "
                f"{ {k: len(v) for k, v in self._free.items()} }, "
                f"pages free {self.num_free_pages()}/{self.num_pages}, wanted {pages})"
            )
        if pages and not self._paged:
            raise ValueError("pages requested but the spec has no paged family")
        slots: SlotMap = {}
        for name, q in self._free.items():
            slot = q.popleft()
            self._allocated[name].add(slot)
            slots[name] = slot
        for name in self._paged:
            ids = tuple(self._free_pages.popleft() for _ in range(pages))
            self._allocated_pages.update(ids)
            slots[name] = ids
        return slots

    def free(self, slots: SlotMap) -> None:
        """Return slots/pages to the free lists. Freed ids go to the *back*
        so recently-freed state contents survive longest for opportunistic
        reuse (the lazy-eviction idea from vLLM's block pool, at slot
        granularity)."""
        for name, slot in slots.items():
            if name in self._paged:
                for page in slot:  # type: ignore[union-attr]
                    if page not in self._allocated_pages:
                        raise ValueError(f"double free: family {name!r} page {page}")
                    self._allocated_pages.remove(page)
                    self._free_pages.append(page)
                continue
            allocated = self._allocated.get(name)
            if allocated is None:
                raise KeyError(f"unknown family {name!r}")
            if slot not in allocated:
                raise ValueError(f"double free: family {name!r} slot {slot}")
            allocated.remove(slot)  # type: ignore[arg-type]
            self._free[name].append(slot)  # type: ignore[arg-type]

    def fork(self, parent_slots: SlotMap) -> SlotMap:
        """Allocate a child slot set next to a live parent (same page count).

        The runner is responsible for the O(MB) state copy parent -> child;
        the arena only guarantees the child ids are valid and distinct. COW
        refcounting arrives with the StateStore (M3); at M0 fork is eager.
        """
        pages = 0
        for name, slot in parent_slots.items():
            if name in self._paged:
                if any(p not in self._allocated_pages for p in slot):  # type: ignore[union-attr]
                    raise ValueError(f"fork of unallocated parent pages: {name!r}")
                pages = len(slot)  # type: ignore[arg-type]
            elif slot not in self._allocated.get(name, ()):
                raise ValueError(f"fork of unallocated parent: {name!r} slot {slot}")
        return self.allocate(pages=pages)


class NoFreeSlots(RuntimeError):
    pass
