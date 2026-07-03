import unittest

from wkvm.core.arena import NoFreeSlots, StateArena
from wkvm.core.config import ModelStateSpec, StateFamilySpec, rwkv7_state_spec


def two_family_spec() -> ModelStateSpec:
    return ModelStateSpec(
        families=(
            StateFamilySpec(name="wkv", bytes_per_slot=1024),
            StateFamilySpec(name="shift", bytes_per_slot=64),
        )
    )


class TestStateArena(unittest.TestCase):
    def test_exact_admission(self) -> None:
        arena = StateArena(two_family_spec(), num_slots=3)
        self.assertEqual(arena.num_free_slots(), 3)
        allocs = [arena.allocate() for _ in range(3)]
        self.assertFalse(arena.can_admit())
        with self.assertRaises(NoFreeSlots):
            arena.allocate()
        arena.free(allocs[1])
        self.assertTrue(arena.can_admit())
        self.assertEqual(arena.num_free_slots(), 1)

    def test_slot_zero_reserved_for_padding(self) -> None:
        arena = StateArena(two_family_spec(), num_slots=4)
        seen = set()
        for _ in range(4):
            for slot in arena.allocate().values():
                seen.add(slot)
                self.assertNotEqual(slot, 0)
        self.assertEqual(seen, {1, 2, 3, 4})

    def test_all_families_allocated_together(self) -> None:
        arena = StateArena(two_family_spec(), num_slots=2)
        slots = arena.allocate()
        self.assertEqual(set(slots), {"wkv", "shift"})

    def test_double_free_rejected(self) -> None:
        arena = StateArena(two_family_spec(), num_slots=2)
        slots = arena.allocate()
        arena.free(slots)
        with self.assertRaises(ValueError):
            arena.free(slots)

    def test_freed_slots_reused_lru(self) -> None:
        arena = StateArena(two_family_spec(), num_slots=2)
        a = arena.allocate()
        b = arena.allocate()
        arena.free(a)
        arena.free(b)
        # Freed slots go to the back: the next allocation must not be the
        # most recently freed one (its state contents live longest).
        c = arena.allocate()
        self.assertEqual(c, a)

    def test_fork_requires_live_parent(self) -> None:
        arena = StateArena(two_family_spec(), num_slots=3)
        parent = arena.allocate()
        child = arena.fork(parent)
        self.assertNotEqual(parent, child)
        arena.free(parent)
        with self.assertRaises(ValueError):
            arena.fork(parent)

    def test_rwkv7_spec_footprint(self) -> None:
        # RWKV-7 7B-class: L32 D4096 head 64 fp16 -> ~16.8MB wkv state.
        spec = rwkv7_state_spec(n_layer=32, d_model=4096)
        wkv = next(f for f in spec.families if f.name == "wkv")
        self.assertEqual(wkv.bytes_per_slot, 32 * 64 * 64 * 64 * 2)
        self.assertAlmostEqual(wkv.bytes_per_slot / 2**20, 16.0, places=1)
        self.assertGreater(spec.bytes_per_request, wkv.bytes_per_slot)


if __name__ == "__main__":
    unittest.main()
