"""Torch-free tests for the M4 guest page pool: paged families in the spec,
page reservation in the arena, and page-aware admission in the scheduler.

Run: ``python -m unittest tests.test_pages -v``
"""

from __future__ import annotations

import unittest

from wkvm.core.arena import NoFreeSlots, StateArena
from wkvm.core.config import ModelStateSpec, SchedulerConfig, StateFamilySpec
from wkvm.core.request import Request, RequestStatus
from wkvm.core.scheduler import Scheduler


def _spec(page_tokens: int = 8) -> ModelStateSpec:
    return ModelStateSpec(
        families=(
            StateFamilySpec(name="gdn_state", bytes_per_slot=1024, layer_ids=(0, 1)),
            StateFamilySpec(name="gdn_conv", bytes_per_slot=64, layer_ids=(0, 1)),
            StateFamilySpec(name="guest_kv", bytes_per_slot=256, layer_ids=(2,), page_tokens=page_tokens),
        )
    )


class TestSpec(unittest.TestCase):
    def test_paged_family_accounting(self) -> None:
        spec = _spec(8)
        self.assertEqual([f.name for f in spec.slot_families], ["gdn_state", "gdn_conv"])
        self.assertEqual([f.name for f in spec.paged_families], ["guest_kv"])
        self.assertEqual(spec.page_tokens, 8)
        self.assertEqual(spec.bytes_per_request, 1024 + 64)
        self.assertEqual(spec.bytes_per_page, 256)
        self.assertEqual([spec.pages_for(n) for n in (0, 1, 8, 9, 16, 17)], [0, 1, 1, 2, 2, 3])

    def test_unpaged_spec_reserves_nothing(self) -> None:
        spec = ModelStateSpec(families=(StateFamilySpec(name="wkv", bytes_per_slot=8),))
        self.assertIsNone(spec.page_tokens)
        self.assertEqual(spec.pages_for(1000), 0)
        self.assertEqual(spec.bytes_per_page, 0)

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):  # two page sizes
            ModelStateSpec(families=(
                StateFamilySpec(name="a", bytes_per_slot=1),
                StateFamilySpec(name="b", bytes_per_slot=1, page_tokens=4),
                StateFamilySpec(name="c", bytes_per_slot=1, page_tokens=8),
            ))
        with self.assertRaises(ValueError):  # no slot family at all
            ModelStateSpec(families=(StateFamilySpec(name="kv", bytes_per_slot=1, page_tokens=4),))
        with self.assertRaises(ValueError):
            StateFamilySpec(name="kv", bytes_per_slot=1, page_tokens=0)


class TestArenaPages(unittest.TestCase):
    def test_requires_pages_for_paged_spec(self) -> None:
        with self.assertRaises(ValueError):
            StateArena(_spec(), num_slots=2)
        arena = StateArena(_spec(), num_slots=2, num_pages=5)
        self.assertEqual(arena.num_pages, 5)
        self.assertEqual(arena.page_tokens, 8)
        self.assertEqual(arena.max_tokens_per_request, 40)
        self.assertEqual(arena.family_names, ("gdn_state", "gdn_conv", "guest_kv"))

    def test_unpaged_arena_ignores_pages(self) -> None:
        spec = ModelStateSpec(families=(StateFamilySpec(name="wkv", bytes_per_slot=8),))
        arena = StateArena(spec, num_slots=2, num_pages=99)
        self.assertEqual(arena.num_pages, 0)
        self.assertIsNone(arena.max_tokens_per_request)
        self.assertTrue(arena.can_admit(pages=0))
        self.assertFalse(arena.can_admit(pages=1))
        self.assertEqual(arena.allocate(), {"wkv": 1})

    def test_allocate_reserves_pages_and_frees_them(self) -> None:
        arena = StateArena(_spec(), num_slots=3, num_pages=5)
        a = arena.allocate(pages=3)
        self.assertEqual(a["gdn_state"], 1)
        self.assertEqual(a["guest_kv"], (1, 2, 3))  # page 0 reserved
        self.assertEqual(arena.num_free_pages(), 2)
        self.assertTrue(arena.can_admit(pages=2))
        self.assertFalse(arena.can_admit(pages=3))
        with self.assertRaises(NoFreeSlots):
            arena.allocate(pages=3)
        b = arena.allocate(pages=2)
        self.assertEqual(b["guest_kv"], (4, 5))
        self.assertEqual(arena.num_free_pages(), 0)
        self.assertEqual(arena.num_free_slots(), 1)  # slots left, pages exhausted
        self.assertFalse(arena.can_admit(pages=1))
        self.assertTrue(arena.can_admit(pages=0))
        arena.free(a)
        self.assertEqual(arena.num_free_pages(), 3)
        self.assertEqual(arena.num_free_slots(), 2)
        c = arena.allocate(pages=3)
        self.assertEqual(c["guest_kv"], (1, 2, 3))  # recycled in order
        with self.assertRaises(ValueError):
            arena.free(a)  # double free of pages

    def test_pages_without_paged_family_rejected(self) -> None:
        spec = ModelStateSpec(families=(StateFamilySpec(name="wkv", bytes_per_slot=8),))
        arena = StateArena(spec, num_slots=1)
        with self.assertRaises(NoFreeSlots):
            arena.allocate(pages=1)

    def test_fork_reserves_same_page_count(self) -> None:
        arena = StateArena(_spec(), num_slots=2, num_pages=6)
        parent = arena.allocate(pages=3)
        child = arena.fork(parent)
        self.assertEqual(len(child["guest_kv"]), 3)
        self.assertNotEqual(child["guest_kv"], parent["guest_kv"])
        self.assertEqual(arena.num_free_pages(), 0)
        arena.free(parent)
        with self.assertRaises(ValueError):
            arena.fork(parent)


class TestSchedulerPages(unittest.TestCase):
    def _sched(self, num_pages: int, num_slots: int = 4) -> Scheduler:
        arena = StateArena(_spec(8), num_slots=num_slots, num_pages=num_pages)
        cfg = SchedulerConfig(max_tokens_per_step=64, max_running_requests=num_slots,
                              max_tokens_per_request_per_step=16)
        return Scheduler(cfg, arena)

    def test_admission_reserves_lifetime_pages(self) -> None:
        sched = self._sched(num_pages=6)
        # 10 prompt + 6 new = 16 tokens -> 2 pages; three such requests need 6.
        reqs = [Request(prompt_token_ids=[1] * 10, max_new_tokens=6) for _ in range(3)]
        for r in reqs:
            sched.add_request(r)
        out = sched.schedule()
        self.assertEqual(len(out.admitted), 3)
        self.assertTrue(all(len(r.slots["guest_kv"]) == 2 for r in reqs))
        self.assertEqual(sched.arena.num_free_pages(), 0)

    def test_head_of_line_waits_for_pages(self) -> None:
        sched = self._sched(num_pages=4)
        big = Request(prompt_token_ids=[1] * 20, max_new_tokens=4)  # 24 tokens -> 3 pages
        small = Request(prompt_token_ids=[1] * 4, max_new_tokens=4)  # 8 tokens -> 1 page
        later = Request(prompt_token_ids=[1] * 4, max_new_tokens=4)  # 1 page, blocked
        for r in (big, small, later):
            sched.add_request(r)
        out = sched.schedule()
        self.assertEqual([r.req_id for r in out.admitted], [big.req_id, small.req_id])
        self.assertEqual(sched.arena.num_free_pages(), 0)
        self.assertIs(later.status, RequestStatus.WAITING)
        # Finish `small` (its single sampled token hits the stop set).
        sched.update_from_output(out, {small.req_id: [7]}, stop_token_ids=frozenset({7}))
        self.assertIs(small.status, RequestStatus.FINISHED_STOPPED)
        self.assertEqual(sched.arena.num_free_pages(), 1)
        out2 = sched.schedule()
        self.assertEqual([r.req_id for r in out2.admitted], [later.req_id])
        self.assertEqual(sched.arena.num_free_pages(), 0)

    def test_abort_returns_pages(self) -> None:
        sched = self._sched(num_pages=2)
        req = Request(prompt_token_ids=[1] * 12, max_new_tokens=4)  # 2 pages
        sched.add_request(req)
        sched.schedule()
        self.assertEqual(sched.arena.num_free_pages(), 0)
        sched.abort_request(req.req_id)
        self.assertEqual(sched.arena.num_free_pages(), 2)
        self.assertEqual(sched.arena.num_free_slots(), 4)

    def test_resumed_request_keeps_given_pages(self) -> None:
        sched = self._sched(num_pages=3)
        slots = sched.arena.allocate(pages=sched.arena.pages_for(20))
        req = Request(prompt_token_ids=[1] * 12, max_new_tokens=8)
        req.num_computed_tokens = 11
        sched.add_resumed_request(req, slots)
        out = sched.schedule()
        self.assertEqual(out.num_scheduled_tokens[req.req_id], 1)
        self.assertEqual(req.slots["guest_kv"], (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
