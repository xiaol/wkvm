import unittest

from wkvm.core.arena import StateArena
from wkvm.core.config import ModelStateSpec, SchedulerConfig, StateFamilySpec
from wkvm.core.request import Request, RequestStatus
from wkvm.core.scheduler import Scheduler, SchedulerOutput


def make_scheduler(
    *,
    num_slots: int = 4,
    max_tokens_per_step: int = 64,
    max_per_request: int = 64,
) -> Scheduler:
    spec = ModelStateSpec(families=(StateFamilySpec(name="wkv", bytes_per_slot=8),))
    return Scheduler(
        SchedulerConfig(
            max_tokens_per_step=max_tokens_per_step,
            max_running_requests=num_slots,
            max_tokens_per_request_per_step=max_per_request,
        ),
        StateArena(spec, num_slots=num_slots),
    )


def run_step(sched: Scheduler, sampled_token: int = 7) -> tuple[SchedulerOutput, list]:
    """Execute one schedule step, sampling one token for every request whose
    gap closed (i.e. whose state is caught up to its last known token)."""
    out = sched.schedule()
    sampled = {}
    for req_id, n in out.num_scheduled_tokens.items():
        req = sched.requests[req_id]
        if req.num_computed_tokens + n == req.num_tokens:
            sampled[req_id] = [sampled_token]
    finished = sched.update_from_output(out, sampled)
    return out, finished


class TestNoPhasesInvariant(unittest.TestCase):
    def test_prefill_then_decode_same_loop(self) -> None:
        sched = make_scheduler(max_tokens_per_step=64)
        req = Request(prompt_token_ids=list(range(10)), max_new_tokens=3)
        sched.add_request(req)

        out1, _ = run_step(sched)
        # Whole prompt fits the budget: one step computes 10 tokens' state
        # and samples the first output token.
        self.assertEqual(out1.num_scheduled_tokens[req.req_id], 10)
        self.assertEqual(len(req.output_token_ids), 1)

        out2, _ = run_step(sched)
        # Steady-state decode is just gap == 1 in the same loop.
        self.assertEqual(out2.num_scheduled_tokens[req.req_id], 1)
        _, finished = run_step(sched)
        self.assertEqual(finished, [req])
        self.assertIs(req.status, RequestStatus.FINISHED_LENGTH)

    def test_chunked_prefill_falls_out_of_budget(self) -> None:
        sched = make_scheduler(max_tokens_per_step=16)
        req = Request(prompt_token_ids=list(range(40)), max_new_tokens=2)
        sched.add_request(req)

        chunks = []
        for _ in range(3):
            out, _ = run_step(sched)
            chunks.append(out.num_scheduled_tokens[req.req_id])
        # 40-token prompt under a 16-token budget: 16 + 16 + 8, and the
        # sampled token arrives only after the last chunk.
        self.assertEqual(chunks, [16, 16, 8])
        self.assertEqual(len(req.output_token_ids), 1)

    def test_running_requests_scheduled_before_waiting(self) -> None:
        sched = make_scheduler(max_tokens_per_step=8, num_slots=4)
        a = Request(prompt_token_ids=list(range(8)), max_new_tokens=4)
        sched.add_request(a)
        run_step(sched)  # a admitted, prefilled, decoding now
        b = Request(prompt_token_ids=list(range(100)), max_new_tokens=1)
        sched.add_request(b)
        out, _ = run_step(sched)
        # a's decode token comes off the budget before b's prefill chunk.
        self.assertEqual(out.num_scheduled_tokens[a.req_id], 1)
        self.assertEqual(out.num_scheduled_tokens[b.req_id], 7)


class TestExactAdmission(unittest.TestCase):
    def test_admission_bounded_by_slots(self) -> None:
        sched = make_scheduler(num_slots=2, max_tokens_per_step=1024)
        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=8) for _ in range(5)
        ]
        for r in reqs:
            sched.add_request(r)
        out, _ = run_step(sched)
        self.assertEqual(len(out.admitted), 2)
        self.assertEqual(len(sched.running), 2)
        self.assertEqual(len(sched.waiting), 3)
        self.assertFalse(sched.arena.can_admit())

    def test_finish_frees_slot_for_waiting(self) -> None:
        sched = make_scheduler(num_slots=1, max_tokens_per_step=1024)
        a = Request(prompt_token_ids=[1, 2], max_new_tokens=1)
        b = Request(prompt_token_ids=[3, 4], max_new_tokens=1)
        sched.add_request(a)
        sched.add_request(b)
        _, finished = run_step(sched)  # a: prefill+sample -> length-finished
        self.assertEqual(finished, [a])
        _, finished = run_step(sched)
        self.assertEqual(finished, [b])
        self.assertEqual(sched.arena.num_free_slots(), 1)

    def test_abort_is_idempotent_and_frees(self) -> None:
        sched = make_scheduler(num_slots=1)
        a = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=8)
        sched.add_request(a)
        sched.schedule()
        self.assertFalse(sched.arena.can_admit())
        sched.abort_request(a.req_id)
        sched.abort_request(a.req_id)
        self.assertTrue(sched.arena.can_admit())
        self.assertIs(a.status, RequestStatus.FINISHED_ABORTED)

    def test_abort_mid_step_ignored_by_update(self) -> None:
        sched = make_scheduler(num_slots=1)
        a = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=8)
        sched.add_request(a)
        out = sched.schedule()
        sched.abort_request(a.req_id)  # abort lands while GPU is "running"
        finished = sched.update_from_output(out, {a.req_id: [9]})
        self.assertEqual(finished, [])
        self.assertEqual(a.output_token_ids, [])  # nothing committed post-abort


class TestStopConditions(unittest.TestCase):
    def test_stop_token_finishes_request(self) -> None:
        sched = make_scheduler()
        req = Request(prompt_token_ids=[1, 2], max_new_tokens=100)
        sched.add_request(req)
        out = sched.schedule()
        finished = sched.update_from_output(
            out, {req.req_id: [42]}, stop_token_ids=frozenset({42})
        )
        self.assertEqual(finished, [req])
        self.assertIs(req.status, RequestStatus.FINISHED_STOPPED)

    def test_budget_conservation(self) -> None:
        sched = make_scheduler(max_tokens_per_step=32, num_slots=4)
        for i in range(4):
            sched.add_request(
                Request(prompt_token_ids=list(range(20)), max_new_tokens=2)
            )
        out = sched.schedule()
        self.assertLessEqual(out.total_tokens, 32)


if __name__ == "__main__":
    unittest.main()
