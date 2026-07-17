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
    completion_prefill_lane_size: int = 0,
) -> Scheduler:
    spec = ModelStateSpec(families=(StateFamilySpec(name="wkv", bytes_per_slot=8),))
    return Scheduler(
        SchedulerConfig(
            max_tokens_per_step=max_tokens_per_step,
            max_running_requests=num_slots,
            max_tokens_per_request_per_step=max_per_request,
            completion_prefill_lane_size=completion_prefill_lane_size,
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

    def test_completion_prefill_lane_finishes_before_next_lane(self) -> None:
        sched = make_scheduler(
            num_slots=4,
            max_tokens_per_step=12,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        requests = [
            Request(prompt_token_ids=list(range(8)), max_new_tokens=2, req_id=req_id)
            for req_id in ("a", "b", "c", "d")
        ]
        for request in requests:
            sched.add_request(request)

        first, _ = run_step(sched)
        second, _ = run_step(sched)
        third, _ = run_step(sched)

        self.assertEqual(first.num_scheduled_tokens, {"a": 4, "b": 4})
        self.assertEqual(second.num_scheduled_tokens, {"a": 4, "b": 4})
        self.assertEqual(
            third.num_scheduled_tokens,
            {"a": 1, "b": 1, "c": 4, "d": 4},
        )
        self.assertEqual(sched.completion_prefill_lane_starts, 2)
        self.assertEqual(sched.completion_prefill_lane_completions, 1)

    def test_completion_prefill_lane_does_not_refill_ragged_lane(self) -> None:
        sched = make_scheduler(
            num_slots=3,
            max_tokens_per_step=11,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        a = Request(prompt_token_ids=list(range(4)), max_new_tokens=3, req_id="a")
        b = Request(prompt_token_ids=list(range(8)), max_new_tokens=3, req_id="b")
        c = Request(prompt_token_ids=list(range(4)), max_new_tokens=3, req_id="c")
        for request in (a, b, c):
            sched.add_request(request)

        first, _ = run_step(sched)
        second, _ = run_step(sched)

        self.assertEqual(first.num_scheduled_tokens, {"a": 4, "b": 4})
        self.assertEqual(second.num_scheduled_tokens, {"a": 1, "b": 4})
        self.assertEqual([request.req_id for request in sched.waiting], ["c"])

    def test_completion_prefill_lane_prioritizes_decode_ready_rows(self) -> None:
        sched = make_scheduler(
            num_slots=3,
            max_tokens_per_step=11,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        decode = Request(prompt_token_ids=[1], max_new_tokens=3, req_id="decode")
        first_prefill = Request(
            prompt_token_ids=list(range(8)), max_new_tokens=2, req_id="prefill-a"
        )
        second_prefill = Request(
            prompt_token_ids=list(range(8)), max_new_tokens=2, req_id="prefill-b"
        )
        for request in (decode, first_prefill, second_prefill):
            sched.add_request(request)

        run_step(sched)
        second, _ = run_step(sched)

        self.assertEqual(
            list(second.num_scheduled_tokens),
            ["decode", "prefill-a"],
        )
        self.assertEqual(second.num_scheduled_tokens["decode"], 1)
        self.assertEqual(second.num_scheduled_tokens["prefill-a"], 4)
        self.assertEqual([request.req_id for request in sched.waiting], ["prefill-b"])

    def test_completion_prefill_lane_preserves_fifo_under_arrivals(self) -> None:
        sched = make_scheduler(
            num_slots=4,
            max_tokens_per_step=12,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        requests = {
            req_id: Request(
                prompt_token_ids=list(range(8)),
                max_new_tokens=1,
                req_id=req_id,
            )
            for req_id in "abcdefgh"
        }
        for req_id in "abcd":
            sched.add_request(requests[req_id])

        admitted_order = []
        for step in range(8):
            if step == 1:
                sched.add_request(requests["e"])
                sched.add_request(requests["f"])
            if step == 2:
                sched.add_request(requests["g"])
                sched.add_request(requests["h"])
            output, _ = run_step(sched)
            admitted_order.extend(request.req_id for request in output.admitted)

        self.assertEqual(admitted_order, list("abcdefgh"))
        self.assertEqual(sched.completion_prefill_lane_starts, 4)
        self.assertEqual(sched.completion_prefill_lane_completions, 4)
        self.assertEqual(sched.completion_prefill_lane_cancellations, 0)
        self.assertFalse(sched.running)
        self.assertFalse(sched.waiting)

    def test_completion_prefill_lane_validates_full_lane_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires max_tokens_per_step"):
            make_scheduler(
                num_slots=4,
                max_tokens_per_step=11,
                max_per_request=4,
                completion_prefill_lane_size=2,
            )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            make_scheduler(
                num_slots=1,
                max_tokens_per_step=10,
                max_per_request=4,
                completion_prefill_lane_size=2,
            )

    def test_completion_prefill_lane_retires_on_terminal_prefill(self) -> None:
        sched = make_scheduler(
            num_slots=2,
            max_tokens_per_step=10,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        requests = [
            Request(prompt_token_ids=[1, 2], max_new_tokens=1, req_id=req_id)
            for req_id in ("a", "b")
        ]
        for request in requests:
            sched.add_request(request)

        out = sched.schedule()
        sched.update_from_output(out, {req.req_id: [9] for req in requests})

        self.assertEqual(sched.completion_prefill_lane_starts, 1)
        self.assertEqual(sched.completion_prefill_lane_completions, 1)
        self.assertEqual(sched.completion_prefill_lane_cancellations, 0)
        self.assertEqual(sched._completion_prefill_lane, [])

    def test_completion_prefill_lane_retires_as_cancelled_on_abort(self) -> None:
        sched = make_scheduler(
            num_slots=2,
            max_tokens_per_step=12,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        requests = [
            Request(prompt_token_ids=list(range(8)), max_new_tokens=2, req_id=req_id)
            for req_id in ("a", "b")
        ]
        for request in requests:
            sched.add_request(request)

        sched.schedule()
        sched.abort_request("a")
        sched.abort_request("b")

        self.assertEqual(sched.completion_prefill_lane_completions, 0)
        self.assertEqual(sched.completion_prefill_lane_cancellations, 1)
        self.assertEqual(sched._completion_prefill_lane, [])

    def test_completion_prefill_lane_retires_as_cancelled_on_failure(self) -> None:
        sched = make_scheduler(
            num_slots=2,
            max_tokens_per_step=12,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        requests = [
            Request(prompt_token_ids=list(range(8)), max_new_tokens=2, req_id=req_id)
            for req_id in ("a", "b")
        ]
        for request in requests:
            sched.add_request(request)

        sched.schedule()
        self.assertIsNotNone(sched.fail_request("a"))
        self.assertIsNotNone(sched.fail_request("b"))

        self.assertEqual(sched.completion_prefill_lane_completions, 0)
        self.assertEqual(sched.completion_prefill_lane_cancellations, 1)
        self.assertEqual(sched._completion_prefill_lane, [])

    def test_completion_prefill_lane_abort_after_prefill_counts_completion(self) -> None:
        sched = make_scheduler(
            num_slots=2,
            max_tokens_per_step=12,
            max_per_request=4,
            completion_prefill_lane_size=2,
        )
        requests = [
            Request(prompt_token_ids=list(range(8)), max_new_tokens=2, req_id=req_id)
            for req_id in ("a", "b")
        ]
        for request in requests:
            sched.add_request(request)

        first = sched.schedule()
        sched.update_from_output(first, {})
        second = sched.schedule()
        sched.update_from_output(second, {})
        sched.abort_request("a")
        sched.abort_request("b")

        self.assertEqual(sched.completion_prefill_lane_completions, 1)
        self.assertEqual(sched.completion_prefill_lane_cancellations, 0)
        self.assertEqual(sched._completion_prefill_lane, [])


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

    def test_parked_finish_retains_and_reuses_same_slot(self) -> None:
        sched = make_scheduler(num_slots=1, max_tokens_per_step=1024)
        session = Request(prompt_token_ids=[1, 2], max_new_tokens=1, req_id="session")
        waiting = Request(prompt_token_ids=[3, 4], max_new_tokens=1, req_id="waiting")
        sched.add_request(session, park_on_finish=True)
        sched.add_request(waiting)

        completed = sched.update_from_output(
            sched.schedule(),
            {session.req_id: [9]},
        )
        retained_slots = dict(session.slots)
        self.assertEqual(completed, [session])
        self.assertIs(session.status, RequestStatus.PARKED)
        self.assertIs(
            session.parked_finish_status,
            RequestStatus.FINISHED_LENGTH,
        )
        self.assertEqual(sched.parked, {session.req_id: session})
        self.assertEqual(sched.arena.num_free_slots(), 0)

        session.prompt_token_ids.extend(session.output_token_ids)
        session.prompt_token_ids.append(8)
        session.output_token_ids.clear()
        session.max_new_tokens = 1
        resumed = sched.resume_parked_request(session.req_id)
        self.assertIs(resumed, session)
        self.assertEqual(session.slots, retained_slots)
        self.assertIs(session.status, RequestStatus.RUNNING)
        self.assertEqual(sched.schedule().num_scheduled_tokens, {session.req_id: 2})

    def test_close_or_abort_parked_request_releases_slot(self) -> None:
        for close in (True, False):
            with self.subTest(close=close):
                sched = make_scheduler(num_slots=1, max_tokens_per_step=1024)
                req = Request(prompt_token_ids=[1, 2], max_new_tokens=1, req_id="session")
                sched.add_request(req, park_on_finish=True)
                sched.update_from_output(sched.schedule(), {req.req_id: [9]})
                self.assertEqual(sched.arena.num_free_slots(), 0)

                if close:
                    sched.close_parked_request(req.req_id)
                    self.assertIs(req.status, RequestStatus.FINISHED_CLOSED)
                else:
                    sched.abort_request(req.req_id)
                    self.assertIs(req.status, RequestStatus.FINISHED_ABORTED)
                self.assertEqual(sched.arena.num_free_slots(), 1)
                self.assertFalse(sched.parked)

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

    def test_fail_request_marks_error_and_frees_slot(self) -> None:
        sched = make_scheduler(num_slots=1)
        a = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=8, req_id="a")
        b = Request(prompt_token_ids=[4, 5, 6], max_new_tokens=8, req_id="b")
        sched.add_request(a)
        sched.add_request(b)

        sched.schedule()
        self.assertFalse(sched.arena.can_admit())

        failed = sched.fail_request("a")
        self.assertIs(failed, a)
        self.assertIs(a.status, RequestStatus.FINISHED_ERROR)
        self.assertTrue(sched.arena.can_admit())
        self.assertEqual(list(sched.waiting), [b])

        failed_waiting = sched.fail_request("b")
        self.assertIs(failed_waiting, b)
        self.assertIs(b.status, RequestStatus.FINISHED_ERROR)
        self.assertEqual(list(sched.waiting), [])


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
