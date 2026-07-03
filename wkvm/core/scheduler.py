"""The no-phases continuous-batching scheduler.

One loop, one invariant: schedule ``num_computed_tokens -> num_tokens`` gaps
under a global token budget. There is no prefill phase and no decode phase;
a "decode" is a request whose gap is 1 and a "chunked prefill" is a request
whose gap exceeds the budget share it was given this step.

Deliberate omissions at M0 (see ROADMAP.md): overlap scheduling (M2 — the
optimistic-advance hooks are already in place via ``update_from_output``
taking sampled counts), swap-based preemption (M3 — preemption currently
recomputes), spec-decode lookahead accounting (deferred).

This class is pure bookkeeping: no tensors, no torch, unit-testable anywhere.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from wkvm.core.arena import StateArena
from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request, RequestStatus


@dataclass
class SchedulerOutput:
    """What the runner executes this step: token counts against slot ids."""

    # req_id -> number of new tokens whose state to compute this step.
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    # Requests entering the batch this step (runner uploads prompt tokens,
    # zeroes/loads their slots). Full payload once; deltas thereafter.
    admitted: list[Request] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(self.num_scheduled_tokens.values())

    @property
    def is_empty(self) -> bool:
        return not self.num_scheduled_tokens


class Scheduler:
    def __init__(self, config: SchedulerConfig, arena: StateArena) -> None:
        self.config = config
        self.arena = arena
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.requests: dict[str, Request] = {}

    # -- intake ------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        if request.req_id in self.requests:
            raise ValueError(f"duplicate req_id {request.req_id}")
        self.requests[request.req_id] = request
        self.waiting.append(request)

    def abort_request(self, req_id: str) -> None:
        req = self.requests.get(req_id)
        if req is None or req.status.is_finished:
            return  # abort is idempotent
        if req.status is RequestStatus.RUNNING:
            self._release(req)
        else:
            try:
                self.waiting.remove(req)
            except ValueError:
                pass
        req.status = RequestStatus.FINISHED_ABORTED

    # -- the loop ----------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.config.max_tokens_per_step

        # 1) RUNNING first: decodes and in-flight chunked prefills. Running
        #    requests already own slots, so this can never fail admission.
        for req in self.running:
            if budget <= 0:
                break
            n = min(
                req.num_scheduled_gap,
                budget,
                self.config.max_tokens_per_request_per_step,
            )
            if n <= 0:
                continue
            out.num_scheduled_tokens[req.req_id] = n
            budget -= n

        # 2) WAITING: admit while there is budget AND a free slot in every
        #    family. Exact admission — the whole point of the state arena.
        while (
            self.waiting
            and budget > 0
            and len(self.running) < self.config.max_running_requests
            and self.arena.can_admit()
        ):
            req = self.waiting.popleft()
            n = min(
                req.num_scheduled_gap,
                budget,
                self.config.max_tokens_per_request_per_step,
            )
            if n <= 0:  # defensive; a waiting request always has a gap
                continue
            req.slots = self.arena.allocate()
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            out.admitted.append(req)
            out.num_scheduled_tokens[req.req_id] = n
            budget -= n

        return out

    # -- results -----------------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        sampled: dict[str, list[int]],
        stop_token_ids: frozenset[int] = frozenset(),
    ) -> list[Request]:
        """Commit one executed step.

        ``sampled`` maps req_id -> newly sampled token ids (empty while a
        request is still mid-prefill). Returns requests finished this step.
        """
        finished: list[Request] = []
        for req_id, n_computed in scheduler_output.num_scheduled_tokens.items():
            req = self.requests.get(req_id)
            if req is None or req.status is not RequestStatus.RUNNING:
                continue  # aborted mid-step; slots already released
            req.num_computed_tokens += n_computed
            assert req.num_computed_tokens <= req.num_tokens, (
                f"{req_id}: computed {req.num_computed_tokens} > "
                f"target {req.num_tokens}"
            )
            for tok in sampled.get(req_id, ()):
                req.output_token_ids.append(tok)
                if tok in stop_token_ids:
                    req.status = RequestStatus.FINISHED_STOPPED
                    break
                if req.reached_length_limit():
                    req.status = RequestStatus.FINISHED_LENGTH
                    break
            if req.status.is_finished:
                self._release(req)
                finished.append(req)
        return finished

    # -- internals -----------------------------------------------------------

    def _release(self, req: Request) -> None:
        self.running.remove(req)
        if req.slots:
            self.arena.free(req.slots)
            req.slots = {}
