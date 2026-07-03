"""GenerationLoop: M1's minimal admit -> prefill -> batched-decode loop.

This is *not* the M0 scheduler (that wiring is M2's job); it is the smallest
loop that proves the exit criterion "correct completions, batch=N decode
from the arena": requests are admitted while every family has a free slot,
prefilled one at a time (chunked), then decoded as one batch per step across
all running slots. A finished request frees its slots, which admits the next
waiting request mid-flight — continuous batching in miniature.

Requests are ``wkvm.core.request.Request`` records: token ids only, state
referenced exclusively through ``request.slots``.
"""

from __future__ import annotations

from collections import deque

import torch

from wkvm.core.arena import StateArena
from wkvm.core.request import Request, RequestStatus
from wkvm.runner.runner import RWKV7Runner
from wkvm.runner.sampling import SamplingParams, make_generator, sample_token


class GenerationLoop:
    def __init__(self, runner: RWKV7Runner, arena: StateArena) -> None:
        self.runner = runner
        self.arena = arena

    def generate(
        self,
        requests: list[Request],
        params: SamplingParams = SamplingParams(),
    ) -> list[Request]:
        """Run every request to completion; returns them in input order.

        One ``SamplingParams`` for the whole call at M1 (per-request params
        arrive with the server frontend); RNG is still per-request so that
        sampled outputs are independent of batch composition.
        """
        waiting: deque[Request] = deque(requests)
        running: list[Request] = []
        generators: dict[str, torch.Generator | None] = {}

        while waiting or running:
            # Admit + prefill while capacity allows. Exact admission: a free
            # slot in every family or the request keeps waiting.
            while waiting and self.arena.can_admit():
                req = waiting.popleft()
                req.slots = self.arena.allocate()
                req.status = RequestStatus.RUNNING
                self.runner.bank.zero_slots(req.slots)
                generators[req.req_id] = make_generator(params, self.runner.device)
                logits = self.runner.prefill(req.prompt_token_ids, req.slots)
                req.num_computed_tokens = req.num_prompt_tokens
                self._commit_token(req, logits, params, generators)
                if not req.status.is_finished:
                    running.append(req)

            if not running:
                continue
            # One batched decode step across every running slot.
            logits = self.runner.decode_step(
                [req.slots for req in running],
                [req.output_token_ids[-1] for req in running],
            )
            still_running: list[Request] = []
            for row, req in enumerate(running):
                req.num_computed_tokens += 1
                self._commit_token(req, logits[row], params, generators)
                if not req.status.is_finished:
                    still_running.append(req)
            running = still_running

        return requests

    def _commit_token(
        self,
        req: Request,
        logits: torch.Tensor,
        params: SamplingParams,
        generators: dict[str, torch.Generator | None],
    ) -> None:
        """Sample one token, append it, apply stop conditions, free slots."""
        token = sample_token(logits, params, generators[req.req_id])
        req.output_token_ids.append(token)
        if token in params.stop_token_ids:
            req.status = RequestStatus.FINISHED_STOPPED
        elif req.reached_length_limit():
            req.status = RequestStatus.FINISHED_LENGTH
        if req.status.is_finished:
            self.arena.free(req.slots)
            req.slots = {}
            generators.pop(req.req_id, None)
