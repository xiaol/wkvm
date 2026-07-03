"""Engine: the M0 scheduler driving the M1 runner (M2-minimal).

One ``step()`` is the whole contract:

    scheduler.schedule() -> execute the SchedulerOutput -> sample for
    gap-closed requests -> scheduler.update_from_output()

Invariants this class maintains:

- **All admission and accounting flow through the scheduler.** The engine
  never touches ``arena.allocate``/``free`` or ``num_computed_tokens``
  directly; it only executes what ``SchedulerOutput`` says and reports
  sampled tokens back. Chunked prefills and decodes scheduled in the same
  step are split internally into a prefill part (per-request chunk forward)
  and one batched decode part — an execution detail invisible to the
  scheduler.
- **Requests stream continuously.** ``add_request`` is legal at any time;
  requests finished in a step free their slots within that same
  ``update_from_output``, so the next ``step()``'s ``schedule()`` can admit
  from the waiting queue.
- **Determinism under batching.** Which tokens a request computes per step
  depends only on the ``SchedulerConfig`` and the request itself (never on
  batch composition) as long as ``max_tokens_per_step`` exceeds
  ``max_running_requests + max_tokens_per_request_per_step``: running
  requests are scheduled first and consume at most one token each in decode,
  so every prefill chunk is exactly ``min(gap, cap)``. With per-request RNG
  (see runner/sampling.py) outputs are then independent of what else is in
  flight — the property the continuous-batching test asserts.

Per-request ``SamplingParams`` carry temperature/seed; stop tokens are
engine-global (a single-model engine has one EOS set), matching the
``update_from_output`` signature.
"""

from __future__ import annotations

import torch

from wkvm.core.arena import StateArena
from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request
from wkvm.core.scheduler import Scheduler, SchedulerOutput
from wkvm.runner.runner import RWKV7Runner
from wkvm.runner.sampling import SamplingParams, make_generator, sample_token
from wkvm.runner.state import RWKV7StateBank


class Engine:
    """Owns Scheduler + StateArena + RWKV7StateBank + RWKV7Runner."""

    def __init__(
        self,
        model,
        layout,
        num_slots: int,
        scheduler_config: SchedulerConfig | None = None,
        device: torch.device | str = "cuda",
        stop_token_ids: frozenset[int] = frozenset(),
        prefill_chunk: int = 512,
    ) -> None:
        self.bank = RWKV7StateBank(layout, num_slots=num_slots, device=device)
        self.arena = StateArena(layout.state_spec(), num_slots=num_slots)
        self.scheduler = Scheduler(
            scheduler_config or SchedulerConfig(max_running_requests=num_slots),
            self.arena,
        )
        self.runner = RWKV7Runner(model, self.bank, prefill_chunk=prefill_chunk)
        self.stop_token_ids = stop_token_ids
        self._params: dict[str, SamplingParams] = {}
        self._generators: dict[str, torch.Generator | None] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_slots: int,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "Engine":
        from wkvm.models.rwkv7 import load_rwkv7

        model, layout = load_rwkv7(model_path, device=device, dtype=dtype)
        return cls(model, layout, num_slots=num_slots, device=device, **kwargs)

    # -- intake ---------------------------------------------------------------

    def add_request(
        self, request: Request, params: SamplingParams = SamplingParams()
    ) -> None:
        """Queue a request. Legal at any time, including between steps."""
        if params.stop_token_ids and params.stop_token_ids != self.stop_token_ids:
            raise ValueError(
                "per-request stop_token_ids must be empty or equal to the "
                "engine-global set (single stop set until the server frontend)"
            )
        self.scheduler.add_request(request)
        self._params[request.req_id] = params

    def abort_request(self, req_id: str) -> None:
        self.scheduler.abort_request(req_id)
        self._params.pop(req_id, None)
        self._generators.pop(req_id, None)

    @property
    def has_unfinished(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    # -- the step ---------------------------------------------------------------

    def step(self) -> list[Request]:
        """One engine step. Returns requests that finished this step (their
        slots are already back in the arena's free lists)."""
        out = self.scheduler.schedule()
        if out.is_empty:
            return []
        for req in out.admitted:
            self.bank.zero_slots(req.slots)
            self._generators[req.req_id] = make_generator(
                self._params[req.req_id], self.runner.device
            )
        sampled = self._execute(out)
        finished = self.scheduler.update_from_output(
            out, sampled, stop_token_ids=self.stop_token_ids
        )
        for req in finished:
            self._params.pop(req.req_id, None)
            self._generators.pop(req.req_id, None)
        return finished

    # -- execution ----------------------------------------------------------------

    def _execute(self, out: SchedulerOutput) -> dict[str, list[int]]:
        """Run the scheduled token counts; sample where the gap closes.

        Split: requests scheduled >1 token run as per-request prefill chunks
        (the runner continues from whatever state the slots hold, so a chunk
        is resume and prefill alike); requests scheduled exactly 1 token —
        steady-state decodes plus any 1-token budget crumbs — run as one
        batched decode step.
        """
        prefills: list[tuple[Request, int]] = []
        decodes: list[Request] = []
        for req_id, n in out.num_scheduled_tokens.items():
            req = self.scheduler.requests[req_id]
            if n == 1:
                decodes.append(req)
            else:
                prefills.append((req, n))

        sampled: dict[str, list[int]] = {}
        for req, n in prefills:
            logits = self.runner.prefill(self._feed_tokens(req, n), req.slots)
            if self._closes_gap(req, n):
                sampled[req.req_id] = [self._sample(req, logits)]
        if decodes:
            logits = self.runner.decode_step(
                [req.slots for req in decodes],
                [self._feed_tokens(req, 1)[0] for req in decodes],
            )
            # One batched argmax + one host sync for the whole batch; the
            # non-greedy path falls back to per-row sampling.
            greedy = logits.argmax(dim=-1).tolist()
            for row, req in enumerate(decodes):
                if not self._closes_gap(req, 1):
                    continue  # mid-prefill crumb: state advanced, no sample
                if self._params[req.req_id].temperature <= 0.0:
                    sampled[req.req_id] = [greedy[row]]
                else:
                    sampled[req.req_id] = [self._sample(req, logits[row])]
        return sampled

    def _feed_tokens(self, req: Request, n: int) -> list[int]:
        """The n token ids whose state this step computes: the slice of
        (prompt + outputs) starting at ``num_computed_tokens``."""
        start = req.num_computed_tokens
        if start < req.num_prompt_tokens:
            tokens = (req.prompt_token_ids + req.output_token_ids)[start:start + n]
        else:  # steady-state decode: avoid rebuilding the full list
            tokens = req.output_token_ids[start - req.num_prompt_tokens:][:n]
        assert len(tokens) == n, f"{req.req_id}: scheduled past known tokens"
        return tokens

    @staticmethod
    def _closes_gap(req: Request, n: int) -> bool:
        """True when this step catches state up to the last known token —
        the moment a next-token distribution exists to sample from."""
        return req.num_computed_tokens + n == req.num_tokens

    def _sample(self, req: Request, logits: torch.Tensor) -> int:
        return sample_token(
            logits, self._params[req.req_id], self._generators[req.req_id]
        )
