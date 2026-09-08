"""Engine: the M0 scheduler driving a model runner (M2-minimal).

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
- **The engine never branches on model family.** A *layout* (see
  ``wkvm/models/rwkv7.py``, ``wkvm/models/qwen35.py``) knows its state
  families and builds its own bank and runner (``make_bank``/``make_runner``);
  the engine only relies on the shared bank/runner contract
  (``zero_slots``, ``prefill``, ``decode_step``, the store protocol).

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
from wkvm.runner.sampling import SamplingParams, make_generator, sample_token


class Engine:
    """Owns Scheduler + StateArena + a state bank + a runner."""

    def __init__(
        self,
        model,
        layout,
        num_slots: int,
        scheduler_config: SchedulerConfig | None = None,
        device: torch.device | str = "cuda",
        stop_token_ids: frozenset[int] = frozenset(),
        prefill_chunk: int = 512,
        num_pages: int | None = None,
    ) -> None:
        """``num_pages`` sizes the guest page pool of hybrid models (ignored
        by models without paged families). Default: 4096 tokens per slot
        worth of pages, shared across all requests."""
        self.layout = layout
        spec = layout.state_spec()
        if spec.paged_families:
            if num_pages is None:
                num_pages = num_slots * spec.pages_for(4096)
        else:
            num_pages = 0
        self.bank = layout.make_bank(num_slots, device, num_pages=num_pages)
        self.arena = StateArena(spec, num_slots=num_slots, num_pages=num_pages)
        self.scheduler = Scheduler(
            scheduler_config or SchedulerConfig(max_running_requests=num_slots),
            self.arena,
        )
        self.runner = layout.make_runner(model, self.bank, prefill_chunk)
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
        """RWKV-7 (fla-format checkpoint)."""
        from wkvm.models.rwkv7 import load_rwkv7

        model, layout = load_rwkv7(model_path, device=device, dtype=dtype)
        return cls(model, layout, num_slots=num_slots, device=device, **kwargs)

    @classmethod
    def from_qwen35(
        cls,
        model_path: str,
        num_slots: int,
        guest_pool_tokens: int | None = None,
        page_tokens: int = 256,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "Engine":
        """Qwen3.5 hybrid (Gated DeltaNet + paged full-attention guests).

        ``guest_pool_tokens`` is the total guest-KV pool shared by all
        requests (default: 4096 per slot); a single request may reserve up
        to the whole pool."""
        from wkvm.models.qwen35 import load_qwen35

        model, layout = load_qwen35(model_path, device=device, dtype=dtype, page_tokens=page_tokens)
        num_pages = None
        if guest_pool_tokens is not None:
            num_pages = layout.state_spec().pages_for(guest_pool_tokens)
        return cls(model, layout, num_slots=num_slots, device=device, num_pages=num_pages, **kwargs)

    # -- intake ---------------------------------------------------------------

    def _check_capacity(self, num_tokens: int, max_new_tokens: int) -> None:
        """A request that could never fit the guest pool is rejected at
        intake, not left waiting forever; a request that fits waits for pages
        under exact admission and can never outgrow its reservation."""
        cap = self.arena.max_tokens_per_request
        if cap is not None and num_tokens + max_new_tokens > cap:
            raise ValueError(
                f"request needs up to {num_tokens + max_new_tokens} tokens, "
                f"guest pool holds {cap}"
            )

    def add_request(
        self, request: Request, params: SamplingParams = SamplingParams()
    ) -> None:
        """Queue a request. Legal at any time, including between steps."""
        if params.stop_token_ids and params.stop_token_ids != self.stop_token_ids:
            raise ValueError(
                "per-request stop_token_ids must be empty or equal to the "
                "engine-global set (single stop set until the server frontend)"
            )
        self._check_capacity(request.num_tokens, request.max_new_tokens)
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

    # -- durable state (M3) ---------------------------------------------------

    def attach_store(self, store_dir) -> None:
        """Create the StateStore and wire snapshot-on-finish."""
        from wkvm.store import StateStore

        self.store = StateStore(self.bank, store_dir)
        self._save_on_finish: dict[str, str] = {}
        self._finish_handles: dict[str, str] = {}
        self.scheduler.on_finish = self._snapshot_on_finish

    def _snapshot_on_finish(self, req: Request) -> None:
        name = self._save_on_finish.pop(req.req_id, None)
        if name is not None:
            self._finish_handles[req.req_id] = self.store.save(
                name,
                req.slots,
                num_computed_tokens=req.num_computed_tokens,
                token_ids=req.prompt_token_ids + req.output_token_ids,
            )

    def save_on_finish(self, req_id: str, name: str) -> None:
        """Arm an automatic snapshot for when this request finishes."""
        self._save_on_finish[req_id] = name

    def snapshot_request(self, req_id: str, name: str) -> str:
        """Snapshot a RUNNING request's state as-of-now; it keeps running."""
        req = self.scheduler.requests[req_id]
        if not req.slots:
            raise ValueError(f"{req_id} holds no slots (not running)")
        # Full known-token list; it may exceed num_computed_tokens (a sampled
        # but unfed token, or an unprefilled remainder) — that surplus IS the
        # schedulable gap a resume starts from.
        return self.store.save(
            name,
            req.slots,
            num_computed_tokens=req.num_computed_tokens,
            token_ids=req.prompt_token_ids + req.output_token_ids,
        )

    def hibernate(self, req_id: str, name: str) -> str:
        """Snapshot a running session and release its slot."""
        handle = self.snapshot_request(req_id, name)
        self.abort_request(req_id)
        return handle

    def import_state(self, name: str, path: str, **kwargs) -> str:
        """Register an externally produced state (e.g. a tuned initial state
        from RNN-StateTuning) as a handle with ``num_computed_tokens = 0`` and
        no tokens: generation from it starts at the prompt, from a state no
        token prefix produces. The layout decides what files it understands."""
        loader = getattr(self.layout, "load_state_adapter", None)
        if loader is None:
            raise NotImplementedError(f"{type(self.layout).__name__} has no state-adapter loader")
        tensors, meta = loader(path, **kwargs)
        return self.store.import_state(name, tensors, metadata=meta)

    def submit_from_handle(
        self,
        handle: str,
        suffix_tokens: list[int] | None = None,
        max_new_tokens: int = 128,
        params: SamplingParams = SamplingParams(),
    ) -> Request:
        """Resume a stored state as a live request.

        The record's token list may run one past its computed count (a
        sampled-but-unfed token); together with any suffix that gap is what
        the scheduler sees — resume needs no special path in the loop."""
        record = self.store.get(handle)
        tokens = list(record.token_ids) + list(suffix_tokens or [])
        if len(tokens) <= record.num_computed_tokens:
            raise ValueError(f"{handle}: nothing to schedule (add suffix tokens)")
        self._check_capacity(len(tokens), max_new_tokens)
        pages = self.arena.pages_for(len(tokens) + max_new_tokens)
        slots = self.arena.allocate(pages=pages)
        try:
            record = self.store.load(handle, slots)
        except Exception:
            self.arena.free(slots)
            raise
        req = Request(prompt_token_ids=tokens, max_new_tokens=max_new_tokens)
        req.num_computed_tokens = record.num_computed_tokens
        self.scheduler.add_resumed_request(req, slots)
        self._params[req.req_id] = params
        self._generators[req.req_id] = make_generator(params, self.runner.device)
        return req

    resume = submit_from_handle
