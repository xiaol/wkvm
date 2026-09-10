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

import dataclasses

import torch

from wkvm.core.arena import StateArena
from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request, RequestStatus
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
        cuda_graphs: bool | dict = False,
    ) -> None:
        """``num_pages`` sizes the guest page pool of hybrid models (ignored
        by models without paged families). Default: 4096 tokens per slot
        worth of pages, shared across all requests.

        ``cuda_graphs``: True (or a dict of ``HybridDecodeGraphs`` kwargs)
        captures decode forwards per batch/length bucket on runners that
        support it (the hybrid runner); batch buckets are capped at
        ``num_slots``."""
        self.layout = layout
        spec = layout.state_spec()
        if spec.paged_families:
            if num_pages is None:
                num_pages = num_slots * spec.pages_for(4096)
        else:
            num_pages = 0
        self.bank = layout.make_bank(num_slots, device, num_pages=num_pages)
        self.arena = StateArena(spec, num_slots=num_slots, num_pages=num_pages)
        self.runner = layout.make_runner(model, self.bank, prefill_chunk)
        # The scheduler may hand a request at most one runner chunk per step:
        # batched prefill needs it, and a runner can bound its chunk below
        # ``prefill_chunk`` (routed guests: the pending buffer and the ring).
        config = scheduler_config or SchedulerConfig(max_running_requests=num_slots)
        if config.max_tokens_per_request_per_step > self._runner_chunk():
            config = dataclasses.replace(config, max_tokens_per_request_per_step=self._runner_chunk())
        self.scheduler = Scheduler(config, self.arena)
        if cuda_graphs and hasattr(self.runner, "enable_cuda_graphs"):
            kwargs = dict(cuda_graphs) if isinstance(cuda_graphs, dict) else {}
            buckets = kwargs.get("batch_buckets", (1, 2, 4, 8, 16, 32))
            buckets = tuple(b for b in buckets if b <= num_slots) or (1,)
            if buckets[-1] < num_slots and num_slots <= 64:
                buckets = buckets + (num_slots,)
            kwargs["batch_buckets"] = buckets
            cap = self.arena.max_tokens_per_request
            if cap is not None:
                lb = kwargs.get("length_buckets", (256, 512, 1024, 2048, 4096))
                kwargs["length_buckets"] = tuple(l for l in lb if l <= cap) or (min(lb),)
            self.runner.enable_cuda_graphs(**kwargs)
        self.stop_token_ids = stop_token_ids
        self._params: dict[str, SamplingParams] = {}
        self._generators: dict[str, torch.Generator | None] = {}
        self.store = None
        self._save_on_finish: dict[str, str] = {}
        self._finish_handles: dict[str, str] = {}
        # The one moment end-of-life state is still addressable: snapshot if
        # armed, then let the bank drop any resident graph row.
        self.scheduler.on_finish = self._on_finish

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
        guest_mode: str = "paged",
        sink_tokens: int = 16,
        ring_tokens: int = 1024,
        routed_params: dict | None = None,
        decode_attention: str = "auto",
        **kwargs,
    ) -> "Engine":
        """Qwen3.5 hybrid (Gated DeltaNet + full-attention guests).

        ``guest_mode="paged"``: exact attention over a paged pool;
        ``guest_pool_tokens`` is the total guest-KV pool shared by all
        requests (default: 4096 per slot); a single request may reserve up
        to the whole pool. ``guest_mode="ring"``: sink + sliding window per
        slot (``sink_tokens`` + ``ring_tokens``), constant memory, unbounded
        context, approximate beyond the window."""
        from wkvm.models.qwen35 import load_qwen35

        model, layout = load_qwen35(
            model_path, device=device, dtype=dtype, page_tokens=page_tokens,
            guest_mode=guest_mode, sink_tokens=sink_tokens, ring_tokens=ring_tokens, **(routed_params or {}),
            decode_attention=decode_attention,
        )
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
        self, request: Request, params: SamplingParams = SamplingParams(),
        park_on_finish: bool = False,
    ) -> None:
        """Queue a request. Legal at any time, including between steps.

        ``park_on_finish``: keep the state slots after the turn completes so
        the session can be continued with ``continue_request`` (multi-turn
        without re-prefill)."""
        if params.stop_token_ids and params.stop_token_ids != self.stop_token_ids:
            raise ValueError(
                "per-request stop_token_ids must be empty or equal to the "
                "engine-global set (single stop set until the server frontend)"
            )
        self._check_capacity(request.num_tokens, request.max_new_tokens)
        self.scheduler.add_request(request, park_on_finish=park_on_finish)
        self._params[request.req_id] = params

    def continue_request(self, req_id: str, new_tokens: list[int], max_new_tokens: int) -> Request:
        """Append a turn to a parked session and put it back in the running
        set: the appended tokens are the schedulable gap, so the ordinary
        loop prefills them (chunked) and decodes on — no special path."""
        req = self.scheduler.parked.get(req_id)
        if req is None:
            raise ValueError(f"{req_id}: not parked")
        if not new_tokens:
            raise ValueError("continue_request needs at least one token")
        self._check_capacity(req.num_tokens + len(new_tokens), max_new_tokens)
        req.output_token_ids.extend(new_tokens)
        req.max_new_tokens = len(req.output_token_ids) + max_new_tokens
        if req.parked_finish_status is not None:
            pass
        self.scheduler.resume_parked_request(req_id)
        self.scheduler._park_on_finish.add(req_id)
        if req_id not in self._generators:
            self._generators[req_id] = make_generator(self._params[req_id], self.runner.device)
        return req

    def close_request(self, req_id: str) -> None:
        """Release a parked session's slots (and any resident graph row)."""
        req = self.scheduler.parked.get(req_id)
        if req is not None and req.slots:
            self._release_bank(req.slots)
        self.scheduler.close_parked_request(req_id)
        self._params.pop(req_id, None)
        self._generators.pop(req_id, None)

    def abort_request(self, req_id: str) -> None:
        req = self.scheduler.requests.get(req_id)
        if req is not None and req.slots:
            self._release_bank(req.slots)
        self.scheduler.abort_request(req_id)
        self._params.pop(req_id, None)
        self._generators.pop(req_id, None)

    def _release_bank(self, slots: dict) -> None:
        release = getattr(self.bank, "release_slots", None)
        if release is not None:
            release(slots)

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
            if req.status is RequestStatus.PARKED:
                continue  # session kept: params/RNG survive for continue_request
            self._params.pop(req.req_id, None)
            self._generators.pop(req.req_id, None)
        return finished

    # -- execution ----------------------------------------------------------------

    def _runner_chunk(self) -> int:
        """Largest prefill chunk the runner takes in one forward."""
        chunk = getattr(self.runner, "chunk_size", None)
        return chunk() if callable(chunk) else self.runner.prefill_chunk

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
        batch_prefill = getattr(self.runner, "prefill_batch", None)
        if batch_prefill is not None and len(prefills) > 1:
            # Group equal-length chunks (<= one runner chunk) into one forward;
            # everything else takes the per-request path below.
            groups: dict[int, list[tuple[Request, int]]] = {}
            chunk = self._runner_chunk()
            for req, n in prefills:
                if n <= chunk:
                    groups.setdefault(n, []).append((req, n))
            done: set[str] = set()
            for n, members in groups.items():
                if len(members) < 2:
                    continue
                logits = batch_prefill([(self._feed_tokens(req, n), req.slots) for req, _ in members])
                for (req, _), row in zip(members, logits):
                    done.add(req.req_id)
                    if self._closes_gap(req, n):
                        sampled[req.req_id] = [self._sample(req, row)]
            prefills = [(req, n) for req, n in prefills if req.req_id not in done]
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
        """Create the StateStore (snapshot-on-finish is wired at construction)."""
        from wkvm.store import StateStore

        self.store = StateStore(self.bank, store_dir)

    def _on_finish(self, req: Request) -> None:
        self._snapshot_on_finish(req)
        if req.req_id in self.scheduler._park_on_finish:
            # Parked: the slots stay; a resident graph row must be written
            # back (parked requests are not in the next batch) — not dropped.
            park = getattr(self.bank, "park_slots", None)
            if park is not None:
                park(req.slots)
            return
        self._release_bank(req.slots)

    def _snapshot_on_finish(self, req: Request) -> None:
        name = self._save_on_finish.pop(req.req_id, None)
        if name is not None:
            if self.store is None:
                raise RuntimeError("save_on_finish armed without a store (call attach_store)")
            self._finish_handles[req.req_id] = self.store.save(
                name,
                req.slots,
                num_computed_tokens=req.num_computed_tokens,
                token_ids=req.prompt_token_ids + req.output_token_ids,
            )

    def save_on_finish(self, req_id: str, name: str) -> None:
        """Arm an automatic snapshot for when this request finishes."""
        if self.store is None:
            raise RuntimeError("attach_store() before save_on_finish()")
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
