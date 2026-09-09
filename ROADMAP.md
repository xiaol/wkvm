# Roadmap

Milestones are cumulative; each one is a working artifact with a demo or a green test suite.
Rationale for every design choice lives in `docs/ANGLE.md` §5 and `docs/RECURRENT_MODE.md` §6.

## M0 — Core bookkeeping (this commit)

- `StateArena`: per-family fixed-slot allocator; exact admission; fork = refcounted slot copy semantics reserved in the interface.
- No-phases scheduler: one integer per request (`num_computed_tokens`) converging to target under a global token budget — chunked prefill, decode, and resume fall out of one loop (stolen from vLLM v1's cleanest invariant).
- Pure Python, no torch import in `wkvm/core`; unit tests run anywhere.

## M1 — RWKV-7 decode on one GPU

- Model: RWKV-7 (start ~1.5B for iteration speed, then 7B).
- FLA kernels: chunked scan for prefill, recurrent update for decode; state arena owns the `[slots, ...]` tensors per layer family.
- Greedy + temperature/top-p sampling; logit parity harness vs the reference implementation (per-layer divergence check).
- Exit: correct completions, batch=N decode from the arena.

## M2 — Continuous batching + CUDA graphs + server

- Wire M0 scheduler to M1 runner; overlap scheduling (bounded future queue) from day one — retrofitting it is what scarred SGLang.
- Whole-step decode CUDA graphs keyed by batch size buckets (uniform batches make this trivially safe).
- `wkvm.core.mixed_batch.MixedBatchMetadata` defines the dependency-free
  ragged contract used to combine decode rows with unequal prefill rows; the
  native runner integration and CPU parity/fallback gates are implemented.
- Minimal OpenAI-compatible HTTP frontend in its own process; engine speaks token ids only.
- Exit: Albatross-comparison benchmark — decode tok/s vs batch size, flat VRAM chart.

## M3 — StateStore + Durable State API (the moat) — ✅ DONE 2026-07-06 (see experiments/results/m3_results.md)

- Position-indexed checkpoint store (boundary snapshots every N tokens), GPU → pinned host → NVMe tiering.
- `/v1/states`: create-from-prompt, save/load (safetensors), fork (COW slot copy), pin, delete; sessions = named states; hibernate/resume; rollback-to-checkpoint.
- Mutation hooks: registered state-update rules (decay / merge / consolidate).
- Exit demos: agent survives engine restart; corpus-state forked to 1k requests; N-thousand warm sessions with sub-100ms resume.

## M4 — Hybrid guest allocator — ✅ DONE 2026-09-09 (docs/HYBRID_ENGINE_PLAN.md, experiments/results/m4_qwen35_hybrid_smoke.md)

- Qwen3.5-9B (24 Gated DeltaNet + 8 full-attention layers) served from the arena with three state families: `gdn_state`, `gdn_conv` (slots) and `guest_kv` (pages). The arena is the one allocator for both: a request reserves `ceil((prompt + max_new_tokens) / page_tokens)` pages for its lifetime at admission, so admission stays exact (free slots AND free pages) and nothing is preempted mid-flight. Page-bytes unification with state pages turned out unnecessary once both kinds live in one exact allocator.
- `Engine`/`StateStore` generalised behind layout factories and a bank protocol; RNN-StateTuning initial states import as durable handles (`/v1/states/import`, `rule="import"` provenance).
- Evidence: CPU parity suite `tests/test_qwen35_hybrid_cpu.py` + torch-free `tests/test_pages.py`; 9B smoke bit-exact vs HF including a 12k-token prompt through the paged pool.
- H5 progress: fla Triton GDN kernels selected at load (`wkvm/runner/kernels.py`); prefill of a 12.6k-token prompt 12.1 s → 4.2 s, decode was launch-bound (~6k launches for ~23 ms of GPU time); CUDA-graphed decode (`wkvm/runner/hybrid_graph.py`) with resident rows brings the 9B step to 21 ms at B=1 and 26 ms at B=16 (616 tok/s) on one A100 — the replay itself. Native RWKV-7 verified on an A100 from official `.pth` weights (`scripts/convert_rwkv7_pth.py`) against the official `rwkv` package.
- Not done here: gather-free paged attention kernel (FA3/FlashInfer) and CUDA graphs on the hybrid decode — H5 in the plan. Pure transformer as all-guest parity baseline deferred (a spec needs at least one slot family; trivial to add a 1-byte identity family when wanted).

## M5 — Recurrent mode for transformers (`docs/RECURRENT_MODE.md`)

1. Ring-only slots (sink + sliding window) — also the SWA path hybrids need anyway.
2. Timescale bank: K states, per-state decay, no boundary logic (static graph ops).
3. Segmented bank: novelty/DLA boundaries + capacity-bounded merge, exposed as a `/v1/states` mutation op.
4. RWKV-MS sidecar loader (hash-bound validation); tau2 parity at batch N vs the single-sequence llama.cpp baseline.
5. Per-model layer budget profiles; flat-memory-vs-context demo chart.

## Deferred / refused

- Speculative decoding via N+1 state slots (only if it earns its complexity), P/D disaggregation (one transport, one RDMA write per session), TP (manual sharding in the importer when needed).
- Refused permanently: model zoo breadth, multi-platform matrix, multi-backend grammar, god-configs. See `docs/ANGLE.md` §3.

## Day-one, retrofit-hostile decisions (enforced from M1)

1. Tag-scoped GPU allocator (weights / states / KV pools) so RL sleep/wake works later.
2. One `process_weights_after_loading` hook — the only quantization coupling point.
3. Metrics ride the output wire; nothing scrapes the busy loop.
4. Per-request sampler/grammar state must be clonable (fork depends on it).
5. Typed frozen configs per subsystem; no god-config threading.
