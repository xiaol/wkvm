# Hybrid Engine Plan (M4): Gated DeltaNet + full-attention guests

Goal: make the README's first sentence true for a GDN hybrid — serve
Qwen3.5-9B (24 Gated DeltaNet layers + 8 full-attention layers, the
"Qwen3-Next class" of ROADMAP M4) from wkvm's own arena/scheduler/store, with
the recurrent state as the first-class object and the attention layers as
guests, and make the Durable State API's third way of obtaining a state —
being *given* one — real on a trained artifact (RNN-StateTuning initial
states).

Status legend: ✅ done in this checkout, ▶ next, ☐ later.

## Why this is the next step

- ROADMAP M0–M3 are done; M4 is the next milestone and its first model is a
  Qwen3-Next-class hybrid. `docs/ANGLE.md` §5: "RWKV-7 first, then one GDN
  hybrid to force the guest-allocator path honest."
- `paper/CLAIMS.md` and the report's limitations exclude "general native
  support for GDN, Mamba2, Qwen3-Next, or arbitrary hybrid architectures";
  `wkvm/models/rwkv7.py` refuses hybrids with "hybrid attn layers are M4".
- The user's RNN-StateTuning line produces tuned GDN initial states for
  Qwen3.5-9B (scene-boundary task: 0% → 100% valid JSON, 34% exact match).
  Those are states that no token prefix produces — the exact capability the
  moat argument rests on — and they had no batched runtime.

## Decisions

1. **Same ownership split as M1.** HF `Qwen3_5DecoderLayer` modules are the
   compute graph; wkvm owns state, positions, masks and the cache object
   (`SlotCache`). No fork of HF, no patched `DynamicCache`.
2. **Guest allocator = paged KV pool inside the arena, reserved for the
   request's lifetime.** `guest_kv` is a *paged family*
   (`StateFamilySpec.page_tokens`); the arena hands out
   `ceil((prompt + max_new_tokens) / page_tokens)` pages at admission next to
   the recurrent slots, so the M0 invariant survives unchanged: admission is
   a count (free slots AND free pages), nothing is preempted or retracted
   mid-flight, and a request that could never fit the pool is rejected at
   intake. The cost is that a request which stops early at EOS held pages it
   did not use; the benefit is exactness and a torch-free allocator that is
   unit-tested without a GPU (`tests/test_pages.py`). ROADMAP's "page-bytes
   unified with state pages" trick exists in vLLM to make one block pool serve
   two kinds of memory; with slots and pages both living in the arena there
   is no second allocator to unify, so it is not needed. (Slice 1 of this
   milestone used a fixed per-slot window; it is superseded.)
3. **The engine never branches on model family.** Layouts build their bank and
   runner (`make_bank`/`make_runner`); the store talks to banks through a
   four-method protocol (`fingerprint_key`, `export_slot`, `import_slot`,
   `memory_families`). RWKV-7 keeps its store keys and fingerprint string, so
   M3 indexes remain valid.
4. **Pure-torch kernels are acceptable for this slice.** Throughput is not a
   claim here; correctness and the state contract are. FLA-class kernels
   plug in unchanged when available (HF dispatches to them by name).

## Milestones

### H0. Contract ✅

`docs/qwen35_hybrid_contract.md`: layer types, three state families and
shapes, zero-state == fresh proof, the exact cache calls the HF layers make,
padded-row attention layout and mask formula, positions, admission rule,
store protocol.

### H1. Layout, bank, runner ✅

- `wkvm/models/qwen35.py`: `Qwen35HybridLayout` (families `gdn_state`,
  `gdn_conv`, `guest_kv`; per-slot bytes; factories), `Qwen35Decoder`
  (wkvm-driven layer loop), `load_qwen35`, `load_state_adapter`.
- `wkvm/runner/hybrid_state.py`: `Qwen35StateBank` (layer-major banks,
  slot 0 reserved) + `SlotCache`.
- `wkvm/runner/hybrid_runner.py`: `Qwen35HybridRunner` with the M1 runner
  contract.

Acceptance (CPU, tiny random model, fp32):

```bash
CUDA_VISIBLE_DEVICES= python -m unittest tests.test_qwen35_hybrid_cpu -v
```

Gates in that file: chunked prefill last-logits vs HF full forward at
1e-4; engine greedy == HF `generate`; batched decode with three prompt
lengths == each alone; guest window enforced at intake; hibernate/resume
exact across WARM → COLD → index rebuild; `decay` touches `gdn_state` only;
an imported tuned initial state reproduces HF's own seeded
`LinearAttentionLayer` cache path token for token; adapter validation
(PLE rejected, wrong layer rejected, partial adapters only with
`strict=False`).

### H2. Engine + store generalisation ✅

`Engine(model, layout, ...)` builds bank/runner from the layout;
`Engine.from_qwen35`; `Engine.import_state(name, path)`;
`StateStore.import_state` records `rule="import"` provenance (source, SHA-256,
tuned layers); `/v1/states/import` on the stdlib server; `--model-type
qwen35 --guest-ctx`.

### H3. Real-checkpoint smoke ✅ (numbers in `experiments/results/m4_qwen35_hybrid_smoke.md`)

```bash
PYTHONPATH=. python experiments/qwen35_hybrid_smoke.py \
  --json experiments/results/m4_qwen35_hybrid_smoke.json
```

Certifies on Qwen3.5-9B bf16 (engine on one A100, independent HF reference
on another): greedy parity vs HF `generate`; continuous batching of distinct
scene prompts vs alone; an RNN-StateTuning adapter imported as a handle and
generated from at batch, compared with the adapter's own runtime and scored
on the scene task (valid JSON / exact match, zero vs tuned state); COLD
resume of the imported handle; per-slot bytes and slot capacity.

### H4. Paged guest pool ✅

- `StateFamilySpec.page_tokens` marks a paged family; `ModelStateSpec`
  exposes `pages_for(num_tokens)`, `bytes_per_page`; `StateArena(spec,
  num_slots, num_pages)` allocates `slots[name] = (page ids...)` for paged
  families and frees/forks them; the scheduler admits with
  `arena.can_admit(pages=pages_for(prompt + remaining budget))` (FCFS,
  head-of-line blocks). Page 0 is the dummy read target of masked positions.
- `Qwen35StateBank` keeps K/V pools `[n_attn, P+1, kv_heads, page_tokens,
  head_dim]`; token `t` of a request lives at `pages[t // page_tokens]`,
  offset `t % page_tokens`. Gather materialises only the batch's resident
  tokens (`B x Lmax`) with one advanced-index op per layer; decode scatter is
  one fused write per layer. Snapshots are token-trimmed and page-size
  independent (the fingerprint excludes `page_tokens`).
- `Engine.from_qwen35(..., guest_pool_tokens, page_tokens)`; default pool is
  4096 tokens per slot, shared, so one request may take the whole pool.
- Acceptance: `tests/test_pages.py` (torch-free) and
  `tests/test_qwen35_hybrid_cpu.py` with 8-token pages so every prompt and
  decode crosses page boundaries; 9B smoke phase E: a 12k-token natural-text
  prompt through 49 pages, greedy-identical to HF.

### H5. Throughput floor ▶ (kernels ✅, profile ✅, graphs ✅, paged kernel ☐)

- **CUDA-graph decode with resident rows ✅** (`wkvm/runner/hybrid_graph.py`,
  `Engine(..., cuda_graphs=True)` / `Engine.from_qwen35(..., cuda_graphs=True)`):
  the decode forward is captured once per (batch bucket, length bucket) over
  static buffers allocated at the largest bucket and sliced as views; mask
  and positions are computed in-graph from a static `lens` tensor. A running
  request's state stays *resident* in a row between steps: the GDN state is
  updated in place and the new K/V token is written in-graph at column
  `lens[row]`, so nothing is gathered or scattered per step. The bank
  (recurrent slots + pages) remains the durable owner: a row is filled once
  when the request first decodes, flushed when eager code needs the state
  (store export, snapshot), evicted when the request leaves the graph path
  (prefill, eager fallback, absent from the batch), dropped on finish/abort.
  Rows stay contiguous by swap-on-departure. Gate:
  `tests/test_hybrid_graph_gpu.py` (eager == graph token-for-token with
  staggered departures and arrivals, bucket transitions, eager fallback,
  hibernate mid-decode, store resume). On the 9B (fla kernels), the step is
  now the replay itself:

  | B | eager step | graphed step (engine.step) | replay alone |
  |---:|---:|---:|---:|
  | 1 | 82 ms | 21 ms (48 tok/s) | 18 ms |
  | 4 | 85 ms | 20 ms (196 tok/s) | 19 ms |
  | 8 | 88 ms | 22 ms (369 tok/s) | 20 ms |
  | 16 | 96 ms | 26 ms (616 tok/s) | 24 ms |

  The full smoke under graphs (`m4_qwen35_hybrid_smoke_fla_graphs.json`)
  keeps every gate: bit-exact parity, batched == alone 8/8, tuned handle ==
  adapter runtime 8/8, cold resume identical, 12.6k-token prompt identical
  through the eager fallback; the 8-prompt batched run drops 5.6 s → 3.7 s.

- **fla Triton kernels ✅** (`wkvm/runner/kernels.py`, `WKVM_KERNELS=auto|fla|torch`):
  transformers dispatches Gated DeltaNet to `fla.ops.gated_delta_rule` when
  `fla` imports; wkvm decides that before any modeling import and, on
  torch < 2.7, imports `fla` with `torch.compile` stubbed (its import-time
  compile trips over Triton >= 3.3). Effect on the 9B: 12.6k-token prefill
  12.1 s → 4.2 s, batched 8-prompt run 8.8 s → 5.6 s, decode step only
  89 → 82 ms at B=1 and 103 → 96 ms at B=16. Parity vs HF unchanged
  (bit-exact, HF on the same kernels).
- **Decode profile ✅** (`experiments/hybrid_decode_profile.py`,
  `experiments/results/m4_hybrid_decode_profile_fla.json`): at B=1 a step
  is 82 ms wall of which the model forward is 77 ms, and only ~23 ms of that
  is GPU kernel time — the rest is ~6,300 kernel launches of HF-module
  glue (`aten::to/copy_/mul` dominate CPU time, the GEMMs dominate GPU
  time). At B=16: forward 94 ms (GPU ~26 ms), gather 12 ms, scatter 9 ms.
  Conclusion: the floor is launch overhead, not kernels.
- **GQA-native decode attention ✅** (`wkvm_gqa`, registered as an HF
  attention implementation in `wkvm/models/qwen35.py`): with any mask
  present HF's sdpa path calls `repeat_kv`, materialising 4 copies of K and
  V per layer per step — at 32 rows x 5.7k routed columns that copy *was*
  the step. The single-query path broadcasts the KV heads over their query
  group in the matmul instead. Default for ring/routed guests (paged keeps
  HF's sdpa so the bit-exact gates hold): 16 x 13.8k ring turn 5.4 → 3.1 s,
  routed 6.7 → 3.8 s. Gate: logits vs sdpa in `tests/test_hybrid_graph_gpu.py`.
- **Next, in order:** (1) the replay itself is now the floor (18 ms at B=1
  for 32 layers of bf16 GEMMs on a 9B: weight-read bound, as expected); the
  levers left are within-graph — a fused GDN step over all 24 layers is not
  possible (layers are sequential), so this is the ceiling for a single
  request on one A100; (2) larger batch buckets (32/64) once the pool and
  static rows are sized for them (each row holds a full-length K/V window:
  B=64 at a 4k window is 8 GiB), or a paged attention kernel so rows can
  drop the window; (3) split mixed batches so one over-long request does not
  push the whole batch to the eager path; (4) pre-capture buckets at startup
  (~0.5 s per bucket on first use today).

### H7. Long-context quality and incumbent comparison ✅ (first pass)

- **RULER-lite** (`experiments/ruler_lite.py`): RULER's 12 synthetic tasks
  (niah x8, vt, cwe, fwe, qa_1) ported onto local SQuAD text with RULER's
  templates and metrics; 20 samples per task at 4k/8k/16k/32k. wkvm (graphs,
  fla) scores 1.00 on every cell; the HF reference on byte-identical prompts
  is the agreement gate (`--compare`), results in
  `experiments/results/ruler_lite_*.json`.
- **vLLM 0.29 on the same box** (`experiments/results/qwen35_hybrid_vs_vllm_20260910.md`):
  run through NVIDIA's CUDA 13 forward-compat library on the 12.8 driver.
  Same prompts, same shape: vLLM is 1.3x (348-token, B=1) to 2.2x
  (13,824-token, B=16) faster end to end. The gap is prefill (one request at
  a time through eager HF modules, ~7k tok/s vs ~21k), guest attention at
  long context (SDPA math over a masked 14k window, 28 ms vs 14 ms per
  step) and kernel granularity (~6.3k kernels per step even inside the
  graph). SGLang cannot run here (aarch64 kernel wheels have no sm_80 code).
- A capture bug surfaced on the way: `torch.cuda.graph` records the
  *current* device's stream, so on `cuda:1` the graph was empty and replays
  were silent no-ops. Capture/replay now run under `torch.cuda.device(bank)`
  and a post-capture replay check refuses graphs that do not recompute.

### H9. Ten times, the way Gemma got it: bounded guest memory ✅ (ring), ✅ (routed bank)

- **Insight.** On an exact hybrid both engines carry the same bytes per
  token, so the memory physics is a wash and vLLM's kernels win. The
  main-branch 10x came from a *bounded* per-session state that never
  re-prefills, measured past the incumbent's capacity wall. The same lever
  exists on the hybrid: the 8 guest layers are the only per-token memory.
- **Ring mode ✅** (`guest_mode="ring"`, `sink_tokens` + `ring_tokens`):
  the `guest_kv` family becomes a fixed per-slot window (33 MiB at 16+1024),
  keys stay post-RoPE at absolute positions, prefill uses a position band
  mask (in-chunk eviction exact), decode writes the new token into its ring
  column in-graph and masks `cols < min(len+1, window)`; one graph per batch
  bucket, no length buckets, unbounded context. Gates: `tests/test_qwen35_ring_cpu.py`
  (from-scratch band-mask reference, wraps, batched, hibernate/resume) and
  the ring case in `tests/test_hybrid_graph_gpu.py`.
- **Result** (`experiments/results/qwen35_ring_wall_20260910.md`): 32
  sessions x 36,864 tokens x 8 turns on one A100 — vLLM re-prefills every
  turn at 129–133 s; wkvm ring 5.25 s per turn: **24.6x per turn, 6.2x over
  8 turns, 16.5x on the README's 48-turn shape**. Below the wall (16 x
  13.8k) vLLM still wins 1.5x.
- **Price** (RULER-lite, ring 16+1024): needle recall only inside the
  window (0.25 at 4k -> ~0 at 32k); the GDN layers do not carry retrievable
  content. Same as Gemma's ring column.
- **Routed span bank ✅** (`guest_mode="routed"`, `wkvm/runner/hybrid_routed.py`,
  results in `experiments/results/qwen35_routed_bank_20260910.md`): per
  session and layer one static column space `[sink 16 | ring 1024 | pending
  2x512 | 64 summaries | 3,072 representatives | scratch]` = 5,201 columns,
  163 MiB across the 8 layers whatever the context. Ring evictions go to
  pending (in-graph during decode), a pass every 512 tokens cuts pending
  into sentence spans, assigns each by mean value to one of 64 slots (a
  running mean-K/V summary column per slot) and keeps exact representatives
  in one pool under the budget. Decisions are made once per session from all
  layers' features and applied to every layer; hibernate/resume carries the
  store and the host bookkeeping. Gates: `tests/test_qwen35_routed_cpu.py`,
  lockstep eager-vs-graph store comparison in `tests/test_hybrid_graph_gpu.py`.
- **The retention signal was found by measurement, not design**
  (`experiments/routed_probe.py`: one needle at a controlled depth, per-layer
  retention vs answer, `--trace` ranks the needle among all routed spans per
  signal). Gemma's rule — farthest-point in mean-V space — kept the needle in
  5/30 probes; token surprisal 11/30 (7-digit needles are single-digit tokens
  at ~2.3 nats; SQuAD names and dates score higher); **local novelty** (1 -
  max cosine to the other spans of the same pass) ranks the needle at the
  0.98–0.99 percentile in every prompt and keeps it **30/30 at 8k/16k/32k,
  all depths, answered correctly**. Readout was never the problem: a kept
  span is found at its original position exactly as in the exact engine.
  Also fixed on the way: a colon in the break set split needles in half;
  per-layer decisions kept a needle in six layers and dropped it in two.
- **Result**: RULER-lite single needle 1.00 at 4k–32k (ring: 0.15 → 0.00),
  4-key/4-value needles 0.75–1.00 through 16k, QA 0.95 → 0.45, haystack-of-
  needles 1.00 → 0.10 (nothing is novel there — the honest limit of a
  query-agnostic memory). Wall workload 32 x 36,864 x 8 turns: **5.5 s per
  turn vs vLLM's 130 s (24x), 229 s vs 1036 s over 8 turns (4.5x), 14x on
  the 48-turn shape**; ring without recall is 3.9 s (20x on 48 turns).
  Below the wall the routed engine is within 7% of vLLM per turn.
- **Costs and next cuts**: routing ~25 ms per pass on the host (turn 0 pays
  58 s over 2,304 passes because each pass syncs the prefill pipeline for
  one row — batch the rows of a prefill forward into one sync); a second retention term
  (similarity to the prompt's instruction prefix ranked the needle at 0.998
  too) is the next probe; the resident-row copy of the store doubles its
  memory under CUDA graphs (R=144 does not fit 32 sessions on 40 GB).

### H8. What the comparison says to build next ▶

1. Batched prefill: run every scheduled prefill chunk of the step as one
   ragged forward (`wkvm.core.mixed_batch` already defines the contract).
2. Paged attention kernel for the guest layers (FlashInfer decode/prefill
   over the pool's pages) — removes the masked SDPA and the row/page
   duplication at once.
3. Fused per-layer decode kernels (RMSNorm + projections + gated norm) to
   close the 4 ms per-step gap at B=1.

### H6. Second hybrid family ☐

Kimi-Linear / Qwen3-Next with MoE: MoE-by-dependency (fused-MoE kernels as
imports), same three-family contract plus a `mlp_only_layers` no-op.

## Effort

| Slice | Estimate | Proves |
|---|---:|---|
| H0–H4 (this checkout) | done | hybrid state contract, paged guests under exact admission, tuned-state import, exact hibernate/resume, batch-composition independence and 12k-token parity on the 9B |
| H5 throughput | 1–2 weeks | numbers comparable to the RWKV-7 M2 table |
| H6 second family | 1–2 weeks | the contract is not Qwen3.5-shaped |

## Risks

- HF Qwen3.5 internals move between releases (5.16 verified); the CPU test
  pins the cache calls and fails loudly on a contract change.
- bf16 batched matmuls can flip a greedy argmax vs single-row execution;
  the smoke reports prefix-match lengths rather than hiding it.
- Copying `B x Lmax` guest tokens per step caps decode throughput at high
  concurrency until H5's paged kernel.
- Lifetime reservation over-reserves for requests that stop early; a lazy
  page allocation with the same admission test is a small follow-up if pool
  pressure shows up in practice.
