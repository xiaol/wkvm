# Fixed state with recall on Qwen3.5-9B: the routed span bank (2026-09-10)

The ring result (`qwen35_ring_wall_20260910.md`) bought 24x per turn past
vLLM's capacity wall with a fixed 33 MiB guest memory per session, and lost
needle recall beyond the window. This note adds recall back while the state
stays fixed: `guest_mode="routed"` on the hybrid's 8 full-attention layers,
163 MiB per session whatever the context, and reports what it recalls, what
it costs, and — the part that took the day — *which retention signal makes
a query-agnostic memory keep the right spans*.

Artifacts: `mt_wkvm_routed48_ctx36864_b32_t8.json`,
`mt_wkvm_ring1024_ctx36864_b32_t8_gqa.json`, `mt_vllm_ctx36864_b32_t8.json`,
`mt_wkvm_routed48_ctx13824_b16_t8.json`, `ruler_lite_wkvm_routed48*.json`,
`ruler_lite_wkvm_routed144*.json`, `routed_probe_r48_*.json`.
Drivers: `experiments/hybrid_multiturn_bench.py --guest-mode routed`,
`experiments/ruler_lite.py --guest-mode routed`, `experiments/routed_probe.py`.
Code: `wkvm/runner/hybrid_routed.py` (mechanism), `wkvm/runner/hybrid_state.py`
(bank integration), `wkvm/runner/hybrid_graph.py` (in-graph eviction).

## The mechanism

Per session and attention layer, one static column space of 5,201 K/V
columns (32 KiB per column across the 8 layers):

```
[ sink 16 | ring 1024 | pending 2x512 | summaries 64 | representatives 64x48 | scratch ]
```

- **sink / ring**: exactly ring mode; keys post-RoPE at absolute positions.
- **pending**: tokens the ring overwrites move here (exact, still visible),
  in-graph during decode, before the forward during prefill. A routing pass
  runs when 512 are waiting. A pass that ends mid-sentence leaves the
  unfinished tail in pending for the next pass rather than cutting it.
- **routing pass**: pending is cut into sentence spans (`.!?` and newline
  tokens; a colon is *not* a break — it cut "X is: 1234567" in half), each
  span is assigned by its mean value vector to one of 64 slots (nearest
  centroid, or a free slot below cosine 0.60). Every slot keeps a running
  mean of its tokens' K and V — one **summary** column. Decisions are made
  once per session from all 8 layers' features and applied to every layer,
  so a span is either in every layer's memory or in none.
- **representatives**: one pool of 3,072 exact token columns under the
  whole budget, kept by **local novelty** — 1 minus the span's maximum cosine
  to the other spans of the same pass — with a 0.98-cosine duplicate floor.
  The pool is re-laid out in position order at every pass.
- **readout**: a decode step attends to every valid column: sink, ring,
  pending, 64 summaries, the pool. Nothing is gathered per step; the CUDA
  graph does the eviction and the mask in-graph, routing runs out-graph on
  the resident row.
- Hibernate/resume carries the store (`rt_*` tensors) and the host
  bookkeeping (`rt_state` blob: centroids, counts, kept spans with position
  and salience).

Everything above is exact eager-vs-graph in `tests/test_hybrid_graph_gpu.py`
(lockstep store comparison across acquisition, routing and row compaction)
and CPU-gated in `tests/test_qwen35_routed_cpu.py`.

## Finding the retention signal (`experiments/routed_probe.py`)

The probe puts one RULER needle ("One of the special magic numbers for
{word} is: {7 digits}.") at a controlled depth in a SQuAD haystack, prefills
through the routed engine, and reports two things separately: whether the
needle's tokens are still materialised in every layer, and whether the model
answers. A RULER score conflates them. R = 48, 64 slots, 2 seeds x 5 depths:

| retention rule | 8k | 16k | 32k | needle kept (all layers) | correct |
|---|---:|---:|---:|---:|---:|
| farthest-point in mean-V space, per layer (the Gemma rule) | 1.00 | 0.50 | 0.00 | 5/30 | 13/30 |
| farthest-point, shared decision | 0.40 | 0.20 | 0.10 | 10/30 | 10/30 |
| token surprisal (lm_head over the chunk, top-4 mean) | 0.80 | 0.20 | 0.10 | 11/30 | 11/30 |
| **local novelty, shared decision** | **1.00** | **1.00** | **1.00** | **30/30** | **30/30** |

Two facts fell out on the way:

1. **Readout is not the problem.** With shared decisions, every retained
   needle was answered correctly and every dropped one was wrong (10/10 and
   0/20 in the farthest-point row). A kept span at its original position is
   found by attention exactly as in the exact engine.
2. **Selection was.** `--trace` ranks the needle span among all routed spans
   of a prompt per candidate signal. At 16k and 32k (16 prompts): mean-V
   distinctness to the *kept set* had already failed; token surprisal put the
   needle at the 0.69 percentile (a 7-digit number is single-digit tokens at
   ~2.3 nats each — SQuAD's names and dates score higher); **local novelty
   ranked it at 0.98–0.99 in every prompt**, as did similarity to the
   prompt's own instruction prefix (0.998). Novelty is generic and needs no
   knowledge of the prompt, so it is the default; the instruction-prefix
   signal is a candidate second term.
3. Two implementation faults found by the same probe: a colon in the break
   set split needles in half (retention fractions of 0.38–0.43), and
   per-layer decisions retained a needle in six layers and dropped it in
   two, with the last layer dropping it at 16k every time.

Capacity was never the limit: a "keep everything" oracle (512-token slots,
per-layer farthest-point) still scored 0.50 on the single needle at 16k and
32k.

## RULER-lite with the bank (R = 48, novelty)

Same prompts as the exact run (all needle cells 1.00) and the ring run.
Full 12-task grid, 20 prompts per cell (`ruler_lite_wkvm_routed48.json`,
960 prompts, 24,150 routing passes, 3,360 s on one A100; the R = 144 grid
`ruler_lite_wkvm_routed144.json` is appended below when it completes):

| task | 4k | 8k | 16k | 32k | exact 4k–32k | ring 16+1024, 4k → 32k |
|---|---:|---:|---:|---:|---|---|
| niah_single_1 (noise haystack) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 → 0.10 |
| niah_single_2 (essay) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.15 → 0.00 |
| niah_single_3 (uuid value) | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 | 0.20 → 0.05 |
| niah_multikey_1 (4 keys) | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0.40 → 0.00 |
| niah_multivalue (4 values) | 1.00 | 0.85 | 0.95 | 1.00 | 1.00 | 0.33 → 0.01 |
| niah_multiquery (4 queries) | 1.00 | 0.90 | 0.98 | 1.00 | 1.00 | 0.30 → 0.06 |
| vt (variable tracking) | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 0.26 → 0.06 |
| fwe (frequent words) | 1.00 | 1.00 | 0.98 | 0.83 | 0.98–1.00 | 0.95 → 0.78 |
| cwe (common words) | 1.00 | 0.77 | 0.79 | 0.62 | 0.90–1.00 | 0.86 → 0.32 |
| qa_1 (SQuAD QA) | 0.95 | 0.70 | 0.45 | 0.15 | 0.85–1.00 | 1.00 → 0.10 |
| niah_multikey_2 (haystack of needles) | 1.00 | 0.45 | 0.10 | 0.05 | 1.00 | 0.20 → 0.05 |
| niah_multikey_3 (haystack of uuid needles) | 1.00 | 0.30 | 0.00 | 0.00 | 1.00 | 0.15 → 0.00 |

**R = 144** (`ruler_lite_wkvm_routed144.json`: 11,345 columns, 355 MiB per
session, a 9,216-token pool ≈ 30% of the evicted tokens at 32k; 4,997 s for
the grid). Budget buys back exactly the tasks the novelty rule cannot
separate — the others were already at the ceiling:

| task | 4k | 8k | 16k | 32k | vs R = 48 at 32k |
|---|---:|---:|---:|---:|---|
| niah_single_1 / 2 / 3 | 1.00 | 1.00 | 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.95 |
| niah_multikey_1 / multivalue / multiquery | 1.00 | 1.00 | 1.00 / 0.95 / 0.97 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| vt | 0.99 | 1.00 | 1.00 | 0.99 | 1.00 |
| fwe / cwe | 1.00 / 1.00 | 1.00 / 0.90 | 1.00 / 0.82 | 0.93 / 0.82 | 0.83 / 0.62 |
| qa_1 | 0.95 | 0.85 | 0.75 | 0.50 | 0.15 |
| niah_multikey_2 / 3 (haystacks of needles) | 1.00 / 1.00 | 1.00 / 1.00 | 0.70 / 0.50 | 0.40 / 0.05 | 0.05 / 0.00 |

Under CUDA graphs the resident-row copy doubles the store, so R = 144 fits
about 16 sessions on a 40 GB A100 next to the 9B weights rather than 32;
de-duplicating that copy is the memory item in the plan.

Five-task subset run first (`ruler_lite_wkvm_routed48_novelty_sub.json`,
same numbers for its five tasks):

| task | 4k | 8k | 16k | 32k | exact | ring 16+1024 (4k → 32k) |
|---|---:|---:|---:|---:|---:|---|
| niah_single_2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.15 → 0.00 |
| niah_multikey_1 (4 keys) | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 | 0.40 → 0.00 |
| niah_multivalue (4 values) | 1.00 | 0.85 | 0.95 | 1.00 | 1.00 | 0.33 → 0.01 |
| niah_multikey_2 (haystack of needles) | 1.00 | 0.45 | 0.10 | 0.05 | 1.00 | 0.20 → 0.05 |
| qa_1 (SQuAD QA) | 0.95 | 0.70 | 0.45 | 0.15 | 0.85–1.00 | 1.00 → 0.10 |

400 prompts, 10,097 routing passes, 262,795 spans, 59% of spans dropped
overall (at 32k the pool holds ~10% of the evicted tokens).

Reading: a needle in prose is recalled at every length; several needles
share the pool fairly (the 8k dips are two prompts whose four needles fell
into one pass and competed with each other for novelty — a multi-scale
novelty term is the fix to probe). Two tasks show the limits of a
query-agnostic memory honestly: when the *haystack itself* is needles
(multikey_2) nothing is novel and the pool holds a random 10% at 16k; QA
needs the one paragraph the question will ask about, which no signal at
eviction time can predict, so it degrades toward the pool fraction plus what
the summaries carry.

## Wall workload: 32 sessions x 36,864-token context, 8 turns

Same shape as the ring note. The decode attention now attends GQA-natively
(`wkvm_gqa`, no `repeat_kv` copies; ring 5.25 → 3.9 s per turn), routing
runs on the host from one device→host transfer per pass, and the drivers pin
`OMP_NUM_THREADS=8` (thread fan-out on tiny host ops had inflated every
earlier routing number on this 128-core box).

| engine | per-session memory | turn 0 (s) | steady turn (s) | total 8 turns (s) | peak GiB |
|---|---:|---:|---:|---:|---:|
| vLLM 0.29, exact full KV, prefix caching | grows (36k x 32 KiB = 1.2 GiB) | 126 | 127–133 | 1036 | 34 (0.85 util) |
| wkvm ring 16+1024 (no recall) | 33 + 49.5 MiB | 133 | 3.9 | 160 | 33.2 |
| **wkvm routed R=48 (recall)** | **163 + 49.5 MiB** | 191 | 5.4–5.6 | 229 | 36.2 |

- **Per turn: 24x** (130 s vs 5.5 s). The turn is 32 x 129-token prefill
  (~1 s) plus 127 graphed steps over 5,713 columns at 32 rows (~35 ms each;
  ring's 1,040 columns take ~27 ms).
- **Over 8 turns: 4.5x.** Turn 0 pays 58 s of routing on top of ring's
  133 s prefill (2,304 passes, ~25 ms each: the pass syncs the prefill
  pipeline once per row; batching the rows of a prefill forward into one
  sync is the obvious next cut).
- **On the README's 48-turn shape: 14x** (191 + 47 x 5.5 = 450 s vs 6,236 s);
  ring is 20x.

Below the wall (16 sessions x 13,824 tokens x 8 turns, everything fits
vLLM's pool): vLLM 3.65 s per turn, 47 s total; ring 3.1 s, 50 s; routed
3.9 s, 67 s (turn 0 40 s vs 22 s). At equal residency the routed engine is
now within 7% of vLLM per turn and pays 1.8x on the initial prefill.

## What this settles

1. **Fixed state with recall exists on the hybrid.** 163 MiB per session,
   unbounded context, single-needle recall 1.00 at 4k–32k and multi-needle
   0.75–1.00 through 16k, 24x per turn past the incumbent's capacity wall
   and 14x on a 48-turn session. The recurrent layers carry the gist, the
   guest bank carries what the gist cannot: the novel spans.
2. **The retention signal is the whole game**, and it is measurable. Mean-V
   farthest-point and surprisal both looked principled and both failed the
   probe; local novelty passed it 30/30. The probe, not the RULER average,
   is the tool for the next signals (instruction similarity, multi-scale
   novelty). Novelty is computed from value features at routing time, so
   generated tokens are scored like prompt tokens; only the surprisal
   option lacks a signal for them.
3. **What it does not do**: recall inside a haystack of near-identical
   facts, or predict which ordinary paragraph a later question needs. Those
   need either a query (retrieval at read time from a cold exact store) or
   more budget; both are outside "fixed state".
4. Remaining engine costs are known and mechanical: turn-0 routing sync per
   row (58 s of 191), prefill through eager HF modules (both engines' turn
   0), and the duplicate resident-row copy of the store for CUDA graphs
   (why R = 144 does not fit 32 sessions on 40 GB: 355 MiB x 64).

Caveats: single runs on an active multi-tenant box; vLLM at its best case
for this workload (prefix caching, 0.85 utilisation); approximate (routed)
vs exact semantics; the probe uses one needle template — the trace signals
should be re-checked on other needle families before the default changes.
