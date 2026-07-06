# Does Recurrent Mode Hurt the Transformer? Yes — Here Is Exactly Where and How Much

*gemma-4-E4B-it, RTX 4090, 2026-07-07. Three-condition protocol (full-KV / ring / ring+bank), depth-stratified synthetic grid + position-resolved NLL on natural documents. Raw tables: `experiments/results/quality_grid.md` (45 cells), `quality_nll.md`. Harness: `experiments/quality_eval.py`. This evaluation was designed to correct for the two traps we had already identified (repetitive-filler flattery, bf16 batch-shape noise) — and the first one turned out to matter a lot.*

## The headline answers

1. **In-window: zero harm, proven.** NLL bins 0–2k are *exactly* equal across all three modes (delta 0.0000); divergence starts one chunk after the ring window. Nothing inside sink+ring is approximated.
2. **Past the window: real, growing harm.** On natural text (6×16k-token docs: LaTeX, markdown, code), ring mode costs **+0.32 nats at 2–3k rising to +0.8–1.4 nats** deep in the document. The bank buys back only **~0.05–0.13 nats** of that. Recurrent mode is not a general-quality repair.
3. **The bank is a targeted-recall device, and task breadth kills it.** Over evicted depths: single planted fact **0.72** (ring: 0.00, full: 1.00); one-of-eight keyed facts **0.26** — collapsing to 0.00 at 16k/32k mid-depths, where eight competing facts fight for 7 representative slots per segment through the merge chain; scattered-list aggregation **0.21**, with one cell where banked scores *below* plain ring.
4. **Our own PoC-2 "perfect recall to 32k" does not survive honest filler.** With non-repetitive text instead of a repeated paragraph, 18/18 becomes the numbers above. The PoC-2 NLL inversion (banked "beating" full) also vanishes on natural documents — both confirmed as filler artifacts. Full-KV is strictly better everywhere past the window, as physics demands.

## The grid, compressed (mean recall on evicted-depth cells)

| task | full-KV | ring | ring+bank |
|---|---|---|---|
| t1: single needle | 1.00 | 0.00 | **0.72** |
| t2: multi-key (1 of 8) | 1.00 | 0.00 | **0.26** |
| t3: aggregation (6 scattered items) | 0.83¹ | 0.00 | **0.21** |

¹ 0.83 is the model's own ceiling on this task even with full attention.

Depth and context matter: t1 recovery is strong at 8k–16k and decays by 32k (0.33 at depth 0.7); t2 survives only when the queried fact is recent (0.9 depth); t3 recovers modestly only at 32k where segments are coarser. Full-KV ran at all context lengths including 32k (chunk 1024) — every comparison has its upper bound measured, not assumed.

## What this means for deploying recurrent mode

- **Safe by construction**: any workload whose dependencies fit sink+ring (recency-dominated chat, streaming, tool loops with recent state). Zero measured cost.
- **Usable with eyes open**: sparse-salient-fact recall over long sessions (the 0.72 regime) — persona facts, key decisions — especially at 8k–16k effective histories.
- **Not fit (training-free)**: dense multi-fact recall, aggregation/summarization over evicted spans, and anything sensitive to the +0.5–1 nat LM-quality drop past the window.
- **The path for the rest is learned, not clever**: these numbers are the training-free ceiling. The RWKV-MS sidecar (trained memory adapters, tau2 0.70 vs 0.20 base) is the credible route to closing the t2/t3/NLL gaps, and this harness is now the regression suite for it — same three conditions plus a fourth: `sidecar`.
- The honest sales pitch stays what the physics supports: **10× session capacity at zero in-window cost and precisely-mapped out-of-window cost** — a trade the user chooses per workload with this document as the map, not a free lunch.

## Method notes (for reuse)

Three modes share one harness; everything batch-1 (bf16 batch-shape flips excluded); synthetics generated from the tokenizer with varied natural-ish filler; NLL is teacher-forced with the cache evolving exactly as in generation, binned per 1k positions with per-doc std; the `NLL_CURVE_OK` marker is emitted only after the pre-eviction exactness gate passes (bins 0–2k within 1e-3). Ledger: `.keel/ledger.jsonl` (findings e8/e12/e16 record the filler-artifact discovery and grid/NLL provenance).
