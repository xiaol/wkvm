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

## Update 2026-07-07: the training-free routed bank (SelectingMemory's idea, no learned parameters)

Replacing the bank's *temporal* segmentation with *content* routing — online spherical clustering into M slots, centroids EMA-updated, decided at the leader layer and replayed downward — changes the recall picture substantially (full tables regenerated in `quality_grid.md`):

| mode (evicted-depth recall) | t1 needle | t2 multi-key | t3 aggregate | overall (full = 0.95) |
|---|---|---|---|---|
| ring | 0.07 | 0.07 | 0.04 | 0.06 |
| banked (temporal segments) | 0.80 | 0.33 | 0.21 | 0.45 |
| routed-value, M=16 (matched budget) | 1.00 | 0.29 | 0.38 | 0.56 |
| **routed-value, M=64** | **1.00** | **0.51** | **0.86** | **0.79** |

Three findings:

1. **The routing feature must be RoPE-free.** Raw-key and residual-key clustering are *position-contaminated* — post-RoPE keys cluster by position, not content, and at M=16 they score 0.00, worse than the temporal bank they replace. **Value vectors carry no RoPE and win outright** (0.625 vs 0.25/0.00 in the sweep). Most KV-compression work clusters keys; on RoPE models that is quietly broken, and this grid demonstrates it.
2. **Allocation and capacity both contribute, and the comparison says how much.** At matched pseudo-slot budget (M=16 ≈ banked's 128 slots), content routing wins t1 (1.00 vs 0.80) and t3 (0.38 vs 0.21) but ties on t2 — the interference task also needs capacity (M=64, 576 max pseudo-slots, reaches 0.51 with 1.00 cells at 32k, and t3 0.86 ≈ the model's own full-attention ceiling).
3. **NLL is unmoved** (+0.10 nats @2–3k → +0.84 @15–16k, ~0.1 better than banked early, converging deep): routing is a **retrieval device, not language-model repair** — consistent with everything else training-free in this document.

Remaining gaps for the learned path (RWKV-MS sidecar) to close: t2's 0.51-vs-1.00 residue and the +0.3–1.4-nat NLL band. The tradeoff ledger for routed-value-m64: same flat footprint class (~4.5 extra MiB/slot for 576 vs 128 pseudo-slots on 4 layers), one online-clustering pass per evicted chunk, zero training.

## Update 2026-07-07 (2): t2 diagnosed by trace, then fixed — span-atomic routing

Before fixing t2's 0.51, we instrumented it (`experiments/t2_trace.py`, results in `t2_trace*.md`): per failed fact, which slots its tokens landed in, whether its answer reps survived, and the attention mass they won. Verdict — **92% of failures were binding scatter** (per-token routing tears the 5–6-token code across slots; name/answer co-location broken in 86% of all evicted facts), 8% sibling eviction, **0% readout** (surviving reps always won attention — but partial reps produced confident hallucinations like `AURORA-001`). Diagnose-then-fix beat guessing: the readout stage we might have "fixed" was never broken.

**The fix (`routed-span-m64`, still training-free):** route sentence-spans as indivisible units — assigned by their *most-novel token's value* (span means re-collided all eight template facts into one slot; the re-trace caught that in one iteration) — plus greedy farthest-point span retention with a near-duplicate floor so redundant filler can't consume slot budget.

| mode | t1 | t2 | t3 | overall (full = 0.95) |
|---|---|---|---|---|
| banked (temporal) | 0.80 | 0.33 | 0.21 | 0.45 |
| routed-value-m64 | 1.00 | 0.51 | 0.86 | 0.79 |
| **routed-span-m64** | **1.00** | **0.89** (0.93 @16k/32k) | 0.81 | **0.90** |

Honest notes: t3 dips 0.86→0.81 (sentence granularity slightly hurts scattered single-item lists); NLL unchanged (still a retrieval device); the span bank costs ~88 MiB/slot at 32k vs the ring's 36 — bounded (ceiling 64 slots × 145 tokens) but no longer tiny. The re-traced residue (all 5 remaining failures) is **evicted-by-sibling under slot budget** — genuinely similar facts overflowing one slot's 144 tokens — zero scatter, zero readout. That residue is exactly what a slot-paging tier (spill hot slots' full KV to pinned host, page in on demand) removes, and now it has a price tag: it buys the last 0.11 of t2 and nothing else.

## Method notes (for reuse)

Three modes share one harness; everything batch-1 (bf16 batch-shape flips excluded); synthetics generated from the tokenizer with varied natural-ish filler; NLL is teacher-forced with the cache evolving exactly as in generation, binned per 1k positions with per-doc std; the `NLL_CURVE_OK` marker is emitted only after the pre-eviction exactness gate passes (bins 0–2k within 1e-3). Ledger: `.keel/ledger.jsonl` (findings e8/e12/e16 record the filler-artifact discovery and grid/NLL provenance).
