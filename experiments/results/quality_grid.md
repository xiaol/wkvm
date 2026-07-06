# Quality grid: full vs ring vs banked vs routed (gemma-4-E4B-it)

Depth-stratified synthetic recall; greedy, substring-scored, batch-1. Filler = cycled varied sentences (non-repetitive). t1/t2: mean of 3 seeds; t3: fraction of 6 items in 48-token output, 1 seed. ring = sink16+window1024; banked = temporal-segment bank K=16/seg=256/reps=8, leader-select; routed-* = training-free routed bank (Raven-style content routing: M persistent slots, cosine routing at the leader layer, EMA centroids, untouched slots never decay; suffix = routing feature + M). full@32k uses prefill chunk 1024.

| ctx | task | depth | full | ring | banked | routed-value-m64 | routed-span-m64 |
|---|---|---|---|---|---|---|---|
| 8192 | t1-needle | 0.1 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.5 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.7 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 8192 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.33 | 0.00 | 0.67 |
| 8192 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.33 | 0.33 | 0.67 |
| 8192 | t2-multikey | 0.7 | 1.00 | 0.00 | 1.00 | 0.67 | 1.00 |
| 8192 | t2-multikey | 0.9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t3-aggregate | 0.1 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 8192 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.00 | 0.83 | 0.83 |
| 8192 | t3-aggregate | 0.5 | 0.83 | 0.00 | 0.00 | 0.83 | 0.83 |
| 8192 | t3-aggregate | 0.7 | 1.00 | 0.17 | 0.50 | 1.00 | 0.83 |
| 8192 | t3-aggregate | 0.9 | 0.83 | 0.50 | 0.17 | 0.83 | 0.83 |
| 16384 | t1-needle | 0.1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.5 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.7 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.9 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 16384 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.00 | 0.67 | 0.67 |
| 16384 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.00 | 0.33 | 1.00 |
| 16384 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 16384 | t2-multikey | 0.7 | 1.00 | 0.00 | 0.00 | 0.33 | 1.00 |
| 16384 | t2-multikey | 0.9 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| 16384 | t3-aggregate | 0.1 | 0.83 | 0.00 | 0.00 | 0.83 | 0.83 |
| 16384 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.67 | 0.83 | 0.83 |
| 16384 | t3-aggregate | 0.5 | 1.00 | 0.00 | 0.00 | 1.00 | 0.83 |
| 16384 | t3-aggregate | 0.7 | 0.83 | 0.00 | 0.17 | 0.83 | 0.67 |
| 16384 | t3-aggregate | 0.9 | 0.83 | 0.00 | 0.00 | 0.83 | 0.83 |
| 32768 | t1-needle | 0.1 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.5 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.7 | 1.00 | 0.00 | 0.33 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.9 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32768 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.33 | 0.67 | 1.00 |
| 32768 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| 32768 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.00 | 1.00 | 0.67 |
| 32768 | t2-multikey | 0.7 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| 32768 | t2-multikey | 0.9 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32768 | t3-aggregate | 0.1 | 0.83 | 0.00 | 0.17 | 0.83 | 0.67 |
| 32768 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.00 | 0.83 | 0.83 |
| 32768 | t3-aggregate | 0.5 | 0.83 | 0.00 | 0.50 | 0.83 | 0.67 |
| 32768 | t3-aggregate | 0.7 | 0.83 | 0.00 | 0.67 | 0.83 | 0.83 |
| 32768 | t3-aggregate | 0.9 | 0.83 | 0.00 | 0.33 | 0.83 | 0.83 |

## Routed-variant sweep (pruning stage: t1/t2 @ 8192, depths 0.1/0.5, 2 seeds)

Only VALUE-feature routing beats temporal banking (values carry no RoPE; raw/residual
key routing is position-contaminated and at M=16 loses the needle entirely):
banked 0.375 | routed-key-m16 0.000 | routed-key-m64 0.250 | routed-resid-m16 0.000
| routed-resid-m64 0.250 | routed-value-m16 0.500 | routed-value-m64 0.625.

## routed-span (p9): span-atomic routing + diversity retention

routed-span-m64 = routed-value-m64 + (f1) spans split at sentence punctuation, routed
atomically by their most-novel token's VALUE vector; (f2) within-slot greedy
farthest-point span retention with a near-duplicate floor (0.10) under a 144-token
slot budget. Fixes the p8 route-scatter failure (92% of t2 failures) and the
follow-on sibling-eviction failure; bank grows to ~3.3k tokens / ~88 MiB at 32k
(vs ring 36 MiB) but stays context-bounded (3.1k @16k -> 3.3k @32k).

## Per-mode means by context (all tasks/depths; 15 cells each)

| mode | @8192 | @16384 | @32768 | overall |
|---|---|---|---|---|
| full | 0.97 | 0.95 | 0.94 | 0.95 |
| ring | 0.18 | 0.00 | 0.00 | 0.06 |
| banked | 0.51 | 0.39 | 0.44 | 0.45 |
| routed-value-m64 | 0.77 | 0.71 | 0.90 | 0.79 |
| routed-span-m64 | 0.89 | 0.91 | 0.90 | 0.90 |

## Per-mode means by task (all ctx/depths)

| mode | t1-needle | t2-multikey | t3-aggregate |
|---|---|---|---|
| full | 1.00 | 1.00 | 0.86 |
| ring | 0.07 | 0.07 | 0.04 |
| banked | 0.80 | 0.33 | 0.21 |
| routed-value-m64 | 1.00 | 0.51 | 0.86 |
| routed-span-m64 | 1.00 | 0.89 | 0.81 |

Residual t2 failures of routed-span-m64 (5/45 traced seeds, t2_trace_span_final.md):
ALL evicted-by-sibling - when the 8 template facts plus diverse filler overflow one
slot's 144-token budget, farthest-point drops the mutually-most-similar fact span.
That residue is the price of fixed slot budgets and is what a slot-paging tier
(spill hot slots to host memory) would remove.
