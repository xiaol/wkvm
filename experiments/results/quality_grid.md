# Quality grid: full vs ring vs banked vs routed (gemma-4-E4B-it)

Depth-stratified synthetic recall; greedy, substring-scored, batch-1. Filler = cycled varied sentences (non-repetitive). t1/t2: mean of 3 seeds; t3: fraction of 6 items in 48-token output, 1 seed. ring = sink16+window1024; banked = temporal-segment bank K=16/seg=256/reps=8, leader-select; routed-* = training-free routed bank (Raven-style content routing: M persistent slots, cosine routing at the leader layer, EMA centroids, untouched slots never decay; suffix = routing feature + M). full@32k uses prefill chunk 1024.

| ctx | task | depth | full | ring | banked | routed-value-m16 | routed-value-m64 |
|---|---|---|---|---|---|---|---|
| 8192 | t1-needle | 0.1 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.5 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.7 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 8192 | t1-needle | 0.9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8192 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.33 | 0.00 | 0.00 |
| 8192 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.33 | 0.00 | 0.33 |
| 8192 | t2-multikey | 0.7 | 1.00 | 0.00 | 1.00 | 0.00 | 0.67 |
| 8192 | t2-multikey | 0.9 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8192 | t3-aggregate | 0.1 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 8192 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.00 | 0.67 | 0.83 |
| 8192 | t3-aggregate | 0.5 | 0.83 | 0.00 | 0.00 | 0.50 | 0.83 |
| 8192 | t3-aggregate | 0.7 | 1.00 | 0.17 | 0.50 | 0.17 | 1.00 |
| 8192 | t3-aggregate | 0.9 | 0.83 | 0.50 | 0.17 | 0.83 | 0.83 |
| 16384 | t1-needle | 0.1 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.5 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.7 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 16384 | t1-needle | 0.9 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 16384 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| 16384 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.00 | 0.33 | 0.33 |
| 16384 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.00 | 0.33 | 0.00 |
| 16384 | t2-multikey | 0.7 | 1.00 | 0.00 | 0.00 | 0.67 | 0.33 |
| 16384 | t2-multikey | 0.9 | 1.00 | 0.00 | 1.00 | 0.33 | 0.00 |
| 16384 | t3-aggregate | 0.1 | 0.83 | 0.00 | 0.00 | 0.00 | 0.83 |
| 16384 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.67 | 0.33 | 0.83 |
| 16384 | t3-aggregate | 0.5 | 1.00 | 0.00 | 0.00 | 0.67 | 1.00 |
| 16384 | t3-aggregate | 0.7 | 0.83 | 0.00 | 0.17 | 0.17 | 0.83 |
| 16384 | t3-aggregate | 0.9 | 0.83 | 0.00 | 0.00 | 0.33 | 0.83 |
| 32768 | t1-needle | 0.1 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.5 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.7 | 1.00 | 0.00 | 0.33 | 1.00 | 1.00 |
| 32768 | t1-needle | 0.9 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| 32768 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.33 | 0.33 | 0.67 |
| 32768 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.00 | 0.33 | 0.67 |
| 32768 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.00 | 0.33 | 1.00 |
| 32768 | t2-multikey | 0.7 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 32768 | t2-multikey | 0.9 | 1.00 | 0.00 | 1.00 | 0.67 | 1.00 |
| 32768 | t3-aggregate | 0.1 | 0.83 | 0.00 | 0.17 | 0.00 | 0.83 |
| 32768 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.00 | 0.17 | 0.83 |
| 32768 | t3-aggregate | 0.5 | 0.83 | 0.00 | 0.50 | 0.67 | 0.83 |
| 32768 | t3-aggregate | 0.7 | 0.83 | 0.00 | 0.67 | 0.67 | 0.83 |
| 32768 | t3-aggregate | 0.9 | 0.83 | 0.00 | 0.33 | 0.50 | 0.83 |

## Routed-variant sweep (pruning stage: t1/t2 @ ctx 8192, depths 0.1/0.5, 2 seeds)

Routing-feature x M sweep; banked as baseline. Key finding: only VALUE-based
routing (values carry no RoPE) beats temporal banking; raw-key and residual-key
routing are position-contaminated and, at M=16, lose the needle entirely.

| variant | t1@0.1 | t1@0.5 | t2@0.1 | t2@0.5 | mean |
|---|---|---|---|---|---|
| banked (baseline) | 0.50 | 1.00 | 0.00 | 0.00 | 0.375 |
| routed-key-m16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| routed-key-m64 | 0.00 | 1.00 | 0.00 | 0.00 | 0.250 |
| routed-resid-m16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 |
| routed-resid-m64 | 0.00 | 1.00 | 0.00 | 0.00 | 0.250 |
| routed-value-m16 | 1.00 | 1.00 | 0.00 | 0.00 | 0.500 |
| routed-value-m64 | 1.00 | 1.00 | 0.00 | 0.50 | 0.625 |

Finalists carried into the full grid above: routed-value-m16, routed-value-m64.

## Per-mode means by context (all tasks, all depths; 15 cells each)

| mode | @8192 | @16384 | @32768 | overall |
|---|---|---|---|---|
| full | 0.97 | 0.95 | 0.94 | 0.95 |
| ring | 0.18 | 0.00 | 0.00 | 0.06 |
| banked | 0.51 | 0.39 | 0.44 | 0.45 |
| routed-value-m16 | 0.54 | 0.54 | 0.58 | 0.56 |
| routed-value-m64 | 0.77 | 0.71 | 0.90 | 0.79 |

## Per-mode means by task (all ctx, all depths)

| mode | t1-needle | t2-multikey | t3-aggregate |
|---|---|---|---|
| full | 1.00 | 1.00 | 0.86 |
| ring | 0.07 | 0.07 | 0.04 |
| banked | 0.80 | 0.33 | 0.21 |
| routed-value-m16 | 1.00 | 0.29 | 0.38 |
| routed-value-m64 | 1.00 | 0.51 | 0.86 |

Takeaway: content routing on VALUE features (routed-value-m64, 576 pseudo-slots max)
recovers 0.79/0.95 of full-KV quality overall vs banked 0.45 and ring 0.06; routed-value-m16
(144 slots, same budget class as banked) still leads banked on t1/t3 but not t2.
