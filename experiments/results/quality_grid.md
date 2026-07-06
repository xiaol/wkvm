# Quality grid: full vs ring vs banked (gemma-4-E4B-it)

Depth-stratified synthetic recall; greedy, substring-scored, batch-1. Filler = cycled varied sentences (non-repetitive). t1/t2: mean of 3 seeds; t3: fraction of 6 items in 48-token output, 1 seed. ring = sink16+window1024; banked = +bank K=16/seg=256/reps=8 leader-select. full@32k uses prefill chunk 1024.

| ctx | task | depth | full | ring | banked |
|---|---|---|---|---|---|
| 8192 | t1-needle | 0.1 | 1.00 | 0.00 | 0.67 |
| 8192 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 |
| 8192 | t1-needle | 0.5 | 1.00 | 0.00 | 1.00 |
| 8192 | t1-needle | 0.7 | 1.00 | 0.00 | 0.67 |
| 8192 | t1-needle | 0.9 | 1.00 | 1.00 | 1.00 |
| 8192 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.00 |
| 8192 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.33 |
| 8192 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.33 |
| 8192 | t2-multikey | 0.7 | 1.00 | 0.00 | 1.00 |
| 8192 | t2-multikey | 0.9 | 1.00 | 1.00 | 1.00 |
| 8192 | t3-aggregate | 0.1 | 1.00 | 0.00 | 0.00 |
| 8192 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.00 |
| 8192 | t3-aggregate | 0.5 | 0.83 | 0.00 | 0.00 |
| 8192 | t3-aggregate | 0.7 | 1.00 | 0.17 | 0.50 |
| 8192 | t3-aggregate | 0.9 | 0.83 | 0.50 | 0.17 |
| 16384 | t1-needle | 0.1 | 1.00 | 0.00 | 1.00 |
| 16384 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 |
| 16384 | t1-needle | 0.5 | 1.00 | 0.00 | 0.67 |
| 16384 | t1-needle | 0.7 | 1.00 | 0.00 | 0.67 |
| 16384 | t1-needle | 0.9 | 1.00 | 0.00 | 0.67 |
| 16384 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.00 |
| 16384 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.00 |
| 16384 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.00 |
| 16384 | t2-multikey | 0.7 | 1.00 | 0.00 | 0.00 |
| 16384 | t2-multikey | 0.9 | 1.00 | 0.00 | 1.00 |
| 16384 | t3-aggregate | 0.1 | 0.83 | 0.00 | 0.00 |
| 16384 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.67 |
| 16384 | t3-aggregate | 0.5 | 1.00 | 0.00 | 0.00 |
| 16384 | t3-aggregate | 0.7 | 0.83 | 0.00 | 0.17 |
| 16384 | t3-aggregate | 0.9 | 0.83 | 0.00 | 0.00 |
| 32768 | t1-needle | 0.1 | 1.00 | 0.00 | 0.67 |
| 32768 | t1-needle | 0.3 | 1.00 | 0.00 | 1.00 |
| 32768 | t1-needle | 0.5 | 1.00 | 0.00 | 0.67 |
| 32768 | t1-needle | 0.7 | 1.00 | 0.00 | 0.33 |
| 32768 | t1-needle | 0.9 | 1.00 | 0.00 | 1.00 |
| 32768 | t2-multikey | 0.1 | 1.00 | 0.00 | 0.33 |
| 32768 | t2-multikey | 0.3 | 1.00 | 0.00 | 0.00 |
| 32768 | t2-multikey | 0.5 | 1.00 | 0.00 | 0.00 |
| 32768 | t2-multikey | 0.7 | 1.00 | 0.00 | 0.00 |
| 32768 | t2-multikey | 0.9 | 1.00 | 0.00 | 1.00 |
| 32768 | t3-aggregate | 0.1 | 0.83 | 0.00 | 0.17 |
| 32768 | t3-aggregate | 0.3 | 0.83 | 0.00 | 0.00 |
| 32768 | t3-aggregate | 0.5 | 0.83 | 0.00 | 0.50 |
| 32768 | t3-aggregate | 0.7 | 0.83 | 0.00 | 0.67 |
| 32768 | t3-aggregate | 0.9 | 0.83 | 0.00 | 0.33 |
