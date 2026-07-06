# t2 stage-attribution trace (routed-span-m64)

16 traced runs (8 cells x 2 seeds); 0 failed, 16 succeeded. Stages: scattered (fact name+answer tokens not co-located in one slot) -> evicted (no exact answer rep survives) -> lost-softmax (surviving reps get < 0.05 head-avg attention at the query step, mean over the 7 full-attention layers) -> decode-other.

## Stage attribution of QUERIED facts

| stage | failed runs | % of failed | succeeded runs |
|---|---|---|---|
| scattered | 0 | 0% | 0 |
| evicted | 0 | 0% | 0 |
| lost-softmax | 0 | 0% | 4 |
| decode-other | 0 | 0% | 12 |
| in-ring | 0 | 0% | 0 |

## All evicted facts across traced runs (routing/retention stats)

- facts fully evicted from ring: 126
- NOT co-located (name/answer split across slots): 0 (0%)
- answer tokens themselves split over >1 slot: 0 (0%)
- >=1 exact answer rep survives at query: 100 (79%)
- evicted by sibling-fact reps: 26; by filler reps: 0

## Per-run detail (queried fact)

| ctx | depth | seed | ok | stage | ans slots | name slots | ans reps | evictor | mean fact-mass | mean ring-mass |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 0.1 | 0 | True | decode-other | [4] | [4] | 7 | - | 0.063 | 0.337 |
| 8192 | 0.1 | 1 | True | decode-other | [2] | [2] | 6 | - | 0.050 | 0.343 |
| 16384 | 0.3 | 0 | True | decode-other | [60] | [60] | 6 | - | 0.051 | 0.313 |
| 16384 | 0.3 | 1 | True | lost-softmax | [58] | [58] | 7 | - | 0.050 | 0.311 |
| 16384 | 0.5 | 0 | True | lost-softmax | [59] | [59] | 7 | - | 0.050 | 0.323 |
| 16384 | 0.5 | 1 | True | lost-softmax | [58] | [58] | 6 | - | 0.032 | 0.324 |
| 16384 | 0.7 | 0 | True | decode-other | [58] | [58] | 6 | - | 0.057 | 0.328 |
| 16384 | 0.7 | 1 | True | decode-other | [58] | [58] | 7 | - | 0.056 | 0.348 |
| 16384 | 0.9 | 0 | True | decode-other | [59] | [59] | 6 | - | 0.053 | 0.319 |
| 16384 | 0.9 | 1 | True | decode-other | [58] | [58] | 6 | - | 0.054 | 0.311 |
| 32768 | 0.1 | 0 | True | decode-other | [11] | [11] | 6 | - | 0.059 | 0.326 |
| 32768 | 0.1 | 1 | True | decode-other | [23] | [23] | 7 | - | 0.062 | 0.335 |
| 32768 | 0.7 | 0 | True | lost-softmax | [15] | [15] | 6 | - | 0.041 | 0.312 |
| 32768 | 0.7 | 1 | True | decode-other | [5] | [5] | 7 | - | 0.056 | 0.320 |
| 32768 | 0.9 | 0 | True | decode-other | [26] | [26] | 6 | - | 0.054 | 0.320 |
| 32768 | 0.9 | 1 | True | decode-other | [20] | [20] | 7 | - | 0.053 | 0.307 |

STAGE_ATTRIBUTION_OK
