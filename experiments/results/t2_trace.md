# t2 stage-attribution trace (routed-value-m64)

16 traced runs (8 cells x 2 seeds); 13 failed, 3 succeeded. Stages: scattered (fact name+answer tokens not co-located in one slot) -> evicted (no exact answer rep survives) -> lost-softmax (surviving reps get < 0.05 head-avg attention at the query step, mean over the 7 full-attention layers) -> decode-other.

## Stage attribution of QUERIED facts

| stage | failed runs | % of failed | succeeded runs |
|---|---|---|---|
| scattered | 12 | 92% | 3 |
| evicted | 1 | 8% | 0 |
| lost-softmax | 0 | 0% | 0 |
| decode-other | 0 | 0% | 0 |
| in-ring | 0 | 0% | 0 |

## All evicted facts across traced runs (routing/retention stats)

- facts fully evicted from ring: 120
- NOT co-located (name/answer split across slots): 103 (86%)
- answer tokens themselves split over >1 slot: 101 (84%)
- >=1 exact answer rep survives at query: 102 (85%)
- evicted by sibling-fact reps: 18; by filler reps: 0

## Per-run detail (queried fact)

| ctx | depth | seed | ok | stage | ans slots | name slots | ans reps | evictor | mean fact-mass | mean ring-mass |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 0.1 | 0 | False | scattered | [2, 15, 55] | [40] | 1 | - | 0.027 | 0.417 |
| 8192 | 0.1 | 1 | False | scattered | [9, 22, 53] | [55] | 1 | - | 0.021 | 0.423 |
| 8192 | 0.3 | 0 | False | scattered | [2, 14, 60] | [27] | 2 | - | 0.021 | 0.408 |
| 8192 | 0.3 | 1 | False | scattered | [3, 34, 51] | [51] | 4 | - | 0.024 | 0.402 |
| 8192 | 0.5 | 0 | True | scattered | [3, 30, 51, 54] | [30, 51] | 2 | - | 0.020 | 0.392 |
| 8192 | 0.5 | 1 | False | scattered | [6, 20, 28, 37] | [28, 37] | 3 | - | 0.034 | 0.389 |
| 8192 | 0.7 | 0 | False | scattered | [5, 11, 50] | [37] | 3 | - | 0.042 | 0.384 |
| 8192 | 0.7 | 1 | True | scattered | [4, 14] | [4] | 2 | - | 0.017 | 0.407 |
| 16384 | 0.3 | 0 | False | scattered | [13, 23] | [23] | 3 | - | 0.028 | 0.376 |
| 16384 | 0.3 | 1 | False | scattered | [6, 19] | [19] | 3 | - | 0.034 | 0.382 |
| 16384 | 0.5 | 0 | True | scattered | [2, 50] | [2, 15] | 3 | - | 0.036 | 0.388 |
| 16384 | 0.5 | 1 | False | scattered | [4, 27] | [21, 27] | 2 | - | 0.013 | 0.390 |
| 16384 | 0.7 | 0 | False | scattered | [3, 24] | [24] | 2 | - | 0.008 | 0.401 |
| 16384 | 0.7 | 1 | False | scattered | [30, 32] | [30] | 1 | - | 0.020 | 0.393 |
| 16384 | 0.9 | 0 | False | scattered | [3, 58] | [3] | 0 | sibling | - | - |
| 16384 | 0.9 | 1 | False | evicted | [28] | [28] | 0 | sibling | - | - |

STAGE_ATTRIBUTION_OK
