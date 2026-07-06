# t2 stage-attribution trace (routed-span-m64)

15 traced runs (5 cells x 3 seeds); 5 failed, 10 succeeded. Stages: scattered (fact name+answer tokens not co-located in one slot) -> evicted (no exact answer rep survives) -> lost-softmax (surviving reps get < 0.05 head-avg attention at the query step, mean over the 7 full-attention layers) -> decode-other.

## Stage attribution of QUERIED facts

| stage | failed runs | % of failed | succeeded runs |
|---|---|---|---|
| scattered | 0 | 0% | 0 |
| evicted | 5 | 100% | 0 |
| lost-softmax | 0 | 0% | 2 |
| decode-other | 0 | 0% | 8 |
| in-ring | 0 | 0% | 0 |

## All evicted facts across traced runs (routing/retention stats)

- facts fully evicted from ring: 111
- NOT co-located (name/answer split across slots): 0 (0%)
- answer tokens themselves split over >1 slot: 0 (0%)
- >=1 exact answer rep survives at query: 79 (71%)
- evicted by sibling-fact reps: 32; by filler reps: 0

## Per-run detail (queried fact)

| ctx | depth | seed | ok | stage | ans slots | name slots | ans reps | evictor | mean fact-mass | mean ring-mass |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 0.1 | 0 | True | decode-other | [4] | [4] | 7 | - | 0.063 | 0.337 |
| 8192 | 0.1 | 1 | True | decode-other | [2] | [2] | 6 | - | 0.050 | 0.343 |
| 8192 | 0.1 | 2 | False | evicted | [3] | [3] | 0 | sibling | - | - |
| 8192 | 0.3 | 0 | True | lost-softmax | [6] | [6] | 6 | - | 0.048 | 0.325 |
| 8192 | 0.3 | 1 | False | evicted | [3] | [3] | 0 | sibling | - | - |
| 8192 | 0.3 | 2 | True | decode-other | [9] | [9] | 7 | - | 0.051 | 0.336 |
| 8192 | 0.5 | 0 | False | evicted | [3] | [3] | 0 | sibling | - | - |
| 8192 | 0.5 | 1 | True | decode-other | [3] | [3] | 7 | - | 0.053 | 0.338 |
| 8192 | 0.5 | 2 | True | lost-softmax | [3] | [3] | 7 | - | 0.045 | 0.334 |
| 16384 | 0.1 | 0 | False | evicted | [58] | [58] | 0 | sibling | - | - |
| 16384 | 0.1 | 1 | True | decode-other | [58] | [58] | 6 | - | 0.056 | 0.328 |
| 16384 | 0.1 | 2 | True | decode-other | [58] | [58] | 7 | - | 0.065 | 0.333 |
| 32768 | 0.5 | 0 | True | decode-other | [2] | [2] | 6 | - | 0.051 | 0.307 |
| 32768 | 0.5 | 1 | False | evicted | [11] | [11] | 0 | sibling | - | - |
| 32768 | 0.5 | 2 | True | decode-other | [19] | [19] | 6 | - | 0.057 | 0.336 |

STAGE_ATTRIBUTION_OK
