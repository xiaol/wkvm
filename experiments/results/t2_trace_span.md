# t2 stage-attribution trace (routed-span-m64)

16 traced runs (8 cells x 2 seeds); 13 failed, 3 succeeded. Stages: scattered (fact name+answer tokens not co-located in one slot) -> evicted (no exact answer rep survives) -> lost-softmax (surviving reps get < 0.05 head-avg attention at the query step, mean over the 7 full-attention layers) -> decode-other.

## Stage attribution of QUERIED facts

| stage | failed runs | % of failed | succeeded runs |
|---|---|---|---|
| scattered | 0 | 0% | 0 |
| evicted | 13 | 100% | 0 |
| lost-softmax | 0 | 0% | 0 |
| decode-other | 0 | 0% | 3 |
| in-ring | 0 | 0% | 0 |

## All evicted facts across traced runs (routing/retention stats)

- facts fully evicted from ring: 126
- NOT co-located (name/answer split across slots): 0 (0%)
- answer tokens themselves split over >1 slot: 0 (0%)
- >=1 exact answer rep survives at query: 36 (29%)
- evicted by sibling-fact reps: 85; by filler reps: 5

## Per-run detail (queried fact)

| ctx | depth | seed | ok | stage | ans slots | name slots | ans reps | evictor | mean fact-mass | mean ring-mass |
|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 0.1 | 0 | False | evicted | [2] | [2] | 0 | sibling | - | - |
| 8192 | 0.1 | 1 | False | evicted | [2] | [2] | 0 | sibling | - | - |
| 16384 | 0.3 | 0 | False | evicted | [63] | [63] | 0 | filler | - | - |
| 16384 | 0.3 | 1 | False | evicted | [58] | [58] | 0 | sibling | - | - |
| 16384 | 0.5 | 0 | True | decode-other | [59] | [59] | 7 | - | 0.057 | 0.373 |
| 16384 | 0.5 | 1 | True | decode-other | [58] | [58] | 6 | - | 0.062 | 0.349 |
| 16384 | 0.7 | 0 | False | evicted | [58] | [58] | 0 | sibling | - | - |
| 16384 | 0.7 | 1 | False | evicted | [58] | [58] | 0 | sibling | - | - |
| 16384 | 0.9 | 0 | False | evicted | [59] | [59] | 0 | sibling | - | - |
| 16384 | 0.9 | 1 | False | evicted | [58] | [58] | 0 | sibling | - | - |
| 32768 | 0.1 | 0 | False | evicted | [44] | [44] | 0 | sibling | - | - |
| 32768 | 0.1 | 1 | False | evicted | [31] | [31] | 0 | filler | - | - |
| 32768 | 0.7 | 0 | False | evicted | [51] | [51] | 0 | sibling | - | - |
| 32768 | 0.7 | 1 | False | evicted | [57] | [57] | 0 | sibling | - | - |
| 32768 | 0.9 | 0 | True | decode-other | [9] | [9] | 6 | - | 0.063 | 0.336 |
| 32768 | 0.9 | 1 | False | evicted | [10] | [10] | 0 | filler | - | - |

STAGE_ATTRIBUTION_OK
