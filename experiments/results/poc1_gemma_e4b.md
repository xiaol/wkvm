# PoC-1: recurrent-mode (sink+ring) serving for gemma-4-E4B-it

- **Hardware**: 1x RTX 4090 24 GB (~2.0–2.4 GB used by an unrelated process throughout).
- **Model**: google/gemma-4-E4B-it (local checkpoint, text tower only via
  `Gemma4ForCausalLM` + `key_mapping`; 13.90 GiB weights resident, bf16, eager attention).
- **Stack**: transformers 5.9.0, torch 2.11.0+cu130 (HRM-Text venv), `HF_HUB_OFFLINE=1`.
- **Script**: `experiments/gemma_recurrent_poc.py`.
- **Config facts** (derived at runtime from config, not hardcoded): 42 layers, last 18 share KV;
  of the 24 KV-owning layers, the 4 full-attention layers **[5, 11, 17, 23]** are the only
  growing-KV layers (sliding layers are already bounded at window 512 by stock
  `DynamicSlidingWindowLayer`). Ring mode replaces exactly those 4 cache layers with a
  sink+ring layer (**sink=16, window=1024**); everything else is stock. The shared
  full-attention tail layers (29/35/41) reuse layer 23's post-update KV via
  `shared_kv_states`, so they are bounded automatically.

## Bench (chunked prefill 2048, greedy decode 64, warmed clocks)

`HF_HUB_OFFLINE=1 .../python experiments/gemma_recurrent_poc.py bench`

| mode | ctx | cache MiB | peak GiB | peak-weights GiB | decode tok/s | NLL(last128) | prefill s |
|---|---|---|---|---|---|---|---|
| full | 900 | 35.0 | 14.27 | 0.36 | 48.55 | 0.5490 | 0.1 |
| full | 2048 | 53.0 | 14.48 | 0.47 | 48.13 | 0.3108 | 0.2 |
| full | 8192 | 149.0 | 15.60 | 1.57 | 43.05 | 0.2833 | 1.0 |
| full | 16384 | 277.0 | 17.24 | 3.12 | 38.45 |  | 2.2 |
| full | 32768 | OOM | OOM | OOM | OOM | OOM | OOM |
| ring | 900 | 35.0 | 14.39 | 0.00 | 48.17 | 0.5490 | 0.1 |
| ring | 2048 | 36.2 | 14.47 | 0.47 | 47.93 | 0.3064 | 0.2 |
| ring | 8192 | 36.2 | 14.74 | 0.73 | 48.74 | 0.3090 | 0.9 |
| ring | 16384 | 36.2 | 14.74 | 0.73 | 48.76 |  | 1.8 |
| ring | 32768 | 36.2 | 14.68 | 0.73 | 47.96 |  | 3.7 |

- **Full KV grows ~linearly** (35 → 277 MiB by 16k) and the 32k prefill **OOMs at chunk 2048**
  (peaked 18.7 GiB against the ~21.6 GiB available). With `--chunk 1024` full mode does finish
  32k: cache **533 MiB**, peak 17.58 GiB, decode **28.11 tok/s** — the linear-growth line
  continues (35→53→149→277→533 MiB, i.e. ~16 KB/token for the 4 growing layers:
  2 KV-heads x 512 head_dim x K+V x bf16 x 4 layers).
- **Ring cache is flat at 36.2 MiB** from 2k through 32k, and **ring decode is flat**
  (47.9–48.8 tok/s at every context length), while full-mode decode degrades
  48 → 43 → 38 → 28 tok/s (eager attention reads the whole growing KV each step).
- Peak-VRAM delta in ring mode is bounded by prefill activations (~0.73 GiB at chunk 2048),
  not by context length.

## Perplexity sanity (mean NLL of last 128 prompt tokens, teacher-forced)

| ctx | full | ring | note |
|---|---|---|---|
| 900 | 0.5490 | 0.5490 | **identical** — nothing evicted (900 < 16+1024); ring == full is the correctness check |
| 2048 | 0.3108 | 0.3064 | eviction active; on this repetitive synthetic filler the loss doesn't move |
| 8192 | 0.2833 | 0.3090 | +0.026 nats drift — the expected lossiness once 7k tokens have been evicted |

(The filler is a repeated paragraph, so NLL is a weak lossiness probe here — the needle
test below is the sharp one.)

## Needle recall ("The secret code is BLUE-742." planted at ~token 200)

`... gemma_recurrent_poc.py needle --ctx 8192` (greedy, 32 tokens, first line shown):

```
[needle ctx=  900 full] recalled=True  out: 'BLUE-742'
[needle ctx=  900 ring] recalled=True  out: 'BLUE-742'
[needle ctx= 8192 full] recalled=True  out: 'BLUE-742'
[needle ctx= 8192 ring] recalled=False out: 'There is no secret code mentioned in the text provided.'
```

- ctx=900 (< sink+window): ring recalls — the ring layer itself does not break attention.
- ctx=8192: full recalls, ring does not (needle was evicted; only sink[0:16] + last 1024
  tokens remain). **This is precisely the gap the PoC-2 state bank is meant to close.**

## Batched ring decode

`... gemma_recurrent_poc.py batch` — B=8 distinct chat prompts, left-padded, batched greedy 64:

- **352.3 tok/s aggregate** (8 x 64 tokens), peak VRAM **13.96 GiB** (weights are 13.90 GiB),
  cache 35.9 MiB total for all 8 sequences. Outputs coherent (e.g. "The capital of France is
  **Paris**.", "12 times 12 is **144**.").

## What this shows / what it doesn't

**Shows:**
1. On a real frontier-ish open checkpoint, only 4 of 42 layers own growing KV; capping them
   with a sink+ring layer that plugs into the stock transformers 5.9 cache/mask machinery
   (no modeling-code changes, ~60 lines for the cache layer) yields **constant memory and
   constant decode latency in context length**, through 32k with no OOM, while stock full KV
   grows linearly and falls over at 32k under the same budget.
2. The ring is *exact* until eviction starts (identical NLL and needle recall at 900 tokens),
   so the serving-mode plumbing is correct; the only quality change comes from eviction itself.
3. Per-request state is a fixed ~36 MiB object — the arena/slot allocation story
   (fixed-size admission, uniform batched decode; 8 concurrent sequences decode at
   ~7.3x the single-stream rate) carries over to transformers, as claimed in
   docs/RECURRENT_MODE.md §4.

**Doesn't show (yet):**
1. **No state bank.** Evicted context is simply gone — needle recall at 8k fails by design.
   PoC-2 adds the segmented RWKV-7 state bank + readout injection to close that gap.
2. No CUDA graphs, paged arena, FA3, or real serving loop — this is eager-mode HF
   transformers; absolute tok/s numbers are floors, only the *flatness* is the result.
3. Keys are cached post-RoPE (StreamingLLM-style eviction). Fine for sink+ring; a state
   bank that *re-reads* evicted KV must handle position aliasing explicitly.
4. The NLL drift at 8k is measured on repetitive synthetic filler; real long-document
   perplexity-vs-position curves (RECURRENT_MODE.md §5 eval plan) remain to be run.

## Concurrency (virtual sessions)

`... gemma_recurrent_poc.py concurrency` — each virtual session has **4096 tokens of
context** (16384 for the long-ctx probes); ONE session is prefilled and its cache tensors
are replicated across the batch dim to B real copies (`batch_repeat_interleave` — honest
memory cost, throwaway quality; each row gets a distinct first token so decode paths
diverge). Batched greedy decode of 128 tokens/session, eager attention. Budget: 21.07 GiB
torch-usable on the shared GPU (~2 GiB other process); **green = completed with >= 1 GiB
headroom on peak reserved**; B_max = largest green B.

### ring @ 4096 ctx/session — B_max = 64 (B=96 runs but breaks the 1 GiB headroom; B=112 hard-OOMs)

| B | cache MiB total | per-slot MiB | peak alloc GiB | peak reserved GiB | agg tok/s | per-stream tok/s | status |
|---|---|---|---|---|---|---|---|
| 8 | 290 | 36.2 | 14.67 | 15.36 | 310.1 | 38.76 | green |
| 16 | 579 | 36.2 | 14.80 | 15.39 | 503.6 | 31.48 | green |
| 32 | 1159 | 36.2 | 15.70 | 16.10 | 669.1 | 20.91 | green |
| 64 | 2318 | 36.2 | 17.49 | 18.24 | 835.4 | 13.05 | green |
| 96 | 3476 | 36.2 | 19.28 | 20.44 | 869.5 | 9.06 | over-budget |
| 112 | - | - | - | 20.63 | - | - | OOM |
| 128 | - | - | - | 20.65 | - | - | OOM |

### full @ 4096 ctx/session — B_max = 8

| B | cache MiB total | per-slot MiB | peak alloc GiB | peak reserved GiB | agg tok/s | per-stream tok/s | status |
|---|---|---|---|---|---|---|---|
| 8 | 688 | 86.0 | 18.44 | 19.89 | 208.3 | 26.04 | green |
| 16 | 1375 | 86.0 | 16.30 | 20.55 | 317.3 | 19.83 | over-budget |
| 32 | - | - | - | 20.63 | - | - | OOM |

### long-context probes (ctx/session 16384)

| mode | B | cache MiB total | per-slot MiB | peak reserved GiB | agg tok/s | per-stream tok/s | status |
|---|---|---|---|---|---|---|---|
| full | 8 | 2224 | 278.0 | 20.65 | 109.2 | 13.65 | over-budget (B_max = 0) |
| full | 16 | - | - | 20.26 | - | - | OOM |
| ring | 64 | 2318 | 36.2 | 19.51 | 782.1 | 12.22 | green (matches ring@4096: same 36.2 MiB/slot, 782 vs 835 tok/s) |

**Read:** Ring admits **8x more sessions than full KV at 4k context (B_max 64 vs 8), and the
gap is unbounded in context length — at 16k full KV cannot green-light even B=8 (B_max 0),
while ring's B=64 numbers are unchanged from 4k (same 36.2 MiB/slot, ~780 vs ~835 tok/s),
because a ring slot is a fixed 36.2 MiB object versus full KV's 86 MiB at 4k / 278 MiB at
16k and growing.** Aggregate throughput scales sub-linearly and saturates around B=64
(~835 tok/s; B=96 adds only +4%): eager-attention compute + per-step Python overhead becomes
the bottleneck before memory does — a serving-stack finding (CUDA graphs / fused attention
are the fix), not a property of the ring cache. Ring's memory ceiling here is dominated by
the 20 stock sliding-window layers plus eager repeat_kv transients, so B_max ~ 64–96 on a
shared 24 GB card; the RECURRENT_MODE.md §4 estimate of ~90 concurrent @ W=1k is consistent
with the measured per-slot size.

## PoC-2: state bank (banked mode)

**Goal**: close the 8k needle gap at constant footprint by folding evicted KV into a
capacity-bounded multi-state bank on the 4 ring layers. **Method chosen**: pseudo-KV
readout (option (a) of the spec) — bank states live as ordinary KV slots inside the cache
(`[sink | bank | pending | ring]`, chronological), entering stock softmax attention through
the same imputed-position causal mask as the ring. No modeling-code changes.

- **Folding**: every `seg` evicted tokens become one segment state of `reps=8` slots:
  slot 0 = segment summary (mean key, mean value); slots 1..7 = the segment's top
  *representatives* — exact post-RoPE (K,V) of the tokens whose keys are most novel vs the
  segment mean (cosine, head-averaged). Fixed boundaries (simplest policy from the design
  doc); post-RoPE keys mean kept entries stay positionally valid, StreamingLLM-style.
- **Capacity**: when the bank exceeds K_STATES segments, the two most similar adjacent
  segments merge — count-weighted mean summary, representatives re-selected by novelty
  against the merged mean (DLA-style capacity-bounded adjacent merging). Footprint is
  therefore flat at any context length.
- **The finding that made it work — cross-layer selection sharing**: per-layer key novelty
  picks the needle tokens *perfectly* at shallow banked layers (5, 11 select exactly
  ` BLUE`,`-`,`7`,`4`,`2`) and *fails* at deep ones (17, 23 pick filler tokens; value-novelty
  and NoPE-dims-only scoring fail there too — the needle signature is smeared by depth).
  Since layer 23 feeds the 3 KV-shared full-attention layers (29/35/41), deep-bank quality
  is decisive: with per-layer selection the 8k needle still failed. Fix: the shallowest
  banked layer is the *leader*; its representative indices and merge decisions are logged
  and replayed by the deeper banked layers (identical eviction schedule, so the op log
  aligns). This mirrors the RWKV-MS finding that the shallow band carries memory.
  `--select per-layer` keeps the failing ablation available.

### Acceptance

| check | result |
|---|---|
| (1) needle @8192 banked | **recalled** (`'BLUE-742'`; full recalls, plain ring fails) |
| (2) footprint flat vs ctx | 36.9–44.4 MiB at 8k/16k/32k (vs ring 36.2 flat, full 149→533 MiB growing) |
| (3) ring-exactness below window | NLL(last128)@900 = **0.549035 for full, ring, banked — identical**; needle@900 banked recalled |
| (4) deeper eviction | recall survives at 16k AND 32k in **all 6 configs**, incl. K=4 (needle reps survive the merge chain) |
| (5) NLL@8192 | banked **0.2572** <= ring 0.3090 (and < full 0.2833 — see caveat) |

### Needle recall sweep (BLUE-742 @ ~token 200; reps=8, sink=16, window=1024)

| K_STATES | seg | recall@8k | recall@16k | recall@32k | bank slots/layer @32k | cache MiB @8k/16k/32k |
|---|---|---|---|---|---|---|
| 4 | 256 | True | True | True | 32 | 36.9 / 36.9 / 36.9 |
| 4 | 512 | True | True | True | 32 | 36.9 / 36.9 / 36.9 |
| 16 | 256 | True | True | True | 128 | 38.4 / 38.4 / 38.4 |
| 16 | 512 | True | True | True | 128 | 38.2 / 38.4 / 38.4 |
| 64 | 256 | True | True | True | 512 | 39.9 / 43.9 / 44.4 |
| 64 | 512 | True | True | True | 496 | 38.2 / 40.2 / 44.2 |

Pseudo-slot cost: K_STATES x reps, i.e. 32 slots/layer at K=4 and 128 at K=16 — tiny vs the
1024-slot ring (512 at K=64 is the ceiling case). The banked cache is a fixed ~37–44 MiB
object, +0.7–8 MiB over plain ring.

### Honest limits

- **Training-free, single synthetic needle task ≠ general quality.** One planted fact in
  repetitive filler is the easiest recall target: its keys are extreme novelty outliers.
  Real long-context quality (RULER/LongBench, multi-needle, paraphrase queries) is
  untested — that is what the learned RWKV-MS sidecar is for later.
- NLL@8192 beating even full KV is a filler artifact: mean-summary slots of a repeated
  paragraph are excellent predictors of more repeated paragraph. Read it as "the bank does
  not hurt" (acceptance 5), not as improved language modeling.
- The mean-summary slot (the actual "compressed state") is not what closed the gap — the
  exact-KV representatives were. Linear-attention readout (option (b)) and an RWKV-7-style
  associative fold remain unexplored here; what shipped is importance-sampled eviction
  (SnapKV/H2O-adjacent) organized as a segmented, merge-bounded, flat-footprint bank.
- Leader-based selection assumes shallow-layer token saliency transfers to deep layers —
  true here and consistent with the shallow-band memory finding, but unvalidated beyond
  this task.
- B=1 only (`batch_repeat_interleave` intentionally raises on banked layers); folding adds
  host-side Python per eviction event, fine at chunk granularity.

## PoC-3: throughput fix (SDPA + mask-free decode + CUDA-graphed static ring)

**Goal**: close the 4k-context aggregate-throughput gap vs vLLM (COMPARISON.md §2:
835 tok/s @ B=64 ours vs 1,356 tok/s @ 38 seqs vLLM). Run date 2026-07-04; same GPU,
desktop process ~1.2 GB (vs 2.0–2.4 GB during PoC-1). **New, stricter budget**: the
allocator is hard-capped at 19 GiB (`--mem-cap-gib`, keeps peak under 19 GB on the
shared desktop), so green = peak reserved <= 18 GiB — PoC-1's green line was 20.07 GiB.
Baseline reproduced under today's conditions: 810.5 tok/s @ B=64 (32-step timing),
consistent with the recorded 835.4 (128-step).

### Profile first (`profile` subcommand; ring, ctx 4096, 32 decode steps)

Per-step breakdown via torch.profiler with record_function ranges around the attention
interface, mask construction and cache update; "GPU busy" sums device-kernel events
only; wall from a separate un-profiled run of the same steps.

| ms/step | eager B=32 | eager B=64 | fixed B=32 | fixed B=64 |
|---|---|---|---|---|
| wall (clean) | 49.0 | 79.0 | 29.6 | 38.3 |
| GPU busy | 47.9 (91%) | 74.3 (95%) | 25.2 | 34.1 |
| attention (full/ring layers) | 10.7 | 20.0 | 1.4 | 2.6 |
| attention (sliding layers) | 14.8 | 26.7 | 1.2 | 2.6 |
| cache update (cat/evict) | 4.7 | 9.4 | 4.4 | 9.6 |
| other model compute | 17.4 | 17.8 | 18.3 | 19.3 |
| python/launch gap | 4.8 | 4.3 | 3.4–10.4 | 3.3 |
| mask build (CPU-side, overlapped) | 13.2 | 18.7 | 0.0 | 0.0 |

The measured bottleneck at B=64 was **eager attention: 46.7 of 79 ms/step (59% of
wall)** — repeat_kv materialization (2 KV heads -> 8) over every cached slot on all 42
attention calls plus fp32 attn-weight buffers — with per-step mask construction burning
another 13–19 ms of CPU (overlapped at these batch sizes, but it becomes the wall once
attention shrinks). Python/launch gap was only ~5% — so CUDA graphs alone would NOT
have fixed this; the profile redirected the effort to attention first. "Other model
compute" is flat in B (weight-bandwidth-bound GEMMs, ~14 GiB of weights read per step)
— that is what makes larger B nearly free once attention is fixed.

### What changed (all in `experiments/gemma_recurrent_poc.py`)

1. **SDPA + mask-free decode** (`--attn sdpa` is the new default). At q=1 decode with
   no padding, every cached slot is unconditionally visible (sink/bank/pending/ring all
   precede the query; a sliding layer stores exactly the last window-1 tokens). We pass
   a pre-built `{"full_attention": None, "sliding_attention": None}` mask mapping, which
   (a) skips the vmap-based per-step mask build entirely and (b) lets SDPA take its
   mask-free path: `is_causal=False` full attention over the cache with `enable_gqa=True`
   — no repeat_kv copy at all. Sliding-layer attention: 26.7 -> 2.6 ms/step at B=64.
2. **Grouped-GEMM decode attention for the wide full-attention heads.** gemma-4's
   full-attention layers use `global_head_dim=512`; no fused SDPA kernel on sm_89
   supports that with GQA, and torch's math fallback materializes GiB-scale fp32
   buffers (OOMed at B=64 under the 19 GiB cap and was *slower* than eager). For the
   exact decode case (mask None, q=1, head_dim>256) we patch the sdpa interface with a
   two-GEMM grouped-query path (q [B,8,1,512] viewed as [B,2,4,512] — the enable_gqa
   head mapping — then fp32 softmax like the eager path): full-layer attention 20.0 ->
   2.6 ms/step at B=64, no big buffer.
3. **CUDA-graphed decode step** (`--graphs`), reusing the M2 GraphedDecode recipe
   (static input/position tensors, side-stream warmup, capture, replay). What had to
   differ from M2: the HF DynamicCache mutates by `torch.cat` (fresh addresses every
   step — unreplayable), so the prefilled cache is first converted to **StaticRingLayers**
   (`to_static_cache`): fixed-address [B,H,cap,D] buffers written in place at a
   *device-tensor* write pointer that advances with captured tensor ops. Slot order
   rotates instead of staying chronological — harmless because decode attention is
   mask-free (order-invariant) and keys are post-RoPE. Capacities chosen for exact
   equivalence with the dynamic layers' per-step visible set (sliding: cap=window;
   ring: cap=sink+window+1, ring over [sink, cap)). The whole step — forward, argmax,
   token feedback, position bump — is captured; the decode loop is `graph.replay()`
   plus one D2D copy per step, one sync at the end (fix 2c: no per-step H2D/D2H).
   Cache eviction thereby also became a 1-slot `index_copy_` instead of the per-step
   cat/evict pair (9.6 ms/step at B=64 in the table above).
4. **Memory**: batch replication now broadcasts the B=1 prefill *directly into* the
   static buffers (no transient B-wide dynamic cache; still real per-slot copies), and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the script default (~1.2 GiB
   less fragmentation at B>=96 for a few % decode throughput). B=128 went from
   hard-OOM to completing within the cap.

### Correctness gates (`verify` subcommand — 8/8 pass, run before any speed claim)

| gate | result |
|---|---|
| NLL(last128)@900 full==ring==banked, eager | PASS — 0.549035 all three (bit-identical) |
| NLL(last128)@900 full==ring==banked, sdpa | PASS — 0.548670 all three (bit-identical) |
| NLL@900 sdpa-vs-eager drift | PASS — 3.65e-4 (<1e-3) |
| greedy tokens eager==sdpa (ring ctx 4096, B=4, 64 steps) | PASS (tie-aware) — free-run 16/256 differ; teacher-forced argmax differs on 2/256, both at top-2 logit gaps <= 0.375 (1–3 bf16 ULPs; one exact 0.0 tie in eager). Numeric tie-flips, not semantic divergence. |
| banked needle@8192 recalled under sdpa | PASS — 'BLUE-742' (bank slots are ordinary KV; SDPA doesn't care) |
| banked greedy output eager==sdpa | PASS — identical first line |
| static-ring tokens == dynamic tokens (B=4, 64 steps) | PASS — strictly token-identical this run |
| graphed tokens == ungraphed static tokens | PASS — 0/256 differ (exact, as required: same kernels, same addresses) |

### Before/after: concurrency ladder, ring @ ctx/session 4096 (128-tok greedy decode)

Before = PoC-1 recorded numbers (eager, masked decode, green line 20.07 GiB).
After = SDPA+mask-free (`sdpa`) and +CUDA graphs (`graphed`), green line **18.0 GiB**
(19 GiB hard cap, 1 GiB headroom — stricter than before).

| B | before agg tok/s | sdpa agg tok/s | graphed agg tok/s | graphed per-stream | graphed peak resv GiB | status (new budget) |
|---|---|---|---|---|---|---|
| 8 | 310.1 | 385.6 | 447.1 | 55.89 | 14.58 | green |
| 16 | 503.6 | 701.1 | 854.6 | 53.41 | 14.62 | green |
| 32 | 669.1 | 1,181.0 | 1,566.3 | 48.95 | 15.27 | green |
| 64 | 835.4 | 1,820.6 | 2,730.8 | 42.67 | 16.42 | green |
| 96 | 869.5 (over-budget) | 2,021.3 (over) | 3,544.9 | 36.93 | 17.64 | **green — new B_max** |
| 112 | OOM | 1,860.7 (over) | 3,983.2 | 35.56 | 18.21 | over-budget |
| 128 | OOM | OOM | 4,344.2 | 33.94 | 18.84 | over-budget (in-cap) |
| 144 | OOM | OOM | OOM | - | 18.95 | OOM |

ctx/session 16384 (graphed) is **identical within noise** — 2,770.4 @ B=64, 3,585.1 @
B=96 (green), 4,330.0 @ B=128 — the flat-slot property carries through the fix.

### New B_max / saturation

- **B_max(green) = 96** (was 64) at **3,545 tok/s** — 4.2x the old green-ceiling
  throughput, under a ~2 GiB *stricter* budget. Hard ceiling B=128 @ 4,344 tok/s
  (5.2x), OOM at 144.
- **The old saturation point is gone**: before, B=64->96 added +4%; now 64->96 adds
  +30% and 96->128 adds +23% (0.69 scaling efficiency — bending but still climbing).
  Throughput no longer flattens before memory runs out: **the ceiling is memory again**,
  dominated by the 20 stock sliding-window layers (21 of every 36.3 MiB/slot) — i.e.
  exactly the arena/slot-shrinking work (paged sliding slabs, smaller windows) that was
  always the roadmap, not kernel overhead.
- Per-stream decode also improved at every B (55.9 tok/s @ B=8 vs 38.8 before).

### Honest limits

- The 2 teacher-forced argmax flips per 256 tokens (bf16 ULP ties) mean greedy streams
  are reproducible-modulo-ties across attention impls, exactly as between any two
  kernel stacks; graphed-vs-ungraphed is bitwise exact.
- `--graphs` requires prefill >= sink+window per layer (static ring must be full) and
  is ring-mode only; banked mode keeps the eager path (its per-step host-side folding
  is inherently uncapturable — and it is the B=1 quality path anyway).
- The concurrency numbers remain replicated-cache virtual sessions (honest memory,
  throwaway quality), 128-token steady-state decode with no arrivals/scheduler.
- Old bench/needle/bank tables above were measured with eager attention
  (`--attn eager --legacy-decode` reproduces that path); PoC-3 changed the defaults.

## Repro

```
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py bench
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py bench --mode full --ctxs 32768 --chunk 1024
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py needle --ctx 8192
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py batch
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py concurrency
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py concurrency --mode ring --ladder 96 112
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py bank
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py needle --mode banked --ctx 8192
# PoC-3 (throughput fix):
HF_HUB_OFFLINE=1 .../python experiments/gemma_recurrent_poc.py profile --attn eager --legacy-decode --batch-size 32 --ctx-per-session 4096   # before
HF_HUB_OFFLINE=1 .../python experiments/gemma_recurrent_poc.py profile --batch-size 64 --ctx-per-session 4096                               # after
HF_HUB_OFFLINE=1 .../python experiments/gemma_recurrent_poc.py verify --ctx-per-session 4096                                                # 8 gates
HF_HUB_OFFLINE=1 .../python experiments/gemma_recurrent_poc.py concurrency --mode ring --graphs --ctx-per-session 4096  --ladder 8 16 32 64 96 112 128 144
HF_HUB_OFFLINE=1 .../python experiments/gemma_recurrent_poc.py concurrency --mode ring --graphs --ctx-per-session 16384 --ladder 8 16 32 64 96 112 128 144
```

Note: model weights live on the NTFS volume mounted at
`/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it` (auto-detected by the
script; mount with `udisksctl mount -b /dev/nvme1n1p2` if absent). E4B fit fine — the E2B
fallback was not needed.
