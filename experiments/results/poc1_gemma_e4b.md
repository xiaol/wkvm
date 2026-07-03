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

## Repro

```
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py bench
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py bench --mode full --ctxs 32768 --chunk 1024
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py needle --ctx 8192
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py batch
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py concurrency
HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python experiments/gemma_recurrent_poc.py concurrency --mode ring --ladder 96 112
```

Note: model weights live on the NTFS volume mounted at
`/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it` (auto-detected by the
script; mount with `udisksctl mount -b /dev/nvme1n1p2` if absent). E4B fit fine — the E2B
fallback was not needed.
