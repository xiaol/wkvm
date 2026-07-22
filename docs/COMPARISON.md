# wkvm vs vLLM vs SGLang vs Albatross

*Measured on one machine: RTX 4090 24GB (Ada, sm_89), local checkpoints, 2026-07-03. Engine-architecture claims verified against main-branch source (see ANGLE.md for the full audit). Measured sections are filled from runs in `/run/media/.../wkvm_bench/` and `experiments/results/`; anything not measured here is marked as such.*

> **Scope update:** the historical 10x-class rows below remain specialized
> steady-state PoC decode. The strict short-session A800 gate remains FAIL at
> 0.790x versus vLLM and 1.400x versus SGLang. A later exact-trace vLLM mode-3
> audit superseded the one-repeat RTX 4090 long-lived 10x-vLLM conclusion: the
> corrected cross-run ratio is 9.827x. The original SGLang row remains 26.079x,
> but it has no matching later optimization audit. All of these comparisons are
> exploratory, and WKVM `routed_span_approximate` semantics differ from
> incumbent `full_kv` semantics.

## Latest provider-HTTP complete-session result

Gemma-4-E4B-it, BF16, B16, 48 synchronized turns, 36,864 initial tokens,
32 new input tokens per continuation, 64 output tokens per request, and a
24,200 MiB whole-device ceiling:

| Engine | Semantic mode | Full wall | Full output tok/s | Peak whole GPU |
|---|---|---:|---:|---:|
| WKVM | `routed_span_approximate` | **180.415 s** | **272.439** | 23,856 MiB |
| vLLM 0.24.0 mode-3 audit | `full_kv` | 1,772.936 s | 27.724 | within 24,200 MiB |
| SGLang 0.5.14 | `full_kv` | 4,705.123 s | 10.446 | 23,597 MiB |

The audited cross-run ratios are **9.827x versus vLLM** and **26.079x versus the
original SGLang row**. Every referenced artifact completes 768/768 requests
with zero errors and uses the same exact 48-turn source/replay trace. The long
session still demonstrates the resident-state advantage, but it no longer
passes a current 10x-vLLM gate. See the
[`full report`](../experiments/results/gemma_4090_48turn_10x_20260717.md) for
launch settings, continuation results, trace identity, and the superseding
audit.

## 1. What each engine is

| | wkvm | vLLM | SGLang | Albatross |
|---|---|---|---|---|
| Primary memory object | fixed-size per-request **state slot** | paged KV blocks (hash-indexed) | paged KV + radix tree | dense per-batch RNN state |
| Model scope | RWKV-7/GDN/hybrids native; transformers as guests or **recurrent mode** | ~287 model families | ~202 model families | RWKV-7 only |
| Codebase | ~2k LOC core (M0-M1) + PoC | ~724k LOC Python | ~626k LOC Python (srt) | ~5k LOC CUDA/C++/Py |
| Scheduler | no-phases token budget, exact slot admission | no-phases token budget, block watermarks | overlap event loop, retraction controller | static batch, no scheduler |
| Concurrency model | slots: admission = counting | blocks: admission = watermark math | tokens: admission = radix-aware estimate | fixed B at launch |
| Sessions/state API | durable, forkable, mutable handles (M3) | prefix cache ≡ f(token prefix) | radix cache ≡ f(token prefix) | manual state tensors |
| Hardware | one target: CUDA | 6 platforms | 7 platforms | one GPU model at a time (layout-tuned) |

The structural difference: vLLM and SGLang index *reusable KV by token prefix*; state is always recomputable-from-tokens. wkvm treats state as the primary, *mutable* object (fork/hibernate/migrate/consolidate) — which prefix-keyed caches cannot represent. Albatross proves the raw physics ceiling for RNN decode but has no serving layer (no continuous batching, no admission, no sessions).

## 2. Measured: gemma-4-E4B-it, 4090, concurrency & memory

Same model, same GPU, same measurement shape (N concurrent sessions at ctx tokens each, 128-token greedy decode). wkvm numbers are the PoC recurrent mode (sink16 + ring1024 on the 4 KV-owning global layers) after the PoC-3 throughput fix: SDPA/GQA mask-free decode + CUDA-graphed static-ring decode step (`experiments/results/poc1_gemma_e4b.md` §PoC-3, run 2026-07-04). Note the wkvm re-run used a *stricter* budget than the 2026-07-03 vLLM run: allocator hard-capped at 19 GiB, green = peak reserved <= 18 GiB (the PoC-1 green line was 20.07 GiB).

**Concurrency (max resident sessions within budget):**

| ctx/session | wkvm recurrent mode (PoC-3) | vLLM 0.24.0 | SGLang 0.5.14¹ | HF full-KV (eager baseline) |
|---|---|---|---|---|
| 4,096 | **96** (36.3 MiB/slot; 128 within the 19 GiB hard cap) | 38 (84 MiB/seq) | 6 (25,360-token pool) | 8 (86 MiB/seq) |
| 16,384 | **96** (36.3 MiB/slot, unchanged) | 9 (276 MiB/seq) | 1 | 0 within headroom |
| 32,768+ | **96** (unchanged² ) | ~4 (would need 552 MiB/seq) | 0 | OOM |

**Aggregate decode throughput (greedy, 128 new tokens/stream):**

| ctx | wkvm PoC-3 (SDPA + CUDA graphs) | vLLM (FlashInfer + CUDA graphs) | SGLang 0.5.14¹ (triton attn) |
|---|---|---|---|
| 4,096 | **3,545 tok/s** @ B=96 green (2,731 @ B=64; 4,344 @ B=128 in-cap) | 1,356 tok/s @ 38 seqs | 410 tok/s @ N=64 (6 concurrent, rest queued) |
| 16,384 | **3,585 tok/s** @ B=96 green | 286 tok/s @ 9 seqs (capacity-starved) | 68 tok/s @ N=8 (1 concurrent) |

¹ SGLang required **9 attempts and three root-cause fixes** to serve this model on this machine (fixed 2026-07-05; full forensic chain in `experiments/results/bench_sglang_gemma4e4b.md`): (i) SWA/full-KV pool configurator starvation on 24GB — manual joint tuning of `max_running_requests` × eviction interval × mem-fraction; (ii) a `Gemma4TextModel` attribute bug in its tc_piecewise prefill graph backend — worked around by disabling prefill graphs; (iii) a **corrupted `nvidia-cutlass-dsl` install** (mixed-version files, NVIDIA/cutlass#3132 failure mode) producing an MLIR ICE that survived every user-facing kill-switch because `sgl_kernel` imports flashinfer behind its own availability flag — fixed by clean reinstall of the same version, CPU-repro-verified; plus (iv) the tvm-ffi JIT then needed the GCC15/glibc `rsqrtf` CPATH shim. Its numbers here carry real handicaps it could shed on a friendlier stack: triton attention backend, no prefill graphs, and multimodal tower weights resident (vLLM's run zeroed them via `limit_mm_per_prompt`), which is much of why its KV pool is 25,360 tokens vs vLLM's 161,584 at comparable mem-fraction. Treat SGLang's row as "what surviving the JIT-chain gauntlet cost", not its potential.

² not re-run at 32k; measured identical at 4k and 16k (flat slots), and ring state is context-independent by construction.

Read: with the PoC-3 fix (profile-driven: eager attention was 59% of the decode step; replaced with SDPA/GQA mask-free decode plus a grouped-GEMM path for the head_dim-512 global layers, then the whole step CUDA-graphed over a fixed-address static ring cache) wkvm now beats vLLM at short context too, not just on capacity: **2,731 tok/s at vLLM-comparable concurrency (B=64) and 3,545 tok/s at its own green B_max=96 vs vLLM's 1,356 — ~2–2.6×** — measured under a ~2 GiB stricter memory budget, and 12.5× at 16k where vLLM is capacity-starved. Two honest qualifiers, in fairness order: (1) this is *not* a same-semantics win — vLLM serves full-KV attention; wkvm's recurrent mode is exact only below the ring window, and the quality cost past the window is now **measured, not waved at** (docs/RECURRENT_MODE_QUALITY.md): +0.3→1.4 nats NLL on natural text with the bank recovering only ~0.1, single-fact recall 0.72 / multi-key 0.26 / aggregation 0.21 vs full-KV's 1.00/1.00/0.83 on evicted depths — and the earlier PoC-2 "perfect recall to 32k" was a repetitive-filler artifact, corrected by that evaluation. vLLM cannot be configured to do constant-footprint serving at all, but a user who needs full-KV semantics gets no benefit from our number; (2) the wkvm ladder is replicated-cache virtual sessions in steady-state decode (no arrivals, no scheduler, no HTTP), while vLLM's number includes its engine loop — shape-matched to its offline `LLM` path but not identical plumbing. Greedy outputs across the fix are verified: 8/8 gates (graphed vs ungraphed bitwise token-identical; SDPA-vs-eager differs only at bf16-ULP logit ties, 2/256 teacher-forced). vLLM's hybrid-KV accounting (sliding layers bounded, 18 shared) remains honest engineering — but its 4 full-attention layers still grow without bound, and past ~8k that is the whole difference. Throughput now scales to the memory wall (no pre-memory saturation: +30% from B=64→96, +23% from 96→128); the ceiling is the 20 stock sliding-window layers' 21 MiB of every 36.3 MiB slot — an arena/paging target, not kernel overhead.

## 3. Measured: RWKV-7, 4090, decode throughput

Albatross is the reference point for what state-slot decode can reach with tuned kernels; wkvm M1 is a first untuned FLA-kernel runner (no CUDA graphs at M1).

Albatross `faster3a_2605`, RWKV-7 World 2.9B fp16, pre-built sm_89 kernels (constants tuned for 5090 — untuned on this 4090), decode BxT=Bx1, p50 of 10 iters:

| B | 1 | 8 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| tok/s | 195 | 840 | 3,595 | 6,042 | 10,158 | 12,707 | 14,029 | **15,356** |

7.6 GiB at model-ready; states add ~11 MiB/session. **1,024 concurrent streams on one 24GB consumer GPU at 15.3k tok/s** — vs the transformer table above (38 sessions, 1.36k tok/s). That ~27×-concurrency / ~11×-throughput gap *is* the state-slot physics wkvm is built on; Albatross proves the ceiling but ships no scheduler, no admission, no sessions, no API — the serving layer is wkvm's job (M1/M2 measured numbers below as they land).

**Measured: wkvm M2 engine, RWKV-7 World 1.5B bf16, same 4090** (`experiments/results/m2_engine_bench.md`). Steady-state decode through the full engine loop — scheduler, arena gather/scatter, batched sampling all in the timed path, unlike Albatross's bare static-batch kernel loop. "Graphed" captures the decode forward in a per-batch-size CUDA graph (verified token-identical to eager):

| B | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| eager tok/s | 67 | 516 | 1,890 | 3,592 | 6,345 | **7,913** |
| graphed tok/s | 195 | 1,084 | 3,565 | 5,274 | 6,886 | **8,077** |
| peak VRAM (GiB, graphed) | 2.96 | 3.19 | 4.36 | 5.92 | 9.84 | 19.02 |

12.19 MiB state/slot; 256 concurrent streams in 19 GiB with a second static state copy for the graph. Honest read: our graphed 1.5B ladder lands at ~0.6–1.0× Albatross's **2.9B** numbers point-for-point — running half the parameters. Per-parameter that is roughly a 2× kernel-efficiency deficit, exactly where it should be: they are hand-tuned sm_89 CUDA with the whole step fused; we are a python engine loop over generic fla kernels with eager per-layer state gather/scatter (72 `index_select` + 72 `index_copy_` per step at 24 layers). The gap is launch/gather overhead, not physics — fusing gather/scatter into the capture is the known next win — and the wkvm number buys what Albatross doesn't have: continuous batching, exact admission, streaming arrivals, per-request sampling. (191M ladder in the results file: 680 → 38,856 tok/s graphed, B=1 → 256.)

## 4. Feature/architecture comparison (verified against source)

| Capability | wkvm (design; M-milestone) | vLLM | SGLang | Albatross |
|---|---|---|---|---|
| Continuous batching | **yes (M2, measured: batch-vs-sequential token-identical)** | yes | yes | no (static batch) |
| Chunked prefill | yes (falls out of token budget; M0 tested) | yes | yes | batched prefill only |
| Prefix/state reuse | checkpoint store, deepest-wins (M3) | hash-block prefix cache; mamba `all` mode | radix tree; mamba checkpoints on nodes | manual |
| Constant-footprint transformers | **yes — recurrent mode (PoC-1/2 measured)** | no (paged KV grows) | no | n/a |
| Mutable/named state handles | **yes (M3, the moat)** | no — violates prefix-hash invariant | no — violates radix-key invariant | states are just tensors (no API, no serving) |
| Fork/rollback of sessions | O(MB) slot copy (M3) | re-prefill through cache | token-granular radix fork (attention only) | manual tensor copy |
| Hibernate/resume sessions | 1 transfer per session (M3) | KV offload connectors (GB-scale) | HiCache tiers (GB-scale) | no |
| CUDA graphs | **yes (M2, decode forward per batch bucket, token-identical)** | full+piecewise dispatcher | 3 capture backends | yes (key to its 15k tps) |
| Spec decode / grammar / LoRA | deferred deliberately | yes (broad) | yes (broad) | no |
| Trainer-kernel parity for RL | same FLA kernels train/serve (M-later) | batch-invariant mode (kernel-level) | deterministic mode (caveat: trainer kernels differ) | n/a |

## 5. Honest read

**Where wkvm wins (measured or structural):**
1. **Long-context concurrency on transformers** (measured): 96 flat slots at any context (under a 19 GiB cap; 128 at the hard ceiling) vs vLLM's 38→9→~4 as context grows; after PoC-3 (SDPA/GQA + CUDA-graphed static ring) also ~2–2.6× vLLM's aggregate throughput at 4k and 12.5× at 16k on the same GPU. The capacity gap widens with context, unboundedly.
2. **Long-context recall at flat footprint** (measured): the PoC-2 segmented state bank recovers needle recall to 32k in 37–44 MiB/slot — a capability class (constant-memory approximate attention) neither incumbent ships as a serving mode.
3. **Linear-model serving exists at all** (structural + measured): zero RWKV support in either incumbent; wkvm M2 serves RWKV-7 with continuous batching + exact admission + CUDA graphs at 8k tok/s (1.5B, B=256, 19 GiB) — Albatross-class physics with a real serving layer.
4. **State-slot economics** (measured): 12.2 MiB/session at 1.5B, 2.3 MiB at 191M, admission = counting — the substrate the M3 durable-handle API (fork/hibernate/mutate) needs, which prefix-keyed caches structurally cannot offer.
5. **Operational surface** (observed in this exercise): wkvm's whole stack is ~3k LOC with one JIT dependency (triton via fla); SGLang's breadth surface (pool configurator × graph backends × cutlass-DSL JIT chain) failed to boot this model on this toolchain in 7 attempts.

**Where the incumbents win, today and durably:** breadth (models × quant × hardware), kernel maturity (FlashInfer/FA3 and paged prefill vs our SDPA-decode PoC — our *prefill* is still stock masked SDPA), operational polish (metrics, distributed serving, APIs), and community. At equal scope wkvm never catches vLLM on mainstream dense transformers with full-KV semantics — that is not the game.

**Where Albatross wins:** peak RWKV decode throughput per GPU from hand-tuned CUDA (its entire codebase optimizes one model on one GPU). wkvm's bet is that Albatross-class physics + a real serving layer (admission, batching, durable state) is worth more than the last 30% of kernel tuning — and Albatross has no answer for hybrids, transformers, or sessions.

**Threats to monitor:** vLLM extending mamba `all`-mode + CPU offload to GDN (absorbs plain checkpointing value); linear/hybrid adoption reversing at frontier labs.

**Measurement caveats, all of them:** single machine, single day, one model per class; wkvm concurrency sweeps used replicated-cache sessions (memory-honest, quality-throwaway); recurrent mode is not full-KV semantics (exact only below the ring window; beyond it, bank recall is validated on needle tasks, not general quality); Albatross numbers are on 5090-tuned constants (expect +some% if retuned for 4090); vLLM ran with prefix caching off (workload had no shared prefixes); SGLang's DNF is toolchain-specific (CUDA 13.1 × nvidia-cutlass-dsl), not fundamental; wkvm's engine bench excludes an HTTP layer (M2 has none — the incumbents' numbers include their engine loop but our offline-API comparison shape matches vLLM's `LLM` offline path). Numbers will move; the *shapes* (flat vs growing memory, capacity-dominated long-context throughput, per-param kernel deficit) are the durable findings.
