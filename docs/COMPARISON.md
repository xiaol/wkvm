# wkvm vs vLLM vs SGLang vs Albatross

*Measured on one machine: RTX 4090 24GB (Ada, sm_89), local checkpoints, 2026-07-03. Engine-architecture claims verified against main-branch source (see ANGLE.md for the full audit). Measured sections are filled from runs in `/run/media/.../wkvm_bench/` and `experiments/results/`; anything not measured here is marked as such.*

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

Same model, same GPU, same measurement shape (N concurrent sessions at ctx tokens each, 128-token greedy decode). wkvm numbers are the PoC recurrent mode (sink16 + ring1024 on the 4 KV-owning global layers; eager attention, no CUDA graphs yet).

**Concurrency (max resident sessions within budget):**

| ctx/session | wkvm recurrent mode (PoC) | vLLM 0.24.0 | SGLang 0.5.14 | HF full-KV (eager baseline) |
|---|---|---|---|---|
| 4,096 | **64** (36.2 MiB/slot) | 38 (84 MiB/seq) | DNF¹ | 8 (86 MiB/seq) |
| 16,384 | **64** (36.2 MiB/slot, unchanged) | 9 (276 MiB/seq) | DNF¹ | 0 within headroom |
| 32,768+ | **64** (unchanged) | ~4 (would need 552 MiB/seq) | DNF¹ | OOM |

**Aggregate decode throughput (greedy, 128 new tokens/stream):**

| ctx | wkvm PoC (eager attention, no graphs) | vLLM (FlashInfer + CUDA graphs) | SGLang |
|---|---|---|---|
| 4,096 | 835 tok/s @ B=64 | **1,356 tok/s** @ 38 seqs | DNF¹ |
| 16,384 | **782 tok/s** @ B=64 | 286 tok/s @ 9 seqs (capacity-starved) | DNF¹ |

¹ SGLang 0.5.14 could not serve this model on this machine after 7 documented attempts (SWA/full-KV pool configurator starvation on 24GB, a `Gemma4TextModel` attribute bug in its tc_piecewise graph backend, and a `nvidia-cutlass-dsl` internal compiler error on the CUDA 13.1 toolchain that survives every user-facing flashinfer kill-switch). Full chain in `wkvm_bench/results_sglang.md`. This is an operational finding about breadth-surface fragility, not an architectural verdict — the same venv-toolchain family served vLLM first try.

Read: vLLM wins per-step efficiency at short context (its kernel stack vs our eager-attention prototype — an M2 gap, not physics); wkvm wins on *capacity*, and past ~8k context capacity dominates: at 16k the PoC's 64 flat slots deliver 2.7× vLLM's aggregate throughput on the same GPU. vLLM's own hybrid-KV accounting (sliding layers bounded, 18 shared) is genuinely good — the 84 MiB/seq at 4k is honest engineering — but the 4 full-attention layers still grow without bound, and that is the whole difference. Quality caveat: wkvm's recurrent mode is exact only below the ring window; beyond it, recall comes from the PoC-2 state bank (needle verified to 32k) — full-KV semantics it is not.

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
1. **Long-context concurrency on transformers** (measured): 64 flat slots at any context vs vLLM's 38→9→~4 as context grows; 2.7× vLLM's aggregate throughput at 16k on the same GPU, with the eager-attention prototype. The gap widens with context, unboundedly.
2. **Long-context recall at flat footprint** (measured): the PoC-2 segmented state bank recovers needle recall to 32k in 37–44 MiB/slot — a capability class (constant-memory approximate attention) neither incumbent ships as a serving mode.
3. **Linear-model serving exists at all** (structural + measured): zero RWKV support in either incumbent; wkvm M2 serves RWKV-7 with continuous batching + exact admission + CUDA graphs at 8k tok/s (1.5B, B=256, 19 GiB) — Albatross-class physics with a real serving layer.
4. **State-slot economics** (measured): 12.2 MiB/session at 1.5B, 2.3 MiB at 191M, admission = counting — the substrate the M3 durable-handle API (fork/hibernate/mutate) needs, which prefix-keyed caches structurally cannot offer.
5. **Operational surface** (observed in this exercise): wkvm's whole stack is ~3k LOC with one JIT dependency (triton via fla); SGLang's breadth surface (pool configurator × graph backends × cutlass-DSL JIT chain) failed to boot this model on this toolchain in 7 attempts.

**Where the incumbents win, today and durably:** breadth (models × quant × hardware), kernel maturity (FlashInfer/FA3 vs our eager-attention PoC), operational polish (metrics, distributed serving, APIs), and community. At equal scope wkvm never catches vLLM on mainstream dense transformers with full-KV semantics — that is not the game.

**Where Albatross wins:** peak RWKV decode throughput per GPU from hand-tuned CUDA (its entire codebase optimizes one model on one GPU). wkvm's bet is that Albatross-class physics + a real serving layer (admission, batching, durable state) is worth more than the last 30% of kernel tuning — and Albatross has no answer for hybrids, transformers, or sessions.

**Threats to monitor:** vLLM extending mamba `all`-mode + CPU offload to GDN (absorbs plain checkpointing value); linear/hybrid adoption reversing at frontier labs.

**Measurement caveats, all of them:** single machine, single day, one model per class; wkvm concurrency sweeps used replicated-cache sessions (memory-honest, quality-throwaway); recurrent mode is not full-KV semantics (exact only below the ring window; beyond it, bank recall is validated on needle tasks, not general quality); Albatross numbers are on 5090-tuned constants (expect +some% if retuned for 4090); vLLM ran with prefix caching off (workload had no shared prefixes); SGLang's DNF is toolchain-specific (CUDA 13.1 × nvidia-cutlass-dsl), not fundamental; wkvm's engine bench excludes an HTTP layer (M2 has none — the incumbents' numbers include their engine loop but our offline-API comparison shape matches vLLM's `LLM` offline path). Numbers will move; the *shapes* (flat vs growing memory, capacity-dominated long-context throughput, per-param kernel deficit) are the durable findings.
