# Why RWKV-7 Has the Serving Advantage: The Arithmetic

*Companion to COMPARISON.md. Every number here is either measured on this RTX 4090 (2026-07-03 benches) or computed from first principles against its specs (1,008 GB/s memory bandwidth, ~165 TFLOPS dense fp16 tensor). The point of this document is that the measured 11× throughput / 27× concurrency gap is not an engineering artifact — it is a roofline consequence, and both engines involved are running at essentially the same hardware efficiency.*

## 1. The one-line answer

**Decode is memory-bandwidth-bound, and the cost of serving one token is (weights ÷ batch) + per-session traffic. RWKV-7's per-session traffic is a fixed ~22 MB (state read+write) while a transformer's is its KV read — 88 MB at 4k context, growing linearly, forever. Worse: the transformer can't raise the batch to amortize the weights, because KV *capacity* caps its batch at 38 on this card while RWKV holds 1024.** Both effects multiply.

## 2. Roofline vs measured (nothing is hidden)

Per decode step at batch B: bytes moved ≈ weights + B × session_traffic; FLOPs ≈ 2 × params × B. Time ≈ max(bytes/1008 GB/s, FLOPs/165 TFLOPS).

**RWKV-7 2.9B fp16 (weights 5.8 GB, state 11.3 MB/session) vs Albatross measured:**

| B | bandwidth bound | compute bound | roofline tok/s | measured | efficiency |
|---|---|---|---|---|---|
| 1 | 5.8 ms | ~0 | 173 | **195** | **113%** ← see §4 |
| 64 | 7.2 ms | 2.2 ms | 8,903 | 6,043 | 68% |
| 128 | 8.6 ms | 4.5 ms | 14,843 | 10,159 | 68% |
| 256 | 11.5 ms | 9.0 ms | 22,273 | 12,705 | 57% |
| 512 | 17.2 ms | 18.0 ms | 28,448 | 14,027 | 49% |
| 1024 | 28.7 ms | **36.0 ms** | 28,448 | 15,357 | 54% |

**Gemma-4-E4B bf16 (weights 16.0 GB, KV read 88 MB/seq at 4k) vs vLLM measured:** at its capacity cap N=38: roofline 1,981 tok/s, measured 1,356 → **68% efficiency — the same as Albatross's mid-range.** vLLM is not leaving performance on the table; it is pinned to a worse roofline.

Three readings:

1. **The engines are equally good; the workloads are not.** 68% MBU on both sides. The 11× measured gap (15,357 / 1,356) is the ratio of achievable rooflines, not engineering quality.
2. **Per-token traffic, the cleanest single number:** at each engine's operating point, E4B moves **509 MB per token** (16 GB of weights amortized over only 38 tokens + KV) while RWKV moves **28.3 MB per token** (5.8 GB amortized over 1024 + state). **18.0×** — that is the whole game on a bandwidth-bound workload.
3. **RWKV crosses into compute-bound at B≈512** — the only good place for an inference workload to be (FLOPs are what the silicon is for). A 24 GB transformer deployment *never gets there*: capacity caps its batch while it is still weight-read-bound.

## 3. Why the transformer cannot follow (three stacked walls)

**Wall 1 — capacity.** 11.3 MB/session vs 84 MB (4k) / 276 MB (16k) / 552 MB (32k). On 24 GB after weights: 1024+ RWKV sessions vs 38 / 9 / ~4. Measured, not estimated. And this is a *favorable* transformer: E4B's 35 sliding-window layers + 18 KV-shared layers already cut KV ~4.6× vs uniform — 4 unbounded full-attention layers still produce this wall.

**Wall 2 — amortization ceiling.** Weight reads amortize as weights/B. The transformer's B is Wall-1-capped at 38, so it pays 421 MB of weight traffic per token; RWKV at B=1024 pays 5.7 MB. Even granting the transformer infinite memory (84 GiB of KV for B=1024 @4k), its roofline is 9,719 tok/s — **still below RWKV's measured 15,357**, because of Wall 3.

**Wall 3 — the asymptote.** As B→∞ the per-token cost converges to per-session traffic itself: ~22.6 MB for RWKV (context-independent) vs 88 MB @4k / 350 MB @16k / linear-in-context for the transformer. The advantage therefore *grows without bound in context length*. This is the same physics that made our recurrent-mode PoC beat vLLM 2.7× at 16k with a python-loop prototype.

**Plus the quiet multiplier — uniformity.** Every RWKV decode step has identical shapes: no block tables, no paged gather, no per-step attention-metadata build, no ragged batches. That is why whole-step CUDA graphs are trivial (M2 captured them first try; token-identical) and why the scheduler collapses to counting slots. vLLM's 2,653-line scheduler and 42-backend attention metadata machinery exist to manage exactly the non-uniformity RWKV doesn't have.

## 4. The B=1 anomaly (worth knowing)

Albatross measures 195 tok/s at B=1 — **above** the 173 tok/s dense weight-read roofline. That is not a measurement error; it is their "lossless sparse FFN": at batch 1 the FFN's active rows are sparse enough to skip reading a fraction of the weights. Two lessons: (a) single-stream decode is so bandwidth-starved that skipping weight bytes is the only lever left; (b) this trick *stops composing at large batch* (the union of active rows densifies), which is why it is a batch-1 feature and why large-batch numbers are the honest ones for serving economics.

## 5. What this does NOT say

- Nothing here is about **quality per parameter**. Exact attention buys exact recall; RWKV-7's fixed state is a lossy summary. The serving advantage is orthogonal to the quality question — which is precisely why frontier labs ship *hybrids* (Qwen3-Next, Kimi-Linear: mostly-linear + a few full-attention layers) and why wkvm serves hybrids with a guest allocator rather than refusing them.
- Albatross's constants are 5090-tuned; its 4090 efficiency (49–68%) would improve with retuning — the gap vs transformers *understates* slightly.
- At very short contexts (≤1k) and low concurrency, the walls barely bind and a transformer with good kernels is simply fine. The advantage is a *scaling* claim: in sessions × context, not a universal one.

## 6. Implications baked into wkvm

1. Slots-as-primary-allocation is the capacity wall turned into an API (exact admission = counting).
2. Whole-step decode graphs are the uniformity dividend (M2: 2.9× at B=1).
3. The Durable State API is the 11 MB session turned into a product (fork/hibernate/migrate are O(MB) operations — meaningless at 84–552 MB/session, transformative at 11).
4. Recurrent mode is the same physics retrofitted onto transformers: cap Wall 1 and 3 by construction (ring + bank), accept approximation beyond the window.
