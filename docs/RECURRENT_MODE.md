# Recurrent Mode for Transformers: Segmented State Banks (Multi-State, Layer- and Context-Length-Wise)

*Tier-2 design from the 2026-07-03 angle analysis (see REPORT.md §5). Borrows the mechanism from `X/Multi-state-RWKV-online-memory` (DLA reproduction + RWKV-MS adapters) and turns it into a serving mode of the state-native engine.*

---

## 1. What we're building and why it's now obligatory

Tier 2 was originally scoped as "sink + sliding-window KV ring": constant per-request footprint for any transformer, Albatross-shaped concurrency, at a lossy-eviction quality cost. Two facts from the user's own research upgrade this:

1. **Naive single-state folding is bad and multi-state fixes it** (measured, `Multi-state-RWKV-online-memory/README.md`): on needle recall at matched state count, a single delta-rule memory scores 0.05–0.23, a single RWKV-7 state 0.63–0.80, but **multi-state RWKV-7 (one state per adaptive block) scores 1.000** — matching DLA. And when block boundaries are imperfect (fixed/noisy/low-K), the RWKV-7 state update beats the plain linear block-sum by +0.11 to +0.37. So: *don't fold evicted context into one state; fold it into a bank of per-segment states, and use the RWKV-7 recurrence, which is robust to imperfect segmentation.*
2. **The RWKV-MS research line has no concurrent runtime.** The trained artifact exists (frozen Gemma-4 E4B + ~800k-param RWKV-MS adapters on layers 0–5, tau2 pass^1 0.70 vs 0.20 base; GGUF sidecar format with hash-bound validation), but the patched llama.cpp runtime is **explicitly one-sequence, `-ub 1`, no continuous batching, no context shift**. The engine's recurrent mode is the missing batched production runtime for this exact model format.

## 2. Slot anatomy (per request, per attention layer)

Everything fixed-size at admission ⇒ same arena allocator, exact slot-count admission, uniform decode batches, whole-step CUDA graphs — the Albatross properties, preserved.

```
Slot(layer ℓ) =
  sink KV        [S tokens]                  # S ≈ 4–16, never evicted
  ring KV        [W tokens, circular]        # exact attention over recent window
  state bank     [K × (heads × d×d state + boundary key summary + meta)]
  bank cursor    (which state is "open", token spans per state)
```

- **Ring**: exact attention (FA3/FlashInfer) over sink + window, standard.
- **Bank**: K RWKV-7-style matrix states, each summarizing one *context segment*. This is the "context-length-wise multi-state": the bank is a segmented, capacity-bounded memory over everything the ring has evicted.
- **Layer-wise multi-state**: per-layer heterogeneous config (the tau2 finding: shallow band 0–5 carries memory; KV-shared tail layers get ring-only; deep layers may get smaller K or K=0). The per-layer-family arena already supports heterogeneous slot layouts — this is the same mechanism that lets RWKV layers and attention layers coexist in hybrids.

### Dataflow

- **Decode step (in-graph, static shapes)**: attention output = exact attention over [sink + ring] ⊕ readout over the K bank states (linear-attention readout per state, combined across states), mixed by a gate. All bank ops are fixed-shape tensor ops over `[B, K, H, d, d]` — CUDA-graph-safe by construction.
- **Eviction (host-side, chunk-granular, off-graph)**: when the ring wraps, the evicted chunk of KV is folded into the *open* bank state via the RWKV-7 read-before-write recurrence (chunked FLA kernel — same kernels as the native RWKV path). This runs every ~W/4 tokens, not every token, so it lives in the between-steps path like vLLM's mamba `align` bookkeeping — but simpler, because slots never move.
- **Boundary decision (host-side, at eviction time)**: decide whether the evicted chunk continues the open state or opens a new one. Policies, cheapest first:
  - `fixed`: every N tokens — the robustness result says RWKV-7 states degrade gracefully here (+0.13–0.32 over linear at fixed boundaries).
  - `novelty`: open a new state when the chunk's mean key is far from the open state's key summary (cosine threshold) — a cheap DLA approximation.
  - `dla`: information-aware boundaries (Algorithm 1 of arXiv 2606.10650) computed on chunk statistics.
- **Merge (when the bank is full)**: capacity-bounded adjacent merging (DLA Algorithm 2): merge the two adjacent states with least information loss. This produces multi-timescale structure automatically — old context ends up in coarse merged states, recent context in fine ones. **Merging is exactly a Durable State mutation hook**: consolidation as a first-class serving primitive.

## 3. The angles (as requested — four, orthogonal, composable)

**A. Segmented bank (the core, above).** Context-length-wise multi-state with adaptive boundaries + capacity-bounded merge. Best recall per byte per the user's own tables; needs boundary logic in the eviction path.

**B. Timescale bank (τ-spectrum) — the zero-logic fallback.** K states over the *same* stream with K different decay rates (fast/medium/slow), no boundaries, no merging — pure static ops, trivially graph-safe, zero host-side logic. Readout gate selects timescale. Worse recall than A on needle tasks (no segment isolation) but strictly better than a single state, and it is the right *first implementation* to bring up the plumbing. A and B share the same slot layout; B is A with `boundary_policy=never, decay=per-state-τ`.

**C. Layer-wise budget shaping.** Three levers, all from measured findings: (i) memory modules on the shallow band only (0–5 finding — 6 layers beat 2 and beat all-24); (ii) skip KV-shared tail layers (Gemma-4 patch already does); (iii) optional per-head ring budgets — retrieval-heavy heads keep longer rings, others shorter (retrieval-heads literature; profile once per model, bake into the slot layout). C changes nothing structurally; it's a per-layer config table consumed at slot-layout time.

**D. Learned sidecar as the deployment format.** Two operating points for the readout/gate:
  - **Training-free**: analytic delta-rule folding, gate calibrated on a small text sample (scale so bank readout ≈ the attention mass the evicted tokens would have had). Zero-shot, research-grade quality; ships as `--recurrent-mode=analytic`.
  - **Learned (RWKV-MS)**: frozen base + memory adapter (r8 q,o deltas, shallow band) — the existing recipe, already validated on tau2. Engine natively loads the **sidecar** (adopt the GGUF sidecar's semantic validation: hash-bound to base checkpoint, state-shape metadata). This makes the engine the production runtime for the `xiaol/gemma-4-e4B-hybrid-rnn-mem-*` line and anything trained with the same recipe: `--recurrent-mode=sidecar path.st`.

## 4. Why it stays Albatross-shaped (concurrency math, estimates)

Qwen3-8B-class (36L, GQA 8×128 → ~147KB KV/token), 32GB card, 16GB weights, fp16:

| Serving mode | Per-request memory | Concurrent @ 8k ctx | @ 32k | @ 128k |
|---|---|---|---|---|
| Full KV (vLLM-style) | 147KB × ctx | ~13 | ~3 | <1 |
| Ring only (W=1k) | ~150MB flat | ~90 | ~90 | ~90 |
| Ring + bank (W=1k, K=16, 6-layer band) | ~180MB flat | ~75 | ~75 | ~75 |
| Ring + bank (K=16, all 36 layers) | ~300MB flat | ~45 | ~45 | ~45 |
| (reference: RWKV-7 7B native) | ~20–35MB flat | ~400+ | ~400+ | ~400+ |

The pitch line: **concurrency independent of context length**. At 8k the win over paged KV is ~6×; at 128k it's two orders of magnitude. Decode batches stay uniform (every slot identical shape), so whole-step graphs and exact admission hold; bandwidth per step is weights + B×(ring+bank), independent of context.

Fork/hibernate/migrate apply unchanged: a transformer session in recurrent mode is a ~180–300MB object — bigger than an RWKV state but still 15–60× smaller than its full-KV equivalent, and fixed-size, so the StateStore tiering (GPU→host→NVMe) and `/v1/states` semantics (fork = slot copy, rollback = boundary-aligned) carry over. Bank states are *mutable by design* (merge/decay/consolidate) — the recurrent mode is the Durable State API's second customer, after native RWKV.

## 5. Evaluation plan (quality is the load-bearing risk)

1. **Mechanism parity first**: port the `dla_poc.py` needle-recall harness to the engine's bank implementation; reproduce the 1.000-recall multi-state row and the fixed-boundary robustness deltas before touching a real model.
2. **Model-level, training-free**: perplexity-vs-position and LongBench/RULER subsets on Qwen3-8B: full KV (upper bound) vs ring-only (lower bound) vs ring+bank at K ∈ {4, 16, 64}, boundary ∈ {fixed, novelty}. Success = bank recovers a large fraction of the ring→full gap at long range.
3. **Model-level, learned**: the tau2 telecom split with the Gemma-4 sidecar checkpoint served by the engine at batch > 1 — must match the single-sequence llama.cpp numbers (0.70 pass^1) bitwise-or-nearly, which doubles as the state-sync correctness test the llama.cpp fork couldn't provide.
4. **Serving benchmark**: N concurrent 64k-context sessions on one 4090/5090 vs vLLM — the flat-memory/flat-latency chart.

Risks, honestly: (a) training-free gating may not reach usable quality on real LMs — the fallback is that the mode ships as *learned-sidecar-first* (quality already demonstrated) with analytic mode flagged experimental; (b) eviction-path host work must stay under the decode-step budget at high concurrency — chunk granularity and batched FLA folding kernels are the mitigation; (c) per-model tuning surface (W, K, band, gates) — ship profiles for 2–3 models, not knobs.

## 6. Build order (increments on the Layer-1 substrate, REPORT.md §5)

1. Ring-only mode: sink+window slots in the arena, FA3 over ring, uniform decode graphs. (This is also just the SWA path hybrids need anyway.)
2. Timescale bank (Angle B): static K-state bank, analytic folding, decode readout in-graph. Mechanism harness green.
3. Segmented bank (Angle A): boundary policies + capacity-bounded merge in the eviction path; merge exposed as a `/v1/states` mutation op.
4. Sidecar loader (Angle D): RWKV-MS adapter format, hash-bound validation, tau2 parity run at batch N.
5. Layer budget table (Angle C) + model profiles; the flat-memory demo chart.
