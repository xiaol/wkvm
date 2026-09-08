# Hybrid Engine Plan (M4): Gated DeltaNet + full-attention guests

Goal: make the README's first sentence true for a GDN hybrid — serve
Qwen3.5-9B (24 Gated DeltaNet layers + 8 full-attention layers, the
"Qwen3-Next class" of ROADMAP M4) from wkvm's own arena/scheduler/store, with
the recurrent state as the first-class object and the attention layers as
guests, and make the Durable State API's third way of obtaining a state —
being *given* one — real on a trained artifact (RNN-StateTuning initial
states).

Status legend: ✅ done in this checkout, ▶ next, ☐ later.

## Why this is the next step

- ROADMAP M0–M3 are done; M4 is the next milestone and its first model is a
  Qwen3-Next-class hybrid. `docs/ANGLE.md` §5: "RWKV-7 first, then one GDN
  hybrid to force the guest-allocator path honest."
- `paper/CLAIMS.md` and the report's limitations exclude "general native
  support for GDN, Mamba2, Qwen3-Next, or arbitrary hybrid architectures";
  `wkvm/models/rwkv7.py` refuses hybrids with "hybrid attn layers are M4".
- The user's RNN-StateTuning line produces tuned GDN initial states for
  Qwen3.5-9B (scene-boundary task: 0% → 100% valid JSON, 34% exact match).
  Those are states that no token prefix produces — the exact capability the
  moat argument rests on — and they had no batched runtime.

## Decisions

1. **Same ownership split as M1.** HF `Qwen3_5DecoderLayer` modules are the
   compute graph; wkvm owns state, positions, masks and the cache object
   (`SlotCache`). No fork of HF, no patched `DynamicCache`.
2. **Guest allocator = paged KV pool inside the arena, reserved for the
   request's lifetime.** `guest_kv` is a *paged family*
   (`StateFamilySpec.page_tokens`); the arena hands out
   `ceil((prompt + max_new_tokens) / page_tokens)` pages at admission next to
   the recurrent slots, so the M0 invariant survives unchanged: admission is
   a count (free slots AND free pages), nothing is preempted or retracted
   mid-flight, and a request that could never fit the pool is rejected at
   intake. The cost is that a request which stops early at EOS held pages it
   did not use; the benefit is exactness and a torch-free allocator that is
   unit-tested without a GPU (`tests/test_pages.py`). ROADMAP's "page-bytes
   unified with state pages" trick exists in vLLM to make one block pool serve
   two kinds of memory; with slots and pages both living in the arena there
   is no second allocator to unify, so it is not needed. (Slice 1 of this
   milestone used a fixed per-slot window; it is superseded.)
3. **The engine never branches on model family.** Layouts build their bank and
   runner (`make_bank`/`make_runner`); the store talks to banks through a
   four-method protocol (`fingerprint_key`, `export_slot`, `import_slot`,
   `memory_families`). RWKV-7 keeps its store keys and fingerprint string, so
   M3 indexes remain valid.
4. **Pure-torch kernels are acceptable for this slice.** Throughput is not a
   claim here; correctness and the state contract are. FLA-class kernels
   plug in unchanged when available (HF dispatches to them by name).

## Milestones

### H0. Contract ✅

`docs/qwen35_hybrid_contract.md`: layer types, three state families and
shapes, zero-state == fresh proof, the exact cache calls the HF layers make,
padded-row attention layout and mask formula, positions, admission rule,
store protocol.

### H1. Layout, bank, runner ✅

- `wkvm/models/qwen35.py`: `Qwen35HybridLayout` (families `gdn_state`,
  `gdn_conv`, `guest_kv`; per-slot bytes; factories), `Qwen35Decoder`
  (wkvm-driven layer loop), `load_qwen35`, `load_state_adapter`.
- `wkvm/runner/hybrid_state.py`: `Qwen35StateBank` (layer-major banks,
  slot 0 reserved) + `SlotCache`.
- `wkvm/runner/hybrid_runner.py`: `Qwen35HybridRunner` with the M1 runner
  contract.

Acceptance (CPU, tiny random model, fp32):

```bash
CUDA_VISIBLE_DEVICES= python -m unittest tests.test_qwen35_hybrid_cpu -v
```

Gates in that file: chunked prefill last-logits vs HF full forward at
1e-4; engine greedy == HF `generate`; batched decode with three prompt
lengths == each alone; guest window enforced at intake; hibernate/resume
exact across WARM → COLD → index rebuild; `decay` touches `gdn_state` only;
an imported tuned initial state reproduces HF's own seeded
`LinearAttentionLayer` cache path token for token; adapter validation
(PLE rejected, wrong layer rejected, partial adapters only with
`strict=False`).

### H2. Engine + store generalisation ✅

`Engine(model, layout, ...)` builds bank/runner from the layout;
`Engine.from_qwen35`; `Engine.import_state(name, path)`;
`StateStore.import_state` records `rule="import"` provenance (source, SHA-256,
tuned layers); `/v1/states/import` on the stdlib server; `--model-type
qwen35 --guest-ctx`.

### H3. Real-checkpoint smoke ✅ (numbers in `experiments/results/m4_qwen35_hybrid_smoke.md`)

```bash
PYTHONPATH=. python experiments/qwen35_hybrid_smoke.py \
  --json experiments/results/m4_qwen35_hybrid_smoke.json
```

Certifies on Qwen3.5-9B bf16 (engine on one A100, independent HF reference
on another): greedy parity vs HF `generate`; continuous batching of distinct
scene prompts vs alone; an RNN-StateTuning adapter imported as a handle and
generated from at batch, compared with the adapter's own runtime and scored
on the scene task (valid JSON / exact match, zero vs tuned state); COLD
resume of the imported handle; per-slot bytes and slot capacity.

### H4. Paged guest pool ✅

- `StateFamilySpec.page_tokens` marks a paged family; `ModelStateSpec`
  exposes `pages_for(num_tokens)`, `bytes_per_page`; `StateArena(spec,
  num_slots, num_pages)` allocates `slots[name] = (page ids...)` for paged
  families and frees/forks them; the scheduler admits with
  `arena.can_admit(pages=pages_for(prompt + remaining budget))` (FCFS,
  head-of-line blocks). Page 0 is the dummy read target of masked positions.
- `Qwen35StateBank` keeps K/V pools `[n_attn, P+1, kv_heads, page_tokens,
  head_dim]`; token `t` of a request lives at `pages[t // page_tokens]`,
  offset `t % page_tokens`. Gather materialises only the batch's resident
  tokens (`B x Lmax`) with one advanced-index op per layer; decode scatter is
  one fused write per layer. Snapshots are token-trimmed and page-size
  independent (the fingerprint excludes `page_tokens`).
- `Engine.from_qwen35(..., guest_pool_tokens, page_tokens)`; default pool is
  4096 tokens per slot, shared, so one request may take the whole pool.
- Acceptance: `tests/test_pages.py` (torch-free) and
  `tests/test_qwen35_hybrid_cpu.py` with 8-token pages so every prompt and
  decode crosses page boundaries; 9B smoke phase E: a 12k-token natural-text
  prompt through 49 pages, greedy-identical to HF.

### H5. Throughput floor ▶

- Gather-free paged attention (FlashInfer/FA3 paged decode) when the host
  has the kernels; until then the per-step `B x Lmax` gather is the cost of
  long contexts at high concurrency.
- Mask-free decode when all rows share a length (already taken), then
  length-bucketed batches to hit the SDPA fast path more often.
- CUDA-graph the GDN decode step (static shapes: `[B, H, K, V]`), attention
  eager — the SGLang out-graph/in-graph split.
- FLA kernels when the host has triton (`kernels`/`fla` names are what HF
  dispatches on; nothing in wkvm changes).

### H6. Second hybrid family ☐

Kimi-Linear / Qwen3-Next with MoE: MoE-by-dependency (fused-MoE kernels as
imports), same three-family contract plus a `mlp_only_layers` no-op.

## Effort

| Slice | Estimate | Proves |
|---|---:|---|
| H0–H4 (this checkout) | done | hybrid state contract, paged guests under exact admission, tuned-state import, exact hibernate/resume, batch-composition independence and 12k-token parity on the 9B |
| H5 throughput | 1–2 weeks | numbers comparable to the RWKV-7 M2 table |
| H6 second family | 1–2 weeks | the contract is not Qwen3.5-shaped |

## Risks

- HF Qwen3.5 internals move between releases (5.16 verified); the CPU test
  pins the cache calls and fails loudly on a contract change.
- bf16 batched matmuls can flip a greedy argmax vs single-row execution;
  the smoke reports prefix-match lengths rather than hiding it.
- Copying `B x Lmax` guest tokens per step caps decode throughput at high
  concurrency until H5's paged kernel.
- Lifetime reservation over-reserves for requests that stop early; a lazy
  page allocation with the same admission test is a small follow-up if pool
  pressure shows up in practice.
