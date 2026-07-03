# Building an Inference Engine From Scratch: vLLM/SGLang Map + The Angle

*Produced 2026-07-03 from fresh main-branch clones of both repos (`/home/xiaol/X/vllm`, `/home/xiaol/X/sglang`). 15 agents, ~524k tokens of source reading; every decision-relevant claim spot-checked against code. Full structured data: `full_analysis.json` in this directory.*

---

## 0. Executive summary

**The bet: a 12–18k LOC Python/Triton *state-native* engine for RWKV-7 / GDN / Mamba2 hybrid models, whose primary allocation object is a fixed-size per-request state slot (not a paged KV block chain), with a "Durable State API" — named, versioned, forkable, exportable, MUTABLE state handles — as the differentiating product.**

Three verified facts make this the angle:

1. **White space**: neither vLLM nor SGLang contains a single RWKV model file (verified by grep — SGLang's only match is a chat-template name). Both ship GDN/Mamba hybrids (qwen3_next, kimi_linear, minimax_m2, falcon_h1), so the *category* is mainstream, but the RWKV family — your family — has no serious server.
2. **Structural commitment**: both engines are built on the invariant `state ≡ f(token-prefix)`. Their entire cache indexes (vLLM's chained block hashes, SGLang's radix keys) are content-addressed by token prefix. A *mutable* state handle (consolidated / decayed / merged memory — your multi-state online-memory research) cannot be represented in either index at all. That is a data-structure inversion they cannot flag their way into; it's the only capability in the whole angle set the incumbents structurally cannot follow.
3. **Physics**: an RWKV-7-class 7B model's full recurrent state is tens of MB — ~1000× smaller than long-context KV. Sessions become O(MB) objects you can hibernate, resume, fork, migrate, and store. Thousands of warm sessions on one GPU; sub-100ms resume; zero re-prefill.

The killed alternatives (and why): a Rust single-binary engine (llama.cpp/mistral.rs occupy the ground, zero comparative advantage), a local consumer-GPU product (llama.cpp already has RWKV-7 + recurrent-state slot save/restore), "batch-invariant determinism" as a headline (vLLM ships `batch_invariant.py`, SGLang ships `--enable-deterministic-inference`), and a plugin-ABI "architecture lab" as the *first* product (ABIs designed before serving three real architectures rot; sequence it after).

---

## 1. What the incumbents actually are (numbers, verified)

| | vLLM | SGLang |
|---|---|---|
| Total Python LOC | **724,262** (`vllm/`) + 90k csrc | **625,570** (`python/sglang/srt/`) |
| Model zoo | 287 files / 188k LOC | 202 files / 128k LOC |
| Attention backends | 42 (36 attn + 6 mamba enum members) | ~22 registered names over ~43 files |
| Quant methods | ~35 | ~33 |
| Platforms | 6 | 7 |
| Config surface | VllmConfig: 25 sub-configs, 12.2k LOC + 2.7k argparse | ServerArgs: **420 fields**, 7.8k LOC |
| Scheduler | `v1/core/sched/scheduler.py` 2,653 LOC | `managers/scheduler.py` **4,335 LOC**, 6 mixins, ~10 event-loop variants |
| Model runner | `gpu_model_runner.py` **7,594 LOC** (v2 rewrite: 13,256 LOC dir, experimental) | `model_runner.py` + runner/ split |
| Irreducible core | **~40k LOC** (~6% of repo) | similar; mem_cache alone is 47k LOC with ~10k of near-duplicate radix forks |

**Read: ~83–94% of both codebases is breadth (models × hardware × quant × features), not engine.** The engine ideas themselves fit in a few thousand lines — that's why a from-scratch build is viable and why nano-vllm exists.

### Architecture both engines converged on independently (treat as validated)

1. **2–3 process split**: tokenizer/detokenizer + HTTP in frontend process(es); the GPU busy loop speaks only token IDs over ZMQ + msgspec/msgpack. String work never stalls GPU dispatch.
2. **No-phases continuous batching**: one integer per request (`num_computed_tokens`) converging to `num_tokens` under a global token budget. Chunked prefill, decode, prefix-cache resume, and spec decode all fall out of one loop. (vLLM's cleanest artifact.)
3. **Overlap scheduling**: schedule step N+1 on CPU while GPU runs step N. vLLM: optimistic-advance + 75-line `async_scheduler.py` subclass. SGLang: dual CUDA stream + `FutureMap` GPU-resident token relay (never CPU-syncs on token values).
4. **Two-table indirection** (SGLang): `req_to_token` row per request + token-to-KV free list; the radix tree holds only index tensors. Makes cache, allocator, layout independently replaceable.
5. **Shape-keyed CUDA-graph dispatch**: enumerate every valid (mode, padded batch descriptor) at startup; runtime dispatch is a set lookup. Full graphs for uniform decode, eager/piecewise for ragged prefill = 90% of the win.
6. **Attention metadata-builder seam**: one `build(common_metadata) → backend_metadata` function per backend; the runner never knows kernel details. SGLang's refinement: the **out_graph/in_graph split** (host-side dynamic planning vs graph-recordable static ops) — the contract that makes any kernel CUDA-graph-safe by construction.

### How they differ where it matters for you (recurrent/hybrid state)

- **vLLM**: retrofits state onto the paged-KV machinery. `MambaSpec` in the same block-table world; page-byte unification (`attention_page_bytes == state_page_bytes`, grow/pad to match) lets ONE allocator serve both — the load-bearing trick. Three coexisting mamba cache modes (`none`/`align`/`all`); `all` mode = block-boundary state snapshots giving mamba prefix caching (`SupportsMambaPrefixCaching`, 7 model families and growing). The `align` mode is an acknowledged mess (null-block padding, delayed frees, per-step block relocation).
- **SGLang**: `mamba_radix_cache.py` bolts state checkpoints onto tree nodes ("deepest checkpoint wins", COW on forward stream) — one of ~6 near-duplicate radix-cache forks (SWA/mamba/hierarchical/unified…, ~10k LOC of duplication their own `unified_cache` work is trying to consolidate).
- **Both**: recurrent state is a second-class guest in a paged-KV house. Feature matrix by assert (mamba × DCP/cascade/async-sched constraints live as assert strings in 5+ files). Spec-decode state rollback: vLLM does N+1 state slots + copy-based rollback (steal this).

---

## 2. Steal list (the ideas that carry their weight)

**Scheduler/core** (mostly vLLM v1):
- No-phases token-budget loop; preempt-to-prefix-cache (no CPU swap bookkeeping); optimistic `num_computed_tokens` advance + rollback → async scheduling as a tiny subclass.
- Delta wire format (full request payload once, then block-id deltas only); scheduler is pure bookkeeping over ints and ids — **unit-testable without a GPU**.
- Intrusive doubly-linked LRU free list; lazy hash eviction (freed blocks keep cache entries until reallocated; free in reverse order so chain tails die first).
- Chained per-block hashes with extra-keys (LoRA, salt, mm) folded in; cap cache hits at n−1 so the last token always recomputes logits.
- Watermark headroom applied only to NEW admissions, never running decodes.

**Runner/graphs**:
- CudagraphDispatcher: startup-enumerated keys, runtime = set lookup; persistent max-sized GPU buffers sliced per step; numpy-on-pinned-memory input prep; capture largest shapes first to reuse the memory pool; reserved null slot/row 0 as padded-batch write sink.
- Model Runner V2 shape: fixed GPU-resident request-state table + per-step `idx_mapping` (batch_idx→state_idx) — but note V2 is already 13k LOC; copy the *shape*, not the scope.
- Two-phase execute/sample split (grammar bitmask + scheduler work overlap the forward).
- SGLang: BreakableCudaGraph mempool-pinning trick (piecewise graphs in ~500 LOC, no compiler); plan-stream/forward-stream metadata overlap; `needs_cpu_seq_lens` opt-out flag.

**Recurrent-state specifics** (vLLM hybrid subsystem):
- One tiny state-discovery interface: layer declares (shapes, dtypes) → engine builds spec/allocates/views. `MambaBase.get_state_shape` covers 6 layer families in 64 lines.
- Decode-first batch reordering → recurrent layers `torch.split` into [decode|prefill], run recurrent-update vs chunked-scan kernels with zero gather.
- Right-to-left "only need the LAST state" cache-hit scan; sparse checkpoint retention (one snapshot per interval); spec decode as N+1 state slots + fused batch-copy rollback.
- Same-step hit poisoning ("return pool_size+1 to defer one step") — 10-line fix for the state-not-yet-written race.

**Scheduling economics** (SGLang):
- `new_token_ratio` feedback controller + retraction (admit optimistically, retract youngest on KV exhaustion, re-estimate).
- Evictable-cache-bytes counted as schedulable budget, evicted lazily at allocation time.
- LPM queue sort with in-batch prefix dedup (~40 lines); FCFS fallback when queue >128.
- Chunked prefill committed to cache between chunks so chunk N+1 is just a prefix hit.

**Process/wire** (vLLM):
- msgspec structs (`array_like`, `gc=False`) + zero-copy tensor frames; single-byte type prefix; one generic utility-RPC (call_id → Future); sentinel-based liveness (ENGINE_CORE_DEAD, mp sentinels in the ZMQ poller).

**Features**:
- Grammar ABC with exactly 5 ops (fill_bitmask / accept / validate / rollback / is_terminated) — xgrammar only.
- Spec-decode rejection as pure accounting (decrement `num_computed_tokens`, overwrite slots next step — no free/compact path).
- LoRA coupled to the scheduler ONLY as a set-cardinality admission check.

**P/D & tiering** (later phases):
- Connector as "remote KV = async external prefix-cache hit"; pointer-exchange-once then integer-index-forever; compatibility hash in handshake; lease+heartbeat block lifetime. For a state engine, P/D transfer is ONE RDMA write of MB — trivially simpler than descriptor-list KV streaming.

## 3. Refuse list (where 600k of the 700k LOC comes from)

1. **The model zoo** (188k/128k LOC): hand-ported model definitions drive breadth in every layer beneath. You ship 3–5 model families, importer-based.
2. **The hardware × attention × quant matrix**: 42/22 attention backends, 35 quant methods, 6–7 platforms, platform conditionals in 357/276 files. One target (CUDA), 1–2 attention backends (FA3 or FlashInfer + your FLA kernels), quantization at load-time via one post-load hook.
3. **God-config threading**: 420-field ServerArgs / 25-sub-config VllmConfig passed into every constructor. Typed per-subsystem configs, frozen after startup.
4. **Feature cross-product in the hot loop**: SGLang's `_get_new_batch_prefill_raw` branches on dllm/mamba/SWA/LoRA/sessions/hicache inline; vLLM's runner has spec-decode × mamba × LoRA × connectors × pooling as inline branches of a 7.6k-line file. Strategy objects chosen at construction, or don't ship the feature.
5. **Fork-and-vendor / shim strata**: SGLang carries 124 files of drifted vLLM forks; vLLM carries 8k LOC of op-shims and alias modules. Version the API; break it.
6. **N cloned event loops** (SGLang: ~10) and subclass-per-family radix caches (~6): one loop parameterized by (admission source, result sink, overlap flag); one tree.
7. **70–100 field batch dataclasses** where filter/merge hand-enumerate fields: schema-driven SoA or per-feature sub-structs from day one.
8. **Union-of-modes metadata** (20+ Optional tensor fields): per-mode typed metadata.

---

## 4. The nine candidate angles and verdicts (adversarial critic, evidence-checked)

| # | Angle | Lens | Score | Verdict |
|---|---|---|---|---|
| 1 | **StateVM** — state checkpoint store as the memory core, not a block pool | research | 7 | **keep** → substrate |
| 2 | **Durable State** — named/forkable/MUTABLE state handles, O(MB) migration | research | 7 | **keep** → the moat |
| 3 | Architecture Lab — bring-your-own-token-mixer ABI + checkpoint→endpoint | research | 5 | merge (month ~9 extension story; ABI-first is premature) |
| 4 | Fixed-footprint engine — state slots as primary allocation | systems | 7 | merge into #1 (it IS #1's engineering plan; its "refuse full attention" purism is the one fatal scoping error — Qwen3-Next hybrid layers are FULL attention, not SWA) |
| 5 | Single-binary Rust engine | systems | 3 | **kill** (llama.cpp/mistral.rs/TRT-LLM occupy every side; vLLM shipping its own Rust frontend) |
| 6 | GPU-autonomous bitwise-deterministic decode for RL | systems | 4 | merge (vLLM `batch_invariant.py` + SGLang `--enable-deterministic-inference` already exist; the survivor: **same-FLA-kernels-in-train-and-serve** → free trainer/sampler logprob parity, which SGLang's docs admit they can't give) |
| 7 | StateServe — warm-state engine for massive idle-session fleets | niche | 6 | merge → headline benchmark ("N-thousand warm sessions, sub-100ms resume, zero re-prefill") |
| 8 | ForkServe — branching-native rollout engine (fork/rollback wire API) | niche | 4 | merge (SGLang has had `fork()` since its founding paper and token-granular radix; survivor: fork/rollback of *recurrent* state + "sampler/grammar state must be clonable" as a day-one interface rule) |
| 9 | PocketCtx — million-token consumer-GPU local engine | niche | 3 | **kill** (llama.cpp has RWKV-7 + `llama_memory_recurrent` + slot save/restore; keep only the flat-memory-vs-context demo chart as marketing) |

**Critic's recommended portfolio (verbatim summary)**: One engine, one bet, three layers — StateVM/fixed-footprint as substrate, Durable State API as moat, StateServe framing as go-to-market; Architecture-Lab importer + parity harness and deterministic-RL mode as later features. Include a *deliberately dumb full-attention guest allocator* so Qwen3-Next/Kimi-class hybrids are servable day one. The moat claim to defend: mutable state handles violate `state≡f(prefix)` — the invariant under incumbents' entire cache indexes. Plain state *checkpointing* is NOT a moat (vLLM `all` mode ships it today for 7 families).

---

## 5. The engine (concrete shape)

**Name-shaped summary**: state-native serving for linear/hybrid models + durable state handles.

**Layer 1 — substrate (months 0–3, ~8–12k LOC)**
- Dense per-layer-family state arenas: `[slots, ...]` tensors; a request = one slot index per family. Admission = counting free slots (exact, no watermark math, no fragmentation).
- Full-attention guest allocator for hybrid models: simple paged pool, page-bytes-unified with state pages (steal vLLM's unification trick). Deliberately dumb; it's a guest, not the core.
- No-phases scheduler (vLLM invariant) + overlap scheduling from day one (bounded future queue; retrofitting it is what created SGLang's contortions).
- Whole-step decode CUDA graphs (uniform fixed-footprint decode batches make these trivially safe); eager or single-bucket prefill first.
- Kernels: FLA (chunked scan for prefill, recurrent update for decode — your kernels), FA3/FlashInfer for guest attention layers. Sampling: temperature/top-p + seeds. Models: RWKV-7 first, then one GDN hybrid (qwen3_next) to force the guest-allocator path honest.
- 3-process split OR single process with tokenizer thread — but keep the token-ids-only engine boundary either way.

**Layer 2 — the moat (months 3–6)**
- StateStore: position-indexed checkpoint store (boundary snapshots every N tokens), chained-prefix-hash addressed, spilling GPU→pinned host→NVMe. "Prefix cache hit" = deepest checkpoint ≤ prompt length.
- `/v1/states` API: create-from-prompt, get/save/load (safetensors), **fork** (lazy COW slot copy — O(MB)), pin, delete; sessions = named states with hibernate/resume; rollback-to-token via checkpoint cadence.
- **Mutation hooks**: registered state-update rules (decay, merge, consolidate) — your multi-state online-memory research as a serving primitive. This is the part with no incumbent answer.
- Day-one interface rule inherited from ForkServe's autopsy: per-request sampler/grammar state must be clonable.

**Layer 3 — proof and reach (months 6–9)**
- Demos lead the roadmap, not trail it: (a) persistent agent surviving engine restart; (b) corpus-state artifact forked to 1k concurrent requests; (c) live O(MB) request migration between GPUs; (d) N-thousand warm sessions on one GPU with sub-100ms NVMe resume; (e) flat tok/s + flat memory vs context length on a 4090; (f) bitwise on-policy RWKV RL (same FLA kernels in trainer and server → logprob parity).
- Then: checkpoint→endpoint importer + logit-parity harness (engine-vs-training per-layer divergence) as the FLA-community wedge; EAGLE-style spec decode via N+1 state slots if it earns its complexity.

**Design decisions to make on day one** (from the completeness pass — retrofit-hostile):
1. **Tag-scoped memory allocator** (weights / states / KV pools separately tagged, CUDA VMM) so sleep/wake + in-place weight update for RL rollout work later — vLLM's `CuMemAllocator` (377 LOC) is the reference. If the engine ever serves as an RL rollout worker, this cannot be retrofitted.
2. **TP seam**: pick parallel-layer-classes (Megatron-style Column/RowParallel + per-param `weight_loader`) vs manual sharding before writing model code. Both incumbents converged on the former; for TP≤8 with 3 model families, manual sharding in the importer is defensible and much smaller.
3. **Weight-loading hook**: one `process_weights_after_loading` seam — it's where quantization actually couples to an engine (load-time repack), not runtime dispatch.
4. **Metrics ride the output wire** (scheduler stats piggybacked per-step, aggregation frontend-side) — both engines converged; never scrape in-process.
5. One CUDA-graph strategy, one grammar backend (xgrammar), one transport (when P/D comes), one radix/checkpoint tree. Refuse mode multiplicity structurally.

**Kill risks (monitor, quarterly)**
1. Linear/hybrid adoption reversing among frontier labs (MiniMax M2 walked back to full attention).
2. vLLM extending mamba `all`-mode + CPU offload to GDN faster than your durable-handle demos land. Their ceiling is still `state≡f(prefix)` — but plain checkpointing/session-resume value gets absorbed. Hence: mutation, fork, migration, and RL-parity demos are the roadmap spine, in that order.

---

## 6. Reading list (now scoped by the map)

**vLLM** (read against your own build, in this order):
- `vllm/v1/core/sched/scheduler.py` (2,653) + `sched/output.py` (267) + `async_scheduler.py` (75) — the loop + wire format + overlap trick.
- `vllm/v1/core/kv_cache_utils.py` (2,229) — hashing + free-list; `block_pool.py` (723).
- `vllm/v1/kv_cache_interface.py` (944) + `single_type_kv_cache_manager.py` (1,560) — the spec abstraction; MambaManager's right-to-left scan.
- `vllm/model_executor/layers/mamba/` — `MambaBase` state discovery; `vllm/v1/worker/gpu/` (V2) — request-state table + idx_mapping shape.
- `vllm/v1/engine/core.py` (2,303) — EngineCoreProc thread layout; `vllm/device_allocator/cumem.py` (377) — tag-scoped allocator.
- `vllm/v1/cudagraph_dispatcher.py` (350).

**SGLang**:
- `srt/mem_cache/radix_cache.py` (830) — the original; refuse the forks. `mamba_radix_cache.py` — checkpoint-on-node "deepest wins".
- `srt/managers/scheduler.py` — event_loop_overlap + FutureMap relay (`overlap_utils.py`) + retraction/new_token_ratio.
- `srt/model_executor/runner/` — capture-backend ABC, BreakableCudaGraph, ShapeKey; `layers/attention/` — out_graph/in_graph contract, hybrid-by-composition wrapper.
- `srt/disaggregation/base/conn.py` (221) — the transfer ABC to imitate (later).

**Elsewhere**: nano-vllm (GeeeekExplorer) for the 1.2k-line skeleton; flex-nano-vllm for FlexAttention paging; llama.cpp `llama_memory_recurrent` (know thy neighbor); FLA library (kernels you already know); RWKV-Infer (what the community has vs needs).
