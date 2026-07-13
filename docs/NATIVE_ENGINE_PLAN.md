# Native wkvm Engine Plan

Goal: replace the patched-HF Gemma recurrent-mode PoC with a native wkvm engine path that can be measured honestly against vLLM and SGLang on the same long-prompt concurrency workload.

## Decision

Do not fork vLLM or SGLang.

Use wkvm-native scheduler, arena, runner, and durable-state boundaries. Borrow narrow design patterns:

- From vLLM: no-phases token-budget scheduler shape, async scheduling direction, process split, CUDA graph dispatcher shape, tag-scoped allocator, hybrid page-byte unification.
- From SGLang: overlap planning ideas, retraction/new-token-ratio admission control, radix/LPM cache lessons, and the out-graph/in-graph graph-safety contract.

Reason: vLLM and SGLang both index reusable cache by token prefix. wkvm's differentiator is state as a first-class mutable object: fork, hibernate, resume, decay, merge, consolidate. Building inside either incumbent means fighting their primary cache invariant.

## Audit Basis

This plan is based on a local source audit of:

- vLLM `ec0ffaa`: scheduler, KV cache/block pool, GPU runner, CUDA graph dispatcher, engine core, OpenAI API server, and metrics stats.
- SGLang `9588cac`: scheduler and schedule batches, radix cache, memory pools, model runner, CUDA graph backend interface, HTTP server, tokenizer manager, and metrics reporter.

The audit does not treat either project as a single "Python script". Both production engines are Python orchestration around optimized CUDA/Triton/C++ kernels, static buffers, graph capture, worker processes, HTTP frontends, and metrics. wkvm-native should use the same implementation model: Python orchestration for admission/control flow, with GPU kernels and static CUDA buffers for the hot path. A C++ rewrite is not the next step.

## vLLM/SGLang Component Audit

| Component | vLLM component shape | SGLang component shape | wkvm-native implication |
|---|---|---|---|
| Frontend and tokenizer boundary | OpenAI API server plus engine client/process boundary. | FastAPI/uvloop HTTP server, tokenizer manager, OpenAI/Anthropic/Ollama-style surfaces, health and metrics wiring. | Keep tokenizer outside the GPU loop. Start with a documented token-id or narrow OpenAI-compatible endpoint, then harden health/auth/streaming. |
| Scheduler/admission | `v1/core/sched/scheduler.py` is a no-phase token-budget scheduler: running requests first, chunked prefill, prefix cache, speculative decode, connectors, Mamba alignment. | `srt/managers/scheduler.py` plus `ScheduleBatch` handles overlap, prefill/decode scheduling, retraction, disaggregation, policy controls, and watchdog behavior. | Borrow the no-phase token-budget shape and SGLang-style retraction/admission controls. Do not import their feature cross-product. |
| Cache/memory object | `KVCacheBlock`, `FreeKVCacheBlockQueue`, and `BlockPool` manage ref-counted token-prefix blocks and prefix-hash lookup. | `RadixCache` handles prefix reuse; `ReqToTokenPool` and token-to-KV pools provide request-row to physical-KV indirection and graph padding rows. | Build a wkvm `StateSlotArena`, not a token-prefix block cache. Steal the allocator discipline, refcounts, dummy rows, and Req-to-state indirection, but make the identity a mutable state slot/span bank/ring lineage. |
| Runner hot path | `GPUModelRunner` is a large production runner with model loading, attention backends, LoRA, KV connectors, speculative decode, CUDA graphs, pooling, distributed handling, and sampling. | `ModelRunner` owns model loading, attention backend selection, memory pool config, graph runners, quantization, distributed modes, sampling, canary/update hooks. | First runner should be narrow: `GemmaRoutedSpanRunner` only, bf16, one GPU, greedy first. General model breadth is a later product, not part of the proof. |
| CUDA graph execution | `CudagraphDispatcher` enumerates valid graph keys and pads batch sizes before capture/replay. | `BaseCudaGraphBackend` gives a clean `capture_session`, `capture_one`, `can_run`, `replay_session`, `replay`, `cleanup` contract. | Copy the shape-key discipline: explicit graph buckets, static addresses, masked dummy rows, eager-vs-graph token parity tests. |
| Metrics and observability | `SchedulerStats`, `RequestStateStats`, prefix-cache stats, KV eviction events, graph stats, performance stats. | Prometheus middleware, scheduler metrics reporter, forward-pass metrics, cache/event metrics, idle logging. | Benchmark-grade wkvm needs first-class p50/p95, queue time, prefill/decode throughput, state-slot occupancy, graph hit rate, errors, and GPU memory. README/video numbers should come only from these artifacts. |
| Process and resilience | Engine core process, executor abstraction, ready handshake, failure callback, IPC, batch queue. | Multi-process scheduler/server layout, watchdog paths, warmup, health checks, metrics endpoint, request receiver/output streamer. | Minimal proof can be single process; production subset needs separate frontend/engine process, health checks, watchdog, backpressure, graceful cancellation, and soak tests. |
| Model breadth | Many model families and feature modes. | Many model families and serving modes. | Explicitly out of scope. Supporting Gemma routed-span well is the proof; becoming a vLLM/SGLang-class general engine is a separate multi-quarter project. |

## Component Decisions

- Do not build wkvm-native by forking vLLM or SGLang. Their reusable-cache identity is token-prefix based; wkvm's useful identity is durable mutable state.
- Borrow architecture, not code bulk: vLLM's no-phase scheduler and graph dispatcher shape; SGLang's graph backend interface, request-to-token indirection, dummy padding row, retraction/backpressure ideas, and Prometheus-style metrics.
- Make the core object a state slot: request row -> state slot -> ring/span-bank/pending-span buffers -> lineage/hibernation metadata. Prefix tokens are inputs to state construction, not the durable cache key.
- Keep the first native runner narrow enough to finish: Gemma-4-E4B-it routed-span on one CUDA GPU. Add OpenAI-compatible polish only after the engine boundary is stable.

## Target Artifact

A native single-model Gemma routed-span runner and minimal serving path:

- `GemmaRoutedSpanRunner` executes Gemma-4-E4B-it without HF `DynamicCache` patches.
- wkvm owns the ring, span-bank, pending-span, valid-mask, and position buffers.
- Existing wkvm scheduler admits distinct long-prompt requests as normal requests.
- A benchmark script reports p50/p95 latency, success/error counts, aggregate throughput, resident sessions, and GPU memory.
- README comparison table can be updated from native wkvm results, not HF PoC results.

This is not a general model zoo. It is a narrow proof that the recurrent-mode memory shape can live inside wkvm's actual engine contract.

## Scope

In scope:

- Gemma-4-E4B-it only.
- Greedy decode first; temperature/top-p can reuse existing sampler after the core path works.
- CUDA, one GPU, bf16.
- Long-context benchmark targets:
  - 13,824 input tokens + 128 output tokens for concurrency.
  - 13,824 input tokens + 512 output tokens for single long-output quality/latency.
  - B = 1, 8, 16, 32, optionally 48 if memory allows.
- Routed-span mode: sink16 + ring1024 + value-routed span bank m64, sentence-punctuation span breaks.
- Minimal HTTP or token-stream endpoint sufficient to prove engine behavior.

Out of scope for this milestone:

- General transformer model support.
- vLLM-compatible feature breadth: LoRA, grammar, speculative decode, multimodal, distributed serving.
- Full-KV exact semantics for routed-span mode.
- Learned routers.
- Multi-GPU tensor parallelism.

## Milestones

### N0. Extract Gemma Execution Contract

Deliverable: a short code/design note listing Gemma-4-E4B-it layer types, KV-owning layers, sliding/full attention behavior, KV-sharing behavior, position-id rules, and mask requirements.

Acceptance:

```bash
test -f docs/gemma_native_contract.md &&
grep -q 'KV-owning' docs/gemma_native_contract.md &&
grep -q 'position' docs/gemma_native_contract.md &&
grep -q 'routed-span' docs/gemma_native_contract.md
```

Notes:

- Start from `experiments/gemma_recurrent_poc.py`; do not re-discover the algorithm from scratch.
- Record which HF behavior is still being relied on during the transition.

### N1. Native Cache Data Structures

Deliverable: wkvm-owned cache/state classes for Gemma routed-span:

- Sliding/ring cache for bounded local layers.
- Routed span bank for global KV-owning layers.
- Pending span buffer and span-boundary bookkeeping.
- Padded valid-mask representation for distinct-cache batching.
- Memory accounting hooks.
- Req-to-state indirection and dummy/padded rows for graph-safe batching.

Acceptance:

```bash
python -m pytest tests/test_gemma_routed_cache.py -q
```

Required tests:

- Ring capacity is constant after long prefill.
- Sink tokens remain stable.
- Span routing is value-based, not RoPE-key-based.
- Padded valid masks hide pad slots.
- Memory estimate is monotonic with resident sessions and bounded with context length.

### N2. Offline `GemmaRoutedSpanRunner`

Deliverable: `GemmaRoutedSpanRunner` with prefill and decode methods matching wkvm's runner contract.

Acceptance:

```bash
HF_HUB_OFFLINE=1 python experiments/native_gemma_smoke.py --ctx 2048 --out 32 --batch 2 | grep -q NATIVE_GEMMA_SMOKE_OK
```

Requirements:

- No HF `DynamicCache` replacement classes in the hot path.
- No CPU-staged independent-prefill merge for normal request execution.
- Short-context token parity check against HF full-KV or current PoC for a fixed greedy prompt.
- Routed-span recall smoke includes the known `BLUE-742`, `Samarkand`, `lantern` facts.

### N3. Scheduler Integration

Deliverable: native Gemma runner plugs into `wkvm.engine.Engine` or a transformer sibling with the same scheduler semantics.

Acceptance:

```bash
python experiments/native_gemma_engine_smoke.py --ctx 4096 --out 64 --concurrency 4 | grep -q NATIVE_ENGINE_SMOKE_OK
```

Requirements:

- Requests are admitted by wkvm scheduler/arena, not a bespoke benchmark loop.
- Distinct prompts are first-class requests.
- Running decode batches can contain requests with different prompt histories.
- Report success/error counts.
- Admission must expose queue depth, runnable rows, resident state slots, and retraction/backpressure decisions.

### N4. CUDA Graph Decode Path

Deliverable: fixed-address decode buffers and graph replay for routed-span decode.

Acceptance:

```bash
python experiments/native_gemma_graph_check.py --ctx 4096 --out 64 --batch 8 | grep -q GRAPH_TOKEN_PARITY_OK
```

Requirements:

- Eager-vs-graph greedy tokens match for the check prompt.
- Graph bucket keys are explicit and enumerable.
- Padded rows write into safe sink rows or masked slots.
- No per-token CPU sync on the greedy path except benchmark timing boundaries.
- Graph backend API follows the SGLang-style capture/replay interface, while graph bucket dispatch follows vLLM-style explicit valid keys.

### N5. Minimal Serving Endpoint

Deliverable: a native wkvm endpoint for generation, either OpenAI-compatible enough for benchmark clients or a documented `/v1/generate` token-id endpoint.

Acceptance:

```bash
python experiments/native_gemma_server_smoke.py --ctx 2048 --out 32 --concurrency 4 | grep -q SERVER_SMOKE_OK
```

Requirements:

- Engine boundary speaks token IDs.
- Tokenizer/detokenizer is outside the GPU busy loop.
- Streaming or stepwise token output is supported.
- Per-request metrics are surfaced: latency, output tokens, finish reason, errors.
- Endpoint can be narrow, but must have health, cancellation, and bounded queue behavior.

### N6. Observability and Ops Floor

Deliverable: metrics and failure handling sufficient to trust head-to-head benchmark output.

Acceptance:

```bash
python experiments/native_gemma_metrics_smoke.py --ctx 2048 --out 32 --concurrency 4 | grep -q NATIVE_METRICS_OK
```

Requirements:

- Per-request: enqueue time, prefill time, decode time, first-token latency, total latency, output tokens, finish reason, error.
- Per-step: scheduled rows, token budget used, runnable/waiting counts, graph bucket, graph hit/miss, GPU memory allocated/reserved/device-used.
- Per-state: resident slots, ring tokens, span-bank occupancy, pending-span occupancy, evictions/retractions.
- Process/server: health endpoint, startup readiness, controlled shutdown, exception reporting.

### N7. Head-to-Head Benchmark

Deliverable: native result artifacts for the README comparison.

Acceptance:

```bash
python experiments/native_gemma_bench.py \
  --ctx 13824 \
  --out 128 \
  --concurrency 1,8,16,32 \
  --prompt-lengths staggered \
  --synthetic-prompts \
  --mem-cap-gib 19 \
  --headroom-gib 1 \
  --decode-microbatch-rows 16 \
  --route-chunk 512 \
  --persistent-padded-decode-steps 8 \
  --persistent-padded-decode-cuda-graph \
  --persistent-padded-decode-graph-warmup-iters 1 \
  --use-native-gemma-forward \
  --native-gemma-checkpoint-loader \
  --native-gemma-attention-backend sdpa_single_gqa \
  --native-gemma-projection-backend separate \
  --require-native-no-hf \
  --json experiments/results/native_gemma_routed_span_concurrency.json &&
python - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('experiments/results/native_gemma_routed_span_concurrency.json').read_text())
assert d['engine'] == 'wkvm-native'
assert d['context_tokens_per_session'] == 13824
assert d['prompt_token_source'] == 'synthetic'
assert d['uses_hf_tokenizer'] is False
assert d['uses_hf_config'] is False
assert d['native_gemma_config_loader'] is True
assert d['native_gemma_checkpoint_loader'] is True
assert d['native_no_hf_requirement']['passed'] is True
assert any(r['success_count'] == r['B'] for r in d['rows'])
assert all(k in d['rows'][0] for k in [
    'p50_latency_s',
    'p95_latency_s',
    'agg_decode_tok_s',
    'peak_reserved_gib',
    'prompt_token_ids_sha256',
])
print('NATIVE_BENCH_OK')
PY
```

Benchmark report must include:

- Exact launch command and git commit.
- Model path and dtype.
- Prompt length, output length, concurrency, prompt-token source, and exact prompt-token fingerprint.
- p50/p95 latency.
- Queue time, first-token latency, prefill time, decode time, finish reason, and errors when available.
- Success/error counts.
- Aggregate decode tok/s and end-to-end output tok/s.
- GPU peak allocated/reserved/device-used memory.
- Native setup provenance: `uses_hf_tokenizer`, `uses_hf_config`, `native_gemma_config_loader`, `uses_hf_model_construction`, `uses_hf_transformer_forward`, `native_gemma_checkpoint_loader`, and `native_no_hf_requirement`.
- Scheduler/engine evidence: max waiting/running/runnable rows, resident state slots, backpressure/retraction counts, graph capture/cache-hit/replay/skip counts, graph-shape reuse/mismatch counts.
- Whether each row is under the 19 GiB cap with 1 GiB headroom.
- Comparison rows for vLLM and SGLang must state when they are nearest existing full-KV runs rather than identical semantics.
- Strict same-workload reports should be rendered with `experiments/gemma_bench_report.py --require-same-shape --require-same-prompt-fingerprint --require-native-no-hf --require-comparable`.
- Strict HTTP reports additionally require `--require-comparable`, a complete shared concurrency ladder, explicit semantic labels, disjoint prompt rows, exact output-token accounting, and identical request/warmup/sampling policies.

### N8. README and Demo Update

Deliverable: README table and video labels use native wkvm numbers.

Acceptance:

```bash
grep -q 'wkvm-native' README.md &&
grep -q 'native_gemma_routed_span_concurrency.json' README.md &&
test -f experiments/results/gemma_routed_span_demo.mp4
```

Requirements:

- Replace or clearly demote patched-HF PoC rows.
- Preserve the caveat: routed-span is approximate recurrent mode, not full-KV semantics.
- If the demo video is updated, labels must say "native wkvm" only for native results.

## Expected Effort

| Work package | Estimate |
|---|---:|
| N0 contract | 0.5-1 day |
| N1 cache/state arena | 4-7 days |
| N2 offline runner | 5-9 days |
| N3 scheduler integration | 4-7 days |
| N4 CUDA graph decode | 5-10 days |
| N5 serving endpoint | 3-6 days |
| N6 observability/ops floor | 2-4 days |
| N7 benchmark and comparison | 2-4 days |
| N8 README/video update | 1-2 days |

Updated estimate:

| Target | Estimate | What it proves |
|---|---:|---|
| Native single-model proof | 4-6 focused weeks | Gemma routed-span runs without HF `DynamicCache` patching, uses wkvm-owned state/cache, and can complete offline plus scheduler smoke tests. |
| Benchmark-grade native engine | 6-10 weeks | Adds graph buckets, serving boundary, metrics, p50/p95, memory accounting, and head-to-head artifacts that can be honestly compared with vLLM/SGLang. |
| Production-grade wkvm subset | 3-5 months | Adds process split, health/auth/cancellation, watchdog/restart behavior, soak tests, backpressure, operational metrics, packaging, and enough failure handling for real users. |
| vLLM/SGLang-class general engine | 9-18+ months | Broad model support, distributed modes, LoRA/spec decode/grammar/multimodal/quantization breadth, compatibility surfaces, and production operations. This is not the goal. |

The previous 3-5 week estimate was reasonable only for a narrow proof that reuses HF module math while wkvm owns cache/state. After auditing vLLM and SGLang, the credible benchmark-grade target is 6-10 weeks because graph capture, metrics, server/process edges, and honest comparison artifacts are real work.

## Risks

- HF Gemma internals hide cache/mask behavior that takes time to reproduce exactly.
- Routed-span padded distinct-cache batching can force per-layer masks if layer layouts diverge.
- CUDA graph capture may expose hidden dynamic allocation or shape drift.
- Full-KV engines will remain faster for B=1 exact generation; the native proof must emphasize resident long-context concurrency and memory shape.
- Quality claims must stay tied to existing recall/NLL artifacts until native results reproduce them.

## First Implementation Slice

Build N0-N2 before touching the server:

1. Write `docs/gemma_native_contract.md`.
2. Add cache data classes and CPU/GPU unit tests.
3. Implement `experiments/native_gemma_smoke.py`.
4. Prove a 2048-token smoke prompt and one routed-span recall prompt.

Only after that should the runner be wired into scheduler/server code. This keeps the first failure domain small: cache semantics before serving mechanics.
