# Native wkvm Engine Plan

Goal: replace the patched-HF Gemma recurrent-mode PoC with a native wkvm engine path that can be measured honestly against vLLM and SGLang on the same long-prompt concurrency workload.

## Decision

Do not fork vLLM or SGLang.

Use wkvm-native scheduler, arena, runner, and durable-state boundaries. Borrow narrow design patterns:

- From vLLM: no-phases token-budget scheduler shape, async scheduling direction, process split, CUDA graph dispatcher shape, tag-scoped allocator, hybrid page-byte unification.
- From SGLang: overlap planning ideas, retraction/new-token-ratio admission control, radix/LPM cache lessons, and the out-graph/in-graph graph-safety contract.

Reason: vLLM and SGLang both index reusable cache by token prefix. wkvm's differentiator is state as a first-class mutable object: fork, hibernate, resume, decay, merge, consolidate. Building inside either incumbent means fighting their primary cache invariant.

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

### N6. Head-to-Head Benchmark

Deliverable: native result artifacts for the README comparison.

Acceptance:

```bash
python experiments/native_gemma_bench.py \
  --ctx 13824 \
  --out 128 \
  --concurrency 1,8,16,32 \
  --mem-cap-gib 19 \
  --headroom-gib 1 \
  --json experiments/results/native_gemma_routed_span_concurrency.json &&
python - <<'PY'
import json
from pathlib import Path
d=json.loads(Path('experiments/results/native_gemma_routed_span_concurrency.json').read_text())
assert d['engine'] == 'wkvm-native'
assert d['context_tokens_per_session'] == 13824
assert any(r['success_count'] == r['B'] for r in d['rows'])
assert all(k in d['rows'][0] for k in ['p50_latency_s','p95_latency_s','agg_decode_tok_s','peak_reserved_gib'])
print('NATIVE_BENCH_OK')
PY
```

Benchmark report must include:

- Exact launch command and git commit.
- Model path and dtype.
- Prompt length, output length, concurrency.
- p50/p95 latency.
- Success/error counts.
- Aggregate decode tok/s and end-to-end output tok/s.
- GPU peak allocated/reserved/device-used memory.
- Whether each row is under the 19 GiB cap with 1 GiB headroom.
- Comparison rows for vLLM and SGLang must state when they are nearest existing full-KV runs rather than identical semantics.

### N7. README and Demo Update

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
| N1 cache structures | 3-5 days |
| N2 offline runner | 4-7 days |
| N3 scheduler integration | 3-5 days |
| N4 CUDA graph decode | 4-8 days |
| N5 serving endpoint | 2-4 days |
| N6 benchmark and comparison | 2-4 days |
| N7 README/video update | 1-2 days |

Total: about 3-5 focused weeks for a credible native single-model proof. The low end assumes reuse of HF module math for weights/layers while wkvm owns cache/state. The high end assumes replacing more of the model-forward path and fixing graph-capture edge cases.

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
