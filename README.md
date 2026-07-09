# wkvm

**A hypervisor for model state.** State-native inference engine for RWKV-7 / GDN / Mamba2 and hybrid-linear models — where the primary allocation object is a fixed-size per-request **state slot**, not a paged KV block chain.

> WKV + KVM: states are the VMs — create, snapshot, fork, hibernate, resume, live-migrate. The engine is the hypervisor.

## Routed-Span Demo

[![Gemma routed-span recurrent-mode demo](experiments/results/gemma_routed_span_demo.gif)](experiments/results/gemma_routed_span_demo.mp4)

Full-quality MP4: [`experiments/results/gemma_routed_span_demo.mp4`](experiments/results/gemma_routed_span_demo.mp4)

Previous ring/concurrency demo: [`experiments/results/gemma_wkvm_style_demo.mp4`](experiments/results/gemma_wkvm_style_demo.mp4)

## Routed-span vs vLLM/SGLang

Gemma-4-E4B-it on one RTX 4090. vLLM and SGLang are full-KV transformer engines; wkvm routed-span is approximate recurrent mode (`sink16 + ring1024 + routed span bank m64`), so this is a memory/throughput comparison, not a same-semantics quality claim. Rows labelled `wkvm-native` use the native wkvm scheduler/arena/runner/server boundary, but still call HF Gemma model math; older rows labelled PoC are retained only as historical context.

**Single long prompt + long output**: 13,824-token prompt + 512-token output, greedy decode, `ignore_eos=True`.

| engine | semantics | facts recovered | prefill+1st | full wall | decode tok/s | e2e output tok/s | memory observed | raw result |
|---|---|---:|---:|---:|---:|---:|---|---|
| wkvm routed-span m64 | approximate recurrent | yes | 1.380s | 11.237s | 51.8 | 45.6 | 14.67 GiB reserved; 52.9 MiB cache | [`json`](experiments/results/long_gen_13824_512_wkvm_routed_span_m64.json) |
| vLLM 0.24.0 | full KV | yes | 1.813s | 8.251s | 79.4 | 62.1 | 22.54 GiB device used; 18.42 GiB alloc | [`json`](experiments/results/long_gen_13824_512_vllm.json) |
| SGLang 0.5.14 | full KV | yes | 1.257s | 8.515s | 70.4 | 60.1 | 21.79 GiB peak device | [`json`](experiments/results/long_gen_13824_512_sglang.json) |

**Distinct long-prompt concurrency**: the `wkvm-native` row is the fresh native engine run at 13,824 context tokens/session and 128 decode tokens/session. The older PoC row is faster because it used a specialized resident-decode harness, not the native scheduler/server contract. vLLM/SGLang rows are the nearest tracked full-KV engine capacity runs at 16,384 context tokens/session and 128 decode tokens/session, included to anchor the incumbent memory shape.

| engine | workload | resident sessions | aggregate decode | memory/capacity note | latency note | source |
|---|---|---:|---:|---|---|---|
| wkvm-native routed-span m64 | 13,824 ctx, distinct prompts | **32 green** | **57.9 tok/s** at B=32 | 17.96 GiB reserved; green means 19 GiB cap with 1 GiB headroom; byte-capped padded decode | p50 72.935s, p95 74.122s at B=32 | [`json`](experiments/results/native_gemma_bytecap_ctx13824_out128_b32_370m.json), [`frontier`](experiments/results/native_gemma_throughput_frontier.md) |
| HF Transformers full-KV | 13,824 ctx, distinct prompts | **2 green**; B=4 over headroom | **52.6 tok/s** green at B=2; 86.2 tok/s over headroom at B=4 | 16.88 GiB reserved at B=2; 20.31 GiB at B=4 | p50/p95 7.457s at B=2; 12.088s at B=4 | [`json`](experiments/results/hf_gemma_batched_chunked_ctx13824_out128_ladder.json), [`frontier`](experiments/results/native_gemma_throughput_frontier.md) |
| wkvm routed-span m64 PoC | 13,824 ctx, distinct prompts | **16 green**; 32/48 completed over headroom | **643.9 tok/s** green; 1039.4 tok/s over headroom | 15.97 GiB reserved, 913 MiB routed cache; specialized resident-decode harness | p50=p95 3.181s decode | [`json`](experiments/results/gemma_routed_span_distinct_concurrency.json) |
| vLLM 0.24.0 | nearest 16,384 ctx full-KV run | 9 cap; N=8 measured | 285.6 tok/s | 21.81 GiB device used; 18.26 GiB alloc | wall 13.42s at N=8; p50/p95 not recorded | [`bench`](experiments/results/bench_vllm_gemma4e4b.md) |
| SGLang 0.5.14 | nearest 16,384 ctx full-KV run | 1 true concurrent; N=8 queue-limited | 68 tok/s | 25,360-token KV pool on this stack | queue-limited; p50/p95 not recorded | [`bench`](experiments/results/bench_sglang_gemma4e4b.md) |

Readout: routed-span is slower for one exact long generation than vLLM/SGLang full-KV, and the current native engine is only modestly faster than green HF Transformers on aggregate decode (**57.9 vs 52.6 tok/s**) while supporting many more resident long sessions (**32 vs 2**). The measured advantage is bounded-memory long-context concurrency when approximate recurrent semantics are acceptable; it is not a replacement for full-KV serving when exact transformer behavior is required.

### Serving-path benchmark

The direct native benchmark is useful for engine throughput, but vLLM/SGLang production comparisons should use the server path. `wkvm.gemma_server` exposes OpenAI-compatible token-id `/v1/completions`, token-id `/v1/stream` SSE events, blocking `/v1/generate`, async-style `/v1/submit` + `/v1/status/<id>`, `/v1/cancel`, `/health`, and `/metrics`. The server now bounds retained completed-request metadata, marks model-step failures as `FINISHED_ERROR`, and supports request timeout cancellation.

```bash
python -m wkvm.gemma_server --model /path/to/gemma-4-E4B-it --slots 32 --port 8000 \
  --max-queue 128 --request-timeout-s 600 --max-completed-requests 4096

python experiments/wkvm_serving_bench.py --backend openai-completions \
  --engine wkvm-native-openai-completions --url http://127.0.0.1:8000 \
  --served-model gemma-4-E4B-it --ctx 13824 --out 128 \
  --concurrency 1,2,4,8,16,32 \
  --json experiments/results/wkvm_serving_ctx13824_out128.json

python experiments/wkvm_serving_bench.py --backend openai-completions \
  --engine vllm-http-stream --url http://127.0.0.1:8001 \
  --served-model gemma-4-E4B-it --ctx 13824 --out 128 \
  --concurrency 1,2,4,8,16,32 \
  --json experiments/results/vllm_serving_ctx13824_out128.json

python experiments/wkvm_serving_bench.py --backend openai-completions \
  --engine sglang-http-stream --url http://127.0.0.1:8002 \
  --served-model gemma-4-E4B-it --ctx 13824 --out 128 \
  --concurrency 1,2,4,8,16,32 \
  --json experiments/results/sglang_serving_ctx13824_out128.json

python experiments/gemma_bench_report.py experiments/results/wkvm_serving_ctx13824_out128.json \
  experiments/results/vllm_serving_ctx13824_out128.json \
  experiments/results/sglang_serving_ctx13824_out128.json \
  --out experiments/results/serving_compare_ctx13824_out128.md
```

This records TTFT, ITL, end-to-end latency, success/error counts, and HTTP-stream output throughput. Fair rows must keep the same `ctx`, `out`, prompt-length mode, greedy/ignore-eos behavior, and concurrency ladder. The OpenAI-compatible path sends token-id prompts to `/v1/completions`; wkvm and vLLM can return streamed `token_ids`, while SGLang currently relies on usage/text/logprob fields unless its server exposes token IDs. This is the correct next comparison shape for vLLM/SGLang serving benchmarks; it still does not remove the deeper gaps of full non-HF Gemma kernel ownership, CUDA graphs/static buffers for the Gemma hot path, and broader OpenAI compatibility.

## Why

For linear/hybrid models, per-request memory is **constant and tiny** (an RWKV-7 7B state is ~20MB — ~1000× smaller than long-context KV). Built around that physics instead of paged KV, an engine gets:

- **Exact admission** — scheduling is counting free slots; no fragmentation, no watermark math.
- **Uniform decode batches** — whole-step CUDA graphs, Albatross-class throughput scaling.
- **Sessions as objects** — hibernate/resume in one transfer, fork in one slot copy, migrate in one RDMA write.
- **The Durable State API** — named, versioned, forkable, exportable, **mutable** state handles (`/v1/states`). Mutable state violates the `state ≡ f(token-prefix)` invariant that paged-KV cache indexes are built on; it is the capability incumbent engines structurally cannot follow.

Full-attention layers in hybrid models (Qwen3-Next / Kimi-Linear class) run in a deliberately simple paged **guest pool**. Pure transformers are supported as guests for parity — and, later, at constant footprint via **recurrent mode** (sink + KV ring + a segmented bank of RWKV-7 states over evicted context; see `docs/RECURRENT_MODE.md`).

## Design documents

- [`docs/ANGLE.md`](docs/ANGLE.md) — the full vLLM/SGLang architecture map (724k/626k LOC audited), what to steal, what to refuse, the nine candidate angles and why this one survived adversarial review.
- [`docs/RECURRENT_MODE.md`](docs/RECURRENT_MODE.md) — constant-footprint transformer serving via multi-state (layer-wise and context-length-wise) memory banks.
- [`ROADMAP.md`](ROADMAP.md) — milestones.

## Status

**M3 — the Durable State API is real and measured** (`experiments/results/m3_results.md`): named/versioned/forkable/**mutable** state handles over a tiered StateStore (GPU slot / pinned host / NVMe safetensors) with `/v1/states`. Demos on one 4090: 2000 hibernated sessions at 2.29 MiB each resuming in **p50 8.2ms / p99 9.5ms** (16/16 exactness); an agent session surviving a **real process restart bit-exactly**, forked 64× at 8.5ms/fork, and mutated (`decay`) with recorded provenance while its parent stays intact — the operation prefix-keyed caches cannot represent. Trainer/server logprob drift quantified honestly (mean 2e-2 at bf16 across fused-vs-chunked paths; trainer-identical chunked rescoring available by construction).

**M2**: the engine serves RWKV-7 end to end — no-phases scheduler driving FLA-kernel decode from arena state slots, continuous batching (batch-vs-sequential token-identical), per-batch-bucket CUDA graphs, 8.1k tok/s at B=256 on a 4090 (1.5B). Plus a measured PoC of **recurrent mode** for transformers on gemma-4-E4B (constant footprint + segmented state bank; needle recall to 32k). See [`docs/COMPARISON.md`](docs/COMPARISON.md) for head-to-head numbers vs vLLM, SGLang, and Albatross on the same GPU, and `experiments/results/` for all raw tables.

```bash
python -m unittest discover -s tests -v
```

## Layout

```
wkvm/core/config.py     # typed specs: state families, slot layouts, engine limits
wkvm/core/request.py    # Request lifecycle, the num_computed_tokens invariant
wkvm/core/arena.py      # StateArena: per-family slot allocator, exact admission
wkvm/core/scheduler.py  # no-phases continuous-batching scheduler
tests/                  # CPU-only invariant tests
```

## License

Apache-2.0 (LICENSE file pending).
