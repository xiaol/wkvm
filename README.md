# wkvm

**A hypervisor for model state.** State-native inference engine for RWKV-7 / GDN / Mamba2 and hybrid-linear models — where the primary allocation object is a fixed-size per-request **state slot**, not a paged KV block chain.

> WKV + KVM: states are the VMs — create, snapshot, fork, hibernate, resume, live-migrate. The engine is the hypervisor.

## Routed-Span Demo

[![Gemma routed-span recurrent-mode demo](experiments/results/gemma_routed_span_demo.gif)](experiments/results/gemma_routed_span_demo.mp4)

Full-quality MP4: [`experiments/results/gemma_routed_span_demo.mp4`](experiments/results/gemma_routed_span_demo.mp4)

Previous ring/concurrency demo: [`experiments/results/gemma_wkvm_style_demo.mp4`](experiments/results/gemma_wkvm_style_demo.mp4)

## Routed-span vs vLLM/SGLang

Gemma-4-E4B-it on one RTX 4090. vLLM and SGLang are full-KV transformer engines; wkvm routed-span is approximate recurrent mode (`sink16 + ring1024 + routed span bank m64`), so this is a memory/throughput comparison, not a same-semantics quality claim. Rows labelled `wkvm-native` use the native wkvm scheduler/arena/runner/server boundary. Older native rows reused HF Gemma module math; current checkpoint-native rows in [`experiments/results/gemma_native_vllm_sglang_current_compare_20260708.md`](experiments/results/gemma_native_vllm_sglang_current_compare_20260708.md) report `uses_hf_transformer_forward=false` and `uses_hf_model_construction=false`. Older rows labelled PoC are retained only as historical context.

### Final controlled Gemma B16 comparison (2026-07-13)

Uniform B16, 128 fixed output tokens, BF16, greedy decode, and a 19 GiB cap with 1 GiB required headroom. Memory is the whole-GPU delta measured by `nvidia-smi`; it is not process-attributed. Multi-sample values below are means.

| context | engine | samples | E2E output tok/s | comparable decode tok/s | whole-GPU delta | 18 GiB gate |
|---:|---|---:|---:|---:|---:|---|
| 16,384 | **WKVM current** | 3 | **64.769** | **256.467** | **16.871 GiB** | **pass 3/3** |
| 16,384 | vLLM 0.24.0 controlled | 1 | 64.713 | 67.091 | 18.405 GiB | fail |
| 16,384 | SGLang 0.5.14 archived | 1 | 26.435 | not comparable | 18.737 GiB | fail |
| 32,768 | **WKVM current** | 1 | **34.830** | **253.746** | **16.926 GiB** | **pass** |
| 32,768 | vLLM 0.24.0 archived | 1 | 24.543 | 25.345 | 18.407 GiB | fail |
| 32,768 | SGLang 0.5.14 archived | 1 | 26.130 | not comparable | 18.759 GiB | fail |

**Readout:** 16K E2E is a tie, not a robust throughput win: vLLM's point lies inside WKVM's 64.685-64.919 three-run range. WKVM has a clear 16K decode-interval advantage (**3.82x**) and uses **1.534 GiB less** whole-GPU memory. At 32K, the current WKVM run wins E2E against the archived incumbent runs, but that claim is single-sample and the incumbent GPU baseline differed. SGLang decode timing uses separate-run subtraction and is therefore excluded from decode comparisons. The final native quality grid passes **105 cases / 45 cells** with a **0.911111** cell mean; B16 matches B2 on all scorer-visible text and scores, while 29/35 full fixed-length sequences are token-exact.

See the [`final evidence audit`](experiments/results/gemma_b16_evidence_audit_20260713.md) and [`verified provenance bundle`](experiments/results/gemma_b16_evidence_bundle_20260713/README.md) for ranges, configurations, fingerprints, source identity, raw artifacts, and caveats.

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

Readout: routed-span is slower for one exact long generation than vLLM/SGLang full-KV, and the shown routed-span native row is only modestly faster than green HF Transformers on aggregate decode (**57.9 vs 52.6 tok/s**) while supporting many more resident long sessions (**32 vs 2**). The measured advantage is bounded-memory long-context concurrency when approximate recurrent semantics are acceptable; it is not a replacement for full-KV serving when exact transformer behavior is required.

The latest strict HTTP Stage-1 smoke reaches **33.586/52.948 tok/s** for WKVM at B=1/B=2, versus **62.307/89.009** for vLLM 0.24.0 and **57.646/78.967** for SGLang 0.5.14. All rows use exact shared prompt fingerprints and output accounting; the full configuration and semantic caveats are recorded in [`experiments/results/gemma_serving_stage1_strict_20260711.md`](experiments/results/gemma_serving_stage1_strict_20260711.md). This is not a WKVM throughput win.

### Serving-path benchmark

The direct native benchmark is useful for engine throughput, but vLLM/SGLang production comparisons should use the server path. `wkvm.gemma_server` exposes OpenAI-compatible token-id `/v1/completions`, token-id `/v1/stream` SSE events, blocking `/v1/generate`, async-style `/v1/submit` + `/v1/status/<id>`, `/v1/cancel`, `/health`, and `/metrics`. The server now bounds retained completed-request metadata, marks model-step failures as `FINISHED_ERROR`, and supports request timeout cancellation.

```bash
python -m pip install -e '.[gemma-server]'

wkvm-gemma-server --model /path/to/gemma-4-E4B-it --slots 32 --port 8000 \
  --native-gemma-production-profile \
  --max-queue 128 --request-timeout-s 600 --max-completed-requests 4096 \
  --max-request-body-bytes 8388608 --request-read-timeout-s 30

# Replace these with versions reported by the environments running each server.
export GPU_DEVICE=0
export WKVM_SERVER_VERSION=0.0.1
export VLLM_SERVER_VERSION=0.24.0
export SGLANG_SERVER_VERSION=0.5.14
export WKVM_SERVER_LAUNCH_COMMAND='wkvm-gemma-server --model /path/to/gemma-4-E4B-it --slots 32 --port 8000 --native-gemma-production-profile --max-queue 128 --request-timeout-s 600 --max-completed-requests 4096 --max-request-body-bytes 8388608 --request-read-timeout-s 30'
export VLLM_SERVER_LAUNCH_COMMAND='vllm serve /path/to/gemma-4-E4B-it --port 8001 --served-model-name gemma-4-E4B-it --tensor-parallel-size 1'
export SGLANG_SERVER_LAUNCH_COMMAND='python -m sglang.launch_server --model-path /path/to/gemma-4-E4B-it --port 8002 --served-model-name gemma-4-E4B-it --tp-size 1'

python experiments/wkvm_serving_bench.py --backend openai-completions \
  --engine wkvm-native-openai-completions --url http://127.0.0.1:8000 \
  --engine-version "$WKVM_SERVER_VERSION" --engine-version-source server_environment \
  --target-server-launch-command "$WKVM_SERVER_LAUNCH_COMMAND" \
  --target-server-config-json '{"production_profile":true,"slots":32}' \
  --gpu-memory-device "$GPU_DEVICE" --gpu-memory-sample-interval-s 0.1 \
  --served-model gemma-4-E4B-it --semantics routed_span_approximate \
  --ctx 13824 --out 128 \
  --concurrency 1,2,4,8,16,32 --requests-per-row 32 \
  --synthetic-prompts --warmup-requests 1 --warmup-output-tokens 4 \
  --json experiments/results/wkvm_serving_ctx13824_out128.json

python experiments/wkvm_serving_bench.py --backend openai-completions \
  --engine vllm-http-stream --url http://127.0.0.1:8001 \
  --engine-version "$VLLM_SERVER_VERSION" --engine-version-source server_environment \
  --target-server-launch-command "$VLLM_SERVER_LAUNCH_COMMAND" \
  --target-server-config-json '{"tensor_parallel_size":1}' \
  --gpu-memory-device "$GPU_DEVICE" --gpu-memory-sample-interval-s 0.1 \
  --served-model gemma-4-E4B-it --semantics full_kv --ctx 13824 --out 128 \
  --concurrency 1,2,4,8,16,32 --requests-per-row 32 \
  --synthetic-prompts --warmup-requests 1 --warmup-output-tokens 4 \
  --json experiments/results/vllm_serving_ctx13824_out128.json

python experiments/wkvm_serving_bench.py --backend openai-completions \
  --engine sglang-http-stream --url http://127.0.0.1:8002 \
  --engine-version "$SGLANG_SERVER_VERSION" --engine-version-source server_environment \
  --target-server-launch-command "$SGLANG_SERVER_LAUNCH_COMMAND" \
  --target-server-config-json '{"tp_size":1}' \
  --gpu-memory-device "$GPU_DEVICE" --gpu-memory-sample-interval-s 0.1 \
  --served-model gemma-4-E4B-it --semantics full_kv --ctx 13824 --out 128 \
  --concurrency 1,2,4,8,16,32 --requests-per-row 32 \
  --synthetic-prompts --warmup-requests 1 --warmup-output-tokens 4 \
  --json experiments/results/sglang_serving_ctx13824_out128.json

python experiments/gemma_bench_report.py experiments/results/wkvm_serving_ctx13824_out128.json \
  experiments/results/vllm_serving_ctx13824_out128.json \
  experiments/results/sglang_serving_ctx13824_out128.json \
  --require-same-shape --require-same-prompt-fingerprint --require-comparable \
  --out experiments/results/serving_compare_ctx13824_out128.md

# Bounded no-weight CUDA proof for same-shape graph reuse across fresh cohorts.
python experiments/token_pool_graph_reuse_smoke.py --hits 4 \
  --json experiments/results/token_pool_graph_reuse_smoke.json

# Bounded primitive benchmark for the fused Gemma layer-exit residual boundary.
python experiments/native_gemma_fused_residual_bench.py \
  > experiments/results/native_gemma_fused_residual_bench.json
```

`python -m wkvm.gemma_server` is equivalent to the `wkvm-gemma-server` entrypoint. The HTTP frontend requires one valid `Content-Length`, rejects transfer-encoded or larger-than-8-MiB bodies by default, and allows 30 seconds by default to receive a complete body; tune these with `--max-request-body-bytes` and `--request-read-timeout-s`. The base install remains dependency-free; the `gemma-server` extra adds PyTorch, Transformers `>=5.7,<6`, safetensors for the checkpoint-native loader, and Accelerate for the Hugging Face `device_map` loader. Token-pool serving is opt-in through `--enable-token-pool-attention`, `--token-pool-max-context-len`, `--token-pool-capacity`, `--token-pool-paged-block-size`, and `--persistent-padded-sliding-metadata-padding`; capacity is workload-dependent, so the production profile does not silently choose it.

This records per-request TTFT/end-to-end latency, output throughput, success/error counts, exact prompt fingerprints, and ITL only when the stream exposes token-exact boundaries. `--requests-per-row 32` keeps each concurrency level busy for up to 32 requests instead of timing one short cohort; prompt rows are disjoint across the ladder and every run gets unique request IDs, preventing exact-prompt cache reuse across rows and retained-ID collisions. `--engine-version` and `--target-server-launch-command` are deliberately operator-reported because a benchmark client cannot safely infer the version or startup arguments of an arbitrary local or remote HTTP server. The optional `--target-server-config-json` records an engine-specific JSON object for settings that are not obvious from the command. The command is recorded verbatim and the config is preserved as structured JSON, so do not include credentials in either. New structured artifacts must include target launch provenance for `--require-comparable`; legacy artifacts remain readable and receive a warning. The top-level `launch_command` remains the benchmark-client invocation, while client Python/package versions, the selected GPU, and its driver are recorded separately. `--gpu-memory-device` is optional and polls `nvidia-smi` only around measured rows. Its baseline, peak, and delta cover the **whole physical GPU**, including unrelated processes, so use an otherwise idle device and do not interpret the delta as process-attributed engine memory. Enable the same monitoring interval for every strict comparison because polling is instrumentation overhead. The strict report requires the complete concurrency ladder, exact output-token accounting, identical sampling/warmup/request/monitoring policies, and matching prompt fingerprints. The OpenAI-compatible path sends token-id prompts to `/v1/completions`; wkvm and vLLM can return streamed `token_ids`, while an SGLang text-chunk stream uses final usage for goodput and is excluded from ITL aggregation. Because routed-span is approximate and vLLM/SGLang are full-KV, the report labels the semantic mismatch and does not treat throughput as quality-equivalent. The bounded graph smoke proves one capture plus fresh-metadata hits without loading Gemma; it is not a server-throughput result. The fused residual benchmark is likewise a primitive measurement. A strict sustained full-model server ladder and larger fused paged/ragged attention and projection/MLP kernels remain the performance target.

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
