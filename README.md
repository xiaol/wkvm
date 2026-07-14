# wkvm

**A hypervisor for model state.** State-native inference engine for RWKV-7 / GDN / Mamba2 and hybrid-linear models — where the primary allocation object is a fixed-size per-request **state slot**, not a paged KV block chain.

> WKV + KVM: states are the VMs — create, snapshot, fork, hibernate, resume, live-migrate. The engine is the hypervisor.

## Routed-Span Demo

[![Gemma routed-span recurrent-mode demo](experiments/results/gemma_routed_span_demo.gif)](experiments/results/gemma_routed_span_demo.mp4)

Full-quality MP4: [`experiments/results/gemma_routed_span_demo.mp4`](experiments/results/gemma_routed_span_demo.mp4)

Previous ring/concurrency demo: [`experiments/results/gemma_wkvm_style_demo.mp4`](experiments/results/gemma_wkvm_style_demo.mp4)

## Routed-span vs vLLM/SGLang

Gemma-4-E4B-it on one RTX 4090. vLLM and SGLang are full-KV transformer engines; wkvm routed-span is approximate recurrent mode (`sink16 + ring1024 + a bounded routed-span bank`, with each result naming its m32/m64 profile), so this is a memory/throughput comparison, not a same-semantics quality claim. Rows labelled `wkvm-native` use the native wkvm scheduler/arena/runner/server boundary. Older native rows reused HF Gemma module math; current checkpoint-native rows in [`experiments/results/gemma_native_vllm_sglang_current_compare_20260708.md`](experiments/results/gemma_native_vllm_sglang_current_compare_20260708.md) report `uses_hf_transformer_forward=false` and `uses_hf_model_construction=false`. Older rows labelled PoC are retained only as historical context.

### Real Open WebUI B32 x 8 comparison (2026-07-14)

Open WebUI 0.10.2 used one authenticated Socket.IO connection across 32 persisted chats for eight synchronized turns. The harness dispatched 32 HTTP requests concurrently per turn and waited for the last terminal Socket.IO event. Every engine completed 256/256 requests and generated exactly 32,768 output tokens. This is offered B32 application concurrency, not 32 users, 32 browser connections, or 32 GPU rows executing at once.

| engine | completion / validation | 8-turn wall | output tok/s | API-accounted total tok/s | unique app tok/s | peak whole GPU |
|---|---|---:|---:|---:|---:|---:|
| **vLLM 0.24.0** | **256/256, pass** | **355.921s** | **92.065** | **10,451.441** | **1,355.087** | **20,494 MiB** |
| SGLang 0.5.14 | 256/256, pass | 605.763s | 54.094 | 6,136.393 | 796.192 | 20,970 MiB |
| WKVM current m32 | 256/256 accounting pass; strict reuse fail | 607.226s | 53.963 | 6,127.860 | 794.274 | 22,858 MiB |

Generated-output tok/s is the primary end-to-end rate: fixed output tokens divided by the sum of the eight cohort walls. API-accounted total tok/s repeats cumulative prompt history on every later request, so it is not fresh model-compute throughput. Unique application tok/s counts the 442,368 initial prompt tokens, 7,168 new continuation-input tokens, and 32,768 output tokens once.

vLLM finished 251.305 seconds sooner than WKVM and delivered 70.6% higher output throughput. SGLang and WKVM were effectively tied in this single sample: SGLang was 0.24% faster. WKVM retained 32 states but decoded two rows per microbatch. Likewise, vLLM's 9.481 and SGLang's 2.504 full-history equivalents describe retained KV capacity, not request concurrency; both accepted B32 and scheduled internally.

WKVM completed transport and token accounting correctly, but only 125/224 continuations exactly reused parked state; 99 persisted and re-rendered histories no longer matched the exact parked token prefix, so they were safely retired and restarted. See the [`full Open WebUI report`](experiments/results/open_webui_b32_t8_compare_20260714.md) and [`aggregate JSON`](experiments/results/open_webui_b32_t8_compare_20260714.json).

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

### One-shot offered-B32 comparison (2026-07-13)

Direct-engine cohort with 32 uniform 16,384-token prompts and 128 fixed output tokens on the same RTX 4090. Performance cells are `min / mean / max` over two runs. WKVM's resident count is engine-proven; vLLM/SGLang values are offered B32 with much smaller profiled full-length KV capacity and queued waves.

| engine | completion / capacity | E2E output tok/s | comparable decode tok/s | batch-wall p95 | whole-GPU delta | 18 GiB gate |
|---|---|---:|---:|---:|---:|---|
| **WKVM current** | **32/32 truly resident**, zero pressure events | 53.991 / **54.757** / 55.522 | 191.148 / **193.236** / 195.323 | 73.722 / **74.769** / 75.816s | 20.255 / **20.859** / 21.463 GiB | fail 2/2 |
| vLLM 0.24.0 | 32/32 offered; 3.134-3.314 full-length KV equivalents | 58.334 / **61.297** / 64.259 | 59.121 / **62.162** / 65.203 | 63.742 / **66.980** / 70.217s | 18.530 / **18.862** / 19.193 GiB | fail 2/2 |
| SGLang 0.5.14 | 32/32 offered; 2.995-3.476 full-length KV equivalents | 45.137 / **50.859** / 56.581 | not comparable | 72.392 / **81.569** / 90.746s | 19.131 / **19.145** / 19.159 GiB | fail 2/2 |

**Readout:** WKVM proves the architecture's resident-capacity benefit—32 resident long-context states versus roughly three full-length KV equivalents—but not an overall B32 win. vLLM is 11.9% faster on mean E2E goodput; WKVM is 3.11x faster on the same-run decode interval; WKVM/SGLang E2E ranges overlap. All B32 rows miss the memory gate, while current-source WKVM B16 remains green at 16.797 GiB. SGLang at its lower 0.78 memory fraction passes the memory gate but cannot admit one 16K prompt under the measured desktop baseline.

The experiment also fixed a high-concurrency CUDA-graph bug: lazy token-pool growth could invalidate captured buffer addresses above one decode microbatch. Graph replay now checks the pool generation and safely recaptures; guarded B24/B32 runs complete with unchanged output fingerprints. See the [`high-concurrency audit and raw artifacts`](experiments/results/gemma_high_concurrency_b32_audit_20260713.md) and [`source/artifact provenance`](experiments/results/gemma_high_concurrency_b32_provenance_20260713/README.md) for telemetry, the two-pass measurement method, graph-fault evidence, and limitations. This remains a direct-engine cohort result. The real Open WebUI B32 x 8 test above now covers sustained application-path serving, but it is one offered-concurrency point, not a full HTTP ladder.

### Sustained multi-turn B32 session comparison (2026-07-13)

Direct-engine synchronized-turn workload on the same RTX 4090: 32 logical sessions, 8 turns, 13,824 initial tokens/session, 32 new input tokens/continuation, and 128 output tokens/request. Each turn submits 32 requests (`offered B32`), so every engine completes 256 session-turn requests and emits 32,768 tokens. Alternating request order avoids deterministic forward-scan cache thrashing.

| engine | completion | retained-history evidence | turn-0 output tok/s | continuation output tok/s | all-turn output tok/s | 8-turn wall | completed req/s |
|---|---:|---|---:|---:|---:|---:|---:|
| vLLM 0.24.0 | 256/256 | 49,114 KV-token capacity = 3.255 full-history equivalents; about 6 full-prefix hits/continuation turn | **80.120** | **85.063** | **84.412** | **388.190s** | **0.659** |
| WKVM current (tuned m32) | 256/256 | **32/32 parked states**, 224/224 continuation reuse hits, zero full reprefills | 32.177 | 65.507 | 57.997 | 564.990s | 0.453 |
| SGLang 0.5.14 | 256/256 | 33,736 effective KV tokens = 2.236 full-history equivalents; about 2 full-prefix hits/continuation turn | 56.932 | 56.068 | 56.175 | 583.324s | 0.439 |

**Readout:** yes, vLLM finishes the complete workload faster: **176.800s sooner than WKVM**, with 45.5% higher all-turn output throughput and 29.9% higher continuation throughput. Counting each unique application input and output token once gives the same ordering: **1,242.444 tok/s vLLM**, 853.650 WKVM, and 826.820 SGLang; this is application goodput, not model-compute throughput. WKVM is 3.24% faster overall than SGLang and retains far more long histories, but residency is a memory-capacity advantage, not automatically a speed advantage. `32 parked states` means all 32 compact WKVM histories survive between turns; it does not mean 32 rows execute simultaneously (this run decodes in two-row microbatches). Likewise, `3.255 KV equivalents` is vLLM's profiled KV-token capacity divided by one provisioned full history—not 3.255 requests completed and not a hard request-concurrency limit. vLLM accepts all 32 requests and schedules them in waves.

The primary throughput number is generated output tokens divided by summed synchronized-turn wall time, so it includes prefill, queueing, cache reuse/recomputation, and decode. Counting every logical input token as fresh compute would be misleading because WKVM advances retained state while vLLM/SGLang may reuse prefix KV. The successful WKVM row required `m_slots=32`, one prefill row, two decode rows, and nonpersistent full-attention rows; the default m64 B32 attempt OOMed. Only turn-0 prompts are token-identical across engines; greedy outputs diverge, so later turns are equal-shape autonomous histories rather than token-identical requests. WKVM remains approximate routed-span semantics, not full-KV quality equivalence. See the [`sustained B32 experiment report`](experiments/results/gemma_multiturn_b32_compare_20260713.md) for turn-level data, order controls, B3 low-concurrency results, parity evidence, launch flags, artifact hashes, and limitations.

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

Readout: routed-span is slower for one exact long generation than vLLM/SGLang full-KV, and the shown routed-span native row is only modestly faster than green HF Transformers on aggregate decode (**57.9 vs 52.6 tok/s**) while supporting many more resident long sessions (**32 vs 2**). The measured advantage is bounded-memory long-context residency when approximate recurrent semantics are acceptable; it is not a replacement for full-KV serving when exact transformer behavior is required.

The latest strict HTTP Stage-1 smoke reaches **33.586/52.948 tok/s** for WKVM at B=1/B=2, versus **62.307/89.009** for vLLM 0.24.0 and **57.646/78.967** for SGLang 0.5.14. All rows use exact shared prompt fingerprints and output accounting; the full configuration and semantic caveats are recorded in [`experiments/results/gemma_serving_stage1_strict_20260711.md`](experiments/results/gemma_serving_stage1_strict_20260711.md). This is not a WKVM throughput win.

### Serving-path benchmark

The direct native benchmark is useful for engine throughput, but vLLM/SGLang production comparisons should use the server path. `wkvm.gemma_server` exposes OpenAI-compatible token-id `/v1/completions`, token-id `/v1/stream` SSE events, blocking `/v1/generate`, async-style `/v1/submit` + `/v1/status/<id>`, `/v1/cancel`, `/v1/models`, `/health`, and `/metrics`. The server now bounds retained completed-request metadata, marks model-step failures as `FINISHED_ERROR`, and supports request timeout cancellation.

Chat compatibility is opt-in: `--enable-openai-chat` loads the tokenizer and enables blocking and SSE `/v1/chat/completions`; token-ID serving remains the default startup path. This is a text-only greedy subset, not full Chat Completions feature parity. Forwarded `X-OpenWebUI-User-Id` and `X-OpenWebUI-Chat-Id` isolate parked sessions by model, user, and chat, and reuse requires exact token-prefix continuity. `--ignore-eos` is a fixed-output benchmark control, not a normal chat requirement.

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

This records per-request TTFT/end-to-end latency, output throughput, success/error counts, exact prompt fingerprints, and ITL only when the stream exposes token-exact boundaries. `--requests-per-row 32` keeps each concurrency level busy for up to 32 requests instead of timing one short cohort; prompt rows are disjoint across the ladder and every run gets unique request IDs, preventing exact-prompt cache reuse across rows and retained-ID collisions. `--engine-version` and `--target-server-launch-command` are deliberately operator-reported because a benchmark client cannot safely infer the version or startup arguments of an arbitrary local or remote HTTP server. The optional `--target-server-config-json` records an engine-specific JSON object for settings that are not obvious from the command. The command is recorded verbatim and the config is preserved as structured JSON, so do not include credentials in either. New structured artifacts must include target launch provenance for `--require-comparable`; legacy artifacts remain readable and receive a warning. The top-level `launch_command` remains the benchmark-client invocation, while client Python/package versions, the selected GPU, and its driver are recorded separately. `--gpu-memory-device` is optional and polls `nvidia-smi` only around measured rows. Its baseline, peak, and delta cover the **whole physical GPU**, including unrelated processes, so use an otherwise idle device and do not interpret the delta as process-attributed engine memory. Enable the same monitoring interval for every strict comparison because polling is instrumentation overhead. The strict report requires the complete concurrency ladder, exact output-token accounting, identical sampling/warmup/request/monitoring policies, and matching prompt fingerprints. The OpenAI-compatible path sends token-id prompts to `/v1/completions`; wkvm and vLLM can return streamed `token_ids`, while an SGLang text-chunk stream uses final usage for goodput and is excluded from ITL aggregation. Because routed-span is approximate and vLLM/SGLang are full-KV, the report labels the semantic mismatch and does not treat throughput as quality-equivalent. The bounded graph smoke proves one capture plus fresh-metadata hits without loading Gemma; it is not a server-throughput result. The fused residual benchmark is likewise a primitive measurement. The real Open WebUI run covers one sustained B32 application-path point; a full `1/2/4/8/16/32` HTTP ladder and larger fused paged/ragged attention and projection/MLP kernels remain performance targets.

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
