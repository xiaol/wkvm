# Real Open WebUI B32 x 8 Comparison (2026-07-14)

This experiment drives Open WebUI 0.10.2 through the same backend protocol used
by its browser: authenticated `POST /api/chat/completions` task creation,
provider SSE, and completion delivery on `/ws/socket.io`. One authenticated
Socket.IO client manages 32 persisted conversations. This is not 32 human users
or 32 browser connections.

## Result

Gemma-4-E4B-it ran on one RTX 4090 with 32 conversations, 8 synchronized turns,
13,824 rendered tokens on turn 0, 32 new user-content tokens on each later turn,
and exactly 128 generated tokens per request. Every engine completed 256/256
requests and emitted 32,768 tokens.

| Engine | Completion / validation | 8-turn wall | Output tok/s | API total tok/s | Unique app tok/s | Requests/s | Peak whole GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| **vLLM 0.24.0** | **256/256, pass** | **355.921s** | **92.065** | **10,451.441** | **1,355.087** | **0.719** | **20,494 MiB** |
| SGLang 0.5.14 | 256/256, pass | 605.763s | 54.094 | 6,136.393 | 796.192 | 0.423 | 20,970 MiB |
| WKVM current m32 | 256/256 accounting pass; strict reuse fail | 607.226s | 53.963 | 6,127.860 | 794.274 | 0.422 | 22,858 MiB |

The WKVM raw status is `failed` only because the optional strict architecture
gate expected 224 exact parked-state reuse hits and observed 125. Its transport,
usage, prompt-accounting, fixed-output, chat-ID, and completion gates all pass.

| Engine | Turn-0 output tok/s | Continuation output tok/s | UI TTFT p50 / p95 / p99 | E2E p50 / p95 / p99 |
|---|---:|---:|---:|---:|
| **vLLM 0.24.0** | **99.600** | **91.081** | **20.634 / 39.580 / 43.668s** | **32.720 / 45.858 / 46.960s** |
| SGLang 0.5.14 | 54.506 | 54.035 | 37.921 / 71.808 / 74.247s | 40.558 / 74.832 / 76.709s |
| WKVM current m32 | 41.756 | 56.315 | 23.612 / 53.866 / 54.029s | 73.688 / 97.906 / 98.053s |

Readout:

- vLLM finishes 251.305 seconds sooner than WKVM, uses 41.4% less wall time,
  and has 70.6% higher generated-output throughput.
- WKVM and SGLang are a single-sample tie: SGLang is only 0.24% faster in
  output tok/s and finishes 1.463 seconds sooner.
- WKVM reaches all 32 resident state slots, but the end-to-end integration does
  not preserve every parked history. Persisted and re-rendered assistant text
  can differ from the exact parked token prefix, causing safe session restarts,
  so capacity does not become a throughput win.
- WKVM uses 2,364 MiB more peak whole-GPU memory than vLLM and 1,888 MiB more
  than SGLang in these B32 server configurations.

## What E2E Measures

The measured request timeline is:

```text
HTTP POST start
  -> Open WebUI task acknowledgement
  -> provider SSE
  -> Open WebUI processing and persistence
  -> first non-empty Socket.IO content (UI-path TTFT)
  -> terminal Socket.IO done/error event (E2E)
```

The throughput wall is the sum of the eight synchronized cohort walls, each
from its first request dispatch to its last terminal event. Workload
construction, provider-metrics snapshots, and gaps between turns are excluded.
ACK latency, UI-path TTFT, and E2E use `time.perf_counter_ns` at the client.

Inter-token latency is not reported. Open WebUI may coalesce events and emit
cumulative content, so Socket.IO event boundaries are not token boundaries.

## Three Throughput Numbers

The user-facing phrase "total tokens/s" is ambiguous, so the harness reports
three rates:

| Metric | Numerator | Interpretation |
|---|---|---|
| Generated output tok/s | 32,768 completion tokens | Primary end-to-end generation throughput |
| API-accounted total tok/s | Sum of provider `prompt_tokens + completion_tokens` for all 256 requests | Standard API accounting; repeats cumulative history on every turn |
| Unique application goodput | 442,368 turn-0 prompt tokens + 7,168 later user tokens + 32,768 output tokens = 482,304 | Counts each logical application token once |

API-accounted totals are about 3.72 million tokens because every later request
contains its cumulative conversation. They do not mean the engine recomputed
every token; prefix caches and recurrent state may reuse work. Unique application
goodput avoids that double counting. All three rates give the same ordering:
vLLM first, then an effective WKVM/SGLang tie.

## Capacity Is Not Concurrency

Every engine accepts an offered cohort of 32 and completes it. Capacity evidence
describes retained history, not a hard request-concurrency limit.

| Engine | Measured retained-history evidence | Meaning |
|---|---|---|
| WKVM | 32 state slots; 32 resident/parked sessions at every final barrier | All 32 compact states fit, but decode uses two-row microbatches rather than 32 simultaneous rows |
| vLLM | 144,406 profiled KV tokens / 15,232 = 9.481 full-history equivalents | Accepts 32 and schedules waves; reported 85,760 cached tokens across the run |
| SGLang | 38,146 full-attention pool tokens / 15,232 = 2.504 equivalents; separate 30,516-token sliding pool | Accepts 32 and schedules waves; reported 2,170 mostly short shared-prefix cached tokens |

The vLLM capacity differs from the earlier direct-engine 3.255-equivalent result
because this text-only OpenAI server profile allocates a larger measured KV pool.
Neither 9.481 nor 2.504 is a count of completed requests.

## WKVM Session Finding

WKVM identifies a retained chat by model, forwarded Open WebUI user ID, and
forwarded chat ID. It reuses state only when the next rendered prompt has the
exact prior token history as a prefix.

Open WebUI normalizes assistant text before persistence, including a final
`.strip()`. Decode-to-text-to-token round trips can therefore differ from the
exact generated token sequence parked by WKVM. The final run records:

- 32 resident and parked sessions at the final barrier;
- 125 exact reuse hits out of 224 continuation turns (55.8%);
- 99 safe session retire-and-restart events;
- 131 sessions opened and 99 closed;
- zero engine-level reuse misses or full-reprefill counters, because the HTTP
  service rejects the mismatched prefix before asking the engine to continue;
- 11 allocator cleanup calls after first-token prefill and before decode.

This is a real application-contract limitation. Weakening the exact-prefix
check would silently attach a recurrent state to different text, so the server
correctly restarts instead.

## Workload Controls

- Open WebUI 0.10.2, one authenticated client, 32 persisted chats.
- Background title, tag, follow-up, context-compaction, and realtime-save work
  disabled.
- `function_calling=legacy` explicitly disables hidden builtin tool injection
  and is removed before provider forwarding. Open WebUI's `tools: []` opt-out
  was not used because vLLM 0.24.0 rejects empty tool arrays.
- Turn 0 is exactly 13,824 rendered tokens for every conversation.
- Each later user content is exactly 32 tokenizer tokens.
- Greedy decode, `top_p=1`, `ignore_eos=true`, 128 output tokens.
- Forward/reverse request order alternates by turn.
- Identical initial and user-delta fingerprints across all engines.
- Later complete histories are autonomous because each engine appends its own
  generated assistant text.
- The server context ceiling is 15,232, not the direct token-ID benchmark's
  15,088. Observed final prompts reach 15,014 tokens, and another 128 output
  tokens require at least 15,142 positions.

## Historical benchmark reproduction (not a serving profile)

The commands below reproduce this frozen B32 artifact. They include fixed-output
and near-device-ceiling benchmark controls that are inappropriate for normal
chat. For a conservative local UI setup, use the
[`Open WebUI demo guide`](../../docs/OPEN_WEBUI_DEMO.md) instead.

Install the repository-side benchmark dependencies:

```bash
python -m pip install -e '.[gemma-server,open-webui-bench]'
```

The `open-webui-bench` extra installs the harness client, not Open WebUI itself.
This artifact used Open WebUI 0.10.2 installed as a separate application.

Configure Open WebUI to forward session identity and point at the active engine:

```bash
export ENABLE_FORWARD_USER_INFO_HEADERS=true
export ENABLE_WEBSOCKET_SUPPORT=true
export OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1
export DATA_DIR=/path/to/open-webui-data

open-webui serve --host 127.0.0.1 --port 8080
```

Start WKVM with the measured B32 profile:

```bash
export MODEL=/path/to/gemma-4-E4B-it

env \
  TOKENIZERS_PARALLELISM=false \
  WKVM_ENABLE_TOKEN_POOL_TRITON=1 \
  WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON=1 \
  WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON=1 \
  WKVM_TOKEN_POOL_TRITON_STRICT=1 \
  WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY=1 \
  python -m wkvm.gemma_server \
  --model "$MODEL" --served-model-name gemma-4-E4B-it --port 8000 \
  --slots 32 --max-queue 128 --request-timeout-s 1200 \
  --enable-openai-chat --ignore-eos \
  --chat-session-ttl-s 3600 --max-chat-sessions 32 --batch-wait-s 0.05 \
  --empty-cuda-cache-before-decode \
  --native-gemma-checkpoint-loader \
  --native-gemma-attention-backend sdpa_single_gqa \
  --native-gemma-projection-backend separate \
  --enable-token-pool-attention \
  --token-pool-max-context-len 15232 --token-pool-capacity 65536 \
  --token-pool-paged-block-size 16 \
  --persistent-padded-sliding-metadata-padding \
  --persistent-padded-decode-cuda-graph \
  --persistent-padded-decode-graph-warmup-iters 0 \
  --persistent-padded-decode-steps 32 \
  --disable-persistent-padded-full-attention-rows \
  --prefill-microbatch-rows 1 --decode-microbatch-rows 2 \
  --sink 16 --window 1024 --m-slots 32 --route-chunk 512 --device cuda
```

Do not combine this command with `--native-gemma-production-profile`; that
profile intentionally applies its own m64 and 128-step defaults after parsing.

The equivalent incumbent launches are:

```bash
vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 --served-model-name gemma-4-E4B-it \
  --dtype bfloat16 --max-model-len 15232 --max-num-seqs 32 \
  --gpu-memory-utilization 0.74 --enable-prefix-caching \
  --language-model-only --limit-mm-per-prompt '{"image":0,"audio":0}' \
  --compilation-config \
  '{"mode":0,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32],"max_cudagraph_capture_size":32}' \
  --generation-config vllm --enable-prompt-tokens-details

python experiments/sglang_gemma_server.py \
  --model-path "$MODEL" --host 127.0.0.1 --port 8000 \
  --served-model-name gemma-4-E4B-it --dtype bfloat16 \
  --context-length 15232 --mem-fraction-static 0.82 \
  --max-running-requests 32 --attention-backend triton \
  --cuda-graph-backend-decode full \
  --cuda-graph-backend-prefill disabled
```

Run the UI-path workload with an authenticated token supplied only through the
environment:

```bash
export OPEN_WEBUI_TOKEN='...'

python experiments/open_webui_multiturn_bench.py \
  --open-webui-url http://127.0.0.1:8080 \
  --tokenizer-path "$MODEL" --model gemma-4-E4B-it \
  --sessions 32 --turns 8 \
  --initial-context-tokens 13824 --turn-input-tokens 32 \
  --output-tokens-per-turn 128 --request-order-policy alternating \
  --gpu-memory-device 0 --engine-name ENGINE --engine-version VERSION \
  --json /path/to/result.json
```

For WKVM, add `--provider-metrics-url http://127.0.0.1:8000/metrics` and
`--require-wkvm-session-reuse` to audit the strict state-reuse invariant.

## Artifacts

The committed aggregate is
[`open_webui_b32_t8_compare_20260714.json`](open_webui_b32_t8_compare_20260714.json).
Request-level raw JSON remains in the local external artifact store and is
anchored by these hashes:

| Engine | Raw basename | Bytes | SHA-256 |
|---|---|---:|---|
| WKVM | `wkvm_openwebui_b32_t8_ctx13824_out128_legacy_20260714.json` | 1,153,164 | `a0ca0bc524ac8d9de348e801c96f79f39fa47b63972b59ca46173b83383f877c` |
| vLLM | `vllm_openwebui_b32_t8_ctx13824_out128_20260714.json` | 543,567 | `78150b4db0585167ba27fef3f33946da688f6acdd923da21d6bc9f378262cb4c` |
| SGLang | `sglang_openwebui_b32_t8_ctx13824_out128_20260714.json` | 551,610 | `75580ba760ea39f4fe6f62d8f7395f623a7a727b9727f4ee3f424ebf70fef6d0` |

The real-path preflight also passed a mock-provider B2 x 2 run and proved that
Open WebUI forwarded non-empty user and chat identity headers on all requests.

## Limitations

- One measured sample per final engine row; no randomized engine-order repeat.
- Whole-GPU memory includes desktop processes and uses different post-load
  baselines, so compare peak used memory rather than process allocation.
- One Open WebUI instance and database served all engines with distinct chats;
  database growth was not reset between engines.
- The test exercises the real REST, persistence, provider SSE, and Socket.IO
  backend path, but it does not render a browser DOM.
- WKVM routed-span mode is approximate recurrent semantics; vLLM and SGLang use
  full/hybrid KV semantics. This is not a token-quality equivalence claim.
- Fixed `ignore_eos` output is a throughput control and may generate beyond a
  model's natural end-of-turn token.
- Later full histories differ because generated text differs by engine.
