# Gemma 16K High-Concurrency Audit: WKVM vs vLLM and SGLang

Date: 2026-07-13

## Outcome

The high-concurrency result is a capacity win for WKVM, not an overall serving
win.

- WKVM proves 32/32 simultaneous resident 16K sessions with zero scheduler
  backpressure or retractions. vLLM and SGLang accept an offered B32 load but
  expose only about 3.0-3.5 full-length KV-context equivalents and process the
  requests in waves.
- vLLM wins B32 end-to-end output goodput in both unpolled repeats. Its mean is
  61.297 tok/s versus WKVM at 54.757 tok/s, an 11.9% advantage.
- WKVM's same-run decode interval is 193.236 tok/s mean versus vLLM at 62.162
  tok/s, a 3.11x advantage. The vLLM interval includes queued waves, so this is
  interval goodput rather than a claim about 32 simultaneous full-KV decodes.
- WKVM and SGLang do not have a robust B32 E2E ordering: WKVM's
  53.991-55.522 tok/s range overlaps SGLang's 45.137-56.581 range. SGLang's
  decode estimate uses separate-run subtraction and is excluded.
- Every successful B32 configuration fails the declared 18 GiB whole-GPU
  engine-delta gate. WKVM's best B32 delta is 20.255 GiB, vLLM's is 18.530
  GiB, and SGLang's is 19.131 GiB.
- The experiment exposed a real WKVM CUDA-graph safety defect above one decode
  microbatch. A generation guard now rejects graphs whose token-pool buffers
  grew after capture; guarded B24 and B32 runs complete without a driver fault.

## Benchmark contract

| Field | Value |
|---|---|
| Model | Gemma-4-E4B-it |
| GPU | One NVIDIA RTX 4090, 24,564 MiB |
| Dtype | BF16 |
| Shape | Uniform 16,384 input tokens, offered B32, 128 output tokens |
| Prompt source | Deterministic synthetic token IDs; no tokenizer |
| Sampling | Greedy, fixed length, EOS ignored |
| Prompt fingerprint | `6ee3f368883b8088bb29fe51ac6dd424d98ed9345c823eafa491a2aa0739dccf` |
| Output fingerprint | `c24e9fe709ea9dcc41a3dc7181817ff2c0e54265e750c5789162b6c8bfbdb465` for every successful promoted run |
| Memory policy | 19 GiB cap with 1 GiB required headroom; pass at `<=18.000 GiB` whole-GPU delta |
| Memory source | `nvidia-smi`, 0.1-second whole-device sampling |
| Repeats | Two unpolled performance runs per engine at B32 |

WKVM uses approximate routed-span recurrent semantics. vLLM and SGLang use
full-KV transformer semantics. Exact output identity on this synthetic prompt
set does not generalize that semantic equivalence.

## B32 comparison

Multi-run cells are `min / mean / max` over two performance runs. Incumbent
running/waiting values come from separate 50 ms telemetry passes so telemetry
collection does not tax the promoted throughput measurements; WKVM uses its
promoted engine counters.

| Engine | B32 completion and residence | Full-length context capacity | Telemetry peak running / waiting | E2E output tok/s | Comparable decode tok/s | Batch-wall p95 field | Whole-GPU delta GiB | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **WKVM current** | **32/32; all 32 proven resident**, 0 backpressure/retractions | **32 resident states** | **32 / 32\*** | 53.991 / **54.757** / 55.522 | 191.148 / **193.236** / 195.323 | 73.722 / **74.769** / 75.816 s | 20.255 / **20.859** / 21.463 | fail 2/2 |
| vLLM 0.24.0 | 32/32 offered; queued waves | 3.134-3.314 KV equivalents | 5 / 30 | 58.334 / **61.297** / 64.259 | 59.121 / **62.162** / 65.203 | 63.742 / **66.980** / 70.217 s | 18.530 / **18.862** / 19.193 | fail 2/2 |
| SGLang 0.5.14 | 32/32 offered; queued/retracted waves | 2.995-3.476 KV equivalents | 3 / 31 | 45.137 / **50.859** / 56.581 | not comparable | 72.392 / **81.569** / 90.746 s | 19.131 / **19.145** / 19.159 | fail 2/2 |

The incumbent `p95_latency_s` field is the full synchronous batch wall copied
across outputs, not a per-request streaming percentile. It is retained only as
a batch completion-time indicator.

\* WKVM's `max_waiting=32` is the initial submission-staging high-water mark,
not a pressure event. Both B32 rows record zero backpressure and zero
retractions while proving all 32 states resident.

### Residence interpretation

WKVM records `max_running=32`, `max_resident_state_slots=32`,
`max_runnable_rows=32`, zero backpressure, and zero retractions in both B32
runs. Its token-pool high-water mark is 63,615 of 65,536 slots.

vLLM reports 51,748-54,718 KV tokens in the unpolled runs, or
3.134-3.314 full-length 16,512-token request equivalents. The 50 ms telemetry
pass observes at most five running and 30 waiting requests with zero
preemptions. Scheduler-running requests can be partially prefilling, so five
must not be called five fully resident 16K contexts.

SGLang reports 49,453-57,389 effective tokens in the unpolled runs, or
2.995-3.476 full-length equivalents. Its 50 ms telemetry pass observes three
running, 31 waiting, and 33,010 used tokens. The unpolled repeats record 10 and
zero output retractions. `max_running_requests=32` is only a scheduler limit,
not resident capacity.

At `mem_fraction_static=0.78`, SGLang stays below the memory gate at 17.521
GiB delta but profiles only 13,922 tokens and rejects every 16,384-token input.
The successful rows require `mem_fraction_static=0.82` and then fail the 18 GiB
gate.

## WKVM resident ladder

| B | Samples | Completion | E2E tok/s | Decode tok/s | p95 latency | Whole-GPU delta | Token-pool HWM | Graph captures / replays | Gate |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 16 | 1 | 16/16 resident | 62.374 | 246.453 | 32.805 s | 16.797 GiB | 28,866 / 36,864 | 1 / 127 | pass |
| 24 | 1 | 24/24 resident | 63.592 | 213.589 | 48.269 s | 19.818 GiB | 47,595 / 49,152 | 4 / 252 | fail |
| 32 | 2 | 32/32 resident | 53.991-55.522 | 191.148-195.323 | 73.722-75.816 s | 20.255-21.463 GiB | 63,615 / 65,536 | 4 / 252 each | fail 2/2 |

B16 remains the highest tested resident concurrency that passes the declared
memory gate. B24 and B32 prove physical residence and safe completion, but not
green capacity.

## CUDA graph fault and fix

The first graph-enabled B32 attempt serialized `CUDA error: an illegal memory
access was encountered`. During diagnosis, the kernel log showed an NVIDIA Xid
31 MMU write fault and local no-graph B24/B32 controls completed. Those kernel
logs and control JSONs are intentionally unpromoted observations, not sealed
evidence in this bundle.

The cause was lazy `TokenKVPool` growth. A graph captured generation 24 buffer
addresses, a later decode microbatch grew/replaced those buffers and advanced
the generation to 44 and then 64, and the attached graph previously replayed
the stale addresses. The new pre-replay guard compares the current pool object
and `buffer_generation` with the captured values. A mismatch raises a bounded
decode fallback, evicts the stale graph, and allows safe recapture.

Guarded B24 and B32 each record the two expected invalidations,
`24 -> 44` and `44 -> 64`, then four captures and 252 replays with zero Triton
runtime errors and the same promoted output fingerprint.

## Measurement method

The promoted incumbent throughput rows use `--telemetry-sample-interval-s
1000`, so only boundary snapshots occur outside the timed generation interval.
Separate 50 ms telemetry passes provide running/waiting/capacity observations.
This avoids charging vLLM/SGLang for frequent registry or shared-memory metric
reads that WKVM does not perform.

All rows use the same prompt fingerprint and produce the same 4,096-token B32
output fingerprint. vLLM uses language-model-only loading, full CUDA graphs,
Inductor disabled, prefix caching disabled, and GPU memory utilization 0.74.
SGLang uses language-model-only override, Triton attention, full decode graphs,
prefill graphs disabled, radix cache disabled, and static memory fraction 0.82.

Whole-GPU deltas are not process-attributed. Desktop GPU baselines varied from
1,636 to 2,500 MiB and affected the profiled incumbent KV capacity. The B32
capacity ordering is large enough to survive that variation, but the exact
delta and throughput values should not be treated as datacenter-stable.

## Promoted artifacts

| Role | Artifact | SHA-256 |
|---|---|---|
| WKVM B16 | [`wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b16_cap19_20260713.json`](wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b16_cap19_20260713.json) | `cce93f42b5ae8bc56aa01f14a2b9edf131d253bdcd847724e12650d56d62b2c8` |
| WKVM B24 | [`wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b24_cap19_20260713.json`](wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b24_cap19_20260713.json) | `e281bb9c40e172a94fe3b57644f49dbef168c3def21a99a48c579acc03ede574` |
| WKVM B32 r1 | [`wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b32_cap19_20260713.json`](wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b32_cap19_20260713.json) | `2a82afb4fb5c0659dd3d9d6b8d178d518357cd339daf0485fdec02dfed8c69e7` |
| WKVM B32 r2 | [`wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b32_cap19_repeat2_20260713.json`](wkvm_gemma_native_graphguard_ctx16384_out128_uniform_b32_cap19_repeat2_20260713.json) | `6bea04aab5e0beea673f0c9621c564f4831d2cdc9d0263f882aaa74f5614414f` |
| WKVM pre-fix failure | [`wkvm_gemma_native_ctx16384_out128_uniform_b32_cap19_20260713.json`](wkvm_gemma_native_ctx16384_out128_uniform_b32_cap19_20260713.json) | `46c53d862e08fb2159aed83b9551b50614c88e1f462f915fec44efcc0997e714` |
| vLLM telemetry pass | [`vllm_gemma_textonly_noinductor_fullgraph_ctx16384_out128_uniform_b32_cap19_20260713.json`](vllm_gemma_textonly_noinductor_fullgraph_ctx16384_out128_uniform_b32_cap19_20260713.json) | `b381dbfa74205e2af23421f7bad6e35c5e8e30f46be7ea8c7007052281160dcc` |
| vLLM performance r1 | [`vllm_gemma_textonly_noinductor_fullgraph_ctx16384_out128_uniform_b32_cap19_final_20260713.json`](vllm_gemma_textonly_noinductor_fullgraph_ctx16384_out128_uniform_b32_cap19_final_20260713.json) | `9c9826def136800b40ae77652100333d3b5347ea5168289a06b6c7e3496403a4` |
| vLLM performance r2 | [`vllm_gemma_textonly_noinductor_fullgraph_ctx16384_out128_uniform_b32_cap19_final2_20260713.json`](vllm_gemma_textonly_noinductor_fullgraph_ctx16384_out128_uniform_b32_cap19_final2_20260713.json) | `daf330afcd5c67e6cb7e0d203f65d78bd6053ea59e938d1b0f309aa78e2a1e1f` |
| SGLang 0.78 admission failure | [`sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_cap19_20260713.json`](sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_cap19_20260713.json) | `e3b2d806ed5d8e53cacc885387210967b602205a8183f0dedfd9d629821bc2cd` |
| SGLang telemetry pass | [`sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_mem082_cap19_20260713.json`](sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_mem082_cap19_20260713.json) | `cac7e695846dfe1ad4a2f5d643c68b4143e5075c89742fb163e65d1d7139da63` |
| SGLang performance r1 | [`sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_mem082_cap19_final_20260713.json`](sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_mem082_cap19_final_20260713.json) | `cbb9c9e585f486b8295ac80befbe72f78a3c8153d763d02e5fb828442697558e` |
| SGLang performance r2 | [`sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_mem082_cap19_final2_20260713.json`](sglang_gemma_textonly_fullgraph_ctx16384_out128_uniform_b32_mem082_cap19_final2_20260713.json) | `33d9f122d65b71afe1aa019ea4a5fffd4389135a97d3caffdd477a4ec8975e0c` |

## Remaining work

The [`source/artifact provenance directory`](gemma_high_concurrency_b32_provenance_20260713/README.md)
records the pre-commit HEAD, binary tracked patch, promoted-file hashes, status,
and a verification script for the exact report-time worktree.

This is a direct-engine resident/offered cohort comparison, not a sustained
OpenAI-compatible HTTP ladder. The next production-path experiment should run
multiple B32 waves through each server while keeping engine-load memory and
request-time memory separate.

For WKVM, the immediate high-concurrency limit is now memory and prefill, not
decode correctness: even the best B32 run must remove at least 2.255 GiB to
pass the 18 GiB gate, and prefill dominates B32 wall time. The graph-generation
guard makes that optimization work safe to continue, but does not itself
improve the memory frontier.
