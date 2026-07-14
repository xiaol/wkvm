# Gemma Sustained Multi-Turn B32 Comparison

Date: 2026-07-13

## Outcome

For this workload, vLLM completes the work faster than WKVM even though WKVM
retains many more long session histories.

- Every engine completes 256/256 session-turn requests with zero errors and
  emits the same 32,768-token output count.
- vLLM finishes in 388.190 seconds, WKVM in 564.990 seconds, and SGLang in
  583.324 seconds.
- vLLM is 176.800 seconds faster than WKVM. Its all-turn output throughput is
  45.5% higher and its continuation-only output throughput is 29.9% higher.
- WKVM retains 32/32 compact parked states and reuses all 224 continuations.
  vLLM profiles 3.255 provisioned full-history KV equivalents and SGLang
  profiles 2.236, but both incumbents still accept and finish every offered
  B32 turn by scheduling work in waves.
- WKVM is 3.24% faster overall and 16.8% faster on continuations than SGLang.
- This is a capacity result for WKVM, not a total-throughput win.

## Benchmark Contract

| Field | Value |
|---|---|
| Model | Gemma-4-E4B-it |
| Hardware | One NVIDIA GeForce RTX 4090 |
| Dtype | BF16 |
| Benchmark path | Direct engine; synchronized turn barriers; not HTTP |
| Logical sessions | 32 |
| Turns | 8 |
| Initial prompt | 13,824 tokens/session |
| Continuation delta | 32 new input tokens/session/turn |
| Output | 128 tokens/request, fixed length |
| Final logical history | 15,072 tokens after the eighth output |
| Sampling | Greedy: temperature 0, top-p 1, EOS ignored |
| Request order | Alternating forward/reverse by turn |
| Total completions | 32 sessions x 8 turns = 256 requests |
| Total generated output | 256 x 128 = 32,768 tokens |
| Unique application tokens | 442,368 initial input + 7,168 continuation input + 32,768 output = 482,304 |
| Source revision | `7b9e50b0195af428782bb30e56c443e90952f119` |
| Provenance | Tracked tree clean in all three final artifacts; unrelated untracked paths recorded separately |

The initial prompt fingerprint is identical in all three final artifacts:
`6f44af738ce415e1d5c859c2278265cd4d29ec8785a37d3ae82d4f2f4de808bd`.
Each engine appends its own greedy output to its next prompt, so turn 1 onward
has the same shape but not the same token content across engines.

WKVM uses approximate routed-span recurrent semantics. vLLM and SGLang use
full-KV transformer semantics. This report therefore compares capacity and
performance, not quality-equivalent inference.

## Completion And Throughput

`Output tok/s` is generated output tokens divided by the relevant wall time.
The full-run numerator is identical for every engine, so full-run wall time and
all-turn output tok/s give the same ordering.

| Engine | Success/error | Turn-0 wall / output tok/s | Continuation wall / output tok/s | Full wall / output tok/s | Unique app tok/s | Completed req/s |
|---|---:|---:|---:|---:|---:|---:|
| vLLM 0.24.0 | 256 / 0 | 51.123s / **80.120** | 337.066s / **85.063** | **388.190s / 84.412** | **1,242.444** | **0.659** |
| WKVM tuned m32 | 256 / 0 | 127.298s / 32.177 | 437.692s / 65.507 | 564.990s / 57.997 | 853.650 | 0.453 |
| SGLang 0.5.14 | 256 / 0 | 71.945s / 56.932 | 511.379s / 56.068 | 583.324s / 56.175 | 826.820 | 0.439 |

`Unique app tok/s` counts each initial input token once, each continuation
delta once, and every generated token once. It is useful application goodput,
not model-compute throughput: retained WKVM state and incumbent prefix KV mean
that logical input tokens do not correspond one-for-one with tokens recomputed
by the model.

The aggregate request-latency fields are:

| Engine | E2E p50 | E2E p95 | TTFT p50 | TTFT p95 | Whole-GPU peak delta |
|---|---:|---:|---:|---:|---:|
| vLLM 0.24.0 | 27.273s | 48.147s | 19.174s | 43.792s | 19,443 MiB |
| WKVM tuned m32 | 63.229s | 127.147s | 0.731s | 63.000s | 21,550 MiB |
| SGLang 0.5.14 | 37.734s | 71.135s | unavailable | unavailable | 19,646 MiB |

SGLang's direct non-streaming interface did not expose TTFT. Whole-GPU memory
comes from `nvidia-smi`; it includes unrelated device users and is not
process-attributed.

### Per-Turn Results

| Turn | Prompt tokens/request | WKVM wall / output tok/s | vLLM wall / output tok/s | SGLang wall / output tok/s |
|---:|---:|---:|---:|---:|
| 0 | 13,824 | 127.298s / 32.177 | **51.123s / 80.120** | 71.945s / 56.932 |
| 1 | 13,984 | 64.604s / 63.402 | **46.687s / 87.732** | 70.510s / 58.091 |
| 2 | 14,144 | 61.918s / 66.152 | **48.318s / 84.772** | 70.823s / 57.834 |
| 3 | 14,304 | 58.260s / 70.306 | **48.696s / 84.114** | 71.137s / 57.579 |
| 4 | 14,464 | 60.279s / 67.950 | **48.628s / 84.232** | 71.795s / 57.052 |
| 5 | 14,624 | 64.777s / 63.232 | **47.672s / 85.920** | 75.416s / 54.312 |
| 6 | 14,784 | 62.607s / 65.424 | **48.118s / 85.123** | 75.077s / 54.557 |
| 7 | 14,944 | 65.248s / 62.776 | **48.946s / 83.683** | 76.622s / 53.458 |

vLLM wins every synchronized turn. WKVM's retained-state path removes full
history reprefill after turn 0, but that saving does not overcome the current
engine's slower prefill and decode execution.

## Concurrency, Residency, And KV Capacity

These are three different quantities:

1. **Offered concurrency**: each turn submits a cohort of 32 requests to every
   engine.
2. **Resident session capacity**: WKVM proves that all 32 compact session
   states remain parked between turns.
3. **Full-history KV equivalents**: an incumbent's profiled KV-token capacity
   divided by one provisioned maximum-length request.

| Engine | Offered cohort | Retention/capacity evidence | Execution note |
|---|---:|---|---|
| WKVM tuned m32 | 32 | 32 resident and parked states; 224/224 reuse hits; zero reuse misses and zero full reprefills | Decode microbatch rows = 2, so 32 resident states do not mean 32 rows execute simultaneously |
| vLLM 0.24.0 | 32 | 49,114 KV tokens / 15,088 provisioned tokens = 3.255 full-history equivalents | Accepts all 32 and schedules waves; alternating order yields six full-prefix hits per continuation turn |
| SGLang 0.5.14 | 32 | 33,736 effective KV tokens / 15,088 provisioned tokens = 2.236 full-history equivalents | Accepts all 32 and schedules waves; alternating order yields about two full-prefix hits per continuation turn |

The provisioned 15,088-token length includes the engine's block-aligned
headroom around the 15,072-token final logical history. `3.255 KV equivalents`
does not mean 3.255 completed requests or a hard concurrency ceiling. It means
the static KV token pool can hold roughly 3.255 maximum-length histories at
one instant. Requests can still queue, run in waves, release blocks, and all
complete.

vLLM serving six full-prefix hits during a turn does not contradict its 3.255
static capacity. Those hits occur across a wave-scheduled cohort while blocks
are reused; six histories need not remain fully resident simultaneously.

WKVM's 32 retained histories are roughly 9.8 times vLLM's provisioned
full-history count and 14.3 times SGLang's. That is the architectural benefit
demonstrated here. It is not a claim that WKVM provides 9.8x throughput or
32-way simultaneous matrix execution.

## Request-Order Controls

A fixed forward scan can evict the incumbent cache tail just before those
sessions are revisited. The final comparison therefore alternates forward and
reverse request order. This is deliberately cache-friendly to vLLM/SGLang.

The first four turns of each final alternating run can be compared with the
separate four-turn forward-order controls:

| Engine | Order | Continuation output tok/s | Useful full-history hits/continuation turn |
|---|---|---:|---:|
| vLLM 0.24.0 | alternating | **85.511** | 6 |
| vLLM 0.24.0 | forward | 79.734 | 0 |
| SGLang 0.5.14 | alternating | **57.834** | about 2 |
| SGLang 0.5.14 | forward | 54.716 | effectively 0 |

All cache telemetry fields are available for every request. The controls show
that the chosen order helps rather than handicaps the incumbent caches. They
do not replace randomized-order repeats; see the limitations below.

## Low-Concurrency B3 Control And Parity

The B3 control uses the same 13,824 + 32 + 128 shape for four turns. vLLM's
52,084-token pool fits all three provisioned histories; SGLang's 34,296-token
pool retains about two.

| Engine | Full wall | All-turn output tok/s | Continuation output tok/s |
|---|---:|---:|---:|
| vLLM 0.24.0 | **17.485s** | **87.848** | **112.375** |
| SGLang 0.5.14 | 28.097s | 54.669 | 61.470 |
| WKVM | 34.936s | 43.967 | 56.339 |

When vLLM's full KV fits, WKVM has no speed advantage: vLLM continuation
throughput is almost 2x WKVM's. SGLang also narrowly exceeds WKVM despite
retaining only about two full prefixes. This supports the narrower conclusion
that WKVM's current benefit is retained-history capacity under memory pressure.

The clean tracked B3 WKVM run also compared parked-state continuation with a
fresh full-history WKVM rerun in report-only mode:

- 11/12 first continuation tokens are exact.
- 6/12 full 128-token sequences are exact.
- The check is not enforced and does not pass exact long-history equivalence.
- The final B32 m32 run has no fresh-history parity pass.

The result proves operational session reuse and bounded retained state, not
token-exact equivalence to fresh full-history execution.

## WKVM Capacity Tuning

The successful WKVM result is not the default m64 configuration. A clean-HEAD
default B32 attempt OOMs before recording turn 0. The successful run uses:

- `m_slots=32`;
- one prefill microbatch row;
- two decode microbatch rows;
- 32 persistent padded decode steps;
- nonpersistent padded full-attention rows;
- allocator cleanup before decode;
- token-pool capacity 65,536 with a recorded 24,453-slot high-water mark.

It reaches 23,515 MiB whole-GPU used, a 21,550 MiB delta from the recorded
baseline. After explicit session close, WKVM reports zero resident/parked
sessions, zero active cache bytes, zero active token-pool request slots, and
all 32 state slots free.

## Reproduction Commands

Use separate environments containing the recorded WKVM source, vLLM 0.24.0,
and SGLang 0.5.14. Set paths locally rather than copying the original machine's
absolute paths:

```bash
export MODEL_PATH=/path/to/gemma-4-E4B-it
export RESULT_DIR=/path/to/results
export WKVM_PY=/path/to/wkvm/python
export VLLM_PY=/path/to/vllm/python
export SGLANG_PY=/path/to/sglang/python

COMMON_FLAGS="--model-path $MODEL_PATH --sessions 32 --turns 8 \
--initial-context-tokens 13824 --turn-input-tokens 32 \
--output-tokens-per-turn 128 --request-order-policy alternating"

$WKVM_PY experiments/gemma_multiturn_bench.py --engine wkvm $COMMON_FLAGS \
  --m-slots 32 --prefill-microbatch-rows 1 --decode-microbatch-rows 2 \
  --persistent-padded-decode-steps 32 \
  --no-persistent-padded-full-attention-rows \
  --wkvm-empty-cache-before-decode \
  --json "$RESULT_DIR/multiturn_wkvm_b32_t8_alternating_m32.json"

$VLLM_PY experiments/gemma_multiturn_bench.py --engine vllm $COMMON_FLAGS \
  --json "$RESULT_DIR/multiturn_vllm_b32_t8_alternating.json"

$SGLANG_PY experiments/gemma_multiturn_bench.py --engine sglang $COMMON_FLAGS \
  --json "$RESULT_DIR/multiturn_sglang_b32_t8_alternating.json"
```

Forward-order controls replace `--turns 8 --request-order-policy alternating`
with `--turns 4 --request-order-policy forward` for vLLM and SGLang.

The focused validation command for the implementation is:

```bash
python -m unittest -q \
  tests.test_gemma_token_pool tests.test_gemma_routed_cache \
  tests.test_scheduler tests.test_gemma_engine \
  tests.test_gemma_multiturn_bench
```

## Audited Artifacts

The raw JSON files remain in the benchmark artifact store. These hashes anchor
the exact files used for this report.

| Role | Artifact basename | SHA-256 |
|---|---|---|
| Final WKVM B32 m32 | `multiturn_wkvm_b32_t8_alternating_m32_7b9e50b_20260713.json` | `49bf3291d3c68f1a55ca64d1bfbac1aff45369cffcec0e092bc8b737c65cb37e` |
| Final vLLM B32 | `multiturn_vllm_b32_t8_alternating_7b9e50b_20260713.json` | `46b7990d9233f03c463d1eea3d8e20b319c698fcfec875311e5c5b621657773e` |
| Final SGLang B32 | `multiturn_sglang_b32_t8_alternating_7b9e50b_20260713.json` | `a63cb521ca9015b70b70d67d4875c9c35f09f29aa0032ddb8e23d3489bffe473` |
| Default WKVM B32 OOM | `multiturn_wkvm_b32_t8_alternating_7b9e50b_20260713.json` | `8ac57bea925e6dc43816359f37ad270dab349fd9211b970195a771d12629168f` |
| vLLM forward control | `control_vllm_b32_t4_forward_7b9e50b_20260713.json` | `9c40ae07183e9ed8ccb787146dfa48b4d2e46675a48017a78fae70fde95bfa9c` |
| SGLang forward control | `control_sglang_b32_t4_forward_7b9e50b_20260713.json` | `09e1a821c53ee2bc8dd601eca136dba642f1c07f5d0d6fcd4b29a21f1e38a619` |
| WKVM B3 parity report | `multiturn_wkvm_b3_graph_parity_report_ca26d93_20260713.json` | `36468a83d4fb26f482e5d7e37ccd9b12dcbadf01ad67311f1d4ec5b067cacad1` |
| vLLM B3 | `multiturn_vllm_b3_alternating_ca26d93_20260713.json` | `6a98f861594d921165170f2beffb8968a38236bf8417c9825121823ece6695d5` |
| SGLang B3 | `multiturn_sglang_b3_alternating_ca26d93_20260713.json` | `55ccf65829ab1533d2dd44e292c2f64e8eb4676629ea6b56c2a072618c571c6e` |

## Limitations

- The final B32 table is one measured run per engine, not a repeated
  confidence interval.
- It is a direct-engine synchronized-round benchmark, not a sustained HTTP
  arrival-rate or production tail-latency test.
- Only turn-0 prompts are token-identical across engines; later histories are
  equal-shape autonomous trajectories.
- WKVM and the incumbents do not have equivalent attention semantics.
- B32 m32 does not have a passing fresh-history parity result.
- Alternating order is a deterministic cache-friendly policy. Forward controls
  exist, but randomized-order repeats do not.
- Whole-GPU memory is not process-attributed, and the successful WKVM run
  nearly fills the card.
- Capacity-equivalent figures describe profiled token pools, not observed
  simultaneous full-history execution and not a universal user limit.
