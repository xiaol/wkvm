# A scoped route to 10x E2E

This document is the implementation and measurement plan for a defensible
10x result. It does not turn the current numbers into a blanket claim.

## Decision

WKVM cannot honestly claim 10x vLLM and SGLang on the current cold
`B64 / ctx16K / out32` full-request shape. The repeated A800 evidence is:

| Engine | Semantics | E2E output tok/s | Batch wall | Peak GPU |
|---|---|---:|---:|---:|
| WKVM native | `routed_span_approximate` | 21.886 | 93.57 s | 28.44 GiB |
| vLLM | `full_kv` | 27.608 | 74.18 s | 74.74 GiB |
| SGLang | `full_kv` | 15.587 | 131.39 s | 69.93 GiB |

The conservative ratios are 0.790x versus vLLM and 1.400x versus SGLang.
There are only 2,048 generated tokens in this cell. A 10x-vLLM result would
need at least 276.08 tok/s, or a batch wall below 7.42 seconds. The measured
WKVM wall would have to fall by roughly 12.6x. Decode-only work cannot remove
the approximately 85-second prefill.

The viable claim is narrower:

> On a named GPU and model, for a long-lived stateful continuation workload,
> WKVM routed-span approximate mode delivers at least 10x continuation E2E
> output throughput versus the measured vLLM and SGLang configurations under
> the same memory cap and request trace.

The wording must include `sessions`, initial context, turns, continuation input,
output length, request order, engine versions, memory cap, and semantic labels.
It must not imply full-KV equivalence or a general engine-wide speedup.

## What the comparison says

| Area | vLLM | SGLang | WKVM today | Required WKVM change |
|---|---|---|---|---|
| Batch representation | One flattened token vector with `q_lens`, cumulative offsets, positions, and slot mappings | `ForwardMode.MIXED` with `extend_lens`, `prefix_lens`, and `out_cache_loc` | `_execute` partitions decode and prefill; `prefill_batch_step` requires equal widths | Introduce one ragged batch contract and one model call for mixed rows |
| KV metadata | Backend-owned persistent block tables and slot mapping | Radix/request tables plus backend metadata | Metadata is built after scheduling and is still Gemma-engine-local | Keep fixed GPU metadata buffers alive across steps and graphs |
| Attention | Paged/ragged kernels handle prefill and decode together | FlashInfer/Triton backends handle mixed batches | Custom token-pool Triton plus dense/padded fallbacks | Use a paged/ragged backend for sliding and full/routed owners; retain a tested fallback |
| Scheduling | Token-budget loop and overlap between CPU planning and GPU execution | Mixed batch plus dual-stream/FutureMap overlap | CPU scheduler and GPU execution are serialized | Add bounded optimistic advance and a future metadata queue |
| State reuse | Prefix/KV cache, constrained by resident KV capacity | Radix cache, constrained by resident KV capacity | Compact parked state for every session, but decode rows are still small | Make compact state authoritative for every supported layer and raise effective decode batch size |

The relevant local code is `wkvm/gemma_engine.py:1207` (`_execute`),
`wkvm/gemma_engine.py:1276` (equal-width prefill), and
`wkvm/runner/gemma_runner.py:3017` (`prefill_batch_step`). The native layer
loop and attention dispatch are in `wkvm/runner/gemma_native_forward.py`.
The incumbent shapes are visible in the local vLLM
`vllm/v1/worker/gpu_model_runner.py` and
`vllm/v1/attention/ops/chunked_prefill_paged_decode.py`, and SGLang
`sglang/srt/managers/schedule_batch.py`.

## Implementation sequence

### P0 — Make every run explainable (implemented)

The multi-turn harness now records prefill/decode model-call counts, maximum
batch rows, graph captures/replays/skips, authoritative token-pool writes,
covered layer types, and full-attention coverage splits in each per-turn WKVM
barrier snapshot. A throughput change is not accepted unless these counters
show which execution path ran.

### P1 — Finish authoritative compact state

Sliding-layer authoritative prefill exists and can release dense sliding tails.
Complete the same ownership rule for routed/full layers:

1. Allocate the routed/full page table before the forward pass.
2. Write K/V directly to the owner page slots with one slot mapping.
3. Read the owner pages during decode; do not materialize a dense routed readout.
4. Release the source dense readout after a parity checkpoint.

Acceptance: serial-vs-authoritative logits agree on a bounded exact test; zero
full/routed mirror writes on the continuation path; compact state remains
resident for every requested session; no allocator growth during steady-state
decode.

### P2 — Add a ragged mixed batch (highest priority)

Add a backend-independent `MixedBatchMetadata` contract containing:

`request_ids`, flattened `input_ids`, `q_lens`, `cu_q_lens`, `prefix_lens`,
absolute `position_ids`, request indices, `initial`/`sample_mask`, per-layer
output slot mappings, and decode/prefill row indices. The contract must
preserve scheduler order and support `q_len=1` decode rows beside
`q_len>1` prefill rows.

The current `wkvm.core.mixed_batch` scaffold validates dependency-free row
offsets and sampling indices only. The GPU implementation still needs
per-layer `kv_indptr`/`kv_indices`, block tables, write-slot mappings, and a
rollback transaction around allocator, table, and cache updates; the scaffold
is not itself mixed execution.

The first implementation may use eager attention and the current rectangular
path as a fallback. It must not silently pad a ragged batch into full dense KV.
Add parity tests for `q_lens=[1,N]` and `[N,1,N]` across sliding, full/routed,
and shared-KV layers, checking logits, cache writes, and sampled tokens.

### P3 — Replace the attention hot path

Benchmark an opt-in FlashInfer paged decode/prefill backend against the current
Triton token-pool backend. Keep a shape- and dtype-gated fallback when the
Gemma head dimension or metadata layout is unsupported. Do not enable it by
default until the parity and memory gates pass.

The installed FlashInfer probe is encouraging but not a 10x argument: warmed
kernel-only gains range from about 1.0x to 1.9x against the current split path
at long contexts, and planning costs several seconds on first use. Gemma's
`head_dim=512` path must force CUDA-core mode. Plan once per static page shape,
keep metadata buffers graph-stable, and compare whole-step E2E rather than
kernel microseconds.

Target for the warm continuation cell: at least 1,000 aggregate output tok/s
at `B32 / initial ctx16K / continuation input32 / output128`, with no dense
full-history mirror and with all 32 states resident. This is the approximate
rate needed for a practical 10x result against the observed vLLM continuation
rate (about 85 tok/s); the exact target is recomputed from each incumbent run.

### P4 — Overlap planning and execution

After P2 is correct, advance scheduler bookkeeping optimistically while the
GPU executes the current ragged batch. Keep a bounded future queue and roll
back `num_computed_tokens` on a failed forward. Metadata construction must use
persistent GPU buffers; token values must never be synchronized to the CPU in
the hot loop.

Do not implement this as two unsynchronized CUDA streams over the existing
cache objects. The cache transaction and slot mapping must be part of the
mixed-batch contract first.

## Measurement gate

Use `experiments/gemma_multiturn_bench.py` for synchronized turns. The minimum
claim matrix is:

| Dimension | Values |
|---|---|
| Sessions | 32, 64, 96 (stop at the largest memory-safe common point) |
| Initial context | 16,384 and 32,768 |
| Turns | 8 and 16 |
| Continuation input | 32 tokens |
| Output per turn | 128, 512, 1,024 tokens |
| Order | alternating plus seeded shuffle controls |
| Repeats | three cold runs per engine and cell |

Report both turn 0 and continuation-only aggregates. The 10x continuation
gate is:

```text
min(WKVM continuation tok/s across repeats)
----------------------------------------------- >= 10
max(incumbent continuation tok/s across repeats)
```

Every artifact must have complete requests, stable output fingerprints,
matching workload fingerprints, exact token accounting, and passing WKVM reuse
invariants. A run with `full_reprefill_turns > 0`, a missing cache-telemetry
field, a graph fallback, or a semantic mismatch is not a passing claim cell.

Build the conservative report with:

```bash
python experiments/multiturn_10x_report.py "$OUT_DIR"/*.json \
  --markdown "$OUT_DIR/continuation_report.md" \
  --summary-json "$OUT_DIR/continuation_summary.json"
```

The command exits nonzero until all three engines have the required repeats and
both incumbent ratios pass. Use `--allow-fail` only for exploratory smoke
reports.

Before publication repeats, tune the incumbent prefill budgets under the same
whole-device memory ceiling. Sweep vLLM `--vllm-gpu-mem-util` together with
`--vllm-max-num-batched-tokens` (at least 8192, 16384, and 32768), and sweep
SGLang `--sglang-chunked-prefill-size` (at least 4096, 8192, and 16384) plus its
supported prefill graph modes. Select the fastest complete configuration that
stays below the ceiling, then freeze it for all three repeats. The artifacts
record the selected token budget/chunk size and the engines' measured cache
capacity; an automatic/default value is recorded as `null` rather than inferred.

Publication mode also requires the per-turn prompt fingerprints to match across
engines. Engine-local generated histories are useful shape scouts, but they are
not a fixed shared request trace and therefore cannot pass `--strict`.

Capture one complete source trace, then replay it through all three engines.
The replay forces each selected token before it enters that engine's cache, so
later prompts and resident cache histories are identical rather than merely
having the same synthetic turn deltas:

```bash
COMMON=(
  --model-path "$MODEL" --sessions 16 --turns 8
  --initial-context-tokens 32768 --turn-input-tokens 32
  --output-tokens-per-turn 128 --request-order-policy alternating
  --request-order-seed 0 --gpu-memory-device 0
  --gpu-memory-sample-interval-s 0.1
)
WKVM_FLAGS=(
  --slots 16 --m-slots 32 --route-chunk 2048 --chunk 2048
  --prefill-microbatch-rows 2 --continuation-prefill-microbatch-rows 8
  --decode-microbatch-rows 16 --persistent-padded-decode-steps 128
  --persistent-padded-decode-cuda-graph
  --persistent-padded-sliding-metadata-padding
  --token-pool-capacity 131072 --token-pool-max-context-len 34032
  --native-gemma-checkpoint-loader
  --native-gemma-attention-backend sdpa_single_gqa
  --native-gemma-projection-backend separate
  --enable-token-pool-attention --enable-token-pool-triton
  --enable-token-pool-paged-triton --enable-token-pool-paged-split-triton
  --token-pool-triton-strict --token-pool-sliding-paged-metadata-only
  --token-pool-route-boundary-batch
)
mkdir -p "$TRACE_DIR" "$OUT_DIR"

"$WKVM_PY" experiments/gemma_multiturn_bench.py --engine wkvm \
  "${COMMON[@]}" "${WKVM_FLAGS[@]}" \
  --write-shared-history-trace-json "$TRACE_DIR/b16_ctx32k_t8.trace.json" \
  --json "$TRACE_DIR/wkvm_trace_source.json"

"$WKVM_PY" experiments/gemma_multiturn_bench.py --engine wkvm \
  "${COMMON[@]}" "${WKVM_FLAGS[@]}" \
  --shared-history-trace-json "$TRACE_DIR/b16_ctx32k_t8.trace.json" \
  --json "$OUT_DIR/wkvm_r1.json"

"$VLLM_PY" experiments/gemma_multiturn_bench.py --engine vllm \
  "${COMMON[@]}" --max-model-len 34032 \
  --vllm-gpu-mem-util "$VLLM_GPU_MEM_UTIL" \
  --vllm-max-num-batched-tokens "$VLLM_MAX_BATCHED_TOKENS" \
  --vllm-language-model-only --vllm-disable-inductor \
  --shared-history-trace-json "$TRACE_DIR/b16_ctx32k_t8.trace.json" \
  --json "$OUT_DIR/vllm_r1.json"

"$SGLANG_PY" experiments/gemma_multiturn_bench.py --engine sglang \
  "${COMMON[@]}" --sglang-context-length 34032 \
  --sglang-max-total-tokens 544512 --sglang-mem-fraction 0.95 \
  --sglang-chunked-prefill-size "$SGLANG_CHUNKED_PREFILL_SIZE" \
  --sglang-max-running-requests 16 --sglang-attention-backend triton \
  --sglang-language-model-only --sglang-decode-graph full \
  --sglang-prefill-graph disabled \
  --shared-history-trace-json "$TRACE_DIR/b16_ctx32k_t8.trace.json" \
  --json "$OUT_DIR/sglang_r1.json"
```

Run each replay command in three fresh processes for the publication repeats;
keep the engine-generated trace-source artifact outside the report glob. The
trace is workload-bound and content-addressed, and `--strict` verifies its SHA,
per-turn selected-output proofs, and the resulting common prompt trace. The
vLLM and SGLang forcing hooks mutate only the one selected logit per row with a
single batched scatter; WKVM overwrites one still-pending sampled token before
cache commit. Artifacts record this timed mutation scope, and strict mode rejects
a full-vocabulary masking implementation.

For a publication candidate, add `--strict` and one explicit ceiling shared by
the entire cohort:

```bash
python experiments/multiturn_10x_report.py "$OUT_DIR"/*.json \
  --strict \
  --whole-device-memory-ceiling-mib "$MEMORY_CEILING_MIB" \
  --markdown "$OUT_DIR/continuation_report.md" \
  --summary-json "$OUT_DIR/continuation_summary.json"
```

The strict gate keeps the ratio checks but also requires every artifact to
record a clean source tree, an idle GPU baseline (at most 1 GiB), valid
whole-device peak/delta memory accounting below that ceiling, explicit enabled
cache configuration, complete engine-specific capacity telemetry, the expected
fixed history policy and semantic label, and a common GPU identity. WKVM
artifacts must additionally show zero fallback decode calls, zero full-attention
coverage splits, zero CUDA-graph skips, and an execution mode consistent with
the recorded mixed-batch counters. A workload with mixed opportunities must
execute them as `mixed_ragged`; this synchronized barrier workload has zero
mixed opportunities and legitimately records `partitioned_prefill_decode`.
Without `--strict`, these checks are still printed for diagnosis but do not turn
an exploratory report into a claim.

The cold full-request gate remains separate and intentionally strict. It is
useful evidence, but it is not replaced by a warm-state result.

## Latest shared-trace 36K scout

The strongest current scoped cell is an exploratory RTX 4090 run at
`B16 / ctx36,864 / delta32 / out64 / 2 turns`. SGLang generated the history
trace natively, then WKVM and vLLM replayed the same selected tokens. The
common trace SHA is
`3028af366ec24fb1e960ce3a7fc5124521c2263cbcd4cc611d2f34e41eec4046`.

| Engine | Trace role and semantics | Tuned configuration | Continuation tok/s | Whole-device peak |
|---|---|---|---:|---:|
| WKVM | forced replay; `routed_span_approximate` | continuation-prefill B8 | 169.933 | 23,729 MiB |
| vLLM | forced replay; `full_kv` | memory utilization 0.84; max batched tokens 4,096 | 11.708 | 23,894 MiB |
| SGLang | native trace source; `full_kv` | memory fraction 0.94; chunked prefill 2,048 | 6.168 | 23,934 MiB |

The resulting ratios are 14.514x versus tuned vLLM and 27.551x versus
SGLang. SGLang is the natural trace source because its sequence logits
processor has not remained reliable across scheduler overlap and preemption;
forcing there can attach the target to stale sequence state and diverge from
the trace. The report now supports one `native_trace_source` plus forced
replays from the other engines, and verifies that role contract together with
the common prompt and selected-output proofs.

This is exploratory evidence, not a publication result. There is only one
complete 36K repeat, the workload has only two turns, the source tree is dirty,
and every run began with more than 1 GiB already in use on the GPU. The report
therefore passes both ratio checks but fails the minimum-repeat, clean-tree,
and idle-baseline publication gates.

The 32K tuned fixed-trace cell has much less margin: WKVM measured
340.655 tok/s against vLLM at 33.014 tok/s, or 10.319x. A second successful
WKVM run measured 335.573 tok/s, only 10.165x against that same vLLM result.
Continuation-prefill B8 is also memory-fragile on this desktop: another B8
repeat failed with an allocation OOM unless expandable segments were used.
The longer `out128` scout is below the target after fairer vLLM provisioning:
357.864 tok/s versus 43.890 tok/s, or 8.154x. These cells are exploratory for
the same dirty-tree and non-idle-GPU reasons; none justifies a general 10x
inference claim.

The exact 36K cohort and report are:

```text
/home/xiaol/X/results/4090/sglang_native_trace_ctx36k_20260716/b16_ctx36k_t2_o64.trace.json
/home/xiaol/X/results/4090/sglang_native_trace_ctx36k_20260716/sglang_source_mem094_c2048.json
/home/xiaol/X/results/4090/sglang_native_trace_ctx36k_20260716/wkvm_replay_cprefill8_expandable.json
/home/xiaol/X/results/4090/sglang_native_trace_ctx36k_20260716/vllm_replay_mem084_mbt4096.json
/home/xiaol/X/results/4090/sglang_native_trace_ctx36k_20260716/exploratory_10x_report.md
/home/xiaol/X/results/4090/sglang_native_trace_ctx36k_20260716/exploratory_10x_summary.json
```

The comparison scouts are recorded at:

```text
/home/xiaol/X/results/4090/sglang_native_trace_20260716/exploratory_10x_report_tuned.md
/home/xiaol/X/results/4090/sglang_native_trace_20260716/wkvm_replay_cprefill8.json
/home/xiaol/X/results/4090/sglang_native_trace_20260716/wkvm_replay_cprefill8_expandable_r2.json
/home/xiaol/X/results/4090/sglang_native_trace_20260716/wkvm_replay_cprefill8_r2.json
/home/xiaol/X/results/4090/sglang_native_trace_20260716/vllm_replay_mem084_mbt4096.json
/home/xiaol/X/results/4090/warm_10x_probe_20260716/wkvm_b16_ctx32768_t8_o128_route2048_boundary_r2.json
/home/xiaol/X/results/4090/warm_10x_probe_20260716/vllm_b16_ctx32768_t2_o128_mem082.json
```

To turn the 36K scout into a publication candidate, capture an eight-turn
native SGLang trace, freeze the fastest memory-safe configurations and common
memory ceiling, run three fresh complete processes per engine, then repeat on
a clean tracked tree with an idle GPU baseline no greater than 1 GiB.

## Latest boundary-batching scout

On the local RTX 4090, the opt-in
`--token-pool-route-boundary-batch` path removed routed-fold single-row
fallbacks for `B16 / ctx32K / delta32 / out128 / 8 turns`, with
`route_chunk=2048` and token-pool capacity `131072`:

| Run | Continuation tok/s | Continuation wall | Decode calls | Fallback calls | Boundary batches | Peak delta |
|---|---:|---:|---:|---:|---:|---:|
| Before boundary batching | 305.544 | 46.9196 s | 1,181 | 176 | 0 | 21,070 MiB |
| Boundary batching repeat 1 | 358.498 | 39.9891 s | 1,016 | 0 | 11 | 20,997 MiB |
| Boundary batching repeat 2 | 357.864 | 40.0599 s | 1,016 | 0 | 11 | 20,557 MiB |

The two patched repeats produced identical prompt and output fingerprints. The
minimum patched aggregate is 12.57x the observed vLLM continuation rate
(28.480 tok/s) and 15.70x the observed SGLang rate (22.788 tok/s); even the
slowest patched continuation turn was 295.426 tok/s, or 10.37x vLLM.

That vLLM denominator used `gpu_memory_utilization=0.74` and is superseded for
claim planning by the `0.82` memory scout: KV capacity rose from 74,437 to
140,640 tokens, six of sixteen continuation histories hit the prefix cache, and
continuation throughput rose to 43.890 tok/s. Against the conservative WKVM
repeat above, the ratio is 8.15x, not 10x. Any new 10x gate must therefore beat
the fastest memory-safe tuned incumbent rather than reuse the 0.74 artifact.

This is still exploratory evidence: the worktree is dirty, the desktop GPU
baseline is above 1 GiB, and incumbents have not yet been rerun for the same
eight-turn fixed prompt trace. Its zero mixed opportunities legitimately yield
`partitioned_prefill_decode`; that mode is not a failure for this synchronized
barrier workload. Boundary batching changes the model-call batch shape at
routed folds, so its output trace is stable across patched repeats but is not
bitwise identical to the earlier single-row fallback trace. A focused
regression verifies that the routed tensors, centroids, counters, retained
spans, and route features after the fold are bit-identical to independent
single-row commits; the output divergence comes from B16-versus-B1 model-kernel
floating-point behavior before greedy sampling.

## Stop conditions

Do not publish 10x if any of these remain true:

- WKVM leaves an observed mixed prefill/decode opportunity unexecuted, or its
  recorded execution mode contradicts the mixed-batch counters.
- Routed/full layers still mirror dense readouts on every continuation step.
- Effective decode rows collapse below the advertised offered concurrency.
- The incumbent is denied prefix caching, a memory budget, or the same request
  order.
- The result compares `routed_span_approximate` to `full_kv` without saying so.

The current dense CUDA-graph eligibility patch and packed-projection/GQA probes
are useful correctness and instrumentation steps, but neither is a 10x path:
the measured isolated gains are single-digit percentages and attention remains
the dominant decode cost.

The latest local smoke demonstrates that the instrumentation is live, not that
the target is met: B8/2K/64 produced 313.877 continuation tok/s, with 8/8
authoritative prefill reservations, 16,384 authoritative tokens, 160 layer
writes, and 122 persistent graph replays. It observed zero mixed opportunities,
so `partitioned_prefill_decode` is the correct execution mode. It is a short,
single-engine smoke on a desktop GPU and is intentionally excluded from any
comparative claim.

## Latest paired three-repeat no-graph cohort

The paired RTX 4090 cohort clears the conservative continuation ratio gate at
`B16 / ctx36,864 / delta32 / out64 / 8 turns`. Each repeat has one native
SGLang source trace and matching vLLM and WKVM replays, with unique run
identities and a common 24,200 MiB whole-device ceiling:

| Engine | Min continuation tok/s | Median continuation tok/s | Max continuation tok/s |
|---|---:|---:|---:|
| WKVM no-graph | 267.179 | 273.354 | 279.358 |
| vLLM tuned | 22.635 | 22.842 | 23.754 |
| SGLang tuned/source | 11.609 | 11.792 | 11.833 |

Using the required conservative witness, `min(WKVM) / max(incumbent)`, the
ratios are 11.248x versus vLLM and 22.579x versus SGLang. The core gate passes:
all nine artifacts are complete, paired to their per-repeat traces, and below
the common ceiling. WKVM records zero request errors, fallback decode calls,
full-attention coverage splits, graph skips, graph shape mismatches, mixed
opportunities, and mixed model calls. Its truthful execution mode is therefore
`partitioned_prefill_decode`, not `mixed_ragged`.

vLLM memory utilization `0.84` was not repeatably memory-safe: the first run
completed, but the second repeat failed with an allocation OOM. Utilization
`0.82` completed all three repeats and is the frozen stable configuration with
`max_num_batched_tokens=4096`. The highest accepted whole-device peak was
24,059 MiB, below the 24,200 MiB campaign ceiling.

The graph-enabled WKVM run reached only 235.997 tok/s. Across the workload it
captured 28 graphs: one ordinary capture per turn plus two extra captures for
each of 10 routed-fold boundaries. There are only 504 continuation decode
steps, so capture and row-rebuild costs do not amortize. The frozen campaign
therefore uses `--no-persistent-padded-decode-cuda-graph`; longer-output cells
must independently re-evaluate the graph break-even point.

Increasing `route_chunk` to 4,096 is not a viable substitute on this card. The
otherwise identical replay failed at the first continuation prefill while
requesting another 112 MiB, after reaching a 24,010 MiB whole-device peak. It
also changes the routed-span state profile, so it would require a separately
declared semantic/configuration cohort even if it fit.

This result is still not a 10x cold or server-wide E2E result. The strict report
fails only the publication provenance conditions: the source worktree was
dirty and every run began above the 1 GiB idle-GPU baseline limit. The paired
repeat matrix, benchmark identities, raw trace integrity, stable engine
configurations, common ceiling, execution contract, and both 10x ratios pass.
A claim run must repeat this frozen cohort from a clean identical commit with
an idle GPU baseline no greater than 1 GiB.

The paired evidence and strict publication report are recorded at:

```text
/home/xiaol/X/results/4090/wkvm_10x_e2e_triton_nograph_r3b_20260716/strict_10x_report.md
/home/xiaol/X/results/4090/wkvm_10x_e2e_triton_nograph_r3b_20260716/strict_10x_summary.json
```

## Latest no-allocator-purge improvement

The next RTX 4090 optimization removes `--wkvm-empty-cache-before-decode` from
the otherwise identical winning configuration. Three fresh WKVM processes
replayed the same three paired traces while the incumbent artifacts remained
fixed:

| Engine | Min continuation tok/s | Median continuation tok/s | Max continuation tok/s |
|---|---:|---:|---:|
| WKVM no graph, no allocator purge | 281.263 | 285.043 | 286.476 |
| vLLM stable 0.82 | 22.635 | 22.842 | 23.754 |
| SGLang tuned/source | 11.609 | 11.792 | 11.833 |

The conservative ratios rise to 11.840x versus vLLM and 23.769x versus
SGLang. All three WKVM runs completed with identical shared-trace outputs,
zero fallbacks, zero full reprefills, zero mixed opportunities, and the same
10 routed-boundary batches. The maximum whole-device peak was 23,578 MiB,
still below the 24,200 MiB ceiling. Continuation p50 E2E latency in the median
run fell to 3.232 seconds.

This shows that allocator purging was synchronization overhead rather than
required memory headroom for this cell. The frozen runner therefore no longer
empties the CUDA allocator between continuation prefill and decode. The new
evidence remains exploratory for the same dirty-tree and idle-GPU reasons, and
publication still requires complete fresh incumbent repeats in the final clean
campaign.

```text
/home/xiaol/X/results/4090/wkvm_10x_tuning_20260717/exploratory_no_empty_cache_10x_report.md
/home/xiaol/X/results/4090/wkvm_10x_tuning_20260717/exploratory_no_empty_cache_10x_summary.json
```

## Why high concurrency wins this cell

The B16 result is a capacity result amplified by batching, not a 10x raw-kernel
result. All three engines are offered the same 16 requests. WKVM also executes
all 8,064 continuation decode rows in 504 model calls, exactly 16 rows per call,
but the decisive difference is how much long-history state remains resident:

| Engine | Resident-history evidence | Continuation prefix reuse | Observed consequence |
|---|---|---:|---|
| WKVM | 16/16 compact parked states | 99.91% | Every continuation computes only the 32-token delta plus the pending token |
| vLLM | 215,680 KV tokens, or 5.73 full 37.6K histories | 40.10% | The median request has zero cached prefix on every continuation turn |
| SGLang | 56,140 effective tokens, or about 1.49 full histories | about 9% | The median request retains only one cached token |

This is why increasing concurrency can raise WKVM throughput while hurting the
incumbents: once offered long-lived sessions exceed full-KV residency, vLLM and
SGLang repeatedly evict and re-prefill histories while WKVM continues to batch
resident compact states. It does not imply that higher concurrency always
helps. Below the incumbent residency limit, or when WKVM decode rows collapse
into small microbatches, the gap can shrink or reverse. A `{B1, B4, B8, B16}`
ladder is still required to attribute the crossover independently.

The scope boundary is visible in the same artifacts. Conservative all-eight-
turn ratios are only 3.960x versus vLLM and 7.505x versus SGLang, and cold
turn-0 ratios are 0.982x and 1.362x. At the current B16 rates, roughly 125
continuation turns would be needed merely to amortize turn 0 into a 10x
all-turn ratio against vLLM. Do not present this as cold-request or general
server E2E.

## Route to client-observed 10x E2E

The shortest defensible next claim is provider-HTTP warm continuation E2E on
the exact winning workload. It must measure from client request dispatch until
the final streamed token, including HTTP parsing, queueing, engine scheduling,
SSE serialization, and transport, but excluding model startup and the initial
session-building turn.

The existing Open WebUI B32 x 8 result is not that measurement. It used a
shorter 13,824-token context and an older WKVM profile with
`sdpa_single_gqa`, `route_chunk=512`, graph capture enabled, prefill B1, and
decode microbatch B2. WKVM reached only 53.963 output tok/s versus vLLM at
92.065 tok/s. More importantly, Open WebUI decoded, normalized, persisted, and
re-tokenized assistant text; only 125 of 224 continuation turns matched the
exact parked-token prefix, so 99 turns safely discarded state and restarted.

Implement the server gate in this order:

1. Expose the existing token-native session path through `/v1/stream` with an
   explicit `session_id`, so the benchmark never round-trips generated tokens
   through text before the next turn.
2. Add a multi-turn HTTP harness that reuses the content-addressed trace and
   records per-turn client wall time, output tok/s, TTFT, p50/p95 latency,
   errors, exact token counts, GPU memory, and session-reuse telemetry.
3. Use the existing bounded teacher-forcing processors through vLLM's allowed
   logits-processor path and SGLang's custom-logit-processor path, so all three
   servers receive the same cumulative token history without full-vocabulary
   masking.
4. Start WKVM with the winning direct-engine profile: `triton_dense_gqa`,
   continuation-prefill B8, `route_chunk=2048`, boundary batching, token-pool
   capacity 114,688, decode B16, and no persistent graph for this out64 cell.
5. Run `{B1, B4, B8, B16}` at ctx36,864, delta32, out64, eight turns, with
   three fresh server processes per engine and the common 24,200 MiB ceiling.
6. Require `min(WKVM) / max(incumbent) >= 10` on client-observed continuation
   throughput, plus complete requests and p95 latency. Keep turn 0 and the
   all-turn aggregate visible as explicit non-10x controls.

The present direct-engine witness allows only about 0.48 seconds of additional
WKVM wall time per B16 continuation turn before the stable vLLM ratio falls
below 10x. If vLLM utilization 0.84 becomes repeatably safe on an idle GPU, that
budget shrinks to about 0.21 seconds per turn. Target at least 300 continuation
output tok/s before adding the application layer, then profile before changing
the workload. The first fixes are to remove avoidable allocator-emptying stalls,
keep token-pool metadata buffers and decode groups persistent across routed
folds, execute real mixed opportunities for staggered arrivals, and raise the
effective decode row width. CUDA graphs should remain disabled for out64 until
captures stop recurring at routed-fold boundaries or a longer-output cell
proves that capture cost amortizes.

After provider HTTP passes, fix the application contract separately. Open
WebUI or a thin adapter must send a session identifier plus a canonical token
or message delta tied to a parent-history digest. WKVM should own the canonical
parked history instead of accepting a decode-to-text-to-token reconstruction as
equivalent state. Only then rerun the full Open WebUI path. Even with that fix,
the honest target remains warm continuation E2E; a 10x cold unique-prompt claim
would require a fundamentally faster initial prefill or a separately disclosed
precomputed-state/snapshot workload.

Every performance result also needs a separately predeclared quality floor.
Teacher forcing makes histories identical for timing, but it does not establish
quality equivalence. Publish unforced long-context task scores or an agreed
logit/output-retention metric beside the speed table, and keep
`routed_span_approximate` in the claim text.

## Provider-HTTP result after admission fixes

The provider-HTTP scout now has an exact SGLang-native source trace and exact
WKVM/vLLM replays for `B16 / ctx36,864 / delta32 / out64 / 8 turns` on the RTX
4090. SGLang used its stock token-only `/generate` endpoint; WKVM used stateful
`/v1/stream`; vLLM used `/v1/completions`. Every valid artifact returned exact
token IDs and stayed below the 24,200 MiB whole-device ceiling.

| Engine | Valid repeats | Continuation tok/s min / median / max | Worst p50 / p95 E2E | Peak GPU |
|---|---:|---:|---:|---:|
| WKVM | 2 | 193.133 / 282.217 / 371.301 | 5.165 / 5.945 s | 23,715 MiB |
| vLLM | 1 | 15.795 / 15.795 / 15.795 | 55.309 / 66.936 s | 23,225 MiB |
| SGLang | 1 | 11.069 / 11.069 / 11.069 | 48.764 / 91.559 s | 23,511 MiB |

The exploratory conservative ratios are 12.227x versus vLLM and 17.447x
versus SGLang. The slow WKVM repeat is retained in the minimum. A third WKVM
repeat is excluded because two cold-turn clients observed pre-overwrite
candidate tokens; server telemetry proved the final selected tokens were
correct. The stream race is fixed by publishing only the processed forced-token
prefix, and the focused server/HTTP suite now passes 73 tests.

The main HTTP throughput loss was not model execution. Python's default listen
backlog was five, so four requests in a synchronized B16 wave could arrive
about one second late and permanently fragment the decode cohort. The server
now raises the listen backlog to at least `max_queue`. With backlog 64 and the
original 10 ms collection window, the best run used 505 batched decode calls,
zero single-row fallbacks, and 112/112 continuation session reuses.

The capacity contrast is explicit in server telemetry:

| Engine | Long-context capacity evidence | Consequence |
|---|---:|---|
| WKVM | 16 resident compact states | Full B16 continuation decode cohorts |
| vLLM | 17,617 blocks x 16 tokens; about 7 full 37.6K histories observed running | Remaining requests queue and most prefixes are evicted |
| SGLang | 56,908 effective KV tokens; about 1.51 full histories | Requests are processed almost serially with near-full reprefill |

This validates the scoped claim only:

> WKVM delivers at least 10x provider-HTTP warm stateful continuation output
> throughput on this named workload, hardware, memory ceiling, and
> `routed_span_approximate` semantic mode.

It is still not 10x for the complete eight-turn session. Using the worst valid
WKVM wall, the all-turn ratios are 5.028x versus vLLM and 7.140x versus
SGLang. Turn 0 alone is approximately tied with vLLM. A true cold/all-turn 10x
claim therefore needs a substantially faster initial prefill, a longer
predeclared session that amortizes turn 0, or both. Do not shorten the wording
to "WKVM is 10x vLLM/SGLang."

The exploratory report is:

```text
/home/xiaol/X/results/4090/wkvm_10x_http_20260717/provider_report/exploratory_report.md
/home/xiaol/X/results/4090/wkvm_10x_http_20260717/provider_report/exploratory_summary.json
```

Its ratio checks pass, but the overall gate intentionally fails because this
was not one unified three-repeat campaign, the source tree is dirty, and the
prelaunch desktop GPU baseline is about 2.9 GiB rather than at most 1 GiB. The
frozen `scripts/run_wkvm_10x_http_4090.sh` runner exists to produce the final
SGLang-source/WKVM-replay/vLLM-replay cohort from fresh processes with common
campaign identities and strict memory provenance.

## Fair KV-sharing fast-prefill scout

The July 17 fair scout changes the cold-path conclusion materially. It uses
the same provider-HTTP trace, RTX 4090, BF16 model, B16 concurrency, and 24,200
MiB whole-device ceiling as the earlier result. vLLM's Gemma4 KV-sharing fast
prefill is enabled, and WKVM now omits the shared tail, final norm, and LM head
on every intermediate prompt chunk. WKVM runs the selected shared tail only on
closing chunks with `Q >= 128`; the Q33 continuation chunks deliberately use
the ordinary full path because the selected-tail path regressed that shape.

The aligned r1 artifacts are:

| Engine/profile | Turn 0 wall | Continuation wall | Full wall | Continuation tok/s | Full tok/s | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|
| WKVM owner-only intermediate chunks | 41.359 s | 19.462 s | 60.821 s | 368.306 | 134.690 | 23,657 MiB |
| vLLM 0.24.0 KV-sharing fast prefill | 40.402 s | 270.340 s | 310.742 s | 26.515 | 26.363 | 23,220 MiB |
| SGLang 0.5.14 | 90.637 s | 639.772 s | 730.409 s | 11.204 | 11.216 | 23,514 MiB |

The provider-HTTP continuation ratios are 13.891x versus vLLM and 32.873x
versus SGLang. The full eight-turn ratios are 5.109x versus vLLM and 12.009x
versus SGLang. WKVM is therefore now 10x SGLang for the complete named session,
but it is still not 10x vLLM for the complete session. Turn 0 is effectively a
tie: vLLM is only 0.957 seconds faster.

Three fresh WKVM processes make the improvement repeatable:

| Metric | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Continuation tok/s | 368.306 | 369.357 | 371.325 |
| Full wall | 59.506 s | 59.868 s | 60.821 s |
| Whole-device peak | 23,648 MiB | 23,657 MiB | 23,732 MiB |

All three runs completed 128/128 requests with zero errors, identical output
fingerprints, the same logical trace SHA, and the intended execution counters:
136 owner-only calls over 552,960 tokens, zero owner-only fallbacks, nine
closing selected-tail calls, and 14 Q33 fallbacks. The extra WKVM repeats are
stability evidence only; they replay the r1 source trace and therefore do not
replace the required per-repeat SGLang source artifacts in a strict campaign.

### Why the first split failed

The first correct 24/18 split still ran the 18-layer Q1 tail and LM head on
every 2K prompt chunk. It reduced layer-token work but regressed the measured
profile:

| WKVM profile | Turn 0 | Continuations | Full wall | Continuation tok/s |
|---|---:|---:|---:|---:|
| Fast prefill disabled | 62.471 s | 19.415 s | 81.886 s | 369.203 |
| Selected tail on every chunk | 63.957 s | 23.389 s | 87.347 s | 306.463 |
| Owner-only intermediate chunks | 41.359 s | 19.462 s | 60.821 s | 368.306 |

The all-chunk selected-tail implementation was 6.67% slower overall and 16.99%
slower on continuation throughput. Owner-only chunking is 1.510x faster on turn
0 and 1.346x faster for the full session than the disabled control, while
preserving continuation throughput. This demonstrates why theoretical
layer-token counts were insufficient: work that produces no sampled logit must
be removed at the scheduling/model boundary, not merely reduced to Q1.

### Fair vLLM configuration

The memory-safe working vLLM configuration is:

```text
--kv-sharing-fast-prefill
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY",...}'
--gpu-memory-utilization 0.82
--max-num-batched-tokens 4096
```

It exposes 209,590 KV tokens, or 5.57 complete 37,616-token histories. Two
other apparently attractive configurations are invalid on the installed
vLLM 0.24.0 build:

- `mode=3 / FULL_AND_PIECEWISE` compiles both decoder portions but reports
  negative 0.86 GiB available KV memory at utilization 0.82.
- `mode=0 / FULL` fails during CUDA-graph memory profiling because its dummy
  prefill lacks `logits_indices_padded` required by the fast-prefill metadata.

`FULL_DECODE_ONLY` retains full decode graphs while keeping prefill eager and
fits below the common ceiling. The frozen HTTP runner now selects this mode by
default when vLLM fast prefill is enabled, records the configuration in every
artifact, and keeps WKVM owner-only fast prefill enabled by default.

The WKVM HTTP launch must also export the token-pool backend policy explicitly:
`WKVM_ENABLE_TOKEN_POOL_TRITON=1`, `WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON=1`,
`WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON=1`,
`WKVM_TOKEN_POOL_TRITON_STRICT=1`,
`WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY=1`, and
`WKVM_TOKEN_POOL_ROUTE_BOUNDARY_BATCH=1`. These controls are environment-based;
the July 17 exploratory launch did not record them in its command, so its
low-level dispatch mix is not publication provenance. The runner now exports
them and the regression test asserts their presence.

### Route from here

High concurrency is now doing exactly what the architecture promises: WKVM
keeps 16 compact session states resident while vLLM holds only 5.57 full
histories and SGLang about 1.5. It does not make the one-time cold computation
10x faster. The owner-only change removes avoidable cold work and brings WKVM
turn 0 to parity with vLLM; the remaining 13.9x continuation advantage comes
from resident-state capacity and batching.

There are two honest next claims:

1. Publish the warm provider-HTTP continuation result after three complete,
   paired, clean-tree, headless repeats for all engines.
2. If complete-session 10x versus both engines is required, predeclare a
   genuinely long-lived session. A linear extrapolation of this single fair
   cohort crosses 10x vLLM at 36 total turns; use at least 48 turns for margin
   and measure it rather than publishing the extrapolation.

For the fixed eight-turn BF16 workload, 10x vLLM would require WKVM full wall
at or below 31.074 seconds. The measured cold turn alone is 40.202--41.359
seconds, so no concurrency, SSE, scheduler, or small fusion adjustment can
close that gap. A fixed-eight-turn 10x result needs a fundamentally different
cold path such as a quality-gated FP8 implementation compared against FP8
incumbents, or a separately disclosed precomputed-state/resume workload.

Artifacts:

```text
/home/xiaol/X/results/4090/wkvm_fast_prefill_fair_http_20260717_r1/owner_only_fair_report.md
/home/xiaol/X/results/4090/wkvm_fast_prefill_fair_http_20260717_r1/owner_only_fair_summary.json
/home/xiaol/X/results/4090/wkvm_fast_prefill_fair_http_20260717_r1/artifacts/wkvm-owner-only-fast-r1.json
/home/xiaol/X/results/4090/wkvm_fast_prefill_fair_http_20260717_r1/artifacts/vllm-fast-full-decode-only-r1.json
/home/xiaol/X/results/4090/wkvm_fast_prefill_fair_http_20260717_r1/artifacts/sglang-source-r1.json
```

## Measured 48-turn complete-session crossover

The predeclared long-lived campaign completed on the RTX 4090. It used B16,
48 synchronized turns, 36,864 initial tokens/session, 32 new input tokens per
continuation, 64 output tokens/request, and a 24,200 MiB whole-device ceiling.
SGLang generated the canonical trace; WKVM and vLLM replayed the same 48-turn
token IDs from fresh processes.

| Engine | Semantic mode | Turn 0 | Continuations | Full wall | Full tok/s | Peak GPU |
|---|---|---:|---:|---:|---:|---:|
| WKVM | `routed_span_approximate` | 40.265 s | 140.150 s | **180.415 s** | **272.439** | 23,856 MiB |
| vLLM 0.24.0 | `full_kv` | 40.469 s | 1,971.421 s | 2,011.890 s | 24.431 | 23,200 MiB |
| SGLang 0.5.14 | `full_kv` | 90.711 s | 4,614.412 s | 4,705.123 s | 10.446 | 23,597 MiB |

The complete-session ratios are 11.151462x versus vLLM and 26.079455x versus
SGLang. All three engines complete 768/768 requests with zero errors, exact
output-ID accounting, one shared trace SHA, and valid memory measurements.
Every report check passes.

This establishes the scoped exploratory claim:

> WKVM delivers at least 10x provider-HTTP complete-session output throughput
> on the named 48-turn long-lived workload, hardware, and memory ceiling, in
> `routed_span_approximate` mode versus vLLM/SGLang `full_kv` mode.

It does not establish fixed-eight-turn 10x versus vLLM, same-semantics 10x, or
a universal engine claim. It is one paired cohort from a dirty, non-headless
tree. Publication still requires three paired clean-tree headless repeats.

The detailed report is
`experiments/results/gemma_4090_48turn_10x_20260717.md`. Generated artifacts
are under:

```text
/home/xiaol/X/results/4090/wkvm_10x_e2e_t48_20260717_r1/
```
