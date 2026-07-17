# WKVM Open WebUI Live Demo Report

**Status:** PASSED

**Offered UI concurrency:** 4 chats

| Act | Offered concurrency | Success | Output tokens | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long context | 1 | 1/1 | 23 | 2.497 s | 2.497 s | 4.639 s | 4.639 s |
| Classic first turn | 4 | 4/4 | 157 | 0.703 s | 1.123 s | 3.527 s | 3.790 s |
| Common follow-up | 4 | 4/4 | 108 | 0.327 s | 0.332 s | 2.426 s | 2.693 s |

The four classic first turns are submitted as one synchronized UI cohort. Their browser-observed TTFT is p50 0.703 s and p95 1.123 s; E2E is p50 3.527 s and p95 3.790 s.

## Runtime Evidence

Provider maxima are observed lifetime high-water gauges from the act's provider probe snapshots; request fields are counter deltas.

| Act | Whole-GPU baseline | Whole-GPU peak | Provider requests | Errors | Cancelled | Timed out | Max running | Max runnable rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long context | 17,372 MiB | 17,769 MiB | 1 | 0 | 0 | 0 | 1 | 1 |
| Concurrency | 17,769 MiB | 17,805 MiB | 8 | 0 | 0 | 0 | 4 | 4 |

| Concurrency provider phase | Requests | Errors | Cancelled | Timed out |
|---|---:|---:|---:|---:|
| Classic first turn | 4 | 0 | 0 | 0 |
| Common follow-up | 4 | 0 | 0 | 0 |

**Follow-up reuse:** 4 reuse hits; 0 sessions opened; 333 prefix tokens reused.

**Capture health:** 0 capture errors; 0 probe errors.

**Launch semantic declaration:** `routed_span_approximate` (source: `scenario.claim_scope.semantics`).

**Observed provider engine config:** `persistent_padded_decode=true`; `persistent_padded_decode_steps=128`; `persistent_padded_decode_cuda_graph=false`; `use_native_gemma_forward=true`; `native_gemma_attention_backend=sdpa_single_gqa`; `native_gemma_projection_backend=separate`; `native_gemma_weight_backend=hf_live`; `native_gemma_checkpoint_loader=true`. Source: `capture.acts.concurrency.provider.after.metrics.values.engine`.

## Validation

| Prompt | Phase | Result | Output tokens |
|---|---|---:|---:|
| Long Context Needle | long_prompt | PASS | 23 |
| Reasoning | first_turn | PASS | 53 |
| Common Follow-up | follow_up | PASS | 37 |
| Code | first_turn | PASS | 58 |
| Common Follow-up | follow_up | PASS | 28 |
| JSON | first_turn | PASS | 14 |
| Common Follow-up | follow_up | PASS | 23 |
| Systems | first_turn | PASS | 32 |
| Common Follow-up | follow_up | PASS | 20 |

## Caveats

- This is a normal four-slot Open WebUI demo, not a controlled load test.
- WKVM serves this demo with routed_span_approximate model-state semantics.
- This capture is not a vLLM/SGLang comparison or proof of a 10x claim.
