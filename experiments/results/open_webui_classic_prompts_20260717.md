# WKVM Open WebUI Live Demo Report

**Status:** PASSED

**Offered UI concurrency:** 4 chats

| Act | Offered concurrency | Success | Output tokens | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long context | 1 | 1/1 | 23 | 1.736 s | 1.736 s | 3.173 s | 3.173 s |
| Classic first turn | 4 | 4/4 | 157 | 0.367 s | 0.525 s | 2.047 s | 2.217 s |
| Common follow-up | 4 | 4/4 | 108 | 0.265 s | 0.269 s | 1.763 s | 1.926 s |

The four classic first turns are submitted as one synchronized UI cohort. Their browser-observed TTFT is p50 0.367 s and p95 0.525 s; E2E is p50 2.047 s and p95 2.217 s.

## Runtime Evidence

Provider maxima are observed lifetime high-water gauges from the act's provider probe snapshots; request fields are counter deltas.

| Act | Whole-GPU baseline | Whole-GPU peak | Provider requests | Errors | Cancelled | Timed out | Max running | Max runnable rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long context | 17,442 MiB | 18,246 MiB | 2 | 0 | 0 | 0 | 2 | 2 |
| Concurrency | 18,230 MiB | 18,340 MiB | 8 | 0 | 0 | 0 | 4 | 4 |

| Concurrency provider phase | Requests | Errors | Cancelled | Timed out |
|---|---:|---:|---:|---:|
| Classic first turn | 4 | 0 | 0 | 0 |
| Common follow-up | 4 | 0 | 0 | 0 |

**Follow-up reuse:** 4 reuse hits; 0 sessions opened; 333 prefix tokens reused.

**Capture health:** 0 capture errors; 0 probe errors.

**Launch semantic declaration:** `routed_span_approximate` (source: `scenario.claim_scope.semantics`).

**Observed provider engine config:** `persistent_padded_decode=true`; `persistent_padded_decode_steps=128`; `persistent_padded_decode_cuda_graph=false`; `use_native_gemma_forward=true`; `native_gemma_attention_backend=sdpa_single_gqa`; `native_gemma_projection_backend=separate`; `native_gemma_weight_backend=hf_live`; `native_gemma_checkpoint_loader=true`. Source: `capture.acts.concurrency.provider.after.metrics.values.engine`.

## Long-Context Source

The 12,000-token lane uses a contiguous natural-text excerpt, not repeated filler.

- Work: *Alice's Adventures in Wonderland* by Lewis Carroll
- Hugging Face dataset: `common-pile/project_gutenberg`
- Document ID: `11`
- Dataset revision: `01dc90a5002f8977c7fb03a372c14bca29c65cf1`
- Parquet revision: `d0bf09a2c2f6f73952733d7a1fe9a34b1cb4348c`
- License: `Public Domain`
- Document text SHA-256: `f17aa0bf7466424a8b357b688678666bad7a0148963ef349016a3098faa6bd1e`
- Selected body SHA-256: `39a7d2489030568740bd76d860a15e705e6fc2e1330051eb21ed66ae2f260034`

## Validation

| Prompt | Phase | Result | Output tokens |
|---|---|---:|---:|
| 12K Natural-Text Recall | long_prompt | PASS | 23 |
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
