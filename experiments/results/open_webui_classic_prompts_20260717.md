# WKVM Open WebUI Live Demo Report

**Status:** PASSED

**Offered UI concurrency:** 4 chats

| Act | Offered concurrency | Success | Output tokens | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long context | 1 | 1/1 | 23 | 2.162 s | 2.162 s | 3.705 s | 3.705 s |
| Classic first turn | 4 | 4/4 | 2730 | 0.438 s | 0.495 s | 15.173 s | 18.509 s |
| Common follow-up | 4 | 4/4 | 2462 | 0.457 s | 0.459 s | 16.241 s | 16.651 s |

The four classic first turns are submitted as one synchronized UI cohort. Their browser-observed TTFT is p50 0.438 s and p95 0.495 s; E2E is p50 15.173 s and p95 18.509 s.

**Act 2 output length:** first-turn minimum 500 tokens; follow-up minimum 584 tokens; required minimum 500 tokens per turn.

## Runtime Evidence

Provider maxima are observed lifetime high-water gauges from the act's provider probe snapshots; request fields are counter deltas.

| Act | Whole-GPU baseline | Whole-GPU peak | Provider requests | Errors | Cancelled | Timed out | Max running | Max runnable rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long context | 17,575 MiB | 18,121 MiB | 1 | 0 | 0 | 0 | 1 | 1 |
| Concurrency | 18,117 MiB | 18,360 MiB | 8 | 0 | 0 | 0 | 4 | 4 |

| Concurrency provider phase | Requests | Errors | Cancelled | Timed out |
|---|---:|---:|---:|---:|
| Classic first turn | 4 | 0 | 0 | 0 |
| Common follow-up | 4 | 0 | 0 | 0 |

**Follow-up reuse:** 3 reuse hits; 1 session opened; 2,605 prefix tokens reused.

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
| Monty Hall Reasoning | first_turn | PASS | 500 |
| Common Follow-up | follow_up | PASS | 643 |
| Python Grouping | first_turn | PASS | 885 |
| Common Follow-up | follow_up | PASS | 601 |
| DR JSON Runbook | first_turn | PASS | 790 |
| Common Follow-up | follow_up | PASS | 634 |
| GPU Admission Control | first_turn | PASS | 555 |
| Common Follow-up | follow_up | PASS | 584 |

## Caveats

- This is a normal four-slot Open WebUI demo, not a controlled load test.
- WKVM serves this demo with routed_span_approximate model-state semantics.
- This capture is not a vLLM/SGLang comparison or proof of a 10x claim.
