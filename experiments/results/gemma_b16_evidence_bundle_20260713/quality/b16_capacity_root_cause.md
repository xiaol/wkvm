# B16 Native Quality Capacity Root Cause

Status: accounting corrected; strict diverse-history coverage remains
unresolved for the 36,864-slot performance pool.

Update: a partial-coverage graph guard now routes dense/partial batches through
eager decode. The guarded 16K B16 run restores scorer-visible semantic parity
with B2 while using eager fallback for three partial-coverage batches. This
fixes the unsafe graph behavior; it does not correct the stale physical
admission bound or prove strict full-row token-pool coverage at B16.

The exact B16/full-grid quality attempts do not reach scoring. They fail while
preparing or decoding diverse routed full-attention rows, so they provide no
quality-pass evidence.

## Failure mechanism

The first quality cohort contains sixteen independently routed 8K prompts.
Their materialized routed readouts are approximately 6.35K tokens per request;
the observed padded dense width reaches 6,656. Full-row token-pool preparation
therefore needs about 102,416 slots for that wave, or 106,752 under the observed
6,624-token upper bound plus per-request reserve. The performance configuration
provides only 36,864 slots.

When allocation fails,
`TokenPoolFullAttentionRowManager.try_prepare_decode_batch()` catches the
recoverable allocation exception, clears full rows, restores dense routed
storage, and returns `None`. Full attention then drops out of token-pool
coverage and padded decode uses a dense materialized fallback. The initial
fallback repeated grouped-query K/V and requested 832 MiB per tensor; a later
manual-GQA fix removed that expansion, but subsequent attempts still exhausted
the GPU from the aggregate diverse-row/materialization and graph lifetime.

## Historical admission-accounting bug

The pre-fix `GemmaMemoryModel.routed_materialized_tokens` estimated:

```text
sink + ring + pending + routed_slots * (1 + reps_per_slot)
= 16 + 1024 + 512 + 64 * 9
= 2,128 tokens
```

The runtime routed-span implementation may retain up to 144 span tokens per
slot. The corresponding bound is:

```text
16 + 1024 + 512 + 64 * (1 + 144) = 10,832 tokens
```

The root-cause trace reported the runtime ceiling as 10,831 because of the
implementation's endpoint convention. Current model accounting uses that
authoritative retention/readout bound instead of the old eight-representative
estimate. The separate 36,864-slot token pool is still too small to strictly
cover the observed diverse B16 wave.

Strict token-pool coverage for the observed first B16 quality wave requires
roughly 106,752 slots, adding about 1.07 GiB over the 36,864-slot pool. Covering
the full runtime bound would require approximately 174,320 slots, adding about
2.10 GiB. Neither alternative has been shown to satisfy the shared 18 GiB
whole-GPU gate.

## Evidence artifacts

| Artifact | Runtime progress | Failure |
|---|---|---|
| `history/native_quality_fullgrid_b16_cap36864_failure.json` | 8K prefill; no graph capture | 832 MiB OOM, process reported at 22.14 GiB |
| `history/native_quality_fullgrid_b16_route512_failure_after_graph.json` | B16 graph capture + 31 replays | 64 MiB OOM, process reported at 22.45 GiB |
| `history/native_quality_fullgrid_b16_route512_failure_after_cohort_gc.json` | progressed into 16K; 4 captures + 70 replays | 240 MiB OOM, process reported at 22.42 GiB |

Each artifact records a stable pre/post runtime-source identity for its own
attempt. None contains a completed score summary, `quality_validation`, or
`runtime_validation`, and none should be described as a quality result.

## Required closure

A quality-qualified B16 claim requires one of the following, followed by a new
full-grid run:

1. admit and execute the true materialized bound while remaining under the
   shared memory gate;
2. redesign full-row representation so diverse routed readouts stay bounded by
   the existing pool without dense fallback; or
3. establish a smaller representation with independently revalidated quality.

A lower-concurrency full grid can validate routed semantics, but it cannot by
itself certify B16 batch-shape parity or the B16 serving path.
