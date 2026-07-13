# Native Quality Status

## Passing final-source evidence

`native_quality_fullgrid_b2_final_o1.json` is the final O(1)-accounting native
grid: 105 cases, 45 cells, contexts 8K/16K/32K, five depths, three t1/t2 seeds,
and one t3 seed. Runtime source identity
`3be6a1b031fdcec8630c30a6f9bc6f03f16efbb723bf3bf3b49eae4fa7b38d9c`
is unchanged across the run. All 105 requests finish successfully and both
runtime and quality validation pass.

| Gate | Result | Required |
|---|---:|---:|
| Overall cell mean | 0.911111 | 0.900000 |
| t1 needle mean / minimum cell | 1.000000 / 1.000000 | 1.000000 / 1.000000 |
| t2 multi-key mean / minimum cell | 0.911111 / 0.666667 | 0.888889 / 0.666667 |
| t3 aggregate mean / minimum cell | 0.822222 / 0.666667 | 0.800000 / 0.666667 |

`native_quality_16k_b16_final_o1.json` uses the same final runtime source and
validates the guarded B16 partial-coverage path at 16K. Compared request-by-
request with the corresponding B2 full-grid rows:

- 35/35 scorer-visible decoded texts match;
- 35/35 scores match;
- 35/35 scorer-visible token prefixes match;
- 29/35 full fixed-length token sequences match;
- six sequences first diverge after their scoring stop point.

The correct claim is **scorer-visible semantic parity**, not full token-exact
parity. The B16 run records two graph captures, 47 replays, and three graph
skips with reason `partial_token_pool_coverage`. Full-attention token-pool
coverage is 47 batches/133 rows versus sliding coverage 117 batches/1,165 rows;
partial coverage executes eagerly by design.

## Historical diagnostics

`history/` preserves the original three-case smoke, exact-shape OOMs, the
unsafe pre-guard B16 quality collapse, the B2 diagnostic, and earlier-source
passing/parity artifacts. These are regression and provenance evidence, not
promoted results.

See [b16_capacity_root_cause.md](b16_capacity_root_cause.md) for the capacity and
fallback history. Current model accounting now uses the real approximately
10,831-token routed materialization bound. The 36,864-slot performance pool
still cannot strictly cover all diverse B16 full rows, so guarded eager fallback
remains part of the supported boundary.

## Remaining limits

- The complete quality grid passes at B2, not B16.
- B16 scorer-visible parity is proven only for the 16K 35-case slice.
- Six post-stop raw token sequences differ between final B2 and B16 runs.
- Native natural-document NLL has not been rerun; the existing NLL report is
  patched-HF PoC evidence.
- Strict diverse-history B16 full-row token-pool capacity is not proven under
  the 18 GiB gate.
