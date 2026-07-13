# Gemma B16 Evidence Audit: WKVM vs vLLM and SGLang

Date: 2026-07-13

Durable evidence: [checksummed provenance bundle](README.md)

## Outcome

The current evidence does **not** prove a robust overall win against vLLM and
SGLang.

- At 16K, WKVM and vLLM are tied on end-to-end output throughput. WKVM's
  three-run mean is 64.769 tok/s versus one controlled-baseline vLLM result at
  64.713 tok/s, only +0.087%. The vLLM point lies inside WKVM's
  64.685-64.919 range.
- At 16K, WKVM has a clear decode-interval and memory advantage: 256.467 versus
  67.091 comparable decode tok/s (3.82x), and 16.871 versus 18.405 GiB mean/
  point whole-GPU engine delta. WKVM passes the 18 GiB gate; vLLM does not.
- At 32K, the current WKVM sample beats the archived vLLM and SGLang samples on
  E2E output throughput: 34.830 versus 24.543 and 26.130 tok/s. This is a
  single-run comparison, and the incumbent artifacts use a different 1,007 MiB
  GPU baseline versus 1,627 MiB for WKVM.
- SGLang decode remains non-comparable because its artifacts use separate-run
  subtraction. SGLang is compared only on E2E output throughput.
- The complete native 105-case quality grid passes at B2. A guarded 16K B16 run
  has scorer-visible semantic parity with B2, but not full token-exact parity
  and not full-grid B16 capacity proof.

This is strong evidence that bounded routed state creates context-flat decode
and a substantial memory advantage. It is not yet evidence that WKVM robustly
beats vLLM on the common 16K E2E metric.

## Benchmark contract

| Field | Value |
|---|---|
| Model | `/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it` |
| Dtype | BF16 |
| Prompt source | deterministic synthetic token IDs; no HF tokenizer |
| Shape | uniform 16,384 or 32,768 input tokens, B16, 128 output tokens |
| Sampling | greedy/fixed length, EOS ignored |
| Memory policy | `mem_cap_gib=19`, `headroom_gib=1`; green at `<=18.000 GiB` |
| Shared memory metric | `wkvm.whole_gpu_memory.v1`, whole physical GPU, `nvidia-smi`, 0.1 s samples |
| Prompt SHA at 16K | `cc789e0d6752a2593030a7365e3804e4b756a1328c6b172e552c7b8dffad4fd7` |
| Prompt SHA at 32K | `0928d1d3d33ce6e9c502f9b1558fa6fd6e6389aede1bacf15d25a349eb459c54` |

Every promoted row completes 16/16 requests with zero reported errors.

## Performance results

Values with three samples are `min / mean / max`. Single samples are shown
once. `Baseline` is whole-GPU used memory immediately before engine load.

| Context | Engine | n | E2E output tok/s | Comparable decode tok/s | Engine delta GiB | Baseline MiB | Gate |
|---:|---|---:|---:|---:|---:|---:|---|
| 16,384 | WKVM current | 3 | 64.685 / **64.769** / 64.919 | 255.878 / **256.467** / 257.221 | 16.800 / **16.871** / 17.008 | 1,620-1,627 | pass 3/3 |
| 16,384 | vLLM 0.24.0 controlled | 1 | **64.713** | **67.091** | **18.405** | 1,627 | fail |
| 16,384 | SGLang 0.5.14 archived | 1 | **26.435** | non-comparable (raw 37.577) | **18.737** | 1,007 | fail |
| 32,768 | WKVM current | 1 | **34.830** | **253.746** | **16.926** | 1,627 | pass |
| 32,768 | vLLM 0.24.0 archived | 1 | **24.543** | **25.345** | **18.407** | 1,007 | fail |
| 32,768 | SGLang 0.5.14 archived | 1 | **26.130** | non-comparable (raw 111.022) | **18.759** | 1,007 | fail |

### Interpretation

The controlled 16K E2E result is a tie. WKVM's mean lead is 0.056 tok/s, or
0.087%, and one vLLM run is insufficient to estimate its controlled-baseline
dispersion. The old 16K vLLM three-run set also overlaps the new WKVM range.
Do not call this a robust E2E win.

WKVM's 16K decode interval is 3.82x vLLM and its mean whole-GPU delta is
1.534 GiB lower. At 32K, WKVM is 41.9% above the archived vLLM E2E sample and
33.3% above the archived SGLang E2E sample; WKVM decode is 10.01x the comparable
vLLM value. Those 32K ratios need controlled-baseline repeats.

### Timing comparability

WKVM divides the 127 post-first-token outputs per successful request by the
same measured run's earliest-first-token to latest-finish interval. vLLM uses
the analogous interval from
`RequestOutput.metrics.first_token_ts/last_token_ts` and records
`decode_timing_comparable=true`.

SGLang records `decode_timing_comparable=false` and
`decode_timing_method=separate_run_subtraction`: a separate `max_tokens=1` run
is subtracted from a `max_tokens=128` run. Its raw decode estimates are retained
only as diagnostics and are excluded from speedup claims.

### Memory comparability

The 16K WKVM and controlled vLLM rows start at 1,620-1,627 MiB and have zero
monitor query errors, making their whole-GPU deltas directly useful. WKVM's
worst 16K delta is 17.008 GiB, leaving 0.992 GiB below the gate; vLLM exceeds
the gate by 0.405 GiB.

The 32K incumbent rows start at 1,007 MiB while WKVM starts at 1,627 MiB. Delta
subtraction reduces but does not eliminate different-background-process risk.
The 32K memory ordering is therefore supportive, not a final controlled memory
comparison. Whole-GPU polling is never process-attributed.

## Bounded-state evidence

| Evidence | 16K current | 32K current | Readout |
|---|---:|---:|---|
| Token-pool high-watermark | 28,866 in 3/3 | 29,052 | +186 slots (+0.64%) as context doubles |
| Whole-GPU delta | 16.871 GiB mean | 16.926 GiB | +0.055 GiB |
| Comparable decode | 256.467 tok/s mean | 253.746 tok/s | nearly context-flat |
| p50 prefill time | 23.456 s mean | 50.326 s | prompt compute grows with context |
| p50 decode time | 6.462 s mean | 6.535 s | decode remains nearly flat |
| Scheduler pressure | 0 backpressure, 0 retractions | 0 backpressure, 0 retractions | all 16 rows remain resident |

The strongest next-work leverage is prefill: bounded routed state has already
flattened decode and physical slot residency, while prefill time doubles with
context. A route-aligned 2,560-token diagnostic increased decode throughput but
reduced 16K E2E to 64.381 tok/s, so it is not promoted.

## Physical accounting boundary

The current `GemmaRoutedSpanConfig` now accounts for authoritative retained
span storage rather than the old eight-representative estimate. With
`sink=16`, `ring=1024`, `pending=512`, `m_slots=64`, and a 144-token per-slot
retention budget, the routed materialized bound is approximately 10,831 tokens
per row, not 2,128.

The 36,864-slot performance pool fully covers the uniform synthetic B16 timing
rows. Diverse quality rows are different: the first B16 quality wave needs
roughly 102K-107K slots for strict full-row token-pool coverage. The guarded
runtime detects partial coverage and executes those batches eagerly instead of
capturing an unsafe graph. Therefore:

- current uniform performance rows have full sliding/full token-pool coverage,
  one graph capture, and 127 replays;
- the 16K B16 quality slice records three
  `partial_token_pool_coverage` graph skips;
- no strict B16 full-row-capacity claim is made for diverse routed histories.

## Native quality evidence

### Full B2 grid

[`native_quality_fullgrid_b2_final_o1.json`](quality/native_quality_fullgrid_b2_final_o1.json)
uses the final O(1)-accounting source and records unchanged pre/post runtime
source identity `3be6a1b031fdcec8630c30a6f9bc6f03f16efbb723bf3bf3b49eae4fa7b38d9c`.

| Gate | Result | Required |
|---|---:|---:|
| Cases / cells | 105 / 45 | 105 / 45 |
| Overall cell mean | 0.911111 | 0.900000 |
| t1 mean / minimum cell | 1.000000 / 1.000000 | 1.000000 / 1.000000 |
| t2 mean / minimum cell | 0.911111 / 0.666667 | 0.888889 / 0.666667 |
| t3 mean / minimum cell | 0.822222 / 0.666667 | 0.800000 / 0.666667 |

All quality and runtime gates pass with zero violations.

### B16 scorer-visible parity

[`native_quality_16k_b16_final_o1.json`](quality/native_quality_16k_b16_final_o1.json)
uses the same final runtime-source identity and completes 35/35 16K cases at
B16. Against the corresponding rows in the full B2 grid:

- 35/35 decoded scored texts match;
- 35/35 scores match;
- 35/35 scorer-visible token prefixes match;
- 29/35 full fixed-length token sequences match;
- six sequences diverge only after the scorer stop point.

This is **scorer-visible semantic parity**, not full token-exact parity. It is
also a 16K slice, not a full 8K/16K/32K B16 grid. Native natural-document NLL
has not been rerun; the existing NLL report remains patched-HF PoC evidence.

## Output fingerprints

| Scope | Fingerprint evidence |
|---|---|
| 16K WKVM timing, 3/3 | complete; 16 requests/2,048 output IDs; `d83caf51c96411dc6ca24d68e5b320ffd93055e349d10e0ad96692c17a899cc4` |
| 16K controlled vLLM | complete; exact same 16-request/2,048-token hash as WKVM |
| 32K WKVM timing | complete; `757e3e47b14e35e7f53a4e10416fd06ea544ef7f1f1503601d8b0954ec701a4a` |
| 32K vLLM / both SGLang rows | generated-token hashes absent; output counts only |
| Final B2 quality | prompt `4319098823c7d416b58a07e8186fc2b2992df32d0232309c4debe83ddb40330c`; output `7bce902de9a915562437f0c6a95c21141fdcbd08528421ec0a1567c4c0e575c1`; break masks `687994b494fb4aa1b0131886f98688d24bd6406b260e49b532a2bcf52332e2be` |

The 16K WKVM/vLLM token match proves exact output identity for this synthetic
benchmark set. It does not imply general equality between routed approximate
attention and exact full-KV transformer semantics.

## Exact promoted artifacts

| Evidence | Original path | File SHA-256 |
|---|---|---|
| WKVM 16K r1 | `/tmp/wkvm-final-o1-accounting-16384-b16-probe.json` | `c25d3b90a736409875ac9e26451e348bc909e3fe605029499bbf43b9aff964a4` |
| WKVM 16K r2 | `/tmp/wkvm-final-o1-accounting-16384-b16-2.json` | `db0b3e55ca87e8134beb085da3828c19d5eb1b872dfb35131ede1cd2a6763332` |
| WKVM 16K r3 | `/tmp/wkvm-final-o1-accounting-16384-b16-3.json` | `6024f85f4d9d98b6c499144d0c43b185111c67af91f247e66310876fa5f88f1f` |
| WKVM 32K r1 | `/tmp/wkvm-final-o1-accounting-32768-b16-1.json` | `6d00a466225e4195e2654a7797dad064522c6d16d2d63fd7163c7edb36808633` |
| vLLM 16K controlled | `/tmp/vllm-final-baseline1620-16384-b16-1.json` | `cf79f25366ef1eabe4eeb8b1fe0f15b9c4ad79ab76094de87c1f1c5496185752` |
| vLLM 32K archived | `/tmp/vllm-gemma-32768-b16-current-monitor-1.json` | `159a2e155b36f0738b0aaaa35853aa989ddf77049d81bb75f4136e55dc16e3ad` |
| SGLang 16K archived | `/tmp/sglang-gemma-16384-b16-current-monitor-1.json` | `14229240976b4fa43d573eaa135adb76ebca90c547de66456aecfe9e579b0b27` |
| SGLang 32K archived | `/tmp/sglang-gemma-32768-b16-current-monitor-1.json` | `afc78da483f69b52b097d84fe2937b2a859a56e5786e986f49e6764d65611c40` |
| Final B2 quality | `/tmp/wkvm-native-quality-fullgrid-b2-final-o1.json` | `d6e845bac1525e8014b581c379338f147c5c9215e27e4855fc40240459ad9894` |
| Final 16K B16 quality | `/tmp/wkvm-native-quality-16k-b16-final-o1.json` | `85ce178c60488b90d2b9779df3b0fcae24115ac9720036a32f62a43b948fb734` |

Durable copies, exact launch commands, timestamps, and structured configs are
in the [bundle](README.md) and
[`provenance/commands.tsv`](provenance/commands.tsv).

## Engine configurations

### WKVM current

- native checkpoint/config loaders; no HF model construction or HF transformer
  forward;
- `sdpa_single_gqa`, separate projections;
- `sink=16`, `window=1024`, `m_slots=64`, `route_chunk=512`;
- prefill chunk 2,048, prefill microbatch rows 2;
- decode microbatch rows 16, 128 persistent padded decode steps, graph warmup 0;
- token-pool capacity 36,864, page size 16, strict flat/paged/split Triton;
- token-pool maximum context 16,640 at 16K and 33,024 at 32K;
- partial token-pool coverage is graph-ineligible and falls back to guarded
  eager decode.

### vLLM

vLLM 0.24.0 uses language-model-only loading, max 16 sequences, prefix caching
disabled, memory utilization 0.74, request metrics enabled, a short warmup, and
full CUDA graphs with capture sizes `[1,2,4,16]`. Maximum model length is
16,528/32,912.

### SGLang

SGLang 0.5.14 uses language-model-only loading, Triton attention, radix cache
disabled, max 16 running requests, static memory fraction 0.78, full decode
graphs, disabled prefill graphs, and a short warmup. Context is 16,528/32,912;
maximum total tokens are 264,448/526,592.

## Provenance and remaining limits

The bundle provides a full 15.99 GB model-file hash, GPU/driver/CUDA/PyTorch and
package identities, exact commands, copied JSONs, a binary tracked diff,
untracked-source copies, and a tracked-plus-untracked SHA-256 manifest.

Remaining limits:

1. Only WKVM has three current 16K runs; controlled vLLM has one.
2. All promoted 32K performance cells have one run; incumbents use a different
   whole-GPU baseline and pre-fix artifacts.
3. SGLang lacks same-run decode timestamps.
4. vLLM 32K and SGLang generated-token hashes are missing.
5. Warmup policy differs: incumbents warm separately; WKVM captures its graph
   in the measured workload.
6. Performance artifacts record committed HEAD rather than a runtime-source
   manifest; the bundle's post-run source snapshot and quality runtime identity
   provide the stronger current-source binding.
7. Full B16 quality capacity and native natural-document NLL remain unproven.

## Priority recommendations

1. Interleave at least three controlled-baseline WKVM and vLLM 16K runs. The
   current E2E ranges overlap, so this is the first requirement for a robust
   win claim.
2. Rerun vLLM and SGLang 32K from the same 1,627 MiB baseline with generated-
   token fingerprints and repeats.
3. Optimize prefill without spending the bounded-state memory advantage; the
   current 16K E2E tie is prefill-limited, not decode-limited.
4. Extend B16 scorer-visible parity to the full context/depth grid and rerun
   native natural-document NLL.
5. Either provision strict diverse-history B16 full-row coverage under the
   memory gate or keep the guarded partial-coverage eager boundary explicit.

Until the controlled 16K repeat sets separate, the correct headline is:
**WKVM ties vLLM at 16K E2E, wins decode and memory, and wins the current 32K
single-run E2E comparison.**
