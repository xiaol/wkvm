# WKVM report claim ledger

This ledger defines which claims are promoted into `paper/main.tex`, the
artifact that supports each claim, and the caveat that must travel with it.
It is intentionally stricter than historical roadmap or marketing language.

## Promoted claims

| ID | Claim | Evidence | Required caveat |
|---|---|---|---|
| C1 | The implemented native RWKV-7 allocator admits requests by fixed state-family slots, and the scheduler advances one computed-token gap across prefill, decode, and resume. | `wkvm/core/arena.py`, `wkvm/core/scheduler.py`, `wkvm/core/request.py`; 60 focused current-checkout tests pass. | Native loader supports pure FLA-format RWKV-7, not general Mamba/GDN/hybrids. |
| C2 | The M2 checkpoint reports 8,077 tok/s for RWKV-7 1.5B at B256 on one RTX 4090 with 12.19 MiB state per slot. | `experiments/results/m2_engine_bench.md`, SHA-256 `899d26677dadcf27f1443a510a0b6efc20e5c7244ab87e91486d4aa41e508df7`, commit `cb421afc185b3acdbbf4d0239c0a6ea16018671f`. | Single 64-step timing; no repeats, confidence interval, model hash, or matched incumbent. CUDA graph captures model forward only. |
| C3 | The M3 demonstration created 2,000 191M snapshots at 2.29 MiB each, reported median warm/disk-backed restore of 8.2/8.6 ms, and reproduced 16/16 interrupted continuations. | `experiments/results/m3_results.md`, SHA-256 `15d336e000c7c4a9d6fd7e35938d75dcae7d28af16e3ea57f7a0c32520b951f9`, commit `9e129f5ba74b9e361a2ed157ba8573a46ada3efa`. | Single run. Disk-backed reload did not flush OS page cache. Exclude reported p99. Fingerprint is layout/dtype only. |
| C4 | Span-atomic routed KV reaches 0.90 overall synthetic recall versus 0.95 for full KV across 8K/16K/32K. | `experiments/results/quality_grid.md`, SHA-256 `a5826b978281ecb09ad0849651b799ac8f4b05ebd37be3f3be161fba3464f5c2`, commit `84dd3a12bf6ae2b9a1330d7de7c01de7daefa08d`. | Custom substring-scored synthetic tasks; t1/t2 use three seeds and t3 one seed. Not quality equivalence. |
| C5 | On 4090 B16/16K/128, WKVM and vLLM tie on E2E throughput while WKVM reports 3.82x comparable decode and 1.534 GiB lower whole-GPU engine delta. | `experiments/results/gemma_b16_evidence_audit_20260713.md`, SHA-256 `b5eef0bf411ca5be4edd626b12bd05284c9e4964fc0d06164ef448ea0d0fae46`, commit `2b4e642d6f9a2e53287e1755db5d711efcab88f4`. | WKVM has three runs; vLLM has one controlled run. Approximate versus full-KV semantics. |
| C6 | On clean repeated A800 B64/16K/32-output runs, WKVM is 4.467x vLLM on comparable decode but 0.793x on E2E throughput. | `experiments/results/gemma_a800_reliable_20260716/report.md`, SHA-256 `83371ce4686212cd3357d9f403c1dce7ef3e2512187b76cb837d9d0abb386776`, benchmark commit `234cc04867a93f1352d2a1c220c216b698f46560`. | Approximate versus full-KV semantics. Peak memory reflects configured pools and is not minimum required memory. |
| C7 | The strict Open WebUI B32x8 checkpoint reports 345.097 tok/s and 224/224 continuation reuse, 2.153x the tested vLLM profile and 5.305x the tested SGLang profile. | `experiments/results/open_webui_parent_token_b32_t8_20260723.md`, SHA-256 `aea5cce4adcba999f5d30720b36935a012bd58ff0498044917ae0439b37fa908`, commit `f128d1252d5aac7200eea214a7a504705580d39a`. | Exploratory single cross-run; external raw artifacts; autonomous later histories; approximate versus full KV; repeats pending. |

## Claims excluded from the report

- Any workload-independent or engine-wide “10x faster” statement.
- The provisional 12.107x A800 scout, which used a dirty worktree and has no
  matched SGLang result.
- The superseded 11.151x RTX 4090 vLLM ratio; the later exact-trace audit reduces
  it to 9.827x.
- Full-quality equivalence between Gemma routed-span and full KV.
- “Native recurrent mode” for Gemma. The implemented mechanism retains actual
  K/V spans and is not the RWKV-style segmented matrix-state bank described in
  older design documents.
- General native support for GDN, Mamba2, Qwen3-Next, or arbitrary hybrid
  architectures.
- Integrated whole-engine CUDA graphs for native RWKV. The promoted M2 graph
  captures fixed-shape model forward while scheduler and gather/scatter remain
  eager.
- Production GPU-to-host-to-NVMe tiering, live migration, automatic checkpoint
  boundaries, rollback, or copy-on-write GPU slots.
- Strong checkpoint compatibility. The current state-store fingerprint binds
  layout and dtype, not checkpoint weight identity.
- Trainer/server parity. The M3 experiment reports 0/8 rollouts within 5e-3.
- “First stateful serving,” “first recurrent prefix cache,” or “first
  linear-attention server.” Pensieve, Marconi, and KVBuffer are direct prior
  art.

## Required before a stronger revision

1. Interleave and randomize at least three runs for every promoted engine/cell.
2. Add confidence intervals or a predeclared nonparametric uncertainty summary.
3. Commit request-level artifacts for application and provider-HTTP results.
4. Run standard long-context quality benchmarks and native natural-document NLL.
5. Bind state handles to a cryptographic checkpoint manifest.
6. Measure truly cold storage by controlling the operating-system page cache.
7. Add offered-load/latency curves, minimum-memory sweeps, energy per token, and
   thermal/clock controls.
