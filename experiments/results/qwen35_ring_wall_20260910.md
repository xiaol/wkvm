# Ten times on Qwen3.5-9B: bounded guest memory past the capacity wall (2026-09-10)

The main-branch 10x rows (Gemma routed-span vs vLLM full KV) came from one
mechanism: a bounded per-session state that never re-prefills, measured on a
long multi-turn workload past the incumbent's KV capacity. This note repeats
that mechanism on the hybrid: Qwen3.5-9B with the 8 full-attention layers
in **ring mode** (16 sink tokens + a 1024-token sliding window per session,
`guest_mode="ring"`), vs vLLM 0.29 with exact full KV and prefix caching,
same GPU (A100-PCIE-40GB, aarch64, CUDA 13 compat library for vLLM), same
prompts, greedy, exactly 128 output tokens per turn.

Artifacts: `mt_wkvm_ring1024_ctx36864_b32_t8.json`,
`mt_vllm_ctx36864_b32_t8.json`, `mt_wkvm_ring1024_ctx13824_b16_t8.json`,
`mt_vllm_ctx13824_b16_t8.json`, `ruler_lite_wkvm_ring1024.json`.
Driver: `experiments/hybrid_multiturn_bench.py`.

## Semantics, stated up front

Ring mode is approximate beyond the window: tokens older than the last 1024
are evicted from the attention layers (the 24 Gated DeltaNet layers keep
their recurrent state over the whole context). vLLM computes exact attention
over the whole conversation. This is the same asymmetry as the Gemma rows.
The quality cost is measured below with RULER-lite; it is large.

## Workload 1: 16 sessions x 13,824-token context, 8 turns (no capacity wall)

16 x 15.7k tokens of KV (7.7 GiB at 32 KiB/token) fits vLLM's pool, so its
prefix cache never evicts and each turn prefills only the 128 new tokens.

| engine | per turn (s) | overall output tok/s | total 8 turns (s) |
|---|---:|---:|---:|
| vLLM 0.29, exact | 3.7 | 347 | 47 |
| wkvm ring 16+1024 | 5.4 | 230 | 71 |

vLLM 1.5x faster. Without a wall the incumbent's kernels win, as in the
single-turn ladders.

## Workload 2: 32 sessions x 36,864-token context, 8 turns (past the wall)

32 x 37k tokens = 37 GiB of KV against a pool of roughly 14 GiB after the
weights: vLLM cannot keep the sessions resident, its prefix cache evicts,
and every turn re-prefills the whole conversation. wkvm's per-session guest
memory is 33 MiB regardless of context (plus the 49.5 MiB recurrent state),
so 32 sessions occupy 2.6 GiB and never leave the GPU.

| engine | turn 0 (s) | steady-state turn (s) | output tok/s (steady) | total 8 turns (s) |
|---|---:|---:|---:|---:|
| vLLM 0.29, exact, prefix caching | 118 | 129–133 | 31 | 1036 |
| wkvm ring 16+1024 | 131 | 5.25 | 780 | 168 |

- **Per turn: 24.6x** (129 s vs 5.25 s). vLLM's turn is a 32 x 38k-token
  re-prefill plus decode; wkvm's is a 32 x 129-token prefill plus 127 graphed
  decode steps over a 1040-column window (~35 ms each at B=32).
- **Over 8 turns: 6.2x**, because wkvm's initial prefill of 1.18M tokens is
  slow (131 s, one request at a time through eager HF modules, ~9k tok/s).
- Extrapolated to the README's 48-turn shape: wkvm 131 + 47 x 5.25 = 378 s
  vs vLLM 48 x 130 = 6,240 s, **16.5x**. The steady-state ratio is what the
  long-session workloads see.
- Memory: wkvm peak 26.1 GiB reserved (weights 17.9 + 32 rows of window and
  state + graph pools); vLLM at its configured 0.85 x 40 GB.

This reproduces the Gemma result's mechanism on the hybrid, with the same
caveat as the main branch: it is a workload-specific claim about sessions
past the incumbent's capacity, not an engine-wide speedup.

## The price: RULER-lite in ring mode (sink 16 + ring 1024)

Same prompts as the exact run (`ruler_lite_wkvm.json`, all cells 1.00 on
needle tasks). Ring mode:

| task | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|
| niah_single_1/2/3 | 0.25 / 0.15 / 0.20 | 0.05 / 0.15 / 0.15 | 0.20 / 0.00 / 0.00 | 0.10 / 0.00 / 0.05 |
| niah_multikey_1/2/3 | 0.40 / 0.20 / 0.15 | 0.10 / 0.20 / 0.05 | 0.15 / 0.00 / 0.00 | 0.00 / 0.05 / 0.00 |
| niah_multivalue / multiquery | 0.33 / 0.30 | 0.15 / 0.11 | 0.11 / 0.09 | 0.01 / 0.06 |
| vt | 0.26 | 0.20 | 0.10 | 0.06 |
| cwe | 0.86 | 0.82 | 0.57 | 0.32 |
| fwe | 0.95 | 0.98 | 0.92 | 0.78 |
| qa_1 | 1.00 | 0.10 | 0.00 | 0.10 |

Reading: needle recall survives only when the needle falls inside the last
1024 tokens (about 25% at 4k, near zero at 32k). **The 24 Gated DeltaNet
layers do not carry retrievable content on their own** — this is the same
finding as the Gemma "ring" column (0.00 at evicted depths) and the reason
the Gemma path grew the routed span bank (recall 0.90 vs 0.95 for full KV).
Frequency-style tasks (fwe, cwe at short lengths) degrade gracefully.

## What this settles

1. The ten-times mechanism transfers to the hybrid: bounded guest memory,
   no re-prefill, sessions past the incumbent's wall. 24.6x per turn, 16.5x
   on a 48-turn shape, 6.2x on 8 turns including a slow initial prefill.
2. It is bought with recall. Ring-only is not shippable for retrieval; the
   routed span bank (value-routed spans of evicted tokens, 64 slots, exact
   representatives under a budget) is the next step, and RULER-lite is the
   gate for it: the target is the Gemma number, ~0.9 recall at 8k–32k with
   memory still bounded at ~4k materialized columns.
3. wkvm's remaining engine debt is prefill: turn 0 at 9k tok/s vs vLLM's
   ~21k. Batched prefill across requests is now in the engine
   (`Engine._execute` groups equal-length chunks into one forward) and the
   wall run was repeated with it: identical, 5.25 s per turn and 131 s for
   turn 0. A [8 x 129]-token batched forward takes 151 ms (6.8k tok/s), the
   same rate as eight single forwards — the prefill is bound by the HF
   module forward at ~40% of peak (unfused ops, SDPA math attention with a
   mask), not by per-request overhead. Fused prefill kernels are the lever.

Caveats: single runs; vLLM configured at `gpu_memory_utilization=0.85` with
prefix caching on (its best case for this workload); an active multi-tenant
box; approximate vs exact semantics on the wkvm side.
