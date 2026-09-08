# M4 slice 1 smoke: Qwen3.5-9B hybrid (Gated DeltaNet + full attention) on wkvm

Date: 2026-09-09. Raw artifact: `m4_qwen35_hybrid_smoke.json`
(`experiments/qwen35_hybrid_smoke.py`, base commit 93e7a87 + the M4 change set).

Setup: one NVIDIA A100-PCIE-40GB for the engine (`Engine.from_qwen35`,
16 slots, `guest_ctx` 4096, prefill chunk 512, bf16), a second A100 for an
independent HF reference (`Qwen3_5ForConditionalGeneration` through
RNN-StateTuning's loader, SDPA). torch 2.6.0+cu128, transformers 5.16.1.
**Kernels are transformers' pure-torch Gated DeltaNet fallbacks** (this host has
no `fla`, `triton`, `causal_conv1d` or `flash_attn`), so no number below is a
throughput claim; they are correctness and footprint evidence.

## A. Parity vs HF (chat prompts, greedy, 32 new tokens)

| prompt tokens | prefill last-logit max abs diff | argmax equal | greedy continuation |
|---:|---:|---|---|
| 29 | 0.0 | yes | identical (32/32) |
| 24 | 0.0 | yes | identical (18/18, EOS) |
| 25 | 0.0 | yes | identical (32/32) |

Bit-exact: the arena gather → HF layer forward → scatter plumbing adds no
numeric difference to HF's own cached path on the real checkpoint.

## B. Continuous batching of 8 distinct scene-boundary prompts (226–676 tokens, ≤48 new)

- Outputs identical to running each request alone: 8/8 (prefix match =
  generated length for every row, including the two that stopped at 2 and 9
  tokens).
- Wall: 26.6 s sequential vs 8.5 s batched (mixed prefill + decode);
  47 pure-decode steps at 58 tok/s aggregate as rows finished.
- Peak allocated 20.2 GiB (weights 16.7 GiB + 16-slot bank 2.9 GiB).

Pure-decode ladder (same 324-token prompt replicated, 8 timed steps each,
348 resident tokens per row):

| B | decode step ms | tok/s |
|---:|---:|---:|
| 1 | 87.1 | 11.5 |
| 4 | 89.7 | 44.6 |
| 8 | 92.6 | 86.4 |
| 16 | 98.7 | 162.1 |

The step time is nearly flat in batch size — the state-native decode is
weight-read bound, as for RWKV-7 — but the absolute floor is the pure-torch
GDN kernel (24 layers of python-level ops per step). See H5 in
`docs/HYBRID_ENGINE_PLAN.md`.

## C. Tuned initial state as a durable handle

Adapter: `/root/x/RNN-StateTuning/outputs/qwen35-state-novel-scene-e1`
(state-only CE adapter, scene-boundary task; full-benchmark numbers in that
repo: 34.2% exact match / 100% valid JSON on 149 scenes).

- `engine.import_state("scene-ce", adapter)` → handle `scene-ce@0`,
  `rule="import"`, `num_computed_tokens=0`, no tokens, 24 tuned layers,
  SHA-256 recorded; import 0.17 s; COLD file 50.3 MB (`gdn_state` only).
- Generation from the handle (prompt as suffix, batched through the same
  scheduler) vs the adapter's own HF runtime (`prepare_model_for_state_tuning`
  + `load_adapter` + `generate`): **identical on 8/8 scenes**.
- Zero state vs tuned state on the same 8 rows:

| state | valid JSON | exact match |
|---|---:|---:|
| zero (frozen Qwen3.5-9B) | 0/8 | 0/8 |
| tuned handle (wkvm) | 8/8 | 3/8 |
| tuned (adapter runtime, reference) | 8/8 | 3/8 |

- Persist → evict → resume from COLD: identical output.

This is the Durable State API's third way of obtaining a state — it is
*given*, not computed from tokens or mutated from a parent — exercised on a
trained artifact: a state that no token prefix produces, served at batch with
exact hibernate/resume.

## D. Footprint (Qwen3.5-9B, bf16)

| item | bytes |
|---|---:|
| weights (text stack, vision dropped) | 17.9 GB |
| `gdn_state` per slot (24 × 32×128×128 fp32) | 48 MiB |
| `gdn_conv` per slot (24 × 8192×4 bf16) | 1.5 MiB |
| `guest_kv` per slot at 4096-token window (8 layers, 4 kv-heads × 256) | 128 MiB |
| guest window per token | 32 KiB |

Slot capacity on this 40 GB card after weights and 2 GiB headroom:
120 slots at a 4k window, 37 at 16k, 10 at 64k. The recurrent part is a
constant 49.5 MiB per session; the guest window is the whole context cost,
which is what the paged pool (H4) is for.

## Caveats

- Single run, one GPU, greedy only; no incumbent comparison.
- Fixed guest window: prompts + `max_new_tokens` above `guest_ctx` are
  rejected at intake, not paged.
- Guest K/V are gathered per step (B × Lmax tokens × 8 layers); fine at this
  scale, the first thing to remove for long contexts.
