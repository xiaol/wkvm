# M4 smoke: Qwen3.5-9B hybrid (Gated DeltaNet + paged full-attention guests) on wkvm

Date: 2026-09-09. Raw artifact: `m4_qwen35_hybrid_smoke.json`
(`experiments/qwen35_hybrid_smoke.py`, base commit 93e7a87 + the M4 change set).

Setup: one NVIDIA A100-PCIE-40GB for the engine (`Engine.from_qwen35`,
16 slots, guest pool 65,536 tokens = 256 pages of 256 tokens, prefill chunk
512, bf16), a second A100 for an independent HF reference
(`Qwen3_5ForConditionalGeneration` through RNN-StateTuning's loader, SDPA).
torch 2.6.0+cu128, transformers 5.16.1. **Kernels are transformers'
pure-torch Gated DeltaNet fallbacks** (this host has no `fla`, `triton`,
`causal_conv1d` or `flash_attn`), so no number below is a throughput claim;
they are correctness and footprint evidence.

## A. Parity vs HF (chat prompts, greedy, 32 new tokens)

| prompt tokens | prefill last-logit max abs diff | argmax equal | greedy continuation |
|---:|---:|---|---|
| 29 | 0.0 | yes | identical (32/32) |
| 24 | 0.0 | yes | identical (18/18, EOS) |
| 25 | 0.0 | yes | identical (32/32) |

Bit-exact: the arena gather → HF layer forward → scatter plumbing, now
through paged K/V, adds no numeric difference to HF's own cached path.

## B. Continuous batching of 8 distinct scene-boundary prompts (226–676 tokens, ≤48 new)

- Outputs identical to running each request alone: 8/8 (prefix match =
  generated length for every row, including the two that stopped at 2 and 9
  tokens).
- Wall: 27.1 s sequential vs 8.8 s batched (mixed prefill + decode);
  pure-decode steps ran at 57 tok/s aggregate as rows finished.
- Peak allocated 20.1 GiB (weights 16.7 GiB + recurrent slots 0.8 GiB +
  page pool 2.0 GiB).

Pure-decode ladder (same 324-token prompt replicated, 8 timed steps each,
348 resident tokens per row):

| B | decode step ms | tok/s |
|---:|---:|---:|
| 1 | 88.9 | 11.2 |
| 4 | 92.2 | 43.4 |
| 8 | 95.3 | 84.0 |
| 16 | 103.0 | 155.3 |

The step time is nearly flat in batch size — the state-native decode is
weight-read bound, as for RWKV-7 — but the absolute floor is the pure-torch
GDN kernel (24 layers of python-level ops per step). Paging cost vs the
slice-1 fixed window: +2 to +4 ms per step at these sizes. See H5 in
`docs/HYBRID_ENGINE_PLAN.md`.

## C. Tuned initial state as a durable handle

Adapter: `/root/x/RNN-StateTuning/outputs/qwen35-state-novel-scene-e1`
(state-only CE adapter, scene-boundary task; full-benchmark numbers in that
repo: 34.2% exact match / 100% valid JSON on 149 scenes).

- `engine.import_state("scene-ce", adapter)` → handle `scene-ce@0`,
  `rule="import"`, `num_computed_tokens=0`, no tokens, 24 tuned layers,
  SHA-256 recorded; COLD file 50.3 MB (`gdn_state` only).
- Generation from the handle (prompt as suffix, batched through the same
  scheduler, pages reserved at resume) vs the adapter's own HF runtime
  (`prepare_model_for_state_tuning` + `load_adapter` + `generate`):
  **identical on 8/8 scenes**.
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

## E. Long context through the paged pool

A 12,637-token natural-text prompt (concatenated novel paragraphs from the
scene test set, chat-templated, one summarising question), 16 greedy tokens:

| item | value |
|---|---:|
| pages reserved (256 tokens each) | 50 |
| greedy continuation vs HF full-KV `generate` | identical (16/16) |
| engine wall (prefill + 16 decodes) | 12.2 s |
| HF reference wall | 5.2 s |
| peak allocated | 20.6 GiB |

The slice-1 fixed 4k window could not have admitted this request; the pool
admits it as 50 pages next to the 49.5 MiB recurrent slot, exactly, with no
preemption path involved. The engine is slower than HF here because prefill
runs in 512-token chunks that each re-gather the resident K/V (25 chunks,
`Lmax` growing to 12k) — the per-step gather is the H5 item.

## D. Footprint (Qwen3.5-9B, bf16)

| item | bytes |
|---|---:|
| weights (text stack, vision dropped) | 17.9 GB |
| `gdn_state` per request (24 × 32×128×128 fp32) | 48 MiB |
| `gdn_conv` per request (24 × 8192×4 bf16) | 1.5 MiB |
| guest page (256 tokens × 8 layers × 4 kv-heads × 256 × K,V bf16) | 8 MiB |
| guest per token | 32 KiB |

Session capacity on this 40 GB card after weights and 2 GiB headroom, if
every session held the given context: 120 at 4k, 37 at 16k, 10 at 64k. The
recurrent part is a constant 49.5 MiB per session; the pages are the whole
context cost and are shared across sessions of different lengths.

## Caveats

- Single run, one GPU, greedy only; no incumbent comparison.
- Pages are reserved for `prompt + max_new_tokens` at admission; a request
  that stops early at EOS held pages it did not use.
- Guest K/V are gathered per step (B × Lmax tokens × 8 layers); fine at this
  scale, the first thing to remove for long contexts at high concurrency.
