# Qwen3.5-9B: wkvm hybrid engine vs vLLM 0.29 on one A100-40GB (aarch64), 2026-09-10

Same-shape, same-prompt, greedy, `ignore_eos` ladders on the same GPU class.
Both engines run the same bf16 checkpoint (`/root/x/Qwen3.5-9B`, 24 Gated
DeltaNet + 8 full-attention layers) with **exact** semantics — this is a
like-for-like comparison, unlike the approximate routed-span Gemma rows.

Artifacts: `wkvm_hybrid_qwen35_ctx13824_out128_ladder.json`,
`vllm_qwen35_ctx13824_out128_ladder.json`,
`wkvm_hybrid_qwen35_ctx348_out128_ladder.json`,
`vllm_qwen35_ctx348_out128_ladder.json`.

## Setup

- Host: aarch64, driver 570.86 (CUDA 12.8), 4x A100-PCIE-40GB; one GPU per
  engine run (wkvm on GPU 1, vLLM on GPU 3 / GPU 1), other GPUs busy with
  unrelated jobs — treat as an active box, not a quiet lab.
- wkvm: branch `m4-qwen35-hybrid`, fla 0.5.2 Triton kernels, CUDA-graph
  decode with resident rows, paged guest pool (256-token pages), eager
  chunked prefill (1024-token chunks, one request at a time), SDPA guest
  attention with a boolean mask. `experiments/hybrid_bench.py`.
- vLLM 0.29.0 (PyPI aarch64 wheel, CUDA 13 build) run through NVIDIA's
  CUDA 13.0 forward-compatibility library (`cuda-compat-13-0_580.95.05`,
  `LD_LIBRARY_PATH=.../cuda-13.0/compat`) on the 12.8 driver; torch
  2.13.0+cu130, FlashInfer 0.6.18, `gpu_memory_utilization=0.85`, default
  CUDA graphs and torch.compile; `experiments/incumbent_gemma_bench.py`
  (one line removed: the `swap_space` argument vLLM 0.29 no longer accepts).
- SGLang could not be run: every Qwen3.5-capable SGLang release needs an
  aarch64 kernel wheel whose `common_ops` is built only for sm_100
  (`sgl-kernel` 0.3.21) or sm_90 (`sglang-kernel` 0.4.x); no A100 code.
- Prompts: `build_prompt` / `prompt_lengths` from `native_gemma_engine_smoke`
  with the Qwen tokenizer, staggered lengths (ctx − 7·i), identical token ids
  for both engines. 128 output tokens per request, greedy.

## 13,824-token context, 128 output tokens

| B | engine | TTFT p50 (s) | latency p50 (s) | end-to-end output tok/s | peak GiB |
|---:|---|---:|---:|---:|---:|
| 1 | vLLM 0.29 | 1.23 | 2.99 | 42.8 | 34.3 (pre-allocated pool) |
| 1 | wkvm hybrid | 1.94 | 5.59 | 22.9 | 36.3 |
| 8 | vLLM 0.29 | 5.51 | 11.39 | 89.2 | 34.3 |
| 8 | wkvm hybrid | 15.52 | 23.41 | 43.7 | 37.6 |
| 16 | vLLM 0.29 | 10.39 | 21.10 | 96.2 | 34.3 |
| 16 | wkvm hybrid | 29.32 | 47.34 | 43.3 | 36.3 |

vLLM completes the workload 1.9x (B=1) to 2.2x (B=16) faster.

## 348-token context, 128 output tokens (decode-dominated)

| B | engine | TTFT p50 (s) | latency p50 (s) | end-to-end output tok/s | decode step (ms) |
|---:|---|---:|---:|---:|---:|
| 1 | vLLM 0.29 | 0.08 | 1.81 | 70.8 | 13.6 |
| 1 | wkvm hybrid | 0.12 | 2.38 | 53.8 | 17.8 |
| 8 | vLLM 0.29 | 0.28 | 2.09 | 490.5 | 15.0 |
| 8 | wkvm hybrid | 0.94 | 4.06 | 252.0 | 24.6 |
| 16 | vLLM 0.29 | 0.45 | 2.36 | 866.8 | 17.1 |
| 16 | wkvm hybrid | 1.89 | 5.02 | 408.2 | 24.6 |

vLLM's decode step at B=1 (13.6 ms for 17.9 GB of weights) is ~85% of the
A100-PCIe's memory bandwidth; wkvm's graphed replay of the HF module graph
sits at 17.8 ms (~65%), and its batched steps grow faster with B.

## Where the gap is

1. **Prefill.** wkvm prefills one request at a time through eager HF
   modules in 1024-token chunks: ~7k tokens/s (220k tokens in 31.6 s at
   B=16) vs vLLM's batched, fused prefill at ~21k tokens/s. On a
   13.8k-in/128-out workload prefill is most of the wall, so this alone
   explains the long-context rows.
2. **Guest attention at long context.** wkvm's decode attention is SDPA's
   math kernel over a `B x 14k` boolean mask with `repeat_kv`, ~28 ms/step
   at B=1 and 13.8k tokens vs vLLM's FlashInfer paged decode at ~14 ms.
3. **Kernel granularity.** Even inside a CUDA graph, the HF decoder layer is
   ~6,300 kernels per step (unfused norms, gates, projections); vLLM's
   fused RMSNorm / gated-delta / MoE-free path is fewer, larger kernels. The
   4 ms gap at B=1 short context is this.
4. **Memory.** wkvm holds every resident request's K/V twice (pages as the
   durable copy, static graph rows as the working set), which is why the
   36–38 GiB peaks match vLLM's pre-allocated 34 GiB pool despite fewer
   sessions.

## What does *not* differ

- Output tokens: both engines produce the same greedy text on the parity
  prompts (`vllm_smoke.log`, `m4_qwen35_hybrid_smoke_fla_graphs.json`).
- The GDN recurrence: both run fla-derived Triton kernels; the recurrent
  state is not where the time goes at either context length.

## Long-context quality: RULER-lite (`experiments/ruler_lite.py`)

RULER's synthetic tasks ported onto local SQuAD text (the official
generators fetch Paul Graham essays and nltk data from hosts this box cannot
reach), RULER templates, answer prefixes and metrics kept; 20 samples per
task per length, greedy, wkvm with fla kernels and CUDA-graph decode vs HF
`generate` on byte-identical prompts.

| task | 4k | 8k | 16k | 32k | outputs identical to HF (of 80) |
|---|---:|---:|---:|---:|---:|
| niah_single_1 | 1.00 | 1.00 | 1.00 | 1.00 | 80 |
| niah_single_2 | 1.00 | 1.00 | 1.00 | 1.00 | 80 |
| niah_single_3 | 1.00 | 1.00 | 1.00 | 1.00 | 80 |
| niah_multikey_1 | 1.00 | 1.00 | 1.00 | 1.00 | 80 |
| niah_multikey_2 | 1.00 | 1.00 | 1.00 | 1.00 | 80 |
| niah_multikey_3 | 1.00 | 1.00 | 1.00 | 1.00 | 79 |

HF scores the same 1.00 on every cell. 479/480 outputs are token-identical
across 4k–32k; the one difference (a 32k multikey_3 sample) is a
post-answer near-tie, both engines score it correct.

| task | 4k wkvm / HF | 8k | 16k | 32k | identical (of 80) |
|---|---:|---:|---:|---:|---:|
| niah_multivalue | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 76 |
| niah_multiquery | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 77 |
| vt | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 69 |
| cwe | 1.00 / 1.00 | 0.90 / 0.91 | 0.99 / 0.99 | 0.99 / 1.00 | 38 |
| fwe | 1.00 / 1.00 | 1.00 / 1.00 | 0.98 / 1.00 | 1.00 / 1.00 | 71 |
| qa_1 | 0.95 / 0.95 | 0.85 / 0.85 | 1.00 / 1.00 | 0.85 / 0.85 | 78 |

Over all 12 tasks and 4 lengths: 888/960 outputs token-identical, every
score within 0.02 of HF (one or two samples on the 120-token list-style
answers of cwe/fwe, where a 4-row batched GEMM and HF's single-row GEMM
break bf16 ties differently mid-list). The paged attention is exact at 32k;
the residual is batch-shape numerics, the same effect documented for the
1.5B RWKV gate. Qwen3.5-9B's own RULER-lite profile: perfect needle recall
to 32k, cwe/qa_1 are the model's weak cells, not the engine's.

## Conclusion for the long-context benchmark

On Qwen3.5-9B, wkvm's hybrid path is a correct, exact engine that is behind
vLLM 0.29 by ~1.3x (short context, B=1) to ~2.2x (13.8k context, B=16) on
this hardware. The differentiators that survive this comparison are the
state-object features — tuned initial states imported as handles, exact
hibernate/resume/fork of a 50 MiB + pages object, mutation with provenance —
not throughput. To change the throughput picture the engine needs, in
order: batched prefill across requests, a paged attention kernel
(FlashInfer/FA3) for the guest layers, and fused per-layer kernels in place
of the HF module graph. None of those is a state-engine problem; all are
kernel work the incumbents have already done.

Caveats: single runs, an active multi-tenant box, vLLM under a
forward-compatibility library rather than a native CUDA 13 driver, no SGLang
row, `ignore_eos` semantics on the wkvm side implemented by an empty stop
set (every request generates exactly 128 tokens on both engines).
