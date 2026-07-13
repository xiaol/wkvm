# Gemma serving Stage-1 comparison (2026-07-11)

This is a bounded HTTP-serving comparison on one NVIDIA GeForce RTX 4090
(driver 595.71.05). It is a readiness smoke, not a long-duration saturation
claim. The strict comparison contract passed for all three artifacts.

## Workload

- Model: local `gemma-4-E4B-it` checkpoint.
- HTTP path: streaming OpenAI-compatible `/v1/completions`.
- Shape: 512 input tokens, 8 requested output tokens, staggered prompts.
- Concurrency: B=1 and B=2, four measured requests per row.
- Warmup: one disjoint request per row with two output tokens.
- Sampling: greedy, `ignore_eos=true`, exact output-token accounting required.
- Monitoring: whole-GPU `nvidia-smi` sampling every 0.05 seconds.
- B=1 prompt SHA-256: `a2ea97671a6d66d7faf595b03da4f7de38da4ad6deabf6acd7f0ce14db97a22e`.
- B=2 prompt SHA-256: `822545a1161f5eafa494d24f6c4d9b9616c482577d5070b354d82055b283fb1f`.

## Results

| engine | semantics | B=1 output tok/s | B=2 output tok/s | B=1 p50 TTFT | B=2 p50 TTFT | whole-GPU peak |
|---|---|---:|---:|---:|---:|---:|
| WKVM native | routed-span approximate | 33.586 | 52.948 | 0.151 s | 0.194 s | 17.730 GiB |
| vLLM 0.24.0 | full KV | 62.307 | 89.009 | 0.039 s | 0.064 s | 17.658 GiB |
| SGLang 0.5.14 | full KV | 57.646 | 78.967 | 0.052 s | 0.103 s | 20.249 GiB |

Every row completed 4/4 requests with exactly eight output tokens per request.
WKVM is 46.1% slower than vLLM and 41.7% slower than SGLang at B=1; it is
40.5% slower than vLLM and 32.9% slower than SGLang at B=2.

## Server configurations

- WKVM used `--native-gemma-production-profile`, two slots, checkpoint-native
  construction, and no Hugging Face transformer forward.
- vLLM used its language-only path, a workload-sized 128 MiB KV cache, prefix
  caching disabled, and full CUDA graphs for B=1/B=2. Inductor compilation was
  disabled because the default O2 startup tried to allocate an additional
  5.25 GiB LM-head-shaped tensor and could not coexist with the desktop GPU
  processes. This deviation is explicit; eager execution was not used.
- SGLang used Triton attention, full decode CUDA graphs for B=1/B=2, disabled
  prefill graphs, and disabled radix/chunked-prefix caches. SGLang 0.5.14 does
  not support `--language-only` for this Gemma4 conditional checkpoint, so its
  multimodal towers remained resident. Its context limit was set to 522 because
  the scheduler reserves two slots beyond the 512+8 measured workload.

## Interpretation

The result does not show a WKVM throughput win. It also is not a
quality-equivalent comparison: WKVM serves approximate routed-span semantics,
whereas vLLM and SGLang serve full KV. WKVM's demonstrated advantage remains
bounded-memory long-context residency, not short-context token throughput.

The machine-readable run artifacts and generated strict report for this run
were written to `/tmp/wkvm-serving-production-v2-clean.json`,
`/tmp/vllm-serving-production-smoke-v2-clean.json`,
`/tmp/sglang-serving-production-v2.json`, and
`/tmp/wkvm-vllm-sglang-serving-strict-v2-clean.md`.
