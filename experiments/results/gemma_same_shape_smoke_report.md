# Gemma Throughput Report

Rows are normalized across wkvm-native, HF Transformers, vLLM, and SGLang JSON schemas. Only rows with the same `ctx`, `out`, and prompt mode should be treated as same-shape comparisons.

| engine | ctx | out | prompt mode | B | success | green | agg decode tok/s | e2e output tok/s | p50 s | p95 s | memory | error | source |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| sglang | 512 | 8 | staggered | 1 | 1/1 | no | 74.075 | 58.094 | 0.138 | 0.138 | 19.849 GiB engine delta | - | [json](experiments/results/sglang_gemma_smoke_ctx512_out8_b1.json) |
| vllm | 512 | 8 | staggered | 1 | 1/1 | no | 88.939 | 68.540 | 0.117 | 0.117 | 20.124 GiB engine delta | - | [json](experiments/results/vllm_gemma_smoke_ctx512_out8_b1.json) |
| HF Transformers (batched) | 512 | 8 | staggered | 2 | 2/2 | yes | 74.658 | 34.026 | 0.469 | 0.469 | 14.033 GiB reserved | - | [json](experiments/results/hf_gemma_smoke_batched_chunked_ctx512_out8_b2.json) |
| wkvm-native row-cap 2 | 512 | 8 | staggered | 2 | 2/2 | yes | 50.923 | 32.252 | 0.490 | 0.492 | 14.096 GiB reserved | - | [json](experiments/results/native_gemma_default_temp_metrics_smoke_ctx512_out8_b2.json) |
