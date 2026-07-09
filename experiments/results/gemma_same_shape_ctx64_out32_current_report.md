# Gemma Throughput Report

Rows are normalized across wkvm-native, HF Transformers, vLLM, and SGLang JSON schemas. Serving rows use HTTP stream output throughput in the `agg decode tok/s` column. Only rows with the same `ctx`, `out`, prompt mode, and benchmark path should be treated as same-shape comparisons.

| engine | ctx | out | prompt mode | B | success | green | agg decode tok/s | e2e output tok/s | p50 s | p95 s | memory | max model batch | padded temp | persistent padded | error | source |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| sglang-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 1 | 1/1 | - | 70.616 | - | 0.453 | 0.453 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/sglang_ctx64_out32_ladder_warm.json) |
| vllm-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 1 | 1/1 | - | 67.033 | - | 0.477 | 0.477 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/vllm_ctx64_out32_ladder_warm.json) |
| wkvm-native row-cap 16 | 64 | 32 | uniform | 1 | 1/1 | yes | 43.702 | 35.363 | 0.904 | 0.904 | 13.938 GiB reserved | 1 rows / 5.2 MiB | 0 | 0 starts / 0 reuses | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/native_gemma_current_ctx64_out32_uniform_ladder.json) |
| wkvm-native-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 1 | 1/1 | - | 44.805 | - | 0.714 | 0.714 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/wkvm_ctx64_out32_ladder_warm.json) |
| sglang-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 2 | 2/2 | - | 123.316 | - | 0.518 | 0.518 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/sglang_ctx64_out32_ladder_warm.json) |
| vllm-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 2 | 2/2 | - | 127.090 | - | 0.503 | 0.503 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/vllm_ctx64_out32_ladder_warm.json) |
| wkvm-native row-cap 16 | 64 | 32 | uniform | 2 | 2/2 | yes | 80.031 | 80.197 | 0.797 | 0.797 | 13.949 GiB reserved | 2 rows / 10.4 MiB | 0 | 0 starts / 0 reuses | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/native_gemma_current_ctx64_out32_uniform_ladder.json) |
| wkvm-native-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 2 | 2/2 | - | 78.583 | - | 0.814 | 0.814 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/wkvm_ctx64_out32_ladder_warm.json) |
| sglang-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 4 | 4/4 | - | 257.362 | - | 0.496 | 0.497 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/sglang_ctx64_out32_ladder_warm.json) |
| vllm-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 4 | 4/4 | - | 246.089 | - | 0.507 | 0.517 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/vllm_ctx64_out32_ladder_warm.json) |
| wkvm-native row-cap 16 | 64 | 32 | uniform | 4 | 4/4 | yes | 147.879 | 148.397 | 0.861 | 0.861 | 13.996 GiB reserved | 4 rows / 20.8 MiB | 0 | 0 starts / 0 reuses | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/native_gemma_current_ctx64_out32_uniform_ladder.json) |
| wkvm-native-openai-completions-ctx64-out32-ladder | 64 | 32 | uniform | 4 | 4/4 | - | 121.872 | - | 1.036 | 1.049 | - | - | - | - | - | [json](/run/media/xiaol/B214449214445C0B/wkvm_bench/results/wkvm_ctx64_out32_ladder_warm.json) |
