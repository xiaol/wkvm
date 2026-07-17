# Reliable Gemma Repeated-Run Report

Evidence gate: **PASS**.

- Shape: ctx=16384, out=32, prompt=uniform, dtype=bfloat16
- Model: `/home/aiuser/X/models/gemma-4-E4B-it`
- Model manifest SHA-256: `0829868c14a87b8dc1323f17b5a0808865acd81d15d320ace171c46e2ce4f946`
- Model identity exclusions: `.cache/**`
- GPU cohort: NVIDIA A800-SXM4-80GB `GPU-d14ae640-d8a8-de48-c9cc-3b73f99604f0` with driver 595.58.03
- Warmup policy: `False` (cold one-shot requires `False`)
- Pre-load GPU baseline ceiling: 1.000 GiB
- GPU memory sample interval: 0.100 s
- Minimum repeats: 3 per engine/B
- Benchmark commit: `234cc04867a93f1352d2a1c220c216b698f46560`
- Exact source/worktree SHA-256: `28763ceb275e942aea3b51fe516b9ed362e84a3aabf29f03b752989b9ec960c4`
- Exact greedy output fingerprints: B=64 `bfabb16acea48e733714142e1120307f6f54a3264ff7f9298bc5dfe8bf53f6c0`
- Source identity exclusions: `experiments/results/**`, `**/__pycache__/**`, `.pytest_cache/**`, `**/*.egg-info/**`, `.venv/**`, `build/**`, `dist/**`
- Semantics: `routed_span_approximate` and `full_kv` are reported separately and are not equivalent.

## Aggregates

Cells are median [min, max] across validated repeated runs.

| B | Engine | Semantics | n | Cohort input tok/s | Cohort prefill wall | E2E output tok/s | TTFT p50 | TTFT p95 | Batch wall | GPU peak used | Decode interval | Comparable decode tok/s |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | wkvm-native 0.0.1 | `routed_span_approximate` | 3 | 12368.410 [12331.504, 12413.080] tok/s | 84.779 [84.473, 85.032] s | 21.886 [21.822, 22.003] tok/s | 80.812 [80.482, 81.180] s | 84.639 [84.335, 84.892] s | 93.574 [93.080, 93.852] s | 28.438 [28.433, 28.618] GiB | 16.314 [16.198, 16.363] s | 121.612 [121.250, 122.481] tok/s |
| 64 | vllm 0.25.1 | `full_kv` | 3 | 14247.043 [14242.409, 14249.204] tok/s | 73.600 [73.588, 73.623] s | 27.608 [27.598, 27.612] tok/s | 38.399 [38.397, 38.424] s | 70.969 [70.953, 70.992] s | 74.182 [74.171, 74.207] s | 74.743 [74.743, 74.743] GiB | 72.869 [72.858, 72.889] s | 27.227 [27.220, 27.231] tok/s |
| 64 | sglang 0.5.15.post1 | `full_kv` | 3 | reported only; ratio excluded (8706.142 [8705.931, 8709.324] tok/s) | 120.441 [120.397, 120.444] s | 15.587 [15.585, 15.588] tok/s | excluded | excluded | 131.389 [131.380, 131.411] s | 69.930 [69.930, 69.930] GiB | excluded (separate_run_subtraction) | excluded (separate_run_subtraction) |

## WKVM / Incumbent Median Ratios

Every ratio is `wkvm-native / incumbent` (numerator / denominator). Cross-semantics ratios describe measured workload performance, not semantic equivalence.

| B | Ratio | Cohort input | E2E output | Comparable decode |
|---:|---|---:|---:|---:|
| 64 | wkvm-native / vllm | 0.868x | 0.793x | 4.467x |
| 64 | wkvm-native / sglang | excluded (incomparable_methods:same_run_max_request_ttft,separate_run_batch_wall) | 1.404x | excluded (separate_run_subtraction) |

## 10x E2E Claim Gate

This observed-run gate passes only when minimum WKVM E2E output throughput divided by maximum incumbent throughput is at least 10.000x for both vLLM and SGLang at the same B. It is deliberately stricter than a median ratio and is not a statistical confidence interval.

| B | Conservative WKVM / vLLM | Conservative WKVM / SGLang | All incumbents |
|---:|---:|---:|---|
| 64 | 0.790x | 1.400x | **FAIL** |

## Artifacts

- `experiments/results/gemma_a800_reliable_20260716/artifacts/sglang-r1.json`
  - Engine: `sglang 0.5.15.post1`; semantics: `full_kv`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-sglang/bin/python experiments/incumbent_gemma_bench.py --engine sglang --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --prompt-lengths uniform --synthetic-prompts --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --mem-sample-interval-s 0.1 --telemetry-sample-interval-s 0.05 --sglang-mem-fraction 0.83 --sglang-context-length 16432 --sglang-chunked-prefill-size 8192 --sglang-attention-backend triton --sglang-language-model-only --sglang-max-running-requests 32 --sglang-decode-graph full --sglang-prefill-graph breakable --sglang-log-level warning --no-warmup --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/sglang-r1.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/sglang-r2.json`
  - Engine: `sglang 0.5.15.post1`; semantics: `full_kv`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-sglang/bin/python experiments/incumbent_gemma_bench.py --engine sglang --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --prompt-lengths uniform --synthetic-prompts --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --mem-sample-interval-s 0.1 --telemetry-sample-interval-s 0.05 --sglang-mem-fraction 0.83 --sglang-context-length 16432 --sglang-chunked-prefill-size 8192 --sglang-attention-backend triton --sglang-language-model-only --sglang-max-running-requests 32 --sglang-decode-graph full --sglang-prefill-graph breakable --sglang-log-level warning --no-warmup --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/sglang-r2.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/sglang-r3.json`
  - Engine: `sglang 0.5.15.post1`; semantics: `full_kv`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-sglang/bin/python experiments/incumbent_gemma_bench.py --engine sglang --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --prompt-lengths uniform --synthetic-prompts --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --mem-sample-interval-s 0.1 --telemetry-sample-interval-s 0.05 --sglang-mem-fraction 0.83 --sglang-context-length 16432 --sglang-chunked-prefill-size 8192 --sglang-attention-backend triton --sglang-language-model-only --sglang-max-running-requests 32 --sglang-decode-graph full --sglang-prefill-graph breakable --sglang-log-level warning --no-warmup --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/sglang-r3.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/vllm-r1.json`
  - Engine: `vllm 0.25.1`; semantics: `full_kv`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-vllm/bin/python experiments/incumbent_gemma_bench.py --engine vllm --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --prompt-lengths uniform --synthetic-prompts --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --mem-sample-interval-s 0.1 --telemetry-sample-interval-s 0.05 --max-model-len 16432 --vllm-gpu-mem-util 0.92 --vllm-max-num-batched-tokens 16384 --vllm-language-model-only --no-warmup --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/vllm-r1.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/vllm-r2.json`
  - Engine: `vllm 0.25.1`; semantics: `full_kv`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-vllm/bin/python experiments/incumbent_gemma_bench.py --engine vllm --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --prompt-lengths uniform --synthetic-prompts --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --mem-sample-interval-s 0.1 --telemetry-sample-interval-s 0.05 --max-model-len 16432 --vllm-gpu-mem-util 0.92 --vllm-max-num-batched-tokens 16384 --vllm-language-model-only --no-warmup --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/vllm-r2.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/vllm-r3.json`
  - Engine: `vllm 0.25.1`; semantics: `full_kv`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-vllm/bin/python experiments/incumbent_gemma_bench.py --engine vllm --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --prompt-lengths uniform --synthetic-prompts --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --mem-sample-interval-s 0.1 --telemetry-sample-interval-s 0.05 --max-model-len 16432 --vllm-gpu-mem-util 0.92 --vllm-max-num-batched-tokens 16384 --vllm-language-model-only --no-warmup --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/vllm-r3.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/wkvm-r1.json`
  - Engine: `wkvm-native 0.0.1`; semantics: `routed_span_approximate`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-wkvm/bin/python experiments/native_gemma_bench.py --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --slots 64 --prompt-lengths uniform --synthetic-prompts --native-gemma-checkpoint-loader --native-gemma-attention-backend sdpa_single_gqa --native-gemma-projection-backend separate --enable-token-pool-attention --token-pool-max-context-len 16640 --token-pool-capacity 262144 --token-pool-paged-block-size 16 --enable-token-pool-triton --enable-token-pool-paged-triton --enable-token-pool-paged-split-triton --token-pool-triton-strict --token-pool-sliding-paged-metadata-only --persistent-padded-sliding-metadata-padding --persistent-padded-decode-steps 32 --persistent-padded-decode-cuda-graph --persistent-padded-decode-graph-warmup-iters 0 --sink 16 --window 1024 --m-slots 32 --route-chunk 512 --chunk 2048 --prefill-microbatch-rows 8 --decode-microbatch-rows 32 --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --gpu-memory-sample-interval-s 0.1 --require-native-no-hf --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/wkvm-r1.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/wkvm-r2.json`
  - Engine: `wkvm-native 0.0.1`; semantics: `routed_span_approximate`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-wkvm/bin/python experiments/native_gemma_bench.py --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --slots 64 --prompt-lengths uniform --synthetic-prompts --native-gemma-checkpoint-loader --native-gemma-attention-backend sdpa_single_gqa --native-gemma-projection-backend separate --enable-token-pool-attention --token-pool-max-context-len 16640 --token-pool-capacity 262144 --token-pool-paged-block-size 16 --enable-token-pool-triton --enable-token-pool-paged-triton --enable-token-pool-paged-split-triton --token-pool-triton-strict --token-pool-sliding-paged-metadata-only --persistent-padded-sliding-metadata-padding --persistent-padded-decode-steps 32 --persistent-padded-decode-cuda-graph --persistent-padded-decode-graph-warmup-iters 0 --sink 16 --window 1024 --m-slots 32 --route-chunk 512 --chunk 2048 --prefill-microbatch-rows 8 --decode-microbatch-rows 32 --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --gpu-memory-sample-interval-s 0.1 --require-native-no-hf --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/wkvm-r2.json`
- `experiments/results/gemma_a800_reliable_20260716/artifacts/wkvm-r3.json`
  - Engine: `wkvm-native 0.0.1`; semantics: `routed_span_approximate`; B=64
  - GPU provenance: NVIDIA A800-SXM4-80GB, driver 595.58.03
  - Launch: `/home/aiuser/X/.venv-wkvm/bin/python experiments/native_gemma_bench.py --model-path /home/aiuser/X/models/gemma-4-E4B-it --ctx 16384 --out 32 --concurrency 64 --slots 64 --prompt-lengths uniform --synthetic-prompts --native-gemma-checkpoint-loader --native-gemma-attention-backend sdpa_single_gqa --native-gemma-projection-backend separate --enable-token-pool-attention --token-pool-max-context-len 16640 --token-pool-capacity 262144 --token-pool-paged-block-size 16 --enable-token-pool-triton --enable-token-pool-paged-triton --enable-token-pool-paged-split-triton --token-pool-triton-strict --token-pool-sliding-paged-metadata-only --persistent-padded-sliding-metadata-padding --persistent-padded-decode-steps 32 --persistent-padded-decode-cuda-graph --persistent-padded-decode-graph-warmup-iters 0 --sink 16 --window 1024 --m-slots 32 --route-chunk 512 --chunk 2048 --prefill-microbatch-rows 8 --decode-microbatch-rows 32 --mem-cap-gib 80 --headroom-gib 4 --max-baseline-gpu-used-gib 1 --gpu-memory-device 7 --gpu-memory-sample-interval-s 0.1 --require-native-no-hf --stop-on-failure --json experiments/results/gemma_a800_reliable_20260716/artifacts/wkvm-r3.json`

## Caveats

- routed_span_approximate and full_kv are different model-state semantics
- whole-device memory includes every process on the selected GPU
- SGLang separate max_tokens=1 prefill is excluded from cohort-input ratios
- SGLang separate-run subtraction is excluded from decode ratios
- the 10x gate applies to E2E output throughput and uses the worst observed repeated-run envelope
