# vLLM bench: gemma-4-E4B-it (2026-07-03 20:50)

- vllm 0.24.0, torch 2.11.0+cu130, GPU NVIDIA GeForce RTX 4090
- gpu_memory_utilization=0.82, enforce_eager=False, max_model_len=16896, prefix_caching=off, loaded with limit_mm_per_prompt={image:0,audio:0}

## A. KV-cache capacity (vs ring engine 64 slots)

- GPU KV cache size: **161,584 tokens** (engine cache_config: 10,099 blocks x 16 tokens; log said -)
- max concurrent 4k+128 seqs (uniform block math): **38**
- max concurrent 16k+128 seqs (uniform block math): **9**
- config KV math: 4 full + 20 sliding(w=512) own-KV layers (18 KV-shared of 42); per-seq KV = 84.0 MiB @4k / 276.0 MiB @16k (structured), vs 384.0 / 1536.0 MiB naive-uniform

Note: with hybrid sliding/full + KV-sharing, vLLM's own reported concurrency line is the authoritative apples-to-apples number; uniform block math over- counts long-context cost for sliding layers.

## B. Throughput (greedy, max_tokens=128, ignore_eos, synthetic exact-length prompts)

| ctx | N | eff. conc (cap) | prefill+1st (s) | full wall (s) | decode tok/s | aggregate tok/s | peak VRAM (GiB) | device used (GiB) |
|---|---|---|---|---|---|---|---|---|
| 4096 | 8 | 8 (38) | 2.22 | 3.80 | 643.9 | 269.5 | 18.26 | 21.81 |
| 4096 | 16 | 16 (38) | 4.09 | 6.04 | 1,039.7 | 339.1 | 18.26 | 21.81 |
| 4096 | 32 | 32 (38) | 8.23 | 12.03 | 1,069.5 | 340.4 | 18.26 | 21.81 |
| 4096 | 64 | 38 (38) | 16.44 | 22.44 | 1,356.2 | 365.1 | 18.26 | 21.81 |
| 16384 | 2 | 2 (9) | 3.03 | 4.32 | 197.0 | 59.3 | 18.26 | 21.81 |
| 16384 | 4 | 4 (9) | 4.89 | 6.85 | 258.5 | 74.7 | 18.26 | 21.81 |
| 16384 | 8 | 8 (9) | 9.86 | 13.42 | 285.6 | 76.3 | 18.26 | 21.81 |

- decode tok/s = N*127 / (wall@128 - wall@1); aggregate = N*128 / wall@128 (includes prefill).
- 16k rows with N above the admission cap are queue-limited: vLLM runs cap-many at once and queues the rest; wall time reflects that serialization.
- peak VRAM is torch.cuda.max_memory_allocated of this process; 'device used' is total minus free (includes the other ~2GB process).
