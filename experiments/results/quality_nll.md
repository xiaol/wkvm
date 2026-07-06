# Position-resolved NLL: full vs recurrent modes (gemma-4-E4B-it)

6 natural documents x 16384 tokens, teacher-forced with chunk-2048 prefill (cache evolves as in generation; divergence granularity is one chunk). Modes as in quality_grid.md. Provenance:

- papers-tex#0: Autoresearch_ideas/papers/associative_state_universal_transformers/main.tex, Autoresearch_ideas/papers/generic_triple_latent/generic_triple_latent.tex
- papers-tex#1: Autoresearch_ideas/papers/on_device_meta_learning_agents/paper58.tex, Autoresearch_ideas/papers/recurrent_ffn/main.tex
- repo-docs-md#0: Multi-state-RWKV-online-memory/README.md, causalab/ARCHITECTURE.md, HRM-Text/README.md
- repo-docs-md#1: rwkv-lm/README.md
- vllm-py#0: vllm/vllm/_custom_ops.py
- vllm-py#1: vllm/vllm/envs.py

| pos bin | full | ring | banked | routed-value-m16 | routed-value-m64 | d(ring-full) ± std | d(banked-full) ± std | d(routed-value-m16-full) ± std | d(routed-value-m64-full) ± std |
|---|---|---|---|---|---|---|---|---|---|
| 0k-1k | 4.5177 | 4.5177 | 4.5177 | 4.5177 | 4.5177 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 1k-2k | 2.8012 | 2.8012 | 2.8012 | 2.8012 | 2.8012 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 2k-3k | 2.8642 | 3.1821 | 3.0485 | 3.0869 | 2.9601 | +0.3179 ± 0.1914 | +0.1843 ± 0.1330 | +0.2227 ± 0.1268 | +0.0959 ± 0.1172 |
| 3k-4k | 2.1866 | 2.4089 | 2.3108 | 2.3616 | 2.2495 | +0.2223 ± 0.1354 | +0.1242 ± 0.1351 | +0.1749 ± 0.1151 | +0.0629 ± 0.0762 |
| 4k-5k | 2.3289 | 3.0976 | 2.9442 | 3.0412 | 2.8320 | +0.7687 ± 0.4677 | +0.6153 ± 0.4283 | +0.7123 ± 0.4509 | +0.5031 ± 0.3348 |
| 5k-6k | 2.1712 | 2.6793 | 2.5856 | 2.6432 | 2.5223 | +0.5081 ± 0.2278 | +0.4144 ± 0.2093 | +0.4720 ± 0.2535 | +0.3511 ± 0.2201 |
| 6k-7k | 1.9142 | 2.5288 | 2.4293 | 2.4885 | 2.4064 | +0.6146 ± 0.4358 | +0.5151 ± 0.3398 | +0.5743 ± 0.4039 | +0.4922 ± 0.3365 |
| 7k-8k | 2.2667 | 2.8414 | 2.7572 | 2.8115 | 2.7385 | +0.5747 ± 0.2821 | +0.4905 ± 0.1997 | +0.5449 ± 0.3020 | +0.4718 ± 0.2734 |
| 8k-9k | 2.0003 | 2.8527 | 2.7388 | 2.8144 | 2.6989 | +0.8525 ± 0.4878 | +0.7386 ± 0.4590 | +0.8141 ± 0.4663 | +0.6987 ± 0.4569 |
| 9k-10k | 2.3797 | 2.8929 | 2.8500 | 2.8794 | 2.8313 | +0.5131 ± 0.2734 | +0.4703 ± 0.2338 | +0.4996 ± 0.2872 | +0.4516 ± 0.2714 |
| 10k-11k | 1.8149 | 2.6402 | 2.5617 | 2.6048 | 2.5608 | +0.8252 ± 0.7021 | +0.7468 ± 0.6476 | +0.7898 ± 0.6726 | +0.7459 ± 0.5912 |
| 11k-12k | 1.6156 | 2.2501 | 2.2360 | 2.2425 | 2.1969 | +0.6346 ± 0.4120 | +0.6204 ± 0.4070 | +0.6269 ± 0.4020 | +0.5813 ± 0.3709 |
| 12k-13k | 1.9710 | 3.3471 | 3.2965 | 3.3344 | 3.2366 | +1.3761 ± 0.8126 | +1.3255 ± 0.7908 | +1.3634 ± 0.7980 | +1.2656 ± 0.7019 |
| 13k-14k | 1.8076 | 2.5592 | 2.5368 | 2.5322 | 2.4782 | +0.7516 ± 0.3594 | +0.7292 ± 0.3691 | +0.7246 ± 0.3458 | +0.6706 ± 0.3206 |
| 14k-15k | 1.7955 | 2.7567 | 2.6354 | 2.7335 | 2.6809 | +0.9613 ± 0.3778 | +0.8399 ± 0.3458 | +0.9380 ± 0.4014 | +0.8855 ± 0.3912 |
| 15k-16k | 1.8309 | 2.6823 | 2.6436 | 2.6616 | 2.6742 | +0.8514 ± 0.6622 | +0.8128 ± 0.6433 | +0.8307 ± 0.6237 | +0.8433 ± 0.6101 |

Sanity (pre-eviction exactness): bin 0-1k max |delta| vs full: ring 0.00e+00, banked 0.00e+00, routed-value-m16 0.00e+00, routed-value-m64 0.00e+00 -> PASS

NLL_CURVE_OK
