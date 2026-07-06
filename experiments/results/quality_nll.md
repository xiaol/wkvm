# Position-resolved NLL: full vs recurrent modes (gemma-4-E4B-it)

6 natural documents x 16384 tokens, teacher-forced with chunk-2048 prefill (cache evolves as in generation; divergence granularity is one chunk). Modes as in quality_grid.md. Provenance:

- papers-tex#0: Autoresearch_ideas/papers/associative_state_universal_transformers/main.tex, Autoresearch_ideas/papers/generic_triple_latent/generic_triple_latent.tex
- papers-tex#1: Autoresearch_ideas/papers/on_device_meta_learning_agents/paper58.tex, Autoresearch_ideas/papers/recurrent_ffn/main.tex
- repo-docs-md#0: Multi-state-RWKV-online-memory/README.md, causalab/ARCHITECTURE.md, HRM-Text/README.md
- repo-docs-md#1: rwkv-lm/README.md
- vllm-py#0: vllm/vllm/_custom_ops.py
- vllm-py#1: vllm/vllm/envs.py

| pos bin | full | ring | banked | routed-value-m64 | routed-span-m64 | d(ring-full) ± std | d(banked-full) ± std | d(routed-value-m64-full) ± std | d(routed-span-m64-full) ± std |
|---|---|---|---|---|---|---|---|---|---|
| 0k-1k | 4.5177 | 4.5177 | 4.5177 | 4.5177 | 4.5177 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 1k-2k | 2.8012 | 2.8012 | 2.8012 | 2.8012 | 2.8012 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 2k-3k | 2.8642 | 3.1821 | 3.0485 | 2.9601 | 2.8665 | +0.3179 ± 0.1914 | +0.1843 ± 0.1330 | +0.0959 ± 0.1172 | +0.0023 ± 0.0133 |
| 3k-4k | 2.1866 | 2.4089 | 2.3108 | 2.2495 | 2.1974 | +0.2223 ± 0.1354 | +0.1242 ± 0.1351 | +0.0629 ± 0.0762 | +0.0107 ± 0.0141 |
| 4k-5k | 2.3289 | 3.0976 | 2.9442 | 2.8320 | 2.3643 | +0.7687 ± 0.4677 | +0.6153 ± 0.4283 | +0.5031 ± 0.3348 | +0.0354 ± 0.0523 |
| 5k-6k | 2.1712 | 2.6793 | 2.5856 | 2.5223 | 2.2113 | +0.5081 ± 0.2278 | +0.4144 ± 0.2093 | +0.3511 ± 0.2201 | +0.0401 ± 0.0394 |
| 6k-7k | 1.9142 | 2.5288 | 2.4293 | 2.4064 | 1.9634 | +0.6146 ± 0.4358 | +0.5151 ± 0.3398 | +0.4922 ± 0.3365 | +0.0492 ± 0.0427 |
| 7k-8k | 2.2667 | 2.8414 | 2.7572 | 2.7385 | 2.3379 | +0.5747 ± 0.2821 | +0.4905 ± 0.1997 | +0.4718 ± 0.2734 | +0.0712 ± 0.0287 |
| 8k-9k | 2.0003 | 2.8527 | 2.7388 | 2.6989 | 2.1045 | +0.8525 ± 0.4878 | +0.7386 ± 0.4590 | +0.6987 ± 0.4569 | +0.1043 ± 0.0499 |
| 9k-10k | 2.3797 | 2.8929 | 2.8500 | 2.8313 | 2.4901 | +0.5131 ± 0.2734 | +0.4703 ± 0.2338 | +0.4516 ± 0.2714 | +0.1104 ± 0.0607 |
| 10k-11k | 1.8149 | 2.6402 | 2.5617 | 2.5608 | 1.9560 | +0.8252 ± 0.7021 | +0.7468 ± 0.6476 | +0.7459 ± 0.5912 | +0.1411 ± 0.1164 |
| 11k-12k | 1.6156 | 2.2501 | 2.2360 | 2.1969 | 1.7587 | +0.6346 ± 0.4120 | +0.6204 ± 0.4070 | +0.5813 ± 0.3709 | +0.1431 ± 0.1546 |
| 12k-13k | 1.9710 | 3.3471 | 3.2965 | 3.2366 | 2.3195 | +1.3761 ± 0.8126 | +1.3255 ± 0.7908 | +1.2656 ± 0.7019 | +0.3485 ± 0.2109 |
| 13k-14k | 1.8076 | 2.5592 | 2.5368 | 2.4782 | 1.9975 | +0.7516 ± 0.3594 | +0.7292 ± 0.3691 | +0.6706 ± 0.3206 | +0.1899 ± 0.0886 |
| 14k-15k | 1.7955 | 2.7567 | 2.6354 | 2.6809 | 2.0436 | +0.9613 ± 0.3778 | +0.8399 ± 0.3458 | +0.8855 ± 0.3912 | +0.2481 ± 0.1537 |
| 15k-16k | 1.8309 | 2.6823 | 2.6436 | 2.6742 | 2.1009 | +0.8514 ± 0.6622 | +0.8128 ± 0.6433 | +0.8433 ± 0.6101 | +0.2701 ± 0.2616 |

Sanity (pre-eviction exactness): bin 0-1k max |delta| vs full: ring 0.00e+00, banked 0.00e+00, routed-value-m64 0.00e+00, routed-span-m64 0.00e+00 -> PASS

NLL_CURVE_OK
