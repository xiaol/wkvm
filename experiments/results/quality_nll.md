# Position-resolved NLL: full vs ring vs banked (gemma-4-E4B-it)

6 natural documents x 16384 tokens, teacher-forced with chunk-2048 prefill (cache evolves as in generation; divergence granularity is one chunk). Modes as in quality_grid.md. Provenance:

- papers-tex#0: Autoresearch_ideas/papers/associative_state_universal_transformers/main.tex, Autoresearch_ideas/papers/generic_triple_latent/generic_triple_latent.tex
- papers-tex#1: Autoresearch_ideas/papers/on_device_meta_learning_agents/paper58.tex, Autoresearch_ideas/papers/recurrent_ffn/main.tex
- repo-docs-md#0: Multi-state-RWKV-online-memory/README.md, causalab/ARCHITECTURE.md, HRM-Text/README.md
- repo-docs-md#1: rwkv-lm/README.md
- vllm-py#0: vllm/vllm/_custom_ops.py
- vllm-py#1: vllm/vllm/envs.py

| pos bin | full | ring | banked | d(ring-full) ± std | d(banked-full) ± std |
|---|---|---|---|---|---|
| 0k-1k | 4.5177 | 4.5177 | 4.5177 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 1k-2k | 2.8012 | 2.8012 | 2.8012 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 2k-3k | 2.7398 | 3.0587 | 2.9250 | +0.3189 ± 0.1924 | +0.1852 ± 0.1350 |
| 3k-4k | 2.1135 | 2.3311 | 2.2332 | +0.2176 ± 0.1373 | +0.1197 ± 0.1338 |
| 4k-5k | 2.2961 | 3.0364 | 2.9098 | +0.7403 ± 0.4657 | +0.6137 ± 0.4284 |
| 5k-6k | 2.1647 | 2.6683 | 2.5764 | +0.5036 ± 0.2338 | +0.4117 ± 0.2125 |
| 6k-7k | 1.9402 | 2.6086 | 2.4782 | +0.6683 ± 0.4696 | +0.5380 ± 0.3538 |
| 7k-8k | 2.2826 | 2.8369 | 2.7722 | +0.5543 ± 0.2674 | +0.4896 ± 0.1988 |
| 8k-9k | 2.0294 | 2.8351 | 2.7321 | +0.8056 ± 0.5267 | +0.7026 ± 0.4857 |
| 9k-10k | 2.3849 | 2.8785 | 2.8365 | +0.4936 ± 0.2804 | +0.4515 ± 0.2375 |
| 10k-11k | 1.7604 | 2.5845 | 2.5162 | +0.8241 ± 0.7032 | +0.7558 ± 0.6382 |
| 11k-12k | 1.6113 | 2.2800 | 2.2629 | +0.6687 ± 0.3856 | +0.6515 ± 0.3831 |
| 12k-13k | 2.0104 | 3.3702 | 3.2704 | +1.3598 ± 0.8178 | +1.2600 ± 0.8316 |
| 13k-14k | 1.7797 | 2.5263 | 2.4670 | +0.7466 ± 0.3591 | +0.6873 ± 0.3837 |
| 14k-15k | 1.7604 | 2.7123 | 2.5866 | +0.9519 ± 0.3813 | +0.8262 ± 0.3538 |
| 15k-16k | 1.8996 | 2.7109 | 2.6682 | +0.8113 ± 0.6858 | +0.7686 ± 0.6713 |

Sanity (pre-eviction exactness): bin 0-1k max |delta| vs full: ring 0.00e+00, banked 0.00e+00 -> PASS

NLL_CURVE_OK
