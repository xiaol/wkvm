# wkvm

**A hypervisor for model state.** State-native inference engine for RWKV-7 / GDN / Mamba2 and hybrid-linear models — where the primary allocation object is a fixed-size per-request **state slot**, not a paged KV block chain.

> WKV + KVM: states are the VMs — create, snapshot, fork, hibernate, resume, live-migrate. The engine is the hypervisor.

## Routed-Span Demo

[![Gemma routed-span recurrent-mode demo](experiments/results/gemma_routed_span_demo.gif)](experiments/results/gemma_routed_span_demo.mp4)

Full-quality MP4: [`experiments/results/gemma_routed_span_demo.mp4`](experiments/results/gemma_routed_span_demo.mp4)

Previous ring/concurrency demo: [`experiments/results/gemma_wkvm_style_demo.mp4`](experiments/results/gemma_wkvm_style_demo.mp4)

## Routed-span vs vLLM/SGLang

Gemma-4-E4B-it on one RTX 4090. vLLM and SGLang are full-KV transformer engines; wkvm here is the patched-HF routed-span recurrent-mode PoC (`sink16 + ring1024 + routed span bank m64`), so this is a memory/throughput comparison, not a same-semantics quality claim.

**Single long prompt + long output**: 13,824-token prompt + 512-token output, greedy decode, `ignore_eos=True`.

| engine | semantics | facts recovered | prefill+1st | full wall | decode tok/s | e2e output tok/s | memory observed | raw result |
|---|---|---:|---:|---:|---:|---:|---|---|
| wkvm routed-span m64 | approximate recurrent | yes | 1.380s | 11.237s | 51.8 | 45.6 | 14.67 GiB reserved; 52.9 MiB cache | [`json`](experiments/results/long_gen_13824_512_wkvm_routed_span_m64.json) |
| vLLM 0.24.0 | full KV | yes | 1.813s | 8.251s | 79.4 | 62.1 | 22.54 GiB device used; 18.42 GiB alloc | [`json`](experiments/results/long_gen_13824_512_vllm.json) |
| SGLang 0.5.14 | full KV | yes | 1.257s | 8.515s | 70.4 | 60.1 | 21.79 GiB peak device | [`json`](experiments/results/long_gen_13824_512_sglang.json) |

**Distinct long-prompt concurrency**: wkvm row is the fresh routed-span run at 13,824 context tokens/session and 128 decode tokens/session. vLLM/SGLang rows are the nearest tracked full-KV engine capacity runs at 16,384 context tokens/session and 128 decode tokens/session, included to anchor the incumbent memory shape.

| engine | workload | resident sessions | aggregate decode | memory/capacity note | latency note | source |
|---|---|---:|---:|---|---|---|
| wkvm routed-span m64 | 13,824 ctx, distinct prompts | **16 green**; 32/48 completed over headroom | **643.9 tok/s** green; 1039.4 tok/s over headroom | 15.97 GiB reserved, 913 MiB routed cache; green means 19 GiB cap with 1 GiB headroom | p50=p95 3.181s decode | [`json`](experiments/results/gemma_routed_span_distinct_concurrency.json) |
| vLLM 0.24.0 | nearest 16,384 ctx full-KV run | 9 cap; N=8 measured | 285.6 tok/s | 21.81 GiB device used; 18.26 GiB alloc | wall 13.42s at N=8; p50/p95 not recorded | [`bench`](experiments/results/bench_vllm_gemma4e4b.md) |
| SGLang 0.5.14 | nearest 16,384 ctx full-KV run | 1 true concurrent; N=8 queue-limited | 68 tok/s | 25,360-token KV pool on this stack | queue-limited; p50/p95 not recorded | [`bench`](experiments/results/bench_sglang_gemma4e4b.md) |

Readout: routed-span is slower for one exact long generation than vLLM/SGLang full-KV, but it keeps many independent long sessions resident under a tighter memory line. The measured advantage is bounded-memory long-context concurrency when approximate recurrent semantics are acceptable; it is not a replacement for full-KV serving when exact transformer behavior is required.

## Why

For linear/hybrid models, per-request memory is **constant and tiny** (an RWKV-7 7B state is ~20MB — ~1000× smaller than long-context KV). Built around that physics instead of paged KV, an engine gets:

- **Exact admission** — scheduling is counting free slots; no fragmentation, no watermark math.
- **Uniform decode batches** — whole-step CUDA graphs, Albatross-class throughput scaling.
- **Sessions as objects** — hibernate/resume in one transfer, fork in one slot copy, migrate in one RDMA write.
- **The Durable State API** — named, versioned, forkable, exportable, **mutable** state handles (`/v1/states`). Mutable state violates the `state ≡ f(token-prefix)` invariant that paged-KV cache indexes are built on; it is the capability incumbent engines structurally cannot follow.

Full-attention layers in hybrid models (Qwen3-Next / Kimi-Linear class) run in a deliberately simple paged **guest pool**. Pure transformers are supported as guests for parity — and, later, at constant footprint via **recurrent mode** (sink + KV ring + a segmented bank of RWKV-7 states over evicted context; see `docs/RECURRENT_MODE.md`).

## Design documents

- [`docs/ANGLE.md`](docs/ANGLE.md) — the full vLLM/SGLang architecture map (724k/626k LOC audited), what to steal, what to refuse, the nine candidate angles and why this one survived adversarial review.
- [`docs/RECURRENT_MODE.md`](docs/RECURRENT_MODE.md) — constant-footprint transformer serving via multi-state (layer-wise and context-length-wise) memory banks.
- [`ROADMAP.md`](ROADMAP.md) — milestones.

## Status

**M3 — the Durable State API is real and measured** (`experiments/results/m3_results.md`): named/versioned/forkable/**mutable** state handles over a tiered StateStore (GPU slot / pinned host / NVMe safetensors) with `/v1/states`. Demos on one 4090: 2000 hibernated sessions at 2.29 MiB each resuming in **p50 8.2ms / p99 9.5ms** (16/16 exactness); an agent session surviving a **real process restart bit-exactly**, forked 64× at 8.5ms/fork, and mutated (`decay`) with recorded provenance while its parent stays intact — the operation prefix-keyed caches cannot represent. Trainer/server logprob drift quantified honestly (mean 2e-2 at bf16 across fused-vs-chunked paths; trainer-identical chunked rescoring available by construction).

**M2**: the engine serves RWKV-7 end to end — no-phases scheduler driving FLA-kernel decode from arena state slots, continuous batching (batch-vs-sequential token-identical), per-batch-bucket CUDA graphs, 8.1k tok/s at B=256 on a 4090 (1.5B). Plus a measured PoC of **recurrent mode** for transformers on gemma-4-E4B (constant footprint + segmented state bank; needle recall to 32k). See [`docs/COMPARISON.md`](docs/COMPARISON.md) for head-to-head numbers vs vLLM, SGLang, and Albatross on the same GPU, and `experiments/results/` for all raw tables.

```bash
python -m unittest discover -s tests -v
```

## Layout

```
wkvm/core/config.py     # typed specs: state families, slot layouts, engine limits
wkvm/core/request.py    # Request lifecycle, the num_computed_tokens invariant
wkvm/core/arena.py      # StateArena: per-family slot allocator, exact admission
wkvm/core/scheduler.py  # no-phases continuous-batching scheduler
tests/                  # CPU-only invariant tests
```

## License

Apache-2.0 (LICENSE file pending).
