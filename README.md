# wkvm

**A hypervisor for model state.** State-native inference engine for RWKV-7 / GDN / Mamba2 and hybrid-linear models — where the primary allocation object is a fixed-size per-request **state slot**, not a paged KV block chain.

> WKV + KVM: states are the VMs — create, snapshot, fork, hibernate, resume, live-migrate. The engine is the hypervisor.

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

M0: core bookkeeping — state arena allocator and the no-phases scheduler, pure Python, unit-tested without a GPU (the scheduler never touches a tensor; it schedules token counts against slot ids).

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
