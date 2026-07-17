# wkvm

**A hypervisor for model state.** State-native inference for RWKV-7, GDN,
Mamba2, and hybrid-linear models, where the primary allocation object is a
fixed-size per-request **state slot**, not a paged KV block chain.

> WKV + KVM: states are the VMs—create, snapshot, fork, hibernate, resume, and
> live-migrate them. The engine is the hypervisor.

## Open WebUI quick start

This is the shortest path to a local browser chat backed by WKVM's Gemma
routed-span guest mode. It is a **compatibility demo**, not a same-semantics
replacement for full-KV Gemma serving: routed-span keeps a bounded approximate
memory, and the current chat API is a text-only greedy subset.

The tested target is Linux, an NVIDIA GPU with 24 GB VRAM, Python 3.12, and
roughly 40 GB of free disk for the checkpoint, environments, and caches. Read
the [complete installation and troubleshooting guide](docs/OPEN_WEBUI_DEMO.md)
before adapting it to another machine.

```bash
# Install isolated WKVM and Open WebUI 0.10.2 environments.
./scripts/open_webui_demo.sh install

# Accept google/gemma-4-E4B-it on Hugging Face, then authenticate and download.
hf auth login
hf download google/gemma-4-E4B-it \
  --local-dir "$HOME/models/gemma-4-E4B-it"

# Check the machine, start both loopback services, and verify a real chat.
WKVM_MODEL_DIR="$HOME/models/gemma-4-E4B-it" \
  ./scripts/open_webui_demo.sh doctor
WKVM_MODEL_DIR="$HOME/models/gemma-4-E4B-it" \
  ./scripts/open_webui_demo.sh start
./scripts/open_webui_demo.sh smoke
```

Open <http://127.0.0.1:3000>, create the first local account, and select
`wkvm-gemma-4-e4b-it`. Stop both services with
`./scripts/open_webui_demo.sh stop`. A pinned Docker Compose alternative is in
[`deploy/open-webui/compose.yaml`](deploy/open-webui/compose.yaml); it uses
Linux host networking because the WKVM CLI binds to `127.0.0.1`.

The helper starts WKVM with `--enable-openai-chat`, four conservative state
slots, normal EOS handling, and the checkpoint-native production profile. It
pins greedy UI defaults, disables unsupported background/tool features, and
does **not** reuse the high-memory B32 benchmark recipe or `--ignore-eos`.

### Current chat contract

- Blocking and streaming `/v1/chat/completions`, `/v1/models`, `/health`, and
  `/metrics` are available.
- Requests must use text messages, `temperature=0`, `top_p=1`, and `n=1`;
  tools, images, logprobs, and custom stop sequences are not supported.
- Open WebUI forwards user/chat identity headers so WKVM can park state per
  chat. Reuse still requires exact rendered-token prefix continuity; UI text
  normalization can safely force a session restart.
- Both services bind to loopback. WKVM currently has no API-key enforcement,
  so do not expose port 8000 to a LAN or the Internet.
- This UI path exercises approximate Gemma routed-span mode. WKVM's native
  RWKV-7 durable-state engine and API are separate paths.

## Performance evidence

There is no honest workload-independent “WKVM is 10x faster” claim. The
current evidence supports a scoped long-lived-session claim and also contains
workloads where vLLM remains faster.

| measured workload | WKVM result | vs vLLM | vs SGLang | outcome |
|---|---:|---:|---:|---|
| RTX 4090 provider HTTP, B16, 48 turns, 36,864-token initial context | 180.415s complete wall | **11.151x faster** | **26.079x faster** | scoped exploratory pass |
| Real Open WebUI 0.10.2, offered B32, 8 turns | 53.963 output tok/s | 0.586x; vLLM is 70.6% faster | 0.998x; effectively tied | accounting pass, strict reuse fail |
| Repeated A800 strict short-session gate, B64/ctx16K/out32 | worst-repeat envelope | 0.790x | 1.400x | overall gate fail |

The 48-turn result completed all 2,304 requests, but it is one paired run on a
desktop RTX 4090 with WKVM `routed_span_approximate` versus incumbent `full_kv`
semantics. The safe claim is: **on that predeclared long-lived workload, WKVM
measured 11.151x vLLM and 26.079x SGLang end to end**. It does not establish a
universal engine ranking or quality equivalence.

The real Open WebUI run completed 256/256 requests for every engine. WKVM
retained 32 states, but only 125/224 continuations exactly matched parked token
history; 99 normalized histories were safely restarted. High concurrency
increases residency and batching opportunity, but it does not by itself remove
WKVM's decode-kernel and microbatch bottlenecks.

Evidence and methodology:

- [48-turn RTX 4090 E2E report](experiments/results/gemma_4090_48turn_10x_20260717.md)
- [Real Open WebUI B32 x 8 report](experiments/results/open_webui_b32_t8_compare_20260714.md)
- [Repeated A800 comparison](experiments/results/gemma_a800_reliable_20260716/report.md)
- [Controlled B16 evidence audit](experiments/results/gemma_b16_evidence_audit_20260713.md)
- [10x E2E scope and optimization plan](docs/10X_E2E_PLAN.md)

## Routed-span demo

[![Gemma routed-span recurrent-mode demo](experiments/results/gemma_routed_span_demo.gif)](experiments/results/gemma_routed_span_demo.mp4)

Full-quality video:
[`experiments/results/gemma_routed_span_demo.mp4`](experiments/results/gemma_routed_span_demo.mp4)

## Serving API

`wkvm.gemma_server` exposes token-ID completion, streaming, submit/status,
cancellation, health, metrics, and model-discovery routes.
`/v1/chat/completions` is deliberately opt-in because enabling it loads the
tokenizer. `python -m wkvm.gemma_server` is equivalent to the
`wkvm-gemma-server` entry point.

```bash
python -m pip install -e '.[gemma-server]'

wkvm-gemma-server \
  --model /path/to/gemma-4-E4B-it \
  --served-model-name wkvm-gemma-4-e4b-it \
  --enable-openai-chat \
  --native-gemma-production-profile \
  --slots 4 --max-chat-sessions 4 --max-queue 16 \
  --request-timeout-s 600 --port 8000
```

`--ignore-eos`, forced output length, cache-emptying, and fixed B32 pool/graph
settings are benchmark controls, not normal serving defaults. For exact launch
provenance and comparison commands, use the linked result reports rather than
copying a historical benchmark profile into production.

## Why

For linear and hybrid models, per-request memory is **constant and tiny**. An
RWKV-7 7B state is roughly 20 MB—about 1000x smaller than long-context KV.
Building around that physics gives the engine:

- **Exact admission**—scheduling is counting free slots, with no fragmentation
  or watermark math.
- **Uniform decode batches**—whole-step CUDA graphs and high throughput scaling.
- **Sessions as objects**—hibernate/resume in one transfer, fork in one slot
  copy, and migrate in one RDMA write.
- **The Durable State API**—named, versioned, forkable, exportable, and mutable
  state handles (`/v1/states`). Mutable state is a capability prefix-keyed KV
  caches cannot represent.

Full-attention layers in hybrid models run in a deliberately simple paged
**guest pool**. Pure transformers are supported as guests for parity and, in
Gemma routed-span mode, with bounded approximate memory. See
[`docs/RECURRENT_MODE.md`](docs/RECURRENT_MODE.md).

## Documentation

- [`docs/OPEN_WEBUI_DEMO.md`](docs/OPEN_WEBUI_DEMO.md)—user install, first chat,
  verification, security, and troubleshooting.
- [`docs/ANGLE.md`](docs/ANGLE.md)—vLLM/SGLang architecture audit and design
  choices.
- [`docs/COMPARISON.md`](docs/COMPARISON.md)—engine comparison and historical
  measurements.
- [`docs/RECURRENT_MODE.md`](docs/RECURRENT_MODE.md)—bounded transformer
  recurrent-mode design.
- [`docs/NATIVE_ENGINE_PLAN.md`](docs/NATIVE_ENGINE_PLAN.md)—native engine plan.
- [`ROADMAP.md`](ROADMAP.md)—milestones.

## Status

**M3—Durable State API:** named, versioned, forkable, mutable state handles over
a tiered StateStore (GPU slot, pinned host, and NVMe safetensors). Measured demos
include 2,000 hibernated sessions at 2.29 MiB each, p50 8.2 ms / p99 9.5 ms
resume, process-restart recovery, 64-way fork, and provenance-recorded mutation.

**M2—native RWKV-7 serving:** a no-phases scheduler drives FLA-kernel decode
from arena state slots with continuous batching and per-batch-bucket CUDA
graphs. A 1.5B model measured 8.1k tok/s at B256 on an RTX 4090.

## Development

The base package is dependency-free so scheduler and arena invariants remain
CPU-testable.

```bash
python -m unittest discover -s tests -v
```

```text
wkvm/core/config.py     typed state families, slot layouts, and engine limits
wkvm/core/request.py    request lifecycle and computed-token invariant
wkvm/core/arena.py      per-family state-slot allocation and exact admission
wkvm/core/scheduler.py  no-phases continuous-batching scheduler
tests/                  CPU-first invariant and integration tests
```

## License

Apache-2.0 (LICENSE file pending).
