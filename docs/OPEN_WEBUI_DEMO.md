# Open WebUI demo

This guide runs Open WebUI 0.10.2 against WKVM's OpenAI-compatible Gemma
server. It is designed for a first local chat, not for reproducing the tuned
B32 benchmark.

The demo uses Gemma routed-span guest mode. That mode keeps bounded approximate
memory instead of full transformer KV, so compare its throughput and memory as
a different serving mode—not as exact full-KV quality equivalence. WKVM's
native RWKV-7 durable-state path is separate.

## Requirements

The verified configuration is:

- Linux x86_64 with an NVIDIA CUDA-capable GPU;
- an RTX 4090-class 24 GB GPU for the documented Gemma profile;
- 35–40 GB of free disk for the model, two isolated environments, and caches;
- `git`, `curl`, and [`uv`](https://docs.astral.sh/uv/);
- ports 8000 and 3000 free on loopback;
- access to the gated `google/gemma-4-E4B-it` repository on Hugging Face.

The helper pins Python 3.12 for Open WebUI because Open WebUI 0.10.2 does not
support Python 3.13 or newer. WKVM itself requires Python 3.11 or newer.

If the system disk is small, put all demo state on a larger volume before
installing:

```bash
export WKVM_DEMO_HOME=/mnt/fast/wkvm-open-webui
export WKVM_MODEL_DIR=/mnt/fast/models/gemma-4-E4B-it
```

These variables must be set again in later shells, or added to a private shell
profile.

## 1. Install the environments

From the repository root:

```bash
./scripts/open_webui_demo.sh install
```

This creates a dedicated WKVM virtual environment under `WKVM_DEMO_HOME` and
installs the checkout with the `gemma-server` extra. It installs Open WebUI
0.10.2 separately as a CPU-side `uv` tool so Open WebUI does not pull a second
CUDA runtime into the WKVM environment. It also installs the current `hf` CLI.

Inspect the commands without changing the machine:

```bash
DRY_RUN=1 ./scripts/open_webui_demo.sh install
```

## 2. Download the model

First accept the Gemma license on the Hugging Face model page. Authenticate
without placing a token in shell history, then download the complete repository
to a local directory:

```bash
hf auth login
hf auth whoami

hf download google/gemma-4-E4B-it \
  --local-dir "${WKVM_MODEL_DIR:-$HOME/models/gemma-4-E4B-it}"
```

The native loader needs a local directory containing `config.json`, tokenizer
files, and the model safetensors. Passing only the Hub repository ID to
`--model` is not supported by this profile.

For users who deliberately use HF-Mirror, the same authenticated download can
be routed through it:

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download google/gemma-4-E4B-it \
  --local-dir "${WKVM_MODEL_DIR:-$HOME/models/gemma-4-E4B-it}"
```

The mirror does not replace Hugging Face authorization. Use the current `hf`
command; `huggingface-cli` is deprecated.

## 3. Check and start

Run the preflight first:

```bash
./scripts/open_webui_demo.sh doctor
```

Then start WKVM, wait for `/health`, start Open WebUI, and wait for its health
endpoint:

```bash
./scripts/open_webui_demo.sh start
```

Model loading can take several minutes on the first run. The helper prints the
log locations and waits up to 15 minutes by default. Inspect lifecycle state at
any time:

```bash
./scripts/open_webui_demo.sh status
./scripts/open_webui_demo.sh logs
```

The launch environment pins greedy model defaults, selects legacy function
calling, and disables background title, tag, follow-up, and context-compaction
requests so a fresh UI does not inject unsupported tools or sampling settings.

## 4. Verify the provider

The bundled smoke test checks health, model discovery, one real non-streaming
chat completion, and the Open WebUI health endpoint:

```bash
./scripts/open_webui_demo.sh smoke
```

The equivalent provider checks are:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer wkvm-local' \
  -H 'Content-Type: application/json' \
  -H 'X-OpenWebUI-User-Id: smoke-user' \
  -H 'X-OpenWebUI-Chat-Id: smoke-chat' \
  -d '{
    "model": "wkvm-gemma-4-e4b-it",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 64,
    "stream": false
  }'
```

WKVM does not currently enforce the dummy bearer key. Loopback binding is the
provider's security boundary.

## 5. Send the first browser chat

Open <http://127.0.0.1:3000>. Keep Open WebUI authentication enabled; the first
registered local account becomes the administrator. Select
`wkvm-gemma-4-e4b-it`, create a new chat, and send a text-only prompt.

For this provider:

- leave sampling at greedy (`temperature=0`, `top_p=1`);
- do not enable tools, web search, image input, memory injection, or custom stop
  sequences;
- keep `n=1`; logprobs are not supported;
- both streaming and blocking text responses work.

Open WebUI forwards `X-OpenWebUI-User-Id` and `X-OpenWebUI-Chat-Id`. WKVM uses
the model, user, and chat tuple as the parked-session identity. After two turns,
inspect the counters:

```bash
curl -fsS http://127.0.0.1:8000/metrics
```

Relevant fields are `engine.session_reuse_hits`,
`engine.session_reuse_misses`, `engine.parked_sessions`, and
`server.total_errors`. Reuse is deliberately strict: the next rendered prompt
must contain the exact prior token history as a prefix. Open WebUI normalizes
persisted assistant text, so a safe retire-and-restart can occur instead of a
reuse hit. The measured B32 run reused 125 of 224 continuations and safely
restarted 99 normalized histories.

## Docker Compose alternative

The native `uv` path is the primary flow because it works without a container
runtime. If Docker Compose is installed, start WKVM with the helper or the
manual command below, then launch the pinned UI container:

```bash
docker compose -f deploy/open-webui/compose.yaml up -d
curl -fsS http://127.0.0.1:3000/health
```

Stop only the container with:

```bash
docker compose -f deploy/open-webui/compose.yaml down
```

The Compose file uses Linux `network_mode: host`. This is required because the
current WKVM CLI binds only to `127.0.0.1`; a normal bridge container cannot
reach that listener through `host.docker.internal`. The UI also binds to
loopback, and its database persists in the `open-webui-data` volume.

## Manual launch

The helper is a convenience wrapper around these two services. To launch WKVM
manually from an environment where `.[gemma-server]` is installed:

```bash
wkvm-gemma-server \
  --model "$WKVM_MODEL_DIR" \
  --served-model-name wkvm-gemma-4-e4b-it \
  --enable-openai-chat \
  --native-gemma-production-profile \
  --slots 4 \
  --max-chat-sessions 4 \
  --max-queue 16 \
  --request-timeout-s 600 \
  --chat-session-ttl-s 1800 \
  --port 8000
```

In a second shell, launch the separately installed Open WebUI tool:

```bash
export DATA_DIR="${WKVM_DEMO_HOME:-$HOME/.local/share/wkvm-open-webui-demo}/open-webui-data"
export WEBUI_AUTH=true
export ENABLE_OLLAMA_API=false
export ENABLE_OPENAI_API=true
export OPENAI_API_BASE_URLS=http://127.0.0.1:8000/v1
export OPENAI_API_KEYS=wkvm-local
export ENABLE_FORWARD_USER_INFO_HEADERS=true
export ENABLE_WEBSOCKET_SUPPORT=true
export ENABLE_PERSISTENT_CONFIG=false
export DEFAULT_MODELS=wkvm-gemma-4-e4b-it
export DEFAULT_MODEL_PARAMS='{"temperature":0,"top_p":1,"function_calling":"legacy"}'
export DEFAULT_MODEL_METADATA='{"capabilities":{"builtin_tools":false,"vision":false,"file_upload":false,"file_context":false,"web_search":false,"image_generation":false,"code_interpreter":false,"terminal":false,"memory":false}}'
export ENABLE_TITLE_GENERATION=false
export ENABLE_TAGS_GENERATION=false
export ENABLE_FOLLOW_UP_GENERATION=false
export ENABLE_CONTEXT_COMPACTION=false
export ENABLE_REALTIME_CHAT_SAVE=false
export ENABLE_CODE_INTERPRETER=false
export ENABLE_MEMORIES=false
export ENABLE_WEB_SEARCH=false
export ENABLE_IMAGE_GENERATION=false

open-webui serve --host 127.0.0.1 --port 3000
```

`OPENAI_API_BASE_URLS` must include `/v1`; Open WebUI discovers the model from
`/v1/models`. `ENABLE_PERSISTENT_CONFIG=false` makes these environment values
authoritative instead of allowing an older database connection setting to
override them. Model- or request-level Advanced Parameters can override the
global defaults, so WKVM still validates every request and safely rejects
non-greedy or tool-bearing payloads.

## Stop, logs, and reset

```bash
./scripts/open_webui_demo.sh stop
./scripts/open_webui_demo.sh logs
```

Open WebUI conversations and accounts live under
`$WKVM_DEMO_HOME/open-webui-data`. To reset the UI, stop both services and move
that directory aside. Do not delete it unless losing every local chat and
account is intentional.

## Troubleshooting

### `hf download` returns 401 or 403

Accept the model license in the browser, run `hf auth whoami`, and authenticate
with a token that has read access. A mirror endpoint does not bypass the gate.

### The model directory is rejected

Confirm the download completed and that the directory contains `config.json`,
tokenizer assets, and all safetensor shards. Point `WKVM_MODEL_DIR` at the local
directory, not at `google/gemma-4-E4B-it`.

### WKVM exits or reports CUDA OOM

Stop other GPU processes and inspect the WKVM log. The documented profile is
for a 24 GB RTX 4090-class GPU. Four state slots are intentionally conservative;
the historical B32 profile is close to the device ceiling and is not a fallback
for normal chat.

### Open WebUI shows no model

Verify `curl http://127.0.0.1:8000/v1/models`, confirm the provider base ends in
`/v1`, and restart Open WebUI. If using a manually configured old data directory,
set `ENABLE_PERSISTENT_CONFIG=false` or update the connection in the admin UI.

### A request returns HTTP 400

Check that the UI did not add tools, images, non-greedy sampling, multiple
choices, logprobs, or custom stop sequences. The error body names the unsupported
field. Normal chat must use `temperature=0` and `top_p=1`.

### Reuse does not increase on every turn

Serving is still correct. Exact parked-state reuse is conditional on token-prefix
continuity; Open WebUI persistence normalization can change the reconstructed
history, in which case WKVM safely starts a fresh state.

### A port is already in use or a PID is stale

Run `./scripts/open_webui_demo.sh status`, then `stop`. If another application
owns the port, stop it or choose `WKVM_PORT` and `OPEN_WEBUI_PORT` overrides.
Read the printed log paths before removing any PID file manually.

## Security

This setup is local-only. Keep both listeners on `127.0.0.1`, keep Open WebUI
authentication enabled, and treat its data directory and secret as private. For
LAN or Internet access, add provider authentication, a TLS reverse proxy, and a
reviewed network policy first; changing a bind address alone is unsafe.

## Benchmark reproduction is different

The historical Open WebUI B32 x 8 result uses 32 slots, fixed output length,
`--ignore-eos`, explicit token-pool sizing, strict Triton settings, cache
emptying, and benchmark-specific UI controls. Those settings reproduce one
artifact and can OOM or distort normal chat behavior. Use the
[historical report](../experiments/results/open_webui_b32_t8_compare_20260714.md)
only when reproducing that exact measurement.
