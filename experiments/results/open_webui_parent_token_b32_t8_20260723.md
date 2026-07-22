# Strict Open WebUI Parent-Token B32 x 8 (2026-07-23)

This checkpoint drives Open WebUI 0.10.2 through the browser backend path:
authenticated task creation, provider SSE, persistence, and terminal Socket.IO
events. One authenticated client manages 32 persisted conversations. It is not
32 browser connections or 32 human users.

## Result

Gemma-4-E4B-it ran on one RTX 4090 with 32 conversations, eight synchronized
turns, 13,824 rendered tokens on turn 0, 32 new user-content tokens on later
turns, and exactly 128 generated tokens per request. Every row completed
256/256 requests and emitted 32,768 tokens.

| Engine | Validation | Wall | Output tok/s | Continuation tok/s | p95 UI TTFT | p95 E2E | Peak whole GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| **WKVM `parent-token-v1`** | **pass; 224/224 reuse** | **117.888s** | **277.959** | **345.607** | 26.262s | 34.752s | 22,947 MiB |
| vLLM optimized | pass | 204.388s | 160.322 | 162.546 | **22.283s** | **27.297s** | 23,970 MiB |
| SGLang optimized | pass | 503.700s | 65.055 | 64.779 | 58.755s | 61.186s | 23,791 MiB |

On this exact workload, WKVM measured:

- **1.734x vLLM** and **4.273x SGLang** in complete-session generated-output throughput;
- **2.126x vLLM** and **5.335x SGLang** on the seven continuation turns;
- 117.888s complete wall, 86.500s sooner than vLLM and 385.812s sooner than SGLang.

The overall p95 includes the 32 cold first turns. WKVM continuation p95 was
12.035s E2E with 2.913s UI-path TTFT; its cold-turn p95 was 34.908s E2E with
26.303s TTFT.

## Strict continuity gate

The result passes the contract rather than counting safe restarts as reuse:

| Counter | Observed | Required |
|---|---:|---:|
| Eligible continuations | 224 | 224 |
| Engine session reuse hits | **224** | **224** |
| Exact-prefix + parent-bound hits | **12 + 212** | **224** |
| Parent-bound misses / rejections | **0 / 0** | **0 / 0** |
| Sessions opened / closed | **32 / 0** | **32 / 0** |
| Cache builds | **32** | **32** |
| Final parked sessions | **32** | **32** |
| Full reprefill turns | **0** | **0** |

WKVM reused 3,239,163 prefix tokens while scheduling 484,471 actual engine
tokens. The state contract binds model, user, chat, current and parent message
IDs, exact visible parent history, and the digest of the retained raw tokens.
Edits, branches, stale parents, expired state, or changed content restart safely.

## What fixed the Open WebUI restart leak

Visible text is not a lossless representation of generated token IDs. The
original exact-prefix path re-encoded persisted assistant text and therefore
lost hidden or noncanonical token history. `parent-token-v1` keeps the original
raw tokens and verifies the Open WebUI parent relationship before appending only
the new structural/user delta.

Two text details also had to match the real application path:

1. WKVM now binds the text actually emitted by its incremental SSE decoder, not
   a separate whole-sequence decode.
2. Open WebUI 0.10.2 applies `.strip()` to the final output item before database
   persistence, terminal Socket.IO delivery, and next-turn reconstruction.
   WKVM binds that same outer-whitespace-normalized form.

The benchmark sets `reasoning_tags=false`; reasoning extraction intentionally
changes next-turn content and is outside this stateful contract. Content-changing
filters remain safe because the visible-history check rejects them and rebuilds
state.

## Why this is not 10x

The valid claim is 1.734x vLLM and 4.273x SGLang here, not 10x.

- vLLM's 204.388s wall would require WKVM at or below 20.439s for 10x, but
  WKVM's first turn alone took 34.926s.
- This workload has seven warm continuations. The separate 11.151x vLLM result
  uses 47 continuations and a 36,864-token initial context, so recurrent-state
  reuse is amortized much longer while incumbent KV histories face more memory
  pressure.
- WKVM retained and ran 32 sessions, but the model-call batch ceiling was 16.
  The run recorded 1,015 decode microbatch splits and 2,061 decode model calls.
- WKVM used no decode CUDA graph and averaged 61.48% GPU utilization. The
  optimized vLLM comparison enabled prefix caching, Gemma KV-sharing fast
  prefill, and full-decode CUDA graphs, averaging 93.28% GPU utilization.
- Open WebUI task, persistence, and Socket.IO work is real, but p95 task ACK was
  about 1.14s for all three engines. It is not the dominant missing factor.

High concurrency provides state residency and batching opportunity; it does not
automatically make the decode kernels execute 32 rows efficiently.

## Comparison scope

This is a controlled single cross-run checkpoint, not a repeated publication
envelope. All rows use the same Open WebUI version, initial prompt fingerprint,
token counts, sampling controls, request order, and browser-backend protocol.
Later histories are autonomous because each engine appends its own generated
assistant text.

The incumbent rows are optimized rather than intentionally weakened:

- vLLM uses prefix caching, Gemma KV-sharing fast prefill, and
  `FULL_DECODE_ONLY` CUDA graphs;
- SGLang uses its radix cache and full decode graph;
- WKVM uses approximate routed-span semantics, while the incumbents use full KV
  attention semantics, so this report is performance evidence, not a quality
  equivalence proof.

Three rotated clean repeats remain the next publication gate.

## Artifacts

The committed [summary JSON](open_webui_parent_token_b32_t8_20260723.json)
contains audited aggregates, launch-profile details, and SHA-256 hashes for the
request-level artifacts. Raw JSON remains in the external benchmark store.

Reproduce the application path with
`experiments/open_webui_multiturn_bench.py`, including both
`--configure-wkvm-parent-token-contract` and `--require-wkvm-session-reuse`.
The high-memory B32 server profile is a benchmark configuration, not the normal
four-slot demo profile documented in [the Open WebUI guide](../../docs/OPEN_WEBUI_DEMO.md).
