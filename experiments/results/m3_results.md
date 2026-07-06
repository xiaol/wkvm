# M3 results: Durable State API + the three generation-claim demos

*RWKV-7 191M (fla format), RTX 4090, 2026-07-06. Full log: `m3_run.log`. Test suite: 29/29 OK (5 new StateStore gates: cross-tier exactness, fork isolation, mutate provenance, rebuild-from-COLD-index, fingerprint rejection).*

## Demo (a) — warm-session fleet ✅

2000 sessions created + snapshotted through continuous batching (64 slots) in **31.7s**; WARM tier 4.46 GiB (**2.29 MiB/session**); 500 evicted to COLD safetensors in 4.9s.

| resume-to-next-token | n | p50 | p99 | max |
|---|---|---|---|---|
| WARM (pinned host → slot) | 224 | **8.2 ms** | 9.5 ms | 35.7 ms |
| COLD (NVMe safetensors → slot) | 76 | **8.6 ms** | 17.6 ms | 18.4 ms |

Exactness: **16/16** interrupted+resumed continuations token-identical to uninterrupted twins (both sides batch-1 to isolate the store from bf16 batch-shape effects). The p99-under-10ms number is the "thousands of hibernated sessions, sub-100ms resume" claim, beaten by 10×; per-session memory says one 24 GB card + one 64 GB host can keep ~25k sessions warm at this model size (1.5B: 12.2 MiB/session → ~5k).

## Demo (b) — persistent, forkable, mutable agent ✅

- **RESTART-EXACT: True** — session `agent@2` persisted to COLD, engine process exited, a fresh process rebuilt the store from `index.json` alone and produced the identical 24-token greedy continuation.
- **Fork**: 64 children in 544.8 ms (**8.5 ms/fork** — metadata + shared bytes, no state copy); 8 decoded concurrently with distinct seeds → 8 distinct continuations; parent unperturbed.
- **Mutate**: `decay(alpha=0.2)` produced `agent@3` with recorded provenance (`rule='decay' <- agent@2`); its continuation differs while the parent's remains bit-exact. This is the operation no prefix-keyed cache can represent: `agent@3` is not `f(any token prefix)`.

## Demo (c) — trainer/server kernel parity: measured, and the claim needs sharpening ⚠️

8 seeded rollouts × 64 tokens, scored two ways over the *same* fla kernels and weights (bf16): serving path (chunked prefill + per-token `fused_recurrent` decode) vs trainer path (one `chunk_rwkv7` full-sequence forward, `use_cache=False`):

- **max |Δlogprob| = 1.38e-1, mean = 1.99e-2, rollouts within 5e-3 everywhere: 0/8.**

Honest reading: *having* the same kernels on both sides does not make decode-time logprobs match trainer recompute — the fused-recurrent and chunked paths accumulate bf16 differently, and 64 recurrent steps compound it. What survives, still differentiated:

1. wkvm can score/verify with the **trainer-identical chunked path** (it is the same `model(input_ids, use_cache=False)` call) — a scoring mode that is bitwise-equal to the trainer *by construction*, no trainer modification needed. That remains beyond SGLang's deterministic mode, whose docs require modifying the training engine.
2. The decode↔train drift is now **quantified** (mean 2e-2 at bf16, 191M/64 tokens) — RL users can decide between importance-ratio correction (standard) or chunked-path rescoring per rollout.
3. Untested here: fp32 state + fp32 accumulation variants, and whether drift shrinks at larger models; both are follow-ups before the "bitwise on-policy RL" claim is used in anger.

## What M3 adds to the engine

`wkvm/store.py` (tiered StateStore, `name@version` lineage, fork, mutation-rule registry with `decay`/`merge`, model fingerprint binding), scheduler `on_finish` hook + `add_resumed_request` (resume = a request with a nonzero starting point — no special path), engine `snapshot/hibernate/resume/submit_from_handle`, `/v1/states` + `/v1/generate` HTTP layer (`wkvm/server.py`), and `experiments/m3_demos.py` (all three demos, reproducible).
