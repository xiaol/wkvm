# Qwen3.5 Hybrid Execution Contract (M4)

This note is the counterpart of `docs/gemma_native_contract.md` for the first
hybrid-linear model wkvm serves natively: Qwen3.5 (Gated DeltaNet + full
attention, the "Qwen3-Next class" of ROADMAP M4). It records what the engine
owns, what it still borrows from HF, and the exact state semantics the
`Qwen35StateBank` / `SlotCache` pair must preserve
(`wkvm/models/qwen35.py`, `wkvm/runner/hybrid_state.py`,
`wkvm/runner/hybrid_runner.py`).

## Model scope

- Target checkpoint: Qwen3.5-9B (`Qwen3_5ForConditionalGeneration`, text
  tower `Qwen3_5TextModel`, 32 decoder layers). Text-only serving; the vision
  tower is dropped at load. `qwen3_5_text` checkpoints load the same way.
- Same ownership split as M1 RWKV-7: HF decoder layers are the compute
  graph; wkvm owns state, positions, masks and the cache object. HF's
  `Qwen3_5TextModel.forward` is never called (it would build masks against a
  `DynamicCache`); `Qwen35Decoder.forward` runs the layer loop.
- One GPU, bf16, greedy/temperature sampling through the existing sampler.

## Layer types and state families

`config.layer_types` (Qwen3.5-9B: `full_attention` every 4th layer):

| layer type | count (9B) | family | unit | shape (layer-major bank) | dtype |
|---|---:|---|---|---|---|
| `linear_attention` | 24 | `gdn_state` | slot | `[24, H_v=32, K=128, V=128]` | fp32 |
| `linear_attention` | 24 | `gdn_conv` | slot | `[24, conv_dim=8192, kernel=4]` | model |
| `full_attention` | 8 | `guest_kv` | **page** | K and V `[8, kv_heads=4, page_tokens, head_dim=256]` per page | model |

Per request: 48 MiB + 1.5 MiB of recurrent state (constant) plus
`ceil((prompt + max_new_tokens) / page_tokens)` pages at 32 KiB per token
(8 MiB per 256-token page). The arena hands out slots and pages together
(`slots["guest_kv"]` is the tuple of page ids); token `t` lives at
`pages[t // page_tokens]`, offset `t % page_tokens`. Page 0 is reserved as
the read target of masked-out positions.

The recurrent state is fp32 because the HF reference kernels
(`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`) emit and
consume fp32 states. `conv_dim = 2*num_k_heads*head_k_dim +
num_v_heads*head_v_dim`.

`guest_len` (tokens resident in the pages) is bank-owned host metadata keyed
by the request's `gdn_state` slot and exported with snapshots; the engine's
`num_computed_tokens` equals it for a live request.

## Zero state is fresh state

For every family, all-zeros is exactly the cache-less first forward:

- `gdn_state = 0` is what the kernels use for `initial_state=None`.
- `gdn_conv = 0` prepended to the new tokens reproduces the left zero padding
  of the cache-less causal conv (`padding = kernel-1`).
- an empty guest window is length 0.

Hence `SlotCache.has_previous_state()` is unconditionally `True`: the HF
GDN layer always takes the seeded path, and takes the single-token recurrent
kernel exactly when `seq_len == 1`. Admission is `zero_slots`.

## Gated DeltaNet cache calls (mirrors `LinearAttentionLayer`)

| HF call | wkvm behaviour |
|---|---|
| `cache.has_previous_state(i, state_idx=0)` | `True` |
| `cache.layers[i].conv_states[0]` | gathered `[B, conv_dim, kernel]`; updated **in place** by `causal_conv1d_update` on the decode path |
| `cache.layers[i].recurrent_states[0]` | gathered `[B, H_v, K, V]` fp32 |
| `cache.layers[i].record_past` | `False` |
| `cache.update_conv_state(x, i, conv_kernel_size=k)` | returns `cat([window, x])`; window := last `k` columns |
| `cache.update_recurrent_state(s, i)` | copies `s` into the gathered state |

`scatter` writes the gathered tensors back with `index_copy_`.

## Guest attention calls

| HF call | wkvm behaviour |
|---|---|
| `cache.update(k, v, i)` | remembers `k_new, v_new`; returns `cat([k_past, k], -2)` (past gathered from pages, padded to `Lmax = max(lens)`) |

Keys are stored **post-RoPE** at their absolute positions; nothing is
re-rotated. `gather` materialises the batch's resident tokens out of their
pages with one advanced-index op per layer (`pool[j][page_idx, :, off_idx]`,
positions beyond a row's length pointing at page 0); `scatter` writes the new
tokens back with one fused write per layer for decode. Batch rows are laid
out `[past_0 .. past_{Lmax-1} | new_0 .. new_{T-1}]`;
a row with fewer resident tokens has its unused past columns masked. Because
attention is permutation-invariant over keys, this padded layout is exact and
the boolean mask (`True` = attend) is the whole contract:

```
allowed(b, i, j) = j < len_b  or  Lmax <= j <= Lmax + i
```

Mask-free (SDPA fast path) cases: a single fresh prefill row (pure causal),
and a decode batch whose rows all share one length.

## Positions

`positions[b, t] = len_b + t` (absolute). Passed as 2-D `[B, T]` to
`Qwen3_5TextRotaryEmbedding`, which expands them to the 3 mRoPE axes itself
(text-only: identical on all axes, interleaved sections `[11, 11, 10]`,
partial rotary factor 0.25).

## Scheduler split

Unchanged from M1: requests scheduled `> 1` token run as per-request prefill
chunks (`prefill_chunk`, keep it a multiple of the GDN scan chunk 64);
requests scheduled exactly 1 token — decodes and 1-token crumbs — run as one
batched decode step. Both call `bank.gather -> decoder.forward -> bank.scatter`.

## Exact admission with paged guests

A request is admissible iff every slot family has a free slot **and** the
pool has `ceil((num_tokens + max_new_tokens) / page_tokens)` free pages. The
scheduler reserves those pages for the request's whole lifetime at
admission (FCFS; a head-of-line request that does not fit blocks the queue
until pages return), so a request can never outgrow its reservation
mid-flight; `GuestCapacityExceeded` from the bank is a bug, not a runtime
path. A request larger than the whole pool is rejected at intake
(`add_request`, `submit_from_handle`) rather than left waiting forever.

## Durable-state protocol

`Qwen35StateBank` implements the store protocol
(`fingerprint_key`, `export_slot`, `import_slot`, `memory_families`):

- snapshots gather the guest tokens out of their pages, trimmed to
  `guest_len` (O(tokens)); the format is page-size independent and the
  fingerprint excludes `page_tokens`;
- `import_slot` zeroes the recurrent families first, then copies the
  families present into the pages the caller reserved — a record holding
  only `gdn_state` is valid and means "tuned initial state, no tokens";
- `decay` scales `gdn_state` only; `merge` averages families of equal shape
  and keeps the receiver's guest window otherwise.

`Qwen35HybridLayout.load_state_adapter(path)` reads an RNN-StateTuning
adapter (`layers.{i}.recurrent_state`, fp32 `[32, 128, 128]` per linear
layer) into that form, verifies layer ids/shapes, rejects per-layer-embedding
tensors (model weights, not state) and records the file SHA-256 in the
handle's provenance (`rule="import"`).

## HF behaviour still relied on

- tokenizer / chat template outside the engine;
- `Qwen3_5DecoderLayer` math: projections, RoPE application, gated RMSNorm,
  MLP, `lm_head`, and the pure-torch GDN / causal-conv kernels (this host has
  no `fla`, `triton`, `causal_conv1d` or `flash_attn`);
- `Qwen3_5TextRotaryEmbedding` for cos/sin;
- config discovery for layer types and head shapes.

It must not rely on `DynamicCache` or any HF mask builder: the wkvm slot is
the durable owner of GDN state, conv window, guest KV, lengths and masks.

## Known limits (see `docs/HYBRID_ENGINE_PLAN.md`, H5)

- Guest K/V are gathered per step (`B x Lmax` tokens x 8 layers copied);
  a gather-free paged attention kernel is the next slice.
- Lifetime page reservation over-reserves for requests that stop early.
- Pure-torch GDN kernels: throughput is not a claim of this slice.
- No CUDA graphs on the hybrid path yet.
