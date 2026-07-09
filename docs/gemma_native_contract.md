# Gemma Native Execution Contract

This note is the transition contract from `experiments/gemma_recurrent_poc.py`
to the native wkvm Gemma path. It records the behavior the first
`GemmaRoutedSpanRunner` must preserve while moving cache/state ownership out of
HF `DynamicCache` subclasses and into wkvm slot-owned buffers.

## Model Scope

- Target model: Gemma-4-E4B-it text tower loaded as `Gemma4ForCausalLM`.
- First native milestone is single GPU, bf16, greedy decode.
- The native path is routed-span recurrent mode, not full-KV exact Gemma
  semantics.
- The runner may reuse HF module math during the transition, but the hot-path
  state object is wkvm-owned state, not patched HF `DynamicCache` replacement
  classes.

## Layer Types And KV-owning Layers

Gemma-4 uses a mixed attention stack. `cfg.layer_types` identifies
`full_attention` and `sliding_attention` decoder layers. The E4B text tower has
42 decoder layers, with the last 18 sharing KV. Only the first 24 decoder layers
own cache entries. Shared tail layers consume the cache published by earlier
layers of the same attention type.

For Gemma-4-E4B-it as observed in the PoC, the growing full-attention
KV-owning layers are `[5, 11, 17, 23]`, selected with:

```python
n_owned = cfg.num_hidden_layers - cfg.num_kv_shared_layers
full_kv_layers = [
    i for i in range(n_owned) if cfg.layer_types[i] == "full_attention"
]
```

The recurrent routed-span cache is applied only to those full-attention
KV-owning layers. Sliding-attention owners keep bounded local KV behavior. The
native cache must still allocate request slots for every KV-owning layer family
that participates in decode, including sliding/ring state, routed span-bank
state, pending-span buffers, valid masks, and per-request position counters.

## Sliding And Full Attention Behavior

Sliding layers are bounded local cache layers. They keep the recent window that
the model can legally attend to and do not need the routed span bank.

Full-attention KV-owning layers are converted to routed-span recurrent memory:

- `sink16`: first 16 tokens remain stable for the life of the request.
- `ring1024`: the most recent 1024 tokens are kept exactly.
- `routed-span m64`: evicted middle tokens accumulate in pending storage, split
  into sentence/punctuation spans, then route to 64 durable content slots.
- Each routed slot stores a mean KV summary plus retained exact span
  representatives under a token budget.

The native object identity is the state slot, not a token prefix. Prefix tokens
construct and mutate the slot state; they are not the reusable cache key.

## Routed-span Routing Rules

The current PoC's `RoutedSpanLayer` is the reference behavior for the first
native cache implementation:

- Sentence punctuation defines preferred span boundaries; long runs are capped
  by `max_span`, and a fixed fallback span is used when no punctuation exists.
- Span routing is value-based. In particular, routing must use value vectors,
  not RoPE-positioned key vectors, because the key path can overfit position or
  template structure.
- A span is routed atomically so a fact's name/code/object tokens stay in one
  slot.
- Per-slot retention uses farthest-point selection with a near-duplicate floor
  so repeated filler cannot consume all retained span budget.
- Pending spans stay exactly visible until they are routed.

The recall smoke for this mode must continue to include the known facts
`BLUE-742`, `Samarkand`, and `lantern`.

## Position And Mask Requirements

For prefill, position ids follow the ordinary prompt positions. During chunked
prefill, the query positions advance monotonically across chunks. During decode,
each request's next token position is its current cumulative sequence length.

Routed-span entries are stored as already-positioned KV. The native runner must
not re-rotate retained keys after eviction. Materialized KV order is a readout
layout: sink, routed summaries/representatives, pending, ring, then current
token. For causal q=1 decode without padding, every stored entry is in the
past, so mask-free full attention is valid.

Distinct-cache batching requires padded valid masks when rows have different
materialized routed-span layouts. Pad slots must be invisible to attention and
must write only into dummy/safe rows on graph-safe paths. Mask metadata is part
of the wkvm state contract, not an HF cache-layer side effect.

## HF Behavior Still Relied On During Transition

Until `GemmaRoutedSpanRunner` owns the full attention call directly, the native
path may still rely on HF for:

- tokenizer and chat-template handling outside the GPU busy loop;
- Gemma module math, projection weights, RoPE application, layernorms, MLPs,
  and lm head;
- config discovery for `layer_types`, `num_kv_shared_layers`, sliding window
  size, head shapes, dtype, and vocabulary size.

It must not rely on HF `DynamicCache` replacement classes as the durable
hot-path state owner. The wkvm state slot must own ring, span-bank,
pending-span, valid-mask, and position buffers.
