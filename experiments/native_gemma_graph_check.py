#!/usr/bin/env python
"""Offline CUDA graph parity check for native Gemma routed-span decode.

This is the first N4 slice, deliberately narrower than engine graph dispatch:
one explicit graph bucket, identical-layout rows, fixed-address decode buffers,
and exact eager-static vs graph-static greedy token parity.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_smoke import filler_ids, load_model, resolve_model_path

from wkvm.models.gemma import gemma4_e4b_routed_span_config
from wkvm.runner.gemma_runner import (
    GemmaRoutedSpanRunner,
    NativeGemmaRoutedCache,
    NativeRoutedSpanLayer,
    NativeSlidingWindowLayer,
)
from wkvm.runner.gemma_state import GemmaRoutedStateBank


class StaticSlidingDecodeLayer:
    is_sliding = True

    def __init__(self, keys, values, cumulative_length: int, sliding_window: int):
        import torch

        bsz, heads, stored, dim = keys.shape
        self.cumulative_length = int(cumulative_length)
        self.sliding_window = int(sliding_window)
        self.cap = self.sliding_window
        if stored != self.sliding_window - 1:
            raise ValueError("static sliding decode requires window-1 stored tokens")
        self.keys = torch.zeros(bsz, heads, self.cap, dim, dtype=keys.dtype, device=keys.device)
        self.values = torch.zeros_like(self.keys)
        self.keys[:, :, :stored].copy_(keys)
        self.values[:, :, :stored].copy_(values)
        self.ptr = torch.tensor([stored], dtype=torch.long, device=self.keys.device)
        self.dtype, self.device = self.keys.dtype, self.keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        import torch

        if key_states.shape[-2] != 1:
            raise NotImplementedError("static sliding graph layer is decode-only")
        self.keys.index_copy_(2, self.ptr, key_states)
        self.values.index_copy_(2, self.ptr, value_states)
        self.ptr.copy_(torch.remainder(self.ptr + 1, self.cap))
        self.cumulative_length += 1
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cap, max(self.cumulative_length - self.cap + 1, 0)

    def get_max_cache_shape(self) -> int:
        return self.cap

    def snapshot(self):
        return (
            self.keys.clone(),
            self.values.clone(),
            self.ptr.clone(),
            self.cumulative_length,
        )

    def restore(self, snap) -> None:
        keys, values, ptr, cumulative_length = snap
        self.keys.copy_(keys)
        self.values.copy_(values)
        self.ptr.copy_(ptr)
        self.cumulative_length = int(cumulative_length)


class StaticRoutedSpanDecodeLayer:
    is_sliding = False

    def __init__(
        self,
        keys,
        values,
        cumulative_length: int,
        *,
        cap: int,
        full_mask,
        pending_tail: int,
        route_chunk: int,
    ):
        import torch

        bsz, heads, stored, dim = keys.shape
        if stored >= cap:
            raise ValueError("static routed-span cap must leave decode headroom")
        self.keys = torch.zeros(bsz, heads, cap, dim, dtype=keys.dtype, device=keys.device)
        self.values = torch.zeros_like(self.keys)
        self.keys[:, :, :stored].copy_(keys)
        self.values[:, :, :stored].copy_(values)
        self.ptr = torch.tensor([stored], dtype=torch.long, device=keys.device)
        self.cumulative_length = int(cumulative_length)
        self.cap = int(cap)
        self.full_mask = full_mask
        self.pending_tail = int(pending_tail)
        self.route_chunk = int(route_chunk)
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        import torch

        if key_states.shape[-2] != 1:
            raise NotImplementedError("static routed-span graph layer is decode-only")
        if self.pending_tail + 1 >= self.route_chunk:
            raise NotImplementedError("static graph check would cross routed-span fold boundary")
        self.keys.index_copy_(2, self.ptr, key_states)
        self.values.index_copy_(2, self.ptr, value_states)
        self.full_mask.index_fill_(3, self.ptr, 0.0)
        self.ptr.copy_(torch.remainder(self.ptr + 1, self.cap))
        self.cumulative_length += 1
        self.pending_tail += 1
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.cap, 0

    def get_max_cache_shape(self) -> int:
        return self.cap

    def snapshot(self):
        return (
            self.keys.clone(),
            self.values.clone(),
            self.ptr.clone(),
            self.cumulative_length,
            self.pending_tail,
        )

    def restore(self, snap) -> None:
        keys, values, ptr, cumulative_length, pending_tail = snap
        self.keys.copy_(keys)
        self.values.copy_(values)
        self.ptr.copy_(ptr)
        self.cumulative_length = int(cumulative_length)
        self.pending_tail = int(pending_tail)


class StaticGemmaGraphCache:
    is_compileable = False

    def __init__(self, hf_config, layers: list[Any], full_mask):
        decoder = hf_config.get_text_config(decoder=True) if hasattr(hf_config, "get_text_config") else hf_config
        self.hf_config = decoder
        self.layers = layers
        self.full_mask = full_mask

    @property
    def is_sliding(self) -> list[bool]:
        return [layer.is_sliding for layer in self.layers]

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            layer_idx = 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.layers):
            return -1
        return self.layers[layer_idx].get_max_cache_shape()

    def attention_mask(self) -> dict[str, Any]:
        return {"full_attention": self.full_mask, "sliding_attention": None}

    def snapshot(self):
        return (
            [layer.snapshot() for layer in self.layers],
            self.full_mask.clone(),
        )

    def restore(self, snap) -> None:
        layer_snaps, full_mask = snap
        for layer, layer_snap in zip(self.layers, layer_snaps):
            layer.restore(layer_snap)
        self.full_mask.copy_(full_mask)


def build_prompt(tok, ctx: int) -> list[int]:
    prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
    header = tok("Static graph parity prompt. ", add_special_tokens=False).input_ids
    budget = ctx - len(prefix) - len(header)
    if budget < 32:
        raise ValueError("ctx too small for graph check")
    return prefix + header + filler_ids(tok, budget)


def repeat_batch(tensor, batch: int):
    if tensor.shape[0] != 1:
        raise ValueError("static graph clone expects a batch-1 native prefill")
    return tensor.expand(batch, *([-1] * (tensor.ndim - 1))).clone().contiguous()


def static_clone(
    native_cache: NativeGemmaRoutedCache,
    *,
    batch: int,
    decode_steps: int,
    warmup_iters: int,
) -> StaticGemmaGraphCache:
    import torch

    routed_layers = [
        layer for layer in native_cache.layers if isinstance(layer, NativeRoutedSpanLayer)
    ]
    if not routed_layers:
        raise ValueError("native cache has no routed-span layers")
    stored_lengths = {layer.keys.shape[2] for layer in routed_layers}
    pending_tails = {layer._pend_k.shape[2] for layer in routed_layers}
    route_chunks = {layer.route_chunk for layer in routed_layers}
    if len(stored_lengths) != 1 or len(pending_tails) != 1 or len(route_chunks) != 1:
        raise ValueError("graph check supports identical routed-span layer layouts only")
    pending_tail = pending_tails.pop()
    route_chunk = route_chunks.pop()
    if pending_tail + decode_steps + warmup_iters + 2 >= route_chunk:
        raise ValueError(
            f"decode would cross routed-span fold boundary: pending={pending_tail}, "
            f"out={decode_steps}, warmup={warmup_iters}, route_chunk={route_chunk}"
        )

    stored = stored_lengths.pop()
    cap = stored + decode_steps + warmup_iters + 4
    dtype = routed_layers[0].keys.dtype
    device = routed_layers[0].keys.device
    full_mask = torch.full(
        (batch, 1, 1, cap),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    full_mask[:, :, :, :stored] = 0.0

    layers: list[Any] = []
    for layer in native_cache.layers:
        if isinstance(layer, NativeSlidingWindowLayer):
            keys = repeat_batch(layer.keys, batch)
            values = repeat_batch(layer.values, batch)
            layers.append(
                StaticSlidingDecodeLayer(
                    keys,
                    values,
                    layer.cumulative_length,
                    layer.sliding_window,
                )
            )
        elif isinstance(layer, NativeRoutedSpanLayer):
            layers.append(
                StaticRoutedSpanDecodeLayer(
                    repeat_batch(layer.keys, batch),
                    repeat_batch(layer.values, batch),
                    layer.cumulative_length,
                    cap=cap,
                    full_mask=full_mask,
                    pending_tail=layer._pend_k.shape[2],
                    route_chunk=layer.route_chunk,
                )
            )
        else:
            raise TypeError(f"unsupported native layer type: {type(layer).__name__}")
    return StaticGemmaGraphCache(native_cache.hf_config, layers, full_mask)


def static_eager_decode(model, cache: StaticGemmaGraphCache, first_token, steps: int):
    import torch

    ids = first_token.to(model.device).clone()
    pos = torch.full((1, 1), cache.get_seq_length(), dtype=torch.long, device=model.device)
    tokens = []
    with torch.inference_mode():
        for _ in range(steps):
            out = model(
                input_ids=ids,
                position_ids=pos,
                attention_mask=cache.attention_mask(),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(ids)
            pos = pos + 1
    return torch.cat(tokens, dim=1)


class GraphedGemmaStep:
    def __init__(
        self,
        model,
        cache: StaticGemmaGraphCache,
        batch_size: int,
        *,
        warmup_iters: int = 3,
    ) -> None:
        import torch

        self.model = model
        self.cache = cache
        self.batch_size = batch_size
        self.ids = torch.zeros(batch_size, 1, dtype=torch.long, device=model.device)
        self.pos = torch.full((1, 1), cache.get_seq_length(), dtype=torch.long, device=model.device)
        self._pos0 = int(cache.get_seq_length())

        with torch.inference_mode():
            snap = cache.snapshot()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup_iters):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            cache.restore(snap)
            self.ids.zero_()
            self.pos.fill_(self._pos0)

            snap = cache.snapshot()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()
            torch.cuda.synchronize()
            cache.restore(snap)
            self.ids.zero_()
            self.pos.fill_(self._pos0)

    def _step(self) -> None:
        out = self.model(
            input_ids=self.ids,
            position_ids=self.pos,
            attention_mask=self.cache.attention_mask(),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        self.ids.copy_(nxt)
        self.pos.add_(1)

    def decode(self, first_token, steps: int):
        import torch

        tokens = torch.empty(self.batch_size, steps, dtype=torch.long, device=self.ids.device)
        self.ids.copy_(first_token.to(self.ids.device))
        self.pos.fill_(self._pos0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(steps):
            self.graph.replay()
            tokens[:, i].copy_(self.ids[:, 0])
        torch.cuda.synchronize()
        return tokens, time.perf_counter() - start


def run(args) -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for native Gemma graph check")

    path = resolve_model_path(args.model_path)
    model = load_model(path, args.device, args.attn)
    tok = AutoTokenizer.from_pretrained(path)
    cfg = gemma4_e4b_routed_span_config(
        num_hidden_layers=model.config.num_hidden_layers,
        num_kv_shared_layers=getattr(model.config, "num_kv_shared_layers", 0),
        layer_types=tuple(model.config.layer_types),
        num_kv_heads=getattr(model.config, "num_global_key_value_heads", None)
        or getattr(model.config, "num_key_value_heads", 2),
        head_dim=getattr(model.config, "global_head_dim", None)
        or getattr(model.config, "head_dim", 512),
        sink_tokens=args.sink,
        ring_tokens=args.window,
        routed_slots=args.m_slots,
        pending_tokens=args.route_chunk,
        sliding_window=getattr(model.config, "sliding_window", 1024),
    )
    bank = GemmaRoutedStateBank(cfg, num_slots=1)
    runner = GemmaRoutedSpanRunner(model, bank, prefill_chunk=args.chunk)
    slots = {"gemma_routed_span": 1, "gemma_sliding_kv": 1, "gemma_routed_meta": 1}
    bank.zero_slots(slots)
    prompt = build_prompt(tok, args.ctx)
    logits, native_cache = runner.prefill(prompt, slots, break_mask=[False] * len(prompt))
    if not isinstance(native_cache, NativeGemmaRoutedCache):
        raise AssertionError("native prefill did not return a native cache")
    first = logits.argmax().reshape(1, 1).repeat(args.batch, 1)

    eager_cache = static_clone(
        native_cache,
        batch=args.batch,
        decode_steps=args.out,
        warmup_iters=args.warmup_iters,
    )
    graph_cache = static_clone(
        native_cache,
        batch=args.batch,
        decode_steps=args.out,
        warmup_iters=args.warmup_iters,
    )

    eager_tokens = static_eager_decode(model, eager_cache, first, args.out)
    graph_step = GraphedGemmaStep(
        model,
        graph_cache,
        args.batch,
        warmup_iters=args.warmup_iters,
    )
    graph_tokens, graph_elapsed = graph_step.decode(first, args.out)
    if not torch.equal(eager_tokens, graph_tokens):
        diffs = (eager_tokens != graph_tokens).nonzero(as_tuple=False)
        raise AssertionError(
            f"graph token parity failed at {diffs[:8].tolist()} "
            f"eager={eager_tokens.cpu().tolist()} graph={graph_tokens.cpu().tolist()}"
        )

    print("GRAPH_TOKEN_PARITY_OK")
    print(f"graph_bucket=gemma-routed-span:b{args.batch}")
    print(f"ctx={len(prompt)}")
    print(f"out={args.out}")
    print(f"batch={args.batch}")
    print(f"graph_decode_s={graph_elapsed:.4f}")
    print(f"tokens={graph_tokens[0].detach().cpu().tolist()}")
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="eager")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=512)
    ap.add_argument("--warmup-iters", type=int, default=3)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
