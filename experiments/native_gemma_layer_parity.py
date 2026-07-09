#!/usr/bin/env python
"""Parity check for the first native Gemma4 decoder-layer slice."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_smoke import build_parity_prompt, break_mask_for, load_model, resolve_model_path

from wkvm.models.gemma import gemma4_e4b_routed_span_config
from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
from wkvm.runner.gemma_runner import NativeGemmaRoutedCache


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def causal_mask(batch: int, seq_len: int, *, dtype, device):
    import torch

    mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=dtype, device=device)
    blocked = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask.masked_fill(blocked.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)


def native_config_from_model(model):
    return gemma4_e4b_routed_span_config(
        num_hidden_layers=model.config.num_hidden_layers,
        num_kv_shared_layers=getattr(model.config, "num_kv_shared_layers", 0),
        layer_types=tuple(model.config.layer_types),
        num_kv_heads=getattr(model.config, "num_global_key_value_heads", None)
        or getattr(model.config, "num_key_value_heads", 2),
        head_dim=getattr(model.config, "global_head_dim", None)
        or getattr(model.config, "head_dim", 512),
        sliding_window=getattr(model.config, "sliding_window", 512),
    )


def layer_inputs(text_model, input_ids):
    inputs_embeds = text_model.embed_tokens(input_ids)
    per_layer_inputs = None
    if text_model.hidden_size_per_layer_input:
        token_ple = text_model.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = text_model.project_per_layer_inputs(inputs_embeds, token_ple)
    return inputs_embeds, per_layer_inputs


def pre_layer_hidden(
    text_model,
    hidden,
    per_layer_inputs,
    position_ids,
    causal_mask_mapping,
    *,
    stop_layer: int,
):
    shared_kv_states = {}
    for i, decoder_layer in enumerate(text_model.layers[:stop_layer]):
        layer_type = text_model.config.layer_types[i]
        per_layer_input = None if per_layer_inputs is None else per_layer_inputs[:, :, i, :]
        hidden = decoder_layer(
            hidden,
            per_layer_input,
            shared_kv_states=shared_kv_states,
            position_embeddings=text_model.rotary_emb(hidden, position_ids, layer_type),
            attention_mask=causal_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=None,
        )
    return hidden


def run(args) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    model = load_model(path, args.device, args.attn)
    tok = AutoTokenizer.from_pretrained(path)
    text_model = model.model
    if args.ctx > int(model.config.sliding_window):
        raise ValueError("layer-0 parity ctx must fit in the sliding window")
    layer_idx = int(args.layer)
    if layer_idx < 0 or layer_idx >= model.config.num_hidden_layers:
        raise ValueError("layer must be inside the Gemma text stack")
    first_kv_shared_layer_idx = model.config.num_hidden_layers - getattr(
        model.config,
        "num_kv_shared_layers",
        0,
    )
    if layer_idx >= first_kv_shared_layer_idx:
        raise ValueError("native layer parity does not support shared-KV tail layers yet")

    prompt = build_parity_prompt(tok, args.ctx)
    input_ids = torch.tensor(prompt, dtype=torch.long, device=model.device).unsqueeze(0)
    decode_id = input_ids[:, -1:]
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=model.device).unsqueeze(0)
    decode_position_ids = torch.tensor([[input_ids.shape[1]]], dtype=torch.long, device=model.device)

    layer = text_model.layers[layer_idx]
    native_layer = NativeGemma4TextDecoderLayer(
        layer,
        native_projection_backend=args.native_gemma_projection_backend,
        native_weight_backend=args.native_gemma_weight_backend,
    )
    native_cfg = native_config_from_model(model)
    hf_cache = NativeGemmaRoutedCache(model.config, native_cfg)
    native_cache = NativeGemmaRoutedCache(model.config, native_cfg)
    breaks = break_mask_for(tok, prompt)
    hf_cache.set_span_break_mask(breaks)
    native_cache.set_span_break_mask(breaks)

    with torch.inference_mode():
        hidden, ple = layer_inputs(text_model, input_ids)
        mask = causal_mask(1, input_ids.shape[1], dtype=hidden.dtype, device=hidden.device)
        causal_mask_mapping = {
            "sliding_attention": mask,
            "full_attention": mask,
        }
        hidden = pre_layer_hidden(
            text_model,
            hidden,
            ple,
            position_ids,
            causal_mask_mapping,
            stop_layer=layer_idx,
        )
        ple_layer = None if ple is None else ple[:, :, layer_idx, :]
        layer_type = model.config.layer_types[layer_idx]
        pos_emb = text_model.rotary_emb(hidden, position_ids, layer_type)
        mask = causal_mask(1, input_ids.shape[1], dtype=hidden.dtype, device=hidden.device)

        hf_prefill = layer(
            hidden,
            ple_layer,
            shared_kv_states={},
            position_embeddings=pos_emb,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=hf_cache,
        )
        native_prefill = native_layer(
            hidden,
            ple_layer,
            shared_kv_states={},
            position_embeddings=pos_emb,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=native_cache,
        )

        decode_hidden, decode_ple = layer_inputs(text_model, decode_id)
        decode_ple_layer = None if decode_ple is None else decode_ple[:, :, layer_idx, :]
        decode_hidden = pre_layer_hidden(
            text_model,
            decode_hidden,
            decode_ple,
            decode_position_ids,
            {"sliding_attention": None, "full_attention": None},
            stop_layer=layer_idx,
        )
        decode_pos_emb = text_model.rotary_emb(
            decode_hidden,
            decode_position_ids,
            layer_type,
        )
        hf_decode = layer(
            decode_hidden,
            decode_ple_layer,
            shared_kv_states={},
            position_embeddings=decode_pos_emb,
            attention_mask=None,
            position_ids=decode_position_ids,
            past_key_values=hf_cache,
        )
        native_decode = native_layer(
            decode_hidden,
            decode_ple_layer,
            shared_kv_states={},
            position_embeddings=decode_pos_emb,
            attention_mask=None,
            position_ids=decode_position_ids,
            past_key_values=native_cache,
        )

    prefill_max = float((hf_prefill - native_prefill).abs().max().item())
    decode_max = float((hf_decode - native_decode).abs().max().item())
    hf_layer_cache = hf_cache.layers[layer_idx]
    native_layer_cache = native_cache.layers[layer_idx]
    if hf_layer_cache.keys is None or native_layer_cache.keys is None:
        raise AssertionError(f"layer {layer_idx} did not populate native cache")
    key_max = float((hf_layer_cache.keys - native_layer_cache.keys).abs().max().item())
    ok = (
        prefill_max <= args.atol
        and decode_max <= args.atol
        and key_max <= args.atol
        and hf_layer_cache.keys.shape == native_layer_cache.keys.shape
    )
    payload = {
        "schema": "wkvm.native_gemma_layer_parity.v1",
        "model_path": path,
        "device": str(model.device),
        "attn": args.attn,
        "ctx": len(prompt),
        "layer": layer_idx,
        "layer_type": layer_type,
        "native_gemma_projection_backend": args.native_gemma_projection_backend,
        "native_gemma_weight_backend": args.native_gemma_weight_backend,
        "atol": args.atol,
        "prefill_max_abs_diff": prefill_max,
        "decode_max_abs_diff": decode_max,
        "cache_key_max_abs_diff": key_max,
        "cache_shape": list(hf_layer_cache.keys.shape),
        "ok": ok,
    }
    if args.json:
        atomic_write_json(Path(args.json), payload)
    print("NATIVE_GEMMA_LAYER_PARITY_OK" if ok else "NATIVE_GEMMA_LAYER_PARITY_FAIL")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="eager")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument(
        "--native-gemma-projection-backend",
        choices=["separate", "qkv_packed", "gate_up_packed", "qkv_gate_up_packed"],
        default="separate",
    )
    ap.add_argument(
        "--native-gemma-weight-backend",
        choices=["hf_live", "owned", "owned_cpu"],
        default="hf_live",
    )
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
