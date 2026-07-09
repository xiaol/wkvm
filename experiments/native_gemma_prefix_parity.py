#!/usr/bin/env python
"""Parity check for a native Gemma4 text-layer prefix."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import UserDict
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
from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM, NativeGemma4TextPrefix
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


def hf_prefix_forward(
    text_model,
    input_ids,
    position_ids,
    cache,
    attention_mask_mapping,
    *,
    num_layers: int,
):
    inputs_embeds = text_model.embed_tokens(input_ids)
    per_layer_inputs = None
    if text_model.hidden_size_per_layer_input:
        token_ple = text_model.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = text_model.project_per_layer_inputs(inputs_embeds, token_ple)

    position_embeddings = {
        layer_type: text_model.rotary_emb(inputs_embeds, position_ids, layer_type)
        for layer_type in set(text_model.config.layer_types[:num_layers])
    }
    shared_kv_states = UserDict()
    hidden = inputs_embeds
    for i, decoder_layer in enumerate(text_model.layers[:num_layers]):
        layer_type = text_model.config.layer_types[i]
        per_layer_input = None if per_layer_inputs is None else per_layer_inputs[:, :, i, :]
        hidden = decoder_layer(
            hidden,
            per_layer_input,
            shared_kv_states=shared_kv_states,
            position_embeddings=position_embeddings[layer_type],
            attention_mask=attention_mask_mapping[layer_type],
            position_ids=position_ids,
            past_key_values=cache,
        )
    return hidden


def cache_diffs(hf_cache, native_cache, *, num_layers: int) -> list[dict[str, Any]]:
    diffs = []
    owned_layers = min(num_layers, len(hf_cache.layers), len(native_cache.layers))
    for layer_idx in range(owned_layers):
        hf_layer = hf_cache.layers[layer_idx]
        native_layer = native_cache.layers[layer_idx]
        if hf_layer.keys is None or native_layer.keys is None:
            diffs.append(
                {
                    "layer": layer_idx,
                    "initialized": False,
                    "ok": hf_layer.keys is None and native_layer.keys is None,
                }
            )
            continue
        key_diff = float((hf_layer.keys - native_layer.keys).abs().max().item())
        value_diff = float((hf_layer.values - native_layer.values).abs().max().item())
        diffs.append(
            {
                "layer": layer_idx,
                "initialized": True,
                "layer_type": hf_cache.hf_config.layer_types[layer_idx],
                "key_shape": list(hf_layer.keys.shape),
                "value_shape": list(hf_layer.values.shape),
                "key_max_abs_diff": key_diff,
                "value_max_abs_diff": value_diff,
                "ok": key_diff == 0.0 and value_diff == 0.0,
            }
        )
    return diffs


def _last_logits(logits):
    if logits.ndim == 3:
        return logits[:, -1, :]
    if logits.ndim == 2:
        return logits
    raise ValueError(f"unsupported logits shape for choice metrics: {tuple(logits.shape)}")


def logit_choice_metrics(hf_logits, native_logits, *, topk: int = 10) -> dict[str, Any]:
    hf_last = _last_logits(hf_logits)
    native_last = _last_logits(native_logits)
    if hf_last.shape != native_last.shape:
        raise ValueError(
            "HF/native logits shapes differ: "
            f"{tuple(hf_last.shape)} vs {tuple(native_last.shape)}"
        )
    if hf_last.shape[0] != 1:
        raise ValueError("choice metrics currently expect a single batch row")

    k = min(int(topk), hf_last.shape[-1])
    hf_values, hf_indices = hf_last[0].topk(k)
    native_values, native_indices = native_last[0].topk(k)
    hf_top_token = int(hf_indices[0].item())
    native_top_token = int(native_indices[0].item())
    hf_top2_margin = None
    native_top2_margin = None
    if k >= 2:
        hf_top2_margin = float((hf_values[0] - hf_values[1]).item())
        native_top2_margin = float((native_values[0] - native_values[1]).item())
    hf_topk_set = {int(x.item()) for x in hf_indices}
    native_topk_set = {int(x.item()) for x in native_indices}
    return {
        "topk": k,
        "hf_top_token": hf_top_token,
        "native_top_token": native_top_token,
        "top_token_match": hf_top_token == native_top_token,
        "hf_top_logit": float(hf_values[0].item()),
        "native_top_logit": float(native_values[0].item()),
        "hf_top2_margin": hf_top2_margin,
        "native_top2_margin": native_top2_margin,
        "hf_top_logit_native_delta": float(
            (native_last[0, hf_top_token] - hf_last[0, hf_top_token]).item()
        ),
        "native_top_logit_hf_delta": float(
            (hf_last[0, native_top_token] - native_last[0, native_top_token]).item()
        ),
        "topk_overlap_count": len(hf_topk_set & native_topk_set),
        "topk_overlap_fraction": len(hf_topk_set & native_topk_set) / k,
        "hf_topk_tokens": [int(x.item()) for x in hf_indices],
        "native_topk_tokens": [int(x.item()) for x in native_indices],
    }


def run(args) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    model = load_model(path, args.device, args.attn)
    tok = AutoTokenizer.from_pretrained(path)
    text_model = model.model
    num_layers = int(args.layers)
    if num_layers < 1 or num_layers > model.config.num_hidden_layers:
        raise ValueError("layers must cover at least one layer and fit inside the model")
    if args.ctx > int(model.config.sliding_window):
        raise ValueError("prefix parity ctx must fit in the sliding window for this check")

    prompt = build_parity_prompt(tok, args.ctx)
    input_ids = torch.tensor(prompt, dtype=torch.long, device=model.device).unsqueeze(0)
    decode_id = input_ids[:, -1:]
    position_ids = torch.arange(
        input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    decode_position_ids = torch.tensor(
        [[input_ids.shape[1]]],
        dtype=torch.long,
        device=model.device,
    )

    native_prefix = NativeGemma4TextPrefix(
        text_model,
        num_layers=num_layers,
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
        inputs_embeds = text_model.embed_tokens(input_ids)
        mask = causal_mask(1, input_ids.shape[1], dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        prefill_mask_mapping = {
            "sliding_attention": mask,
            "full_attention": mask,
        }

        hf_prefill = hf_prefix_forward(
            text_model,
            input_ids,
            position_ids,
            hf_cache,
            prefill_mask_mapping,
            num_layers=num_layers,
        )
        native_prefill = native_prefix(
            input_ids=input_ids,
            attention_mask=prefill_mask_mapping,
            position_ids=position_ids,
            past_key_values=native_cache,
        ).hidden_states

        decode_mask_mapping = {
            "sliding_attention": None,
            "full_attention": None,
        }
        hf_decode = hf_prefix_forward(
            text_model,
            decode_id,
            decode_position_ids,
            hf_cache,
            decode_mask_mapping,
            num_layers=num_layers,
        )
        native_decode = native_prefix(
            input_ids=decode_id,
            attention_mask=decode_mask_mapping,
            position_ids=decode_position_ids,
            past_key_values=native_cache,
        ).hidden_states

    prefill_max = float((hf_prefill - native_prefill).abs().max().item())
    decode_max = float((hf_decode - native_decode).abs().max().item())
    diffs = cache_diffs(hf_cache, native_cache, num_layers=num_layers)
    cache_max = max(
        [
            max(
                float(row.get("key_max_abs_diff", 0.0)),
                float(row.get("value_max_abs_diff", 0.0)),
            )
            for row in diffs
        ],
        default=0.0,
    )
    logit_prefill_max = None
    logit_decode_max = None
    logit_prefill_choice = None
    logit_decode_choice = None
    if args.compare_logits:
        if num_layers != model.config.num_hidden_layers:
            raise ValueError("--compare-logits requires --layers to cover the full text stack")
        hf_lm_cache = NativeGemmaRoutedCache(model.config, native_cfg)
        native_lm_cache = NativeGemmaRoutedCache(model.config, native_cfg)
        hf_lm_cache.set_span_break_mask(breaks)
        native_lm_cache.set_span_break_mask(breaks)
        native_lm = NativeGemma4ForCausalLM(
            model,
            native_projection_backend=args.native_gemma_projection_backend,
            native_weight_backend=args.native_gemma_weight_backend,
        )
        with torch.inference_mode():
            hf_lm_prefill = model(
                input_ids=input_ids,
                attention_mask=prefill_mask_mapping,
                position_ids=position_ids,
                past_key_values=hf_lm_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            native_lm_prefill = native_lm(
                input_ids=input_ids,
                attention_mask=prefill_mask_mapping,
                position_ids=position_ids,
                past_key_values=native_lm_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            hf_lm_decode = model(
                input_ids=decode_id,
                attention_mask=decode_mask_mapping,
                position_ids=decode_position_ids,
                past_key_values=hf_lm_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            native_lm_decode = native_lm(
                input_ids=decode_id,
                attention_mask=decode_mask_mapping,
                position_ids=decode_position_ids,
                past_key_values=native_lm_cache,
                use_cache=True,
                logits_to_keep=1,
            )
        logit_prefill_max = float(
            (hf_lm_prefill.logits - native_lm_prefill.logits).abs().max().item()
        )
        logit_decode_max = float(
            (hf_lm_decode.logits - native_lm_decode.logits).abs().max().item()
        )
        logit_prefill_choice = logit_choice_metrics(
            hf_lm_prefill.logits,
            native_lm_prefill.logits,
            topk=args.choice_topk,
        )
        logit_decode_choice = logit_choice_metrics(
            hf_lm_decode.logits,
            native_lm_decode.logits,
            topk=args.choice_topk,
        )

    ok = (
        prefill_max <= args.atol
        and decode_max <= args.atol
        and cache_max <= args.atol
        and (logit_prefill_max is None or logit_prefill_max <= args.atol)
        and (logit_decode_max is None or logit_decode_max <= args.atol)
    )
    payload = {
        "schema": "wkvm.native_gemma_prefix_parity.v1",
        "model_path": path,
        "device": str(model.device),
        "attn": args.attn,
        "ctx": len(prompt),
        "layers": num_layers,
        "layer_types": list(model.config.layer_types[:num_layers]),
        "native_gemma_projection_backend": args.native_gemma_projection_backend,
        "native_gemma_weight_backend": args.native_gemma_weight_backend,
        "atol": args.atol,
        "prefill_max_abs_diff": prefill_max,
        "decode_max_abs_diff": decode_max,
        "cache_max_abs_diff": cache_max,
        "logit_prefill_max_abs_diff": logit_prefill_max,
        "logit_decode_max_abs_diff": logit_decode_max,
        "logit_prefill_choice": logit_prefill_choice,
        "logit_decode_choice": logit_decode_choice,
        "logit_top_token_match": (
            None
            if logit_prefill_choice is None or logit_decode_choice is None
            else bool(
                logit_prefill_choice["top_token_match"]
                and logit_decode_choice["top_token_match"]
            )
        ),
        "cache_diffs": diffs,
        "ok": ok,
    }
    if args.json:
        atomic_write_json(Path(args.json), payload)
    print("NATIVE_GEMMA_PREFIX_PARITY_OK" if ok else "NATIVE_GEMMA_PREFIX_PARITY_FAIL")
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
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--compare-logits", action="store_true")
    ap.add_argument(
        "--choice-topk",
        type=int,
        default=10,
        help="Top-k size for --compare-logits greedy-choice diagnostics.",
    )
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
