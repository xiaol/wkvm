#!/usr/bin/env python
"""Offline smoke for the native Gemma routed-span runner boundary."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from wkvm.models.gemma import gemma4_e4b_routed_span_config
from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner, NativeGemmaRoutedCache
from wkvm.runner.gemma_state import GemmaRoutedStateBank

MODEL_CANDIDATES = [
    "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it",
    "google/gemma-4-E4B-it",
]

FILLER = (
    "The old lighthouse keeper walked along the shore every morning, checking "
    "the tide tables and noting the weather in his worn leather journal. "
)
RECALL_PROMPT = (
    "Remember these facts exactly. The secret code is BLUE-742. "
    "The archive city is Samarkand. The object on the brass table is a lantern. "
)
RECALL_QUESTION = (
    "\n\nAnswer with the three remembered facts: secret code, city, and object."
)
PARITY_PROMPT = (
    "A small deterministic cache check: Mira labels the red box alpha, "
    "the blue box beta, and the green box gamma. The next label is"
)


def resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in MODEL_CANDIDATES:
        if os.path.isdir(candidate):
            return candidate
    return MODEL_CANDIDATES[-1]


def break_mask_for(tok, ids: list[int]) -> list[bool]:
    return [any(c in tok.decode([tid]) for c in ".!?\n") for tid in ids]


def filler_ids(tok, n: int) -> list[int]:
    unit = tok(FILLER, add_special_tokens=False).input_ids
    reps = (n + len(unit) - 1) // len(unit)
    return (unit * reps)[:n]


def build_prompt(tok, ctx: int) -> list[int]:
    prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
    recall = tok(RECALL_PROMPT, add_special_tokens=False).input_ids
    question = tok(RECALL_QUESTION, add_special_tokens=False).input_ids
    budget = ctx - len(prefix) - len(recall) - len(question)
    if budget < 16:
        raise ValueError(f"ctx={ctx} too small for recall smoke")
    pre = min(64, budget // 3)
    return prefix + filler_ids(tok, pre) + recall + filler_ids(tok, budget - pre) + question


def build_parity_prompt(tok, ctx: int) -> list[int]:
    prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
    fixed = tok(PARITY_PROMPT, add_special_tokens=False).input_ids
    budget = max(0, ctx - len(prefix) - len(fixed))
    return prefix + filler_ids(tok, budget) + fixed


def load_model(
    path: str,
    device: str,
    attn: str,
    *,
    native_checkpoint_loader: bool = False,
    native_gemma_attention_backend: str = "manual",
    native_gemma_projection_backend: str = "separate",
):
    import torch

    if native_checkpoint_loader:
        from wkvm.runner.gemma_native_forward import load_native_gemma4_from_checkpoint

        return load_native_gemma4_from_checkpoint(
            path,
            device=device,
            dtype=torch.bfloat16,
            native_attention_backend=native_gemma_attention_backend,
            native_projection_backend=native_gemma_projection_backend,
        )

    from transformers import AutoConfig
    from transformers.models.gemma4 import Gemma4ForCausalLM

    full_cfg = AutoConfig.from_pretrained(path)
    text_cfg = full_cfg.get_text_config(decoder=True)
    model = Gemma4ForCausalLM.from_pretrained(
        path,
        config=text_cfg,
        dtype=torch.bfloat16,
        attn_implementation=attn,
        key_mapping={r"^model\.language_model": "model"},
        device_map=device,
    )
    model.eval()
    return model


def hf_greedy_tokens(model, prompt: list[int], max_new_tokens: int, device: str) -> list[int]:
    import torch

    ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    out_tokens: list[int] = []
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        tok = int(out.logits[0, -1].float().argmax().item())
        out_tokens.append(tok)
        for _ in range(max_new_tokens - 1):
            out = model(
                input_ids=torch.tensor([[tok]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                attention_mask={"full_attention": None, "sliding_attention": None},
            )
            cache = out.past_key_values
            tok = int(out.logits[0, -1].float().argmax().item())
            out_tokens.append(tok)
    return out_tokens


def native_greedy_tokens(
    runner: GemmaRoutedSpanRunner,
    bank: GemmaRoutedStateBank,
    tok,
    prompt: list[int],
    slots: dict[str, int],
    max_new_tokens: int,
) -> list[int]:
    bank.zero_slots(slots)
    logits, cache = runner.prefill(prompt, slots, break_mask=break_mask_for(tok, prompt))
    out_tokens: list[int] = [int(logits.argmax().item())]
    for _ in range(max_new_tokens - 1):
        pos = len(prompt) + len(out_tokens) - 1
        logits = runner.decode_step(cache, [out_tokens[-1]], position_ids=[pos])
        out_tokens.append(int(logits[0].argmax().item()))
    return out_tokens


def run_short_parity(
    model,
    tok,
    runner: GemmaRoutedSpanRunner,
    bank: GemmaRoutedStateBank,
    args,
) -> None:
    import torch

    if args.parity_ctx > args.sink + args.window:
        raise ValueError("parity_ctx must not trigger routed-span eviction")
    slots = {"gemma_routed_span": 1, "gemma_sliding_kv": 1, "gemma_routed_meta": 1}
    prompt = build_parity_prompt(tok, args.parity_ctx)
    hf_tokens = hf_greedy_tokens(model, prompt, args.parity_out, args.device)
    native_tokens = native_greedy_tokens(runner, bank, tok, prompt, slots, args.parity_out)
    if native_tokens != hf_tokens:
        raise AssertionError(
            "short-context parity failed: "
            f"native={native_tokens} hf={hf_tokens}"
        )
    print("NATIVE_GEMMA_PARITY_OK")
    print(f"parity_tokens={native_tokens}")
    torch.cuda.empty_cache()


def run_metadata_only() -> None:
    cfg = gemma4_e4b_routed_span_config()
    bank = GemmaRoutedStateBank(cfg, num_slots=2)
    slots = {"gemma_routed_span": 1}
    bank.ingest_positions(slots, list(range(2048)), [i % 23 == 0 for i in range(2048)])
    state = bank.slot_state(slots)
    assert cfg.full_kv_layers == (5, 11, 17, 23)
    assert all(len(layer.ring_positions) <= cfg.ring_tokens for layer in state.full_layers.values())
    print("NATIVE_GEMMA_METADATA_OK")
    print("NATIVE_GEMMA_SMOKE_OK")


def run_model(args) -> None:
    import torch
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    if args.native_gemma_checkpoint_loader:
        args.use_native_gemma_forward = True
        if args.native_gemma_weight_backend != "hf_live":
            raise ValueError(
                "--native-gemma-checkpoint-loader owns checkpoint tensors directly "
                "and requires --native-gemma-weight-backend hf_live"
            )
    model = load_model(
        path,
        args.device,
        args.attn,
        native_checkpoint_loader=args.native_gemma_checkpoint_loader,
        native_gemma_attention_backend=args.native_gemma_attention_backend,
        native_gemma_projection_backend=args.native_gemma_projection_backend,
    )
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
    bank = GemmaRoutedStateBank(cfg, num_slots=max(args.batch, 1))
    runner = GemmaRoutedSpanRunner(
        model,
        bank,
        prefill_chunk=args.chunk,
        use_native_gemma_forward=args.use_native_gemma_forward,
        native_gemma_attention_backend=args.native_gemma_attention_backend,
        native_gemma_projection_backend=args.native_gemma_projection_backend,
        native_gemma_weight_backend=args.native_gemma_weight_backend,
    )

    if not args.skip_parity:
        run_short_parity(model, tok, runner, bank, args)

    slots = {"gemma_routed_span": 1, "gemma_sliding_kv": 1, "gemma_routed_meta": 1}
    bank.zero_slots(slots)
    prompt = build_prompt(tok, args.ctx)
    breaks = break_mask_for(tok, prompt)
    logits, cache = runner.prefill(prompt, slots, break_mask=breaks)
    if isinstance(cache, NativeGemmaRoutedCache) is False:
        raise AssertionError("runner did not return native Gemma cache")
    if "DynamicCache" in type(cache).__name__:
        raise AssertionError("HF DynamicCache is not allowed in native smoke")
    tokens: list[int] = [int(logits.argmax().item())]
    for _ in range(args.out - 1):
        logits = runner.decode_step(cache, [tokens[-1]])
        tokens.append(int(logits[0].argmax().item()))
    text = tok.decode(tokens)

    state = bank.slot_state(slots)
    assert all(len(layer.ring_positions) <= cfg.ring_tokens for layer in state.full_layers.values())
    print("NATIVE_GEMMA_SMOKE_OK")
    print(
        "model_forward_backend="
        f"{getattr(runner.model, 'wkvm_forward_backend', 'hf_transformers_gemma4_forward')}"
    )
    print(
        "uses_hf_transformer_forward="
        f"{not bool(getattr(runner.model, 'wkvm_no_hf_transformer_forward', False))}"
    )
    print(
        "uses_hf_model_construction="
        f"{bool(getattr(runner.model, 'wkvm_uses_hf_model_construction', True))}"
    )
    print(
        "native_gemma_checkpoint_loader="
        f"{bool(getattr(runner.model, 'wkvm_checkpoint_native_loader', False))}"
    )
    print(
        "native_gemma_projection_backend="
        f"{getattr(runner.model, 'native_projection_backend', args.native_gemma_projection_backend)}"
    )
    print(
        "native_gemma_weight_backend="
        f"{getattr(runner.model, 'native_weight_backend', args.native_gemma_weight_backend)}"
    )
    print(f"native_cache_bytes={cache.state_bytes()}")
    print(f"native_resident_tokens={state.resident_tokens}")
    print(f"contains_BLUE-742={'BLUE-742' in text}")
    print(f"contains_Samarkand={'Samarkand' in text}")
    print(f"contains_lantern={'lantern' in text.lower()}")
    if args.require_recall and not all(x in text for x in ("BLUE-742", "Samarkand")):
        raise SystemExit("recall smoke failed")
    if args.require_recall and "lantern" not in text.lower():
        raise SystemExit("recall smoke failed")
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--out", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=256)
    ap.add_argument("--parity-ctx", type=int, default=128)
    ap.add_argument("--parity-out", type=int, default=8)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--metadata-only", action="store_true")
    ap.add_argument("--require-recall", action="store_true")
    ap.add_argument(
        "--use-native-gemma-forward",
        action="store_true",
        help=(
            "Run runner model calls through wkvm's native Gemma forward bridge "
            "instead of transformers.Gemma4ForCausalLM.forward."
        ),
    )
    ap.add_argument(
        "--native-gemma-checkpoint-loader",
        action="store_true",
        help=(
            "Load Gemma4 text tensors directly from safetensors into wkvm's native "
            "forward bridge instead of constructing transformers.Gemma4ForCausalLM. "
            "Still uses Transformers for config/tokenizer metadata."
        ),
    )
    ap.add_argument(
        "--native-gemma-attention-backend",
        choices=["manual", "manual_gqa", "sdpa", "sdpa_single_gqa", "triton_dense_gqa"],
        default="manual",
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
    args = ap.parse_args()
    if args.metadata_only:
        run_metadata_only()
        return
    run_model(args)


if __name__ == "__main__":
    main()
