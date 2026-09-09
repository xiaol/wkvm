"""Convert an official RWKV-7 ``.pth`` (BlinkDL layout) into an fla-format
directory that ``wkvm.models.rwkv7.load_rwkv7`` / ``fla.models.rwkv7`` load.

Mapping (verified against ``fla/layers/rwkv7.py`` 0.5.2 and the G1 series
checkpoints; the fla project's own converter copies the same tensors):

    emb.weight                       -> model.embeddings.weight
    blocks.0.ln0.{weight,bias}       -> model.layers.0.pre_norm.*
    blocks.i.ln1 / ln2               -> model.layers.i.attn_norm / ffn_norm
    blocks.i.att.x_{r,w,k,v,a,g}     -> model.layers.i.attn.x_*        (1,1,D)
    blocks.i.att.k_k / k_a           -> attn.k_k / k_a                 (D)
    blocks.i.att.r_k                 -> attn.r_k                       (H, hd)
    blocks.i.att.{receptance,key,value,output}.weight -> attn.{r,k,v,o}_proj.weight
    blocks.i.att.w1 / w2 / w0        -> attn.w_lora.lora.0.weight^T / lora.2.weight^T / lora.2.bias
    blocks.i.att.a1 / a2 / a0        -> attn.a_lora.*                  (same pattern)
    blocks.i.att.v1 / v2 / v0        -> attn.v_lora.*   (i >= 1 only; layer 0 has no v_lora)
    blocks.i.att.g1 / g2             -> attn.g_lora.lora.0.weight^T / lora.2.weight^T (no bias)
    blocks.i.att.ln_x.{weight,bias}  -> attn.g_norm.*
    blocks.i.ffn.x_k                 -> ffn.x_k                        (D)
    blocks.i.ffn.{key,value}.weight  -> ffn.{key,value}.weight
    ln_out.{weight,bias}             -> model.norm.*
    head.weight                      -> lm_head.weight

Decay convention: fla's layer feeds ``-0.6065 * sigmoid(w_lora(x))`` to a
kernel that applies ``exp(w)``; the official kernel applies ``exp(-exp(w))``
to ``-softplus(-x) - 0.5``. Both equal ``exp(-0.6065 * sigmoid(x))``, so
``w0/w1/w2`` copy unchanged.

Usage:
  python scripts/convert_rwkv7_pth.py <in.pth> <out_dir> [--dtype bfloat16] \
      [--vocab rwkv_vocab_v20230424.txt]

With ``--vocab`` the World vocabulary and the slow HF tokenizer shim in
``scripts/rwkv_world_tokenizer/`` are copied next to the weights, so
``AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)`` works and
the M1–M3 GPU tests run with ``WKVM_RWKV7_PATH=<out_dir>``.

Verified 2026-09-09 on the G1 0.4B checkpoint: greedy continuations through
``wkvm`` (fla kernels, arena slots, chunked prefill) equal the official
``rwkv`` package (``RWKV(model=..., strategy="cuda bf16")``) on 4 prompts x
24 tokens; last-logit max |diff| 0.3–0.75 at logit scale 10–22 (bf16, two
kernel implementations).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def convert(state: dict[str, torch.Tensor], dtype: torch.dtype) -> tuple[dict[str, torch.Tensor], dict]:
    n_layer = max(int(k.split(".")[1]) for k in state if k.startswith("blocks.")) + 1
    hidden = state["emb.weight"].shape[1]
    vocab = state["emb.weight"].shape[0]
    head_dim = state["blocks.0.att.r_k"].shape[1]
    out: dict[str, torch.Tensor] = {}

    def put(dst: str, src: str | torch.Tensor, transpose: bool = False, reshape=None) -> None:
        t = state[src] if isinstance(src, str) else src
        if transpose:
            t = t.t()
        if reshape is not None:
            t = t.reshape(*reshape)
        out[dst] = t.to(dtype).contiguous()

    put("model.embeddings.weight", "emb.weight")
    put("model.norm.weight", "ln_out.weight")
    put("model.norm.bias", "ln_out.bias")
    put("lm_head.weight", "head.weight")
    put("model.layers.0.pre_norm.weight", "blocks.0.ln0.weight")
    put("model.layers.0.pre_norm.bias", "blocks.0.ln0.bias")
    for i in range(n_layer):
        b, L = f"blocks.{i}", f"model.layers.{i}"
        for src, dst in (("ln1", "attn_norm"), ("ln2", "ffn_norm")):
            put(f"{L}.{dst}.weight", f"{b}.{src}.weight")
            put(f"{L}.{dst}.bias", f"{b}.{src}.bias")
        a = f"{b}.att"
        A = f"{L}.attn"
        for x in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            put(f"{A}.{x}", f"{a}.{x}", reshape=(1, 1, hidden))
        put(f"{A}.k_k", f"{a}.k_k", reshape=(hidden,))
        put(f"{A}.k_a", f"{a}.k_a", reshape=(hidden,))
        put(f"{A}.r_k", f"{a}.r_k")
        for src, dst in (("receptance", "r_proj"), ("key", "k_proj"), ("value", "v_proj"), ("output", "o_proj")):
            put(f"{A}.{dst}.weight", f"{a}.{src}.weight")
        for lora in ("w", "a") + (("v",) if i > 0 else ()):
            put(f"{A}.{lora}_lora.lora.0.weight", f"{a}.{lora}1", transpose=True)
            put(f"{A}.{lora}_lora.lora.2.weight", f"{a}.{lora}2", transpose=True)
            put(f"{A}.{lora}_lora.lora.2.bias", f"{a}.{lora}0", reshape=(-1,))
        put(f"{A}.g_lora.lora.0.weight", f"{a}.g1", transpose=True)
        put(f"{A}.g_lora.lora.2.weight", f"{a}.g2", transpose=True)
        put(f"{A}.g_norm.weight", f"{a}.ln_x.weight")
        put(f"{A}.g_norm.bias", f"{a}.ln_x.bias")
        put(f"{L}.ffn.x_k", f"{b}.ffn.x_k", reshape=(hidden,))
        put(f"{L}.ffn.key.weight", f"{b}.ffn.key.weight")
        put(f"{L}.ffn.value.weight", f"{b}.ffn.value.weight")

    used = {
        "emb.weight", "ln_out.weight", "ln_out.bias", "head.weight", "blocks.0.ln0.weight", "blocks.0.ln0.bias",
    }
    for i in range(n_layer):
        b = f"blocks.{i}"
        used.update(f"{b}.{n}" for n in (
            "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias",
            "att.x_r", "att.x_w", "att.x_k", "att.x_v", "att.x_a", "att.x_g", "att.k_k", "att.k_a", "att.r_k",
            "att.receptance.weight", "att.key.weight", "att.value.weight", "att.output.weight",
            "att.w0", "att.w1", "att.w2", "att.a0", "att.a1", "att.a2", "att.g1", "att.g2",
            "att.ln_x.weight", "att.ln_x.bias", "ffn.x_k", "ffn.key.weight", "ffn.value.weight",
        ))
        used.update(f"{b}.att.v{j}" for j in (0, 1, 2))  # layer 0's v-lora is unused by design
    unused = sorted(set(state) - used)
    if unused:
        raise KeyError(f"unmapped tensors in checkpoint: {unused[:10]}")

    config = {
        "model_type": "rwkv7",
        "architectures": ["RWKV7ForCausalLM"],
        "attn_mode": "chunk",
        "hidden_size": hidden,
        "hidden_ratio": 4,
        "intermediate_size": state["blocks.0.ffn.key.weight"].shape[0],
        "num_hidden_layers": n_layer,
        "head_dim": head_dim,
        "num_heads": hidden // head_dim,
        "decay_low_rank_dim": state["blocks.0.att.w1"].shape[1],
        "gate_low_rank_dim": state["blocks.0.att.g1"].shape[1],
        "a_low_rank_dim": state["blocks.0.att.a1"].shape[1],
        "v_low_rank_dim": state["blocks.1.att.v1"].shape[1] if n_layer > 1 else 32,
        "hidden_act": "sqrelu",
        "norm_first": True,
        "norm_bias": True,
        "norm_eps": 1e-5,
        "fuse_norm": False,
        "fuse_cross_entropy": False,
        "vocab_size": vocab,
        "tie_word_embeddings": False,
        "use_cache": True,
        "torch_dtype": str(dtype).removeprefix("torch."),
    }
    return out, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--vocab", default=None, help="rwkv_vocab_v20230424.txt to ship with a tokenizer shim")
    args = ap.parse_args()
    from safetensors.torch import save_file

    state = torch.load(args.src, map_location="cpu", mmap=True, weights_only=True)
    tensors, config = convert(state, getattr(torch, args.dtype))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt", "source": str(args.src)})
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if args.vocab:
        import shutil

        shim = Path(__file__).resolve().parent / "rwkv_world_tokenizer"
        shutil.copyfile(args.vocab, out / "rwkv_vocab_v20230424.txt")
        for name in ("tokenization_rwkv_world.py", "tokenizer_config.json"):
            shutil.copyfile(shim / name, out / name)
    total = sum(t.numel() for t in tensors.values())
    print(f"wrote {out}: {len(tensors)} tensors, {total/1e6:.1f}M params, "
          f"{config['num_hidden_layers']} layers, hidden {config['hidden_size']}, vocab {config['vocab_size']}")


if __name__ == "__main__":
    main()
