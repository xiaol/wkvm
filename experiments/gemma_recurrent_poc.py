#!/usr/bin/env python
"""PoC-1: "recurrent mode" serving for gemma-4-E4B-it on one RTX 4090.

Demonstrates that capping the growing-KV layers (the full-attention layers that
own KV: derived from config, e.g. layers 5/11/17/23 of gemma-4-E4B) with a fixed
sink+ring cache gives FLAT memory and FLAT decode speed vs context length, while
stock full KV grows linearly. A needle-recall probe documents the quality gap the
ring introduces (to be closed by the state bank in PoC-2).

Design notes (verified against transformers 5.9.0 in the HRM-Text venv):
- v5 caches are per-layer objects. DynamicCache(config=...) builds one layer per
  NON-shared decoder layer (num_kv_shared_layers tail layers own no cache) using
  LAYER_TYPE_CACHE_MAPPING; sliding layers already get a bounded
  DynamicSlidingWindowLayer. We replace only the full_attention owned layers with
  SinkRingLayer below.
- KV sharing: the last owned layer of each type (store_full_length_kv) publishes
  its post-cache-update KV into shared_kv_states[layer_type]; the shared tail
  layers consume that. So bounding layer 23's cache automatically bounds the
  shared full-attention layers 29/35/41 too.
- Masks: create_causal_mask() builds ONE mask for the whole full_attention group
  from past_key_values.is_sliding.index(False) -> our layer 5. It asks the layer
  for (kv_length, kv_offset) via get_mask_sizes(). We report kv_offset such that
  imputed KV positions run contiguously up to the current position: for a causal
  (full) mask every cached slot is then visible to every query (correct - all
  cached tokens ARE in the past), and the last Q slots line up exactly with the
  query positions, so intra-chunk causality during chunked prefill is exact.
- StreamingLLM caveat: keys are stored post-RoPE, so evicting middle positions
  keeps absolute positions valid on the kept sink/ring entries; no re-rotation is
  needed. The model simply no longer sees the evicted middle - that is the lossy
  part that the PoC-2 state bank will compensate.

Run (offline, no HF_HOME override):
  HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python \
      experiments/gemma_recurrent_poc.py bench
"""

import argparse
import gc
import math
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

MODEL_CANDIDATES = [
    "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it",
    "google/gemma-4-E4B-it",
]
FALLBACK_E2B = "google/gemma-4-E2B-it"

FILLER = (
    "The old lighthouse keeper walked along the shore every morning, checking the "
    "tide tables and noting the weather in his worn leather journal. Gulls wheeled "
    "overhead while fishing boats returned with the dawn catch, their hulls heavy "
    "and their crews tired but satisfied after a long night on the water. "
)
NEEDLE = "The secret code is BLUE-742. Remember it carefully. "
QUESTION = (
    "What is the secret code mentioned earlier in this text? "
    "Answer with just the code."
)


# --------------------------------------------------------------------------- #
# Sink+ring cache layer
# --------------------------------------------------------------------------- #
class SinkRingLayer(DynamicLayer):
    """Bounded KV layer: keeps the first `sink` tokens ever seen plus a ring of
    the most recent `window` tokens. Everything in between is evicted.

    is_sliding stays False so masking_utils keeps this layer in the
    full_attention mask group (create_causal_mask picks the first non-sliding
    layer's get_mask_sizes to shape the group mask).

    get_mask_sizes() imputes contiguous positions [cum - stored, cum + Q) onto
    the stored slots. Sink slots therefore masquerade as recent positions, which
    is harmless under a *causal* mask (they are unconditionally in the past) and
    keeps the returned KV length consistent with the mask width. Keys are cached
    post-RoPE (StreamingLLM-style), so kept entries stay positionally valid.
    """

    is_sliding = False

    def __init__(self, sink: int = 16, window: int = 1024):
        super().__init__()
        self.sink = int(sink)
        self.window = int(window)
        self.cumulative_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]

        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        cap = self.sink + self.window
        if full_keys.shape[-2] > cap:
            # First `sink` positions ever seen + last `window` positions.
            self.keys = torch.cat(
                [full_keys[..., : self.sink, :], full_keys[..., -self.window :, :]], dim=-2
            )
            self.values = torch.cat(
                [full_values[..., : self.sink, :], full_values[..., -self.window :, :]], dim=-2
            )
        else:
            self.keys, self.values = full_keys, full_values

        # Return the *un-evicted* states for this step's attention (the current
        # chunk always attends to everything still cached + itself).
        return full_keys, full_values

    def get_mask_sizes(self, query_length: int):
        stored = 0 if self.keys is None or not self.is_initialized else self.keys.shape[-2]
        kv_length = stored + query_length
        kv_offset = self.cumulative_length - stored
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.sink + self.window


def build_cache(model, mode: str, sink: int, window: int) -> DynamicCache:
    cfg = model.config.get_text_config(decoder=True)
    cache = DynamicCache(config=model.config)
    if mode == "ring":
        n_owned = cfg.num_hidden_layers - getattr(cfg, "num_kv_shared_layers", 0)
        ring_idx = [
            i for i in range(n_owned) if cfg.layer_types[i] == "full_attention"
        ]
        for i in ring_idx:
            cache.layers[i] = SinkRingLayer(sink=sink, window=window)
    return cache


def cache_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            if t is not None and isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
    return total


# --------------------------------------------------------------------------- #
# Model / tokenizer loading
# --------------------------------------------------------------------------- #
def resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in MODEL_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return MODEL_CANDIDATES[-1]


def load_model(path: str, device: str = "cuda"):
    """Load the text tower only (Gemma4ForCausalLM) from the multimodal
    checkpoint via key_mapping; skips vision/audio tower weights entirely."""
    from transformers.models.gemma4 import Gemma4ForCausalLM

    full_cfg = AutoConfig.from_pretrained(path)
    text_cfg = full_cfg.get_text_config(decoder=True)
    model = Gemma4ForCausalLM.from_pretrained(
        path,
        config=text_cfg,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        key_mapping={r"^model\.language_model": "model"},
        device_map=device,
    )
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def chat_affixes(tok):
    """Derive (prefix_ids, suffix_ids) around user content from the chat
    template, so we can token-budget the middle exactly."""
    marker = "XQZMARKERQZX"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True
    )
    if not isinstance(ids, list):  # v5 returns a BatchEncoding
        ids = ids["input_ids"]
    marker_ids = tok(marker, add_special_tokens=False).input_ids
    pos = find_subsequence(ids, marker_ids)
    assert pos >= 0, "marker not found in chat template output"
    return ids[:pos], ids[pos + len(marker_ids) :]


def filler_ids(tok, n: int) -> list[int]:
    unit = tok(FILLER, add_special_tokens=False).input_ids
    reps = math.ceil(n / len(unit))
    return (unit * reps)[:n]


def build_plain_prompt(tok, ctx: int) -> torch.Tensor:
    """Synthetic long prompt of exactly `ctx` tokens (BOS + filler)."""
    ids = [tok.bos_token_id] + filler_ids(tok, ctx - 1)
    return torch.tensor([ids], dtype=torch.long)


def build_needle_prompt(tok, ctx: int) -> torch.Tensor:
    """Chat-formatted prompt of ~ctx tokens with the needle at ~token 200."""
    prefix, suffix = chat_affixes(tok)
    needle_ids = tok(NEEDLE, add_special_tokens=False).input_ids
    question_ids = tok("\n\n" + QUESTION, add_special_tokens=False).input_ids
    budget = ctx - len(prefix) - len(suffix) - len(needle_ids) - len(question_ids)
    assert budget > 250, f"ctx={ctx} too small for needle prompt"
    pre_needle = max(0, 200 - len(prefix))
    ids = (
        prefix
        + filler_ids(tok, pre_needle)
        + needle_ids
        + filler_ids(tok, budget - pre_needle)
        + question_ids
        + suffix
    )
    return torch.tensor([ids], dtype=torch.long)


# --------------------------------------------------------------------------- #
# Prefill / decode primitives
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def chunked_prefill(model, cache, input_ids, chunk: int, keep_last_logits: int = 1):
    """Prefill input_ids through the cache in chunks; returns logits of the last
    `keep_last_logits` positions (bf16, [1, keep, V])."""
    n = input_ids.shape[1]
    logits = None
    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        keep = keep_last_logits if end == n else 1
        out = model(
            input_ids=input_ids[:, pos:end].to(model.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=keep,
        )
        logits = out.logits
        pos = end
    return logits


@torch.inference_mode()
def greedy_decode(model, cache, first_token, steps, attention_mask=None):
    """Batched greedy decode. Returns (tokens [B, steps], elapsed_seconds)."""
    device = model.device
    cur = first_token.to(device)
    tokens = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        kwargs = {}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
            )
            kwargs["attention_mask"] = attention_mask
            kwargs["position_ids"] = (attention_mask.cumsum(-1) - 1)[:, -1:]
        out = model(
            input_ids=cur,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            **kwargs,
        )
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(cur)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return torch.cat(tokens, dim=1), elapsed


def last_token_nll(model, cache, input_ids, chunk, tail=128):
    """Mean NLL of the last `tail` tokens of input_ids, teacher-forced.

    Prefills everything but the final (tail+1) tokens, then runs the final
    (tail+1)-token chunk in one forward and scores positions 1..tail of it."""
    n = input_ids.shape[1]
    assert n > tail + 1
    head, tail_ids = input_ids[:, : n - (tail + 1)], input_ids[:, n - (tail + 1) :]
    chunked_prefill(model, cache, head, chunk)
    logits = chunked_prefill(model, cache, tail_ids, chunk=tail + 1, keep_last_logits=tail + 1)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = tail_ids[:, 1:].to(logits.device)
    nll = -logprobs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item(), logits[:, -1:]


def free_cache(cache):
    del cache
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def warmup(model, tok, args):
    """Stabilize GPU clocks so per-run decode timings are comparable."""
    cache = build_cache(model, "full", args.sink, args.window)
    prompt = build_plain_prompt(tok, 256)
    chunked_prefill(model, cache, prompt, args.chunk)
    greedy_decode(model, cache, prompt[:, -1:], 32)
    free_cache(cache)


def run_bench(model, tok, args):
    ctxs = args.ctxs or [900, 2048, 8192, 16384, 32768]
    ppl_ctxs = {900, 2048, 8192}
    warmup(model, tok, args)
    modes = ["full", "ring"] if args.mode == "both" else [args.mode]
    rows = []
    weights_bytes = torch.cuda.memory_allocated()
    print(f"# weights resident: {weights_bytes / 2**30:.2f} GiB")

    for mode in modes:
        for ctx in ctxs:
            prompt = build_plain_prompt(tok, ctx)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            cache = build_cache(model, mode, args.sink, args.window)
            oom = False
            nll = float("nan")
            tps = float("nan")
            cbytes = 0
            t_prefill = float("nan")
            try:
                t0 = time.perf_counter()
                if ctx in ppl_ctxs:
                    nll, last_logits = last_token_nll(model, cache, prompt, args.chunk)
                else:
                    last_logits = chunked_prefill(model, cache, prompt, args.chunk)
                torch.cuda.synchronize()
                t_prefill = time.perf_counter() - t0
                first = last_logits[:, -1].argmax(dim=-1, keepdim=True)
                _, elapsed = greedy_decode(model, cache, first, args.decode_tokens)
                tps = args.decode_tokens / elapsed
                cbytes = cache_bytes(cache)
            except torch.OutOfMemoryError:
                oom = True
                torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated()
            rows.append(
                dict(
                    mode=mode, ctx=ctx, oom=oom,
                    cache_mib=cbytes / 2**20,
                    peak_gib=peak / 2**30,
                    delta_gib=(peak - base) / 2**30,
                    tps=tps, nll=nll, prefill_s=t_prefill,
                )
            )
            free_cache(cache)
            r = rows[-1]
            print(
                f"[{mode:4s}] ctx={ctx:6d} oom={oom} cache={r['cache_mib']:.1f}MiB "
                f"peak={r['peak_gib']:.2f}GiB (delta {r['delta_gib']:.2f}) "
                f"decode={r['tps']:.2f}tok/s nll_last128={r['nll']:.4f} "
                f"prefill={r['prefill_s']:.1f}s"
            )

    print("\n## Bench: full KV vs sink+ring "
          f"(sink={args.sink}, window={args.window}, decode={args.decode_tokens} tok, "
          f"chunked prefill {args.chunk})\n")
    hdr = ("| mode | ctx | cache MiB | peak GiB | peak-weights GiB | decode tok/s "
           "| NLL(last128) | prefill s |")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["oom"]:
            print(f"| {r['mode']} | {r['ctx']} | OOM | OOM | OOM | OOM | OOM | OOM |")
        else:
            nll = "" if math.isnan(r["nll"]) else f"{r['nll']:.4f}"
            print(
                f"| {r['mode']} | {r['ctx']} | {r['cache_mib']:.1f} | {r['peak_gib']:.2f} "
                f"| {r['delta_gib']:.2f} | {r['tps']:.2f} | {nll} | {r['prefill_s']:.1f} |"
            )
    return rows


def run_needle(model, tok, args):
    modes = ["full", "ring"] if args.mode == "both" else [args.mode]
    results = []
    for ctx in [900, args.ctx]:
        for mode in modes:
            prompt = build_needle_prompt(tok, ctx)
            cache = build_cache(model, mode, args.sink, args.window)
            logits = chunked_prefill(model, cache, prompt, args.chunk)
            first = logits[:, -1].argmax(dim=-1, keepdim=True)
            rest, _ = greedy_decode(model, cache, first, 31)
            out_ids = torch.cat([first, rest], dim=1)[0].tolist()
            eos_ids = {1, 106}  # <eos>, <end_of_turn>
            cut = next((i for i, t in enumerate(out_ids) if t in eos_ids), len(out_ids))
            out_ids = out_ids[:cut]
            text = tok.decode(out_ids, skip_special_tokens=True).strip()
            text = text.split("\n")[0][:120]
            hit = "BLUE-742" in text
            results.append(dict(ctx=ctx, mode=mode, hit=hit, text=text))
            print(f"[needle ctx={ctx:5d} {mode:4s}] recalled={hit} out: {text!r}")
            free_cache(cache)
    return results


def run_concurrency(model, tok, args):
    """Virtual-session concurrency sweep. Prefill ONE session of ctx-per-session
    tokens, then replicate its cache tensors across the batch dim to B (real
    copies via the layer API's batch_repeat_interleave — honest memory cost,
    throwaway quality) and decode batched greedy with a distinct first token per
    row. B_max = largest B that completes with >= 1 GiB device headroom."""
    steps = args.decode_tokens
    headroom = 1 << 30
    # Total bytes torch could ever use: free device memory now + what it holds.
    avail = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()
    print(f"# concurrency: torch-usable {avail / 2**30:.2f} GiB, headroom 1 GiB, "
          f"decode {steps} tok/session")
    word_ids = tok(" one two three four five six seven eight nine ten red blue"
                   " green gold iron salt north south east west",
                   add_special_tokens=False).input_ids

    ladder = args.ladder or [8, 16, 32, 64, 128, 192, 256, 320, 384]
    modes = ["ring", "full"] if args.mode == "both" else [args.mode]
    plans = [(m, args.ctx_per_session, ladder) for m in modes]
    if args.mode == "both":
        plans.append(("full", 16384, [8, 16]))  # long-ctx collapse of full KV
        plans.append(("ring", 16384, [64]))     # must match ring @ 4096

    warmup(model, tok, args)
    tables = []
    for mode, ctx, ladder_b in plans:
        prompt = build_plain_prompt(tok, ctx)
        rows = []
        for B in ladder_b:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            cache = build_cache(model, mode, args.sink, args.window)
            row = dict(B=B, ok=False, green=False, agg=float("nan"),
                       per=float("nan"), cache_mib=float("nan"),
                       peak_gib=float("nan"), resv_gib=float("nan"))
            try:
                chunked_prefill(model, cache, prompt, args.chunk)
                for layer in cache.layers:
                    layer.batch_repeat_interleave(B)
                first = torch.tensor(
                    [[word_ids[i % len(word_ids)]] for i in range(B)], dtype=torch.long)
                _, elapsed = greedy_decode(model, cache, first, steps)
                resv = torch.cuda.max_memory_reserved()
                row.update(ok=True, agg=B * steps / elapsed, per=steps / elapsed,
                           cache_mib=cache_bytes(cache) / 2**20,
                           peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                           resv_gib=resv / 2**30, green=resv <= avail - headroom)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                row["resv_gib"] = torch.cuda.max_memory_reserved() / 2**30
            free_cache(cache)
            rows.append(row)
            print(f"[conc {mode:4s} ctx={ctx:5d} B={B:3d}] ok={row['ok']} "
                  f"green={row['green']} agg={row['agg']:.1f}tok/s "
                  f"per={row['per']:.2f} cache={row['cache_mib']:.0f}MiB "
                  f"reserved={row['resv_gib']:.2f}GiB")
            if not row["ok"]:
                break  # OOM: larger B is hopeless
        tables.append((mode, ctx, rows))

    for mode, ctx, rows in tables:
        green_bs = [r["B"] for r in rows if r["green"]]
        print(f"\n### concurrency {mode} @ ctx/session {ctx} — "
              f"B_max(green) = {max(green_bs) if green_bs else 0}\n")
        print("| B | cache MiB total | per-slot MiB | peak alloc GiB "
              "| peak reserved GiB | agg tok/s | per-stream tok/s | status |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if not r["ok"]:
                print(f"| {r['B']} | - | - | - | {r['resv_gib']:.2f} | - | - | OOM |")
            else:
                status = "green" if r["green"] else "over-budget"
                print(f"| {r['B']} | {r['cache_mib']:.0f} | {r['cache_mib']/r['B']:.1f} "
                      f"| {r['peak_gib']:.2f} | {r['resv_gib']:.2f} "
                      f"| {r['agg']:.1f} | {r['per']:.2f} | {status} |")
    return tables


def run_batch(model, tok, args):
    prompts = [
        "Write one sentence about the ocean.",
        "Name three prime numbers.",
        "What is the capital of France?",
        "Give a synonym for 'happy'.",
        "What color is chlorophyll?",
        "State Newton's second law briefly.",
        "Name a famous composer.",
        "What is 12 times 12?",
    ][: args.batch_size]
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    B = input_ids.shape[0]

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cache = build_cache(model, args.mode if args.mode != "both" else "ring",
                        args.sink, args.window)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    first = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    rest, elapsed = greedy_decode(model, cache, first, args.decode_tokens - 1,
                                  attention_mask=attn)
    toks = torch.cat([first, rest], dim=1)
    peak = torch.cuda.max_memory_allocated()
    agg = B * args.decode_tokens / elapsed
    print(f"\n[batch B={B} mode=ring] aggregate decode: {agg:.1f} tok/s "
          f"({args.decode_tokens} tok/seq), peak VRAM {peak / 2**30:.2f} GiB, "
          f"cache {cache_bytes(cache) / 2**20:.1f} MiB")
    for i, p in enumerate(prompts):
        text = tok.decode(toks[i], skip_special_tokens=True).strip().split("\n")[0][:80]
        print(f"  - {p!r} -> {text!r}")
    free_cache(cache)
    return agg, peak


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["bench", "needle", "batch", "concurrency"])
    ap.add_argument("--mode", choices=["full", "ring", "both"], default="both")
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=8192, help="needle context length")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk size")
    ap.add_argument("--decode-tokens", type=int, default=None,
                    help="decode steps (default: 64; 128 for concurrency)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ctx-per-session", type=int, default=4096,
                    help="context tokens per virtual session (concurrency)")
    ap.add_argument("--ladder", type=int, nargs="*", default=None,
                    help="override concurrency batch-size ladder")
    ap.add_argument("--ctxs", type=int, nargs="*", default=None,
                    help="override bench context lengths")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    if args.decode_tokens is None:
        args.decode_tokens = 128 if args.cmd == "concurrency" else 64

    path = resolve_model_path(args.model_path)
    print(f"# loading {path} (text tower only, eager, bf16)")
    t0 = time.perf_counter()
    model = load_model(path)
    tok = AutoTokenizer.from_pretrained(path)
    print(f"# loaded in {time.perf_counter() - t0:.1f}s; "
          f"weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    cfg = model.config.get_text_config(decoder=True)
    n_owned = cfg.num_hidden_layers - getattr(cfg, "num_kv_shared_layers", 0)
    ring_idx = [i for i in range(n_owned) if cfg.layer_types[i] == "full_attention"]
    print(f"# growing-KV (ring-capped) layers: {ring_idx}; "
          f"owned layers: {n_owned}/{cfg.num_hidden_layers}")

    if args.cmd == "bench":
        run_bench(model, tok, args)
    elif args.cmd == "needle":
        run_needle(model, tok, args)
    elif args.cmd == "batch":
        run_batch(model, tok, args)
    elif args.cmd == "concurrency":
        run_concurrency(model, tok, args)


if __name__ == "__main__":
    main()
