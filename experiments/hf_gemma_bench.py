#!/usr/bin/env python
"""Hugging Face Transformers baseline for Gemma throughput.

This benchmark intentionally uses the same local Gemma loader and prompt builder
as the native wkvm Gemma benchmark. It measures the framework path wkvm still
uses for model math, but without wkvm's scheduler/routed-span cache.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
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

from native_gemma_engine_smoke import build_prompt, prompt_lengths
from native_gemma_smoke import load_model, resolve_model_path


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def parse_concurrency(raw: str) -> list[int]:
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        val = int(part)
        if val < 1:
            raise argparse.ArgumentTypeError("concurrency values must be >= 1")
        vals.append(val)
    if not vals:
        raise argparse.ArgumentTypeError("--concurrency must contain at least one value")
    return vals


def bench_prompt_lengths(ctx: int, concurrency: int, mode: str) -> list[int]:
    if mode == "staggered":
        return prompt_lengths(ctx, concurrency)
    if mode == "uniform":
        return [ctx] * concurrency
    raise ValueError(f"unknown prompt length mode: {mode}")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def round_or_none(x: float | None, ndigits: int = 3) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(x, ndigits)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def cuda_peak_reserved_gib() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_reserved() / 2**30


def torch_usable_gib(mem_cap_gib: float) -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    usable = free + torch.cuda.memory_reserved()
    usable = min(usable, int(mem_cap_gib * 2**30))
    return usable / 2**30


def synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cuda_empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_serial_request(
    model,
    prompt: list[int],
    out_tokens: int,
    device: str,
    *,
    prefill_chunk: int,
) -> tuple[float, float, int]:
    import torch

    ids = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        cache = None
        out = None
        for start in range(0, ids.shape[1], prefill_chunk):
            chunk = ids[:, start : start + prefill_chunk]
            out = model(
                input_ids=chunk,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
        if out is None:
            raise ValueError("empty prompt")
        cache = out.past_key_values
        tok = int(out.logits[0, -1].float().argmax().item())
        first_token_time = time.perf_counter()
        produced = 1
        for _ in range(out_tokens - 1):
            out = model(
                input_ids=torch.tensor([[tok]], dtype=torch.long, device=device),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                attention_mask={"full_attention": None, "sliding_attention": None},
            )
            cache = out.past_key_values
            tok = int(out.logits[0, -1].float().argmax().item())
            produced += 1
    synchronize()
    finished = time.perf_counter()
    del cache, out, ids
    return finished - started, finished - first_token_time, produced


def pad_left(prompts: list[list[int]], pad_id: int) -> tuple[Any, Any]:
    import torch

    max_len = max(len(prompt) for prompt in prompts)
    input_ids = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        offset = max_len - len(prompt)
        input_ids[row, offset:] = torch.tensor(prompt, dtype=torch.long)
        attention_mask[row, offset:] = 1
    return input_ids, attention_mask


def run_batched_requests(
    model,
    prompts: list[list[int]],
    out_tokens: int,
    device: str,
    pad_id: int,
    *,
    prefill_chunk: int,
) -> tuple[float, float, int]:
    import torch

    input_ids, attention_mask = pad_left(prompts, pad_id)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        cache = None
        out = None
        for start in range(0, input_ids.shape[1], prefill_chunk):
            end = min(start + prefill_chunk, input_ids.shape[1])
            out = model(
                input_ids=input_ids[:, start:end],
                attention_mask=attention_mask[:, :end],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
        if out is None:
            raise ValueError("empty prompt batch")
        cache = out.past_key_values
        next_tokens = out.logits[:, -1].float().argmax(dim=-1)
        first_token_time = time.perf_counter()
        produced = len(prompts)
        for _ in range(out_tokens - 1):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((len(prompts), 1), dtype=torch.long, device=device)],
                dim=1,
            )
            out = model(
                input_ids=next_tokens.reshape(-1, 1),
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            next_tokens = out.logits[:, -1].float().argmax(dim=-1)
            produced += len(prompts)
    synchronize()
    finished = time.perf_counter()
    del cache, out, input_ids, attention_mask, next_tokens
    return finished - started, finished - first_token_time, produced


def run_row(model, tok, B: int, args, usable_gib: float | None) -> dict[str, Any]:
    reset_cuda_peak()
    row: dict[str, Any] = {
        "B": B,
        "mode": args.mode,
        "success_count": 0,
        "error_count": 0,
        "p50_latency_s": None,
        "p95_latency_s": None,
        "agg_decode_tok_s": None,
        "e2e_output_tok_s": None,
        "peak_reserved_gib": None,
        "green": False,
        "elapsed_s": None,
        "error": None,
    }
    started = time.perf_counter()
    try:
        lengths = bench_prompt_lengths(args.ctx, B, args.prompt_lengths)
        prompts = [build_prompt(tok, n, i) for i, n in enumerate(lengths)]
        latencies: list[float] = []
        decode_s = 0.0
        output_tokens = 0

        if args.mode == "serial":
            for prompt in prompts:
                latency, req_decode_s, produced = run_serial_request(
                    model,
                    prompt,
                    args.out,
                    args.device,
                    prefill_chunk=args.prefill_chunk,
                )
                latencies.append(latency)
                decode_s += req_decode_s
                output_tokens += produced
                gc.collect()
                cuda_empty_cache()
        elif args.mode == "batched":
            pad_id = tok.pad_token_id
            if pad_id is None:
                pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
            latency, decode_s, output_tokens = run_batched_requests(
                model,
                prompts,
                args.out,
                args.device,
                int(pad_id),
                prefill_chunk=args.prefill_chunk,
            )
            latencies = [latency] * B
        else:
            raise ValueError(f"unknown mode: {args.mode}")

        elapsed = time.perf_counter() - started
        peak_reserved = cuda_peak_reserved_gib()
        decode_tokens = B * max(0, args.out - 1)
        row.update(
            {
                "success_count": B,
                "error_count": 0,
                "p50_latency_s": round_or_none(statistics.median(latencies)),
                "p95_latency_s": round_or_none(percentile(latencies, 0.95)),
                "agg_decode_tok_s": round_or_none(
                    decode_tokens / decode_s if decode_s > 0 else None
                ),
                "e2e_output_tok_s": round_or_none(
                    output_tokens / elapsed if elapsed > 0 else None
                ),
                "peak_reserved_gib": round_or_none(peak_reserved),
                "green": bool(
                    peak_reserved is not None
                    and usable_gib is not None
                    and peak_reserved <= usable_gib - args.headroom_gib
                ),
                "elapsed_s": round(elapsed, 3),
                "prompt_lengths": [len(prompt) for prompt in prompts],
                "decode_seconds": round_or_none(decode_s),
            }
        )
    except Exception as exc:
        row["error_count"] = max(B, 1)
        row["error"] = str(exc).splitlines()[0]
        row["elapsed_s"] = round(time.perf_counter() - started, 3)
        row["peak_reserved_gib"] = round_or_none(cuda_peak_reserved_gib())
        gc.collect()
        cuda_empty_cache()
    return row


def run(args) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    model = load_model(path, args.device, args.attn)
    tok = AutoTokenizer.from_pretrained(path)
    usable_gib = torch_usable_gib(args.mem_cap_gib)

    rows = []
    for B in args.concurrency:
        row = run_row(model, tok, B, args, usable_gib)
        rows.append(row)
        print(
            f"[hf-transformers mode={args.mode} ctx={args.ctx} out={args.out} B={B}] "
            f"success={row['success_count']}/{B} "
            f"p50={row['p50_latency_s']}s p95={row['p95_latency_s']}s "
            f"agg={row['agg_decode_tok_s']}tok/s "
            f"reserved={row['peak_reserved_gib']}GiB green={row['green']}"
        )
        if row.get("error") and args.stop_on_failure:
            break

    payload: dict[str, Any] = {
        "schema": "wkvm.hf_gemma_bench.v1",
        "engine": "hf-transformers",
        "mode": args.mode,
        "context_tokens_per_session": args.ctx,
        "prompt_lengths_mode": args.prompt_lengths,
        "decode_tokens_per_session": args.out,
        "prefill_chunk": args.prefill_chunk,
        "mem_cap_gib": args.mem_cap_gib,
        "headroom_gib": args.headroom_gib,
        "torch_usable_gib": round_or_none(usable_gib),
        "model_path": path,
        "dtype": "bfloat16",
        "device": args.device,
        "attn": args.attn,
        "git_commit": git_commit(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "concurrency": args.concurrency,
        "notes": [
            "Uses Hugging Face Gemma4ForCausalLM forward and full HF cache semantics.",
            "serial mode measures one request at a time through one loaded model.",
            "batched mode left-pads prompts and runs one HF batch with an attention mask.",
            "Both modes chunk prefill to avoid measuring one large full-context prefill allocation.",
        ],
        "summary": {
            "bmax_green": max((r["B"] for r in rows if r["green"]), default=0),
            "max_success_B": max(
                (r["B"] for r in rows if r["success_count"] == r["B"]),
                default=0,
            ),
            "best_green_agg_decode_tok_s": max(
                (r["agg_decode_tok_s"] or 0.0 for r in rows if r["green"]),
                default=0.0,
            ),
        },
        "rows": rows,
    }
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency("1,2,4,8"))
    ap.add_argument("--prompt-lengths", choices=["staggered", "uniform"], default="staggered")
    ap.add_argument("--mode", choices=["serial", "batched"], default="batched")
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--mem-cap-gib", type=float, default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)))
    ap.add_argument("--headroom-gib", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--stop-on-failure", action="store_true")
    args = ap.parse_args()
    if args.prefill_chunk < 1:
        raise SystemExit("--prefill-chunk must be >= 1")
    run(args)


if __name__ == "__main__":
    main()
