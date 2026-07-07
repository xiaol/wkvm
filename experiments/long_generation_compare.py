#!/usr/bin/env python3
"""Long-prompt / long-output token comparison for gemma-4-E4B-it.

This harness is intentionally separate from gemma_demo_video.py.  The video is
a presentation artifact; this script is a reproducible benchmark/comparison:

  1. build the same exact-token long prompt in every engine,
  2. generate a fixed long greedy output with ignore-eos semantics,
  3. write output token ids + timing to JSON,
  4. compare JSON records from wkvm recurrent mode, vLLM, and/or SGLang.

The run step is cross-venv friendly:

  # wkvm recurrent mode / patched HF path
  HF_HUB_OFFLINE=1 python experiments/long_generation_compare.py run \
      --engine wkvm --wkvm-mode ring --ctx 16384 --out-tokens 1024 \
      --out experiments/results/long_gen_wkvm_ring.json

  # vLLM, from the vLLM venv used in the existing bench
  /run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/vllm/bin/python \
      experiments/long_generation_compare.py run --engine vllm \
      --ctx 16384 --out-tokens 1024 \
      --out experiments/results/long_gen_vllm.json

  # SGLang, from the SGLang venv used in the existing bench
  /run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/sglang/bin/python \
      experiments/long_generation_compare.py run --engine sglang \
      --ctx 16384 --out-tokens 1024 \
      --out experiments/results/long_gen_sglang.json

  python experiments/long_generation_compare.py compare \
      experiments/results/long_gen_wkvm_ring.json \
      experiments/results/long_gen_vllm.json \
      experiments/results/long_gen_sglang.json \
      --out experiments/results/long_generation_compare.md
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "experiments" / "results"

DEFAULT_MODEL = "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it"

FILLER = (
    "The benchmark document describes a serving system under sustained load. "
    "Requests arrive with long histories, the scheduler admits work according "
    "to a memory budget, and each decode step streams one more token through "
    "the model. Engineers record latency, throughput, cache size, and output "
    "tokens so the same prompt can be audited across engines. "
)

NEEDLE = (
    "CRITICAL RECORD: the codename is BLUE-742, the deployment city is "
    "Samarkand, and the checksum word is lantern. "
)

QUESTION = (
    "\n\nTask: In the first sentence, state the codename, deployment city, "
    "and checksum word from the critical record. Then continue with a detailed "
    "technical explanation of the serving benchmark.\n\nAnswer:\n"
)


def now() -> float:
    return time.perf_counter()


def sha_ids(ids: list[int]) -> str:
    h = hashlib.sha256()
    h.update((" ".join(map(str, ids))).encode("ascii"))
    return h.hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def repeated_ids(tok, n: int, text: str = FILLER) -> list[int]:
    unit = tok(text, add_special_tokens=False).input_ids
    if not unit:
        raise ValueError("filler encoded to zero tokens")
    return (unit * math.ceil(n / len(unit)))[:n]


def build_prompt_ids(
    tok,
    ctx: int,
    kind: str,
    needle_pos: int,
) -> list[int]:
    """Build an exact-length token prompt.

    ``recall`` places the critical record near ``needle_pos`` so ring-only
    recurrent mode must rely on memory beyond the sink/window if ctx is long.
    ``continuation`` is a neutral synthetic continuation prompt for timing and
    numeric-drift checks.
    """
    bos = [] if tok.bos_token_id is None else [tok.bos_token_id]
    if ctx <= len(bos) + 8:
        raise ValueError(f"ctx={ctx} is too small")

    if kind == "continuation":
        return (bos + repeated_ids(tok, ctx - len(bos)))[:ctx]

    if kind != "recall":
        raise ValueError(f"unknown prompt kind: {kind}")

    needle = tok(NEEDLE, add_special_tokens=False).input_ids
    question = tok(QUESTION, add_special_tokens=False).input_ids
    if len(bos) + len(needle) + len(question) >= ctx:
        raise ValueError(
            f"ctx={ctx} too small for recall prompt "
            f"(fixed={len(bos) + len(needle) + len(question)} tokens)"
        )

    before = max(0, needle_pos - len(bos))
    max_before = ctx - len(bos) - len(needle) - len(question)
    before = min(before, max_before)
    after = ctx - len(bos) - before - len(needle) - len(question)
    ids = (
        bos
        + repeated_ids(tok, before)
        + needle
        + repeated_ids(tok, after)
        + question
    )
    assert len(ids) == ctx, (len(ids), ctx)
    return ids


def decode_preview(tok, ids: list[int], limit: int = 700) -> str:
    text = tok.decode(ids, skip_special_tokens=True)
    text = text.replace("\r\n", "\n")
    return text[:limit]


_BREAK_CACHE: dict[int, bool] = {}


def break_mask_for(tok, ids: list[int]) -> list[bool]:
    """Sentence-break mask per token position for RoutedSpanLayer."""
    mask: list[bool] = []
    for tid in ids:
        b = _BREAK_CACHE.get(tid)
        if b is None:
            b = any(c in tok.decode([tid]) for c in ".!?\n")
            _BREAK_CACHE[tid] = b
        mask.append(b)
    return mask


def extract_sglang_output_ids(obj: Any) -> list[int]:
    """SGLang output shape has changed across versions; handle common forms."""
    if isinstance(obj, list):
        if len(obj) != 1:
            raise ValueError(f"expected one SGLang output, got {len(obj)}")
        obj = obj[0]
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected SGLang output type: {type(obj).__name__}")
    for key in ("output_ids", "output_token_ids"):
        if key in obj and obj[key] is not None:
            return [int(x) for x in obj[key]]
    meta = obj.get("meta_info") or {}
    for key in ("output_ids", "output_token_ids"):
        if key in meta and meta[key] is not None:
            return [int(x) for x in meta[key]]
    raise KeyError(
        "SGLang did not return output token ids. Try a newer SGLang build or "
        "inspect the raw JSON printed by engine.generate()."
    )


def gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return int(out.strip().splitlines()[0])


class VramMonitor:
    """Whole-GPU memory sampler for engines that own CUDA outside torch."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.peak_mib: int | None = None
        self.baseline_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_mem_used_mib()
            if used is not None:
                self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.baseline_mib = gpu_mem_used_mib()
        self.peak_mib = self.baseline_mib
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def base_record(args, tok, prompt_ids: list[int]) -> dict[str, Any]:
    return {
        "schema": "wkvm.long_generation_compare.v1",
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "engine": args.engine,
        "model_path": args.model,
        "prompt": {
            "kind": args.prompt_kind,
            "ctx_tokens": len(prompt_ids),
            "needle_pos": args.needle_pos if args.prompt_kind == "recall" else None,
            "sha256": sha_ids(prompt_ids),
            "preview": decode_preview(tok, prompt_ids),
        },
        "generation": {
            "requested_output_tokens": args.out_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }


def run_wkvm(args, tok, prompt_ids: list[int]) -> dict[str, Any]:
    import gc
    import torch

    sys.path.insert(0, str(HERE))
    import gemma_recurrent_poc as poc

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.mem_cap_gib * 2**30 / total)
    )
    model = poc.load_model(args.model, attn=args.attn)

    class PArgs:
        sink = args.sink
        window = args.window
        k_states = args.k_states
        seg = args.seg
        reps = args.reps
        select = args.select
        m_slots = args.m_slots
        route_on = args.route_on
        span = args.span
        trace = False

    pargs = PArgs()
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    def one_pass(max_tokens: int, keep_output: bool) -> tuple[float, list[int], dict[str, Any]]:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cache = poc.build_cache(model, args.wkvm_mode, pargs)
        if args.span:
            poc.set_span_break_mask(cache, break_mask_for(tok, prompt_ids))
        torch.cuda.synchronize()
        t0 = now()
        logits = poc.chunked_prefill(model, cache, prompt, args.chunk)
        first = logits[:, -1].argmax(dim=-1, keepdim=True)
        if max_tokens > 1:
            rest, _decode_s = poc.greedy_decode(model, cache, first, max_tokens - 1)
            out = torch.cat([first, rest], dim=1)
        else:
            torch.cuda.synchronize()
            out = first
        torch.cuda.synchronize()
        wall = now() - t0
        ids = out[0].tolist() if keep_output else []
        metrics = {
            "cache_mib": round(poc.cache_bytes(cache) / 2**20, 3),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
        }
        poc.free_cache(cache)
        return wall, [int(x) for x in ids], metrics

    # Warm clocks/JIT once without affecting measured passes.
    with contextlib.suppress(Exception):
        poc.warmup(model, tok, argparse.Namespace(sink=args.sink, window=args.window, chunk=args.chunk))

    t_first, _, first_metrics = one_pass(1, keep_output=False)
    t_full, out_ids, full_metrics = one_pass(args.out_tokens, keep_output=True)
    decode_s = max(t_full - t_first, 1e-9)
    actual_out = len(out_ids)
    decode_out = max(actual_out - 1, 0)

    rec = base_record(args, tok, prompt_ids)
    rec["generation"].update(
        {
            "actual_output_tokens": actual_out,
            "completed_requested": actual_out == args.out_tokens,
        }
    )
    rec.update(
        {
            "engine_config": {
                "wkvm_mode": args.wkvm_mode,
                "attn": args.attn,
                "sink": args.sink,
                "window": args.window,
                "chunk": args.chunk,
                "mem_cap_gib": args.mem_cap_gib,
                "m_slots": args.m_slots,
                "route_on": args.route_on,
                "span": args.span,
                "span_break_mask": "sentence_punctuation" if args.span else None,
            },
            "timing": {
                "prefill_plus_first_s": round(t_first, 6),
                "full_wall_s": round(t_full, 6),
                "decode_delta_s": round(decode_s, 6),
                "decode_tok_s": round(decode_out / decode_s, 3),
                "e2e_output_tok_s": round(actual_out / t_full, 3),
            },
            "memory": {
                "first_pass": first_metrics,
                "full_pass": full_metrics,
            },
            "output_ids": out_ids,
            "output_text": tok.decode(out_ids, skip_special_tokens=True),
        }
    )
    return rec


def run_vllm(args, tok, prompt_ids: list[int]) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams
    import vllm

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len or (len(prompt_ids) + args.out_tokens + 16),
        gpu_memory_utilization=args.vllm_gpu_mem_util,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        swap_space=0,
        disable_log_stats=True,
        dtype="bfloat16",
    )
    try:
        llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "audio": 0})
        mm_note = "limit_mm_per_prompt={image:0,audio:0}"
    except TypeError:
        llm = LLM(**kwargs)
        mm_note = "default multimodal budget"

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.out_tokens,
        ignore_eos=True,
    )
    sp1 = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
    )

    def generate(params):
        t0 = now()
        outs = llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)
        wall = now() - t0
        ids = [int(x) for x in outs[0].outputs[0].token_ids]
        return wall, ids

    torch.cuda.reset_peak_memory_stats()
    t_first, _ = generate(sp1)
    t_full, out_ids = generate(sp)
    decode_s = max(t_full - t_first, 1e-9)
    actual_out = len(out_ids)
    decode_out = max(actual_out - 1, 0)
    free_b, total_b = torch.cuda.mem_get_info()

    rec = base_record(args, tok, prompt_ids)
    rec["generation"].update(
        {
            "actual_output_tokens": actual_out,
            "completed_requested": actual_out == args.out_tokens,
        }
    )
    rec.update(
        {
            "engine_config": {
                "vllm_version": vllm.__version__,
                "gpu_memory_utilization": args.vllm_gpu_mem_util,
                "enforce_eager": args.enforce_eager,
                "max_model_len": kwargs["max_model_len"],
                "prefix_caching": False,
                "mm_note": mm_note,
            },
            "timing": {
                "prefill_plus_first_s": round(t_first, 6),
                "full_wall_s": round(t_full, 6),
                "decode_delta_s": round(decode_s, 6),
                "decode_tok_s": round(decode_out / decode_s, 3),
                "e2e_output_tok_s": round(actual_out / t_full, 3),
            },
            "memory": {
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "device_used_gib": round((total_b - free_b) / 2**30, 3),
            },
            "output_ids": out_ids,
            "output_text": tok.decode(out_ids, skip_special_tokens=True),
        }
    )
    return rec


def run_sglang(args, tok, prompt_ids: list[int]) -> dict[str, Any]:
    import sglang as sgl

    kwargs = dict(
        model_path=args.model,
        mem_fraction_static=args.sglang_mem_fraction,
        disable_radix_cache=True,
        enable_multimodal=False,
        log_level="info",
        max_running_requests=args.sglang_max_running_requests,
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
    )
    if args.sglang_attention_backend:
        kwargs["attention_backend"] = args.sglang_attention_backend

    engine = sgl.Engine(**kwargs)
    sp = {
        "temperature": 0.0,
        "max_new_tokens": args.out_tokens,
        "ignore_eos": True,
    }
    sp1 = dict(sp)
    sp1["max_new_tokens"] = 1

    def generate(params):
        with VramMonitor() as mon:
            t0 = now()
            out = engine.generate(input_ids=[prompt_ids], sampling_params=params)
            wall = now() - t0
        return wall, extract_sglang_output_ids(out), mon

    t_first, _, mon1 = generate(sp1)
    t_full, out_ids, mon2 = generate(sp)
    decode_s = max(t_full - t_first, 1e-9)
    actual_out = len(out_ids)
    decode_out = max(actual_out - 1, 0)

    with contextlib.suppress(Exception):
        engine.shutdown()

    rec = base_record(args, tok, prompt_ids)
    rec["generation"].update(
        {
            "actual_output_tokens": actual_out,
            "completed_requested": actual_out == args.out_tokens,
        }
    )
    rec.update(
        {
            "engine_config": {
                "sglang_version": getattr(sgl, "__version__", "unknown"),
                "mem_fraction_static": args.sglang_mem_fraction,
                "attention_backend": args.sglang_attention_backend or "default",
                "disable_radix_cache": True,
                "max_running_requests": args.sglang_max_running_requests,
            },
            "timing": {
                "prefill_plus_first_s": round(t_first, 6),
                "full_wall_s": round(t_full, 6),
                "decode_delta_s": round(decode_s, 6),
                "decode_tok_s": round(decode_out / decode_s, 3),
                "e2e_output_tok_s": round(actual_out / t_full, 3),
            },
            "memory": {
                "baseline_mib": mon1.baseline_mib,
                "peak_mib": max(x for x in (mon1.peak_mib, mon2.peak_mib) if x is not None),
            },
            "output_ids": out_ids,
            "output_text": tok.decode(out_ids, skip_special_tokens=True),
        }
    )
    return rec


def run_cmd(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = build_prompt_ids(
        tok,
        ctx=args.ctx,
        kind=args.prompt_kind,
        needle_pos=args.needle_pos,
    )
    if args.print_prompt:
        print(decode_preview(tok, prompt_ids, limit=2000))

    runners = {
        "wkvm": run_wkvm,
        "vllm": run_vllm,
        "sglang": run_sglang,
    }
    rec = runners[args.engine](args, tok, prompt_ids)
    atomic_write_json(Path(args.out), rec)
    print(
        f"WROTE {args.out} engine={args.engine} "
        f"ctx={len(prompt_ids)} out={len(rec['output_ids'])} "
        f"decode_tok_s={rec['timing']['decode_tok_s']}"
    )


def lcp_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def fact_flags(text: str) -> str:
    flags = [
        ("BLUE-742", "code"),
        ("Samarkand", "city"),
        ("lantern", "checksum"),
    ]
    return ", ".join(name for needle, name in flags if needle.lower() in text.lower()) or "-"


def compare_cmd(args) -> None:
    records = [json.loads(Path(p).read_text()) for p in args.records]
    if len(records) < 2:
        raise SystemExit("compare needs at least two JSON records")
    prompt_hashes = {r["prompt"]["sha256"] for r in records}
    if len(prompt_hashes) != 1:
        raise SystemExit(f"prompt hash mismatch: {sorted(prompt_hashes)}")

    names = [
        r["engine"] + (f":{r['engine_config'].get('wkvm_mode')}" if r["engine"] == "wkvm" else "")
        for r in records
    ]
    lines: list[str] = []
    first = records[0]
    lines += [
        "# Long Generation Token Compare",
        "",
        f"- prompt kind: **{first['prompt']['kind']}**",
        f"- prompt tokens: **{first['prompt']['ctx_tokens']:,}**",
        f"- prompt sha256: `{first['prompt']['sha256'][:16]}...`",
        f"- requested output tokens: **{first['generation']['requested_output_tokens']:,}**",
        "",
        "## Runs",
        "",
        "| engine | output tokens | prefill+1st s | full wall s | decode tok/s | e2e out tok/s | facts in output |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in zip(names, records):
        t = r["timing"]
        lines.append(
            f"| {name} | {len(r['output_ids']):,} | "
            f"{t['prefill_plus_first_s']:.3f} | {t['full_wall_s']:.3f} | "
            f"{t['decode_tok_s']:.1f} | {t['e2e_output_tok_s']:.1f} | "
            f"{fact_flags(r.get('output_text', ''))} |"
        )

    lines += [
        "",
        "## Pairwise Token Agreement",
        "",
        "| left | right | exact | LCP tokens | equal positions / min len | first mismatch |",
        "|---|---|---:|---:|---:|---|",
    ]
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i]["output_ids"], records[j]["output_ids"]
            n = min(len(a), len(b))
            lcp = lcp_len(a, b)
            equal = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
            if lcp < n:
                mismatch = f"{lcp}: {a[lcp]} vs {b[lcp]}"
            elif len(a) != len(b):
                mismatch = f"length differs after shared {n}"
            else:
                mismatch = "-"
            lines.append(
                f"| {names[i]} | {names[j]} | {str(a == b)} | {lcp:,} | "
                f"{equal:,}/{n:,} | {mismatch} |"
            )

    lines += ["", "## Output Heads", ""]
    for name, r in zip(names, records):
        text = (r.get("output_text") or "").replace("\n", "\\n")
        lines += [f"### {name}", "", text[:1200], ""]

    md = "\n".join(lines)
    if args.out:
        atomic_write_text(Path(args.out), md)
        print(f"WROTE {args.out}")
    else:
        print(md)


def add_run_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--engine", choices=["wkvm", "vllm", "sglang"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument("--prompt-kind", choices=["recall", "continuation"], default="recall")
    ap.add_argument("--needle-pos", type=int, default=200)
    ap.add_argument("--out", default=str(RESULTS / "long_generation_run.json"))
    ap.add_argument("--print-prompt", action="store_true")

    # wkvm recurrent-mode options.
    ap.add_argument("--wkvm-mode", choices=["full", "ring", "banked", "routed"], default="ring")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--mem-cap-gib", type=float, default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)))
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--k-states", type=int, default=16)
    ap.add_argument("--seg", type=int, default=512)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--select", choices=["shared", "per-layer"], default="shared")
    ap.add_argument("--m-slots", type=int, default=16)
    ap.add_argument("--route-on", choices=["key", "resid", "value"], default="resid")
    ap.add_argument("--span", action="store_true")

    # vLLM options.
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.82)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")

    # SGLang options.
    ap.add_argument("--sglang-mem-fraction", type=float, default=0.88)
    ap.add_argument("--sglang-attention-backend", default=None)
    ap.add_argument("--sglang-max-running-requests", type=int, default=64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    add_run_args(run)
    run.set_defaults(fn=run_cmd)

    cmp_ap = sub.add_parser("compare")
    cmp_ap.add_argument("records", nargs="+")
    cmp_ap.add_argument("--out", default=None)
    cmp_ap.set_defaults(fn=compare_cmd)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
