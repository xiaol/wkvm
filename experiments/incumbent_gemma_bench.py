#!/usr/bin/env python
"""Same-shape vLLM/SGLang Gemma throughput benchmark.

This is the incumbent-engine counterpart to ``native_gemma_bench.py`` and
``hf_gemma_bench.py``. It intentionally reuses the same prompt builder, output
length, concurrency ladder, and JSON row shape where the incumbent APIs expose
the same facts.

Run this script from the target engine's environment, for example:

  /run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/vllm/bin/python \
      experiments/incumbent_gemma_bench.py --engine vllm --ctx 13824 \
      --out 128 --concurrency 1,2,4,8 \
      --json experiments/results/vllm_gemma_ctx13824_out128_ladder.json

  /run/media/xiaol/B214449214445C0B/wkvm_bench/venvs/sglang/bin/python \
      experiments/incumbent_gemma_bench.py --engine sglang --ctx 13824 \
      --out 128 --concurrency 1,2,4,8 \
      --json experiments/results/sglang_gemma_ctx13824_out128_ladder.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import threading
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
from native_gemma_smoke import resolve_model_path
from bench_prompt_utils import (
    SyntheticBenchTokenizer,
    prompt_fingerprint_row_fields,
    prompt_set_fingerprint,
)


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


def prompt_token_source(args) -> str:
    return "synthetic" if getattr(args, "synthetic_prompts", False) else "hf_tokenizer"


def uses_hf_tokenizer(args) -> bool:
    return prompt_token_source(args) == "hf_tokenizer"


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
    first = out.strip().splitlines()[0]
    return int(first)


class VramMonitor:
    """Whole-GPU memory sampler for engines that do not expose torch peaks."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.baseline_mib: int | None = None
        self.peak_mib: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_mem_used_mib()
            if used is not None:
                if self.peak_mib is None:
                    self.peak_mib = used
                else:
                    self.peak_mib = max(self.peak_mib, used)
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.baseline_mib = gpu_mem_used_mib()
        self.peak_mib = self.baseline_mib
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        final = gpu_mem_used_mib()
        if final is not None:
            self.peak_mib = final if self.peak_mib is None else max(self.peak_mib, final)

    @property
    def engine_peak_delta_gib(self) -> float | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0, self.peak_mib - self.baseline_mib) / 1024.0


def monitor_memory_row(monitor: VramMonitor) -> dict[str, Any]:
    return {
        "baseline_gpu_mib": monitor.baseline_mib,
        "peak_gpu_mib": monitor.peak_mib,
        "peak_engine_delta_gib": round_or_none(monitor.engine_peak_delta_gib),
    }


def synchronize_cuda() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def reset_torch_peak() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def torch_peak_alloc_gib() -> float | None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**30
    return None


def torch_peak_reserved_gib() -> float | None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved() / 2**30
    return None


def cleanup_cuda() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def normalize_vllm_outputs(outputs: list[Any], expected: int) -> list[list[int]]:
    if len(outputs) != expected:
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {expected} prompts")
    rows: list[list[int]] = []
    for out in outputs:
        rows.append([int(x) for x in out.outputs[0].token_ids])
    return rows


def sglang_output_ids(obj: Any) -> list[int]:
    if not isinstance(obj, dict):
        raise TypeError(f"unexpected SGLang output item type: {type(obj).__name__}")
    for key in ("output_ids", "output_token_ids"):
        if obj.get(key) is not None:
            return [int(x) for x in obj[key]]
    meta = obj.get("meta_info") or {}
    for key in ("output_ids", "output_token_ids"):
        if meta.get(key) is not None:
            return [int(x) for x in meta[key]]
    raise KeyError("SGLang output did not contain output token ids")


def normalize_sglang_outputs(obj: Any, expected: int) -> list[list[int]]:
    if expected == 1 and isinstance(obj, dict):
        return [sglang_output_ids(obj)]
    if not isinstance(obj, list):
        raise TypeError(f"unexpected SGLang output type: {type(obj).__name__}")
    if len(obj) != expected:
        raise RuntimeError(f"SGLang returned {len(obj)} outputs for {expected} prompts")
    return [sglang_output_ids(item) for item in obj]


def make_row(
    *,
    B: int,
    prompt_lens: list[int],
    first_wall_s: float,
    full_wall_s: float,
    outputs: list[list[int]],
    mem: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    success_count = sum(1 for ids in outputs if ids)
    latencies = [full_wall_s] * success_count
    output_tokens = sum(len(ids) for ids in outputs)
    decode_tokens = sum(max(0, len(ids) - 1) for ids in outputs)
    decode_s = max(full_wall_s - first_wall_s, 0.0)
    row: dict[str, Any] = {
        "B": B,
        "success_count": success_count,
        "error_count": B - success_count + (1 if error else 0),
        "p50_latency_s": round_or_none(statistics.median(latencies) if latencies else None),
        "p95_latency_s": round_or_none(percentile(latencies, 0.95)),
        "prefill_plus_first_s": round_or_none(first_wall_s),
        "decode_seconds": round_or_none(decode_s),
        "agg_decode_tok_s": round_or_none(
            decode_tokens / decode_s if decode_s > 0 else None
        ),
        "e2e_output_tok_s": round_or_none(
            output_tokens / full_wall_s if full_wall_s > 0 else None
        ),
        "elapsed_s": round_or_none(full_wall_s),
        "prompt_lengths": prompt_lens,
        "output_token_counts": [len(ids) for ids in outputs],
        "error": error,
        **mem,
    }
    return row


def row_green(row: dict[str, Any], args) -> bool:
    candidates = [
        row.get("peak_engine_delta_gib"),
        row.get("peak_reserved_gib"),
        row.get("peak_alloc_gib"),
    ]
    mem_gib = next((float(x) for x in candidates if x is not None), None)
    if mem_gib is None:
        return False
    return mem_gib <= args.mem_cap_gib - args.headroom_gib


def run_vllm(
    args,
    tok,
    prompts_by_b: dict[int, list[list[int]]],
    monitor: VramMonitor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    max_prompt_len = max(len(p) for prompts in prompts_by_b.values() for p in prompts)
    max_model_len = args.max_model_len or (max_prompt_len + args.out + 16)
    kwargs = {
        "model": args.model_path,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": args.vllm_gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "swap_space": 0,
        "disable_log_stats": True,
        "dtype": "bfloat16",
    }
    try:
        llm = LLM(**kwargs, limit_mm_per_prompt={"image": 0, "audio": 0})
        mm_note = "limit_mm_per_prompt={image:0,audio:0}"
    except TypeError:
        llm = LLM(**kwargs)
        mm_note = "default multimodal budget"

    sp1 = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1, ignore_eos=True)
    spn = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.out,
        ignore_eos=True,
    )

    if args.warmup:
        with contextlib.suppress(Exception):
            llm.generate(
                [{"prompt_token_ids": next(iter(prompts_by_b.values()))[0][:128]}],
                sp1,
                use_tqdm=False,
            )

    rows: list[dict[str, Any]] = []
    reset_torch_peak()
    for B, prompts in prompts_by_b.items():
        fingerprint_fields = prompt_fingerprint_row_fields(
            prompt_set_fingerprint(
                prompts,
                prompt_token_source=prompt_token_source(args),
            )
        )
        try:
            reqs = [{"prompt_token_ids": p} for p in prompts]
            synchronize_cuda()
            t0 = time.perf_counter()
            first = llm.generate(reqs, sp1, use_tqdm=False)
            synchronize_cuda()
            t_first = time.perf_counter() - t0

            synchronize_cuda()
            t0 = time.perf_counter()
            full = llm.generate(reqs, spn, use_tqdm=False)
            synchronize_cuda()
            t_full = time.perf_counter() - t0
            outputs = normalize_vllm_outputs(full, B)
            mem = {
                "peak_alloc_gib": round_or_none(torch_peak_alloc_gib()),
                "peak_reserved_gib": round_or_none(torch_peak_reserved_gib()),
                **monitor_memory_row(monitor),
            }
            row = make_row(
                B=B,
                prompt_lens=[len(p) for p in prompts],
                first_wall_s=t_first,
                full_wall_s=t_full,
                outputs=outputs,
                mem=mem,
            )
        except Exception as exc:
            row = {
                "B": B,
                "success_count": 0,
                "error_count": B,
                "error": str(exc).splitlines()[0],
                "prompt_lengths": [len(p) for p in prompts],
                "peak_alloc_gib": round_or_none(torch_peak_alloc_gib()),
                "peak_reserved_gib": round_or_none(torch_peak_reserved_gib()),
            }
        row.update(fingerprint_fields)
        row["green"] = row_green(row, args)
        rows.append(row)
        print_row(args.engine, args, row)
        if row.get("error") and args.stop_on_failure:
            break

    engine_info = {
        "vllm_version": vllm.__version__,
        "gpu_memory_utilization": args.vllm_gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "max_model_len": max_model_len,
        "prefix_caching": False,
        "mm_note": mm_note,
    }
    with contextlib.suppress(Exception):
        del llm
    cleanup_cuda()
    return rows, engine_info


def run_sglang(
    args,
    tok,
    prompts_by_b: dict[int, list[list[int]]],
    monitor: VramMonitor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import sglang as sgl

    kwargs = {
        "model_path": args.model_path,
        "mem_fraction_static": args.sglang_mem_fraction,
        "context_length": args.sglang_context_length,
        "max_total_tokens": args.sglang_max_total_tokens,
        "disable_radix_cache": True,
        "enable_multimodal": False,
        "log_level": args.sglang_log_level,
        "max_running_requests": args.sglang_max_running_requests,
        "cuda_graph_backend_decode": args.sglang_decode_graph,
        "cuda_graph_backend_prefill": args.sglang_prefill_graph,
    }
    if args.sglang_attention_backend:
        kwargs["attention_backend"] = args.sglang_attention_backend

    engine = sgl.Engine(**kwargs)
    sp1 = {"temperature": 0.0, "max_new_tokens": 1, "ignore_eos": True}
    spn = {
        "temperature": 0.0,
        "max_new_tokens": args.out,
        "ignore_eos": True,
    }

    if args.warmup:
        with contextlib.suppress(Exception):
            engine.generate(
                input_ids=[next(iter(prompts_by_b.values()))[0][:128]],
                sampling_params=sp1,
            )

    rows: list[dict[str, Any]] = []
    for B, prompts in prompts_by_b.items():
        fingerprint_fields = prompt_fingerprint_row_fields(
            prompt_set_fingerprint(
                prompts,
                prompt_token_source=prompt_token_source(args),
            )
        )
        try:
            synchronize_cuda()
            t0 = time.perf_counter()
            first = engine.generate(input_ids=prompts, sampling_params=sp1)
            synchronize_cuda()
            t_first = time.perf_counter() - t0
            normalize_sglang_outputs(first, B)

            synchronize_cuda()
            t0 = time.perf_counter()
            full = engine.generate(input_ids=prompts, sampling_params=spn)
            synchronize_cuda()
            t_full = time.perf_counter() - t0
            outputs = normalize_sglang_outputs(full, B)
            row = make_row(
                B=B,
                prompt_lens=[len(p) for p in prompts],
                first_wall_s=t_first,
                full_wall_s=t_full,
                outputs=outputs,
                mem=monitor_memory_row(monitor),
            )
        except Exception as exc:
            row = {
                "B": B,
                "success_count": 0,
                "error_count": B,
                "error": str(exc).splitlines()[0],
                "prompt_lengths": [len(p) for p in prompts],
            }
        row.update(fingerprint_fields)
        row["green"] = row_green(row, args)
        rows.append(row)
        print_row(args.engine, args, row)
        if row.get("error") and args.stop_on_failure:
            break

    engine_info = {
        "sglang_version": getattr(sgl, "__version__", "unknown"),
        "mem_fraction_static": args.sglang_mem_fraction,
        "context_length": args.sglang_context_length,
        "max_total_tokens": args.sglang_max_total_tokens,
        "attention_backend": args.sglang_attention_backend or "default",
        "disable_radix_cache": True,
        "max_running_requests": args.sglang_max_running_requests,
        "cuda_graph_backend_decode": args.sglang_decode_graph,
        "cuda_graph_backend_prefill": args.sglang_prefill_graph,
    }
    with contextlib.suppress(Exception):
        engine.shutdown()
    cleanup_cuda()
    return rows, engine_info


def print_row(engine: str, args, row: dict[str, Any]) -> None:
    print(
        f"[{engine} ctx={args.ctx} out={args.out} B={row['B']}] "
        f"success={row.get('success_count')}/{row['B']} "
        f"p50={row.get('p50_latency_s')}s p95={row.get('p95_latency_s')}s "
        f"agg={row.get('agg_decode_tok_s')}tok/s "
        f"reserved={row.get('peak_reserved_gib')}GiB "
        f"engine_delta={row.get('peak_engine_delta_gib')}GiB "
        f"green={row.get('green')}"
    )


def build_payload(args, rows: list[dict[str, Any]], engine_info: dict[str, Any], monitor: VramMonitor) -> dict[str, Any]:
    for row in rows:
        if row.get("peak_engine_delta_gib") is None:
            row["peak_engine_delta_gib"] = round_or_none(monitor.engine_peak_delta_gib)
            row["baseline_gpu_mib"] = monitor.baseline_mib
            row["peak_gpu_mib"] = monitor.peak_mib
            row["green"] = row_green(row, args)
    return {
        "schema": "wkvm.incumbent_gemma_bench.v1",
        "engine": args.engine,
        "context_tokens_per_session": args.ctx,
        "prompt_lengths_mode": args.prompt_lengths,
        "decode_tokens_per_session": args.out,
        "mem_cap_gib": args.mem_cap_gib,
        "headroom_gib": args.headroom_gib,
        "model_path": args.model_path,
        "prompt_token_source": prompt_token_source(args),
        "uses_hf_tokenizer": uses_hf_tokenizer(args),
        "dtype": "bfloat16",
        "git_commit": git_commit(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "engine_config": engine_info,
        "memory": {
            "baseline_gpu_mib": monitor.baseline_mib,
            "peak_gpu_mib": monitor.peak_mib,
            "peak_engine_delta_gib": round_or_none(monitor.engine_peak_delta_gib),
            "green_note": (
                "green uses peak_engine_delta_gib when available; this is "
                "whole-GPU peak minus whole-GPU baseline sampled before engine load"
            ),
        },
        "summary": {
            "bmax_green": max((r["B"] for r in rows if r.get("green")), default=0),
            "max_success_B": max(
                (r["B"] for r in rows if r.get("success_count") == r["B"]),
                default=0,
            ),
            "best_green_agg_decode_tok_s": max(
                (r.get("agg_decode_tok_s") or 0.0 for r in rows if r.get("green")),
                default=0.0,
            ),
        },
        "rows": rows,
    }


def run(args) -> dict[str, Any]:
    args.model_path = resolve_model_path(args.model_path)
    if getattr(args, "synthetic_prompts", False):
        tok = SyntheticBenchTokenizer(vocab_size=args.synthetic_vocab_size)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model_path)
    prompts_by_b = {
        B: [
            build_prompt(tok, n, row)
            for row, n in enumerate(bench_prompt_lengths(args.ctx, B, args.prompt_lengths))
        ]
        for B in args.concurrency
    }

    runners = {
        "vllm": run_vllm,
        "sglang": run_sglang,
    }
    with VramMonitor(interval_s=args.mem_sample_interval_s) as monitor:
        rows, engine_info = runners[args.engine](args, tok, prompts_by_b, monitor)
    payload = build_payload(args, rows, engine_info, monitor)
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency("1,2,4,8"))
    ap.add_argument("--prompt-lengths", choices=["staggered", "uniform"], default="staggered")
    ap.add_argument(
        "--synthetic-prompts",
        action="store_true",
        help=(
            "Generate deterministic prompt token IDs locally instead of loading "
            "a Hugging Face tokenizer. The prompt IDs are passed directly to "
            "the incumbent engine."
        ),
    )
    ap.add_argument(
        "--synthetic-vocab-size",
        type=int,
        default=262_144,
        help="Vocabulary size used by --synthetic-prompts.",
    )
    ap.add_argument("--mem-cap-gib", type=float, default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)))
    ap.add_argument("--headroom-gib", type=float, default=1.0)
    ap.add_argument("--mem-sample-interval-s", type=float, default=0.1)
    ap.add_argument("--json", default=None)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stop-on-failure", action="store_true")

    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.82)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")

    ap.add_argument("--sglang-mem-fraction", type=float, default=0.88)
    ap.add_argument("--sglang-context-length", type=int, default=None)
    ap.add_argument("--sglang-max-total-tokens", type=int, default=None)
    ap.add_argument("--sglang-attention-backend", default="triton")
    ap.add_argument("--sglang-max-running-requests", type=int, default=64)
    ap.add_argument("--sglang-decode-graph", default="full")
    ap.add_argument("--sglang-prefill-graph", default="disabled")
    ap.add_argument("--sglang-log-level", default="warning")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
