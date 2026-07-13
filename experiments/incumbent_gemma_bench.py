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
    GENERATED_OUTPUT_FINGERPRINT_SCHEMA,
    SyntheticBenchTokenizer,
    generated_output_fingerprint,
    generated_output_fingerprint_row_fields,
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


def sglang_language_model_override(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise ValueError(f"{config_path}: text_config must be an object")
    override = dict(text_config)
    if text_config.get("global_head_dim") is not None:
        override["swa_head_dim"] = text_config["head_dim"]
        override["swa_v_head_dim"] = text_config["head_dim"]
        override["head_dim"] = text_config["global_head_dim"]
        override["v_head_dim"] = text_config["global_head_dim"]
    if text_config.get("num_global_key_value_heads") is not None:
        override["swa_num_key_value_heads"] = text_config["num_key_value_heads"]
        override["num_key_value_heads"] = text_config[
            "num_global_key_value_heads"
        ]
    override["architectures"] = ["Gemma4ForCausalLM"]
    return override


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
        self.sample_count = 0
        self.query_error_count = 0
        self.first_error: str | None = None
        self._process: subprocess.Popen[str] | None = None

    def _record_samples(self, output: str) -> None:
        for raw in output.splitlines():
            value = raw.strip()
            if not value:
                continue
            try:
                used = int(value)
            except ValueError:
                self.query_error_count += 1
                if self.first_error is None:
                    self.first_error = f"unexpected nvidia-smi sample {value!r}"
                continue
            self.sample_count += 1
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)

    def __enter__(self):
        self.baseline_mib = gpu_mem_used_mib()
        self.peak_mib = self.baseline_mib
        interval_ms = max(1, int(round(self.interval_s * 1000)))
        try:
            self._process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self.query_error_count += 1
            self.first_error = str(exc).splitlines()[0]
        return self

    def __exit__(self, *exc) -> None:
        process = self._process
        if process is not None:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
            self._record_samples(stdout)
            if process.returncode not in (0, -15) and stderr.strip():
                self.query_error_count += 1
                if self.first_error is None:
                    self.first_error = stderr.strip().splitlines()[0]
        final = gpu_mem_used_mib()
        if final is not None:
            self.sample_count += 1
            self.peak_mib = final if self.peak_mib is None else max(self.peak_mib, final)

    @property
    def engine_peak_delta_gib(self) -> float | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0, self.peak_mib - self.baseline_mib) / 1024.0

    def result(self) -> dict[str, Any]:
        peak_delta_mib = None
        if self.baseline_mib is not None and self.peak_mib is not None:
            peak_delta_mib = max(0, self.peak_mib - self.baseline_mib)
        return {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": "0",
            "device_index": 0,
            "device_uuid": None,
            "sample_interval_s": self.interval_s,
            "sample_count": self.sample_count,
            "baseline_used_mib": self.baseline_mib,
            "peak_used_mib": self.peak_mib,
            "peak_delta_mib": peak_delta_mib,
            "query_error_count": self.query_error_count,
            "error": self.first_error,
        }


def monitor_memory_row(monitor: VramMonitor) -> dict[str, Any]:
    return {
        "baseline_gpu_mib": monitor.baseline_mib,
        "peak_gpu_mib": monitor.peak_mib,
        "peak_engine_delta_gib": round_or_none(monitor.engine_peak_delta_gib),
        "gpu_memory": monitor.result(),
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


def vllm_request_metrics_timing(
    outputs: list[Any],
    expected: int,
) -> dict[str, Any] | None:
    """Return batch-wide decode timing from one vLLM generation run."""

    if len(outputs) != expected:
        return None
    first_token_timestamps: list[float] = []
    last_token_timestamps: list[float] = []
    first_token_latencies: list[float] = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        first_token_ts = getattr(metrics, "first_token_ts", None)
        last_token_ts = getattr(metrics, "last_token_ts", None)
        if (
            not isinstance(first_token_ts, (int, float))
            or isinstance(first_token_ts, bool)
            or not math.isfinite(first_token_ts)
            or first_token_ts <= 0
            or not isinstance(last_token_ts, (int, float))
            or isinstance(last_token_ts, bool)
            or not math.isfinite(last_token_ts)
            or last_token_ts < first_token_ts
        ):
            return None
        first_token_timestamps.append(float(first_token_ts))
        last_token_timestamps.append(float(last_token_ts))
        first_token_latency = getattr(metrics, "first_token_latency", None)
        if (
            isinstance(first_token_latency, (int, float))
            and not isinstance(first_token_latency, bool)
            and math.isfinite(first_token_latency)
            and first_token_latency >= 0
        ):
            first_token_latencies.append(float(first_token_latency))

    return {
        "prefill_plus_first_s": (
            min(first_token_latencies)
            if len(first_token_latencies) == expected
            else None
        ),
        "decode_seconds": max(last_token_timestamps) - min(first_token_timestamps),
        "decode_timing_method": "same_run_request_metrics",
        "decode_timing_source": (
            "RequestOutput.metrics.first_token_ts/last_token_ts"
        ),
        "decode_timing_comparable": True,
        "decode_timing_request_count": expected,
        "decode_timing_note": (
            "Batch interval is earliest first token to latest last token from "
            "the measured max_tokens=N run."
        ),
    }


def measure_vllm_generation(
    llm: Any,
    reqs: list[dict[str, Any]],
    sp1: Any,
    spn: Any,
) -> tuple[list[Any], float, dict[str, Any]]:
    """Measure vLLM, preferring exact timestamps from the full generation."""

    synchronize_cuda()
    started = time.perf_counter()
    full = llm.generate(reqs, spn, use_tqdm=False)
    synchronize_cuda()
    full_wall_s = time.perf_counter() - started

    timing = vllm_request_metrics_timing(full, len(reqs))
    if timing is not None:
        return full, full_wall_s, timing

    synchronize_cuda()
    started = time.perf_counter()
    llm.generate(reqs, sp1, use_tqdm=False)
    synchronize_cuda()
    first_wall_s = time.perf_counter() - started
    return full, full_wall_s, {
        "prefill_plus_first_s": first_wall_s,
        "decode_seconds": max(full_wall_s - first_wall_s, 0.0),
        "decode_timing_method": "separate_run_subtraction",
        "decode_timing_source": (
            "max_tokens=N wall time minus a separate max_tokens=1 wall time"
        ),
        "decode_timing_comparable": False,
        "decode_timing_request_count": 0,
        "decode_timing_note": (
            "Fallback only: separate-run subtraction is not directly comparable "
            "to same-run first-token-to-finish timing."
        ),
    }


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
    first_wall_s: float | None,
    full_wall_s: float,
    outputs: list[list[int]],
    mem: dict[str, Any],
    error: str | None = None,
    decode_seconds: float | None = None,
    decode_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    success_count = sum(1 for ids in outputs if ids)
    latencies = [full_wall_s] * success_count
    output_tokens = sum(len(ids) for ids in outputs)
    decode_tokens = sum(max(0, len(ids) - 1) for ids in outputs)
    decode_s = (
        max(full_wall_s - first_wall_s, 0.0)
        if decode_seconds is None and first_wall_s is not None
        else decode_seconds
    )
    timing_fields = decode_timing or {
        "decode_timing_method": "separate_run_subtraction",
        "decode_timing_source": (
            "max_tokens=N wall time minus a separate max_tokens=1 wall time"
        ),
        "decode_timing_comparable": False,
        "decode_timing_request_count": 0,
        "decode_timing_note": (
            "Separate-run subtraction is not directly comparable to same-run "
            "first-token-to-finish timing."
        ),
    }
    timing_fields = {
        key: value
        for key, value in timing_fields.items()
        if key not in {"prefill_plus_first_s", "decode_seconds"}
    }
    row: dict[str, Any] = {
        "B": B,
        "success_count": success_count,
        "error_count": B - success_count + (1 if error else 0),
        "p50_latency_s": round_or_none(statistics.median(latencies) if latencies else None),
        "p95_latency_s": round_or_none(percentile(latencies, 0.95)),
        "prefill_plus_first_s": round_or_none(first_wall_s),
        "decode_seconds": round_or_none(decode_s),
        "agg_decode_tok_s": round_or_none(
            decode_tokens / decode_s
            if decode_s is not None and decode_s > 0
            else None
        ),
        "e2e_output_tok_s": round_or_none(
            output_tokens / full_wall_s if full_wall_s > 0 else None
        ),
        "elapsed_s": round_or_none(full_wall_s),
        "prompt_lengths": prompt_lens,
        "output_token_counts": [len(ids) for ids in outputs],
        "error": error,
        **timing_fields,
        **mem,
    }
    if success_count == B:
        row.update(
            generated_output_fingerprint_row_fields(
                generated_output_fingerprint(
                    (f"bench-{B}-{row_index}", token_ids)
                    for row_index, token_ids in enumerate(outputs)
                )
            )
        )
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
        "max_num_seqs": max(prompts_by_b),
        "gpu_memory_utilization": args.vllm_gpu_mem_util,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "swap_space": 0,
        "disable_log_stats": False,
        "dtype": "bfloat16",
    }
    if args.vllm_language_model_only:
        kwargs["language_model_only"] = True
    compilation_config = None
    if args.vllm_disable_inductor:
        capture_sizes = sorted({1, 2, 4, max(prompts_by_b)})
        compilation_config = {
            "mode": 0,
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": max(capture_sizes),
        }
        kwargs["compilation_config"] = compilation_config
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
            full, t_full, timing = measure_vllm_generation(llm, reqs, sp1, spn)
            outputs = normalize_vllm_outputs(full, B)
            mem = {
                "peak_alloc_gib": round_or_none(torch_peak_alloc_gib()),
                "peak_reserved_gib": round_or_none(torch_peak_reserved_gib()),
                **monitor_memory_row(monitor),
            }
            row = make_row(
                B=B,
                prompt_lens=[len(p) for p in prompts],
                first_wall_s=timing["prefill_plus_first_s"],
                full_wall_s=t_full,
                outputs=outputs,
                mem=mem,
                decode_seconds=timing["decode_seconds"],
                decode_timing=timing,
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
        "max_num_seqs": max(prompts_by_b),
        "prefix_caching": False,
        "request_metrics_enabled": True,
        "decode_timing_preferred_method": "same_run_request_metrics",
        "decode_timing_fallback_comparable": False,
        "language_model_only": args.vllm_language_model_only,
        "compilation_config": compilation_config,
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
    model_override = None
    if args.sglang_language_model_only:
        model_override = sglang_language_model_override(args.model_path)
        kwargs["json_model_override_args"] = json.dumps(model_override)

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
        "language_model_only": args.sglang_language_model_only,
        "model_override_architectures": (
            None if model_override is None else model_override["architectures"]
        ),
        "model_override_model_type": (
            None if model_override is None else model_override.get("model_type")
        ),
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


def generated_output_fingerprint_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_rows = [
        row
        for row in rows
        if row.get("success_count") == row.get("B")
        and row.get("error_count") == 0
    ]
    by_batch: dict[str, dict[str, object]] = {}
    for row in successful_rows:
        fingerprint = row.get("generated_output_fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        if fingerprint.get("schema") != GENERATED_OUTPUT_FINGERPRINT_SCHEMA:
            continue
        digest = fingerprint.get("request_output_token_ids_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            continue
        by_batch[str(row["B"])] = dict(fingerprint)
    return {
        "successful_rows": len(successful_rows),
        "fingerprinted_successful_rows": len(by_batch),
        "complete": len(by_batch) == len(successful_rows),
        "by_batch": by_batch,
    }


def build_payload(args, rows: list[dict[str, Any]], engine_info: dict[str, Any], monitor: VramMonitor) -> dict[str, Any]:
    gpu_memory = monitor.result()
    for row in rows:
        row["peak_engine_delta_gib"] = round_or_none(monitor.engine_peak_delta_gib)
        row["baseline_gpu_mib"] = monitor.baseline_mib
        row["peak_gpu_mib"] = monitor.peak_mib
        row["gpu_memory"] = dict(gpu_memory)
        row["green"] = row_green(row, args)
    output_fingerprints = generated_output_fingerprint_summary(rows)
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
        "generated_output_fingerprint_schema": (
            GENERATED_OUTPUT_FINGERPRINT_SCHEMA
        ),
        "generated_output_fingerprint_coverage": {
            key: output_fingerprints[key]
            for key in (
                "successful_rows",
                "fingerprinted_successful_rows",
                "complete",
            )
        },
        "generated_output_fingerprints_by_batch": output_fingerprints[
            "by_batch"
        ],
        "engine_config": engine_info,
        "memory": {
            "baseline_gpu_mib": monitor.baseline_mib,
            "peak_gpu_mib": monitor.peak_mib,
            "peak_engine_delta_gib": round_or_none(monitor.engine_peak_delta_gib),
            "gpu_memory": gpu_memory,
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
        try:
            rows, engine_info = runners[args.engine](args, tok, prompts_by_b, monitor)
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            rows = []
            for B, prompts in prompts_by_b.items():
                rows.append(
                    {
                        "B": B,
                        "success_count": 0,
                        "error_count": B,
                        "error": setup_error,
                        "prompt_lengths": [len(prompt) for prompt in prompts],
                        "green": False,
                        **prompt_fingerprint_row_fields(
                            prompt_set_fingerprint(
                                prompts,
                                prompt_token_source=prompt_token_source(args),
                            )
                        ),
                    }
                )
            engine_info = {"setup_error": setup_error}
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
    ap.add_argument("--vllm-language-model-only", action="store_true")
    ap.add_argument(
        "--vllm-disable-inductor",
        action="store_true",
        help="Disable Inductor compilation while retaining full CUDA graphs.",
    )

    ap.add_argument("--sglang-mem-fraction", type=float, default=0.88)
    ap.add_argument("--sglang-context-length", type=int, default=None)
    ap.add_argument("--sglang-max-total-tokens", type=int, default=None)
    ap.add_argument("--sglang-attention-backend", default="triton")
    ap.add_argument("--sglang-language-model-only", action="store_true")
    ap.add_argument("--sglang-max-running-requests", type=int, default=64)
    ap.add_argument("--sglang-decode-graph", default="full")
    ap.add_argument("--sglang-prefill-graph", default="disabled")
    ap.add_argument("--sglang-log-level", default="warning")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
