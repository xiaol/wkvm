#!/usr/bin/env python
"""HTTP streaming benchmark for the native wkvm Gemma server.

This benchmark measures the serving path rather than calling
``GemmaNativeEngine`` directly. It is intentionally token-id only so it can use
the same prompts as ``native_gemma_bench.py`` while recording serving metrics
that are comparable to vLLM/SGLang style harnesses: TTFT, ITL, E2E latency,
success/error counts, and output throughput. ITL is aggregated only when the
stream exposes token-exact event boundaries; ``--requests-per-row`` supports
sustained multi-wave measurements instead of a single request cohort.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
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


PROTECTED_OPENAI_REQUEST_FIELDS = frozenset(
    {
        "ignore_eos",
        "best_of",
        "max_tokens",
        "min_p",
        "model",
        "n",
        "prompt",
        "return_token_ids",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "top_k",
        "top_p",
    }
)
PROVENANCE_SCHEMA = "wkvm.serving_bench.provenance.v2"
GPU_MEMORY_SCHEMA = "wkvm.whole_gpu_memory.v1"
PROVENANCE_PACKAGES = ("wkvm", "torch", "transformers", "vllm", "sglang")


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


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a valid JSON object: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must decode to a JSON object")
    return value


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


def round_or_none(x: float | None, ndigits: int = 6) -> float | None:
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


def installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except Exception:
            versions[package] = None
    return versions


def query_nvidia_gpu(device: str) -> dict[str, Any]:
    fields = ("index", "uuid", "name", "driver_version", "memory.total", "memory.used")
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=5.0,
    )
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(
            f"nvidia-smi returned {len(rows)} GPUs for device selector {device!r}"
        )
    values = [value.strip() for value in rows[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi output for device {device!r}")
    return {
        "index": int(values[0]),
        "uuid": values[1],
        "name": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(values[4]),
        "memory_used_mib": int(values[5]),
    }


class WholeGpuMemoryMonitor:
    """Opt-in nvidia-smi sampler for the selected GPU's total used memory."""

    def __init__(self, device: str, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError("GPU memory sample interval must be > 0")
        self.device = str(device)
        self.interval_s = float(interval_s)
        self.baseline_used_mib: int | None = None
        self.peak_used_mib: int | None = None
        self.sample_count = 0
        self.query_error_count = 0
        self.first_error: str | None = None
        self.gpu: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            sample = query_nvidia_gpu(self.device)
        except Exception as exc:
            self.query_error_count += 1
            if self.first_error is None:
                self.first_error = str(exc).splitlines()[0]
            return
        used_mib = int(sample["memory_used_mib"])
        self.sample_count += 1
        self.gpu = sample
        if self.baseline_used_mib is None:
            self.baseline_used_mib = used_mib
        self.peak_used_mib = (
            used_mib
            if self.peak_used_mib is None
            else max(self.peak_used_mib, used_mib)
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "WholeGpuMemoryMonitor":
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=max(6.0, self.interval_s * 2.0))
        self._sample()

    def result(self) -> dict[str, Any]:
        peak_delta_mib = None
        if self.baseline_used_mib is not None and self.peak_used_mib is not None:
            peak_delta_mib = max(0, self.peak_used_mib - self.baseline_used_mib)
        gpu = self.gpu or {}
        return {
            "schema": GPU_MEMORY_SCHEMA,
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": self.device,
            "device_index": gpu.get("index"),
            "device_uuid": gpu.get("uuid"),
            "sample_interval_s": self.interval_s,
            "sample_count": self.sample_count,
            "baseline_used_mib": self.baseline_used_mib,
            "peak_used_mib": self.peak_used_mib,
            "peak_delta_mib": peak_delta_mib,
            "query_error_count": self.query_error_count,
            "error": self.first_error,
        }


def collect_gpu_provenance(device: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if device is None:
        return None, None
    try:
        sample = query_nvidia_gpu(str(device))
    except Exception as exc:
        return None, str(exc).splitlines()[0]
    sample.pop("memory_used_mib", None)
    sample["source"] = "nvidia-smi"
    return sample, None


def build_provenance(
    args,
    *,
    commit: str | None,
    gpu: dict[str, Any] | None,
    gpu_probe_error: str | None,
) -> dict[str, Any]:
    packages = installed_package_versions()
    engine_version = getattr(args, "engine_version", None)
    monitor_device = getattr(args, "gpu_memory_device", None)
    raw_launch_command = getattr(args, "target_server_launch_command", None)
    target_server_launch_command = (
        raw_launch_command
        if isinstance(raw_launch_command, str) and raw_launch_command.strip()
        else None
    )
    target_server_config = getattr(args, "target_server_config", None)
    if target_server_config is not None and not isinstance(target_server_config, dict):
        raise ValueError("target_server_config must be a JSON object")
    return {
        "schema": PROVENANCE_SCHEMA,
        "benchmark": {
            "git_commit": commit,
            "wkvm_package_version": packages.get("wkvm"),
        },
        "client_environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": packages,
        },
        "engine": {
            "label": args.engine,
            "version": engine_version,
            "version_source": (
                getattr(args, "engine_version_source", "operator_supplied")
                if engine_version
                else "unreported"
            ),
        },
        "target_server": {
            "launch_command": target_server_launch_command,
            "launch_command_source": (
                "operator_supplied"
                if target_server_launch_command is not None
                else "unreported"
            ),
            "config": target_server_config,
            "config_source": (
                "operator_supplied"
                if target_server_config is not None
                else "unreported"
            ),
        },
        "gpu": gpu,
        "gpu_probe_error": gpu_probe_error,
        "gpu_memory_monitor": {
            "enabled": monitor_device is not None,
            "scope": "whole_device" if monitor_device is not None else None,
            "source": "nvidia-smi" if monitor_device is not None else None,
            "device_selector": (
                str(monitor_device) if monitor_device is not None else None
            ),
            "sample_interval_s": (
                float(getattr(args, "gpu_memory_sample_interval_s", 0.1))
                if monitor_device is not None
                else None
            ),
            "caveat": (
                "includes every process on the selected GPU; baseline and peak are "
                "not process-attributed"
                if monitor_device is not None
                else None
            ),
        },
    }


def bench_prompt_lengths(ctx: int, concurrency: int, mode: str) -> list[int]:
    if mode == "staggered":
        return prompt_lengths(ctx, concurrency)
    if mode == "uniform":
        return [ctx] * concurrency
    raise ValueError(f"unknown prompt length mode: {mode}")


def prompt_token_source(args) -> str:
    return "synthetic" if getattr(args, "synthetic_prompts", False) else "hf_tokenizer"


def requests_for_row(args, concurrency: int) -> int:
    request_count = getattr(args, "requests_per_row", None)
    request_count = concurrency if request_count is None else int(request_count)
    if request_count < concurrency:
        raise ValueError("requests_per_row must be >= concurrency")
    return request_count


def build_prompts(
    args,
    *,
    row_offset: int = 0,
    request_counts: dict[int, int] | None = None,
) -> dict[int, list[list[int]]]:
    if getattr(args, "synthetic_prompts", False):
        tok = SyntheticBenchTokenizer(
            vocab_size=getattr(args, "synthetic_vocab_size", 262_144),
        )
    else:
        from transformers import AutoTokenizer

        path = resolve_model_path(args.model_path)
        tok = AutoTokenizer.from_pretrained(path)
    prompts_by_b: dict[int, list[list[int]]] = {}
    next_row = int(row_offset)
    for B in args.concurrency:
        base_lengths = bench_prompt_lengths(args.ctx, B, args.prompt_lengths)
        request_count = (
            requests_for_row(args, B)
            if request_counts is None
            else int(request_counts[B])
        )
        if request_count < 1:
            raise ValueError("prompt request counts must be >= 1")
        lengths = [base_lengths[i % len(base_lengths)] for i in range(request_count)]
        prompts_by_b[B] = [
            build_prompt(tok, n, next_row + i) for i, n in enumerate(lengths)
        ]
        next_row += request_count
    return prompts_by_b


def validate_extra_body(extra_body: dict[str, Any] | None) -> None:
    if extra_body is None:
        return
    protected = sorted(PROTECTED_OPENAI_REQUEST_FIELDS.intersection(extra_body))
    if protected:
        raise ValueError(
            "--extra-body-json cannot override benchmark-controlled fields: "
            + ", ".join(protected)
        )


def benchmark_request_id(args, kind: str, concurrency: int, index: int) -> str:
    prefix = getattr(args, "run_id", None)
    suffix = f"{kind}-{concurrency}-{index}"
    return suffix if not prefix else f"{prefix}-{suffix}"


def parse_sse_line(line: bytes | str) -> dict[str, Any] | str | None:
    text = line.decode(errors="replace").strip() if isinstance(line, bytes) else line.strip()
    if not text.startswith("data:"):
        return None
    data = text.removeprefix("data:").strip()
    if data == "[DONE]":
        return data
    return json.loads(data)


def sse_events_from_line(line: bytes) -> list[dict[str, Any] | str]:
    """Parse one raw SSE read chunk.

    ``urllib`` can yield one logical SSE line or an entire event block depending
    on buffering. Keeping this parser block-aware makes the benchmark robust
    against both wkvm's tiny server and ASGI OpenAI servers.
    """
    text = line.decode(errors="replace")
    events: list[dict[str, Any] | str] = []
    for raw in text.splitlines():
        event = parse_sse_line(raw)
        if event is not None:
            events.append(event)
    return events


def request_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode(errors="replace").strip()
    except Exception:
        body = ""
    return body or str(exc)


def stream_request_wkvm(
    *,
    url: str,
    prompt: list[int],
    max_tokens: int,
    req_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "prompt_ids": prompt,
            "max_tokens": max_tokens,
            "req_id": req_id,
            "timeout_s": timeout_s,
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/stream",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_time: float | None = None
    last_token_time: float | None = None
    token_times: list[float] = []
    output_tokens: list[int] = []
    finish_reason = None
    error = None
    saw_finish = False
    try:
        with urllib.request.urlopen(request, timeout=timeout_s + 5.0) as response:
            for line in response:
                for event in sse_events_from_line(line):
                    if not isinstance(event, dict):
                        continue
                    now = time.perf_counter()
                    event_type = event.get("type")
                    if event_type == "token":
                        if first_token_time is None:
                            first_token_time = now
                        last_token_time = now
                        token_times.append(now)
                        output_tokens.append(int(event["token"]))
                    elif event_type == "finish":
                        saw_finish = True
                        finish_reason = event.get("finish_reason")
                        error = event.get("error")
                        break
                    elif event_type == "error":
                        error = event.get("error") or "stream error"
                        break
                if finish_reason is not None or error is not None:
                    break
    except urllib.error.HTTPError as exc:
        error = request_error_body(exc)
    except Exception as exc:
        error = str(exc).splitlines()[0]
    finished = time.perf_counter()
    if error is None and not saw_finish:
        error = "stream ended without a finish event"
    if error is None and finish_reason != "length":
        error = f"unexpected finish_reason {finish_reason!r}"
    if error is None and len(output_tokens) != max_tokens:
        error = (
            f"expected exactly {max_tokens} output tokens, "
            f"received {len(output_tokens)}"
        )
    inter_token_latencies = [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]
    return {
        "req_id": req_id,
        "success": error is None,
        "finish_reason": finish_reason,
        "error": error,
        "output_tokens": len(output_tokens),
        "ttft_s": None if first_token_time is None else first_token_time - started,
        "e2e_latency_s": finished - started,
        "decode_s": None
        if first_token_time is None or last_token_time is None
        else max(0.0, last_token_time - first_token_time),
        "itl_s": inter_token_latencies,
        "itl_valid": True,
        "output_token_count_exact": True,
        "output_token_count_source": "token_events",
        "stream_token_count_sources": ["token_events"],
    }


def openai_delta_token_info(choice: dict[str, Any]) -> tuple[int, str | None, bool]:
    if choice.get("token_ids") is not None:
        count = len(choice["token_ids"])
        return count, "token_ids", count <= 1
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        tokens = logprobs.get("tokens")
        if tokens:
            count = len(tokens)
            return count, "logprobs.tokens", count <= 1
    if "text" in choice and not choice.get("finish_reason"):
        return (1, "text_chunk", False) if choice.get("text") else (0, None, True)
    return 0, None, True


def openai_delta_token_count(choice: dict[str, Any]) -> int:
    return openai_delta_token_info(choice)[0]


def stream_request_openai_completions(
    *,
    url: str,
    prompt: list[int],
    max_tokens: int,
    req_id: str,
    timeout_s: float,
    model: str,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_extra_body(extra_body)
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        body.update(extra_body)
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
            "x-request-id": req_id,
        },
        method="POST",
    )
    started = time.perf_counter()
    first_token_time: float | None = None
    last_token_time: float | None = None
    inter_token_latencies: list[float] = []
    streamed_output_tokens = 0
    usage_output_tokens: int | None = None
    stream_token_count_sources: set[str] = set()
    stream_token_timing_exact = True
    finish_reason = None
    error = None
    saw_done = False
    saw_finish = False
    try:
        with urllib.request.urlopen(request, timeout=timeout_s + 5.0) as response:
            for line in response:
                done = False
                for event in sse_events_from_line(line):
                    if event == "[DONE]":
                        saw_done = True
                        done = True
                        break
                    if not isinstance(event, dict):
                        continue
                    if event.get("error"):
                        error = json.dumps(event["error"], sort_keys=True)
                        break
                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        usage_output_tokens = int(usage["completion_tokens"])
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    choice_finish_reason = choice.get("finish_reason")
                    if choice_finish_reason is not None:
                        saw_finish = True
                        finish_reason = choice_finish_reason
                    n_tokens, count_source, timing_exact = openai_delta_token_info(choice)
                    if n_tokens <= 0:
                        continue
                    if count_source is not None:
                        stream_token_count_sources.add(count_source)
                    stream_token_timing_exact = stream_token_timing_exact and timing_exact
                    now = time.perf_counter()
                    for _ in range(n_tokens):
                        if first_token_time is None:
                            first_token_time = now
                        else:
                            assert last_token_time is not None
                            inter_token_latencies.append(now - last_token_time)
                        last_token_time = now
                        streamed_output_tokens += 1
                if done or error is not None:
                    break
    except urllib.error.HTTPError as exc:
        error = request_error_body(exc)
    except Exception as exc:
        error = str(exc).splitlines()[0]
    finished = time.perf_counter()
    exact_stream_count = bool(
        stream_token_count_sources
        and "text_chunk" not in stream_token_count_sources
    )
    if usage_output_tokens is not None:
        output_tokens = usage_output_tokens
        output_token_count_source = "usage"
        output_token_count_exact = True
        if (
            error is None
            and exact_stream_count
            and streamed_output_tokens != usage_output_tokens
        ):
            error = (
                "streamed token count disagrees with usage: "
                f"streamed={streamed_output_tokens}, usage={usage_output_tokens}"
            )
    elif stream_token_count_sources == {"text_chunk"}:
        output_tokens = streamed_output_tokens
        output_token_count_source = "text_chunks"
        output_token_count_exact = False
    elif stream_token_count_sources:
        output_tokens = streamed_output_tokens
        output_token_count_source = "+".join(sorted(stream_token_count_sources))
        output_token_count_exact = "text_chunk" not in stream_token_count_sources
    else:
        output_tokens = streamed_output_tokens
        output_token_count_source = "none"
        output_token_count_exact = output_tokens == 0
    if error is None and not output_token_count_exact:
        error = "stream has no exact output-token count"
    if error is None and not saw_finish:
        error = "stream ended without a finish event"
    if error is None and not saw_done:
        error = "stream ended without [DONE]"
    if error is None and finish_reason != "length":
        error = f"unexpected finish_reason {finish_reason!r}"
    if error is None and streamed_output_tokens == 0:
        error = "stream contained no token events"
    if error is None and output_tokens != max_tokens:
        error = (
            f"expected exactly {max_tokens} output tokens, received {output_tokens}"
        )
    itl_valid = bool(
        stream_token_timing_exact
        and streamed_output_tokens == output_tokens
        and "text_chunk" not in stream_token_count_sources
    )
    return {
        "req_id": req_id,
        "success": error is None,
        "finish_reason": finish_reason,
        "error": error,
        "output_tokens": output_tokens,
        "ttft_s": None if first_token_time is None else first_token_time - started,
        "e2e_latency_s": finished - started,
        "decode_s": None
        if first_token_time is None or last_token_time is None
        else max(0.0, last_token_time - first_token_time),
        "itl_s": inter_token_latencies,
        "itl_valid": itl_valid,
        "output_token_count_exact": output_token_count_exact,
        "output_token_count_source": output_token_count_source,
        "stream_token_count_sources": sorted(stream_token_count_sources),
    }


def stream_request(
    *,
    backend: str,
    url: str,
    prompt: list[int],
    max_tokens: int,
    req_id: str,
    timeout_s: float,
    model: str,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if backend == "wkvm":
        return stream_request_wkvm(
            url=url,
            prompt=prompt,
            max_tokens=max_tokens,
            req_id=req_id,
            timeout_s=timeout_s,
        )
    if backend == "openai-completions":
        return stream_request_openai_completions(
            url=url,
            prompt=prompt,
            max_tokens=max_tokens,
            req_id=req_id,
            timeout_s=timeout_s,
            model=model,
            extra_body=extra_body,
        )
    raise ValueError(f"unknown backend {backend!r}")


def summarize_row(B: int, results: list[dict[str, Any]], elapsed_s: float) -> dict[str, Any]:
    successes = [r for r in results if r["success"]]
    errors = [r for r in results if not r["success"]]
    ttfts = [r["ttft_s"] for r in successes if r["ttft_s"] is not None]
    e2es = [r["e2e_latency_s"] for r in successes]
    itl_valid_results = [r for r in successes if r.get("itl_valid", True)]
    itls = [lat for r in itl_valid_results for lat in r["itl_s"]]
    output_tokens = sum(int(r["output_tokens"]) for r in successes)
    request_metrics = []
    for result in sorted(results, key=lambda item: item["req_id"]):
        result_itls = result.get("itl_s", [])
        request_metrics.append(
            {
                "req_id": result["req_id"],
                "success": result["success"],
                "finish_reason": result["finish_reason"],
                "error": result["error"],
                "output_tokens": result["output_tokens"],
                "ttft_s": round_or_none(result.get("ttft_s")),
                "e2e_latency_s": round_or_none(result.get("e2e_latency_s")),
                "decode_s": round_or_none(result.get("decode_s")),
                "itl_valid": result.get("itl_valid", True),
                "itl_count": len(result_itls),
                "p50_itl_s": round_or_none(percentile(result_itls, 0.50)),
                "p95_itl_s": round_or_none(percentile(result_itls, 0.95)),
                "output_token_count_exact": result.get("output_token_count_exact", True),
                "output_token_count_source": result.get("output_token_count_source"),
            }
        )
    return {
        "B": B,
        "request_count": len(results),
        "success_count": len(successes),
        "error_count": len(errors),
        "output_tokens": output_tokens,
        "elapsed_s": round_or_none(elapsed_s, 3),
        "request_output_tok_s": round_or_none(output_tokens / elapsed_s if elapsed_s > 0 else None, 3),
        "p50_ttft_s": round_or_none(percentile(ttfts, 0.50), 6),
        "p95_ttft_s": round_or_none(percentile(ttfts, 0.95), 6),
        "p50_itl_s": round_or_none(percentile(itls, 0.50), 6),
        "p95_itl_s": round_or_none(percentile(itls, 0.95), 6),
        "p50_e2e_latency_s": round_or_none(percentile(e2es, 0.50), 6),
        "p95_e2e_latency_s": round_or_none(percentile(e2es, 0.95), 6),
        "itl_valid_request_count": len(itl_valid_results),
        "itl_sample_count": len(itls),
        "output_token_count_exact_requests": sum(
            1 for result in successes if result.get("output_token_count_exact", True)
        ),
        "output_token_count_sources": sorted(
            {
                str(result.get("output_token_count_source"))
                for result in successes
                if result.get("output_token_count_source") is not None
            }
        ),
        "errors": [
            {"req_id": r["req_id"], "error": r["error"], "finish_reason": r["finish_reason"]}
            for r in errors[:8]
        ],
        "request_metrics": request_metrics,
    }


def run_row(
    url: str,
    B: int,
    prompts: list[list[int]],
    args,
    *,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    gpu_memory_device = getattr(args, "gpu_memory_device", None)
    monitor = (
        WholeGpuMemoryMonitor(
            str(gpu_memory_device),
            float(getattr(args, "gpu_memory_sample_interval_s", 0.1)),
        )
        if gpu_memory_device is not None
        else None
    )
    monitor_context = monitor if monitor is not None else contextlib.nullcontext()
    with monitor_context:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=B) as pool:
            futs = [
                pool.submit(
                    stream_request,
                    backend=args.backend,
                    url=url,
                    prompt=prompt,
                    max_tokens=args.out,
                    req_id=benchmark_request_id(args, "serve", B, i),
                    timeout_s=args.request_timeout_s,
                    model=args.served_model,
                    extra_body=extra_body,
                )
                for i, prompt in enumerate(prompts)
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futs)]
        elapsed = time.perf_counter() - started
    row = summarize_row(B, results, elapsed)
    if monitor is not None:
        row["gpu_memory"] = monitor.result()
    row.update(
        prompt_fingerprint_row_fields(
            prompt_set_fingerprint(
                prompts,
                prompt_token_source=prompt_token_source(args),
            )
        )
    )
    print(
        f"[{args.engine} backend={args.backend} ctx={args.ctx} out={args.out} B={B}] "
        f"success={row['success_count']}/{row['request_count']} "
        f"ttft_p50={row['p50_ttft_s']}s e2e_p95={row['p95_e2e_latency_s']}s "
        f"throughput={row['request_output_tok_s']}tok/s"
    )
    return row


def run_warmup(
    url: str,
    B: int,
    prompts: list[list[int]],
    args,
    *,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if args.warmup_requests <= 0:
        return None

    count = min(args.warmup_requests, B)
    warm_prompts = prompts[:count]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        futs = [
            pool.submit(
                stream_request,
                backend=args.backend,
                url=url,
                prompt=prompt,
                max_tokens=args.warmup_output_tokens,
                req_id=benchmark_request_id(args, "warmup", B, i),
                timeout_s=args.request_timeout_s,
                model=args.served_model,
                extra_body=extra_body,
            )
            for i, prompt in enumerate(warm_prompts)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    elapsed = time.perf_counter() - started
    summary = summarize_row(count, results, elapsed)
    summary.update(
        prompt_fingerprint_row_fields(
            prompt_set_fingerprint(
                warm_prompts,
                prompt_token_source=prompt_token_source(args),
            )
        )
    )
    summary["requested_output_tokens"] = args.warmup_output_tokens
    print(
        f"[{args.engine} backend={args.backend} ctx={args.ctx} out={args.warmup_output_tokens} "
        f"B={count} warmup-for={B}] success={summary['success_count']}/{count} "
        f"elapsed={summary['elapsed_s']}s"
    )
    return summary


def run(args) -> dict[str, Any]:
    if not getattr(args, "run_id", None):
        args.run_id = uuid.uuid4().hex
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    validate_extra_body(extra_body)
    url = args.url.rstrip("/")
    commit = git_commit()
    gpu, gpu_probe_error = collect_gpu_provenance(
        getattr(args, "gpu_memory_device", None)
    )
    if getattr(args, "gpu_memory_device", None) is not None and gpu is None:
        raise RuntimeError(
            "GPU memory monitoring was requested but the selected device could not "
            f"be queried: {gpu_probe_error or 'unknown nvidia-smi error'}"
        )
    provenance = build_provenance(
        args,
        commit=commit,
        gpu=gpu,
        gpu_probe_error=gpu_probe_error,
    )
    prompts_by_b = build_prompts(args)
    measured_prompt_rows = sum(
        requests_for_row(args, concurrency) for concurrency in args.concurrency
    )
    warmup_prompts_by_b = (
        build_prompts(
            args,
            row_offset=measured_prompt_rows + args.warmup_row_offset,
            request_counts={
                concurrency: min(args.warmup_requests, concurrency)
                for concurrency in args.concurrency
            },
        )
        if args.warmup_requests > 0
        else {}
    )
    rows = []
    warmups = []
    for B in args.concurrency:
        warmup = run_warmup(
            url,
            B,
            warmup_prompts_by_b.get(B, prompts_by_b[B]),
            args,
            extra_body=extra_body,
        )
        if warmup is not None:
            warmup["B"] = B
            warmups.append(warmup)
            if warmup["error_count"] and args.stop_on_failure:
                break
        rows.append(run_row(url, B, prompts_by_b[B], args, extra_body=extra_body))
        if rows[-1]["error_count"] and args.stop_on_failure:
            break

    payload: dict[str, Any] = {
        "schema": "wkvm.serving_bench.v1",
        "engine": args.engine,
        "backend": args.backend,
        "url": url,
        "run_id": args.run_id,
        "context_tokens_per_session": args.ctx,
        "prompt_lengths_mode": args.prompt_lengths,
        "decode_tokens_per_session": args.out,
        "concurrency": args.concurrency,
        "requests_per_row": args.requests_per_row,
        "prompt_reuse_policy": "disjoint_across_measured_and_warmup_rows",
        "prompt_token_source": prompt_token_source(args),
        "uses_hf_tokenizer": prompt_token_source(args) == "hf_tokenizer",
        "synthetic_vocab_size": (
            args.synthetic_vocab_size if args.synthetic_prompts else None
        ),
        "warmup_requests": args.warmup_requests,
        "warmup_output_tokens": args.warmup_output_tokens,
        "warmup_row_offset": args.warmup_row_offset,
        "request_timeout_s": args.request_timeout_s,
        "served_model": args.served_model,
        "semantics": args.semantics,
        "sampling": {
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "max_tokens": args.out,
        },
        "extra_body": extra_body,
        "model_path": resolve_model_path(args.model_path),
        "git_commit": commit,
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "provenance": provenance,
        "warmups": warmups,
        "rows": rows,
        "summary": {
            "max_success_B": max(
                (
                    r["B"]
                    for r in rows
                    if r["success_count"] == r["request_count"]
                ),
                default=0,
            ),
            "best_output_tok_s": max(
                (r["request_output_tok_s"] or 0.0 for r in rows),
                default=0.0,
            ),
        },
    }
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--backend",
        choices=["wkvm", "openai-completions"],
        default="wkvm",
        help="HTTP API shape to benchmark.",
    )
    ap.add_argument(
        "--engine",
        default=None,
        help="Result label. Defaults to wkvm-native-http-stream or the backend name.",
    )
    ap.add_argument(
        "--engine-version",
        default=None,
        help=(
            "Target server engine version. This is operator-reported because the "
            "HTTP APIs do not expose a portable version endpoint."
        ),
    )
    ap.add_argument(
        "--engine-version-source",
        default="operator_supplied",
        help="Source label stored with --engine-version (for example package or git).",
    )
    ap.add_argument(
        "--target-server-launch-command",
        default=None,
        help=(
            "Exact operator-supplied command used to launch the target server. "
            "Recorded verbatim and required by strict comparisons for new artifacts."
        ),
    )
    ap.add_argument(
        "--target-server-config-json",
        dest="target_server_config",
        type=parse_json_object,
        default=None,
        help=(
            "Optional JSON object with target-server settings not fully expressed "
            "by its launch command."
        ),
    )
    ap.add_argument(
        "--served-model",
        default="gemma-4-E4B-it",
        help="Model name sent to OpenAI-compatible completion servers.",
    )
    ap.add_argument(
        "--semantics",
        choices=["full_kv", "routed_span_approximate", "other"],
        required=True,
        help="Attention/cache semantics served by the target engine.",
    )
    ap.add_argument(
        "--extra-body-json",
        default=None,
        help="JSON object merged into each OpenAI-compatible request body.",
    )
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency("1,2,4,8"))
    ap.add_argument("--prompt-lengths", choices=["staggered", "uniform"], default="staggered")
    ap.add_argument(
        "--requests-per-row",
        type=int,
        default=None,
        help=(
            "Total measured requests at each concurrency. Defaults to one cohort "
            "of B requests; set >= max concurrency for sustained-load percentiles."
        ),
    )
    ap.add_argument(
        "--synthetic-prompts",
        action="store_true",
        help="Use deterministic tokenizer-free prompt IDs for exact cross-engine replay.",
    )
    ap.add_argument(
        "--synthetic-vocab-size",
        type=int,
        default=262_144,
        help="Vocabulary size used by --synthetic-prompts.",
    )
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
    ap.add_argument(
        "--gpu-memory-device",
        "--monitor-gpu",
        default=None,
        help=(
            "Opt in to whole-device nvidia-smi memory sampling for each measured "
            "row; accepts a physical GPU index or UUID."
        ),
    )
    ap.add_argument(
        "--gpu-memory-sample-interval-s",
        type=float,
        default=0.1,
        help="nvidia-smi polling interval used by --gpu-memory-device.",
    )
    ap.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Untimed requests to send before each measured row, capped at that row's concurrency.",
    )
    ap.add_argument(
        "--warmup-output-tokens",
        type=int,
        default=1,
        help="Completion tokens requested by each untimed warmup request.",
    )
    ap.add_argument(
        "--warmup-row-offset",
        type=int,
        default=64,
        help="Prompt row offset for warmup prompts so prefix caches do not prime measured prompts.",
    )
    ap.add_argument("--model-path", default=None)
    ap.add_argument(
        "--run-id",
        default=None,
        help="Request-ID prefix. Defaults to a new random value for every run.",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--stop-on-failure", action="store_true")
    args = ap.parse_args()
    if args.extra_body_json is not None and not isinstance(json.loads(args.extra_body_json), dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be >= 0")
    if args.warmup_output_tokens < 1:
        raise SystemExit("--warmup-output-tokens must be >= 1")
    if args.synthetic_vocab_size < 16:
        raise SystemExit("--synthetic-vocab-size must be >= 16")
    if args.gpu_memory_sample_interval_s <= 0:
        raise SystemExit("--gpu-memory-sample-interval-s must be > 0")
    if args.requests_per_row is not None and args.requests_per_row < max(args.concurrency):
        raise SystemExit("--requests-per-row must be >= the maximum concurrency")
    if args.engine is None:
        args.engine = (
            "wkvm-native-http-stream"
            if args.backend == "wkvm"
            else f"{args.backend}-http-stream"
        )
    run(args)


if __name__ == "__main__":
    main()
