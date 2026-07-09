#!/usr/bin/env python
"""HTTP streaming benchmark for the native wkvm Gemma server.

This benchmark measures the serving path rather than calling
``GemmaNativeEngine`` directly. It is intentionally token-id only so it can use
the same prompts as ``native_gemma_bench.py`` while recording serving metrics
that are comparable to vLLM/SGLang style harnesses: TTFT, ITL, E2E latency,
success/error counts, and output throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


def bench_prompt_lengths(ctx: int, concurrency: int, mode: str) -> list[int]:
    if mode == "staggered":
        return prompt_lengths(ctx, concurrency)
    if mode == "uniform":
        return [ctx] * concurrency
    raise ValueError(f"unknown prompt length mode: {mode}")


def build_prompts(args, *, row_offset: int = 0) -> dict[int, list[list[int]]]:
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    tok = AutoTokenizer.from_pretrained(path)
    prompts_by_b: dict[int, list[list[int]]] = {}
    for B in args.concurrency:
        lengths = bench_prompt_lengths(args.ctx, B, args.prompt_lengths)
        prompts_by_b[B] = [
            build_prompt(tok, n, row_offset + i) for i, n in enumerate(lengths)
        ]
    return prompts_by_b


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
    inter_token_latencies = [
        token_times[i] - token_times[i - 1] for i in range(1, len(token_times))
    ]
    return {
        "req_id": req_id,
        "success": bool(error is None and len(output_tokens) == max_tokens),
        "finish_reason": finish_reason,
        "error": error,
        "output_tokens": len(output_tokens),
        "ttft_s": None if first_token_time is None else first_token_time - started,
        "e2e_latency_s": finished - started,
        "decode_s": None
        if first_token_time is None or last_token_time is None
        else max(0.0, last_token_time - first_token_time),
        "itl_s": inter_token_latencies,
    }


def openai_delta_token_count(choice: dict[str, Any]) -> int:
    if choice.get("token_ids") is not None:
        return len(choice["token_ids"])
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        tokens = logprobs.get("tokens")
        if tokens:
            return len(tokens)
    if "text" in choice and not choice.get("finish_reason"):
        return 1
    return 0


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
    finish_reason = None
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s + 5.0) as response:
            for line in response:
                done = False
                for event in sse_events_from_line(line):
                    if event == "[DONE]":
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
                    finish_reason = choice.get("finish_reason") or finish_reason
                    n_tokens = openai_delta_token_count(choice)
                    if n_tokens <= 0:
                        continue
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
    output_tokens = max(streamed_output_tokens, usage_output_tokens or 0)
    return {
        "req_id": req_id,
        "success": bool(error is None and output_tokens >= max_tokens),
        "finish_reason": finish_reason,
        "error": error,
        "output_tokens": output_tokens,
        "ttft_s": None if first_token_time is None else first_token_time - started,
        "e2e_latency_s": finished - started,
        "decode_s": None
        if first_token_time is None or last_token_time is None
        else max(0.0, last_token_time - first_token_time),
        "itl_s": inter_token_latencies,
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
    itls = [lat for r in successes for lat in r["itl_s"]]
    output_tokens = sum(int(r["output_tokens"]) for r in successes)
    return {
        "B": B,
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
        "errors": [
            {"req_id": r["req_id"], "error": r["error"], "finish_reason": r["finish_reason"]}
            for r in errors[:8]
        ],
    }


def run_row(
    url: str,
    B: int,
    prompts: list[list[int]],
    args,
    *,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=B) as pool:
        futs = [
            pool.submit(
                stream_request,
                backend=args.backend,
                url=url,
                prompt=prompt,
                max_tokens=args.out,
                req_id=f"serve-{B}-{i}",
                timeout_s=args.request_timeout_s,
                model=args.served_model,
                extra_body=extra_body,
            )
            for i, prompt in enumerate(prompts)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    elapsed = time.perf_counter() - started
    row = summarize_row(B, results, elapsed)
    row["prompt_lengths"] = [len(p) for p in prompts]
    print(
        f"[{args.engine} backend={args.backend} ctx={args.ctx} out={args.out} B={B}] "
        f"success={row['success_count']}/{B} "
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
                req_id=f"warmup-{B}-{i}",
                timeout_s=args.request_timeout_s,
                model=args.served_model,
                extra_body=extra_body,
            )
            for i, prompt in enumerate(warm_prompts)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]
    elapsed = time.perf_counter() - started
    summary = summarize_row(count, results, elapsed)
    summary["prompt_lengths"] = [len(p) for p in warm_prompts]
    summary["requested_output_tokens"] = args.warmup_output_tokens
    print(
        f"[{args.engine} backend={args.backend} ctx={args.ctx} out={args.warmup_output_tokens} "
        f"B={count} warmup-for={B}] success={summary['success_count']}/{count} "
        f"elapsed={summary['elapsed_s']}s"
    )
    return summary


def run(args) -> dict[str, Any]:
    prompts_by_b = build_prompts(args)
    warmup_prompts_by_b = (
        build_prompts(args, row_offset=args.warmup_row_offset)
        if args.warmup_requests > 0
        else {}
    )
    url = args.url.rstrip("/")
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
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
        "context_tokens_per_session": args.ctx,
        "prompt_lengths_mode": args.prompt_lengths,
        "decode_tokens_per_session": args.out,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "warmup_output_tokens": args.warmup_output_tokens,
        "warmup_row_offset": args.warmup_row_offset,
        "request_timeout_s": args.request_timeout_s,
        "served_model": args.served_model,
        "extra_body": extra_body,
        "model_path": resolve_model_path(args.model_path),
        "git_commit": git_commit(),
        "launch_command": shlex.join([sys.executable, *sys.argv]),
        "warmups": warmups,
        "rows": rows,
        "summary": {
            "max_success_B": max(
                (r["B"] for r in rows if r["success_count"] == r["B"]),
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
        "--served-model",
        default="gemma-4-E4B-it",
        help="Model name sent to OpenAI-compatible completion servers.",
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
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
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
    ap.add_argument("--json", default=None)
    ap.add_argument("--stop-on-failure", action="store_true")
    args = ap.parse_args()
    if args.extra_body_json is not None and not isinstance(json.loads(args.extra_body_json), dict):
        raise SystemExit("--extra-body-json must decode to a JSON object")
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be >= 0")
    if args.warmup_output_tokens < 1:
        raise SystemExit("--warmup-output-tokens must be >= 1")
    if args.engine is None:
        args.engine = (
            "wkvm-native-http-stream"
            if args.backend == "wkvm"
            else f"{args.backend}-http-stream"
        )
    run(args)


if __name__ == "__main__":
    main()
