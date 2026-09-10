"""Same-shape throughput ladder for the wkvm hybrid engine (Qwen3.5-9B).

The wkvm side of the vLLM/SGLang comparison: identical prompt construction to
``incumbent_gemma_bench.py`` (``build_prompt`` / ``prompt_lengths`` from
``native_gemma_engine_smoke``, HF tokenizer of the same model), identical
shape (``--ctx`` prompt tokens per session, ``--out`` decode tokens,
``--prompt-lengths staggered``), greedy, ``--ignore-eos`` semantics (every
request generates exactly ``out`` tokens), one GPU. Emits the
``wkvm.hybrid_bench.v1`` schema with the fields the benchmark contract and
``gemma_bench_report.py`` read.

Semantics: exact paged attention for the full-attention layers, exact GDN
recurrence — the same math as HF/vLLM, so this is a like-for-like row, not
the approximate routed-span comparison of the Gemma line.

  PYTHONPATH=. python experiments/hybrid_bench.py --ctx 13824 --out 128 \
      --concurrency 1,8,16 --json experiments/results/wkvm_hybrid_qwen35_ctx13824_out128_ladder.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")  # 128-core host: BLAS/torch thread fan-out on tiny host ops is a 10x slowdown
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_gemma_engine_smoke import build_prompt, prompt_lengths  # noqa: E402

SCHEMA = "wkvm.hybrid_bench.v1"


def parse_concurrency(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def fingerprint(prompts: list[list[int]]) -> str:
    h = hashlib.sha256()
    for p in prompts:
        h.update(json.dumps(p).encode())
        h.update(b"\n")
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/root/x/Qwen3.5-9B")
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--concurrency", type=parse_concurrency, default=parse_concurrency("1,8,16"))
    ap.add_argument("--prompt-lengths", choices=["staggered", "uniform"], default="staggered")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--page-tokens", type=int, default=256)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--guest-mode", choices=("paged", "ring"), default="paged")
    ap.add_argument("--sink-tokens", type=int, default=16)
    ap.add_argument("--ring-tokens", type=int, default=1024)
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mem-cap-gib", type=float, default=float(os.environ.get("WKVM_MEM_CAP_GIB", 38)))
    ap.add_argument("--headroom-gib", type=float, default=1.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from wkvm.runner.kernels import select_kernels

    kernels = select_kernels()
    from transformers import AutoTokenizer

    from wkvm.core.config import SchedulerConfig
    from wkvm.core.request import Request
    from wkvm.engine import Engine

    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    max_b = max(args.concurrency)
    lengths_by_b = {b: (prompt_lengths(args.ctx, b) if args.prompt_lengths == "staggered" else [args.ctx] * b)
                    for b in args.concurrency}
    prompts_by_b = {b: [build_prompt(tok, n, row) for row, n in enumerate(lengths_by_b[b])] for b in args.concurrency}
    per_request = args.ctx + args.out
    slots = max_b
    graphs = False if args.no_graphs else {
        "batch_buckets": tuple(sorted({b for b in (1, 2, 4, 8, 16, 32) if b <= max_b} | {max_b})),
        "length_buckets": (per_request + 64,),
    }
    torch.cuda.reset_peak_memory_stats(dev)
    base_alloc = torch.cuda.memory_allocated(dev)
    t0 = time.time()
    engine = Engine.from_qwen35(
        args.model_path, num_slots=slots, guest_pool_tokens=slots * (per_request + 64), page_tokens=args.page_tokens,
        device=dev, stop_token_ids=frozenset(), prefill_chunk=args.prefill_chunk,
        scheduler_config=SchedulerConfig(max_tokens_per_step=max(8192, args.prefill_chunk * slots),
                                         max_running_requests=slots, max_tokens_per_request_per_step=args.prefill_chunk),
        cuda_graphs=graphs, guest_mode=args.guest_mode, sink_tokens=args.sink_tokens, ring_tokens=args.ring_tokens,
    )
    load_s = time.time() - t0
    weights_gib = (torch.cuda.memory_allocated(dev) - base_alloc) / 2**30
    layout = engine.layout
    print(f"engine loaded in {load_s:.1f}s, allocated {weights_gib:.2f} GiB (weights + slots + pool), kernels={kernels}, graphs={bool(graphs)}", flush=True)

    def run_batch(prompts: list[list[int]]):
        reqs = [Request(prompt_token_ids=list(p), max_new_tokens=args.out) for p in prompts]
        first_token_at: dict[str, float] = {}
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        t_start = time.time()
        for r in reqs:
            engine.add_request(r)
        prefill_done = None
        while engine.has_unfinished:
            engine.step()
            now = time.time()
            for r in reqs:
                if r.req_id not in first_token_at and r.output_token_ids:
                    first_token_at[r.req_id] = now
            if prefill_done is None and len(first_token_at) == len(reqs):
                torch.cuda.synchronize(dev)
                prefill_done = time.time()
        torch.cuda.synchronize(dev)
        t_end = time.time()
        return reqs, t_start, t_end, first_token_at, prefill_done

    if args.warmup:
        run_batch(prompts_by_b[args.concurrency[0]][:1])
        run_batch(prompts_by_b[max_b])
        print("warmup done", flush=True)

    rows = []
    for b in args.concurrency:
        prompts = prompts_by_b[b]
        reqs, t_start, t_end, first, prefill_done = run_batch(prompts)
        wall = t_end - t_start
        out_tokens = sum(len(r.output_token_ids) for r in reqs)
        ok = sum(len(r.output_token_ids) == args.out for r in reqs)
        ttft = [first[r.req_id] - t_start for r in reqs if r.req_id in first]
        decode_wall = t_end - prefill_done if prefill_done else None
        peak_alloc = torch.cuda.max_memory_allocated(dev) / 2**30
        peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**30
        row = {
            "B": b,
            "request_count": len(reqs),
            "success_count": ok,
            "error_count": len(reqs) - ok,
            "prompt_lengths": lengths_by_b[b],
            "prompt_total_tokens": sum(len(p) for p in prompts),
            "prompt_token_ids_sha256": fingerprint(prompts),
            "prompt_fingerprint": fingerprint(prompts),
            "output_tokens": out_tokens,
            "wall_s": wall,
            "e2e_output_tok_s": out_tokens / wall,
            "agg_decode_tok_s": (b * (args.out - 1) / decode_wall) if decode_wall else None,
            "prefill_wall_s": (prefill_done - t_start) if prefill_done else None,
            "p50_ttft_s": percentile(ttft, 50),
            "p95_ttft_s": percentile(ttft, 95),
            "p50_latency_s": wall,  # all requests finish together under ignore-eos
            "p95_latency_s": wall,
            "peak_alloc_gib": peak_alloc,
            "peak_reserved_gib": peak_reserved,
            "peak_engine_delta_gib": peak_reserved - weights_gib,
            "green": peak_reserved <= args.mem_cap_gib - args.headroom_gib,
            "graph_stats": dict(engine.runner.graphs.stats) if engine.runner.graphs else None,
        }
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                          if k not in ("prompt_lengths", "prompt_fingerprint", "prompt_token_ids_sha256", "graph_stats")}), flush=True)

    payload = {
        "schema": SCHEMA,
        "engine": "wkvm-hybrid" if args.guest_mode == "paged" else f"wkvm-hybrid-ring{args.sink_tokens}+{args.ring_tokens}",
        "semantics": "exact_paged_attention_gdn" if args.guest_mode == "paged" else "sink_ring_approximate_gdn",
        "guest_mode": args.guest_mode,
        "sink_tokens": args.sink_tokens,
        "ring_tokens": args.ring_tokens,
        "model_path": args.model_path,
        "dtype": "bfloat16",
        "context_tokens_per_session": args.ctx,
        "decode_tokens_per_session": args.out,
        "prompt_lengths_mode": args.prompt_lengths,
        "prompt_token_source": "hf_tokenizer",
        "uses_hf_tokenizer": True,
        "uses_hf_config": True,
        "uses_hf_model_construction": True,
        "uses_hf_transformer_forward": True,
        "ignore_eos": True,
        "sampling": {"temperature": 0.0, "greedy": True},
        "kernels": kernels,
        "cuda_graphs": bool(graphs),
        "prefill_chunk": args.prefill_chunk,
        "page_tokens": args.page_tokens,
        "recurrent_bytes_per_slot": layout.bytes_per_slot,
        "guest_bytes_per_token": layout.guest_bytes_per_token,
        "gpu": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
        "load_s": load_s,
        "weights_plus_state_gib": weights_gib,
        "mem_cap_gib": args.mem_cap_gib,
        "headroom_gib": args.headroom_gib,
        "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "launch_command": " ".join(sys.argv),
        "rows": rows,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1))
    print("HYBRID_BENCH_OK")


if __name__ == "__main__":
    main()
