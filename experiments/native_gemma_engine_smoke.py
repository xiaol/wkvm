#!/usr/bin/env python
"""Scheduler smoke for the native Gemma routed-span engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from native_gemma_smoke import break_mask_for, filler_ids, load_model, resolve_model_path

from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request
from wkvm.gemma_engine import GemmaNativeEngine
from wkvm.models.gemma import gemma4_e4b_routed_span_config


def build_prompt(tok, target_len: int, row: int) -> list[int]:
    prefix = [tok.bos_token_id] if tok.bos_token_id is not None else []
    header = (
        f"Session {row} ledger. The routing color is cobalt-{row}. "
        f"The city marker is Samarkand-{row}. The desk object is lantern-{row}. "
    )
    question = "\nContinue the ledger in one short factual sentence."
    fixed = tok(header + question, add_special_tokens=False).input_ids
    budget = target_len - len(prefix) - len(fixed)
    if budget < 16:
        raise ValueError(f"ctx target {target_len} too small for engine smoke")
    before = min(64 + row * 3, budget // 3)
    return prefix + filler_ids(tok, before) + fixed + filler_ids(tok, budget - before)


def prompt_lengths(ctx: int, concurrency: int) -> list[int]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    stride = 7
    floor = max(64, ctx - stride * (concurrency - 1))
    if floor < 64:
        raise ValueError("ctx too small for distinct-history smoke")
    return [ctx - stride * i for i in range(concurrency)]


def chunked_scheduler_config(
    prompts: list[list[int]],
    *,
    slots: int,
    token_budget: int | None,
    chunk: int,
    require_full_prefill_budget: bool = False,
) -> SchedulerConfig:
    if not prompts:
        raise ValueError("prompts must not be empty")
    if slots < 1:
        raise ValueError("slots must be >= 1")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")
    total_prompt = sum(len(p) for p in prompts)
    max_prompt = max(len(p) for p in prompts)
    budget = token_budget or total_prompt
    if require_full_prefill_budget and budget < total_prompt:
        raise ValueError("token budget must admit all full prefills for the N3 smoke")
    return SchedulerConfig(
        max_tokens_per_step=max(budget, len(prompts)),
        max_running_requests=slots,
        max_tokens_per_request_per_step=max(1, min(max_prompt, budget, chunk)),
    )


def run(args) -> None:
    import torch
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model_path)
    model = load_model(path, args.device, args.attn)
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

    lengths = prompt_lengths(args.ctx, args.concurrency)
    prompts = [build_prompt(tok, n, i) for i, n in enumerate(lengths)]
    slots = args.slots or args.concurrency
    sched_cfg = chunked_scheduler_config(
        prompts,
        slots=slots,
        token_budget=args.token_budget,
        chunk=args.chunk,
        require_full_prefill_budget=True,
    )
    engine = GemmaNativeEngine(
        model,
        cfg,
        num_slots=slots,
        scheduler_config=sched_cfg,
        prefill_chunk=args.chunk,
        decode_microbatch_rows=args.decode_microbatch_rows,
        decode_microbatch_bytes=args.decode_microbatch_bytes,
        decode_batch_planner=args.decode_batch_planner,
    )

    reqs = [
        Request(prompt_token_ids=prompt, max_new_tokens=args.out, req_id=f"gemma-{i}")
        for i, prompt in enumerate(prompts)
    ]
    start = time.perf_counter()
    for req, prompt in zip(reqs, prompts):
        engine.add_request(req, break_mask=break_mask_for(tok, prompt))

    finished: list[Request] = []
    while engine.has_unfinished:
        finished.extend(engine.step())
        if engine.metrics.steps > args.max_steps:
            raise RuntimeError("native Gemma engine smoke did not converge")
    elapsed = time.perf_counter() - start

    success = [
        req
        for req in reqs
        if req.status.is_finished and len(req.output_token_ids) == req.max_new_tokens
    ]
    errors = len(reqs) - len(success) + engine.metrics.error_count
    stats = engine.stats()
    stats.update(
        {
            "engine": "wkvm-native-gemma",
            "success_count": len(success),
            "error_count": errors,
            "requested_concurrency": args.concurrency,
            "slots": slots,
            "prompt_lengths": [len(p) for p in prompts],
            "output_tokens_per_request": args.out,
            "elapsed_s": round(elapsed, 3),
        }
    )
    print(json.dumps(stats, sort_keys=True))
    print(f"success_count={len(success)}")
    print(f"error_count={errors}")
    print(f"queue_depth={stats['queue_depth']}")
    print(f"runnable_rows={stats['runnable_rows']}")
    print(f"resident_state_slots={stats['resident_state_slots']}")
    print(f"backpressure_decisions={stats['backpressure_reasons']}")
    print(f"retraction_events={stats['retraction_events']}")
    print(f"decode_batches={stats['decode_batches']}")
    print(f"max_decode_batch_rows={stats['max_decode_batch_rows']}")
    print(f"distinct_history_decode_batches={stats['distinct_history_decode_batches']}")

    if errors:
        raise SystemExit("native Gemma engine smoke had failed requests")
    if stats["max_resident_state_slots"] < min(args.concurrency, slots):
        raise SystemExit("scheduler/arena did not admit expected resident slots")
    if stats["distinct_history_decode_batches"] < 1:
        raise SystemExit("no distinct-history decode scheduler batch observed")
    print("NATIVE_ENGINE_SMOKE_OK")
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--decode-microbatch-rows", type=int, default=16)
    ap.add_argument("--decode-microbatch-bytes", type=int, default=None)
    ap.add_argument(
        "--decode-batch-planner",
        choices=["scheduler", "length_bucketed"],
        default="scheduler",
    )
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=10_000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
