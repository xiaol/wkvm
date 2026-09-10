"""Multi-turn session benchmark: wkvm hybrid (paged or ring guests) vs vLLM.

The workload shape that produced the Gemma 10x rows: ``B`` concurrent
sessions, each starting from a ``--ctx``-token context, then ``--turns``
turns of (``--turn-in`` new user tokens -> ``--out`` generated tokens). Every
turn's prompt tokens are identical across engines (``build_prompt`` +
``filler_ids`` from the shared prompt builders); generation is greedy with
``ignore_eos`` semantics (exactly ``out`` tokens per turn).

- wkvm keeps each session parked between turns (``park_on_finish`` +
  ``continue_request``): the recurrent state and the guest window/pages stay
  in place, the new turn's tokens are the only prefill. In ring mode the
  per-session memory is constant, so the number of sessions never hits a
  capacity wall.
- vLLM is given the growing conversation each turn with prefix caching
  enabled: when the previous turns' KV is still cached it prefills only the
  new tokens; when the cache was evicted (KV pool full) it re-prefills the
  whole context. That is the incumbent's honest best case.

Outputs differ between ring mode and an exact engine beyond the window; the
comparison is throughput at a stated semantics, like the Gemma rows.

  PYTHONPATH=. python experiments/hybrid_multiturn_bench.py --engine wkvm --guest-mode ring --ring-tokens 1024 \
      --ctx 13824 --turns 8 --sessions 16 --json experiments/results/mt_wkvm_ring.json
  (vllm13-venv, compat lib) python experiments/hybrid_multiturn_bench.py --engine vllm ...
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")  # 128-core host: BLAS/torch thread fan-out on tiny host ops is a 10x slowdown
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_gemma_engine_smoke import build_prompt, prompt_lengths  # noqa: E402
from native_gemma_smoke import filler_ids  # noqa: E402

SCHEMA = "wkvm.multiturn_bench.v1"


def turn_tokens(tok, turn: int, session: int, n: int) -> list[int]:
    head = tok(f"\nUser turn {turn} for session {session}: ", add_special_tokens=False).input_ids
    tail = tok("\nContinue the ledger with the next entry.", add_special_tokens=False).input_ids
    body = filler_ids(tok, max(0, n - len(head) - len(tail)))
    return (head + body + tail)[:n]


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_wkvm(args, tok, prompts):
    import torch

    from wkvm.core.config import SchedulerConfig
    from wkvm.core.request import Request
    from wkvm.engine import Engine
    from wkvm.runner.kernels import select_kernels

    select_kernels()
    dev = torch.device(args.device)
    b = args.sessions
    total_per_session = args.ctx + args.turns * (args.turn_in + args.out) + 64
    graphs = {"batch_buckets": tuple(sorted({x for x in (1, 2, 4, 8, 16, 32) if x <= b} | {b}))}
    if args.guest_mode == "paged":
        graphs["length_buckets"] = (total_per_session,)
    engine = Engine.from_qwen35(
        args.model_path, num_slots=b, guest_pool_tokens=b * total_per_session, device=dev,
        stop_token_ids=frozenset(), prefill_chunk=args.prefill_chunk,
        scheduler_config=SchedulerConfig(max_tokens_per_step=max(8192, args.prefill_chunk * b),
                                         max_running_requests=b, max_tokens_per_request_per_step=args.prefill_chunk),
        cuda_graphs=graphs, guest_mode=args.guest_mode, sink_tokens=args.sink_tokens, ring_tokens=args.ring_tokens,
        routed_params=dict(routed_pending=args.routed_pending, routed_slots=args.routed_slots,
                           routed_reps=args.routed_reps, routed_max_span=args.routed_max_span,
                           routed_retention=args.routed_retention),
    )
    weights_gib = torch.cuda.memory_allocated(dev) / 2**30
    reqs = [Request(prompt_token_ids=list(p), max_new_tokens=args.out) for p in prompts]
    turns = []
    torch.cuda.synchronize(dev)
    t_all = time.time()
    for turn in range(args.turns):
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        if turn == 0:
            for r in reqs:
                engine.add_request(r, park_on_finish=True)
        else:
            for i, r in enumerate(reqs):
                engine.continue_request(r.req_id, turn_tokens(tok, turn, i, args.turn_in), args.out)
        first = {}
        start_out = {r.req_id: len(r.output_token_ids) for r in reqs}
        while engine.has_unfinished:
            engine.step()
            now = time.time()
            for r in reqs:
                if r.req_id not in first and len(r.output_token_ids) > start_out[r.req_id]:
                    first[r.req_id] = now
        torch.cuda.synchronize(dev)
        wall = time.time() - t0
        ttft = [first[r.req_id] - t0 for r in reqs]
        turns.append({"turn": turn, "wall_s": wall, "p50_ttft_s": percentile(ttft, 50), "p95_ttft_s": percentile(ttft, 95),
                      "output_tokens": b * args.out, "output_tok_s": b * args.out / wall,
                      "context_tokens_p50": percentile([r.num_computed_tokens for r in reqs], 50),
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in turns[-1].items()}), flush=True)
    total = time.time() - t_all
    stats = dict(engine.runner.graphs.stats) if engine.runner.graphs else None
    for r in reqs:
        engine.close_request(r.req_id)
    return turns, total, {"weights_plus_state_gib": weights_gib, "graph_stats": stats,
                          "routing_stats": dict(engine.bank.rg.stats) if engine.bank.routed else None,
                          "bytes_per_session_recurrent": engine.layout.bytes_per_slot,
                          "guest_bytes_per_session": (engine.layout.window_tokens * engine.layout.guest_bytes_per_token
                                                      if args.guest_mode != "paged" else None)}


def run_vllm(args, tok, prompts):
    import torch
    from vllm import LLM, SamplingParams

    b = args.sessions
    total_per_session = args.ctx + args.turns * (args.turn_in + args.out) + 64
    llm = LLM(model=args.model_path, dtype="bfloat16", max_model_len=total_per_session, max_num_seqs=b,
              gpu_memory_utilization=args.vllm_gpu_mem_util, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.0, max_tokens=args.out, ignore_eos=True)
    convos = [list(p) for p in prompts]
    turns = []
    dev = torch.device("cuda:0")
    t_all = time.time()
    for turn in range(args.turns):
        if turn > 0:
            for i in range(b):
                convos[i] = convos[i] + turn_tokens(tok, turn, i, args.turn_in)
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": c} for c in convos], sp, use_tqdm=False)
        wall = time.time() - t0
        ttft = []
        for i, o in enumerate(outs):
            ids = list(o.outputs[0].token_ids)
            convos[i] = convos[i] + ids
            m = getattr(o, "metrics", None)
            if m is not None and getattr(m, "first_token_time", None) and getattr(m, "arrival_time", None):
                ttft.append(m.first_token_time - m.arrival_time)
        turns.append({"turn": turn, "wall_s": wall, "p50_ttft_s": percentile(ttft, 50), "p95_ttft_s": percentile(ttft, 95),
                      "output_tokens": b * args.out, "output_tok_s": b * args.out / wall,
                      "context_tokens_p50": percentile([len(c) for c in convos], 50),
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(dev) / 2**30})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in turns[-1].items()}), flush=True)
    total = time.time() - t_all
    return turns, total, {"vllm_gpu_mem_util": args.vllm_gpu_mem_util, "prefix_caching": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("wkvm", "vllm"), default="wkvm")
    ap.add_argument("--model-path", default="/root/x/Qwen3.5-9B")
    ap.add_argument("--ctx", type=int, default=13_824)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--turn-in", type=int, default=128)
    ap.add_argument("--out", type=int, default=128)
    ap.add_argument("--sessions", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--guest-mode", choices=("paged", "ring", "routed"), default="ring")
    ap.add_argument("--sink-tokens", type=int, default=16)
    ap.add_argument("--ring-tokens", type=int, default=1024)
    ap.add_argument("--routed-pending", type=int, default=512)
    ap.add_argument("--routed-slots", type=int, default=64)
    ap.add_argument("--routed-reps", type=int, default=48)
    ap.add_argument("--routed-max-span", type=int, default=48)
    ap.add_argument("--routed-retention", choices=("novelty", "surprisal", "fps"), default="novelty")
    ap.add_argument("--vllm-gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    lengths = prompt_lengths(args.ctx, args.sessions)
    prompts = [build_prompt(tok, n, row) for row, n in enumerate(lengths)]
    turns, total, extra = (run_wkvm if args.engine == "wkvm" else run_vllm)(args, tok, prompts)
    payload = {
        "schema": SCHEMA,
        "engine": (args.engine if args.engine == "vllm" else "wkvm-hybrid-" + {
            "paged": "paged", "ring": f"ring{args.sink_tokens}+{args.ring_tokens}",
            "routed": f"routed{args.sink_tokens}+{args.ring_tokens}+p{args.routed_pending}+{args.routed_slots}x{args.routed_reps}",
        }[args.guest_mode]),
        "semantics": ("exact_full_kv" if args.engine == "vllm" or args.guest_mode == "paged"
                      else {"ring": "sink_ring_approximate", "routed": "routed_span_bank_approximate"}[args.guest_mode]),
        "model_path": args.model_path, "sessions": args.sessions, "context_tokens": args.ctx, "turns": args.turns,
        "turn_in_tokens": args.turn_in, "out_tokens": args.out, "prompt_lengths": lengths,
        "total_wall_s": total, "total_output_tokens": args.sessions * args.out * args.turns,
        "output_tok_s_overall": args.sessions * args.out * args.turns / total,
        "turns_table": turns, "git_commit": os.popen("git rev-parse --short HEAD").read().strip(),
        "launch_command": " ".join(sys.argv), **extra,
    }
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=1))
    print(f"TOTAL {args.engine}: {total:.1f}s for {payload['total_output_tokens']} output tokens "
          f"-> {payload['output_tok_s_overall']:.1f} tok/s")
    print("MULTITURN_OK")


if __name__ == "__main__":
    main()
