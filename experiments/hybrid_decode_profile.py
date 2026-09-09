"""Where does a hybrid decode step spend its time? (H5 diagnostic)

Builds the Qwen3.5-9B engine, admits B replicated requests, prefills them,
then times pure decode steps three ways: wall per step, a component split
(gather / model forward / scatter / sampling glue) and a torch.profiler
table of the top CUDA kernels and CPU ops. Run with WKVM_KERNELS=fla|torch.

  PYTHONPATH=. python experiments/hybrid_decode_profile.py --batch 1 16
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/x/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--prompt-tokens", type=int, default=320)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cuda-graphs", action="store_true", help="capture/replay decode forwards (H5)")
    args = ap.parse_args()

    from wkvm.runner.kernels import select_kernels

    kernels = select_kernels()
    from wkvm.core.config import SchedulerConfig
    from wkvm.core.request import Request
    from wkvm.engine import Engine

    slots = max(args.batch)
    engine = Engine.from_qwen35(
        args.model, num_slots=slots, device=args.device, prefill_chunk=512,
        scheduler_config=SchedulerConfig(max_tokens_per_step=8192, max_running_requests=slots,
                                         max_tokens_per_request_per_step=512),
        cuda_graphs=args.cuda_graphs,
    )
    dev = torch.device(args.device)
    gen = torch.Generator().manual_seed(0)
    prompt = torch.randint(1000, 100000, (args.prompt_tokens,), generator=gen).tolist()
    report = {"kernels": kernels, "cuda_graphs": args.cuda_graphs, "prompt_tokens": args.prompt_tokens, "rows": []}

    def pure_decode() -> bool:
        s = engine.scheduler
        return not s.waiting and all(r.num_scheduled_gap == 1 for r in s.running)

    for b in args.batch:
        reqs = [Request(prompt_token_ids=list(prompt), max_new_tokens=64) for _ in range(b)]
        for r in reqs:
            engine.add_request(r)
        while not pure_decode():
            engine.step()
        engine.step()  # first pure-decode step: graph capture (if enabled) happens here, not in the timing
        torch.cuda.synchronize(dev)
        # 1) wall per step
        times = []
        for _ in range(args.steps):
            t = time.time(); engine.step(); torch.cuda.synchronize(dev); times.append(time.time() - t)
        wall_ms = sum(times) / len(times) * 1000
        # 2) component split on the runner (same work as engine.step minus scheduling)
        runner, bank = engine.runner, engine.bank
        slot_batch = [r.slots for r in engine.scheduler.running]
        last = [r.output_token_ids[-1] for r in engine.scheduler.running]
        comp = {"gather": 0.0, "forward": 0.0, "scatter": 0.0}
        graph_ms = None
        if runner.graphs is not None:
            g = runner.graphs
            lens = [bank.slot_len(s) for s in slot_batch]
            key = g.bucket_for(len(slot_batch), max(lens))
            if key is not None:
                graph, cache, _ = g._graphs[key]
                with torch.inference_mode():  # captured tensors are inference tensors
                    torch.cuda.synchronize(dev); t = time.time()
                    for _ in range(args.steps):
                        graph.replay()
                    torch.cuda.synchronize(dev); graph_ms = (time.time() - t) / args.steps * 1000
        with torch.inference_mode():
            for _ in range(args.steps):
                torch.cuda.synchronize(dev); t = time.time()
                cache = bank.gather(slot_batch, new_tokens=1)
                torch.cuda.synchronize(dev); comp["gather"] += time.time() - t; t = time.time()
                ids = torch.tensor(last, device=dev).unsqueeze(1)
                logits = runner._forward(ids, cache)
                torch.cuda.synchronize(dev); comp["forward"] += time.time() - t; t = time.time()
                bank.scatter(slot_batch, cache)
                torch.cuda.synchronize(dev); comp["scatter"] += time.time() - t
                # undo the length advance so the next iteration repeats the same step
                for s in slot_batch:
                    bank.guest_len[s["gdn_state"]] -= 1
        comp = {k: v / args.steps * 1000 for k, v in comp.items()}
        # 3) profiler on the forward only
        from torch.profiler import ProfilerActivity, profile

        cache = bank.gather(slot_batch, new_tokens=1)
        ids = torch.tensor(last, device=dev).unsqueeze(1)
        with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                runner._forward(ids, cache)
            torch.cuda.synchronize(dev)
        ka = prof.key_averages()
        cuda_total = sum(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) for e in ka
                         if e.key.startswith(("void", "ampere", "sm80", "Cutlass", "cutlass", "triton", "_")) ) / 3 / 1000
        top = sorted(ka, key=lambda e: -getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)))[:12]
        top_rows = [{"op": e.key[:70], "cuda_ms": getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) / 3 / 1000,
                     "cpu_ms": e.cpu_time_total / 3 / 1000, "calls": e.count // 3} for e in top]
        n_kernels = sum(e.count for e in ka if getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)) > 0) // 3
        row = {"B": b, "wall_step_ms": wall_ms, "tok_s": b / (wall_ms / 1000), "components_ms": comp,
               "graph_replay_ms": graph_ms, "eager_forward_cuda_kernel_ms_est": cuda_total,
               "eager_kernel_launches_per_step": n_kernels, "top": top_rows}
        if runner.graphs is not None:
            row["graph_stats"] = dict(runner.graphs.stats)
        report["rows"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "top"}, indent=1), flush=True)
        for r in top_rows:
            print(f"   {r['cuda_ms']:7.2f} ms cuda {r['cpu_ms']:7.2f} ms cpu x{r['calls']:4d}  {r['op']}")
        for r in reqs:
            engine.abort_request(r.req_id)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    print("PROFILE_OK")


if __name__ == "__main__":
    main()
