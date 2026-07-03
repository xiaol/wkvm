"""M2 engine bench: steady-state decode tok/s from the *engine loop*.

Unlike M1's runner micro-bench, every timed token here flows through
``Engine.step()``: scheduler.schedule() -> gather/forward/scatter -> batched
sample -> update_from_output. That is the honest serving number to put next
to Albatross's static-batch ladder (docs/COMPARISON.md §3).

Method: fill all ``B`` slots with requests whose ``max_new_tokens`` exceeds
warmup+steps (nobody finishes mid-measurement), drive to steady state (every
gap == 1), then time ``steps`` engine steps. tok/s = B * steps / dt.

``--graph`` additionally captures the decode-step *model forward* in a CUDA
graph per batch size (uniform batch + static state tensors make this safe)
and reports a second measurement with replay. Gather/scatter and the
scheduler stay in eager python — the graph removes the per-step python module
walk + kernel launch overhead only.

Usage:
    python experiments/m2_engine_bench.py [--model PATH] \
        [--batches 1 8 32 64 128 256] [--steps 64] [--graph]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wkvm.core.config import SchedulerConfig  # noqa: E402
from wkvm.core.request import Request  # noqa: E402
from wkvm.engine import Engine  # noqa: E402
from wkvm.models.rwkv7 import WKV_DTYPE, load_rwkv7  # noqa: E402
from wkvm.runner.runner import RWKV7Runner  # noqa: E402

DEFAULT_MODEL = "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla/rwkv7-1.5B-world"
PROMPT_LEN = 32


class GraphedDecode:
    """CUDA-graph capture of one uniform-batch decode forward.

    Static tensors: ``ids_in`` and per-layer state inputs seeded into a fresh
    fla Cache at capture time. The forward's output state tensors (whatever
    ``cache[i]`` holds after the captured call) live in the graph's private
    pool, so their addresses are stable across replays — we keep references
    and scatter from them after each replay. Warmup runs on a side stream
    (with throwaway caches, so the static inputs stay the read targets) to
    JIT every triton kernel before capture.
    """

    def __init__(self, runner: RWKV7Runner, batch_size: int) -> None:
        bank, layout = runner.bank, runner.bank.layout
        self.runner, self.batch_size = runner, batch_size
        dev = bank.device
        self.ids_in = torch.zeros(batch_size, 1, dtype=torch.long, device=dev)
        self.wkv_in = torch.zeros(
            layout.n_layer, batch_size, *layout.wkv_shape,
            dtype=WKV_DTYPE, device=dev)
        self.conv_in = torch.zeros(
            layout.n_layer, batch_size, *layout.shift_shape,
            dtype=layout.dtype, device=dev)
        self.ffn_in = torch.zeros_like(self.conv_in)

        with torch.inference_mode():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._forward()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.cache_out, self.logits_out = self._forward()

    def _forward(self):
        from fla.models.utils import Cache

        cache = Cache()
        for i in range(self.runner.bank.layout.n_layer):
            cache.update(
                recurrent_state=self.wkv_in[i], conv_state=self.conv_in[i],
                ffn_state=self.ffn_in[i], layer_idx=i, offset=0)
        out = self.runner.model(
            input_ids=self.ids_in, past_key_values=cache,
            use_cache=True, logits_to_keep=1)
        return cache, out.logits

    @torch.inference_mode()
    def decode_step(self, slot_batch, last_tokens) -> torch.Tensor:
        bank = self.runner.bank
        wkv_ids = bank._ids(slot_batch, "wkv")
        shift_ids = bank._ids(slot_batch, "shift")
        for i in range(bank.layout.n_layer):
            torch.index_select(bank.wkv[i], 0, wkv_ids, out=self.wkv_in[i])
            torch.index_select(bank.shift[i, 0], 0, shift_ids, out=self.conv_in[i])
            torch.index_select(bank.shift[i, 1], 0, shift_ids, out=self.ffn_in[i])
        self.ids_in.copy_(
            torch.tensor(last_tokens, dtype=torch.long,
                         device=bank.device).unsqueeze(1))
        self.graph.replay()
        for i in range(bank.layout.n_layer):
            st = self.cache_out[i]
            bank.wkv[i].index_copy_(0, wkv_ids, st["recurrent_state"].to(WKV_DTYPE))
            bank.shift[i, 0].index_copy_(
                0, shift_ids, st["conv_state"].to(bank.shift.dtype))
            bank.shift[i, 1].index_copy_(
                0, shift_ids, st["ffn_state"].to(bank.shift.dtype))
        return self.logits_out[:, -1].float()


class GraphedRunner(RWKV7Runner):
    """Runner that replays a captured graph when the decode batch matches the
    captured bucket, falling back to eager otherwise. Prefill stays eager."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._graph: GraphedDecode | None = None

    def capture(self, batch_size: int) -> None:
        self._graph = GraphedDecode(self, batch_size)

    def decode_step(self, slot_batch, last_tokens) -> torch.Tensor:
        if self._graph is not None and len(slot_batch) == self._graph.batch_size:
            return self._graph.decode_step(slot_batch, last_tokens)
        return super().decode_step(slot_batch, last_tokens)


def steady_state_toks(engine: Engine, batch: int, vocab: int,
                      warmup: int, steps: int) -> float:
    g = torch.Generator().manual_seed(0)
    for _ in range(batch):
        engine.add_request(Request(
            prompt_token_ids=torch.randint(0, vocab, (PROMPT_LEN,),
                                           generator=g).tolist(),
            max_new_tokens=warmup + steps + 16,
        ))
    # Reach steady state: everyone admitted, every gap == 1 (pure decode).
    while engine.scheduler.waiting or any(
            r.num_scheduled_gap > 1 for r in engine.scheduler.running):
        engine.step()
    assert len(engine.scheduler.running) == batch

    for _ in range(warmup):
        engine.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        engine.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    for req in list(engine.scheduler.running):
        engine.abort_request(req.req_id)
    return batch * steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("WKVM_RWKV7_PATH",
                                                      DEFAULT_MODEL))
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 8, 32, 64, 128, 256])
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--graph", action="store_true",
                    help="also measure with CUDA-graphed decode forward")
    args = ap.parse_args()

    model, layout = load_rwkv7(args.model, device="cuda")
    vocab = model.config.vocab_size
    spec = layout.state_spec()
    mib_slot = spec.bytes_per_request / 2**20
    print(f"model: {args.model}")
    print(f"state per slot: {mib_slot:.2f} MiB "
          f"(wkv fp32 {spec.families[0].bytes_per_slot / 2**20:.2f} "
          f"+ shift {spec.families[1].bytes_per_slot / 2**20:.2f})")
    header = "| B | tok/s (eager) |" + (" tok/s (graphed) |" if args.graph else "") \
             + " state MiB total | peak VRAM GiB |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))

    for b in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cfg = SchedulerConfig(
            max_tokens_per_step=max(16384, b * (PROMPT_LEN + 8)),
            max_running_requests=b,
        )
        engine = Engine(model, layout, num_slots=b, scheduler_config=cfg)
        tps = steady_state_toks(engine, b, vocab, args.warmup, args.steps)
        row = f"| {b} | {tps:,.0f} |"
        if args.graph:
            try:
                engine.runner = GraphedRunner(model, engine.bank)
                engine.runner.capture(b)
                gtps = steady_state_toks(engine, b, vocab, args.warmup,
                                         args.steps)
                row += f" {gtps:,.0f} |"
            except Exception as exc:  # capture failure is a result, not a crash
                row += f" failed: {type(exc).__name__}: {exc} |"
        state_mib = engine.bank.state_bytes() / 2**20
        peak = torch.cuda.max_memory_allocated() / 2**30
        row += f" {state_mib:,.0f} | {peak:.2f} |"
        print(row, flush=True)
        del engine
    print(f"\n(per-slot state: {mib_slot:.2f} MiB; prompt {PROMPT_LEN}, "
          f"{args.steps} timed steps, {args.warmup} warmup)")


if __name__ == "__main__":
    main()
