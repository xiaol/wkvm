"""M1 micro-bench: RWKV-7 decode tok/s from arena slots vs batch size.

Numbers produced while another experiment owns most of the GPU are
placeholders — rerun on an idle card before quoting them anywhere.

Usage:
    python experiments/m1_decode_bench.py [--model PATH] [--batches 1 8 32]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wkvm.core.arena import StateArena  # noqa: E402
from wkvm.models.rwkv7 import load_rwkv7  # noqa: E402
from wkvm.runner import RWKV7Runner, RWKV7StateBank  # noqa: E402

DEFAULT_MODEL = "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla/rwkv7-0.1B-world"


def bench(runner: RWKV7Runner, arena: StateArena, batch: int,
          prompt_len: int = 32, steps: int = 64, warmup: int = 8) -> float:
    g = torch.Generator().manual_seed(0)
    vocab = runner.model.config.vocab_size
    slot_batch = []
    tokens = []
    for _ in range(batch):
        slots = arena.allocate()
        runner.bank.zero_slots(slots)
        prompt = torch.randint(0, vocab, (prompt_len,), generator=g).tolist()
        logits = runner.prefill(prompt, slots)
        slot_batch.append(slots)
        tokens.append(int(logits.argmax().item()))

    def run(n: int) -> None:
        toks = list(tokens)
        for _ in range(n):
            logits = runner.decode_step(slot_batch, toks)
            toks = logits.argmax(dim=-1).tolist()

    run(warmup)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run(steps)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    for slots in slot_batch:
        arena.free(slots)
    return batch * steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("WKVM_RWKV7_PATH", DEFAULT_MODEL))
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    model, layout = load_rwkv7(args.model, device="cuda")
    num_slots = max(args.batches)
    bank = RWKV7StateBank(layout, num_slots=num_slots, device="cuda")
    arena = StateArena(layout.state_spec(), num_slots=num_slots)
    runner = RWKV7Runner(model, bank)

    spec = layout.state_spec()
    print(f"model: {args.model}")
    print(f"state/slot: {spec.bytes_per_request / 2**20:.2f} MiB "
          f"({num_slots + 1} slots materialised: {bank.state_bytes() / 2**20:.1f} MiB)")
    for b in args.batches:
        tps = bench(runner, arena, b, steps=args.steps)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"batch {b:>3}: {tps:8.1f} tok/s  (peak alloc {peak:.2f} GiB)")


if __name__ == "__main__":
    main()
