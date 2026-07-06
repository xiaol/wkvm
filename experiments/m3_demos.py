"""M3 demos: the three generation-claim demonstrations (docs/ANGLE.md §5).

  fleet   (a) thousands of hibernated sessions on one GPU, sub-100ms resume,
              exactness spot-checked against uninterrupted twins.
  agent   (b) a persistent agent surviving a REAL process restart bit-exactly,
              then forked to many children and mutated via a registered rule.
  parity  (c) trainer/server kernel parity: serving-path logprobs (chunked
              prefill + fused-recurrent decode) vs the training-path forward
              (one chunked pass, use_cache=False) over the same fla kernels.

Run (191M default):
  HF_HUB_OFFLINE=1 python experiments/m3_demos.py fleet --sessions 2000
  HF_HUB_OFFLINE=1 python experiments/m3_demos.py agent
  HF_HUB_OFFLINE=1 python experiments/m3_demos.py parity
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEIGHTS = os.environ.get(
    "WKVM_RWKV7_PATH",
    "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla/rwkv7-191M-world",
)
STORE_DIR = os.environ.get(
    "WKVM_STORE_DIR",
    "/run/media/xiaol/B214449214445C0B/wkvm_bench/statestore/demos",
)
PROMPTS = [
    "The history of the city of",
    "In distributed systems, consensus means",
    "My favorite recipe starts with",
    "The spacecraft's telemetry showed",
    "Once upon a time in a small village",
    "The quarterly report indicates that revenue",
    "To train a neural network efficiently,",
    "The detective examined the room and noticed",
]


def make_engine(num_slots: int = 64):
    import torch  # noqa: F401

    from wkvm.core.config import SchedulerConfig
    from wkvm.engine import Engine

    cfg = SchedulerConfig(
        max_tokens_per_step=8192,
        max_running_requests=num_slots,
        max_tokens_per_request_per_step=512,
    )
    engine = Engine.from_pretrained(WEIGHTS, num_slots=num_slots, scheduler_config=cfg)
    engine.attach_store(STORE_DIR)
    return engine


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)


def run_all(engine) -> None:
    while engine.has_unfinished:
        engine.step()


def finish(engine, req) -> list[int]:
    while not req.status.is_finished:
        engine.step()
    return list(req.output_token_ids)


# ---------------------------------------------------------------- demo (a) --


def demo_fleet(n_sessions: int, cold_fraction: float, resumes: int) -> None:
    from wkvm.core.request import Request

    engine = make_engine(num_slots=64)
    tok = tokenizer()
    rng = random.Random(0)

    print(f"[fleet] creating {n_sessions} sessions (continuous batching, 64 slots)")
    t0 = time.perf_counter()
    prompt_ids: dict[str, list[int]] = {}
    reqs = []
    for i in range(n_sessions):
        text = f"{rng.choice(PROMPTS)} (session {i})"
        ids = tok(text)["input_ids"]
        req = Request(prompt_token_ids=ids, max_new_tokens=8)
        engine.add_request(req)
        engine.save_on_finish(req.req_id, f"s{i}")
        prompt_ids[f"s{i}"] = ids
        reqs.append(req)
    run_all(engine)
    create_s = time.perf_counter() - t0
    handles = {f"s{i}": engine._finish_handles[r.req_id] for i, r in enumerate(reqs)}
    outputs8 = {f"s{i}": list(r.output_token_ids) for i, r in enumerate(reqs)}

    warm_bytes = sum(
        sum(t.numel() * t.element_size() for t in ts.values())
        for ts in engine.store._warm.values()
    )
    print(
        f"[fleet] created+snapshotted in {create_s:.1f}s; "
        f"WARM tier {warm_bytes / 2**30:.2f} GiB for {n_sessions} sessions "
        f"({warm_bytes / n_sessions / 2**20:.2f} MiB/session)"
    )

    cold = rng.sample(sorted(handles.values()), int(n_sessions * cold_fraction))
    t0 = time.perf_counter()
    for h in cold:
        engine.store.evict(h)
    print(
        f"[fleet] evicted {len(cold)} to COLD (safetensors) in "
        f"{time.perf_counter() - t0:.1f}s"
    )

    def resume_once(handle: str) -> float:
        t0 = time.perf_counter()
        req = engine.submit_from_handle(handle, max_new_tokens=1)
        finish(engine, req)
        return (time.perf_counter() - t0) * 1e3

    lat: dict[str, list[float]] = {"warm": [], "cold": []}
    cold_set = set(cold)
    for h in rng.sample(sorted(handles.values()), resumes):
        tier = "cold" if h in cold_set and h not in engine.store._warm else "warm"
        lat[tier].append(resume_once(h))
    for tier, xs in lat.items():
        if not xs:
            continue
        xs.sort()
        print(
            f"[fleet] resume-to-next-token {tier.upper()}: n={len(xs)} "
            f"p50={statistics.median(xs):.1f}ms "
            f"p99={xs[int(len(xs) * 0.99) - 1]:.1f}ms max={xs[-1]:.1f}ms"
        )

    # Exactness: interrupted-and-resumed == uninterrupted, with BOTH sides
    # run batch-1 so the check isolates the store (batch-shape bf16 GEMM
    # nondeterminism — the known M2 ulp-flip finding — is a separate axis
    # and would contaminate a comparison against the batch-64 creation run).
    ok = 0
    checks = 16
    for i in range(checks):
        ids = tok(f"{PROMPTS[i % len(PROMPTS)]} (exactness {i})")["input_ids"]
        twin = Request(prompt_token_ids=list(ids), max_new_tokens=24)
        engine.add_request(twin)
        twin_out = finish(engine, twin)
        first = Request(prompt_token_ids=list(ids), max_new_tokens=8)
        engine.add_request(first)
        engine.save_on_finish(first.req_id, f"exact{i}")
        first_out = finish(engine, first)
        h = engine._finish_handles[first.req_id]
        if i % 2:  # alternate: check the COLD path too
            engine.store.evict(h)
        resumed = engine.submit_from_handle(h, max_new_tokens=16)
        res_out = finish(engine, resumed)
        ok += twin_out == first_out + res_out
    print(f"[fleet] exactness: {ok}/{checks} interrupted+resumed == uninterrupted")
    assert ok == checks, "resume exactness violated"


# ---------------------------------------------------------------- demo (b) --


REF_FILE = Path(STORE_DIR) / "agent_reference.json"


def demo_agent_phase1() -> None:
    from wkvm.core.request import Request

    engine = make_engine(num_slots=8)
    tok = tokenizer()
    turns = [
        "User: My name is Wei and I keep three ferrets.\nAssistant:",
        "\nUser: I live in Chengdu and work on inference engines.\nAssistant:",
        "\nUser: Remind me what you know about me.\nAssistant:",
    ]
    handle = None
    for i, turn in enumerate(turns):
        ids = tok(turn)["input_ids"]
        if handle is None:
            req = Request(prompt_token_ids=ids, max_new_tokens=32)
            engine.add_request(req)
        else:
            req = engine.submit_from_handle(handle, suffix_tokens=ids, max_new_tokens=32)
        engine.save_on_finish(req.req_id, "agent")
        finish(engine, req)
        handle = engine._finish_handles[req.req_id]
        print(f"[agent:1] turn {i}: {tok.decode(req.output_token_ids)!r} -> {handle}")
    engine.store.persist(handle)

    ref = engine.submit_from_handle(handle, max_new_tokens=24)
    reference = finish(engine, ref)
    REF_FILE.write_text(json.dumps({"handle": handle, "continuation": reference}))
    print(f"[agent:1] persisted {handle}; reference continuation written. Exiting.")


def demo_agent_phase2() -> None:
    ref = json.loads(REF_FILE.read_text())
    engine = make_engine(num_slots=8)  # brand-new process: only COLD index exists
    req = engine.submit_from_handle(ref["handle"], max_new_tokens=24)
    out = finish(engine, req)
    exact = out == ref["continuation"]
    print(f"[agent:2] RESTART-EXACT: {exact} ({ref['handle']} across process boundary)")
    assert exact

    # Fork the agent's memory to 64 children exploring different replies.
    from wkvm.runner.sampling import SamplingParams

    t0 = time.perf_counter()
    children = [engine.store.fork(ref["handle"], f"agent-fork{i}") for i in range(64)]
    fork_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    kids = [
        engine.submit_from_handle(h, max_new_tokens=16,
                                  params=SamplingParams(temperature=0.9, seed=i))
        for i, h in enumerate(children[: engine.arena.num_free_slots()])
    ]
    run_all(engine)
    uniq = len({tuple(k.output_token_ids) for k in kids})
    print(
        f"[agent:2] forked 64 handles in {fork_s * 1e3:.1f}ms (O(name) each); "
        f"{len(kids)} decoded together in {time.perf_counter() - t0:.2f}s, "
        f"{uniq} distinct continuations"
    )

    # Mutation: decayed memory is a NEW state no token prefix reproduces.
    mutated = engine.store.mutate(ref["handle"], "decay", {"alpha": 0.2})
    mut_out = finish(engine, engine.submit_from_handle(mutated, max_new_tokens=24))
    again = finish(engine, engine.submit_from_handle(ref["handle"], max_new_tokens=24))
    print(
        f"[agent:2] mutate(decay 0.2): continuation changed={mut_out != ref['continuation']}, "
        f"parent still exact={again == ref['continuation']}, "
        f"provenance={engine.store.get(mutated).rule!r}<-{engine.store.get(mutated).parent}"
    )
    assert mut_out != ref["continuation"] and again == ref["continuation"]


def demo_agent() -> None:
    Path(STORE_DIR).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    for phase in ("1", "2"):
        r = subprocess.run(
            [sys.executable, __file__, "agent", "--phase", phase], env=env
        )
        if r.returncode != 0:
            raise SystemExit(f"agent phase {phase} failed")
    print("[agent] both phases green: state survived a real process restart.")


# ---------------------------------------------------------------- demo (c) --


def demo_parity(n_rollouts: int, new_tokens: int) -> None:
    import torch

    from wkvm.core.request import Request
    from wkvm.runner.sampling import SamplingParams

    engine = make_engine(num_slots=8)
    tok = tokenizer()
    prompt = tok("The engineer explained the design tradeoff:")["input_ids"]

    rollouts = []
    for i in range(n_rollouts):
        req = Request(prompt_token_ids=list(prompt), max_new_tokens=new_tokens)
        engine.add_request(req, SamplingParams(temperature=1.0, seed=1000 + i))
        rollouts.append(req)
    run_all(engine)

    # Server-path scoring: chunked prefill + fused-recurrent decode (the
    # engine's own kernels), replayed teacher-forced via the runner.
    def serve_logprobs(tokens: list[int]) -> torch.Tensor:
        slots = engine.arena.allocate()
        try:
            engine.bank.zero_slots(slots)
            lps = []
            logits = engine.runner.prefill(tokens[: len(prompt)], slots)
            for t in tokens[len(prompt):]:
                lp = torch.log_softmax(logits.float(), dim=-1)[t]
                lps.append(lp)
                logits = engine.runner.decode_step([slots], [t])[0]
            return torch.stack(lps)
        finally:
            engine.arena.free(slots)

    # Trainer-path scoring: one full-sequence chunked forward, no cache —
    # exactly what an RL trainer computes, same fla kernels underneath.
    @torch.inference_mode()
    def train_logprobs(tokens: list[int]) -> torch.Tensor:
        ids = torch.tensor([tokens], device=engine.runner.device)
        logits = engine.runner.model(input_ids=ids, use_cache=False).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        pos = torch.arange(len(prompt) - 1, len(tokens) - 1, device=lp.device)
        return lp[pos, torch.tensor(tokens[len(prompt):], device=lp.device)]

    diffs, agree = [], 0
    for req in rollouts:
        seq = req.prompt_token_ids + req.output_token_ids
        s, t = serve_logprobs(seq), train_logprobs(seq)
        diffs.append((s - t).abs())
        agree += int(torch.allclose(s, t, atol=5e-3))
    d = torch.cat(diffs)
    print(
        f"[parity] {n_rollouts} rollouts x {new_tokens} tokens: "
        f"max|dlogprob|={d.max():.2e} mean={d.mean():.2e} "
        f"rollouts within 5e-3 everywhere: {agree}/{n_rollouts}"
    )
    print(
        "[parity] server path = chunked prefill + fused_recurrent decode; "
        "trainer path = one chunk_rwkv7 forward (use_cache=False). Same fla "
        "kernels, bf16 activations — residual diff is accumulation order."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", choices=["fleet", "agent", "parity"])
    ap.add_argument("--sessions", type=int, default=2000)
    ap.add_argument("--cold-fraction", type=float, default=0.25)
    ap.add_argument("--resumes", type=int, default=300)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--phase", choices=["1", "2"], default=None)
    args = ap.parse_args()
    if args.demo == "fleet":
        demo_fleet(args.sessions, args.cold_fraction, args.resumes)
    elif args.demo == "agent":
        if args.phase == "1":
            demo_agent_phase1()
        elif args.phase == "2":
            demo_agent_phase2()
        else:
            demo_agent()
    else:
        demo_parity(args.rollouts, args.new_tokens)
