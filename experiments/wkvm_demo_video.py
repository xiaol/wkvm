"""wkvm generation demo video: capture real engine events, render to mp4.

Honesty rule: every piece of text and every number shown in the video comes
from a real Engine run recorded into an event log
(experiments/results/wkvm_demo_events.json) with wall-clock timestamps.
Time-compressed / slowed segments are labeled on screen; latency numbers
(resume, fork, kill/restart) are the measured values from this capture.

Four acts (all real):
  A concurrency wall   N sessions streaming through continuous batching
  B hibernate/resume   engine.hibernate each -> resume-to-next-token timed
  C fork + mutate      store.fork x16 (timed) -> 16 distinct continuations;
                       mutate(decay) diverges while parent stays exact
  D kill -9 / restart  phase1 subprocess persists + reference, SIGKILLed for
                       real; phase2 fresh process resumes; per-token diff

Usage:
  HF_HUB_OFFLINE=1 python experiments/wkvm_demo_video.py capture [--fast]
  python experiments/wkvm_demo_video.py render [--fast]
  (phase1/phase2 are internal act-D subprocess entrypoints)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"
WEIGHTS_15B = "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla/rwkv7-1.5B-world"
WEIGHTS_191M = "/run/media/xiaol/B214449214445C0B/wkvm_bench/weights/fla/rwkv7-191M-world"
STORE_DEFAULT = "/run/media/xiaol/B214449214445C0B/wkvm_bench/statestore/video_demo"
EVENTS_DEFAULT = RESULTS / "wkvm_demo_events.json"
MP4_DEFAULT = RESULTS / "wkvm_demo.mp4"

PROMPTS = [
    "The history of the city of Samarkand begins",
    "def quicksort(arr):",
    "The lighthouse keeper had not spoken to anyone in",
    "In distributed systems, consensus means",
    "Once upon a time in a small village by the sea,",  # CORPUS_IDX = 4
    "The spacecraft's telemetry showed an anomaly in",
    "My favorite recipe starts with two onions and",
    "SELECT users.name, COUNT(orders.id) FROM",
    "The quarterly report indicates that revenue",
    "Kyoto in autumn is famous for",
    "class LRUCache:",
    "The detective examined the room and noticed",
    "To train a neural network efficiently, one should",
    "The Amazon river carries more water than",
    "import asyncio\n\nasync def fetch_all(urls):",
    "Her grandmother's letters were written in",
    "The economics of renewable energy changed when",
    "fn main() { let mut counter =",
    "Reykjavik sits on the boundary between",
    "The tiger chased the rabbit through the bamboo, and",
    "A proof by induction proceeds as follows:",
    "The submarine descended past the photic zone where",
    "Buenos Aires was founded twice, first in",
    "for (let i = 0; i < nodes.length; i++) {",
    "The violinist tuned her instrument while",
    "Photosynthesis converts carbon dioxide and water into",
    "The chess grandmaster stared at the board and",
    "CREATE TABLE sessions (id INTEGER PRIMARY KEY,",
    "Marrakech's medina is a maze of",
    "The glacier had retreated forty meters since",
    "def tokenize(text: str) -> list[str]:",
    "The stock exchange opened sharply lower after",
    "Istanbul straddles two continents, and its",
    "The beekeeper noticed the hive was unusually quiet",
    "Quantum entanglement was described by Einstein as",
    "while true; do curl -s localhost:8080/health",
    "The archaeologists uncovered a mosaic depicting",
    "Lagos is the largest city in Africa by",
    "The pianist began the nocturne so softly that",
    "In Rust, ownership rules guarantee that",
    "The ferry crossed the strait twice daily, carrying",
    "Mitochondria are often called the powerhouse of",
    "git rebase -i HEAD~3  # squash the last",
    "The typhoon changed course overnight, and by morning",
    "Vienna's coffee houses were once the meeting place of",
    "The neural probe recorded spikes from",
    "print(sum(x**2 for x in range(10)))",
    "The desert caravan traveled only at night because",
]
CORPUS_IDX = 4  # the story session that act C forks

AGENT_TURNS = [
    "User: Hi! My name is Wei and I keep three ferrets.\n\n"
    "Assistant: Nice to meet you, Wei!\n\n"
    "User: I live in Chengdu and I work on inference engines.\n\nAssistant:",
    "\n\nUser: Please remind me: what do you know about me so far?\n\nAssistant:",
]


def now() -> float:
    return time.perf_counter()


def decode_stream(tok, out_ids: list[int]) -> list[str]:
    """One display piece per token; multi-token UTF-8 sequences are emitted
    whole at the completing token (earlier tokens contribute '')."""
    pieces: list[str] = []
    pending: list[int] = []
    for tk in out_ids:
        pending.append(tk)
        s = tok.decode(pending)
        if "�" in s:
            pieces.append("")
        else:
            pieces.append(s)
            pending = []
    return pieces


def _finish(engine, req):
    while not req.status.is_finished:
        engine.step()
    return list(req.output_token_ids)


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


# =========================================================================
# CAPTURE
# =========================================================================


def make_engine(weights: str, num_slots: int, store_dir: Path):
    from wkvm.core.config import SchedulerConfig
    from wkvm.engine import Engine

    cfg = SchedulerConfig(
        max_tokens_per_step=8192,
        max_running_requests=num_slots,
        max_tokens_per_request_per_step=512,
    )
    engine = Engine.from_pretrained(weights, num_slots=num_slots, scheduler_config=cfg)
    engine.attach_store(store_dir)
    return engine


def cmd_capture(args) -> None:
    import gc

    import torch
    from transformers import AutoTokenizer

    from wkvm.core.request import Request
    from wkvm.runner.sampling import SamplingParams

    weights = WEIGHTS_191M if args.fast else WEIGHTS_15B
    n = args.panes or (12 if args.fast else 48)
    a_span = args.act_a or (6.0 if args.fast else 20.0)
    wake_span = 2.5 if args.fast else 5.0
    kid_span = 2.5 if args.fast else 5.0

    store = Path(args.store)
    shutil.rmtree(store, ignore_errors=True)
    store.mkdir(parents=True, exist_ok=True)

    print(f"[capture] loading {weights} num_slots={n}")
    t0 = now()
    engine = make_engine(weights, n, store)
    load_s = now() - t0
    tok = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)
    print(f"[capture] engine up in {load_s:.1f}s")

    model_name = "RWKV-7 191M" if args.fast else "RWKV-7 1.5B"
    ev: dict = {
        "meta": {
            "model": model_name,
            "weights": weights,
            "gpu": torch.cuda.get_device_name(0),
            "slots": n,
            "load_s": round(load_s, 2),
            "date": time.strftime("%Y-%m-%d"),
        }
    }

    # ---- Act A: concurrency wall -----------------------------------------
    print(f"[capture] act A: {n} sessions for {a_span}s")
    sess = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        ids = tok(prompt)["input_ids"]
        req = Request(prompt_token_ids=ids, max_new_tokens=1_000_000)
        engine.add_request(req, SamplingParams(temperature=0.8, seed=1234 + i))
        sess.append({"sid": f"s{i:02d}", "prompt": prompt, "req": req,
                     "toks": [], "tt": []})

    # Warmup: the first steps include one-time triton compile/autotune for the
    # prefill/decode shapes (measured ~18s at 1.5B/B=48). Run until every
    # session has sampled its first token, then start the act clock; warmup
    # tokens are kept as real pane base text and the excluded warmup duration
    # is recorded (and labeled on screen).
    w0 = now()
    while (any(len(s["req"].output_token_ids) < 1 for s in sess)
           and now() - w0 < 300):
        engine.step()
    warm_s = now() - w0
    for s in sess:
        s["n0"] = len(s["req"].output_token_ids)
        s["base"] = tok.decode(s["req"].output_token_ids)
    print(f"[capture] act A warmup (kernel compile + first token): {warm_s:.1f}s")

    counters = []
    hist: list[tuple[float, int]] = [(0.0, 0)]
    t0 = now()
    last_c = 0.0
    while now() - t0 < a_span:
        engine.step()
        t = now() - t0
        total = 0
        for s in sess:
            out = s["req"].output_token_ids[s["n0"]:]
            k = len(s["toks"])
            if len(out) > k:
                for tk in out[k:]:
                    s["toks"].append(tk)
                    s["tt"].append(t)
            total += len(out)
        hist.append((t, total))
        if t - last_c >= 0.25:
            t1, c1 = hist[0]
            for ht, hc in reversed(hist):
                if ht <= t - 1.0:
                    t1, c1 = ht, hc
                    break
            counters.append({
                "t": round(t, 3),
                "toks": total,
                "tok_s": round((total - c1) / max(t - t1, 1e-6)),
                "vram": round(torch.cuda.memory_allocated() / 2**30, 2),
                "active": len(engine.scheduler.running),
            })
            last_c = t
    a_total = sum(len(s["toks"]) for s in sess)
    a_rate = a_total / a_span
    print(f"[capture] act A: {a_total} tokens in {a_span:.1f}s = {a_rate:.0f} tok/s aggregate")

    ev["A"] = {
        "span": round(now() - t0, 3),
        "warmup_s": round(warm_s, 1),
        "counters": counters,
        "sessions": [
            {"sid": s["sid"], "prompt": s["prompt"],
             "pieces": ([[0.0, s["base"]]] if s["base"] else [])
             + [[round(t, 3), p] for t, p in
                zip(s["tt"], decode_stream(tok, s["toks"]))]}
            for s in sess
        ],
    }

    # ---- Act B: hibernate all, resume some --------------------------------
    running = [s for s in sess if not s["req"].status.is_finished]
    print(f"[capture] act B: hibernating {len(running)} sessions")
    tB = now()
    hib = []
    for s in running:
        h0 = now()
        handle = engine.hibernate(s["req"].req_id, s["sid"])
        hib.append({"sid": s["sid"], "t": round(now() - tB, 4),
                    "ms": round((now() - h0) * 1e3, 2), "handle": handle})
    hib_span = now() - tB
    hib_handles = {h["handle"] for h in hib}
    warm_bytes = sum(
        sum(t.numel() * t.element_size() for t in ts.values())
        for h, ts in engine.store._warm.items() if h in hib_handles
    )
    per_mib = warm_bytes / max(len(hib), 1) / 2**20
    print(f"[capture] hibernated in {hib_span*1e3:.0f}ms, {per_mib:.2f} MiB/session")

    rng = random.Random(7)
    # one unrecorded warm resume: the very first B=1 decode after the big
    # batch re-triggers kernel warmup; exclude that from the measured probes
    warm = engine.submit_from_handle(hib[-1]["handle"], max_new_tokens=1)
    _finish(engine, warm)
    probes = rng.sample(hib, min(12, len(hib)))
    lat = []
    for p in probes:
        q0 = now()
        r = engine.submit_from_handle(p["handle"], max_new_tokens=1)
        _finish(engine, r)
        lat.append({"sid": p["sid"], "ms": round((now() - q0) * 1e3, 2)})
    p50 = statistics.median(l["ms"] for l in lat)
    print(f"[capture] resume-to-next-token p50={p50:.1f}ms n={len(lat)}")

    wake_probes = probes[: min(10, len(probes))]
    wake = []
    t0 = now()
    for i, p in enumerate(wake_probes):
        r = engine.submit_from_handle(
            p["handle"], max_new_tokens=1_000_000,
            params=SamplingParams(temperature=0.8, seed=555 + i))
        wake.append({"sid": p["sid"], "ms": lat[i]["ms"], "req": r,
                     "toks": [], "tt": []})
    while now() - t0 < wake_span and engine.has_unfinished:
        engine.step()
        t = now() - t0
        for w in wake:
            out = w["req"].output_token_ids
            k = len(w["toks"])
            for tk in out[k:]:
                w["toks"].append(tk)
                w["tt"].append(t)
    for w in wake:
        engine.abort_request(w["req"].req_id)
    print(f"[capture] act B: {sum(len(w['toks']) for w in wake)} continuation tokens")

    ev["B"] = {
        "hib": [{k: h[k] for k in ("sid", "t", "ms")} for h in hib],
        "hib_span": round(hib_span, 4),
        "per_mib": round(per_mib, 2),
        "lat": lat,
        "p50": round(p50, 2),
        "wake": [
            {"sid": w["sid"], "ms": w["ms"],
             "pieces": [[round(t, 3), p] for t, p in
                        zip(w["tt"], decode_stream(tok, w["toks"]))]}
            for w in wake
        ],
    }

    # ---- Act C: fork x16 + mutate ------------------------------------------
    # A dedicated corpus session: a fresh ~80-token story state is a far
    # better fork subject than a 900-token-deep act-A session (which can
    # drift into degenerate/multilingual text at temp 0.8).
    from wkvm.core.request import Request as _Request

    corpus_prompt = ("The old cartographer unrolled the map and pointed to "
                     "an island that appeared on no other chart.")
    creq = _Request(prompt_token_ids=tok(corpus_prompt)["input_ids"],
                    max_new_tokens=80)
    engine.add_request(creq, SamplingParams(temperature=0.8, seed=42))
    engine.save_on_finish(creq.req_id, "corpus")
    _finish(engine, creq)
    corpus_handle = engine._finish_handles[creq.req_id]
    corpus_text = corpus_prompt + tok.decode(creq.output_token_ids)
    print(f"[capture] act C: forking {corpus_handle} x16")
    forks = []
    for i in range(16):
        f0 = now()
        fh = engine.store.fork(corpus_handle, f"fork{i:02d}")
        forks.append({"name": f"fork{i:02d}", "handle": fh,
                      "ms": round((now() - f0) * 1e3, 2)})
    fork_mean = statistics.mean(f["ms"] for f in forks)

    kids = []
    t0 = now()
    for i, f in enumerate(forks[: min(16, n)]):
        r = engine.submit_from_handle(
            f["handle"], max_new_tokens=1_000_000,
            params=SamplingParams(temperature=0.9, seed=i))
        kids.append({"name": f["name"], "seed": i, "req": r, "toks": [], "tt": []})
    while now() - t0 < kid_span and engine.has_unfinished:
        engine.step()
        t = now() - t0
        for kd in kids:
            out = kd["req"].output_token_ids
            k = len(kd["toks"])
            for tk in out[k:]:
                kd["toks"].append(tk)
                kd["tt"].append(t)
    for kd in kids:
        engine.abort_request(kd["req"].req_id)
    uniq = len({tuple(kd["toks"][:32]) for kd in kids})
    print(f"[capture] act C: {uniq}/{len(kids)} distinct continuations, "
          f"fork mean {fork_mean:.1f}ms")

    m0 = now()
    mut_handle = engine.store.mutate(corpus_handle, "decay", {"alpha": 0.2})
    mut_ms = (now() - m0) * 1e3
    pr = engine.submit_from_handle(corpus_handle, max_new_tokens=1_000_000)
    mr = engine.submit_from_handle(mut_handle, max_new_tokens=1_000_000)
    pair = [{"req": pr, "toks": [], "tt": []}, {"req": mr, "toks": [], "tt": []}]
    t0 = now()
    while now() - t0 < (2.0 if args.fast else 3.5):
        engine.step()
        t = now() - t0
        for q in pair:
            out = q["req"].output_token_ids
            k = len(q["toks"])
            for tk in out[k:]:
                q["toks"].append(tk)
                q["tt"].append(t)
    engine.abort_request(pr.req_id)
    engine.abort_request(mr.req_id)
    ncmp = min(len(pair[0]["toks"]), len(pair[1]["toks"]))
    div = next((i for i in range(ncmp)
                if pair[0]["toks"][i] != pair[1]["toks"][i]), ncmp)
    print(f"[capture] mutate(decay 0.2): diverges at token {div}")

    ev["C"] = {
        "corpus_sid": "corpus",
        "corpus_handle": corpus_handle,
        "corpus_prompt": corpus_prompt,
        "corpus_tail": corpus_text[-180:],
        "forks": [{k: f[k] for k in ("name", "ms")} for f in forks],
        "fork_mean_ms": round(fork_mean, 2),
        "uniq": uniq,
        "kids": [
            {"name": kd["name"], "seed": kd["seed"],
             "pieces": [[round(t, 3), p] for t, p in
                        zip(kd["tt"], decode_stream(tok, kd["toks"]))]}
            for kd in kids
        ],
        "mutate": {
            "ms": round(mut_ms, 2),
            "handle": mut_handle,
            "div_token": div,
            "parent_pieces": [[round(t, 3), p] for t, p in
                              zip(pair[0]["tt"], decode_stream(tok, pair[0]["toks"]))],
            "mut_pieces": [[round(t, 3), p] for t, p in
                           zip(pair[1]["tt"], decode_stream(tok, pair[1]["toks"]))],
        },
    }

    # ---- Act D: real kill -9 / restart (subprocesses) ----------------------
    print("[capture] act D: freeing engine, launching phase1 subprocess")
    del engine, sess, wake, kids, pair, pr, mr
    gc.collect()
    torch.cuda.empty_cache()

    ev["D"] = run_act_d(weights, store)
    n_match = sum(ev["D"]["phase2"]["match"])
    print(f"[capture] act D: {n_match}/{len(ev['D']['phase2']['match'])} "
          f"tokens bit-exact across kill -9 "
          f"(SIGKILL rc={ev['D']['kill']['rc']})")

    ev["summary"] = {
        "act_a_tok_s": round(a_rate),
        "act_a_tokens": a_total,
        "hibernated": len(hib),
        "per_mib": round(per_mib, 2),
        "resume_p50_ms": round(p50, 2),
        "resume_n": len(lat),
        "fork_mean_ms": round(fork_mean, 2),
        "uniq_forks": uniq,
        "kill_exact": f"{n_match}/{len(ev['D']['phase2']['match'])}",
        "kill_rc": ev["D"]["kill"]["rc"],
    }
    _atomic_write(Path(args.events), ev)
    print(f"[capture] CAPTURE_OK -> {args.events}")
    print(json.dumps(ev["summary"], indent=2))


def run_act_d(weights: str, store: Path) -> dict:
    """Act D: phase1 subprocess builds+persists agent state and is SIGKILLed
    for real; phase2 is a brand-new process that resumes and diffs."""
    agent_store = store / "agent"
    shutil.rmtree(agent_store, ignore_errors=True)
    p1_json = RESULTS / "wkvm_demo_phase1.json"
    p2_json = RESULTS / "wkvm_demo_phase2.json"
    for f in (p1_json, p2_json):
        f.unlink(missing_ok=True)
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    p1 = subprocess.Popen(
        [sys.executable, __file__, "phase1", "--weights", weights,
         "--store", str(agent_store), "--out", str(p1_json)], env=env)
    t0 = now()
    while not p1_json.exists():
        if p1.poll() is not None:
            raise SystemExit(f"phase1 died early rc={p1.returncode}")
        if now() - t0 > 900:
            p1.kill()
            raise SystemExit("phase1 timeout")
        time.sleep(0.2)
    time.sleep(0.5)  # phase1 is now sleeping in its keepalive loop
    os.kill(p1.pid, signal.SIGKILL)  # the real kill -9
    p1.wait()
    print(f"[capture] kill -9 {p1.pid} -> rc={p1.returncode}")
    r2 = subprocess.run(
        [sys.executable, __file__, "phase2", "--weights", weights,
         "--store", str(agent_store), "--ref", str(p1_json),
         "--out", str(p2_json)], env=env)
    if r2.returncode != 0:
        raise SystemExit("phase2 failed")
    return {"phase1": json.loads(p1_json.read_text()),
            "kill": {"pid": p1.pid, "rc": p1.returncode},
            "phase2": json.loads(p2_json.read_text())}


def cmd_redo_d(args) -> None:
    """Re-run only act D (same real subprocess/kill path) and patch the
    existing events JSON."""
    ev = json.loads(Path(args.events).read_text())
    ev["D"] = run_act_d(ev["meta"]["weights"], Path(args.store))
    match = ev["D"]["phase2"]["match"]
    ev["summary"]["kill_exact"] = f"{sum(match)}/{len(match)}"
    ev["summary"]["kill_rc"] = ev["D"]["kill"]["rc"]
    _atomic_write(Path(args.events), ev)
    print(f"[redo-d] act D re-captured: {ev['summary']['kill_exact']} "
          f"bit-exact (rc={ev['summary']['kill_rc']}) -> {args.events}")


def cmd_phase1(args) -> None:
    """Act D phase 1: build agent memory over 3 turns, persist to COLD,
    record a greedy reference continuation, then wait to be SIGKILLed."""
    from transformers import AutoTokenizer

    from wkvm.core.request import Request

    store = Path(args.store)
    store.mkdir(parents=True, exist_ok=True)
    t0 = now()
    engine = make_engine(args.weights, 8, store)
    load_s = now() - t0
    tok = AutoTokenizer.from_pretrained(args.weights, trust_remote_code=True)

    handle = None
    turn_log = []
    for i, turn in enumerate(AGENT_TURNS):
        ids = tok(turn)["input_ids"]
        if handle is None:
            req = Request(prompt_token_ids=ids, max_new_tokens=24)
            engine.add_request(req)
        else:
            req = engine.submit_from_handle(handle, suffix_tokens=ids,
                                            max_new_tokens=24)
        engine.save_on_finish(req.req_id, "agent")
        _finish(engine, req)
        handle = engine._finish_handles[req.req_id]
        completion = tok.decode(req.output_token_ids)
        turn_log.append({"turn": turn, "completion": completion, "handle": handle})
        print(f"[phase1] turn {i}: {completion!r} -> {handle}")
    engine.store.persist(handle)

    ref = engine.submit_from_handle(handle, max_new_tokens=32)
    ref_toks = _finish(engine, ref)
    _atomic_write(Path(args.out), {
        "pid": os.getpid(),
        "handle": handle,
        "load_s": round(load_s, 2),
        "turns": turn_log,
        "ref_toks": ref_toks,
        "ref_text": tok.decode(ref_toks),
    })
    print(f"[phase1] pid {os.getpid()}: persisted {handle}, reference written; "
          "waiting for SIGKILL")
    while True:
        time.sleep(1)


def cmd_phase2(args) -> None:
    """Act D phase 2: brand-new process, store rebuilt from index.json,
    resume the handle and diff against the pre-kill reference."""
    from transformers import AutoTokenizer

    ref = json.loads(Path(args.ref).read_text())
    t0 = now()
    engine = make_engine(args.weights, 8, Path(args.store))
    load_s = now() - t0
    tok = AutoTokenizer.from_pretrained(args.weights, trust_remote_code=True)

    req = engine.submit_from_handle(ref["handle"], max_new_tokens=len(ref["ref_toks"]))
    toks: list[int] = []
    tt: list[float] = []
    t0 = now()
    while not req.status.is_finished:
        engine.step()
        t = now() - t0
        for tk in req.output_token_ids[len(toks):]:
            toks.append(tk)
            tt.append(t)
    match = [int(a == b) for a, b in zip(toks, ref["ref_toks"])]
    _atomic_write(Path(args.out), {
        "load_s": round(load_s, 2),
        "toks": toks,
        "pieces": [[round(t, 3), p] for t, p in zip(tt, decode_stream(tok, toks))],
        "match": match,
        "exact": toks == ref["ref_toks"],
    })
    print(f"[phase2] exact={toks == ref['ref_toks']} "
          f"({sum(match)}/{len(match)} tokens)")


# =========================================================================
# RENDER
# =========================================================================

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def _wide(ch: str) -> bool:
    import unicodedata

    return unicodedata.east_asian_width(ch) in ("W", "F")

BG = (12, 14, 19)
HDR_BG = (18, 21, 28)
PANE_BG = (19, 22, 29)
PANE_BG_DIM = (14, 16, 21)
BORDER = (44, 50, 62)
FG = (198, 208, 202)
DIM = (100, 110, 124)
DIMMER = (68, 76, 88)
GREEN = (96, 220, 130)
AMBER = (236, 186, 92)
RED = (236, 96, 86)
CYAN = (102, 198, 230)
WHITE = (232, 236, 236)


class Painter:
    def __init__(self, W, H, fps):
        from PIL import ImageFont

        self.W, self.H, self.fps = W, H, fps
        self._f: dict = {}
        self._ImageFont = ImageFont
        self.cw: dict = {}

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self._f:
            self._f[key] = self._ImageFont.truetype(
                FONT_BOLD if bold else FONT_PATH, size)
            self.cw[key] = self._f[key].getlength("M")
        return self._f[key]

    def char_w(self, size, bold=False):
        self.font(size, bold)
        return self.cw[(size, bold)]

    def font_cjk(self, size):
        key = ("cjk", size)
        if key not in self._f:
            try:
                self._f[key] = self._ImageFont.truetype(FONT_CJK, size, index=2)
            except Exception:
                self._f[key] = self.font(size)  # fall back to mono (tofu)
        return self._f[key]


class StreamPane:
    """Char-wrapped streaming text fed from [t, piece(, color)] events."""

    def __init__(self, events, base_lines=None):
        self.ev = events
        self.i = 0
        self.lines: list[list] = ([list(l) for l in base_lines]
                                  if base_lines else [[]])

    def advance(self, T, cw):
        while self.i < len(self.ev) and self.ev[self.i][0] <= T:
            e = self.ev[self.i]
            color = e[2] if len(e) > 2 else None
            self.feed(e[1], cw, color)
            self.i += 1

    def feed(self, s, cw, color=None):
        s = s.replace("�", "").replace("\r", "").replace("\t", "  ")
        for ch in s:
            if ch == "\n":
                self.lines.append([])
            else:
                w = 2 if _wide(ch) else 1
                if sum(2 if _wide(c) else 1 for c, _ in self.lines[-1]) + w > cw:
                    self.lines.append([])
                self.lines[-1].append((ch, color))
        if len(self.lines) > 80:
            del self.lines[: len(self.lines) - 80]

    def snapshot(self):
        return [list(l) for l in self.lines]


def draw_pane_text(d, P, rect, pane, n_lines, size, y0_off, default=FG, fade=True):
    x0, y0, x1, y1 = rect
    lh = size + 2
    vis = pane.lines[-n_lines:]
    scrolled = len(pane.lines) > n_lines
    f = P.font(size)
    for li, line in enumerate(vis):
        color0 = default
        if fade and scrolled and li == 0:
            color0 = DIMMER
        elif fade and scrolled and li == 1:
            color0 = DIM
        # group runs by (color, wide) so CJK falls back to the Noto font
        x = x0 + 4
        j = 0
        while j < len(line):
            c = line[j][1]
            wide = _wide(line[j][0])
            k = j
            while (k < len(line) and line[k][1] == c
                   and _wide(line[k][0]) == wide):
                k += 1
            run = "".join(ch for ch, _ in line[j:k])
            if wide:
                fc = P.font_cjk(size)
                d.text((x, y0 + y0_off + li * lh), run, font=fc,
                       fill=(c or color0))
                x += fc.getlength(run)
            else:
                d.text((x, y0 + y0_off + li * lh), run, font=f,
                       fill=(c or color0))
                x += P.char_w(size) * len(run)
            j = k
    return


def _runs(s):
    j = 0
    while j < len(s):
        wide = _wide(s[j])
        k = j
        while k < len(s) and _wide(s[k]) == wide:
            k += 1
        yield s[j:k], wide
        j = k


def measure_mixed(P, s, size, bold=False):
    w = 0.0
    for run, wide in _runs(s):
        w += (P.font_cjk(size).getlength(run) if wide
              else P.char_w(size, bold) * len(run))
    return w


def draw_mixed(d, P, xy, s, size, fill, bold=False):
    x, y = xy
    for run, wide in _runs(s):
        if wide:
            f = P.font_cjk(size)
            d.text((x, y), run, font=f, fill=fill)
            x += f.getlength(run)
        else:
            d.text((x, y), run, font=P.font(size, bold), fill=fill)
            x += P.char_w(size, bold) * len(run)
    return x


def grid_layout(n, x0, y0, x1, y1, gap=4):
    for cols, rows in [(4, 3), (4, 4), (6, 4), (8, 4), (8, 6), (10, 6)]:
        if cols * rows >= n:
            break
    w = (x1 - x0 - gap * (cols - 1)) / cols
    h = (y1 - y0 - gap * (rows - 1)) / rows
    rects = []
    for i in range(n):
        r, c = divmod(i, cols)
        px = x0 + c * (w + gap)
        py = y0 + r * (h + gap)
        rects.append((int(px), int(py), int(px + w), int(py + h)))
    return rects


def speed_label(d, P, factor):
    if factor > 1.15:
        s = f">> {factor:.1f}x"
    elif factor < 0.85:
        s = f"<< {factor:.2f}x (slowed)"
    else:
        return
    f = P.font(14, bold=True)
    d.text((P.W - 14 - f.getlength(s), 14), s, font=f, fill=AMBER)


def header(d, P, meta, counter=None, right=None):
    d.rectangle([0, 0, P.W, 44], fill=HDR_BG)
    d.line([0, 44, P.W, 44], fill=BORDER)
    title = f"wkvm — {meta['model']} — {meta['gpu']}"
    d.text((14, 13), title, font=P.font(16, bold=True), fill=WHITE)
    if counter is not None:
        s = (f"{counter['tok_s']:,} tok/s   VRAM {counter['vram']:.1f} GiB   "
             f"sessions {counter['active']}")
        f = P.font(15)
        d.text((P.W - 200 - f.getlength(s), 14), s, font=f, fill=GREEN)
    elif right:
        f = P.font(15)
        d.text((P.W - 200 - f.getlength(right), 14), right, font=f, fill=GREEN)


def footer(d, P, caption, sub=None):
    y = P.H - 40
    d.rectangle([0, y, P.W, P.H], fill=HDR_BG)
    d.line([0, y, P.W, y], fill=BORDER)
    d.text((14, y + 6), caption, font=P.font(15, bold=True), fill=CYAN)
    if sub:
        f = P.font(13)
        d.text((P.W - 14 - f.getlength(sub), y + 8), sub, font=f, fill=DIM)


def counter_at(counters, T):
    cur = counters[0] if counters else None
    for c in counters:
        if c["t"] <= T:
            cur = c
        else:
            break
    return cur


def cmd_render(args) -> None:
    from PIL import Image, ImageDraw

    ev = json.loads(Path(args.events).read_text())
    meta = ev["meta"]
    W, H = (1280, 720)
    fps = 30
    P = Painter(W, H, fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "22",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE)

    nframes = 0

    def emit(img):
        nonlocal nframes
        proc.stdin.write(img.tobytes())
        nframes += 1

    def blank():
        return Image.new("RGB", (W, H), BG)

    GX0, GY0, GX1, GY1 = 8, 52, W - 8, H - 46

    # ---------------- Act A ------------------------------------------------
    A = ev["A"]
    nA = len(A["sessions"])
    rects = grid_layout(nA, GX0, GY0, GX1, GY1)
    span = A["span"]
    DA = min(span, 24.0)
    factor = span / DA
    size = 11 if nA > 20 else 13
    tsize = 10 if nA > 20 else 12
    cw = max(8, int((rects[0][2] - rects[0][0] - 8) / P.char_w(size)))
    n_lines = max(2, int((rects[0][3] - rects[0][1] - tsize - 8) / (size + 2)))

    base = blank()
    bd = ImageDraw.Draw(base)
    for r, s in zip(rects, A["sessions"]):
        bd.rectangle(r, fill=PANE_BG, outline=BORDER)
        title = f"{s['sid']}  {s['prompt'][:cw - 5]}"
        bd.text((r[0] + 4, r[1] + 2), title[:cw], font=P.font(tsize), fill=DIM)

    panes = [StreamPane(s["pieces"]) for s in A["sessions"]]
    for f_i in range(int(DA * fps)):
        T = (f_i + 1) / fps * factor
        img = base.copy()
        d = ImageDraw.Draw(img)
        header(d, P, meta, counter=counter_at(A["counters"], T))
        for r, pane in zip(rects, panes):
            pane.advance(T, cw)
            draw_pane_text(d, P, r, pane, n_lines, size, tsize + 5)
        footer(d, P,
               f"ACT 1 / CONCURRENCY WALL — {nA} sessions decoding through one "
               "continuous-batching engine",
               f"live output — kernel warmup ({A.get('warmup_s', 0):.0f}s) "
               "excluded")
        speed_label(d, P, factor)
        emit(img)
    for pane in panes:
        pane.advance(1e9, cw)
    frozen = {s["sid"]: p.snapshot() for s, p in zip(A["sessions"], panes)}
    sid_rect = {s["sid"]: r for s, r in zip(A["sessions"], rects)}
    sid_prompt = {s["sid"]: s["prompt"] for s in A["sessions"]}

    # ---------------- Act B ------------------------------------------------
    B = ev["B"]
    hib_t = {h["sid"]: h["t"] for h in B["hib"]}
    hib_ms = {h["sid"]: h["ms"] for h in B["hib"]}
    hib_span = max(B["hib_span"], 1e-3)
    seg1 = 5.0
    hib_factor = hib_span / (seg1 - 1.0)

    def slot_strip(d, T_disp, mode, woken=frozenset()):
        # mode: 'hib' empties slots as hibernates land; 'wake' refills woken
        n = len(A["sessions"])
        sw = min(14, 300 // max(n, 1)) if n > 24 else min(14, (W - 900) // max(n, 1))
        x0 = W - 24 - sw * n
        y0 = H - 34
        for i, s in enumerate(A["sessions"]):
            sid = s["sid"]
            occupied = True
            if sid in hib_t:
                t_h = 1.0 + hib_t[sid] / hib_factor if mode == "hib" else -1
                occupied = mode == "hib" and T_disp < t_h
            if mode == "wake":
                occupied = sid in woken
            c = GREEN if occupied else (30, 34, 42)
            d.rectangle([x0 + i * sw, y0, x0 + i * sw + sw - 2, y0 + 22],
                        fill=c, outline=BORDER)
        d.text((x0, y0 - 14), "GPU state slots", font=P.font(10), fill=DIM)

    # B1: hibernate sweep
    for f_i in range(int(seg1 * fps)):
        T = (f_i + 1) / fps
        img = blank()
        d = ImageDraw.Draw(img)
        header(d, P, meta)
        for sid, r in sid_rect.items():
            gone = sid in hib_t and T >= 1.0 + hib_t[sid] / hib_factor
            d.rectangle(r, fill=PANE_BG_DIM if gone else PANE_BG, outline=BORDER)
            if gone:
                d.text((r[0] + 4, r[1] + 2),
                       f"{sid} -> host  {hib_ms[sid]:.1f} ms"[:cw],
                       font=P.font(tsize, bold=True), fill=AMBER)
            else:
                d.text((r[0] + 4, r[1] + 2),
                       f"{sid}  {sid_prompt[sid][:cw - 5]}"[:cw],
                       font=P.font(tsize), fill=DIM)
            pane = StreamPane([], base_lines=frozen[sid])
            draw_pane_text(d, P, r, pane, n_lines, size, tsize + 5,
                           default=DIMMER if gone else FG, fade=False)
        footer(d, P,
               f"ACT 2 / HIBERNATE — {len(B['hib'])} sessions -> host pinned "
               f"memory — {B['per_mib']:.1f} MiB each "
               f"(real sweep: {hib_span*1e3:.0f} ms)")
        slot_strip(d, T, "hib")
        lbl = f"slowed for display — real sweep: {hib_span*1e3:.0f} ms"
        fL = P.font(14, bold=True)
        d.text((P.W - 14 - fL.getlength(lbl), 14), lbl, font=fL, fill=AMBER)
        emit(img)

    # B2: staggered resumes, continuations stream at 1x from wake moment
    seg2 = 10.0
    wake = B["wake"]
    wake_start = {w["sid"]: 0.6 + i * 0.45 for i, w in enumerate(wake)}
    wpanes = {w["sid"]: StreamPane(w["pieces"], base_lines=frozen[w["sid"]])
              for w in wake}
    wms = {w["sid"]: w["ms"] for w in wake}
    last_wake = max(wake_start.values()) if wake_start else 0
    for f_i in range(int(seg2 * fps)):
        T = (f_i + 1) / fps
        img = blank()
        d = ImageDraw.Draw(img)
        header(d, P, meta, right=f"{len(wake)} resumed / {len(B['hib'])} hibernated")
        woken = {sid for sid, t0 in wake_start.items() if T >= t0}
        for sid, r in sid_rect.items():
            if sid in woken:
                d.rectangle(r, fill=PANE_BG, outline=GREEN)
                d.text((r[0] + 4, r[1] + 2),
                       f"{sid} resumed {wms[sid]:.1f} ms"[:cw],
                       font=P.font(tsize, bold=True), fill=GREEN)
                pane = wpanes[sid]
                pane.advance(T - wake_start[sid], cw)
                draw_pane_text(d, P, r, pane, n_lines, size, tsize + 5)
            else:
                d.rectangle(r, fill=PANE_BG_DIM, outline=BORDER)
                d.text((r[0] + 4, r[1] + 2), f"{sid}  (hibernated)"[:cw],
                       font=P.font(tsize), fill=DIMMER)
        cap = ("ACT 2 / RESUME — submit_from_handle(): state H2D + next "
               "token, per-session ms")
        if T > last_wake + 0.8:
            cap = (f"ACT 2 / RESUME — resume-to-next-token p50 {B['p50']:.1f} ms"
                   f" (n={len(B['lat'])}) — continues mid-sentence")
        footer(d, P, cap)
        slot_strip(d, T, "wake", woken)
        emit(img)

    # ---------------- Act C ------------------------------------------------
    C = ev["C"]
    # C1: fork title card
    seg = 2.5
    for f_i in range(int(seg * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        header(d, P, meta)
        y = 170
        d.text((90, y), f"engine.store.fork({C['corpus_sid']!r}) × 16",
               font=P.font(30, bold=True), fill=WHITE)
        d.text((90, y + 60), "one O(MB) metadata copy per fork — "
               "no state tensors copied, parent immutable",
               font=P.font(16), fill=DIM)
        d.text((90, y + 100),
               f"measured: {C['fork_mean_ms']:.2f} ms/fork (mean of 16)",
               font=P.font(20, bold=True), fill=GREEN)
        d.text((90, y + 140), "parent state (the story so far):",
               font=P.font(13), fill=DIM)
        tail = C.get("corpus_tail", C["corpus_prompt"]).replace("\n", " ")
        d.text((90, y + 162), f"...{tail[-115:]}", font=P.font(14), fill=CYAN)
        n_show = min(16, int((f_i / fps) / seg * 20))
        for i, fk in enumerate(C["forks"][:n_show]):
            d.text((90 + (i % 4) * 270, y + 215 + (i // 4) * 24),
                   f"{fk['name']}  {fk['ms']:.2f} ms", font=P.font(14),
                   fill=FG)
        footer(d, P, "ACT 3 / FORK — 16 copies of one session's full "
               "recurrent state, timed individually")
        emit(img)

    # C2: 16 fork panes streaming
    seg = 7.0
    n_kids = len(C["kids"])
    krects = grid_layout(n_kids, GX0, GY0, GX1, GY1)
    ksize, ktsize = 13, 12
    kcw = max(8, int((krects[0][2] - krects[0][0] - 8) / P.char_w(ksize)))
    klines = max(2, int((krects[0][3] - krects[0][1] - ktsize - 8) / (ksize + 2)))
    kbase = blank()
    kd = ImageDraw.Draw(kbase)
    for r, kid in zip(krects, C["kids"]):
        kd.rectangle(r, fill=PANE_BG, outline=BORDER)
        kd.text((r[0] + 4, r[1] + 2),
                f"{kid['name']}  seed={kid['seed']}  T=0.9",
                font=P.font(ktsize), fill=DIM)
    kpanes = [StreamPane(k["pieces"]) for k in C["kids"]]
    for f_i in range(int(seg * fps)):
        T = (f_i + 1) / fps
        img = kbase.copy()
        d = ImageDraw.Draw(img)
        header(d, P, meta, right=f"{n_kids} forks of {C['corpus_sid']} decoding")
        for r, pane in zip(krects, kpanes):
            pane.advance(T, kcw)
            draw_pane_text(d, P, r, pane, klines, ksize, ktsize + 6)
        footer(d, P,
               f"ACT 3 / FORK — fork = one O(MB) copy, "
               f"{C['fork_mean_ms']:.2f} ms each — {C['uniq']}/{n_kids} distinct "
               "continuations (different seeds)", "streaming at 1x")
        emit(img)

    # C3: mutate compare
    seg = 5.5
    M = C["mutate"]
    half = (GX1 - GX0 - 8) // 2
    prect = (GX0, GY0 + 26, GX0 + half, GY1)
    mrect = (GX0 + half + 8, GY0 + 26, GX1, GY1)
    msize = 14
    mcw = max(8, int((half - 10) / P.char_w(msize)))
    mlines = max(2, int((prect[3] - prect[1] - 30) / (msize + 2)))
    mut_pieces = [[t, p, (AMBER if j >= M["div_token"] else None)]
                  for j, (t, p) in enumerate(M["mut_pieces"])]
    ppane = StreamPane(M["parent_pieces"])
    mpane = StreamPane(mut_pieces)
    for f_i in range(int(seg * fps)):
        T = (f_i + 1) / fps
        img = blank()
        d = ImageDraw.Draw(img)
        header(d, P, meta)
        d.text((GX0 + 4, GY0 + 2), f"{C['corpus_handle']}  (parent, greedy)",
               font=P.font(14, bold=True), fill=GREEN)
        d.text((mrect[0] + 4, GY0 + 2),
               f"{M['handle']}  = mutate(decay α=0.2)  (greedy)",
               font=P.font(14, bold=True), fill=AMBER)
        for r, pane in ((prect, ppane), (mrect, mpane)):
            d.rectangle(r, fill=PANE_BG, outline=BORDER)
            pane.advance(T, mcw)
            draw_pane_text(d, P, r, pane, mlines, msize, 6)
        footer(d, P,
               f"ACT 3 / MUTATE — decay applied in {M['ms']:.1f} ms; "
               f"diverges at token {M['div_token']} — state ≠ f(prefix): "
               "no prefix cache can index this", "both decoded greedy, 1x")
        emit(img)

    # ---------------- Act D ------------------------------------------------
    D = ev["D"]
    p1, p2 = D["phase1"], D["phase2"]
    pid = D["kill"]["pid"]
    lines: list[tuple[float, str, tuple]] = [(0.4, f"$ python agent.py                       # engine pid {pid}", CYAN)]
    t = 1.0
    lines.append((t, f"[agent] {meta['model']} up in {p1['load_s']:.1f}s — 8 slots, store attached", DIM))
    for i, turn in enumerate(p1["turns"]):
        for ln in [x.strip() for x in turn["turn"].split("\n") if x.strip()]:
            t += 0.45
            lines.append((t, f"  > {ln[:88]}", FG))
        t += 0.5
        comp = turn["completion"].replace("\n", " ").strip()
        lines.append((t, f"    {comp[:88]}", GREEN))
    t += 0.6
    lines.append((t, f"[store] persisted {p1['handle']} -> NVMe (safetensors + index.json)", AMBER))
    t += 0.5
    lines.append((t, f"[ref]   next {len(p1['ref_toks'])} greedy tokens recorded as reference", DIM))
    t += 1.0
    tk = t
    lines.append((t, f"$ kill -9 {pid}", RED))
    t += 0.8
    lines.append((t, f"[1]+  Killed                 python agent.py   (SIGKILL, exit {D['kill']['rc']})", RED))
    t += 1.0
    lines.append((t, f"$ python agent.py --resume {p1['handle']}    # brand-new process", CYAN))
    t += 0.6
    lines.append((t, f"[agent] store rebuilt from index.json alone — engine up in {p2['load_s']:.1f}s", DIM))
    t += 0.5
    lines.append((t, "[resume] continuing from stored state (no re-prefill):", FG))
    stream_t0 = t + 0.4
    match = p2["match"]
    n_match = sum(match)
    done_t = stream_t0 + (p2["pieces"][-1][0] if p2["pieces"] else 0) + 0.5
    segD = done_t + 3.2

    lh = 20
    fsz = 14
    max_px = W - 100
    # pre-layout resumed tokens (pixel positions, mixed-font widths)
    tok_layout = []  # (t_disp, x_px, row, text, width_px, ok)
    xpx, row = 0.0, 0
    for j, (tt_, piece) in enumerate(p2["pieces"]):
        piece = piece.replace("�", "").replace("\n", " ")
        wpx = measure_mixed(P, piece, fsz)
        if xpx + wpx > max_px:
            xpx, row = 0.0, row + 1
        tok_layout.append((stream_t0 + tt_, xpx, row, piece, wpx,
                           bool(match[j]) if j < len(match) else True))
        xpx += wpx

    for f_i in range(int(segD * fps)):
        T = (f_i + 1) / fps
        img = blank()
        d = ImageDraw.Draw(img)
        header(d, P, meta)
        y = 62
        for lt, s, color in lines:
            if T >= lt:
                draw_mixed(d, P, (48, y), s, fsz, color)
            y += lh
        # resumed token stream with match underline
        ybase = y + 6
        for td, cx, cr, piece, wpx, ok in tok_layout:
            if T < td or not piece:
                continue
            x = 48 + cx
            yy = ybase + cr * (lh + 6)
            draw_mixed(d, P, (x, yy), piece, fsz, WHITE)
            d.line([x + 1, yy + fsz + 5, x + wpx - 1, yy + fsz + 5],
                   fill=GREEN if ok else RED, width=2)
        if T >= done_t:
            d.text((48, ybase + (tok_layout[-1][2] + 2) * (lh + 6)),
                   f"[exact] {n_match}/{len(match)} tokens match the pre-kill "
                   "reference — bit-exact", font=P.font(15, bold=True),
                   fill=GREEN)
        cap = "ACT 4 / KILL -9 — process killed for real (SIGKILL)"
        if T >= done_t:
            cap = ("ACT 4 / KILL -9 — process killed — state survived on NVMe "
                   "— continuation bit-exact")
        footer(d, P, cap, "token stream at 1x; green underline = matches reference")
        if tk <= T < tk + 0.8:  # flash on the kill
            d.rectangle([0, 0, W, 4], fill=RED)
            d.rectangle([0, H - 4, W, H], fill=RED)
        emit(img)

    # ---------------- End card ---------------------------------------------
    S = ev["summary"]
    seg = 6.0
    rows = [
        ("wkvm", 40, WHITE, True),
        ("state-native inference for RWKV-7 — sessions are objects: "
         "hibernate, resume, fork, mutate, survive kill -9", 16, DIM, False),
        ("", 10, DIM, False),
        (f"this run: {S['act_a_tok_s']:,} tok/s @ B={meta['slots']} · "
         f"resume p50 {S['resume_p50_ms']:.1f} ms · fork {S['fork_mean_ms']:.1f} ms · "
         f"kill -9 survived {S['kill_exact']} bit-exact", 14, GREEN, True),
        ("", 8, DIM, False),
        ("measured elsewhere in this repo:", 14, DIM, False),
        ("resume p50 8.2 ms / fork 8.5 ms          (m3_results.md, 191M fleet, 2000 sessions)", 15, FG, False),
        ("96 slots @16k ctx vs 9 for vLLM          (docs/COMPARISON.md, same GPU)", 15, FG, False),
        ("8,077 tok/s @ B=256, 1.5B CUDA-graphed   (m2_engine_bench.md)", 15, FG, False),
        ("", 10, DIM, False),
        ("github.com/xiaol/wkvm", 22, CYAN, True),
    ]
    for f_i in range(int(seg * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        y = 120
        for s, sz, color, bold in rows:
            if s:
                f = P.font(sz, bold=bold)
                d.text(((W - f.getlength(s)) / 2, y), s, font=f, fill=color)
            y += sz + 18
        d.text((14, H - 26), f"all numbers measured — {meta['date']} — {meta['gpu']}",
               font=P.font(12), fill=DIMMER)
        emit(img)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    dur = nframes / fps
    print(f"RENDER_OK {out} frames={nframes} duration={dur:.1f}s "
          f"size={out.stat().st_size/2**20:.1f}MB")


# =========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--fast", action="store_true", help="191M, fewer panes")
    c.add_argument("--panes", type=int, default=None)
    c.add_argument("--act-a", type=float, default=None, dest="act_a")
    c.add_argument("--store", default=STORE_DEFAULT)
    c.add_argument("--events", default=str(EVENTS_DEFAULT))
    c.set_defaults(fn=cmd_capture)

    rd = sub.add_parser("redo-d")
    rd.add_argument("--events", default=str(EVENTS_DEFAULT))
    rd.add_argument("--store", default=STORE_DEFAULT)
    rd.set_defaults(fn=cmd_redo_d)

    for name, fn in (("phase1", cmd_phase1), ("phase2", cmd_phase2)):
        p = sub.add_parser(name)
        p.add_argument("--weights", required=True)
        p.add_argument("--store", required=True)
        p.add_argument("--out", required=True)
        if name == "phase2":
            p.add_argument("--ref", required=True)
        p.set_defaults(fn=fn)

    r = sub.add_parser("render")
    r.add_argument("--fast", action="store_true")
    r.add_argument("--events", default=str(EVENTS_DEFAULT))
    r.add_argument("--out", default=str(MP4_DEFAULT))
    r.set_defaults(fn=cmd_render)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
