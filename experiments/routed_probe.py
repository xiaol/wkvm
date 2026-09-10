"""Needle retention probe for the routed span bank.

One needle sentence at a controlled depth in a SQuAD haystack of ``L``
tokens (the RULER niah_single_2 shape), prefilled through the routed
engine. Reports, per attention layer, whether the needle's tokens are still
materialised (ring / pending columns, or a kept representative span whose
position range covers them), and separately whether the model answers —
selection failures and readout failures look the same in a RULER score.

  PYTHONPATH=. python experiments/routed_probe.py --lengths 8192 16384 --depths 10 30 50 70 90 --seeds 2
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "8")  # 128-core host: BLAS/torch thread fan-out on tiny host ops is a 10x slowdown
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ruler_lite import NEEDLE, TEMPLATES, Generator, sent_tokenize  # noqa: E402


def build(gen: Generator, tok, length: int, depth_pct: float, rng_seed: int):
    import random

    rng = random.Random(rng_seed)
    key, value = rng.choice(gen.words.pairs), str(rng.randint(10**6, 10**7 - 1))
    needle = NEEDLE.format(type_needle_v="numbers", key=key, value=value)
    template, prefix = TEMPLATES["niah"]
    template = template.replace("Some", "A").replace("are all", "is").replace("are", "is")
    prefix = prefix.replace("are", "is")
    words = length  # ~1.3 tokens/word: start high, trim below
    while True:
        text = " ".join(gen.essay[:words])
        sents = sent_tokenize(text)
        cut = int(len(sents) * depth_pct / 100)
        context = " ".join(sents[:cut] + [needle] + sents[cut:])
        body = template.format(type_needle_v="number", context=context, query=key)
        chat = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
        n = len(enc["input_ids"])
        if n <= length - 64:
            break
        words = int(words * (length - 96) / n)
    start = chat.index(needle)
    end = start + len(needle)
    needle_toks = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if b > start and a < end]
    ids = list(enc["input_ids"]) + tok(prefix.format(type_needle_v="number", query=key), add_special_tokens=False)["input_ids"]
    return ids, (needle_toks[0], needle_toks[-1] + 1), value


def retention(engine, slots, needle_range: tuple[int, int]) -> list[float]:
    """Per attention layer: fraction of the needle's positions still visible."""
    bank = engine.bank
    st, p = bank.rstore, bank.rg.p
    g, sid = slots["guest_kv"], slots["gdn_state"]
    lo, hi = needle_range
    need = set(range(lo, hi))
    pos = st.pos[g].tolist()
    valid0 = st.valid[0, g].tolist()
    exact_cols = {pos[c] for c in range(p.summ_base) if valid0[c] and pos[c] >= 0}  # sink, ring, pending (shared)
    out = []
    for layer in range(bank.layout.n_attn):
        covered = set(exact_cols)
        for sl in bank.rg.spans_for(sid, layer):
            for sp in sl:
                if sp.get("pos", -1) >= 0:
                    covered.update(range(sp["pos"], sp["pos"] + sp["len"]))
        out.append(len(need & covered) / len(need))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/x/Qwen3.5-9B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384])
    ap.add_argument("--depths", type=float, nargs="+", default=[10, 30, 50, 70, 90])
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--ring-tokens", type=int, default=1024)
    ap.add_argument("--routed-pending", type=int, default=512)
    ap.add_argument("--routed-slots", type=int, default=64)
    ap.add_argument("--routed-reps", type=int, default=48)
    ap.add_argument("--routed-max-span", type=int, default=48)
    ap.add_argument("--per-layer", action="store_true", help="per-layer routing decisions (default: shared)")
    ap.add_argument("--retention", choices=("novelty", "surprisal", "fps"), default="novelty")
    ap.add_argument("--trace", action="store_true", help="rank the needle span among all routed spans per salience signal")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from wkvm.core.config import SchedulerConfig
    from wkvm.core.request import Request
    from wkvm.engine import Engine

    tok = AutoTokenizer.from_pretrained(args.model)
    gen = Generator(tok, seed=42)
    engine = Engine.from_qwen35(
        args.model, num_slots=1, device=args.device, stop_token_ids=frozenset({tok.eos_token_id}), prefill_chunk=1024,
        scheduler_config=SchedulerConfig(max_tokens_per_step=16384, max_running_requests=1,
                                         max_tokens_per_request_per_step=1024),
        cuda_graphs={"batch_buckets": (1,)}, guest_mode="routed", ring_tokens=args.ring_tokens,
        routed_params=dict(routed_pending=args.routed_pending, routed_slots=args.routed_slots,
                           routed_reps=args.routed_reps, routed_max_span=args.routed_max_span,
                           routed_shared=not args.per_layer, routed_retention=args.retention),
    )
    rows = []
    trace_rows = []
    for length in args.lengths:
        for depth in args.depths:
            for seed in range(args.seeds):
                ids, rng_, value = build(gen, tok, length, depth, seed * 1000 + int(depth) + length)
                req = Request(prompt_token_ids=ids, max_new_tokens=12)
                t0 = time.time()
                if args.trace:
                    engine.bank.rg.trace = []
                engine.add_request(req, park_on_finish=True)
                while engine.has_unfinished:
                    engine.step()
                pred = tok.decode(req.output_token_ids, skip_special_tokens=True)
                ret = retention(engine, req.slots, rng_)
                engine.close_request(req.req_id)
                row = {"length": length, "depth": depth, "seed": seed, "prompt_tokens": len(ids), "needle_pos": rng_[0],
                       "retained_per_layer": [round(x, 2) for x in ret], "retained_min": min(ret),
                       "correct": value in pred, "pred": pred.strip()[:40], "wall_s": round(time.time() - t0, 1)}
                if args.trace:
                    tr = engine.bank.rg.trace
                    lo, hi = rng_
                    mine = [t for t in tr if t["pos"] < hi and t["pos"] + t["len"] > lo]
                    pct = {}
                    for key in ("sal", "sal_top4", "sal_mean", "sal_sum", "distinct", "sink_sim"):
                        vals = sorted(t[key] for t in tr)
                        best = max((t[key] for t in mine), default=None)
                        pct[key] = None if best is None else round(sum(v <= best for v in vals) / len(vals), 3)
                    row["needle_spans"] = [(t["pos"], t["len"]) for t in mine]
                    row["routed_spans"] = len(tr)
                    row["needle_percentile"] = pct
                    trace_rows.append(pct)
                rows.append(row)
                print(json.dumps(row), flush=True)
    by = {}
    for r in rows:
        k = (r["length"], r["depth"])
        by.setdefault(k, []).append(r)
    print("\nlength depth  retained(mean over layers)  fully-retained  correct")
    for (length, depth), rs in sorted(by.items()):
        mean_ret = sum(sum(r["retained_per_layer"]) / len(r["retained_per_layer"]) for r in rs) / len(rs)
        full = sum(r["retained_min"] >= 0.99 for r in rs) / len(rs)
        acc = sum(r["correct"] for r in rs) / len(rs)
        print(f"{length:6d} {depth:5.0f}  {mean_ret:8.2f}  {full:14.2f}  {acc:7.2f}")
    if trace_rows:
        keys = [k for k in trace_rows[0] if trace_rows[0][k] is not None]
        print("\nneedle percentile among routed spans (1.0 = most salient), mean over samples:")
        for k in keys:
            vals = [t[k] for t in trace_rows if t.get(k) is not None]
            print(f"  {k:9s} mean {sum(vals)/len(vals):.3f}   >=0.9 in {sum(v >= 0.9 for v in vals)}/{len(vals)}   >=0.8 in {sum(v >= 0.8 for v in vals)}/{len(vals)}")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "routing_stats": dict(engine.bank.rg.stats),
                                               "args": vars(args)}, indent=1))
    print("PROBE_OK")


if __name__ == "__main__":
    main()
