#!/usr/bin/env python
"""p8: stage-attribution trace for routed-value-m64 failures on t2 (multi-key).

For each traced (ctx, depth, seed) cell, and each of the 8 facts, records:
  1. ROUTE-SCATTER  which slot(s) the fact's name / answer tokens landed in
  2. RETENTION      whether exact reps of the ANSWER tokens survive at query time
                    (and, if not, whether sibling-fact or filler reps occupy the
                    slot they were routed to)
  3. READOUT        softmax mass of the query decode step on the fact's surviving
                    rep indices vs the ring, per full-attention layer
Then attributes each FAILED queried fact to its first broken stage:
  scattered -> evicted -> lost-softmax -> decode-other
Writes experiments/results/t2_trace.md ending with STAGE_ATTRIBUTION_OK.

Run: HF_HUB_OFFLINE=1 .../python experiments/t2_trace.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import gemma_recurrent_poc as P
import quality_eval as Q

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BANKED_LAYER_IDX = None  # filled at runtime from config
ATTN_MASS_THRESHOLD = 0.05

# failing cells for routed-value-m64 from quality_grid.md (score < 1.0),
# spread over ctx x depth; seeds chosen per-cell at runtime by actual failure
CELLS = [(8192, 0.1), (8192, 0.3), (8192, 0.5), (8192, 0.7),
         (16384, 0.3), (16384, 0.5), (16384, 0.7), (16384, 0.9)]
SEEDS = [0, 1, 2]


def fact_spans(tok, prompt_ids, meta):
    """Per fact: dict(name, all token span, name-token span, answer-token span)."""
    spans = []
    for name in Q._NAMES:
        fact_text = f"The code for project {name} is {meta['values'][name]}. "
        fids = tok(fact_text, add_special_tokens=False).input_ids
        start = P.find_subsequence(prompt_ids, fids)
        assert start > 0, f"fact not found: {fact_text!r}"
        nids = tok(f" {name}", add_special_tokens=False).input_ids
        n_off = P.find_subsequence(fids, nids)
        aids = tok(f" {meta['values'][name]}", add_special_tokens=False).input_ids
        a_off = P.find_subsequence(fids, aids)
        assert n_off >= 0 and a_off >= 0
        spans.append(dict(
            name=name,
            span=list(range(start, start + len(fids))),
            name_pos=list(range(start + n_off, start + n_off + len(nids))),
            ans_pos=list(range(start + a_off, start + a_off + len(aids))),
        ))
    return spans


@torch.inference_mode()
def trace_cell(model, tok, ctx, depth, seed, mode="routed-value-m64"):
    rng = random.Random(f"{ctx}/t2-multikey/{depth}/{seed}")
    prompt, scorer, gen_len = Q.task_t2(tok, rng, ctx, depth)
    meta = scorer.meta
    ids = prompt[0].tolist()
    facts = fact_spans(tok, ids, meta)

    base, cfg = Q.parse_mode(mode)
    cfg.trace = True
    cache = P.build_cache(model, base, cfg)
    if getattr(cfg, "span", False):
        P.set_span_break_mask(cache, Q.break_mask_for(tok, ids))
    # prefill all but the last token; the final forward is the query step
    P.chunked_prefill(model, cache, prompt[:, :-1], cfg.chunk)
    layers = [l for l in cache.layers if isinstance(l, P.RoutedBankLayer)]
    leader = layers[0]
    layouts = {i: l.slot_layout()
               for i, l in enumerate(cache.layers) if isinstance(l, P.RoutedBankLayer)}
    n_slots_used = sum(1 for c in leader._slot_cnt if c > 0)
    out = model(input_ids=prompt[:, -1:].to(model.device), past_key_values=cache,
                use_cache=True, output_attentions=True, logits_to_keep=1)
    attns = out.attentions  # per model layer: [1, heads, 1, KV]
    first = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    rest, _ = P.greedy_decode(model, cache, first, gen_len - 1)
    toks = torch.cat([first, rest], dim=1)[0].tolist()
    cut = next((i for i, t in enumerate(toks) if t in (1, 106)), len(toks))
    answer = tok.decode(toks[:cut], skip_special_tokens=True)
    ok = meta["values"][meta["target"]] in answer

    # per-fact diagnostics (routing/rep state identical across banked layers)
    rep_positions = {p for slot in leader._rep_pos for p in slot}
    rep_slot_of = {p: s for s, slot in enumerate(leader._rep_pos) for p in slot}
    per_fact = []
    for f in facts:
        ans_slots = sorted({leader.assignment_of(p) for p in f["ans_pos"]
                            if leader.assignment_of(p) is not None})
        name_slots = sorted({leader.assignment_of(p) for p in f["name_pos"]
                             if leader.assignment_of(p) is not None})
        in_ring = all(leader.assignment_of(p) is None for p in f["ans_pos"])
        surv = sorted(set(f["ans_pos"]) & rep_positions)
        # if evicted: what lives in the slot(s) the answer went to?
        evictor = None
        if not surv and not in_ring and ans_slots:
            occ = [p for s in ans_slots for p in leader._rep_pos[s]]
            sib = [p for p in occ if any(p in g["span"] for g in facts)]
            evictor = "sibling" if sib else "filler"
        # readout mass over this fact's surviving rep indices, per banked layer
        mass = {}
        if surv:
            for li, layout in layouts.items():
                a = attns[li][0, :, 0, :].float().mean(0)  # head-avg [KV]
                fact_idx = [i for i, (kind, s, p) in enumerate(layout)
                            if kind == "slot_rep" and p in set(f["span"])]
                ring_idx = [i for i, (kind, s, p) in enumerate(layout) if kind == "ring"]
                mass[li] = (a[fact_idx].sum().item(), a[ring_idx].sum().item())
        per_fact.append(dict(
            name=f["name"], queried=f["name"] == meta["target"], in_ring=in_ring,
            ans_slots=ans_slots, name_slots=name_slots,
            colocated=(len(set(ans_slots) | set(name_slots)) <= 1),
            n_ans_reps=len(surv), evictor=evictor, mass=mass))
    P.free_cache(cache)
    return dict(ctx=ctx, depth=depth, seed=seed, ok=ok, answer=answer,
                target=meta["target"], facts=per_fact, n_slots_used=n_slots_used)


def attribute(fact):
    """First broken stage for a queried fact."""
    if fact["in_ring"]:
        return "in-ring"
    if not fact["colocated"]:
        return "scattered"
    if fact["n_ans_reps"] == 0:
        return "evicted"
    if fact["mass"]:
        avg = sum(m[0] for m in fact["mass"].values()) / len(fact["mass"])
        if avg < ATTN_MASS_THRESHOLD:
            return "lost-softmax"
    return "decode-other"


def main():
    global CELLS
    mode = sys.argv[1] if len(sys.argv) > 1 else "routed-value-m64"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "t2_trace.md"
    if len(sys.argv) > 3:  # optional cells override: ctx:depth,...
        CELLS = [(int(c.split(":")[0]), float(c.split(":")[1]))
                 for c in sys.argv[3].split(",")]
    path = P.resolve_model_path(None)
    print(f"# loading {path}")
    model = P.load_model(path)
    tok = P.AutoTokenizer.from_pretrained(path)
    # from_pretrained(config=..., attn_implementation="eager") is silently
    # ignored in transformers 5.9 -> the model actually runs sdpa; force eager
    # so output_attentions returns real weights for the readout stage
    model.set_attn_implementation("eager")

    results = []
    for ctx, depth in CELLS:
        for seed in SEEDS:
            r = trace_cell(model, tok, ctx, depth, seed, mode)
            q = next(f for f in r["facts"] if f["queried"])
            stage = attribute(q)
            print(f"[trace ctx={ctx} d={depth} s={seed}] ok={r['ok']} stage(q)={stage} "
                  f"q={r['target']} ans_slots={q['ans_slots']} reps={q['n_ans_reps']} "
                  f"evictor={q['evictor']} out={r['answer'][:40]!r}", flush=True)
            results.append((r, q, stage))

    failed = [(r, q, s) for r, q, s in results if not r["ok"]]
    succ = [(r, q, s) for r, q, s in results if r["ok"]]
    stages = ["scattered", "evicted", "lost-softmax", "decode-other", "in-ring"]
    fail_counts = {s: sum(1 for _, _, st in failed if st == s) for s in stages}
    succ_counts = {s: sum(1 for _, _, st in succ if st == s) for s in stages}

    # all-fact aggregates (64 facts per 8 cells x 2 seeds -> 128)
    allf = [f for r, _, _ in results for f in r["facts"] if not f["in_ring"]]
    n = len(allf)
    scat = sum(1 for f in allf if not f["colocated"])
    ans_split = sum(1 for f in allf if len(f["ans_slots"]) > 1)
    kept = sum(1 for f in allf if f["n_ans_reps"] > 0)
    sib = sum(1 for f in allf if f["evictor"] == "sibling")
    fil = sum(1 for f in allf if f["evictor"] == "filler")

    lines = [
        f"# t2 stage-attribution trace ({mode})",
        "",
        f"{len(results)} traced runs ({len(CELLS)} cells x {len(SEEDS)} seeds); "
        f"{len(failed)} failed, {len(succ)} succeeded. Stages: scattered (fact name+answer "
        f"tokens not co-located in one slot) -> evicted (no exact answer rep survives) -> "
        f"lost-softmax (surviving reps get < {ATTN_MASS_THRESHOLD} head-avg attention at the "
        f"query step, mean over the 7 full-attention layers) -> decode-other.",
        "",
        "## Stage attribution of QUERIED facts",
        "",
        "| stage | failed runs | % of failed | succeeded runs |",
        "|---|---|---|---|",
    ]
    for s in stages:
        pct = 100.0 * fail_counts[s] / max(1, len(failed))
        lines.append(f"| {s} | {fail_counts[s]} | {pct:.0f}% | {succ_counts[s]} |")
    lines += [
        "",
        "## All evicted facts across traced runs (routing/retention stats)",
        "",
        f"- facts fully evicted from ring: {n}",
        f"- NOT co-located (name/answer split across slots): {scat} ({100*scat/max(1,n):.0f}%)",
        f"- answer tokens themselves split over >1 slot: {ans_split} ({100*ans_split/max(1,n):.0f}%)",
        f"- >=1 exact answer rep survives at query: {kept} ({100*kept/max(1,n):.0f}%)",
        f"- evicted by sibling-fact reps: {sib}; by filler reps: {fil}",
        "",
        "## Per-run detail (queried fact)",
        "",
        "| ctx | depth | seed | ok | stage | ans slots | name slots | ans reps | evictor | mean fact-mass | mean ring-mass |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r, q, st in results:
        if q["mass"]:
            fm = sum(m[0] for m in q["mass"].values()) / len(q["mass"])
            rm = sum(m[1] for m in q["mass"].values()) / len(q["mass"])
            fm, rm = f"{fm:.3f}", f"{rm:.3f}"
        else:
            fm = rm = "-"
        lines.append(f"| {r['ctx']} | {r['depth']} | {r['seed']} | {r['ok']} | {st} "
                     f"| {q['ans_slots']} | {q['name_slots']} | {q['n_ans_reps']} "
                     f"| {q['evictor'] or '-'} | {fm} | {rm} |")
    lines += ["", "STAGE_ATTRIBUTION_OK"]
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, out_name), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"wrote {os.path.join(RESULTS, out_name)}")


if __name__ == "__main__":
    main()
