#!/usr/bin/env python
"""Quality evaluation: full-KV vs sink+ring vs ring+bank on gemma-4-E4B-it.

Where and how much does recurrent mode hurt? Two probes:
  grid  - depth-stratified synthetic recall tasks (single needle, multi-key
          needle, aggregation) over ctx x depth x mode. NON-repetitive filler
          (cycled varied sentences): repetitive filler flatters lossy modes
          (PoC-2 finding).
  nll   - position-resolved teacher-forced NLL on natural long documents built
          from local files (papers .tex, repo docs .md, vllm .py), binned per
          1k positions, per-mode deltas vs full.

Modes: full (stock cache), ring (sink16+window1024), banked (PoC-2 defaults:
K_STATES=16, seg=256, reps=8, leader selection). Everything batch-1, greedy.

Machinery is imported from gemma_recurrent_poc.py (its CLI stays untouched).

Run:
  HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python \
      experiments/quality_eval.py grid --smoke   # <5min, prints GRID_OK
"""

import argparse
import glob
import os
import random
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import gemma_recurrent_poc as P

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_MODES = "full,ring,banked,routed"


def make_cfg(chunk=2048):
    """Cache/config namespace consumed by P.build_cache. Banked = PoC-2 default;
    routed default = resid routing, M=16 (variant sweep in quality_grid.md)."""
    return SimpleNamespace(sink=16, window=1024, k_states=16, seg=256, reps=8,
                           select="shared", chunk=chunk, m_slots=16, route_on="resid")


def parse_mode(mode, chunk=2048):
    """Mode spec -> (base mode for build_cache, cfg). Routed variants encode
    routing feature and slot count: e.g. 'routed-key-m64', 'routed-value-m16'."""
    cfg = make_cfg(chunk)
    parts = mode.split("-")
    base = parts[0]
    assert base in ("full", "ring", "banked", "routed"), f"bad mode {mode}"
    for p in parts[1:]:
        if p in ("key", "resid", "value"):
            cfg.route_on = p
        elif p.startswith("m") and p[1:].isdigit():
            cfg.m_slots = int(p[1:])
        else:
            raise ValueError(f"bad mode suffix {p!r} in {mode!r}")
    return base, cfg


# --------------------------------------------------------------------------- #
# Non-repetitive filler: cycled varied sentences from combinatorial templates
# --------------------------------------------------------------------------- #
_SUBJ = ["The river barge", "A distant thunderstorm", "The night market", "Her walnut desk",
         "The harbor crane", "An old tram line", "The observatory dome", "A migrating flock",
         "The mountain pass", "The printing press", "A rusted windmill", "The tide pool",
         "The village bakery", "An amber streetlight", "The glacier field", "A courier on horseback",
         "The lecture hall", "The suspension bridge", "A weathered signpost", "The orchard wall"]
_VERB = ["stood silent beside", "cast long shadows over", "slowly drifted past", "was rebuilt near",
         "creaked in the wind above", "drew visitors toward", "had been painted the color of",
         "remained hidden behind", "echoed faintly across", "gradually gave way to",
         "was photographed against", "leaned precariously over"]
_OBJ = ["the flooded meadow", "a row of cypress trees", "the abandoned railway", "a field of barley",
        "the limestone cliffs", "an overgrown courtyard", "the frozen reservoir", "a cluster of houseboats",
        "the terraced vineyards", "a line of fishing huts", "the basalt columns", "an empty amphitheater"]
_TAIL = ["long before the festival began", "while the rain kept falling", "as autumn settled in",
         "despite the morning fog", "just after the harvest ended", "when the ferries stopped running",
         "under a pale winter sun", "though nobody remembered why", "as the church bells rang",
         "before the road was widened"]


def sentence_pool_ids(tok, rng, n=400):
    """Tokenized pool of n varied sentences; cycled to build filler."""
    seen, pool = set(), []
    while len(pool) < n:
        s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_TAIL)}. "
        if s in seen:
            continue
        seen.add(s)
        pool.append(tok(s, add_special_tokens=False).input_ids)
    return pool


def build_middle(tok, rng, budget, insertions):
    """Token ids of length `budget`: cycled varied filler with `insertions`
    ([(token_pos, text), ...]) placed at their positions."""
    pool = sentence_pool_ids(tok, rng)
    order = list(range(len(pool)))
    rng.shuffle(order)
    out, i = [], 0
    for pos, text in sorted(insertions):
        ins = tok(text, add_special_tokens=False).input_ids
        assert pos + len(ins) < budget - 32, "insertion too close to the end"
        while len(out) < pos:
            out += pool[order[i % len(pool)]]
            i += 1
        out = out[:pos] + ins
    while len(out) < budget:
        out += pool[order[i % len(pool)]]
        i += 1
    return out[:budget]


def assemble_prompt(tok, middle_ids, query):
    prefix, suffix = P.chat_affixes(tok)
    q_ids = tok("\n\n" + query, add_special_tokens=False).input_ids
    return torch.tensor([prefix + middle_ids + q_ids + suffix], dtype=torch.long)


def middle_budget(tok, ctx, query):
    prefix, suffix = P.chat_affixes(tok)
    q_ids = tok("\n\n" + query, add_special_tokens=False).input_ids
    return ctx - len(prefix) - len(suffix) - len(q_ids)


# --------------------------------------------------------------------------- #
# Tasks
# --------------------------------------------------------------------------- #
_COLORS = ["CRIMSON", "AZURE", "AMBER", "VIOLET", "TEAL", "COBALT", "SIENNA", "IVORY"]
_NAMES = ["Aurora", "Bastion", "Cascade", "Driftwood", "Ember", "Falcon", "Glacier", "Harbor"]
_ITEMS = ["astrolabe", "theremin", "sextant", "monocle", "gyroscope", "abacus",
          "chalice", "hourglass", "barometer", "spyglass"]


def _code(rng):
    return f"{rng.choice(_COLORS)}-{rng.randint(100, 999)}"


def task_t1(tok, rng, ctx, depth):
    """Single needle. Returns (prompt, scorer, gen_len)."""
    code = _code(rng)
    query = "What is the secret code mentioned in the text? Answer with just the code."
    budget = middle_budget(tok, ctx, query)
    pos = int(depth * (budget - 64))
    mid = build_middle(tok, rng, budget, [(pos, f"The secret code is {code}. ")])
    return assemble_prompt(tok, mid, query), (lambda out: float(code in out)), 32


def task_t2(tok, rng, ctx, depth):
    """Multi-key needle: 8 keyed facts, query the one nearest `depth`."""
    values = {}
    while len(values) < len(_NAMES):  # distinct values
        values[_NAMES[len(values)]] = _code(rng)
    fact_depths = [(j + 0.5) / 8 for j in range(8)]
    target_j = min(range(8), key=lambda j: abs(fact_depths[j] - depth))
    name = _NAMES[target_j]
    query = f"What is the code for project {name}? Answer with just the code."
    budget = middle_budget(tok, ctx, query)
    ins = [(int(d * (budget - 64)), f"The code for project {n} is {values[n]}. ")
           for n, d in zip(_NAMES, fact_depths)]
    mid = build_middle(tok, rng, budget, ins)
    return assemble_prompt(tok, mid, query), (lambda out: float(values[name] in out)), 32


def task_t3(tok, rng, ctx, depth):
    """Aggregation-lite: 6 checklist items clustered around `depth`; repeat all."""
    items = rng.sample(_ITEMS, 6)
    query = "List every item on the expedition checklist mentioned in the text."
    budget = middle_budget(tok, ctx, query)
    lo, hi = max(0.02, depth - 0.06), min(0.93, depth + 0.06)
    ins = [(int((lo + (hi - lo) * j / 5) * (budget - 64)),
            f"Item {j + 1} on the expedition checklist is the {it}. ")
           for j, it in enumerate(items)]
    mid = build_middle(tok, rng, budget, ins)
    scorer = lambda out: sum(it in out for it in items) / len(items)
    return assemble_prompt(tok, mid, query), scorer, 48


TASKS = [("t1-needle", task_t1), ("t2-multikey", task_t2), ("t3-aggregate", task_t3)]


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def gen_answer(model, tok, prompt, mode, gen_len, chunk):
    base, cfg = parse_mode(mode, chunk)
    cache = P.build_cache(model, base, cfg)
    logits = P.chunked_prefill(model, cache, prompt, chunk)
    first = logits[:, -1].argmax(dim=-1, keepdim=True)
    rest, _ = P.greedy_decode(model, cache, first, gen_len - 1)
    ids = torch.cat([first, rest], dim=1)[0].tolist()
    cut = next((i for i, t in enumerate(ids) if t in (1, 106)), len(ids))
    P.free_cache(cache)
    return tok.decode(ids[:cut], skip_special_tokens=True)


def grid_cells(model, tok, modes, ctxs, depths, n_seeds, tasks=TASKS):
    """Shared cell runner: returns rows of {ctx, task, depth, <mode scores>}."""
    rows = []
    for ctx in ctxs:
        for tname, tfn in tasks:
            for depth in depths:
                cell = {m: [] for m in modes}
                for seed in range(n_seeds[tname]):
                    rng = random.Random(f"{ctx}/{tname}/{depth}/{seed}")
                    prompt, scorer, gen_len = tfn(tok, rng, ctx, depth)
                    for mode in modes:
                        chunk = 1024 if (mode == "full" and ctx >= 32768) else 2048
                        try:
                            out = gen_answer(model, tok, prompt, mode, gen_len, chunk)
                            cell[mode].append(scorer(out))
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            cell[mode].append(None)
                score = {m: (None if any(v is None for v in cell[m])
                             else sum(cell[m]) / len(cell[m])) for m in modes}
                rows.append(dict(ctx=ctx, task=tname, depth=depth, scores=score))
                fmt = " ".join(f"{m}={'N/A' if score[m] is None else f'{score[m]:.2f}'}"
                               for m in modes)
                print(f"[grid ctx={ctx:5d} {tname:12s} depth={depth:.1f}] {fmt}", flush=True)
    return rows


def _table(rows, modes):
    lines = ["| ctx | task | depth | " + " | ".join(modes) + " |",
             "|---" * (3 + len(modes)) + "|"]
    for r in rows:
        vals = " | ".join("N/A" if r["scores"][m] is None else f"{r['scores'][m]:.2f}"
                          for m in modes)
        lines.append(f"| {r['ctx']} | {r['task']} | {r['depth']:.1f} | {vals} |")
    return lines


def run_grid(model, tok, args):
    smoke = args.smoke
    modes = args.modes.split(",")
    ctxs = [2048] if smoke else [8192, 16384, 32768]
    depths = [0.2, 0.8] if smoke else [0.1, 0.3, 0.5, 0.7, 0.9]
    n_seeds = {"t1-needle": 1 if smoke else 3,
               "t2-multikey": 1 if smoke else 3,
               "t3-aggregate": 1}
    rows = grid_cells(model, tok, modes, ctxs, depths, n_seeds)
    lines = [
        "# Quality grid: full vs ring vs banked vs routed (gemma-4-E4B-it)",
        "",
        f"Depth-stratified synthetic recall; greedy, substring-scored, batch-1. "
        f"Filler = cycled varied sentences (non-repetitive). "
        f"t1/t2: mean of 3 seeds; t3: fraction of 6 items in 48-token output, 1 seed. "
        f"ring = sink16+window1024; banked = temporal-segment bank K=16/seg=256/reps=8, "
        f"leader-select; routed-* = training-free routed bank (Raven-style content "
        f"routing: M persistent slots, cosine routing at the leader layer, EMA "
        f"centroids, untouched slots never decay; suffix = routing feature + M). "
        f"full@32k uses prefill chunk 1024.",
        "",
    ] + _table(rows, modes)
    text = "\n".join(lines) + "\n"
    if not smoke:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "quality_grid.md"), "w") as fh:
            fh.write(text)
            if getattr(args, "append_text", None):
                fh.write(args.append_text)
        print(f"\nwrote {os.path.join(RESULTS_DIR, 'quality_grid.md')}")
    else:
        print(text)
    return rows


def run_sweep(model, tok, args):
    """Cheap routed-variant sweep to pick finalists: t1+t2 @8k, depths 0.1/0.5,
    2 seeds, all routing features x M."""
    variants = [f"routed-{f}-m{m}" for f in ("key", "resid", "value") for m in (16, 64)]
    modes = ["banked"] + variants
    n_seeds = {"t1-needle": 2, "t2-multikey": 2, "t3-aggregate": 1}
    rows = grid_cells(model, tok, modes, [8192], [0.1, 0.5], n_seeds,
                      tasks=TASKS[:2])
    print("\n### Routed-variant sweep (t1/t2 @ ctx 8192, depths 0.1/0.5, 2 seeds)\n")
    for line in _table(rows, modes):
        print(line)
    means = {m: sum(r["scores"][m] for r in rows) / len(rows) for m in modes}
    print("\nvariant means:", {m: round(v, 3) for m, v in means.items()})
    return rows, means


# --------------------------------------------------------------------------- #
# Position-resolved NLL on natural documents
# --------------------------------------------------------------------------- #
def _doc_sources():
    tex = sorted(glob.glob("/home/xiaol/X/Autoresearch_ideas/papers/*/*.tex")) + [
        "/home/xiaol/X/paper_2605_27734/source/main_text.tex",
        "/home/xiaol/X/gram_manim/paper/src/main_arxiv.tex",
        "/home/xiaol/X/mellum2_video/paper/source/main.tex",
    ]
    md = [
        "/home/xiaol/X/Multi-state-RWKV-online-memory/README.md",
        "/home/xiaol/X/causalab/ARCHITECTURE.md",
        "/home/xiaol/X/HRM-Text/README.md",
        "/home/xiaol/X/rwkv-lm/README.md",
        "/home/xiaol/X/tau2-bench/docs/leaderboard-submission.md",
        "/home/xiaol/X/megatrain/verl/README.md",
        "/home/xiaol/X/ccsp_README.md",
    ]
    py = sorted(glob.glob("/home/xiaol/X/vllm/vllm/*.py"),
                key=lambda p: -os.path.getsize(p))[:24]
    return [("papers-tex", [p for p in tex if os.path.isfile(p)]),
            ("repo-docs-md", [p for p in md if os.path.isfile(p)]),
            ("vllm-py", py)]


def build_docs(tok, docs_per_source, tokens_per_doc):
    docs = []
    for name, files in _doc_sources():
        buf, used, made = [], [], 0
        for f in files:
            try:
                text = open(f, errors="ignore").read()
            except OSError:
                continue
            buf += tok(text, add_special_tokens=False).input_ids
            used.append(os.path.relpath(f, "/home/xiaol/X"))
            if len(buf) >= tokens_per_doc:
                docs.append((f"{name}#{made}", list(used), buf[:tokens_per_doc]))
                buf, used = buf[tokens_per_doc:][:0], []
                made += 1
                if made >= docs_per_source:
                    break
        assert made >= docs_per_source, f"not enough text in source {name}"
    return docs


@torch.inference_mode()
def per_token_nll(model, cache, ids, chunk):
    """Teacher-forced NLL per predicted position (cache evolves chunk-wise, as
    in generation). Returns float32 cpu tensor nll[N-1]; nll[i] = -log p(ids[i+1])."""
    N = ids.shape[1]
    carry = None  # last position's logits from the previous chunk
    nlls = []
    for pos in range(0, N, chunk):
        end = min(pos + chunk, N)
        out = model(input_ids=ids[:, pos:end].to(model.device),
                    past_key_values=cache, use_cache=True, logits_to_keep=0)
        logits = out.logits  # bf16 [1, T, V]
        if carry is not None:
            preds = torch.cat([carry, logits[:, :-1]], dim=1)
            targets = ids[:, pos:end].to(model.device)
        else:
            preds = logits[:, :-1]
            targets = ids[:, pos + 1 : end].to(model.device)
        carry = logits[:, -1:].clone()
        p, t = preds[0], targets[0]
        for s in range(0, p.shape[0], 256):  # fp32 CE in slices (VRAM headroom)
            nlls.append(torch.nn.functional.cross_entropy(
                p[s : s + 256].float(), t[s : s + 256], reduction="none").cpu())
    return torch.cat(nlls)


def run_nll(model, tok, args):
    smoke = args.smoke
    docs_per_source, doc_len = (1, 4096) if smoke else (2, 16384)
    docs = build_docs(tok, docs_per_source, doc_len)
    if smoke:
        docs = docs[:2]
    bin_size = 1024
    n_bins = doc_len // bin_size
    modes = args.modes.split(",")
    assert modes[0] == "full", "nll modes must start with full (delta baseline)"
    per_doc = {}  # (doc, mode) -> [bin means]
    for dname, used, ids_list in docs:
        ids = torch.tensor([ids_list], dtype=torch.long)
        for mode in modes:
            base, cfg = parse_mode(mode)
            cache = P.build_cache(model, base, cfg)
            t0 = time.perf_counter()
            nll = per_token_nll(model, cache, ids, chunk=2048)
            P.free_cache(cache)
            pos = torch.arange(1, ids.shape[1])
            bins = [nll[(pos >= b * bin_size) & (pos < (b + 1) * bin_size)].mean().item()
                    for b in range(n_bins)]
            per_doc[(dname, mode)] = bins
            print(f"[nll {dname:16s} {mode:18s}] mean={nll.mean().item():.4f} "
                  f"bin0={bins[0]:.4f} last={bins[-1]:.4f} ({time.perf_counter()-t0:.0f}s)",
                  flush=True)

    def agg(mode, b):
        vals = [per_doc[(d, mode)][b] for d, _, _ in docs]
        return sum(vals) / len(vals)

    def delta_stats(mode, b):
        ds = [per_doc[(d, mode)][b] - per_doc[(d, "full")][b] for d, _, _ in docs]
        m = sum(ds) / len(ds)
        sd = (sum((x - m) ** 2 for x in ds) / max(1, len(ds) - 1)) ** 0.5
        return m, sd

    lossy = [m for m in modes if m != "full"]
    lines = ["# Position-resolved NLL: full vs recurrent modes (gemma-4-E4B-it)", "",
             f"{len(docs)} natural documents x {doc_len} tokens, teacher-forced with "
             f"chunk-2048 prefill (cache evolves as in generation; divergence "
             f"granularity is one chunk). Modes as in quality_grid.md. Provenance:", ""]
    for dname, used, _ in docs:
        lines.append(f"- {dname}: {', '.join(used[:6])}{' ...' if len(used) > 6 else ''}")
    hdr = "| pos bin | " + " | ".join(modes) + " | " + \
          " | ".join(f"d({m}-full) ± std" for m in lossy) + " |"
    lines += ["", hdr, "|---" * (1 + len(modes) + len(lossy)) + "|"]
    for b in range(n_bins):
        vals = " | ".join(f"{agg(m, b):.4f}" for m in modes)
        deltas = " | ".join("{:+.4f} ± {:.4f}".format(*delta_stats(m, b)) for m in lossy)
        lines.append(f"| {b}k-{b + 1}k | {vals} | {deltas} |")
    max_d0 = {m: max(abs(per_doc[(d, m)][0] - per_doc[(d, "full")][0]) for d, _, _ in docs)
              for m in lossy}
    ok = all(v < 1e-3 for v in max_d0.values())
    lines += ["", "Sanity (pre-eviction exactness): bin 0-1k max |delta| vs full: " +
              ", ".join(f"{m} {v:.2e}" for m, v in max_d0.items()) +
              f" -> {'PASS' if ok else 'FAIL'}", ""]
    lines.append("NLL_CURVE_OK" if ok else "NLL_CURVE_FAIL")
    text = "\n".join(lines) + "\n"
    if not smoke:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "quality_nll.md"), "w") as fh:
            fh.write(text)
        print(f"\nwrote {os.path.join(RESULTS_DIR, 'quality_nll.md')}")
    else:
        print(text)
    return ok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["grid", "nll", "sweep"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid (ctx 2048) + tiny nll (2 docs x 4k); prints GRID_OK")
    ap.add_argument("--modes", default=DEFAULT_MODES,
                    help="comma list; routed variants like routed-key-m64")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()

    path = P.resolve_model_path(args.model_path)
    print(f"# loading {path}")
    model = P.load_model(path)
    tok = P.AutoTokenizer.from_pretrained(path)
    print(f"# loaded; weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    if args.smoke:
        run_grid(model, tok, args)
        ok = run_nll(model, tok, args)
        print("GRID_OK" if ok else "SMOKE_FAILED_NLL_SANITY")
        return
    if args.cmd == "grid":
        run_grid(model, tok, args)
    elif args.cmd == "sweep":
        run_sweep(model, tok, args)
    else:
        run_nll(model, tok, args)


if __name__ == "__main__":
    main()
