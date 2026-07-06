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
MODES = ["full", "ring", "banked"]


def make_cfg(chunk=2048):
    """Cache/config namespace consumed by P.build_cache. Banked = PoC-2 default."""
    return SimpleNamespace(sink=16, window=1024, k_states=16, seg=256, reps=8,
                           select="shared", chunk=chunk)


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
    cfg = make_cfg(chunk)
    cache = P.build_cache(model, mode, cfg)
    logits = P.chunked_prefill(model, cache, prompt, chunk)
    first = logits[:, -1].argmax(dim=-1, keepdim=True)
    rest, _ = P.greedy_decode(model, cache, first, gen_len - 1)
    ids = torch.cat([first, rest], dim=1)[0].tolist()
    cut = next((i for i, t in enumerate(ids) if t in (1, 106)), len(ids))
    P.free_cache(cache)
    return tok.decode(ids[:cut], skip_special_tokens=True)


def run_grid(model, tok, args):
    smoke = args.smoke
    ctxs = [2048] if smoke else [8192, 16384, 32768]
    depths = [0.2, 0.8] if smoke else [0.1, 0.3, 0.5, 0.7, 0.9]
    n_seeds = {"t1-needle": 1 if smoke else 3,
               "t2-multikey": 1 if smoke else 3,
               "t3-aggregate": 1}
    rows = []
    for ctx in ctxs:
        for tname, tfn in TASKS:
            for depth in depths:
                cell = {m: [] for m in MODES}
                for seed in range(n_seeds[tname]):
                    rng = random.Random(f"{ctx}/{tname}/{depth}/{seed}")
                    prompt, scorer, gen_len = tfn(tok, rng, ctx, depth)
                    for mode in MODES:
                        chunk = 1024 if (mode == "full" and ctx >= 32768) else 2048
                        try:
                            out = gen_answer(model, tok, prompt, mode, gen_len, chunk)
                            cell[mode].append(scorer(out))
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            cell[mode].append(None)
                score = {m: (None if any(v is None for v in cell[m])
                             else sum(cell[m]) / len(cell[m])) for m in MODES}
                rows.append(dict(ctx=ctx, task=tname, depth=depth, **score))
                fmt = {m: ("N/A" if score[m] is None else f"{score[m]:.2f}") for m in MODES}
                print(f"[grid ctx={ctx:5d} {tname:12s} depth={depth:.1f}] "
                      f"full={fmt['full']} ring={fmt['ring']} banked={fmt['banked']}",
                      flush=True)
    lines = [
        "# Quality grid: full vs ring vs banked (gemma-4-E4B-it)",
        "",
        f"Depth-stratified synthetic recall; greedy, substring-scored, batch-1. "
        f"Filler = cycled varied sentences (non-repetitive). "
        f"t1/t2: mean of 3 seeds; t3: fraction of 6 items in 48-token output, 1 seed. "
        f"ring = sink16+window1024; banked = +bank K=16/seg=256/reps=8 leader-select. "
        f"full@32k uses prefill chunk 1024.",
        "",
        "| ctx | task | depth | full | ring | banked |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        f = {m: ("N/A" if r[m] is None else f"{r[m]:.2f}") for m in MODES}
        lines.append(f"| {r['ctx']} | {r['task']} | {r['depth']:.1f} "
                     f"| {f['full']} | {f['ring']} | {f['banked']} |")
    text = "\n".join(lines) + "\n"
    if not smoke:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, "quality_grid.md"), "w") as fh:
            fh.write(text)
        print(f"\nwrote {os.path.join(RESULTS_DIR, 'quality_grid.md')}")
    else:
        print(text)
    return rows


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
    per_doc = {}  # (doc, mode) -> [bin means]
    for dname, used, ids_list in docs:
        ids = torch.tensor([ids_list], dtype=torch.long)
        for mode in MODES:
            cache = P.build_cache(model, mode, make_cfg())
            t0 = time.perf_counter()
            nll = per_token_nll(model, cache, ids, chunk=2048)
            P.free_cache(cache)
            pos = torch.arange(1, ids.shape[1])
            bins = [nll[(pos >= b * bin_size) & (pos < (b + 1) * bin_size)].mean().item()
                    for b in range(n_bins)]
            per_doc[(dname, mode)] = bins
            print(f"[nll {dname:16s} {mode:6s}] mean={nll.mean().item():.4f} "
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

    lines = ["# Position-resolved NLL: full vs ring vs banked (gemma-4-E4B-it)", "",
             f"{len(docs)} natural documents x {doc_len} tokens, teacher-forced with "
             f"chunk-2048 prefill (cache evolves as in generation; divergence "
             f"granularity is one chunk). Modes as in quality_grid.md. Provenance:", ""]
    for dname, used, _ in docs:
        lines.append(f"- {dname}: {', '.join(used[:6])}{' ...' if len(used) > 6 else ''}")
    lines += ["", "| pos bin | full | ring | banked | d(ring-full) ± std | d(banked-full) ± std |",
              "|---|---|---|---|---|---|"]
    for b in range(n_bins):
        dr, sr = delta_stats("ring", b)
        db, sb = delta_stats("banked", b)
        lines.append(f"| {b}k-{b + 1}k | {agg('full', b):.4f} | {agg('ring', b):.4f} "
                     f"| {agg('banked', b):.4f} | {dr:+.4f} ± {sr:.4f} | {db:+.4f} ± {sb:.4f} |")
    dr0 = max(abs(per_doc[(d, "ring")][0] - per_doc[(d, "full")][0]) for d, _, _ in docs)
    db0 = max(abs(per_doc[(d, "banked")][0] - per_doc[(d, "full")][0]) for d, _, _ in docs)
    ok = dr0 < 1e-3 and db0 < 1e-3
    lines += ["", f"Sanity (pre-eviction exactness): bin 0-1k max |delta| vs full: "
              f"ring {dr0:.2e}, banked {db0:.2e} -> {'PASS' if ok else 'FAIL'}", ""]
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
    ap.add_argument("cmd", choices=["grid", "nll"])
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid (ctx 2048) + tiny nll (2 docs x 4k); prints GRID_OK")
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
    else:
        run_nll(model, tok, args)


if __name__ == "__main__":
    main()
