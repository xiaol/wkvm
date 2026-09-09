"""RULER-lite: the RULER synthetic long-context tasks, ported onto local data.

Why a port: the official generators (github.com/NVIDIA/RULER, scripts/data/
synthetic) download Paul Graham essays and nltk data from hosts this box
cannot reach. This file keeps RULER's task templates, answer prefixes,
needle/chain/word-list constructions, depth sampling and metrics
(``string_match_all`` / ``string_match_part``), and swaps the essay haystack
for SQuAD contexts (English prose, local parquet). Numbers are therefore
"RULER-lite": same task family, different haystack text, fewer samples.

Tasks (RULER names): niah_single_1/2/3, niah_multikey_1/2/3, niah_multivalue,
niah_multiquery, vt, cwe, fwe, qa_1.

The point for wkvm: the hybrid's guest attention is exact within the paged
pool, so wkvm's outputs must equal HF's token for token (greedy) at every
length — the score is the model's, the agreement is the engine's.

  PYTHONPATH=. python experiments/ruler_lite.py --lengths 4096 8192 16384 --samples 20 \
      --engine wkvm --device cuda:0 --json experiments/results/ruler_lite_wkvm.json
  PYTHONPATH=. python experiments/ruler_lite.py ... --engine hf --device cuda:1 --json ..._hf.json
  PYTHONPATH=. python experiments/ruler_lite.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

SQUAD = "/root/x/data/squad/validation-00000-of-00001.parquet"

TEMPLATES = {
    "niah": (
        "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
        "I will quiz you about the {type_needle_v} afterwards.\n{context}\n"
        "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?",
        " The special magic {type_needle_v} for {query} mentioned in the provided text are",
    ),
    "vt": (
        "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
        "Question: Find all variables that are assigned the value {query} in the text above.",
        " Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are "
        "assigned the value {query}, they are: ",
    ),
    "cwe": (
        "Below is a numbered list of words. In these words, some appear more often than others. "
        "Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?",
        " Answer: The top 10 words that appear most often in the list are:",
    ),
    "fwe": (
        "Read the following coded text and track the frequency of each coded word. Find the three most frequently "
        "appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. "
        "What are the three most frequently appeared words in the above coded text?",
        " Answer: According to the coded text above, the three most frequently appeared words are:",
    ),
    "qa": (
        "Answer the question based on the given documents. Only give me the answer and do not output any other words."
        "\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\nQuestion: {query}",
        " Answer:",
    ),
}

# RULER's synthetic.yaml task configurations.
TASKS = {
    "niah_single_1": dict(kind="niah", haystack="noise", k="words", v="numbers", nk=1, nv=1, nq=1, gen=128),
    "niah_single_2": dict(kind="niah", haystack="essay", k="words", v="numbers", nk=1, nv=1, nq=1, gen=128),
    "niah_single_3": dict(kind="niah", haystack="essay", k="words", v="uuids", nk=1, nv=1, nq=1, gen=128),
    "niah_multikey_1": dict(kind="niah", haystack="essay", k="words", v="numbers", nk=4, nv=1, nq=1, gen=128),
    "niah_multikey_2": dict(kind="niah", haystack="needle", k="words", v="numbers", nk=1, nv=1, nq=1, gen=128),
    "niah_multikey_3": dict(kind="niah", haystack="needle", k="uuids", v="uuids", nk=1, nv=1, nq=1, gen=128),
    "niah_multivalue": dict(kind="niah", haystack="essay", k="words", v="numbers", nk=1, nv=4, nq=1, gen=128),
    "niah_multiquery": dict(kind="niah", haystack="essay", k="words", v="numbers", nk=1, nv=1, nq=4, gen=128),
    "vt": dict(kind="vt", chains=1, hops=4, gen=30),
    "cwe": dict(kind="cwe", freq_cw=30, freq_ucw=3, num_cw=10, gen=120),
    "fwe": dict(kind="fwe", alpha=2.0, wordlen=6, gen=50),
    "qa_1": dict(kind="qa", gen=32),
}

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
DEPTHS = [round(x) for x in [i * 100 / 39 for i in range(40)]]


# -- local corpus ----------------------------------------------------------------------

def load_squad():
    import pyarrow.parquet as pq

    t = pq.read_table(SQUAD).to_pylist()
    docs = sorted({r["context"] for r in t})
    idx = {c: i for i, c in enumerate(docs)}
    by_title: dict[str, list[int]] = {}
    for r in t:
        by_title.setdefault(r["title"], [])
        if idx[r["context"]] not in by_title[r["title"]]:
            by_title[r["title"]].append(idx[r["context"]])
    qas = [{"query": r["question"], "outputs": list(r["answers"]["text"]), "context": [idx[r["context"]]],
            "more_context": [i for i in by_title[r["title"]] if i != idx[r["context"]]]} for r in t]
    return docs, qas


def sent_tokenize(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class Words:
    def __init__(self, seed: int) -> None:
        import wonderwords

        self.nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        self.adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        self.verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
        self.pairs = sorted({f"{a}-{n}" for a in self.adjs for n in self.nouns})
        self.all = sorted(set(self.nouns + self.adjs + self.verbs))
        random.Random(seed).shuffle(self.all)


# -- generators (faithful to RULER's scripts, haystack swapped) -----------------------------

class Generator:
    def __init__(self, tok, seed: int = 42) -> None:
        self.tok = tok
        self.seed = seed
        self.rng = random.Random(seed)
        self.words = Words(seed)
        self.docs, self.qas = load_squad()
        self.essay = re.sub(r"\s+", " ", " ".join(self.docs)).split(" ")

    def ntok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def rand(self, kind: str) -> str:
        if kind == "numbers":
            return str(self.rng.randint(10**6, 10**7 - 1))
        if kind == "words":
            return self.rng.choice(self.words.pairs)
        if kind == "uuids":
            return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))
        raise ValueError(kind)

    # niah -------------------------------------------------------------------------------
    def niah(self, cfg, num_haystack: int):
        nk = max(cfg["nk"], cfg["nq"])
        keys, values, needles = [], [], []
        for _ in range(nk):
            keys.append(self.rand(cfg["k"]))
            vals = []
            for _ in range(cfg["nv"]):
                vals.append(self.rand(cfg["v"]))
                needles.append(NEEDLE.format(type_needle_v=cfg["v"], key=keys[-1], value=vals[-1]))
            values.append(vals)
        random.Random(self.seed).shuffle(needles)
        if cfg["haystack"] == "essay":
            hay = self.essay
            text = " ".join(hay[:num_haystack]) if num_haystack <= len(hay) else " ".join((hay * (num_haystack // len(hay) + 1))[:num_haystack])
            sents = sent_tokenize(text)
            pos = [0] + sorted(int(len(sents) * d / 100) for d in self.rng.sample(DEPTHS, len(needles))) + [len(sents)]
            parts = []
            for i in range(1, len(pos)):
                parts.append(" ".join(sents[pos[i - 1]:pos[i]]))
                if i - 1 < len(needles):
                    parts.append(needles[i - 1])
            context = " ".join(parts)
        else:
            if cfg["haystack"] == "noise":
                sents = [NOISE] * num_haystack
            else:
                sents = [NEEDLE.format(type_needle_v=cfg["v"], key=self.rand(cfg["k"]), value=self.rand(cfg["v"]))
                         for _ in range(num_haystack)]
            for index, element in zip(sorted(self.rng.sample(range(num_haystack), len(needles)), reverse=True), needles):
                sents.insert(index, element)
            context = "\n".join(sents)
        qi = self.rng.sample(range(nk), cfg["nq"])
        queries = [keys[i] for i in qi]
        answers = [a for i in qi for a in values[i]]
        query = ", ".join(queries[:-1]) + ", and " + queries[-1] if len(queries) > 1 else queries[0]
        template, prefix = TEMPLATES["niah"]
        tv = cfg["v"]
        if cfg["nq"] * cfg["nv"] == 1:
            template = template.replace("Some", "A").replace("are all", "is").replace("are", "is")
            prefix = prefix.replace("are", "is")
            tv = tv[:-1]
        return (template.format(type_needle_v=tv, context=context, query=query),
                prefix.format(type_needle_v=tv, query=query), answers)

    # variable tracking ------------------------------------------------------------------
    def vt(self, cfg, num_noise: int):
        chains, hops = cfg["chains"], cfg["hops"]
        vars_all = [''.join(self.rng.choices(string.ascii_uppercase, k=5)) for _ in range((hops + 1) * chains)]
        while len(set(vars_all)) < chains * (hops + 1):
            vars_all.append(''.join(self.rng.choices(string.ascii_uppercase, k=5)))
        chain_list, var_list = [], []
        for i in range(0, len(vars_all), hops + 1):
            v = vars_all[i:i + hops + 1]
            var_list.append(v)
            chain = [f"VAR {v[0]} = {self.rng.randint(10000, 99999)}"]
            for j in range(hops):
                chain.append(f"VAR {v[j + 1]} = VAR {v[j]} ")
            chain_list.append(chain)
        value = chain_list[0][0].split("=")[-1].strip()
        sents = [NOISE] * num_noise
        for chain in chain_list:
            positions = sorted(self.rng.sample(range(len(sents)), len(chain)))
            for p, j in zip(positions, range(len(chain))):
                sents.insert(p + j, chain[j])
        context = "\n".join(sents).replace(". \n", ".\n")
        template, prefix = TEMPLATES["vt"]
        return (template.format(context=context, query=value),
                prefix.format(num_v=hops + 1, query=value), var_list[0])

    # common words extraction --------------------------------------------------------------
    def cwe(self, cfg, num_words: int):
        def example(n, common_repeats, uncommon_repeats, common_nums):
            pool = self.rng.sample(self.words.all, min(n, len(self.words.all)))
            common, uncommon = pool[:common_nums], pool[common_nums:]
            lst = common * common_repeats + uncommon * uncommon_repeats
            random.Random(self.seed).shuffle(lst)
            return " ".join(f"{i + 1}. {w}" for i, w in enumerate(lst)), common
        template, prefix = TEMPLATES["cwe"]
        shot_ctx, shot_ans = example(40, 10, 3, cfg["num_cw"])
        shot = template.format(context=shot_ctx) + prefix + " " + " ".join(f"{i + 1}. {w}" for i, w in enumerate(shot_ans))
        context, answer = example(num_words, cfg["freq_cw"], cfg["freq_ucw"], cfg["num_cw"])
        return shot + "\n" + template.format(context=context), prefix, answer

    # frequent words extraction -------------------------------------------------------------
    def fwe(self, cfg, num_words: int, vocab_size: int):
        from math import fsum

        vocab = sorted({''.join(self.rng.choices(string.ascii_lowercase, k=cfg["wordlen"])) for _ in range(vocab_size * 2)})[:vocab_size]
        random.Random(self.seed).shuffle(vocab)
        vocab[0] = "..."
        alpha = cfg["alpha"]
        zeta = fsum(k ** -alpha for k in range(1, 100000))
        counts = [int(num_words * (k ** -alpha) / zeta) for k in range(1, len(vocab) + 1)]
        sampled = [w for w, c in zip(vocab, counts) for _ in range(c)]
        random.Random(self.seed).shuffle(sampled)
        template, prefix = TEMPLATES["fwe"]
        return template.format(context=" ".join(sampled)), prefix, vocab[1:4]

    # qa ------------------------------------------------------------------------------------
    def qa(self, index: int, num_docs: int):
        q = self.qas[index]
        cur, more = q["context"], q["more_context"]
        if num_docs - len(cur) > len(more):
            others = [i for i in range(len(self.docs)) if i not in cur and i not in more]
            all_docs = cur + more + self.rng.sample(others, max(0, num_docs - len(cur) - len(more)))
        else:
            all_docs = cur + self.rng.sample(more, num_docs - len(cur))
        all_docs = [self.docs[i] for i in all_docs]
        random.Random(self.seed).shuffle(all_docs)
        context = "\n\n".join(f"Document {i + 1}:\n{d}" for i, d in enumerate(all_docs))
        template, prefix = TEMPLATES["qa"]
        return template.format(context=context, query=q["query"]), prefix, q["outputs"]

    # sizing: binary search the haystack size so prompt + generation fits --------------------
    def fit(self, make, max_len: int, gen: int, lo: int, hi: int) -> int:
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            text, prefix, _ = make(mid)
            if self.ntok(text + prefix) + gen <= max_len:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return best

    def samples(self, task: str, max_len: int, n: int):
        cfg = TASKS[task]
        gen = cfg["gen"]
        budget = max_len - 64  # chat-template overhead
        out = []
        if cfg["kind"] == "niah":
            make = lambda h: self.niah(cfg, h)
            per = self.ntok(make(100)[0]) / 100
            size = self.fit(make, budget, gen, 1, int(budget / per * 3))
        elif cfg["kind"] == "vt":
            make = lambda h: self.vt(cfg, h)
            per = self.ntok(make(100)[0]) / 100
            size = self.fit(make, budget, gen, 1, int(budget / per * 3))
        elif cfg["kind"] == "cwe":
            make = lambda h: self.cwe(cfg, h)
            per = self.ntok(make(1000)[0]) / 1000
            size = self.fit(make, budget, gen, 10, int(budget / per * 2))
        elif cfg["kind"] == "fwe":
            vocab_size = (budget - gen) // 50
            make = lambda h: self.fwe(cfg, h, vocab_size)
            per = self.ntok(make(2000)[0]) / 2000
            size = self.fit(make, budget, gen, 10, int(budget / per * 2))
        else:  # qa
            make = None
        for i in range(n):
            if cfg["kind"] == "qa":
                qi = self.rng.randrange(len(self.qas))
                mk = lambda d, qi=qi: self.qa(qi, d)
                size = self.fit(mk, budget, gen, 1, 400)
                text, prefix, answers = mk(size)
            else:
                text, prefix, answers = make(size)
                while self.ntok(text + prefix) + gen > budget and size > 1:
                    size = max(1, int(size * 0.95))
                    text, prefix, answers = make(size)
            out.append({"task": task, "max_len": max_len, "index": i, "input": text, "answer_prefix": prefix,
                        "outputs": answers, "gen": gen})
        return out


# -- metrics (RULER eval) --------------------------------------------------------------------

def string_match_all(pred: str, refs: list[str]) -> float:
    return sum(r.lower() in pred.lower() for r in refs) / len(refs)


def string_match_part(pred: str, refs: list[str]) -> float:
    return float(any(r.lower() in pred.lower() for r in refs))


def score(task: str, pred: str, refs: list[str]) -> float:
    return string_match_part(pred, refs) if task.startswith("qa") else string_match_all(pred, refs)


# -- engines -----------------------------------------------------------------------------------

def render(tok, sample) -> list[int]:
    ids = tok.apply_chat_template([{"role": "user", "content": sample["input"]}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "ids"):
        ids = ids.ids
    return list(ids) + tok(sample["answer_prefix"], add_special_tokens=False)["input_ids"]


def run_wkvm(samples, tok, args):
    import torch

    from wkvm.core.config import SchedulerConfig
    from wkvm.core.request import Request
    from wkvm.engine import Engine

    max_len = max(s["max_len"] for s in samples)
    slots = args.batch
    graphs = {"batch_buckets": tuple(b for b in (1, 2, 4, 8) if b <= slots),
              "length_buckets": tuple(l for l in (4096, 8192, 16384, 32768, 65536) if l <= max_len + 256) or (4096,)}
    engine = Engine.from_qwen35(
        args.model, num_slots=slots, guest_pool_tokens=slots * (max_len + 256), device=args.device,
        stop_token_ids=frozenset({tok.eos_token_id}), prefill_chunk=1024,
        scheduler_config=SchedulerConfig(max_tokens_per_step=16384, max_running_requests=slots,
                                         max_tokens_per_request_per_step=1024),
        cuda_graphs=graphs if not args.no_graphs else False,
        guest_mode=args.guest_mode, sink_tokens=args.sink_tokens, ring_tokens=args.ring_tokens,
    )
    results = []
    for start in range(0, len(samples), slots):
        chunk = samples[start:start + slots]
        reqs = [Request(prompt_token_ids=render(tok, s), max_new_tokens=s["gen"]) for s in chunk]
        t0 = time.time()
        for r in reqs:
            engine.add_request(r)
        while engine.has_unfinished:
            engine.step()
        dt = time.time() - t0
        for s, r in zip(chunk, reqs):
            results.append({**{k: s[k] for k in ("task", "max_len", "index", "outputs")},
                            "prompt_tokens": r.num_prompt_tokens, "output_ids": list(r.output_token_ids),
                            "pred": tok.decode(r.output_token_ids, skip_special_tokens=True), "batch_wall_s": dt})
        print(f"[wkvm] {chunk[0]['task']}@{chunk[0]['max_len']} {start + len(chunk)}/{len(samples)} "
              f"{dt:.1f}s score={sum(score(x['task'], x['pred'], x['outputs']) for x in results[-len(chunk):]) / len(chunk):.2f}",
              flush=True)
    return results, {"graph_stats": dict(engine.runner.graphs.stats) if engine.runner.graphs else None}


def run_hf(samples, tok, args):
    import sys

    import torch

    sys.path.insert(0, "/root/x/RNN-StateTuning/src")
    from rnn_state_tuning.modeling import load_qwen35_model

    model = load_qwen35_model(args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa")
    model = model.to(args.device).eval().requires_grad_(False)
    results = []
    for i, s in enumerate(samples):
        ids = render(tok, s)
        t = torch.tensor([ids], device=args.device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids=t, attention_mask=torch.ones_like(t), max_new_tokens=s["gen"],
                                 do_sample=False, use_cache=True, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen = out[0, len(ids):].tolist()
        dt = time.time() - t0
        results.append({**{k: s[k] for k in ("task", "max_len", "index", "outputs")}, "prompt_tokens": len(ids),
                        "output_ids": gen, "pred": tok.decode(gen, skip_special_tokens=True), "batch_wall_s": dt})
        if (i + 1) % 5 == 0 or i + 1 == len(samples):
            print(f"[hf] {s['task']}@{s['max_len']} {i + 1}/{len(samples)} {dt:.1f}s", flush=True)
    return results, {}


def summarize(results):
    table: dict = {}
    for r in results:
        key = (r["task"], r["max_len"])
        table.setdefault(key, []).append(score(r["task"], r["pred"], r["outputs"]))
    return {f"{t}@{l}": {"score": sum(v) / len(v), "n": len(v)} for (t, l), v in sorted(table.items())}


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    ra = {(r["task"], r["max_len"], r["index"]): r for r in a["results"]}
    rb = {(r["task"], r["max_len"], r["index"]): r for r in b["results"]}
    keys = sorted(set(ra) & set(rb))
    mismatched_prompts = [k for k in keys if ra[k]["prompt_tokens"] != rb[k]["prompt_tokens"]]
    if mismatched_prompts:
        print(f"WARNING: {len(mismatched_prompts)}/{len(keys)} samples have different prompt token counts "
              "(the two runs did not see the same prompts; regenerate from one cache)")
    by: dict = {}
    for k in keys:
        same = ra[k]["output_ids"] == rb[k]["output_ids"]
        by.setdefault((k[0], k[1]), []).append(same)
    print(f"{'task@len':<24} {'n':>3} {'identical':>9} {a['engine']:>8} {b['engine']:>8}")
    tot = 0
    for (t, l), v in sorted(by.items()):
        sa, sb = a["summary"][f"{t}@{l}"]["score"], b["summary"][f"{t}@{l}"]["score"]
        print(f"{t + '@' + str(l):<24} {len(v):>3} {sum(v):>9} {sa:>8.2f} {sb:>8.2f}")
        tot += sum(v)
    print(f"identical outputs: {tot}/{len(keys)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/x/Qwen3.5-9B")
    ap.add_argument("--engine", choices=("wkvm", "hf"), default="wkvm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--guest-mode", choices=("paged", "ring"), default="paged",
                    help="wkvm guest memory: exact paged pool, or sink+ring window (approximate beyond the window)")
    ap.add_argument("--sink-tokens", type=int, default=16)
    ap.add_argument("--ring-tokens", type=int, default=1024)
    ap.add_argument("--json", default=None)
    ap.add_argument("--data-cache", default="experiments/results/ruler_lite_data.json")
    ap.add_argument("--compare", nargs=2, default=None)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    from wkvm.runner.kernels import select_kernels

    kernels = select_kernels()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    cache = Path(args.data_cache)
    # The cache always holds ALL tasks for (seed, lengths, samples); --tasks
    # only filters it, so every engine sees byte-identical prompts.
    key = f"{args.seed}:{','.join(map(str, args.lengths))}:{args.samples}"
    data = json.load(open(cache)) if cache.exists() else {}
    if data.get("key") != key:
        gen = Generator(tok, args.seed)
        samples = []
        for L in args.lengths:
            for task in TASKS:
                t0 = time.time()
                samples.extend(gen.samples(task, L, args.samples))
                print(f"generated {task}@{L}: {args.samples} samples, {time.time() - t0:.1f}s, "
                      f"~{gen.ntok(samples[-1]['input'])} tokens", flush=True)
        data = {"key": key, "samples": samples}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
    wanted = set(args.tasks)
    samples = [s for s in data["samples"] if s["task"] in wanted]
    # group by (max_len, task) so batches share a length bucket
    samples.sort(key=lambda s: (s["max_len"], s["task"], s["index"]))
    t0 = time.time()
    results, extra = (run_wkvm if args.engine == "wkvm" else run_hf)(samples, tok, args)
    summary = summarize(results)
    for k, v in summary.items():
        print(f"{k:<24} {v['score']:.3f} (n={v['n']})")
    payload = {"engine": args.engine, "kernels": kernels, "cuda_graphs": not args.no_graphs and args.engine == "wkvm",
               "guest_mode": args.guest_mode if args.engine == "wkvm" else "exact",
               "sink_tokens": args.sink_tokens, "ring_tokens": args.ring_tokens,
               "model": args.model, "lengths": args.lengths, "samples_per_task": args.samples, "seed": args.seed,
               "haystack": "squad-validation-contexts (RULER essay haystack replaced)", "wall_s": time.time() - t0,
               "summary": summary, "results": results, **extra}
    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False))
    print("RULER_LITE_OK")


if __name__ == "__main__":
    main()
