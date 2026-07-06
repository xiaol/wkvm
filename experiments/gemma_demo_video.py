"""gemma-4-E4B recurrent-mode concurrency demo video: real capture -> mp4.

Honesty rule (same as wkvm_demo_video.py): every piece of text and every
number shown comes from a real run recorded into an event log
(experiments/results/gemma_demo_events.json) with wall-clock timestamps.
Time-compressed segments are labeled on screen. The two structural caveats
are printed ON the frames, not just here:

  * this is the PoC path — patched HF transformers cache layers, NOT the
    wkvm engine (no scheduler/arena, no /v1/states);
  * act 2/3 ladder sessions are ONE real prefill replicated across the
    batch dim (honest memory cost, throwaway text). Act 1 is fully real:
    distinct prompts, distinct decoded text.

Three acts:
  A real concurrency    N distinct chat prompts, batched greedy decode in
                        ring mode, per-token wall-clock timestamps
  B ring ladder         virtual sessions @4k ctx: B=8..128 CUDA-graphed
                        decode, flat MiB/slot; one 16k rung for flatness
  C full-KV contrast    identical ladder with stock full-KV cache under
                        the same allocator cap -> where it goes over budget

Usage:
  HF_HUB_OFFLINE=1 python experiments/gemma_demo_video.py capture [--fast]
  python experiments/gemma_demo_video.py render
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "experiments" / "results"
EVENTS_DEFAULT = RESULTS / "gemma_demo_events.json"
MP4_DEFAULT = RESULTS / "gemma_demo.mp4"

PROMPTS_16 = [
    "Write one sentence about the ocean.",
    "Name three prime numbers.",
    "What is the capital of France?",
    "Give a synonym for 'happy'.",
    "What color is chlorophyll?",
    "State Newton's second law briefly.",
    "Name a famous composer.",
    "What is 12 times 12?",
    "What gas do plants absorb from the air?",
    "Name a river that flows through Egypt.",
    "What is the boiling point of water in Celsius?",
    "Give an antonym for 'ancient'.",
    "What planet is known as the red planet?",
    "Name the author of 'Romeo and Juliet'.",
    "What is 7 squared?",
    "What language is spoken in Brazil?",
]

END_IDS = (1, 106)  # <eos>, <end_of_turn>


def now() -> float:
    return time.perf_counter()


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


# =========================================================================
# CAPTURE
# =========================================================================


def act_a_real_batch(model, tok, args, poc) -> dict:
    """B distinct chat prompts, ring cache, batched greedy decode with a
    per-step wall-clock timestamp (sync each step: honest floor)."""
    import torch
    from wkvm_demo_video import decode_stream

    prompts = PROMPTS_16[: args.panes]
    tok.padding_side = "left"
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True, tokenize=False)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    B = input_ids.shape[0]

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cache = poc.build_cache(model, "ring", args)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)

        # the prefill forward produced the first answer token; show it at t=0
        toks = [[tk] for tk in cur[:, 0].tolist()]
        tstamps: list[float] = [0.0]
        counters = []
        torch.cuda.synchronize()
        t0 = now()
        for step in range(args.decode_tokens):
            attn = torch.cat([attn, torch.ones_like(attn[:, :1])], dim=-1)
            out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                        logits_to_keep=1, attention_mask=attn,
                        position_ids=(attn.cumsum(-1) - 1)[:, -1:])
            cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            torch.cuda.synchronize()
            t = now() - t0
            tstamps.append(t)
            row = cur[:, 0].tolist()
            for i, tk in enumerate(row):
                toks[i].append(tk)
            if step % 8 == 7 or step == args.decode_tokens - 1:
                counters.append(dict(
                    t=round(t, 3),
                    tok_s=int(B * (step + 1) / t),
                    vram=round(torch.cuda.memory_allocated() / 2**30, 2),
                    active=B))
    elapsed = tstamps[-1]
    cache_mib = poc.cache_bytes(cache) / 2**20
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    poc.free_cache(cache)

    sessions, events = [], []
    for i, p in enumerate(prompts):
        ids = toks[i]
        cut = next((j for j, tk in enumerate(ids) if tk in END_IDS), len(ids))
        pieces = decode_stream(tok, ids[:cut])
        tp = [[round(tstamps[j], 3), pieces[j]] for j in range(cut)]
        sessions.append(dict(sid=f"s{i:02d}", prompt=p, pieces=tp))
        events.extend(dict(t=round(tstamps[j], 3), sid=f"s{i:02d}",
                           piece=pieces[j]) for j in range(cut))
    agg = B * args.decode_tokens / elapsed
    print(f"[act A] B={B} distinct prompts, {args.decode_tokens} tok/seq: "
          f"{agg:.1f} tok/s agg, cache {cache_mib:.1f} MiB, peak {peak_gib:.2f} GiB")
    return dict(sessions=sessions, events=events, counters=counters,
                span=round(elapsed, 3), agg_tok_s=round(agg, 1),
                cache_mib=round(cache_mib, 1), peak_gib=round(peak_gib, 2),
                decode_tokens=args.decode_tokens)


def run_ladder(model, tok, args, poc, mode: str, ctx: int, ladder: list[int],
               graphs: bool, t_origin: float) -> list[dict]:
    """One concurrency ladder (run_concurrency semantics, JSON rows +
    wall-clock stamps). Virtual sessions: 1 real prefill replicated xB."""
    import torch

    headroom = 1 << 30
    avail = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()
    avail = min(avail, int(args.mem_cap_gib * 2**30))
    word_ids = tok(" one two three four five six seven eight nine ten red blue"
                   " green gold iron salt north south east west",
                   add_special_tokens=False).input_ids
    prompt = poc.build_plain_prompt(tok, ctx)
    steps = args.decode_tokens
    rows = []
    for B in ladder:
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        cache = poc.build_cache(model, mode, args)
        row = dict(B=B, ctx=ctx, mode=mode, ok=False, green=False,
                   agg=None, per_slot_mib=None, cache_mib=None, resv_gib=None,
                   t=None)
        gs = None
        try:
            poc.chunked_prefill(model, cache, prompt, args.chunk)
            first = torch.tensor([[word_ids[i % len(word_ids)]] for i in range(B)],
                                 dtype=torch.long)
            if graphs:
                assert mode == "ring"
                cache = poc.to_static_cache(cache, repeats=B)
                gs = poc.GraphedStep(model, cache, B)
                _, elapsed = poc.graphed_greedy_decode(gs, first, steps)
            else:
                for layer in cache.layers:
                    layer.batch_repeat_interleave(B)
                _, elapsed = poc.greedy_decode(model, cache, first, steps)
            resv = torch.cuda.max_memory_reserved()
            cache_mib = poc.cache_bytes(cache) / 2**20
            row.update(ok=True, agg=round(B * steps / elapsed, 1),
                       per_slot_mib=round(cache_mib / B, 1),
                       cache_mib=round(cache_mib, 1),
                       resv_gib=round(resv / 2**30, 2),
                       green=bool(resv <= avail - headroom))
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and \
                    "out of memory" not in str(exc).lower():
                raise
            gs = None
            torch.cuda.empty_cache()
            row["resv_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        if gs is not None:
            del gs.graph, gs
        poc.free_cache(cache)
        row["t"] = round(now() - t_origin, 2)
        rows.append(row)
        print(f"[ladder {mode} ctx={ctx} B={B}] ok={row['ok']} green={row['green']} "
              f"agg={row['agg']} per_slot={row['per_slot_mib']}MiB "
              f"resv={row['resv_gib']}GiB")
        if not row["ok"]:
            break
    return rows


def cmd_capture(args) -> None:
    import torch
    import gemma_recurrent_poc as poc
    from transformers import AutoTokenizer

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.mem_cap_gib * 2**30 / total))

    path = poc.resolve_model_path(args.model_path)
    t0 = now()
    model = poc.load_model(path, attn="sdpa")
    tok = AutoTokenizer.from_pretrained(path)
    load_s = now() - t0
    print(f"# loaded in {load_s:.1f}s; weights "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    meta = dict(
        model="gemma-4-E4B-it",
        mode="ring (sink 16 + window 1024 on the 4 growing-KV layers)",
        gpu=torch.cuda.get_device_name(0),
        weights_gib=round(torch.cuda.memory_allocated() / 2**30, 2),
        load_s=round(load_s, 2),
        mem_cap_gib=args.mem_cap_gib,
        date=time.strftime("%Y-%m-%d"),
        harness="PoC: patched HF transformers cache — not the wkvm engine",
    )

    tw0 = now()
    poc.warmup(model, tok, args)
    warmup_s = now() - tw0

    A = act_a_real_batch(model, tok, args, poc)
    A["warmup_s"] = round(warmup_s, 1)

    t_origin = now()
    ring_rows = run_ladder(model, tok, args, poc, "ring", args.ctx_per_session,
                           args.ladder, graphs=True, t_origin=t_origin)
    ring16k = run_ladder(model, tok, args, poc, "ring", 16384,
                         [args.b16k], graphs=True, t_origin=t_origin)
    B = dict(replicated_sessions=True, graphs=True, decode_tokens=args.decode_tokens,
             ctx=args.ctx_per_session, rows=ring_rows, rows_16k=ring16k,
             span=round(now() - t_origin, 2))

    t_origin = now()
    full_rows = run_ladder(model, tok, args, poc, "full", args.ctx_per_session,
                           args.full_ladder, graphs=False, t_origin=t_origin)
    full16k = run_ladder(model, tok, args, poc, "full", 16384,
                         args.full_ladder_16k, graphs=False, t_origin=t_origin)
    C = dict(replicated_sessions=True, graphs=False, decode_tokens=args.decode_tokens,
             ctx=args.ctx_per_session, rows=full_rows, rows_16k=full16k,
             span=round(now() - t_origin, 2))

    green = [r for r in ring_rows if r["green"]]
    fgreen = [r for r in full_rows if r["green"]]
    summary = dict(
        act_a_B=len(A["sessions"]), act_a_tok_s=A["agg_tok_s"],
        act_a_cache_mib=A["cache_mib"],
        ring_bmax=max((r["B"] for r in green), default=0),
        ring_best_agg=max((r["agg"] for r in green), default=0),
        ring_slot_mib=green[-1]["per_slot_mib"] if green else None,
        ring_16k_ok=bool(ring16k and ring16k[0]["green"]),
        full_bmax=max((r["B"] for r in fgreen), default=0),
        full_16k_bmax=max((r["B"] for r in full16k if r["green"]), default=0),
    )
    _atomic_write(Path(args.events),
                  dict(meta=meta, A=A, B=B, C=C, summary=summary))
    print(f"CAPTURE_OK {args.events}")
    print(json.dumps(summary, indent=1))


# =========================================================================
# RENDER
# =========================================================================


def cmd_render(args) -> None:
    import subprocess
    from PIL import Image, ImageDraw
    import wkvm_demo_video as R

    ev = json.loads(Path(args.events).read_text())
    meta = ev["meta"]
    W, H = 1280, 720
    fps = 30
    P = R.Painter(W, H, fps)
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
        return Image.new("RGB", (W, H), R.BG)

    def header(d, counter=None, right=None):
        d.rectangle([0, 0, W, 44], fill=R.HDR_BG)
        d.line([0, 44, W, 44], fill=R.BORDER)
        title = f"wkvm recurrent mode — {meta['model']} — {meta['gpu']}"
        d.text((14, 13), title, font=P.font(16, bold=True), fill=R.WHITE)
        if counter is not None:
            s = (f"{counter['tok_s']:,} tok/s   VRAM {counter['vram']:.1f} GiB   "
                 f"sessions {counter['active']}")
            f = P.font(15)
            d.text((W - 20 - f.getlength(s), 14), s, font=f, fill=R.GREEN)
        elif right:
            f = P.font(15)
            d.text((W - 20 - f.getlength(right), 14), right, font=f, fill=R.GREEN)

    GX0, GY0, GX1, GY1 = 8, 52, W - 8, H - 46

    # ---------------- Title card -------------------------------------------
    seg = 4.0
    rows = [
        ("gemma-4-E4B on one RTX 4090", 30, R.WHITE, True),
        ("recurrent-mode serving: constant memory per session", 18, R.FG, False),
        ("", 8, R.DIM, False),
        ("stock full-attention KV grows with context; here the 4 growing-KV "
         "layers", 15, R.DIM, False),
        ("are capped by a sink+ring cache — every session is a fixed ~36 MiB "
         "slot", 15, R.DIM, False),
        ("", 8, R.DIM, False),
        ("every number on screen: measured in this capture "
         f"({meta['date']})", 15, R.GREEN, True),
        ("PoC harness: patched HF transformers cache — NOT the wkvm engine",
         15, R.AMBER, True),
        ("exact until context exceeds the ring window; quality past it is "
         "measured", 13, R.DIM, False),
        ("in docs/RECURRENT_MODE_QUALITY.md — this is a serving-physics demo",
         13, R.DIM, False),
    ]
    for f_i in range(int(seg * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        header(d)
        y = 150
        for s, sz, color, bold in rows:
            if s:
                f = P.font(sz, bold=bold)
                d.text(((W - f.getlength(s)) / 2, y), s, font=f, fill=color)
            y += sz + 16
        emit(img)

    # ---------------- Act A: real concurrent sessions ----------------------
    A = ev["A"]
    nA = len(A["sessions"])
    rects = R.grid_layout(nA, GX0, GY0, GX1, GY1)
    span = A["span"]
    # play at true 1x (decode is fast), then hold the finished text to read
    DA = min(span + 7.0, 22.0)
    factor = 1.0
    size, tsize = 12, 11
    cw = max(8, int((rects[0][2] - rects[0][0] - 8) / P.char_w(size)))
    n_lines = max(2, int((rects[0][3] - rects[0][1] - tsize - 8) / (size + 2)))
    base = blank()
    bd = ImageDraw.Draw(base)
    for r, s in zip(rects, A["sessions"]):
        bd.rectangle(r, fill=R.PANE_BG, outline=R.BORDER)
        bd.text((r[0] + 4, r[1] + 2), f"{s['sid']}  {s['prompt']}"[:cw],
                font=P.font(tsize), fill=R.DIM)
    panes = [R.StreamPane(s["pieces"]) for s in A["sessions"]]
    for f_i in range(int(DA * fps)):
        T = (f_i + 1) / fps * factor
        img = base.copy()
        d = ImageDraw.Draw(img)
        header(d, counter=R.counter_at(A["counters"], T))
        for r, pane in zip(rects, panes):
            pane.advance(T, cw)
            R.draw_pane_text(d, P, r, pane, n_lines, size, tsize + 5)
        R.footer(d, P,
                 f"ACT 1 / REAL CONCURRENCY — {nA} distinct prompts — "
                 f"ring cache {A['cache_mib']:.0f} MiB total",
                 f"real text, real timestamps — warmup ({A['warmup_s']:.0f}s) "
                 "excluded")
        if factor > 1.01:
            R.speed_label(d, P, factor)
        emit(img)

    # ---------------- Act B/C shared ladder drawing -------------------------
    def ladder_scene(act, title_txt, seg_s, bar_color, other=None):
        """Rows reveal in captured order, bar = agg tok/s, chip = MiB/slot."""
        rows_ = act["rows"] + act.get("rows_16k", [])
        max_agg = max([r["agg"] or 0 for r in rows_] +
                      [r["agg"] or 0 for r in (other or [])] + [1])
        span_ = max(r["t"] for r in rows_)
        # normalized reveal: captured order and spacing kept, but the first
        # rung appears ~0.6s in ("ladder compressed" stays on screen)
        tmin = min(r["t"] for r in rows_)
        for r in rows_:
            r["_td"] = 0.6 + (r["t"] - tmin) / max(span_ - tmin, 1e-6) \
                * (seg_s - 3.0)
        fct = 1.0
        X_LBL, X_BAR0, X_BAR1, X_AGG, X_SLOT, X_RESV = 16, 240, 620, 632, 812, 1000
        rh = min(52, (GY1 - GY0 - 40) // max(len(rows_), 1))
        for f_i in range(int(seg_s * fps)):
            T = (f_i + 1) / fps
            img = blank()
            d = ImageDraw.Draw(img)
            header(d)
            d.text((GX0 + 8, GY0 + 2), title_txt, font=P.font(15, bold=True),
                   fill=R.WHITE)
            lbl = f"ladder compressed — real: {span_:.0f}s"
            fL = P.font(13, bold=True)
            d.text((W - 16 - fL.getlength(lbl), GY0 + 3), lbl, font=fL,
                   fill=R.AMBER)
            y = GY0 + 30
            for r in rows_:
                if r["_td"] > T:
                    break
                ty = y + rh // 2 - 14
                d.text((X_LBL, ty), f"B={r['B']:>3} @{r['ctx']//1024:>2}k",
                       font=P.font(15, bold=True), fill=R.FG)
                grow = min(1.0, (T - r["_td"]) / 0.4 + 0.15)
                if r["ok"]:
                    wpx = int((X_BAR1 - X_BAR0) * (r["agg"] / max_agg))
                    d.rectangle([X_BAR0, y + 4, X_BAR0 + max(2, int(wpx * grow)),
                                 y + rh - 14],
                                fill=bar_color if r["green"] else R.AMBER,
                                outline=R.BORDER)
                    d.text((X_AGG, ty), f"{r['agg']:>6,.0f} tok/s",
                           font=P.font(14), fill=R.WHITE)
                    d.text((X_SLOT, ty), f"{r['per_slot_mib']:.1f} MiB/slot",
                           font=P.font(13), fill=R.CYAN)
                    st = "green" if r["green"] else "over-budget"
                    d.text((X_RESV, ty), f"{r['resv_gib']:.1f}G [{st}]",
                           font=P.font(13),
                           fill=R.GREEN if r["green"] else R.AMBER)
                else:
                    d.rectangle([X_BAR0, y + 4, X_BAR0 + 60, y + rh - 14],
                                fill=(60, 24, 24), outline=R.RED)
                    d.text((X_BAR0 + 70, ty),
                           f"OOM  (reserved hit {r['resv_gib']:.1f} GiB cap)",
                           font=P.font(14, bold=True), fill=R.RED)
                y += rh
            R.footer(d, P, act["_caption"],
                     "1 real prefill replicated xB — memory honest, "
                     "text throwaway")
            emit(img)

    B = ev["B"]
    B["_caption"] = (f"ACT 2 / RING LADDER — {B['ctx']}-token sessions, "
                     "CUDA-graphed decode — flat MiB/slot")
    ladder_scene(B, f"ring (sink+window) — {meta['mem_cap_gib']:.0f} GiB "
                    "allocator cap, green = 1 GiB headroom", 16.0, R.GREEN)

    C = ev["C"]
    C["_caption"] = ("ACT 3 / FULL-KV CONTRAST — stock cache, same budget — "
                     "KV grows with B and ctx")
    ladder_scene(C, "stock full-attention KV — same ladder, same budget",
                 12.0, R.CYAN, other=B["rows"])

    # ---------------- side-by-side + end card ------------------------------
    S = ev["summary"]
    seg = 8.0
    rows = [
        ("one 24 GB GPU, one gemma-4-E4B", 26, R.WHITE, True),
        ("", 6, R.DIM, False),
        (f"real sessions (distinct prompts): B={S['act_a_B']} at "
         f"{S['act_a_tok_s']:,.0f} tok/s — {S['act_a_cache_mib']:.0f} MiB cache "
         "total", 16, R.GREEN, True),
        (f"ring ladder: B_max {S['ring_bmax']} green @ "
         f"{S['ring_best_agg']:,.0f} tok/s — {S['ring_slot_mib']:.1f} MiB/slot, "
         f"flat at 16k ctx: {'yes' if S['ring_16k_ok'] else 'no'}",
         16, R.GREEN, True),
        (f"stock full KV, same budget: B_max {S['full_bmax']} @ 4k ctx, "
         f"{S['full_16k_bmax']} @ 16k", 16, R.CYAN, True),
        ("", 6, R.DIM, False),
        ("measured separately in this repo (docs/COMPARISON.md, same GPU):",
         14, R.DIM, False),
        ("vLLM 0.24: 38 sessions @4k / 9 @16k — SGLang 0.5.14: 6 / 1",
         15, R.FG, False),
        ("", 6, R.DIM, False),
        ("caveats: PoC on patched HF transformers, not the wkvm engine;",
         13, R.AMBER, False),
        ("ring is exact below the window — quality past it: "
         "docs/RECURRENT_MODE_QUALITY.md", 13, R.AMBER, False),
        ("", 6, R.DIM, False),
        ("github.com/xiaol/wkvm", 20, R.CYAN, True),
    ]
    for f_i in range(int(seg * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        y = 108
        for s, sz, color, bold in rows:
            if s:
                f = P.font(sz, bold=bold)
                d.text(((W - f.getlength(s)) / 2, y), s, font=f, fill=color)
            y += sz + 15
        d.text((14, H - 26),
               f"all numbers measured — {meta['date']} — {meta['gpu']}",
               font=P.font(12), fill=R.DIMMER)
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
    c.add_argument("--fast", action="store_true")
    c.add_argument("--panes", type=int, default=16)
    c.add_argument("--decode-tokens", type=int, default=96)
    c.add_argument("--ladder", type=int, nargs="*",
                   default=[8, 16, 32, 64, 96, 128])
    c.add_argument("--b16k", type=int, default=96)
    c.add_argument("--full-ladder", type=int, nargs="*",
                   default=[8, 16, 32, 64])
    c.add_argument("--full-ladder-16k", type=int, nargs="*", default=[8, 16])
    c.add_argument("--ctx-per-session", type=int, default=4096)
    c.add_argument("--chunk", type=int, default=2048)
    c.add_argument("--sink", type=int, default=16)
    c.add_argument("--window", type=int, default=1024)
    c.add_argument("--mem-cap-gib", type=float,
                   default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)))
    c.add_argument("--model-path", default=None)
    c.add_argument("--events", default=str(EVENTS_DEFAULT))
    c.set_defaults(fn=cmd_capture)

    r = sub.add_parser("render")
    r.add_argument("--events", default=str(EVENTS_DEFAULT))
    r.add_argument("--out", default=str(MP4_DEFAULT))
    r.set_defaults(fn=cmd_render)

    args = ap.parse_args()
    if args.cmd == "capture" and args.fast:
        args.panes = 8
        args.decode_tokens = 32
        args.ladder = [8, 16]
        args.b16k = 16
        args.full_ladder = [8, 16]
        args.full_ladder_16k = [8]
    args.fn(args)


if __name__ == "__main__":
    main()
