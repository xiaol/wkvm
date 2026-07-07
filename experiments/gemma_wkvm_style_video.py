"""Render a wkvm-demo-style Gemma recurrent-mode comparison video.

This script is render-only. It combines already captured artifacts:

  * experiments/results/gemma_demo_events.json
  * experiments/results/long_gen_13824_512_*.json
  * experiments/results/quality_grid.md
  * experiments/results/both_quality_concurrency_plan.md

The video is intended to answer two questions at once:

  1. What concurrency did the Gemma recurrent PoC show?
  2. What happens on a long prompt with a long output versus vLLM/SGLang?

Measured numbers are labeled as measured. Routed-span capacity is labeled as a
derived target because there is not yet a native wkvm Gemma routed-span runner.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
RESULTS = EXP / "results"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT))

EVENTS_DEFAULT = RESULTS / "gemma_demo_events.json"
OUT_DEFAULT = RESULTS / "gemma_wkvm_style_demo.mp4"
LONG_RUNS = {
    "wkvm:ring": RESULTS / "long_gen_13824_512_wkvm_ring.json",
    "vLLM": RESULTS / "long_gen_13824_512_vllm.json",
    "SGLang": RESULTS / "long_gen_13824_512_sglang.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def short_num(x: float | int | None, digits: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return f"{x:,}"
    return f"{x:,.{digits}f}"


def clean_text(s: str) -> str:
    # Keep display robust and ASCII-like inside the mono dashboard.
    return (s.replace("\r", "")
             .replace("\t", "  ")
             .replace("\u2014", "-")
             .replace("\u2013", "-")
             .replace("\u2018", "'")
             .replace("\u2019", "'")
             .replace("\u201c", '"')
             .replace("\u201d", '"'))


def wrap_chars(s: str, width: int) -> list[str]:
    out: list[str] = []
    for para in clean_text(s).splitlines() or [""]:
        if not para:
            out.append("")
            continue
        out.extend(textwrap.wrap(para, width=width, break_long_words=False,
                                 replace_whitespace=False) or [""])
    return out


def chunk_text_for_replay(text: str, max_chunk: int = 16) -> list[str]:
    """Chunk completed output text for visual replay.

    The source JSON has the full output text and token count, but not
    per-token detokenized pieces. These display chunks are only for visual
    replay; token progress is still driven by measured output-token counts.
    """
    text = clean_text(text)
    parts = re.findall(r"\s+|[^\s]+", text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        if "\n" in part:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(part)
            continue
        if len(buf) + len(part) > max_chunk and buf:
            chunks.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        chunks.append(buf)
    return chunks or [""]


def replay_events(text: str, duration: float, max_chunk: int = 16) -> list[list]:
    chunks = chunk_text_for_replay(text, max_chunk=max_chunk)
    if len(chunks) == 1:
        return [[0.0, chunks[0]]]
    return [[round(i * duration / (len(chunks) - 1), 3), c]
            for i, c in enumerate(chunks)]


def facts_found(output: str) -> list[str]:
    text = output.lower()
    facts = []
    if "blue-742" in text:
        facts.append("code")
    if "samarkand" in text:
        facts.append("city")
    if "lantern" in text:
        facts.append("checksum")
    return facts


def draw_center(d, P, y: int, text: str, size: int, fill, bold: bool = False) -> int:
    f = P.font(size, bold=bold)
    d.text(((P.W - f.getlength(text)) / 2, y), text, font=f, fill=fill)
    return y + size + 16


def draw_box(d, rect, fill, outline, width: int = 1):
    d.rectangle(rect, fill=fill, outline=outline, width=width)


def draw_panel_title(d, P, rect, title: str, right: str | None = None, fill=None):
    import wkvm_demo_video as R

    x0, y0, x1, _ = rect
    d.text((x0 + 8, y0 + 6), title, font=P.font(14, bold=True),
           fill=fill or R.WHITE)
    if right:
        f = P.font(12, bold=True)
        d.text((x1 - 8 - f.getlength(right), y0 + 8), right, font=f,
               fill=R.GREEN)


def draw_wrapped(d, P, xy, text: str, width_chars: int, size: int, fill,
                 bold: bool = False, line_gap: int = 4,
                 max_lines: int | None = None) -> int:
    x, y = xy
    lines = wrap_chars(text, width_chars)
    if max_lines is not None:
        lines = lines[:max_lines]
    for line in lines:
        d.text((x, y), line, font=P.font(size, bold=bold), fill=fill)
        y += size + line_gap
    return y


def draw_header(d, P, meta: dict, title: str, right: str | None = None):
    import wkvm_demo_video as R

    d.rectangle([0, 0, P.W, 44], fill=R.HDR_BG)
    d.line([0, 44, P.W, 44], fill=R.BORDER)
    head = f"wkvm-style Gemma demo - {meta['model']} - {meta['gpu']}"
    d.text((14, 13), head, font=P.font(16, bold=True), fill=R.WHITE)
    d.text((14, 34), title, font=P.font(9), fill=R.DIMMER)
    if right:
        f = P.font(14, bold=True)
        d.text((P.W - 14 - f.getlength(right), 14), right, font=f, fill=R.GREEN)


def draw_footer(d, P, left: str, right: str | None = None):
    import wkvm_demo_video as R

    y = P.H - 40
    d.rectangle([0, y, P.W, P.H], fill=R.HDR_BG)
    d.line([0, y, P.W, y], fill=R.BORDER)
    d.text((14, y + 7), left, font=P.font(14, bold=True), fill=R.CYAN)
    if right:
        f = P.font(12)
        d.text((P.W - 14 - f.getlength(right), y + 10), right, font=f,
               fill=R.DIM)


def draw_bar(d, P, x0: int, y0: int, w: int, h: int, value: float,
             max_value: float, fill, label: str, outline):
    filled = int(w * min(1.0, max(0.0, value / max_value)))
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(20, 23, 30), outline=outline)
    if filled:
        d.rectangle([x0, y0, x0 + filled, y0 + h], fill=fill)
    d.text((x0 + w + 10, y0 - 1), label, font=P.font(13), fill=fill)


def row_by_ctx(rows: list[dict], ctx: int, b: int | None = None) -> list[dict]:
    out = [r for r in rows if r.get("ctx") == ctx]
    if b is not None:
        out = [r for r in out if r.get("B") == b]
    return out


def title_card(emit, blank, P, ev: dict, duration: float, fps: int):
    import wkvm_demo_video as R
    from PIL import ImageDraw

    meta = ev["meta"]
    for _ in range(int(duration * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header(d, P, meta, "concurrency + throughput + long-output audit")
        y = 125
        y = draw_center(d, P, y, "Gemma recurrent-mode serving", 32, R.WHITE, True)
        y = draw_center(d, P, y, "RWKV-demo dashboard style, Gemma data", 18, R.FG)
        y += 10
        rows = [
            ("measured: 16 distinct prompts, ring/full ladders, vLLM/SGLang long-output runs", R.GREEN),
            ("measured: 13,824-token prompt -> 512 output tokens, ignore_eos=True", R.CYAN),
            ("caveat: current Gemma path is patched HF cache PoC, not native wkvm.engine", R.AMBER),
            ("caveat: routed-span-m64 capacity is derived until native runner is implemented", R.AMBER),
        ]
        for text, color in rows:
            y = draw_center(d, P, y + 6, text, 15, color, color == R.GREEN)
        y += 18
        draw_center(d, P, y, f"sources captured {meta['date']} on {meta['gpu']}",
                    14, R.DIM)
        draw_footer(d, P, "ACT 0 / SOURCE OF TRUTH",
                    "JSON + markdown artifacts under experiments/results")
        emit(img)


def act_real_concurrency(emit, blank, P, ev: dict, duration: float, fps: int):
    import wkvm_demo_video as R
    from PIL import ImageDraw

    meta = ev["meta"]
    A = ev["A"]
    sessions = A["sessions"]
    n = len(sessions)
    GX0, GY0, GX1, GY1 = 8, 52, P.W - 8, P.H - 46
    rects = R.grid_layout(n, GX0, GY0, GX1, GY1)
    size, title_size = 12, 10
    cw = max(8, int((rects[0][2] - rects[0][0] - 8) / P.char_w(size)))
    n_lines = max(2, int((rects[0][3] - rects[0][1] - title_size - 8) /
                         (size + 2)))
    real_span = max(float(A["span"]), 1e-6)
    hold_s = 3.2
    display_decode_s = max(5.5, duration - hold_s)
    speed = real_span / display_decode_s

    base = blank()
    bd = ImageDraw.Draw(base)
    for r, s in zip(rects, sessions):
        draw_box(bd, r, R.PANE_BG, R.BORDER)
        title = f"{s['sid']}  {s['prompt']}"
        bd.text((r[0] + 4, r[1] + 2), title[:cw], font=P.font(title_size),
                fill=R.DIM)

    panes = [R.StreamPane(s["pieces"]) for s in sessions]
    frames = int(duration * fps)
    for f_i in range(frames):
        T = (f_i + 1) / fps
        real_T = min(real_span, T * speed)
        img = base.copy()
        d = ImageDraw.Draw(img)
        counter = R.counter_at(A["counters"], real_T)
        right = (f"{int(counter['tok_s']):,} tok/s  "
                 f"{counter['active']} sessions  "
                 f"{counter['vram']:.1f} GiB") if counter else None
        draw_header(d, P, meta, "real distinct Gemma prompts streaming", right)
        for r, pane in zip(rects, panes):
            pane.advance(real_T, cw)
            R.draw_pane_text(d, P, r, pane, n_lines, size, title_size + 5)
        draw_footer(
            d, P,
            f"ACT 1 / REAL CONCURRENCY - B={n}, {A['decode_tokens']} decode steps",
            f"slowed for readability; real span {A['span']:.3f}s, cache {A['cache_mib']:.1f} MiB",
        )
        emit(img)


def act_concurrency_ladder(emit, blank, P, ev: dict, duration: float, fps: int):
    import wkvm_demo_video as R
    from PIL import ImageDraw

    meta = ev["meta"]
    B = ev["B"]
    C = ev["C"]
    ring_rows = B["rows"] + B.get("rows_16k", [])
    full_rows = C["rows"] + C.get("rows_16k", [])
    max_agg = max([r.get("agg") or 0 for r in ring_rows + full_rows] + [1])
    max_cap = 100

    cap_rows = [
        ("ring PoC", "measured", 96, 96, 96, R.GREEN),
        ("routed-span-m64", "derived target", 43, 43, 43, R.AMBER),
        ("vLLM full KV", "measured/derived", 38, 9, 4, R.CYAN),
        ("SGLang full KV", "measured", 6, 1, 0, R.RED),
    ]
    ladder = [
        ("ring @4k", r["B"], r.get("agg"), r.get("per_slot_mib"),
         r.get("resv_gib"), r.get("green"), r.get("ok"), R.GREEN)
        for r in B["rows"]
    ] + [
        ("ring @16k", r["B"], r.get("agg"), r.get("per_slot_mib"),
         r.get("resv_gib"), r.get("green"), r.get("ok"), R.GREEN)
        for r in B.get("rows_16k", [])
    ] + [
        ("full @4k", r["B"], r.get("agg"), r.get("per_slot_mib"),
         r.get("resv_gib"), r.get("green"), r.get("ok"), R.CYAN)
        for r in C["rows"]
    ] + [
        ("full @16k", r["B"], r.get("agg"), r.get("per_slot_mib"),
         r.get("resv_gib"), r.get("green"), r.get("ok"), R.CYAN)
        for r in C.get("rows_16k", [])
    ]
    reveal = max(1, math.ceil(len(ladder) / (duration - 2.0)))

    for f_i in range(int(duration * fps)):
        T = (f_i + 1) / fps
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header(d, P, meta, "measured concurrency ladder and capacity comparison")

        left = (18, 62, 746, 676)
        right = (764, 62, 1262, 676)
        draw_box(d, left, R.PANE_BG, R.BORDER)
        draw_box(d, right, R.PANE_BG, R.BORDER)
        draw_panel_title(d, P, left, "measured ladder: aggregate decode throughput",
                         "green = 1 GiB headroom")
        draw_panel_title(d, P, right, "resident session capacity", None)

        visible = min(len(ladder), int(T * reveal) + 1)
        y = 100
        for i, (name, b, agg, slot, resv, green, ok, color) in enumerate(ladder[:visible]):
            if y > 650:
                break
            label = f"{name:10s}  B={b:>3}"
            d.text((32, y), label, font=P.font(13, bold=True), fill=R.FG)
            if ok:
                bar_color = color if green else R.AMBER
                draw_bar(d, P, 205, y + 1, 290, 16, float(agg or 0), max_agg,
                         bar_color, f"{float(agg or 0):,.0f}/s", R.BORDER)
                d.text((598, y), f"{short_num(slot, 1)} MiB/slot",
                       font=P.font(12), fill=R.CYAN)
                d.text((705, y), f"{short_num(resv, 2)}G",
                       font=P.font(12), fill=R.GREEN if green else R.AMBER)
            else:
                d.rectangle([205, y + 1, 280, y + 17], fill=(60, 24, 24),
                            outline=R.RED)
                d.text((292, y), f"OOM / over cap at {short_num(resv, 2)}G",
                       font=P.font(12, bold=True), fill=R.RED)
            y += 32

        y = 103
        headers = [("mode", 780), ("basis", 945), ("4k", 1082), ("16k", 1145), ("32k", 1210)]
        for text, x in headers:
            d.text((x, y - 28), text, font=P.font(11, bold=True), fill=R.DIM)
        for name, basis, c4, c16, c32, color in cap_rows:
            d.text((780, y), name, font=P.font(13, bold=True), fill=color)
            d.text((945, y), basis, font=P.font(11), fill=R.DIM)
            for val, x in ((c4, 1082), (c16, 1145), (c32, 1210)):
                shown = "~" + str(val) if basis == "derived target" or (name == "vLLM full KV" and val == 4) else str(val)
                d.text((x, y), shown, font=P.font(13, bold=True),
                       fill=color if val else R.DIMMER)
            draw_bar(d, P, 780, y + 20, 420, 13, c16, max_cap, color,
                     "", R.BORDER)
            y += 82

        notes = [
            "ring PoC: 96 green sessions at 4k and 16k, 36.3 MiB/slot",
            "full KV grows with context: 32 green at 4k, 8 green at 16k",
            "vLLM served Gemma first try; SGLang ran after stack fixes",
            "routed-span is the current quality+concurrency target, not a native measurement yet",
        ]
        y = 486
        for note in notes:
            y = draw_wrapped(d, P, (782, y), note, 55, 12, R.FG, max_lines=2)
            y += 8

        draw_footer(d, P, "ACT 2 / CONCURRENCY COMPARISON",
                    "replicated ladder is memory-honest; Act 1 used distinct prompts")
        emit(img)


def act_long_output(emit, blank, P, ev: dict, runs: dict, duration: float, fps: int):
    import wkvm_demo_video as R
    from PIL import ImageDraw

    meta = ev["meta"]
    names = ["wkvm:ring", "vLLM", "SGLang"]
    cols = [(10, 100, 420, 641), (435, 100, 845, 641), (860, 100, 1270, 641)]
    colors = {"wkvm:ring": R.AMBER, "vLLM": R.CYAN, "SGLang": R.GREEN}
    panes = {}
    durations = {}
    for name in names:
        run = runs[name]
        dt = float(run["timing"]["decode_delta_s"])
        durations[name] = dt
        panes[name] = R.StreamPane(replay_events(run["output_text"], dt, max_chunk=18))
    max_dt = max(durations.values())
    display_speed = max_dt / max(1e-6, duration - 1.0)

    for f_i in range(int(duration * fps)):
        T = (f_i + 1) / fps
        real_T = min(max_dt, T * display_speed)
        img = blank()
        d = ImageDraw.Draw(img)
        prompt_tokens = runs["wkvm:ring"]["prompt"]["ctx_tokens"]
        draw_header(d, P, meta, "long-prompt / long-output audit",
                    f"{prompt_tokens:,} ctx -> 512 output tokens")

        d.text((20, 60),
               "Same prompt, greedy, ignore_eos=True. Output text replayed from completed JSON; token clock uses measured counts.",
               font=P.font(13), fill=R.DIM)
        d.text((20, 78),
               "Expected facts: BLUE-742 / Samarkand / lantern",
               font=P.font(13, bold=True), fill=R.WHITE)

        for rect, name in zip(cols, names):
            run = runs[name]
            color = colors[name]
            gen = run["generation"]
            timing = run["timing"]
            facts = facts_found(run["output_text"])
            found_all = len(facts) == 3
            draw_box(d, rect, R.PANE_BG, color if found_all else R.AMBER, width=2)
            title = f"{name}  {timing['decode_tok_s']:.1f} decode tok/s"
            right = f"{gen['actual_output_tokens']}/{gen['requested_output_tokens']} tokens"
            draw_panel_title(d, P, rect, title, right, color)
            status = ("facts recovered: code, city, checksum"
                      if found_all else "facts missed: generated different record")
            d.text((rect[0] + 8, rect[1] + 31), status, font=P.font(11, bold=True),
                   fill=R.GREEN if found_all else R.AMBER)
            progress = int(min(gen["actual_output_tokens"],
                               gen["actual_output_tokens"] * real_T / durations[name]))
            d.text((rect[0] + 8, rect[1] + 49),
                   f"prefill+1st {timing['prefill_plus_first_s']:.3f}s   wall {timing['full_wall_s']:.3f}s   shown {progress:>3}/512",
                   font=P.font(10), fill=R.DIM)
            cw = max(10, int((rect[2] - rect[0] - 16) / P.char_w(11)))
            n_lines = max(4, int((rect[3] - rect[1] - 86) / 13))
            panes[name].advance(min(real_T, durations[name]), cw)
            inner = (rect[0] + 4, rect[1] + 78, rect[2] - 4, rect[3] - 8)
            R.draw_pane_text(d, P, inner, panes[name], n_lines, 11, 0)

        x0, y0, x1, y1 = 18, 655, 1262, 676
        d.rectangle([x0, y0, x1, y1], fill=(20, 23, 30), outline=R.BORDER)
        xpos = x0 + 3
        for name in names:
            run = runs[name]
            w = int((x1 - x0 - 6) * min(1.0, real_T / max_dt) / 3)
            d.rectangle([xpos, y0 + 3, xpos + w, y1 - 3], fill=colors[name])
            xpos += (x1 - x0 - 6) // 3

        draw_footer(d, P, "ACT 3 / WHOLE-OUTPUT THROUGHPUT",
                    "single stream: ring is slower here; capacity is the recurrent advantage")
        if display_speed > 1.15:
            R.speed_label(d, P, display_speed)
        emit(img)


def act_quality_target(emit, blank, P, ev: dict, duration: float, fps: int):
    import wkvm_demo_video as R
    from PIL import ImageDraw

    meta = ev["meta"]
    recall_rows = [
        ("full KV", 0.95, 1.00, 1.00, 0.86, R.CYAN, "measured quality baseline"),
        ("ring", 0.06, 0.07, 0.07, 0.04, R.AMBER, "measured high-concurrency ablation"),
        ("routed-span-m64", 0.90, 1.00, 0.89, 0.81, R.GREEN, "measured quality, derived capacity"),
    ]
    cap_rows = [
        ("ring", 36.3, 96, "measured"),
        ("routed-span-m64", 88.0, 43, "derived"),
        ("vLLM 16k", 276.0, 9, "measured"),
        ("vLLM 32k", 552.0, 4, "derived"),
    ]
    for f_i in range(int(duration * fps)):
        T = (f_i + 1) / fps
        fade = min(1.0, T / 1.0)
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header(d, P, meta, "quality plus concurrency: current justified target")

        left = (18, 70, 745, 500)
        right = (764, 70, 1262, 500)
        draw_box(d, left, R.PANE_BG, R.BORDER)
        draw_box(d, right, R.PANE_BG, R.BORDER)
        draw_panel_title(d, P, left, "recall on synthetic long-memory grid",
                         "higher is better")
        draw_panel_title(d, P, right, "per-session memory and resident capacity",
                         "16k/32k view")

        y = 125
        for name, overall, t1, t2, t3, color, note in recall_rows:
            d.text((42, y), name, font=P.font(15, bold=True), fill=color)
            vals = [("overall", overall), ("t1", t1), ("t2", t2), ("t3", t3)]
            x = 228
            for label, val in vals:
                d.text((x, y - 18), label, font=P.font(9), fill=R.DIM)
                draw_bar(d, P, x, y + 1, 82, 15, val * fade, 1.0,
                         color, f"{val:.2f}", R.BORDER)
                x += 124
            d.text((42, y + 32), note, font=P.font(11), fill=R.DIM)
            y += 82

        y = 130
        max_cap = 100
        for name, mib, cap, basis in cap_rows:
            color = R.GREEN if name == "routed-span-m64" else (R.AMBER if name == "ring" else R.CYAN)
            d.text((790, y), name, font=P.font(14, bold=True), fill=color)
            d.text((934, y), f"{mib:.1f} MiB/slot", font=P.font(12), fill=R.FG)
            tag = f"{'~' if basis == 'derived' else ''}{cap} sessions"
            draw_bar(d, P, 790, y + 24, 300, 17, cap * fade, max_cap, color,
                     "", R.BORDER)
            d.text((1102, y + 24), tag, font=P.font(11, bold=True),
                   fill=color)
            d.text((1190, y + 24), basis, font=P.font(11, bold=True),
                   fill=R.AMBER if basis == "derived" else R.GREEN)
            y += 78

        bottom = (18, 520, 1262, 676)
        draw_box(d, bottom, R.PANE_BG, R.BORDER)
        draw_panel_title(d, P, bottom, "defensible reading", None)
        lines = [
            "Plain ring wins capacity but fails evicted facts. It is not the 'both' result.",
            "routed-span-m64 is the current both-candidate: 0.90 recall overall and about 43 long sessions under the old green memory line.",
            "The remaining gap is real: t2 sibling eviction under fixed slot budget; native wkvm runner still needs to measure throughput without the HF PoC caveat.",
        ]
        yy = 560
        for line in lines:
            yy = draw_wrapped(d, P, (44, yy), line, 138, 13,
                              R.GREEN if "routed-span" in line else R.FG,
                              bold="routed-span" in line)
            yy += 7
        draw_footer(d, P, "ACT 4 / JUSTIFIED BOTH RESULT",
                    "measured quality + measured/derived capacity, labeled separately")
        emit(img)


def end_card(emit, blank, P, ev: dict, runs: dict, duration: float, fps: int):
    import wkvm_demo_video as R
    from PIL import ImageDraw

    meta = ev["meta"]
    S = ev["summary"]
    wk = runs["wkvm:ring"]["timing"]
    vl = runs["vLLM"]["timing"]
    sg = runs["SGLang"]["timing"]
    rows = [
        ("Gemma recurrent-mode demo", 32, R.WHITE, True),
        ("what the current artifacts justify", 16, R.DIM, False),
        ("", 8, R.DIM, False),
        (f"real distinct prompts: B={S['act_a_B']} at {S['act_a_tok_s']:.1f} tok/s, cache {S['act_a_cache_mib']:.1f} MiB", 15, R.GREEN, True),
        (f"ring ladder: 96 green sessions at 4k and 16k, {S['ring_slot_mib']:.1f} MiB/slot", 15, R.GREEN, True),
        (f"long output single stream: ring {wk['decode_tok_s']:.1f} tok/s, vLLM {vl['decode_tok_s']:.1f}, SGLang {sg['decode_tok_s']:.1f}", 15, R.CYAN, True),
        ("ring long-output fact check fails; vLLM/SGLang recover BLUE-742 / Samarkand / lantern", 14, R.AMBER, True),
        ("routed-span-m64 is the current both target: 0.90 recall, about 43 long sessions derived", 14, R.GREEN, True),
        ("", 8, R.DIM, False),
        ("next real result: native wkvm Gemma routed-span runner, then rerun quality + concurrency gates", 14, R.FG, False),
        ("experiments/results/gemma_wkvm_style_demo.mp4", 18, R.CYAN, True),
    ]
    for _ in range(int(duration * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        draw_header(d, P, meta, "summary")
        y = 105
        for s, sz, color, bold in rows:
            if s:
                f = P.font(sz, bold=bold)
                d.text(((P.W - f.getlength(s)) / 2, y), s, font=f, fill=color)
            y += sz + 15
        d.text((14, P.H - 26),
               f"all measured rows from saved artifacts - {meta['date']} - {meta['gpu']}",
               font=P.font(12), fill=R.DIMMER)
        emit(img)


def cmd_render(args) -> None:
    import wkvm_demo_video as R
    from PIL import Image

    ev = load_json(Path(args.events))
    runs = {name: load_json(path) for name, path in LONG_RUNS.items()}

    W, H = 1280, 720
    fps = args.fps
    P = R.Painter(W, H, fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", args.preset, "-crf", str(args.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )

    nframes = 0

    def blank():
        return Image.new("RGB", (W, H), R.BG)

    def emit(img):
        nonlocal nframes
        assert proc.stdin is not None
        proc.stdin.write(img.tobytes())
        nframes += 1

    title_card(emit, blank, P, ev, 4.0, fps)
    act_real_concurrency(emit, blank, P, ev, 11.0, fps)
    act_concurrency_ladder(emit, blank, P, ev, 14.0, fps)
    act_long_output(emit, blank, P, ev, runs, 15.0, fps)
    act_quality_target(emit, blank, P, ev, 11.0, fps)
    end_card(emit, blank, P, ev, runs, 6.0, fps)

    assert proc.stdin is not None
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed with rc={rc}")
    dur = nframes / fps
    print(f"RENDER_OK {out} frames={nframes} duration={dur:.1f}s "
          f"size={out.stat().st_size / 2**20:.1f}MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=str(EVENTS_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--preset", default="medium")
    args = ap.parse_args()
    cmd_render(args)


if __name__ == "__main__":
    main()
