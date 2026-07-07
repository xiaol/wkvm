"""Render a routed-span Gemma recurrent-mode demo video.

This is a presentation artifact built from saved measurements under
experiments/results. It intentionally separates:

  * measured ring concurrency,
  * measured B=1 routed-span long-output behavior,
  * measured routed-span replicated-session decode capacity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
RESULTS = EXP / "results"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT))

import gemma_wkvm_style_video as G
import wkvm_demo_video as R

EVENTS_DEFAULT = RESULTS / "gemma_demo_events.json"
OUT_DEFAULT = RESULTS / "gemma_routed_span_demo.mp4"
ROUTED_LADDER = RESULTS / "gemma_routed_span_ladder.json"

LONG_RUNS = {
    "wkvm:ring": RESULTS / "long_gen_13824_512_wkvm_ring.json",
    "wkvm:routed-span-m64": RESULTS / "long_gen_13824_512_wkvm_routed_span_m64.json",
    "vLLM": RESULTS / "long_gen_13824_512_vllm.json",
    "SGLang": RESULTS / "long_gen_13824_512_sglang.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _green_best(ladder: dict) -> dict:
    green = [r for r in ladder["rows"] if r.get("green")]
    return max(green, key=lambda r: r["B"]) if green else {}


def _best_ok(ladder: dict) -> dict:
    ok = [r for r in ladder["rows"] if r.get("ok")]
    return max(ok, key=lambda r: r["B"]) if ok else {}


def title_card(emit, blank, P, ev: dict, runs: dict, ladder: dict, duration: float, fps: int) -> None:
    from PIL import ImageDraw

    meta = ev["meta"]
    routed = runs["wkvm:routed-span-m64"]
    ring = runs["wkvm:ring"]
    best_green = _green_best(ladder)
    for _ in range(int(duration * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        G.draw_header(d, P, meta, "routed-span-m64 comparison", "measured + labeled")
        y = 112
        y = G.draw_center(d, P, y, "Gemma routed-span recurrent mode", 32, R.WHITE, True)
        y = G.draw_center(d, P, y, "long prompt, long output, concurrency context", 18, R.FG)
        y += 14
        rows = [
            (
                f"measured routed-span B=1: {routed['timing']['decode_tok_s']:.1f} decode tok/s, "
                "facts recovered",
                R.GREEN,
            ),
            (
                f"measured ring baseline: {ring['timing']['decode_tok_s']:.1f} decode tok/s, "
                "facts missed",
                R.AMBER,
            ),
            (
                "measured ring concurrency: 96 green resident sessions at 4k and 16k",
                R.CYAN,
            ),
            (
                f"measured routed-span replicated ladder: B={best_green['B']} green, "
                f"{best_green['agg']:.0f} aggregate tok/s",
                R.GREEN,
            ),
        ]
        for text, color in rows:
            y = G.draw_center(d, P, y + 7, text, 15, color, color == R.GREEN)
        y += 20
        G.draw_center(d, P, y, "patched HF cache PoC, not native wkvm.engine serving yet", 14, R.DIM)
        G.draw_footer(
            d,
            P,
            "ACT 0 / ROUTED-SPAN SOURCE OF TRUTH",
            "JSON + markdown artifacts under experiments/results",
        )
        emit(img)


def act_routed_span_ladder(
    emit, blank, P, ev: dict, ladder: dict, duration: float, fps: int
) -> None:
    from PIL import ImageDraw

    meta = ev["meta"]
    rows = ladder["rows"]
    summary = ladder["summary"]
    ctx = ladder["context_tokens_per_session"]
    steps = ladder["decode_tokens_per_session"]
    best_green = _green_best(ladder)
    best_ok = _best_ok(ladder)
    max_agg = max([r.get("agg") or 0 for r in rows] + [1])
    reveal = max(1, len(rows))

    for f_i in range(int(duration * fps)):
        T = (f_i + 1) / fps
        img = blank()
        d = ImageDraw.Draw(img)
        G.draw_header(
            d,
            P,
            meta,
            "measured routed-span replicated-session ladder",
            f"{ctx:,} ctx/session -> {steps} decode tokens",
        )

        left = (18, 66, 790, 676)
        right = (810, 66, 1262, 676)
        G.draw_box(d, left, R.PANE_BG, R.BORDER)
        G.draw_box(d, right, R.PANE_BG, R.BORDER)
        G.draw_panel_title(d, P, left, "routed-span-m64 measured rows", "19 GiB cap")
        G.draw_panel_title(d, P, right, "what the measurement means", None)

        visible = min(len(rows), int(T * reveal / max(1.0, duration - 1.0)) + 1)
        y = 114
        headers = [("B", 42), ("cache", 130), ("reserved", 238), ("aggregate decode", 358), ("status", 650)]
        for text, x in headers:
            d.text((x, y - 28), text, font=P.font(11, bold=True), fill=R.DIM)
        for r in rows[:visible]:
            b = r["B"]
            ok = bool(r.get("ok"))
            green = bool(r.get("green"))
            color = R.GREEN if green else (R.AMBER if ok else R.RED)
            d.text((42, y), f"{b:>2}", font=P.font(14, bold=True), fill=R.FG)
            if ok:
                cache = float(r["cache_mib"])
                resv = float(r["resv_gib"])
                agg = float(r["agg"])
                status = "green" if green else "over budget"
                d.text((130, y), f"{cache:,.0f} MiB", font=P.font(12), fill=R.CYAN)
                d.text((238, y), f"{resv:.2f} GiB", font=P.font(12), fill=color)
                G.draw_bar(d, P, 358, y + 1, 235, 16, agg, max_agg, color, f"{agg:,.0f}/s", R.BORDER)
                d.text((650, y), status, font=P.font(12, bold=True), fill=color)
            else:
                resv = r.get("resv_gib")
                d.rectangle([358, y + 1, 438, y + 17], fill=(60, 24, 24), outline=R.RED)
                d.text((456, y), f"OOM at {float(resv):.2f} GiB reserved",
                       font=P.font(12, bold=True), fill=R.RED)
            y += 58

        notes = [
            f"green max: B={summary['bmax_green']} at {summary['best_green_agg_tok_s']:.0f} aggregate tok/s",
            f"largest completed row: B={summary['max_ok_B']} at {best_ok.get('agg', 0):.0f} tok/s, but over the green memory line",
            "basis: one real B=1 routed-span prefill, then real cache tensor copies across batch",
            "not measured here: distinct routed-span prompts doing concurrent prefill/routing",
        ]
        y = 112
        for note in notes:
            fill = R.GREEN if note.startswith("green max") else (R.AMBER if note.startswith("not measured") else R.FG)
            y = G.draw_wrapped(d, P, (834, y), note, 48, 14, fill,
                               bold=note.startswith(("green max", "not measured")), max_lines=3)
            y += 18

        cfg = ladder["config"]
        y += 8
        config_rows = [
            ("mode", ladder["mode"]),
            ("m_slots", str(cfg["m_slots"])),
            ("route_on", cfg["route_on"]),
            ("decode", f"{steps} tokens/session"),
            ("cap", f"{ladder['mem_cap_gib']:.0f} GiB with {ladder['headroom_gib']:.0f} GiB headroom"),
        ]
        for k, v in config_rows:
            d.text((834, y), k, font=P.font(12, bold=True), fill=R.DIM)
            d.text((960, y), v, font=P.font(12), fill=R.CYAN if k == "mode" else R.FG)
            y += 34

        G.draw_footer(
            d,
            P,
            "ACT 2 / ROUTED-SPAN CONCURRENCY",
            "replicated-session decode measured; distinct prompt serving remains separate",
        )
        emit(img)


def act_routed_span_mechanics(
    emit, blank, P, ev: dict, runs: dict, ladder: dict, duration: float, fps: int
) -> None:
    from PIL import ImageDraw

    meta = ev["meta"]
    routed = runs["wkvm:routed-span-m64"]
    cfg = routed["engine_config"]
    mem = routed["memory"]["full_pass"]
    facts = G.facts_found(routed["output_text"])
    best_green = _green_best(ladder)

    for f_i in range(int(duration * fps)):
        T = (f_i + 1) / fps
        reveal = min(1.0, T / max(1.0, duration - 1.0))
        img = blank()
        d = ImageDraw.Draw(img)
        G.draw_header(d, P, meta, "what makes this routed-span, not plain ring")

        left = (18, 66, 736, 504)
        right = (760, 66, 1262, 504)
        bottom = (18, 526, 1262, 676)
        for rect in (left, right, bottom):
            G.draw_box(d, rect, R.PANE_BG, R.BORDER)

        G.draw_panel_title(d, P, left, "span-routing path", "explicit mask")
        G.draw_panel_title(d, P, right, "measured runs", "quality + ladder")
        G.draw_panel_title(d, P, bottom, "why ring loses this prompt", None)

        steps = [
            ("1", "evicted prompt tokens are split at sentence punctuation", R.CYAN),
            ("2", "each span routes atomically into one content slot", R.GREEN),
            ("3", "routing feature = most-novel VALUE token, not RoPE keys", R.GREEN),
            ("4", "within-slot farthest-point retention keeps diverse spans", R.CYAN),
        ]
        y = 114
        for idx, text, color in steps:
            alpha = min(1.0, max(0.0, reveal * 4.5 - (int(idx) - 1)))
            fill = color if alpha > 0.15 else R.DIMMER
            d.ellipse([42, y - 4, 70, y + 24], fill=(22, 28, 35), outline=fill, width=2)
            f = P.font(13, bold=True)
            d.text((56 - f.getlength(idx) / 2, y + 2), idx, font=f, fill=fill)
            G.draw_wrapped(d, P, (86, y), text, 66, 14, fill, bold=alpha > 0.65)
            y += 74

        rows = [
            ("mode", f"{cfg['wkvm_mode']} + span"),
            ("m_slots", str(cfg["m_slots"])),
            ("route_on", cfg["route_on"]),
            ("span_break_mask", cfg["span_break_mask"]),
            ("cache", f"{mem['cache_mib']:.1f} MiB @ 13,824 ctx"),
            ("throughput", f"{routed['timing']['decode_tok_s']:.1f} decode tok/s"),
            ("ladder", f"B={best_green['B']} green, {best_green['agg']:.0f} aggregate tok/s"),
            ("facts", ", ".join(facts) if facts else "-"),
        ]
        y = 112
        for k, v in rows:
            d.text((790, y), k, font=P.font(12, bold=True), fill=R.DIM)
            G.draw_wrapped(d, P, (930, y), v, 34, 13, R.GREEN if k in ("facts", "throughput") else R.FG, bold=k == "facts")
            y += 42
        G.draw_wrapped(
            d,
            P,
            (790, 430),
            "Concurrency shown here is replicated-session decode after one real routed-span prefill; distinct prompt serving is not measured in this artifact.",
            47,
            12,
            R.AMBER,
            bold=True,
        )

        x0, y0, x1, y1 = bottom
        d.text((42, y0 + 40), "13,824-token prompt", font=P.font(14, bold=True), fill=R.FG)
        bar_x0, bar_y = 210, y0 + 45
        bar_w = 820
        d.rectangle([bar_x0, bar_y, bar_x0 + bar_w, bar_y + 16], fill=(22, 25, 33), outline=R.BORDER)
        needle_x = bar_x0 + int(bar_w * 200 / 13824)
        ring_x = bar_x0 + int(bar_w * (13824 - 1024) / 13824)
        d.rectangle([needle_x, bar_y - 10, needle_x + 8, bar_y + 26], fill=R.GREEN)
        d.rectangle([ring_x, bar_y, bar_x0 + bar_w, bar_y + 16], fill=(51, 48, 30), outline=R.AMBER)
        d.text((needle_x + 14, bar_y - 16), "BLUE-742 / Samarkand / lantern", font=P.font(11, bold=True), fill=R.GREEN)
        d.text((ring_x - 20, bar_y + 23), "ring window only sees suffix", font=P.font(11), fill=R.AMBER)
        d.text((42, y0 + 92), "ring output invented Project Chimera / Neo-Kyoto / 7B3A9F", font=P.font(13, bold=True), fill=R.AMBER)
        d.text((610, y0 + 92), "routed-span output starts with the correct record", font=P.font(13, bold=True), fill=R.GREEN)

        G.draw_footer(
            d,
            P,
            "ACT 2 / ROUTED-SPAN MECHANICS",
            "explicit sentence mask applied in long_generation_compare.py",
        )
        emit(img)


def act_long_output(emit, blank, P, ev: dict, runs: dict, duration: float, fps: int) -> None:
    from PIL import ImageDraw

    meta = ev["meta"]
    names = ["wkvm:ring", "wkvm:routed-span-m64", "vLLM", "SGLang"]
    rects = [
        (12, 102, 632, 356),
        (648, 102, 1268, 356),
        (12, 372, 632, 642),
        (648, 372, 1268, 642),
    ]
    colors = {
        "wkvm:ring": R.AMBER,
        "wkvm:routed-span-m64": R.GREEN,
        "vLLM": R.CYAN,
        "SGLang": (160, 190, 255),
    }
    panes = {}
    durations = {}
    for name in names:
        run = runs[name]
        dt = float(run["timing"]["decode_delta_s"])
        durations[name] = dt
        panes[name] = R.StreamPane(G.replay_events(run["output_text"], dt, max_chunk=18))

    max_dt = max(durations.values())
    display_speed = max_dt / max(1e-6, duration - 1.0)

    for f_i in range(int(duration * fps)):
        T = (f_i + 1) / fps
        real_T = min(max_dt, T * display_speed)
        img = blank()
        d = ImageDraw.Draw(img)
        prompt_tokens = runs["wkvm:ring"]["prompt"]["ctx_tokens"]
        G.draw_header(
            d,
            P,
            meta,
            "same long prompt, greedy long output",
            f"{prompt_tokens:,} ctx -> 512 output tokens",
        )
        d.text(
            (20, 62),
            "Expected first sentence: BLUE-742 / Samarkand / lantern. Text replay is from completed JSON; token clocks use measured output counts.",
            font=P.font(13),
            fill=R.DIM,
        )

        for rect, name in zip(rects, names):
            run = runs[name]
            color = colors[name]
            gen = run["generation"]
            timing = run["timing"]
            found_all = len(G.facts_found(run["output_text"])) == 3
            outline = R.GREEN if found_all else R.AMBER
            G.draw_box(d, rect, R.PANE_BG, outline, width=2)
            title = f"{name}  {timing['decode_tok_s']:.1f} tok/s"
            right = f"{gen['actual_output_tokens']}/{gen['requested_output_tokens']}"
            G.draw_panel_title(d, P, rect, title, right, color)
            status = "facts recovered" if found_all else "facts missed"
            d.text(
                (rect[0] + 8, rect[1] + 30),
                status,
                font=P.font(11, bold=True),
                fill=R.GREEN if found_all else R.AMBER,
            )
            progress = int(min(gen["actual_output_tokens"], gen["actual_output_tokens"] * real_T / durations[name]))
            d.text(
                (rect[0] + 122, rect[1] + 30),
                f"prefill+1st {timing['prefill_plus_first_s']:.3f}s   wall {timing['full_wall_s']:.3f}s   shown {progress:>3}/512",
                font=P.font(10),
                fill=R.DIM,
            )
            cw = max(10, int((rect[2] - rect[0] - 16) / P.char_w(11)))
            n_lines = max(4, int((rect[3] - rect[1] - 72) / 13))
            panes[name].advance(min(real_T, durations[name]), cw)
            inner = (rect[0] + 4, rect[1] + 64, rect[2] - 4, rect[3] - 8)
            R.draw_pane_text(d, P, inner, panes[name], n_lines, 11, 0)

        x0, y0, x1, y1 = 20, 656, 1260, 676
        d.rectangle([x0, y0, x1, y1], fill=(20, 23, 30), outline=R.BORDER)
        span = (x1 - x0 - 8) // len(names)
        for i, name in enumerate(names):
            w = int(span * min(1.0, real_T / durations[name]))
            px = x0 + 4 + i * span
            d.rectangle([px, y0 + 4, px + w, y1 - 4], fill=colors[name])
        G.draw_footer(
            d,
            P,
            "ACT 3 / WHOLE-OUTPUT THROUGHPUT",
            "single-stream: full-KV engines decode faster; routed-span fixes ring's factual miss",
        )
        if display_speed > 1.15:
            R.speed_label(d, P, display_speed)
        emit(img)


def end_card(emit, blank, P, ev: dict, runs: dict, ladder: dict, duration: float, fps: int) -> None:
    from PIL import ImageDraw

    meta = ev["meta"]
    S = ev["summary"]
    routed = runs["wkvm:routed-span-m64"]
    ring = runs["wkvm:ring"]
    vllm = runs["vLLM"]
    sglang = runs["SGLang"]
    best_green = _green_best(ladder)
    best_ok = _best_ok(ladder)
    rows = [
        ("routed-span Gemma demo", 32, R.WHITE, True),
        ("what is justified by the current artifacts", 16, R.DIM, False),
        ("", 8, R.DIM, False),
        (
            f"routed-span long output: {routed['timing']['decode_tok_s']:.1f} tok/s, "
            "512/512 tokens, facts recovered",
            15,
            R.GREEN,
            True,
        ),
        (
            f"ring long output: {ring['timing']['decode_tok_s']:.1f} tok/s, facts missed",
            15,
            R.AMBER,
            True,
        ),
        (
            f"full-KV single-stream baselines: vLLM {vllm['timing']['decode_tok_s']:.1f} tok/s, "
            f"SGLang {sglang['timing']['decode_tok_s']:.1f}",
            15,
            R.CYAN,
            True,
        ),
        (
            f"ring concurrency: {S['ring_bmax']} green sessions at 4k/16k, "
            f"{S['ring_slot_mib']:.1f} MiB/slot",
            15,
            R.GREEN,
            True,
        ),
        (
            f"routed-span replicated ladder: B={best_green['B']} green, "
            f"{best_green['agg']:.0f} tok/s; B={best_ok['B']} ran but over budget",
            14,
            R.GREEN,
            True,
        ),
        ("", 8, R.DIM, False),
        (
            "remaining gap: distinct routed-span prompts and native serving are not measured in this artifact",
            14,
            R.FG,
            False,
        ),
        ("experiments/results/gemma_routed_span_demo.mp4", 18, R.CYAN, True),
    ]
    for _ in range(int(duration * fps)):
        img = blank()
        d = ImageDraw.Draw(img)
        G.draw_header(d, P, meta, "summary")
        y = 98
        for s, sz, color, bold in rows:
            if s:
                f = P.font(sz, bold=bold)
                d.text(((P.W - f.getlength(s)) / 2, y), s, font=f, fill=color)
            y += sz + 15
        d.text(
            (14, P.H - 26),
            f"all measured rows from saved artifacts - {meta['date']} - {meta['gpu']}",
            font=P.font(12),
            fill=R.DIMMER,
        )
        emit(img)


def cmd_render(args) -> None:
    from PIL import Image

    ev = load_json(Path(args.events))
    runs = {name: load_json(path) for name, path in LONG_RUNS.items()}
    ladder = load_json(Path(args.ladder))

    W, H = 1280, 720
    fps = args.fps
    P = R.Painter(W, H, fps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
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

    title_card(emit, blank, P, ev, runs, ladder, 4.0, fps)
    G.act_real_concurrency(emit, blank, P, ev, 8.0, fps)
    act_routed_span_ladder(emit, blank, P, ev, ladder, 12.0, fps)
    act_routed_span_mechanics(emit, blank, P, ev, runs, ladder, 8.0, fps)
    act_long_output(emit, blank, P, ev, runs, 16.0, fps)
    G.act_quality_target(emit, blank, P, ev, 10.0, fps)
    end_card(emit, blank, P, ev, runs, ladder, 5.0, fps)

    assert proc.stdin is not None
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"ffmpeg failed with rc={rc}")
    dur = nframes / fps
    print(f"RENDER_OK {out} frames={nframes} duration={dur:.1f}s size={out.stat().st_size / 2**20:.1f}MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", default=str(EVENTS_DEFAULT))
    ap.add_argument("--ladder", default=str(ROUTED_LADDER))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=22)
    ap.add_argument("--preset", default="medium")
    args = ap.parse_args()
    cmd_render(args)


if __name__ == "__main__":
    main()
