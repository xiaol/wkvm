#!/usr/bin/env python
"""Render a normalized Gemma throughput report from benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def fmt(x: Any, suffix: str = "") -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.3f}{suffix}"
    return f"{x}{suffix}"


def fmt_bytes(x: Any) -> str:
    if x is None:
        return "-"
    try:
        value = int(x)
    except (TypeError, ValueError):
        return "-"
    if value == 0:
        return "0"
    gib = value / 2**30
    if gib >= 1:
        return f"{gib:.3f} GiB"
    mib = value / 2**20
    if mib >= 1:
        return f"{mib:.1f} MiB"
    return f"{value} B"


def engine_label(data: dict[str, Any]) -> str:
    engine = data.get("engine", "unknown")
    if engine == "hf-transformers":
        return f"HF Transformers ({data.get('mode', 'unknown')})"
    if engine == "wkvm-native":
        cfg = data.get("config") or {}
        cap = cfg.get("decode_microbatch_bytes")
        rows = cfg.get("decode_microbatch_rows")
        suffix = ""
        if cfg.get("persistent_padded_decode") is False:
            suffix = ", persistent padded off"
        if cap:
            return f"wkvm-native byte-cap {cap}{suffix}"
        return f"wkvm-native row-cap {rows}{suffix}"
    if engine == "wkvm-native-http-stream":
        return "wkvm-native HTTP stream"
    if engine == "wkvm-native-openai-completions":
        return "wkvm-native OpenAI completions"
    if engine == "vllm-http-stream":
        return "vLLM HTTP stream"
    if engine == "sglang-http-stream":
        return "SGLang HTTP stream"
    return str(engine)


def row_memory(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("peak_engine_delta_gib") is not None:
        return "engine delta", fmt(row.get("peak_engine_delta_gib"), " GiB")
    if row.get("peak_reserved_gib") is not None:
        return "reserved", fmt(row.get("peak_reserved_gib"), " GiB")
    if row.get("peak_alloc_gib") is not None:
        return "allocated", fmt(row.get("peak_alloc_gib"), " GiB")
    return "-", "-"


def model_batch_summary(row: dict[str, Any]) -> str:
    rows = row.get("max_decode_model_batch_rows")
    bytes_ = row.get("max_decode_model_batch_bytes")
    if rows is None and bytes_ is None:
        return "-"
    if bytes_ is None:
        return f"{fmt(rows)} rows"
    if rows is None:
        return fmt_bytes(bytes_)
    return f"{fmt(rows)} rows / {fmt_bytes(bytes_)}"


def persistent_summary(row: dict[str, Any]) -> str:
    starts = row.get("persistent_padded_decode_starts")
    reuses = row.get("persistent_padded_decode_reuses")
    if starts is None and reuses is None:
        return "-"
    return f"{fmt(starts)} starts / {fmt(reuses)} reuses"


def decode_timing_summary(row: dict[str, Any]) -> str:
    keys = (
        ("merge", "decode_timing_merge_s"),
        ("model", "decode_timing_model_forward_s"),
        ("commit", "decode_timing_commit_s"),
        ("split", "decode_timing_split_s"),
        ("mask", "decode_timing_mask_s"),
        ("total", "decode_timing_total_s"),
    )
    if all(row.get(key) is None for _label, key in keys):
        return "-"
    return " / ".join(f"{label} {fmt(row.get(key))}" for label, key in keys)


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    rows = []
    for row in data.get("rows", []):
        mem_kind, mem_value = row_memory(row)
        agg_decode = row.get("agg_decode_tok_s")
        if agg_decode is None:
            agg_decode = row.get("request_output_tok_s")
        p50 = row.get("p50_latency_s")
        if p50 is None:
            p50 = row.get("p50_e2e_latency_s")
        p95 = row.get("p95_latency_s")
        if p95 is None:
            p95 = row.get("p95_e2e_latency_s")
        rows.append(
            {
                "path": path,
                "schema": data.get("schema"),
                "engine": engine_label(data),
                "ctx": data.get("context_tokens_per_session"),
                "out": data.get("decode_tokens_per_session"),
                "prompt_mode": data.get("prompt_lengths_mode"),
                "B": row.get("B"),
                "success": f"{row.get('success_count', 0)}/{row.get('B', '?')}",
                "green": row.get("green"),
                "agg_decode": agg_decode,
                "decode_timing": decode_timing_summary(row),
                "e2e_output": row.get("e2e_output_tok_s"),
                "p50": p50,
                "p95": p95,
                "mem_kind": mem_kind,
                "mem_value": mem_value,
                "model_batch": model_batch_summary(row),
                "padded_temp": fmt_bytes(row.get("padded_decode_temp_bytes")),
                "persistent_padded": persistent_summary(row),
                "error": row.get("error"),
            }
        )
    return rows


def render(paths: list[Path]) -> str:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_rows(path))
    rows.sort(
        key=lambda r: (
            r["ctx"] or 0,
            r["out"] or 0,
            r["B"] or 0,
            r["engine"],
            str(r["path"]),
        )
    )
    lines = [
        "# Gemma Throughput Report",
        "",
        "Rows are normalized across wkvm-native, HF Transformers, vLLM, and SGLang JSON schemas. "
        "Serving rows use HTTP stream output throughput in the `agg decode tok/s` column. "
        "Only rows with the same `ctx`, `out`, prompt mode, and benchmark path should be treated as same-shape comparisons.",
        "",
        "| engine | ctx | out | prompt mode | B | success | green | agg decode tok/s | decode timing s | e2e output tok/s | p50 s | p95 s | memory | max model batch | padded temp | persistent padded | error | source |",
        "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        path = row["path"]
        source = f"[json]({path.as_posix()})"
        memory = (
            row["mem_value"]
            if row["mem_kind"] == "-"
            else f"{row['mem_value']} {row['mem_kind']}"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["engine"]),
                    fmt(row["ctx"]),
                    fmt(row["out"]),
                    str(row["prompt_mode"] or "-"),
                    fmt(row["B"]),
                    str(row["success"]),
                    fmt(row["green"]),
                    fmt(row["agg_decode"]),
                    row["decode_timing"],
                    fmt(row["e2e_output"]),
                    fmt(row["p50"]),
                    fmt(row["p95"]),
                    memory,
                    row["model_batch"],
                    row["padded_temp"],
                    row["persistent_padded"],
                    str(row["error"] or "-"),
                    source,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = render(args.json_files)
    if args.out:
        atomic_write_text(args.out, text)
        print(f"WROTE {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
