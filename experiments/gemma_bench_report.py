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


def load_payloads(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    return [(path, json.loads(path.read_text())) for path in paths]


def shape_key(data: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        data.get("context_tokens_per_session"),
        data.get("decode_tokens_per_session"),
        data.get("prompt_lengths_mode"),
    )


def fmt_shape(shape: tuple[Any, Any, Any]) -> str:
    ctx, out, prompt_mode = shape
    return f"ctx={fmt(ctx)} out={fmt(out)} prompt={prompt_mode or '-'}"


def validate_same_shape(payloads: list[tuple[Path, dict[str, Any]]]) -> None:
    groups: dict[tuple[Any, Any, Any], list[Path]] = {}
    for path, data in payloads:
        groups.setdefault(shape_key(data), []).append(path)
    if len(groups) <= 1:
        return
    parts = []
    for shape, paths in sorted(groups.items(), key=lambda item: fmt_shape(item[0])):
        parts.append(
            f"{fmt_shape(shape)}: "
            + ", ".join(path.as_posix() for path in sorted(paths))
        )
    raise ValueError(
        "same-shape requirement failed; inputs contain multiple benchmark shapes: "
        + "; ".join(parts)
    )


def row_has_full_success(row: dict[str, Any]) -> bool:
    return row.get("success_count") == row.get("B")


def native_no_hf_row_problems(row: dict[str, Any]) -> list[str]:
    problems = []
    if row.get("uses_hf_transformer_forward") is not False:
        problems.append("uses_hf_transformer_forward_not_false")
    if row.get("uses_hf_model_construction") is not False:
        problems.append("uses_hf_model_construction_not_false")
    if row.get("native_gemma_checkpoint_loader") is not True:
        problems.append("native_gemma_checkpoint_loader_not_true")
    return problems


def native_no_hf_setup_problems(data: dict[str, Any]) -> list[str]:
    problems = []
    if data.get("uses_hf_tokenizer") is not False:
        problems.append("uses_hf_tokenizer_not_false")
    if data.get("uses_hf_config") is not False:
        problems.append("uses_hf_config_not_false")
    if data.get("native_gemma_config_loader") is not True:
        problems.append("native_gemma_config_loader_not_true")
    return problems


def validate_native_no_hf(payloads: list[tuple[Path, dict[str, Any]]]) -> None:
    violations = []
    checked_rows = 0
    native_payloads = [
        (path, data) for path, data in payloads if data.get("engine") == "wkvm-native"
    ]
    for path, data in native_payloads:
        setup_problems = native_no_hf_setup_problems(data)
        if setup_problems:
            violations.append((path, "setup", setup_problems))
        payload_checked_rows = 0
        for row in data.get("rows", []):
            if not row_has_full_success(row):
                continue
            checked_rows += 1
            payload_checked_rows += 1
            problems = native_no_hf_row_problems(row)
            if problems:
                violations.append((path, row.get("B"), problems))
        if payload_checked_rows == 0:
            violations.append((path, None, ["no_successful_rows_to_check"]))
    if not native_payloads:
        violations.append((None, None, ["no_wkvm_native_payloads_to_check"]))
    if not violations and checked_rows > 0:
        return
    parts = []
    for path, B, problems in violations:
        source = path.as_posix() if path is not None else "<inputs>"
        parts.append(f"{source} B={fmt(B)} problems={','.join(problems)}")
    raise ValueError(
        "native no-HF requirement failed; "
        "wkvm-native payloads must prove no HF tokenizer/config, no HF model "
        "construction/forward, and native checkpoint/config loading: "
        + "; ".join(parts)
    )


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


def row_or_payload_value(
    data: dict[str, Any], row: dict[str, Any], key: str
) -> Any:
    if key in row:
        return row.get(key)
    boundary = data.get("hf_boundary")
    if isinstance(boundary, dict) and key in boundary:
        return boundary.get(key)
    return data.get(key)


def no_hf_guard_summary(data: dict[str, Any], row: dict[str, Any]) -> str:
    if data.get("engine") != "wkvm-native":
        return "n/a"
    report = data.get("native_no_hf_requirement")
    if isinstance(report, dict):
        status = "pass" if report.get("passed") is True else "fail"
        checked = fmt(report.get("checked_successful_rows"))
        required = ", required" if report.get("required") else ""
        return f"{status} ({checked} rows{required})"
    if row_has_full_success(row):
        return "pass (row)" if not native_no_hf_row_problems(row) else "fail (row)"
    return "-"


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


def fatal_error_row(data: dict[str, Any]) -> dict[str, Any] | None:
    fatal = data.get("fatal_error")
    if not isinstance(fatal, dict):
        return None
    parts = []
    phase = fatal.get("phase")
    if phase:
        parts.append(str(phase))
    error_type = fatal.get("type")
    if error_type:
        parts.append(str(error_type))
    message = fatal.get("message")
    if message:
        parts.append(str(message))
    return {
        "B": None,
        "success_count": 0,
        "green": False,
        "error": ": ".join(parts) if parts else "fatal setup error",
    }


def load_rows(path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source_rows = list(data.get("rows", []))
    if not source_rows:
        fatal_row = fatal_error_row(data)
        if fatal_row is not None:
            source_rows = [fatal_row]
    for row in source_rows:
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
                "shape": fmt_shape(shape_key(data)),
                "ctx": data.get("context_tokens_per_session"),
                "out": data.get("decode_tokens_per_session"),
                "prompt_mode": data.get("prompt_lengths_mode"),
                "model_forward_backend": row_or_payload_value(
                    data, row, "model_forward_backend"
                ),
                "uses_hf_transformer_forward": row_or_payload_value(
                    data, row, "uses_hf_transformer_forward"
                ),
                "uses_hf_model_construction": row_or_payload_value(
                    data, row, "uses_hf_model_construction"
                ),
                "uses_hf_tokenizer": row_or_payload_value(
                    data, row, "uses_hf_tokenizer"
                ),
                "uses_hf_config": row_or_payload_value(
                    data, row, "uses_hf_config"
                ),
                "native_gemma_config_loader": row_or_payload_value(
                    data, row, "native_gemma_config_loader"
                ),
                "native_gemma_checkpoint_loader": row_or_payload_value(
                    data, row, "native_gemma_checkpoint_loader"
                ),
                "no_hf_guard": no_hf_guard_summary(data, row),
                "B": row.get("B"),
                "success": f"{row.get('success_count', 0)}/{fmt(row.get('B'))}",
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


def render(
    paths: list[Path],
    *,
    require_same_shape: bool = False,
    require_native_no_hf: bool = False,
) -> str:
    payloads = load_payloads(paths)
    if require_same_shape:
        validate_same_shape(payloads)
    if require_native_no_hf:
        validate_native_no_hf(payloads)
    rows: list[dict[str, Any]] = []
    for path, data in payloads:
        rows.extend(load_rows(path, data))
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
        "Only rows with the same `ctx`, `out`, prompt mode, and benchmark path should be treated as same-shape comparisons. "
        "The native no-HF columns are applicable to wkvm-native rows and are `n/a` for incumbent engines.",
        "",
        "| engine | shape | B | success | green | agg decode tok/s | forward backend | HF fwd | HF construct | HF tok | HF cfg | native cfg | native ckpt | no-HF guard | decode timing s | e2e output tok/s | p50 s | p95 s | memory | max model batch | padded temp | persistent padded | error | source |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|---|---|---|---|---|",
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
                    row["shape"],
                    fmt(row["B"]),
                    str(row["success"]),
                    fmt(row["green"]),
                    fmt(row["agg_decode"]),
                    str(row["model_forward_backend"] or "-"),
                    fmt(row["uses_hf_transformer_forward"]),
                    fmt(row["uses_hf_model_construction"]),
                    fmt(row["uses_hf_tokenizer"]),
                    fmt(row["uses_hf_config"]),
                    fmt(row["native_gemma_config_loader"]),
                    fmt(row["native_gemma_checkpoint_loader"]),
                    row["no_hf_guard"],
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
    ap.add_argument(
        "--require-same-shape",
        action="store_true",
        help="fail if the input JSON files do not share one ctx/out/prompt-mode shape",
    )
    ap.add_argument(
        "--require-native-no-hf",
        action="store_true",
        help=(
            "fail unless every fully successful wkvm-native row proves no HF "
            "Transformer forward, no HF model construction, and native checkpoint loading"
        ),
    )
    args = ap.parse_args()
    text = render(
        args.json_files,
        require_same_shape=args.require_same_shape,
        require_native_no_hf=args.require_native_no_hf,
    )
    if args.out:
        atomic_write_text(args.out, text)
        print(f"WROTE {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
