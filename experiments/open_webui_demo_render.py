"""Render deterministic WKVM/Open WebUI browser evidence.

The renderer consumes a browser-capture artifact and its measured report.  It
does not manufacture benchmark numbers: prompt labels and aggregate metrics
come from the report, while per-video timing comes from capture offsets.

Example:
  python experiments/open_webui_demo_render.py render \
    --capture experiments/results/open_webui_demo_capture.json \
    --report experiments/results/open_webui_demo_report.json \
    --mp4 experiments/results/open_webui_demo.mp4 \
    --gif experiments/results/open_webui_demo.gif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WIDTH = 1280
HEIGHT = 720
FPS = 30
TITLE_SECONDS = 3.0
TRANSITION_SECONDS = 2.0
FINAL_SECONDS = 4.0
LONG_MAX_DISPLAY_SECONDS = 24.0
MONTAGE_MAX_DISPLAY_SECONDS = 20.0
MP4_SIZE_LIMIT = 10 * 1024 * 1024
GIF_SIZE_LIMIT = 5 * 1024 * 1024
SEMANTICS = "routed_span_approximate"
TEN_X_CAVEAT = "not a 10x measurement"

BACKGROUND = (9, 13, 22)
PANEL = (20, 28, 43)
PANEL_ALT = (25, 35, 53)
INK = (238, 244, 249)
MUTED = (153, 168, 187)
TEAL = (70, 222, 190)
CYAN = (91, 201, 255)
AMBER = (251, 190, 80)
RED = (248, 113, 113)

_MISSING = object()


class SchemaError(ValueError):
    """Raised when a capture/report pair cannot support an honest render."""


@dataclass(frozen=True)
class PromptInfo:
    prompt_id: str
    label: str
    summary: str


@dataclass(frozen=True)
class Turn:
    submitted_s: float
    first_token_s: float
    completed_s: float

    @property
    def ttft_s(self) -> float:
        return self.first_token_s - self.submitted_s

    @property
    def e2e_s(self) -> float:
        return self.completed_s - self.submitted_s


@dataclass(frozen=True)
class Act:
    prompt: PromptInfo
    video: Path
    submitted_s: float
    first_token_s: float
    completed_s: float
    rendered_token_count: int | None = None
    follow_up: Turn | None = None

    @property
    def ttft_s(self) -> float:
        return self.first_token_s - self.submitted_s

    @property
    def e2e_s(self) -> float:
        return self.completed_s - self.submitted_s

    @property
    def recording_completed_s(self) -> float:
        if self.follow_up is not None:
            return self.follow_up.completed_s
        return self.completed_s

    @property
    def recording_duration_s(self) -> float:
        return self.recording_completed_s - self.submitted_s


@dataclass(frozen=True)
class Metrics:
    offered_concurrency: int
    success_count: int
    follow_up_success_count: int
    error_count: int
    ttft_p50_s: float
    ttft_p95_s: float
    e2e_p50_s: float
    e2e_p95_s: float
    baseline_vram_mib: float | None
    peak_vram_mib: float
    max_running: int
    max_runnable_rows: int
    exact_reuse_hits: int
    provider_error_count: int
    provider_summary: str | None
    first_turn_min_output_tokens: int | None
    follow_up_min_output_tokens: int | None
    act_2_total_output_tokens: int | None


@dataclass(frozen=True)
class Evidence:
    capture_path: Path
    report_path: Path
    long_act: Act
    classic_acts: tuple[Act, Act, Act, Act]
    metrics: Metrics
    model: str
    gpu: str
    provenance: str
    scoped_evidence: str


@dataclass(frozen=True)
class VisualSpec:
    role: str
    eyebrow: str
    title: str
    subtitle: str = ""
    rows: tuple[str, ...] = ()
    footer: str = ""
    caveat: str = ""
    pane_labels: tuple[tuple[str, str], ...] = ()
    transparent: bool = False

    def all_text(self) -> str:
        parts: list[str] = [self.eyebrow, self.title, self.subtitle]
        parts.extend(self.rows)
        for first, second in self.pane_labels:
            parts.extend((first, second))
        parts.extend((self.footer, self.caveat))
        return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class Timeline:
    long_speed: float
    long_display_s: float
    montage_speed: float
    montage_display_s: float

    @property
    def total_s(self) -> float:
        return (
            TITLE_SECONDS
            + self.long_display_s
            + TRANSITION_SECONDS
            + self.montage_display_s
            + FINAL_SECONDS
        )

    @property
    def montage_start_s(self) -> float:
        return TITLE_SECONDS + self.long_display_s + TRANSITION_SECONDS

    @property
    def final_start_s(self) -> float:
        return self.montage_start_s + self.montage_display_s


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SchemaError(f"{label} JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"{label} JSON is invalid at line {exc.lineno}, column {exc.colno}: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise SchemaError(f"{label} JSON root must be an object: {path}")
    return value


def _at(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _pick(
    value: Any,
    paths: Iterable[Sequence[str]],
    label: str,
    *,
    default: Any = _MISSING,
) -> Any:
    for path in paths:
        candidate = _at(value, path)
        if candidate is not _MISSING and candidate is not None:
            return candidate
    if default is not _MISSING:
        return default
    rendered = ", ".join(".".join(path) for path in paths)
    raise SchemaError(f"missing {label}; expected one of: {rendered}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{label} must be an object")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{label} must be finite, got {value!r}")
    return result


def _integer(value: Any, label: str) -> int:
    number = _number(value, label)
    if not number.is_integer():
        raise SchemaError(f"{label} must be an integer, got {value!r}")
    return int(number)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{label} must be a non-empty string")
    return value.strip()


def _normal_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _recursive_values(value: Any, aliases: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normal_key(str(key)) in aliases:
                found.append(child)
            found.extend(_recursive_values(child, aliases))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_values(child, aliases))
    return found


def _resolve_video(raw: Any, capture_path: Path, label: str) -> Path:
    video = Path(_text(raw, label)).expanduser()
    if not video.is_absolute():
        video = capture_path.parent / video
    return video.resolve()


def _timing(node: Mapping[str, Any], label: str) -> tuple[float, float, float]:
    timing = _mapping(node.get("timing", node), f"{label}.timing")
    submitted = _number(
        _pick(
            timing,
            (
                ("submit_offset_s",),
                ("submitted_offset_s",),
                ("prompt_offset_s",),
                ("submitted_s",),
            ),
            f"{label} submit offset",
        ),
        f"{label} submit offset",
    )
    first_token = _number(
        _pick(
            timing,
            (
                ("first_token_offset_s",),
                ("first_token_s",),
                ("first_token_at_s",),
            ),
            f"{label} first-token offset",
        ),
        f"{label} first-token offset",
    )
    completed = _number(
        _pick(
            timing,
            (
                ("completion_offset_s",),
                ("completed_offset_s",),
                ("completed_s",),
                ("completion_s",),
            ),
            f"{label} completion offset",
        ),
        f"{label} completion offset",
    )
    if submitted < 0 or not submitted <= first_token <= completed:
        raise SchemaError(
            f"{label} offsets must satisfy 0 <= submit <= first token <= completion; "
            f"got {submitted}, {first_token}, {completed}"
        )
    if completed == submitted:
        raise SchemaError(f"{label} completion must be later than submission")
    return submitted, first_token, completed


def _raise_capture_error(node: Mapping[str, Any], label: str) -> None:
    error = node.get("error")
    if error not in (None, "", False):
        raise SchemaError(f"cannot render failed {label}: {error}")


def _require_passed_report(report: Mapping[str, Any]) -> None:
    status = report.get("status")
    if not isinstance(status, str) or status.strip().lower() != "passed":
        raise SchemaError(
            "report status must be 'passed' before rendering; "
            f"got {status!r}"
        )
    overall_passed = _pick(
        report,
        (("overall_passed",), ("summary", "overall_passed")),
        "report overall_passed",
        default=_MISSING,
    )
    if overall_passed is not True:
        raise SchemaError(
            "report overall_passed must be true before rendering; "
            f"got {overall_passed!r}"
        )


def _require_matching_capture(report: Mapping[str, Any], capture_path: Path) -> None:
    declared = _at(report, ("provenance", "capture", "file_sha256"))
    if declared is _MISSING or declared is None:
        return
    if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", declared):
        raise SchemaError(f"report capture file_sha256 is invalid: {declared!r}")
    actual = _sha256(capture_path)
    if declared.lower() != actual:
        raise SchemaError(
            "report was generated from a different capture: "
            f"declared sha256:{declared.lower()[:12]}, actual sha256:{actual[:12]}"
        )


def _prompt_catalog(report: Mapping[str, Any]) -> dict[str, PromptInfo]:
    catalog: dict[str, PromptInfo] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            prompt_id = value.get("prompt_id", value.get("id"))
            label = value.get("label", value.get("prompt_label"))
            summary = value.get(
                "prompt_summary",
                value.get("summary", value.get("exact_prompt", value.get("prompt"))),
            )
            if isinstance(prompt_id, str) and isinstance(label, str):
                rendered_summary = summary if isinstance(summary, str) else label
                catalog.setdefault(
                    prompt_id,
                    PromptInfo(
                        prompt_id=prompt_id,
                        label=label.strip(),
                        summary=" ".join(rendered_summary.split()),
                    ),
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return catalog


def _prompt_for(
    catalog: Mapping[str, PromptInfo],
    prompt_id: str,
    fallback_label: Any,
    label: str,
) -> PromptInfo:
    if prompt_id in catalog:
        return catalog[prompt_id]
    if isinstance(fallback_label, str) and fallback_label.strip():
        clean = " ".join(fallback_label.split())
        return PromptInfo(prompt_id=prompt_id, label=clean, summary=clean)
    raise SchemaError(
        f"report has no prompt summary for {label} prompt_id={prompt_id!r}; "
        "add a records[] entry with prompt_id and label"
    )


def _metric(scope: Mapping[str, Any], key: str) -> float:
    aliases = {
        _normal_key(key),
        _normal_key(key.replace("_s", "")),
        _normal_key(key.replace("ttft", "first_token_latency")),
    }
    values = _recursive_values(scope, aliases)
    if not values:
        raise SchemaError(f"report metrics are missing {key}")
    return _number(values[0], f"report metric {key}")


def _memory_mib(snapshot: Any, label: str) -> float | None:
    if snapshot is None:
        return None
    if isinstance(snapshot, bool):
        return None
    if isinstance(snapshot, (int, float)):
        return _number(snapshot, label)
    if isinstance(snapshot, list):
        values = [_memory_mib(item, label) for item in snapshot]
        present = [value for value in values if value is not None]
        return sum(present) if present else None
    if not isinstance(snapshot, Mapping):
        return None

    direct_aliases = {
        "memoryusedmib",
        "usedmemorymib",
        "usedmib",
        "vramusedmib",
        "vrammib",
        "peakvrammib",
        "baselinevrammib",
        "totalusedmib",
    }
    for key, value in snapshot.items():
        if _normal_key(str(key)) in direct_aliases and isinstance(value, (int, float)):
            return _number(value, f"{label}.{key}")

    for key in ("gpus", "devices", "per_gpu", "by_gpu", "values"):
        if key in snapshot:
            nested = _memory_mib(snapshot[key], f"{label}.{key}")
            if nested is not None:
                return nested

    nested_values: list[float] = []
    for child in snapshot.values():
        if isinstance(child, (Mapping, list)):
            nested = _memory_mib(child, label)
            if nested is not None:
                nested_values.append(nested)
    return sum(nested_values) if nested_values else None


def _gpu_sample_mib(gpu: Any, field: str) -> float | None:
    aliases = {
        "baseline": {"baselineusedmib", "baselinemib", "baselinevrammib"},
        "peak": {"peakusedmib", "peakmib", "peakvrammib"},
    }[field]
    values = _recursive_values(gpu, aliases)
    numeric = [
        _number(value, f"GPU {field} memory")
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(numeric) if numeric else None


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_scalars(child, path))
    elif isinstance(value, (str, int, float, bool)) and not isinstance(value, bytes):
        result[prefix] = value
    return result


def _short_metric_name(path: str) -> str:
    name = path.split(".")[-1]
    return name.replace("_", " ")


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value).lower() if isinstance(value, bool) else str(value)


def _provider_snapshot_max(provider: Mapping[str, Any], key: str) -> int:
    aliases = {_normal_key(key)}
    values: list[Any] = []
    for snapshot_name in ("before", "after_first_turn", "after"):
        snapshot = provider.get(snapshot_name)
        if isinstance(snapshot, Mapping):
            values.extend(_recursive_values(snapshot, aliases))
    numeric = [
        _integer(value, f"provider {key}")
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numeric:
        raise SchemaError(
            f"provider snapshots are missing engine.{key}; expected before/"
            "after_first_turn/after metrics"
        )
    return max(numeric)


def _provider_follow_up_reuse_hits(provider: Mapping[str, Any]) -> int:
    value = _pick(
        provider,
        (
            ("follow_up_session_reuse_delta", "session_reuse_hits"),
            ("follow_up_delta", "engine", "session_reuse_hits"),
        ),
        "provider follow-up session reuse hits",
    )
    return _integer(value, "provider follow-up session reuse hits")


def _provider_error_count(provider: Mapping[str, Any]) -> int:
    value = _at(provider, ("delta", "server", "total_errors"))
    if value is _MISSING:
        value = _at(provider, ("follow_up_delta", "server", "total_errors"))
    if value is not _MISSING:
        return _integer(value, "provider error count")

    before = _at(provider, ("before", "metrics", "values", "server", "total_errors"))
    after = _at(provider, ("after", "metrics", "values", "server", "total_errors"))
    if before is _MISSING or after is _MISSING:
        raise SchemaError(
            "provider metrics are missing total_errors delta or before/after counters"
        )
    delta = _integer(after, "provider after total_errors") - _integer(
        before, "provider before total_errors"
    )
    if delta < 0:
        raise SchemaError(f"provider total_errors counter decreased by {-delta}")
    return delta


def _provider_observations(provider: Any) -> tuple[int, int, int, int]:
    provider = _mapping(provider, "capture concurrency provider metrics")
    return (
        _provider_snapshot_max(provider, "max_running"),
        _provider_snapshot_max(provider, "max_runnable_rows"),
        _provider_follow_up_reuse_hits(provider),
        _provider_error_count(provider),
    )


def _find_text(value: Any, aliases: set[str], default: str) -> str:
    found = _recursive_values(value, aliases)
    for candidate in found:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return default


def _model_identity(capture: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    exact_aliases = {"servedmodel", "modelid", "modelname", "model"}
    for source in (capture, report):
        exact = _find_text(source, exact_aliases, "")
        if exact:
            return exact

    tokenizer_identity = _pick(
        report,
        (
            ("provenance", "tokenizer", "report", "identity"),
            ("provenance", "tokenizer", "scenario", "identity"),
            ("tokenizer", "identity"),
        ),
        "tokenizer identity",
        default="",
    )
    if tokenizer_identity:
        return tokenizer_identity

    tokenizer_class = _find_text(
        report,
        {"tokenizerclass", "class"},
        "",
    )
    if tokenizer_class == "GemmaTokenizer":
        return "Gemma family (GemmaTokenizer)"
    if tokenizer_class:
        return f"WKVM model ({tokenizer_class})"
    return "WKVM model (identity not recorded)"


def _provenance_labels(capture: Mapping[str, Any]) -> tuple[str, str, str]:
    provenance = capture.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    browser_node = provenance.get("browser")
    browser_node = browser_node if isinstance(browser_node, Mapping) else {}
    webui_node = provenance.get("open_webui")
    webui_node = webui_node if isinstance(webui_node, Mapping) else {}
    browser_name = browser_node.get("engine", browser_node.get("channel", "Chromium"))
    browser_version = browser_node.get("version")
    browser = " ".join(
        str(value).strip()
        for value in (browser_name, browser_version)
        if value not in (None, "")
    )
    webui_version = webui_node.get("version")
    webui = f"Open WebUI {webui_version}" if webui_version else "Open WebUI"

    gpu_name = "GPU"
    acts = capture.get("acts")
    concurrency = acts.get("concurrency") if isinstance(acts, Mapping) else None
    gpu = concurrency.get("gpu") if isinstance(concurrency, Mapping) else None
    devices = gpu.get("devices") if isinstance(gpu, Mapping) else None
    if isinstance(devices, list):
        names = [
            str(device["name"]).strip()
            for device in devices
            if isinstance(device, Mapping) and device.get("name")
        ]
        if names:
            gpu_name = " + ".join(names)
    return browser or "recorded browser", webui, gpu_name


def load_evidence(capture_path: Path, report_path: Path) -> Evidence:
    capture_raw = _load_object(capture_path, "capture")
    report = _load_object(report_path, "report")
    _require_passed_report(report)
    _require_matching_capture(report, capture_path)
    capture = _mapping(capture_raw.get("artifact", capture_raw), "capture.artifact")
    acts = _mapping(_pick(capture, (("acts",),), "capture acts"), "capture.acts")
    catalog = _prompt_catalog(report)

    long_node = _mapping(
        _pick(
            acts,
            (("long_prompt",), ("long_chat",), ("long",)),
            "capture long-prompt act",
        ),
        "capture.acts.long_prompt",
    )
    _raise_capture_error(long_node, "long-prompt act")
    long_prompt_id = _text(
        _pick(long_node, (("prompt_id",), ("id",)), "long-prompt prompt_id"),
        "long-prompt prompt_id",
    )
    long_submitted, long_first, long_completed = _timing(long_node, "long-prompt act")
    long_act = Act(
        prompt=_prompt_for(
            catalog, long_prompt_id, long_node.get("label"), "long-prompt"
        ),
        video=_resolve_video(
            _pick(
                long_node,
                (("video_path",), ("webm",), ("video",), ("path",)),
                "long-prompt WebM path",
            ),
            capture_path,
            "long-prompt WebM path",
        ),
        submitted_s=long_submitted,
        first_token_s=long_first,
        completed_s=long_completed,
        rendered_token_count=(
            _integer(long_node["rendered_token_count"], "long-prompt rendered token count")
            if long_node.get("rendered_token_count") is not None
            else None
        ),
    )

    concurrency = _mapping(
        _pick(
            acts,
            (("concurrency",), ("classic_prompts",), ("classic",)),
            "capture concurrency act",
        ),
        "capture.acts.concurrency",
    )
    _raise_capture_error(concurrency, "concurrency act")
    sessions = _pick(
        concurrency,
        (("sessions",), ("videos",), ("acts",)),
        "capture concurrency sessions",
    )
    if not isinstance(sessions, list) or len(sessions) != 4:
        count = len(sessions) if isinstance(sessions, list) else "non-list"
        raise SchemaError(
            f"capture concurrency sessions must contain exactly four videos; got {count}"
        )

    classic_acts: list[Act] = []
    for index, raw_session in enumerate(sessions):
        session = _mapping(raw_session, f"capture concurrency session {index + 1}")
        _raise_capture_error(session, f"concurrency session {index + 1} capture")
        turn = _mapping(
            _pick(
                session,
                (("first_turn",), ("turn",), ("request",)),
                f"concurrency session {index + 1} first turn",
            ),
            f"concurrency session {index + 1}.first_turn",
        )
        _raise_capture_error(turn, f"concurrency session {index + 1}")
        prompt_id = _text(
            _pick(
                turn,
                (("prompt_id",),),
                f"concurrency session {index + 1} prompt_id",
                default=session.get("prompt_id"),
            ),
            f"concurrency session {index + 1} prompt_id",
        )
        submitted, first_token, completed = _timing(
            turn, f"concurrency session {index + 1}"
        )
        follow_up = _mapping(
            _pick(
                session,
                (("follow_up",),),
                f"concurrency session {index + 1} follow-up",
            ),
            f"concurrency session {index + 1}.follow_up",
        )
        _raise_capture_error(follow_up, f"concurrency session {index + 1} follow-up")
        follow_submitted, follow_first_token, follow_completed = _timing(
            follow_up, f"concurrency session {index + 1} follow-up"
        )
        if follow_submitted < completed:
            raise SchemaError(
                f"concurrency session {index + 1} follow-up submission precedes "
                "first-turn completion"
            )
        classic_acts.append(
            Act(
                prompt=_prompt_for(
                    catalog,
                    prompt_id,
                    turn.get("label", session.get("label")),
                    f"concurrency session {index + 1}",
                ),
                video=_resolve_video(
                    _pick(
                        session,
                        (("video_path",), ("webm",), ("video",), ("path",)),
                        f"concurrency session {index + 1} WebM path",
                    ),
                    capture_path,
                    f"concurrency session {index + 1} WebM path",
                ),
                submitted_s=submitted,
                first_token_s=first_token,
                completed_s=completed,
                follow_up=Turn(
                    submitted_s=follow_submitted,
                    first_token_s=follow_first_token,
                    completed_s=follow_completed,
                ),
            )
        )

    report_acts = report.get("acts", {})
    metric_scope = _mapping(
        _pick(
            report_acts,
            (("concurrency_first_turn",), ("concurrency",), ("classic",)),
            "report concurrency metrics",
            default=report.get("summary", report),
        ),
        "report concurrency metrics",
    )
    offered = _integer(
        _pick(
            metric_scope,
            (("offered_concurrency",), ("concurrency",), ("request_count",)),
            "offered concurrency",
            default=4,
        ),
        "offered concurrency",
    )
    success = _integer(
        _pick(
            metric_scope,
            (("success_count",), ("successful_requests",), ("succeeded",)),
            "success count",
        ),
        "success count",
    )
    follow_up_scope = _mapping(
        _pick(
            report_acts,
            (("concurrency_follow_up",),),
            "report concurrency follow-up metrics",
        ),
        "report concurrency follow-up metrics",
    )
    follow_up_success = _integer(
        _pick(
            follow_up_scope,
            (("success_count",), ("successful_requests",), ("succeeded",)),
            "follow-up success count",
        ),
        "follow-up success count",
    )
    error_count = _integer(
        _pick(
            report,
            (
                ("summary", "all_error_count"),
                ("acts", "all_requests", "error_count"),
                ("summary", "error_count"),
            ),
            "all-request error count",
        ),
        "all-request error count",
    )
    if offered != 4:
        raise SchemaError(f"renderer requires offered_concurrency=4; report says {offered}")

    gpu = concurrency.get("gpu", long_node.get("gpu"))
    baseline = _gpu_sample_mib(gpu, "baseline")
    peak = _gpu_sample_mib(gpu, "peak")
    if isinstance(gpu, Mapping):
        if baseline is None:
            baseline = _memory_mib(
                _pick(
                    gpu,
                    (("baseline",), ("before",), ("baseline_mib",)),
                    "GPU baseline",
                    default=None,
                ),
                "GPU baseline",
            )
        if peak is None:
            peak = _memory_mib(
                _pick(
                    gpu,
                    (("peak",), ("after",), ("peak_mib",)),
                    "GPU peak",
                    default=None,
                ),
                "GPU peak",
            )
    if peak is None:
        peak_values = _recursive_values(
            report,
            {"peakvrammib", "gpupeakmib", "peakmemoryusedmib"},
        )
        peak = _number(peak_values[0], "peak VRAM MiB") if peak_values else None
    if peak is None:
        raise SchemaError(
            "missing peak VRAM; capture.acts.concurrency.gpu must include a peak "
            "snapshot with memory_used_mib"
        )

    semantics_values = _recursive_values(
        report,
        {"semantics", "modelsemantics", "servingsemantics"},
    )
    for value in semantics_values:
        if isinstance(value, str) and value and value != SEMANTICS:
            raise SchemaError(
                f"renderer is scoped to semantics={SEMANTICS}; report says {value!r}"
            )

    provider = concurrency.get("provider", concurrency.get("provider_metrics"))
    max_running, max_runnable_rows, exact_reuse_hits, provider_errors = (
        _provider_observations(provider)
    )
    provider_summary = (
        f"max_running {max_running} · max_runnable_rows {max_runnable_rows} · "
        f"exact reuse hits {exact_reuse_hits}/{offered} · errors {provider_errors}"
    )
    first_turn_min_output_tokens = metric_scope.get("output_tokens_min")
    if type(first_turn_min_output_tokens) is not int:
        first_turn_min_output_tokens = None
    follow_up_min_output_tokens = follow_up_scope.get("output_tokens_min")
    if type(follow_up_min_output_tokens) is not int:
        follow_up_min_output_tokens = None
    first_turn_output_tokens = metric_scope.get("output_tokens")
    follow_up_output_tokens = follow_up_scope.get("output_tokens")
    act_2_total_output_tokens = (
        first_turn_output_tokens + follow_up_output_tokens
        if type(first_turn_output_tokens) is int
        and type(follow_up_output_tokens) is int
        else None
    )
    metrics = Metrics(
        offered_concurrency=offered,
        success_count=success,
        follow_up_success_count=follow_up_success,
        error_count=error_count,
        ttft_p50_s=_metric(metric_scope, "ttft_p50_s"),
        ttft_p95_s=_metric(metric_scope, "ttft_p95_s"),
        e2e_p50_s=_metric(metric_scope, "e2e_p50_s"),
        e2e_p95_s=_metric(metric_scope, "e2e_p95_s"),
        baseline_vram_mib=baseline,
        peak_vram_mib=peak,
        max_running=max_running,
        max_runnable_rows=max_runnable_rows,
        exact_reuse_hits=exact_reuse_hits,
        provider_error_count=provider_errors,
        provider_summary=provider_summary,
        first_turn_min_output_tokens=first_turn_min_output_tokens,
        follow_up_min_output_tokens=follow_up_min_output_tokens,
        act_2_total_output_tokens=act_2_total_output_tokens,
    )

    model = _model_identity(capture, report)
    browser, webui, gpu_name = _provenance_labels(capture)
    scoped_evidence = _find_text(
        report,
        {"scopedevidence", "scoped48turnevidence", "benchmarkevidence"},
        "separate scoped 48-turn evidence",
    )
    return Evidence(
        capture_path=capture_path,
        report_path=report_path,
        long_act=long_act,
        classic_acts=tuple(classic_acts),  # type: ignore[arg-type]
        metrics=metrics,
        model=model,
        gpu=gpu_name,
        provenance=f"{browser} · {webui}",
        scoped_evidence=scoped_evidence,
    )


def _fmt_seconds(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"{rendered} s"


def _fmt_mib(value: float) -> str:
    rendered = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{rendered} MiB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_timeline(evidence: Evidence) -> Timeline:
    long_speed = max(1.0, evidence.long_act.e2e_s / LONG_MAX_DISPLAY_SECONDS)
    long_display = evidence.long_act.e2e_s / long_speed
    montage_real = max(act.recording_duration_s for act in evidence.classic_acts)
    montage_speed = max(1.0, montage_real / MONTAGE_MAX_DISPLAY_SECONDS)
    montage_display = montage_real / montage_speed
    return Timeline(
        long_speed=long_speed,
        long_display_s=long_display,
        montage_speed=montage_speed,
        montage_display_s=montage_display,
    )


def build_visual_specs(
    evidence: Evidence, timeline: Timeline, mp4_name: str
) -> dict[str, VisualSpec]:
    metrics = evidence.metrics
    caveat = f"semantics: {SEMANTICS} · functional browser demo · {TEN_X_CAVEAT}"
    vram = f"peak VRAM {_fmt_mib(metrics.peak_vram_mib)}"
    if metrics.baseline_vram_mib is not None:
        vram = (
            f"VRAM {_fmt_mib(metrics.baseline_vram_mib)} baseline → "
            f"{_fmt_mib(metrics.peak_vram_mib)} peak"
        )
    evidence_hash = (
        f"capture sha256:{_sha256(evidence.capture_path)[:12]} · "
        f"report sha256:{_sha256(evidence.report_path)[:12]}"
    )
    provider_row = (
        f"provider observed · max_running {metrics.max_running} · "
        f"max_runnable_rows {metrics.max_runnable_rows}"
    )
    reuse_row = (
        f"follow-ups · {metrics.exact_reuse_hits}/{metrics.offered_concurrency} "
        f"exact reuse hits · report errors {metrics.error_count} · "
        f"provider errors {metrics.provider_error_count}"
    )
    length_row = None
    if (
        metrics.first_turn_min_output_tokens is not None
        and metrics.follow_up_min_output_tokens is not None
    ):
        length_row = (
            "Act 2 output · minimum "
            f"{min(metrics.first_turn_min_output_tokens, metrics.follow_up_min_output_tokens):,} "
            "tokens/turn"
        )
        if metrics.act_2_total_output_tokens is not None:
            length_row += f" · {metrics.act_2_total_output_tokens:,} tokens total"
    long_scale = (
        f"{evidence.long_act.rendered_token_count:,} rendered tokens · "
        if evidence.long_act.rendered_token_count is not None
        else ""
    )

    pane_labels = tuple(
        (
            f"{index + 1}. {act.prompt.label}",
            f"first E2E {_fmt_seconds(act.e2e_s)} · follow-up E2E "
            f"{_fmt_seconds(act.follow_up.e2e_s) if act.follow_up else 'missing'}",
        )
        for index, act in enumerate(evidence.classic_acts)
    )
    speed_note = "capture shown at 1×"
    if timeline.long_speed > 1.001:
        speed_note = (
            f"capture shown at {timeline.long_speed:.2f}×; measured timing labels unchanged"
        )
    montage_speed_note = "four recordings shown at 1×"
    if timeline.montage_speed > 1.001:
        montage_speed_note = (
            f"four recordings shown at {timeline.montage_speed:.2f}×; alignment preserved"
        )

    return {
        "title": VisualSpec(
            role="title",
            eyebrow="WKVM × OPEN WEBUI · RECORDED BROWSER EVIDENCE",
            title="One long chat. Four concurrent chats.",
            subtitle=f"{evidence.model} · {evidence.gpu}",
            rows=tuple(
                row
                for row in (
                f"LONG CHAT · {evidence.long_act.prompt.summary}",
                f"{long_scale}TTFT {_fmt_seconds(evidence.long_act.ttft_s)} · "
                f"E2E {_fmt_seconds(evidence.long_act.e2e_s)}",
                f"CONCURRENCY {metrics.offered_concurrency} · success "
                f"{metrics.success_count}/{metrics.offered_concurrency} first turns · "
                f"{metrics.follow_up_success_count}/{metrics.offered_concurrency} follow-ups",
                f"TTFT p50/p95 {_fmt_seconds(metrics.ttft_p50_s)} / "
                f"{_fmt_seconds(metrics.ttft_p95_s)}",
                f"E2E p50/p95 {_fmt_seconds(metrics.e2e_p50_s)} / "
                f"{_fmt_seconds(metrics.e2e_p95_s)} · {vram}",
                provider_row,
                length_row,
                reuse_row,
                )
                if row is not None
            ),
            footer=f"{evidence_hash} · {evidence.provenance}",
            caveat=caveat,
        ),
        "long_overlay": VisualSpec(
            role="long_overlay",
            eyebrow="ACT 1 · LONG-PROMPT WEBUI CHAT",
            title=evidence.long_act.prompt.label,
            subtitle=(
                f"{long_scale}TTFT {_fmt_seconds(evidence.long_act.ttft_s)} · "
                f"E2E {_fmt_seconds(evidence.long_act.e2e_s)}"
            ),
            rows=(evidence.long_act.prompt.summary, speed_note),
            caveat=f"{SEMANTICS} · {TEN_X_CAVEAT}",
            transparent=True,
        ),
        "transition": VisualSpec(
            role="transition",
            eyebrow="ACT 2 · SYNCHRONIZED BROWSER CAPTURE",
            title="Four classic prompts, then four follow-ups.",
            subtitle="Each pane includes both measured turns from the same chat.",
            rows=tuple(act.prompt.summary for act in evidence.classic_acts),
            footer=montage_speed_note,
            caveat=caveat,
        ),
        "montage_overlay": VisualSpec(
            role="montage_overlay",
            eyebrow="ACT 2 · FOUR CHATS ALIGNED AT FIRST-TURN SUBMIT t=0",
            title="Open WebUI → WKVM · first turns + follow-ups",
            rows=tuple(
                row
                for row in (
                f"offered concurrency {metrics.offered_concurrency} · first turns "
                f"{metrics.success_count}/{metrics.offered_concurrency} · follow-ups "
                f"{metrics.follow_up_success_count}/{metrics.offered_concurrency}",
                f"TTFT p50/p95 {_fmt_seconds(metrics.ttft_p50_s)} / "
                f"{_fmt_seconds(metrics.ttft_p95_s)} · E2E p50/p95 "
                f"{_fmt_seconds(metrics.e2e_p50_s)} / {_fmt_seconds(metrics.e2e_p95_s)}",
                length_row,
                provider_row,
                f"{metrics.exact_reuse_hits}/{metrics.offered_concurrency} exact reuse "
                f"hits · errors {metrics.error_count} report / "
                f"{metrics.provider_error_count} provider · {vram} · {montage_speed_note}",
                )
                if row is not None
            ),
            caveat=f"semantics: {SEMANTICS} · {TEN_X_CAVEAT}",
            pane_labels=pane_labels,
            transparent=True,
        ),
        "final": VisualSpec(
            role="final",
            eyebrow="EVIDENCE BOUNDARY",
            title="A functional demo, with a deliberately narrow claim.",
            subtitle="What this recording establishes",
            rows=(
                "✓ Open WebUI completes a measured long-prompt WKVM chat.",
                f"✓ {metrics.success_count}/{metrics.offered_concurrency} first turns and "
                f"{metrics.follow_up_success_count}/{metrics.offered_concurrency} follow-ups "
                "complete in four recorded chats.",
                f"✓ Provider observed max_running {metrics.max_running} and "
                f"max_runnable_rows {metrics.max_runnable_rows}.",
                (
                    "✓ Every Act 2 response contains at least "
                    f"{min(metrics.first_turn_min_output_tokens, metrics.follow_up_min_output_tokens):,} "
                    "measured output tokens."
                    if metrics.first_turn_min_output_tokens is not None
                    and metrics.follow_up_min_output_tokens is not None
                    else "✓ Act 2 output-length evidence is recorded in the report."
                ),
                f"✓ {metrics.exact_reuse_hits}/{metrics.offered_concurrency} exact reuse "
                f"hits · errors {metrics.error_count} report / "
                f"{metrics.provider_error_count} provider.",
                f"This is {TEN_X_CAVEAT}.",
                f"The {evidence.scoped_evidence} is a separate artifact.",
                f"Serving semantics: {SEMANTICS}.",
            ),
            footer=f"Full video: {mp4_name}",
            caveat="Functional UI evidence ≠ cross-engine throughput comparison",
        ),
    }


def _font_paths() -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    regular = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    bold = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    )
    return regular, bold


def _load_font(size: int, *, bold: bool = False):
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise SystemExit("Pillow is required to render cards: python -m pip install Pillow") from exc
    candidates = _font_paths()[1 if bold else 0]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise SystemExit(
        "No supported sans-serif font found; install fonts-dejavu-core or liberation-fonts"
    )


def _gradient_image(*, transparent: bool = False):
    from PIL import Image, ImageDraw

    mode = "RGBA" if transparent else "RGB"
    image = Image.new(mode, (WIDTH, HEIGHT), (0, 0, 0, 0) if transparent else BACKGROUND)
    if transparent:
        return image
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        mix = y / max(HEIGHT - 1, 1)
        color = (
            int(BACKGROUND[0] + 8 * mix),
            int(BACKGROUND[1] + 12 * mix),
            int(BACKGROUND[2] + 18 * mix),
        )
        draw.line((0, y, WIDTH, y), fill=color)
    draw.ellipse((930, -250, 1450, 270), fill=(18, 65, 81))
    draw.ellipse((-250, 520, 310, 1080), fill=(31, 42, 75))
    return image


def _text_width(draw: Any, text: str, font: Any) -> float:
    return draw.textbbox((0, 0), text, font=font)[2]


def _ellipsize(draw: Any, text: str, font: Any, width: int) -> str:
    clean = " ".join(text.split())
    if _text_width(draw, clean, font) <= width:
        return clean
    suffix = "…"
    while clean and _text_width(draw, clean + suffix, font) > width:
        clean = clean[:-1]
    return clean.rstrip() + suffix


def _wrap(draw: Any, text: str, font: Any, width: int, max_lines: int) -> list[str]:
    words = " ".join(text.split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if _text_width(draw, candidate, font) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and words:
        lines[-1] = _ellipsize(draw, lines[-1], font, width)
    return lines


def _rounded_panel(draw: Any, box: tuple[int, int, int, int], *, fill: Any) -> None:
    draw.rounded_rectangle(box, radius=18, fill=fill, outline=(61, 79, 104), width=2)


def _render_card(spec: VisualSpec, path: Path) -> None:
    from PIL import ImageDraw

    image = _gradient_image(transparent=spec.transparent)
    draw = ImageDraw.Draw(image, "RGBA" if spec.transparent else None)
    if spec.role == "long_overlay":
        draw.rectangle((0, 0, WIDTH, 78), fill=(8, 12, 20, 225))
        draw.rectangle((0, 626, WIDTH, HEIGHT), fill=(8, 12, 20, 235))
        draw.text((28, 14), spec.eyebrow, font=_load_font(16, bold=True), fill=TEAL)
        draw.text((28, 39), _ellipsize(draw, spec.title, _load_font(22, bold=True), 760),
                  font=_load_font(22, bold=True), fill=INK)
        right_font = _load_font(19, bold=True)
        draw.text((WIDTH - 28 - _text_width(draw, spec.subtitle, right_font), 31),
                  spec.subtitle, font=right_font, fill=CYAN)
        draw.text((28, 642), _ellipsize(draw, spec.rows[0], _load_font(17), 1200),
                  font=_load_font(17), fill=INK)
        draw.text((28, 674), spec.rows[1], font=_load_font(14), fill=MUTED)
        caveat_font = _load_font(14, bold=True)
        draw.text((WIDTH - 28 - _text_width(draw, spec.caveat, caveat_font), 674),
                  spec.caveat, font=caveat_font, fill=AMBER)
    elif spec.role == "montage_overlay":
        draw.rectangle((0, 0, WIDTH, 58), fill=(8, 12, 20, 238))
        draw.rectangle((0, 604, WIDTH, HEIGHT), fill=(8, 12, 20, 244))
        draw.text((18, 10), spec.eyebrow, font=_load_font(15, bold=True), fill=TEAL)
        draw.text((18, 31), spec.title, font=_load_font(18, bold=True), fill=INK)
        pane_positions = ((8, 64), (646, 64), (8, 336), (646, 336))
        for (x, y), (first, second) in zip(pane_positions, spec.pane_labels):
            draw.rectangle((x, y, x + 626, y + 48), fill=(8, 12, 20, 218))
            draw.rectangle((x, y, x + 626, y + 260), outline=(70, 222, 190, 210), width=2)
            draw.text((x + 10, y + 6), _ellipsize(draw, first, _load_font(16, bold=True), 600),
                      font=_load_font(16, bold=True), fill=INK)
            draw.text((x + 10, y + 27), second, font=_load_font(13), fill=CYAN)
        draw.text((18, 616), spec.rows[0], font=_load_font(18, bold=True), fill=TEAL)
        draw.text((18, 645), spec.rows[1], font=_load_font(16, bold=True), fill=INK)
        detail_font = _load_font(12)
        draw.text(
            (18, 672),
            _ellipsize(draw, spec.rows[2], detail_font, WIDTH - 36),
            font=detail_font,
            fill=MUTED,
        )
        draw.text(
            (18, 695),
            _ellipsize(draw, spec.rows[3], detail_font, WIDTH - 36),
            font=detail_font,
            fill=MUTED,
        )
        caveat_font = _load_font(14, bold=True)
        draw.text((WIDTH - 18 - _text_width(draw, spec.caveat, caveat_font), 615),
                  spec.caveat, font=caveat_font, fill=AMBER)
    else:
        draw.text((72, 54), spec.eyebrow, font=_load_font(17, bold=True), fill=TEAL)
        title_size = 42 if spec.role != "transition" else 38
        title_font = _load_font(title_size, bold=True)
        title_lines = _wrap(draw, spec.title, title_font, WIDTH - 144, 2)
        y = 96
        for line in title_lines:
            draw.text((72, y), line, font=title_font, fill=INK)
            y += title_size + 8
        if spec.subtitle:
            draw.text((72, y + 4), spec.subtitle, font=_load_font(21), fill=CYAN)
            y += 48
        panel_top = max(220, y + 12)
        panel_bottom = 598
        _rounded_panel(draw, (64, panel_top, WIDTH - 64, panel_bottom), fill=PANEL)
        row_font = _load_font(18, bold=spec.role == "final")
        row_y = panel_top + 22
        for index, row in enumerate(spec.rows):
            color = INK
            if row.startswith("✓"):
                color = TEAL
            elif TEN_X_CAVEAT in row:
                color = AMBER
            elif index == 0 and spec.role == "title":
                color = CYAN
            for line in _wrap(draw, row, row_font, WIDTH - 180, 2):
                draw.text((90, row_y), line, font=row_font, fill=color)
                row_y += 26
            row_y += 8
        draw.text((72, 628), _ellipsize(draw, spec.footer, _load_font(13), WIDTH - 144),
                  font=_load_font(13), fill=MUTED)
        caveat_font = _load_font(15, bold=True)
        caveat_x = WIDTH - 72 - _text_width(draw, spec.caveat, caveat_font)
        draw.rounded_rectangle((caveat_x - 14, 662, WIDTH - 58, 700), radius=14,
                               fill=(74, 54, 20, 210), outline=(140, 101, 34), width=1)
        draw.text((caveat_x, 672), spec.caveat, font=caveat_font, fill=AMBER)
    image.save(path, format="PNG", optimize=False)


def _seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _video_filter(
    input_index: int,
    act: Act,
    speed: float,
    output_label: str,
    *,
    pane: bool,
    pad_to_s: float | None = None,
) -> str:
    filters = [
        f"[{input_index}:v]trim=start={_seconds(act.submitted_s)}:"
        f"end={_seconds(act.recording_completed_s)}",
        f"setpts=(PTS-STARTPTS)/{_seconds(speed)}",
        f"fps={FPS}",
    ]
    if pane:
        filters.extend(
            (
                "scale=626:260:force_original_aspect_ratio=decrease:flags=lanczos",
                "pad=626:260:(ow-iw)/2:(oh-ih)/2:color=0x090d16",
            )
        )
    else:
        filters.extend(
            (
                "scale=1280:720:force_original_aspect_ratio=decrease:flags=lanczos",
                "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x090d16",
            )
        )
    if pad_to_s is not None:
        own_duration = act.recording_duration_s / speed
        pad = max(0.0, pad_to_s - own_duration)
        filters.extend(
            (
                f"tpad=stop_mode=clone:stop_duration={_seconds(pad)}",
                f"trim=duration={_seconds(pad_to_s)}",
            )
        )
    filters.extend(("setsar=1", "format=yuv420p"))
    return ",".join(filters) + f"[{output_label}]"


def build_mp4_command(
    evidence: Evidence,
    timeline: Timeline,
    assets: Mapping[str, Path],
    mp4_path: Path,
) -> list[str]:
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]

    def image_input(path: Path, duration: float) -> None:
        command.extend(
            ("-loop", "1", "-framerate", str(FPS), "-t", _seconds(duration), "-i", str(path))
        )

    image_input(assets["title"], TITLE_SECONDS)  # input 0
    command.extend(("-i", str(evidence.long_act.video)))  # input 1
    image_input(assets["long_overlay"], timeline.long_display_s)  # input 2
    image_input(assets["transition"], TRANSITION_SECONDS)  # input 3
    for act in evidence.classic_acts:  # inputs 4..7
        command.extend(("-i", str(act.video)))
    image_input(assets["montage_overlay"], timeline.montage_display_s)  # input 8
    image_input(assets["final"], FINAL_SECONDS)  # input 9

    filters = [
        f"[0:v]scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={FPS},setsar=1,"
        f"trim=duration={_seconds(TITLE_SECONDS)},setpts=PTS-STARTPTS,"
        "format=yuv420p[title]",
        _video_filter(1, evidence.long_act, timeline.long_speed, "longbase", pane=False),
        f"[2:v]fps={FPS},format=rgba[longov]",
        f"[longbase][longov]overlay=shortest=1:format=auto,"
        f"trim=duration={_seconds(timeline.long_display_s)},setpts=PTS-STARTPTS,"
        "format=yuv420p[long]",
        f"[3:v]scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={FPS},setsar=1,"
        f"trim=duration={_seconds(TRANSITION_SECONDS)},setpts=PTS-STARTPTS,"
        "format=yuv420p[transition]",
    ]
    for index, act in enumerate(evidence.classic_acts):
        filters.append(
            _video_filter(
                4 + index,
                act,
                timeline.montage_speed,
                f"pane{index}",
                pane=True,
                pad_to_s=timeline.montage_display_s,
            )
        )
    filters.extend(
        (
            "[pane0][pane1][pane2][pane3]xstack=inputs=4:"
            "layout=0_0|638_0|0_272|638_272:fill=0x090d16,"
            "pad=1280:720:8:64:color=0x090d16,setsar=1[grid]",
            f"[8:v]fps={FPS},format=rgba[montageov]",
            f"[grid][montageov]overlay=shortest=1:format=auto,"
            f"trim=duration={_seconds(timeline.montage_display_s)},"
            "setpts=PTS-STARTPTS,format=yuv420p[montage]",
            f"[9:v]scale={WIDTH}:{HEIGHT}:flags=lanczos,fps={FPS},setsar=1,"
            f"trim=duration={_seconds(FINAL_SECONDS)},setpts=PTS-STARTPTS,"
            "format=yuv420p[final]",
            "[title][long][transition][montage][final]"
            f"concat=n=5:v=1:a=0,fps={FPS}[outv]",
        )
    )
    target_bits = 8.6 * 1024 * 1024 * 8
    bitrate_kbps = max(350, min(1200, int(target_bits / timeline.total_s / 1000)))
    command.extend(
        (
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-b:v",
            f"{bitrate_kbps}k",
            "-maxrate",
            f"{bitrate_kbps}k",
            "-bufsize",
            f"{bitrate_kbps * 2}k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-map_metadata",
            "-1",
            "-metadata",
            "creation_time=1970-01-01T00:00:00Z",
            str(mp4_path),
        )
    )
    return command


def build_gif_command(timeline: Timeline, mp4_path: Path, gif_path: Path) -> list[str]:
    long_first = TITLE_SECONDS + min(
        timeline.long_display_s,
        max(0.0, timeline.long_display_s / 2),
    )
    long_start = max(TITLE_SECONDS, long_first - 0.75)
    long_end = min(TITLE_SECONDS + timeline.long_display_s, long_start + 2.0)
    montage_end = min(timeline.final_start_s, timeline.montage_start_s + 3.0)
    segments = (
        (0.0, min(1.5, TITLE_SECONDS)),
        (long_start, long_end),
        (timeline.montage_start_s, montage_end),
        (timeline.final_start_s, timeline.final_start_s + 1.5),
    )
    filters: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(segments):
        if end <= start:
            continue
        label = f"g{index}"
        labels.append(f"[{label}]")
        filters.append(
            f"[0:v]trim=start={_seconds(start)}:end={_seconds(end)},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
    filters.append(
        "".join(labels)
        + f"concat=n={len(labels)}:v=1:a=0,fps=10,"
        "scale=960:540:flags=lanczos,split=2[gifbase][palettebase]"
    )
    filters.extend(
        (
            "[palettebase]palettegen=max_colors=96:stats_mode=diff[palette]",
            "[gifbase][palette]paletteuse=dither=bayer:bayer_scale=5:"
            "diff_mode=rectangle[outgif]",
        )
    )
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(mp4_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outgif]",
        "-loop",
        "0",
        str(gif_path),
    ]


def _asset_paths(work_dir: Path) -> dict[str, Path]:
    return {
        "title": work_dir / "00-title.png",
        "long_overlay": work_dir / "01-long-overlay.png",
        "transition": work_dir / "02-transition.png",
        "montage_overlay": work_dir / "03-montage-overlay.png",
        "final": work_dir / "04-final.png",
    }


def _validate_inputs(evidence: Evidence) -> None:
    missing = [act.video for act in (evidence.long_act, *evidence.classic_acts) if not act.video.is_file()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"capture WebM file(s) not found:\n{rendered}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found; install ffmpeg and retry")


def _validate_output(path: Path, limit: int, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"{label} was not created: {path}")
    if path.stat().st_size >= limit:
        raise SystemExit(
            f"{label} is {path.stat().st_size / 2**20:.2f} MiB, above the "
            f"{limit / 2**20:.0f} MiB target: {path}"
        )


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _markdown_link(mp4_path: Path, gif_path: Path) -> str:
    return f"[![WKVM Open WebUI demo]({gif_path.name})]({mp4_path.name})"


def _render_in_work_dir(
    evidence: Evidence,
    timeline: Timeline,
    specs: Mapping[str, VisualSpec],
    work_dir: Path,
    mp4_path: Path,
    gif_path: Path,
    *,
    dry_run: bool,
) -> None:
    assets = _asset_paths(work_dir)
    mp4_command = build_mp4_command(evidence, timeline, assets, mp4_path)
    gif_command = build_gif_command(timeline, mp4_path, gif_path)
    if dry_run:
        print(f"MP4_COMMAND {shlex.join(mp4_command)}")
        print(f"GIF_COMMAND {shlex.join(gif_command)}")
        print(f"MARKDOWN {_markdown_link(mp4_path, gif_path)}")
        return

    work_dir.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        _render_card(spec, assets[name])
    _run(mp4_command)
    _validate_output(mp4_path, MP4_SIZE_LIMIT, "MP4")
    _run(gif_command)
    _validate_output(gif_path, GIF_SIZE_LIMIT, "GIF")
    print(
        f"RENDER_OK mp4={mp4_path} ({mp4_path.stat().st_size / 2**20:.2f} MiB) "
        f"gif={gif_path} ({gif_path.stat().st_size / 2**20:.2f} MiB)"
    )
    print(f"MARKDOWN {_markdown_link(mp4_path, gif_path)}")


def cmd_render(args: argparse.Namespace) -> None:
    capture_path = Path(args.capture).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    mp4_path = Path(args.mp4).expanduser().resolve()
    gif_path = Path(args.gif).expanduser().resolve()
    if mp4_path == gif_path:
        raise SystemExit("--mp4 and --gif must be different output paths")
    if mp4_path.suffix.lower() != ".mp4" or gif_path.suffix.lower() != ".gif":
        raise SystemExit("--mp4 must end in .mp4 and --gif must end in .gif")

    evidence = load_evidence(capture_path, report_path)
    timeline = build_timeline(evidence)
    specs = build_visual_specs(evidence, timeline, mp4_path.name)
    if not args.dry_run:
        _validate_inputs(evidence)

    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        _render_in_work_dir(
            evidence,
            timeline,
            specs,
            work_dir,
            mp4_path,
            gif_path,
            dry_run=args.dry_run,
        )
        return

    if args.dry_run:
        work_dir = mp4_path.parent / f".{mp4_path.stem}-render-dry-run"
        _render_in_work_dir(
            evidence,
            timeline,
            specs,
            work_dir,
            mp4_path,
            gif_path,
            dry_run=True,
        )
        return

    with tempfile.TemporaryDirectory(prefix="wkvm-open-webui-render-") as temporary:
        _render_in_work_dir(
            evidence,
            timeline,
            specs,
            Path(temporary),
            mp4_path,
            gif_path,
            dry_run=False,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="render MP4 and linked GIF teaser")
    render.add_argument("--capture", required=True, help="browser capture artifact JSON")
    render.add_argument("--report", required=True, help="measured demo report JSON")
    render.add_argument("--mp4", required=True, help="output H.264 MP4")
    render.add_argument("--gif", required=True, help="output palette GIF teaser")
    render.add_argument("--work-dir", help="preserve generated card PNGs in this directory")
    render.add_argument(
        "--dry-run",
        action="store_true",
        help="validate JSON and print deterministic ffmpeg commands without writing files",
    )
    render.set_defaults(function=cmd_render)
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except SchemaError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
