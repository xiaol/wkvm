#!/usr/bin/env python
"""Build and report the deterministic WKVM Open WebUI video scenario.

The scenario is intentionally a live UI demonstration rather than a controlled
engine comparison.  ``build`` creates the prompts and their fingerprints;
``report`` validates a browser capture and summarizes browser-observed timing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_KIND = "wkvm.open_webui.demo_scenario"
REPORT_KIND = "wkvm.open_webui.demo_report"
SCHEMA_VERSION = 1
OFFERED_CONCURRENCY = 4
NEEDLE_TARGET_RENDERED_TOKEN = 256
HF_LONG_SOURCE_DATASET_ID = (
    "Thermostatic/project-gutenberg-frankenstein-chapters"
)
HF_LONG_SOURCE_REVISION = "e37ee04474b60bdf4cc680dfc41ed9dd453cf7fc"
HF_LONG_SOURCE_CONFIG = "default"
HF_LONG_SOURCE_SPLIT = "train"
HF_LONG_SOURCE_FILENAME = "frankenstein_chapters.parquet"
HF_LONG_SOURCE_FILE_SHA256 = (
    "512da16deee193ba4fe32e7e8273c1aef02568ab36975811b3345d9b337cad9b"
)
HF_LONG_SOURCE_TEXT_SHA256 = (
    "cbac39268b43c020cc4d9d6ff0e690657ff91417c4165481733b42f5b129dfd4"
)
HF_LONG_SOURCE_ROWS = 28
CAVEATS = (
    "This is a normal four-slot Open WebUI demo, not a controlled load test.",
    "WKVM serves this demo with routed_span_approximate model-state semantics.",
    "This capture is not a vLLM/SGLang comparison or proof of a 10x claim.",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError("tokenizer did not return a token-id list")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("batched tokenization is not supported")
        value = list(value[0])
    if not all(type(token_id) is int for token_id in value):
        raise TypeError("tokenizer returned non-integer token IDs")
    return list(value)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    return _normalize_token_ids(
        tokenizer.encode(text, add_special_tokens=False)
    )


def render_chat_token_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool = True,
) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    except TypeError:
        kwargs.pop("return_dict")
        rendered = tokenizer.apply_chat_template(list(messages), **kwargs)
    return _normalize_token_ids(rendered)


def _safe_tokenizer_identity(tokenizer_path: str, tokenizer: Any) -> str | None:
    requested = Path(tokenizer_path).expanduser()
    if not requested.is_absolute() and not requested.exists():
        return tokenizer_path
    candidate = getattr(tokenizer, "name_or_path", None)
    if not isinstance(candidate, str) or not candidate:
        return None
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_absolute() or candidate_path.exists():
        return None
    return candidate


def tokenizer_fingerprint(
    tokenizer: Any,
    *,
    identity: str | None = None,
) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        chat_template = None
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size < 0:
        try:
            vocab_size = len(tokenizer)
        except (TypeError, AttributeError):
            vocab_size = None
    return {
        "identity": identity,
        "class": type(tokenizer).__name__,
        "vocab_size": vocab_size,
        "chat_template_sha256": (
            text_sha256(chat_template) if chat_template is not None else None
        ),
        "bos_token_id": _optional_nonnegative_int(
            getattr(tokenizer, "bos_token_id", None)
        ),
        "eos_token_id": _optional_nonnegative_int(
            getattr(tokenizer, "eos_token_id", None)
        ),
    }


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def load_tokenizer(tokenizer_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; install the gemma-server or "
            "open-webui-bench extra"
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_path)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def _exact_source_prefix(
    prefix: str,
    source_text: str,
    suffix: str,
    target_tokens: int,
    count_tokens: Any,
) -> tuple[str, int, int]:
    base_count = count_tokens(prefix + suffix)
    if base_count > target_tokens:
        raise ValueError(
            f"fixed prompt text uses {base_count} tokens, above target "
            f"{target_tokens}"
        )
    if base_count == target_tokens:
        return prefix + suffix, base_count, 0
    if not source_text:
        raise ValueError("natural long-context source text is empty")
    if count_tokens(prefix + source_text + suffix) < target_tokens:
        raise ValueError(
            "natural long-context source is too short for the requested "
            f"{target_tokens} rendered tokens"
        )

    low = 0
    high = len(source_text)
    while low < high:
        middle = (low + high) // 2
        count = count_tokens(prefix + source_text[:middle] + suffix)
        if count < target_tokens:
            low = middle + 1
        else:
            high = middle

    exact_candidates: list[int] = []
    for radius in (64, 512, 4096):
        start = max(0, low - radius)
        stop = min(len(source_text), low + radius)
        exact_candidates = [
            character_count
            for character_count in range(start, stop + 1)
            if count_tokens(
                prefix + source_text[:character_count] + suffix
            )
            == target_tokens
        ]
        if exact_candidates:
            break
    if exact_candidates:
        boundary_candidates = [
            character_count
            for character_count in exact_candidates
            if character_count in {0, len(source_text)}
            or source_text[character_count - 1].isspace()
            or source_text[character_count].isspace()
        ]
        character_count = (
            boundary_candidates[-1]
            if boundary_candidates
            else exact_candidates[-1]
        )
        return (
            prefix + source_text[:character_count] + suffix,
            target_tokens,
            character_count,
        )
    raise RuntimeError(
        "could not truncate the natural source to exactly "
        f"{target_tokens} rendered tokens"
    )


def natural_text_quality(text: str) -> dict[str, Any]:
    words = re.findall(r"[\w]+(?:['’][\w]+)?", text.casefold())
    if not words:
        return {
            "word_count": 0,
            "unique_word_fraction": 0.0,
            "dominant_word_fraction": 1.0,
            "repeated_4gram_fraction": 1.0,
        }
    counts: dict[str, int] = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    ngrams = [tuple(words[index : index + 4]) for index in range(len(words) - 3)]
    repeated_4grams = len(ngrams) - len(set(ngrams))
    return {
        "word_count": len(words),
        "unique_word_fraction": len(counts) / len(words),
        "dominant_word_fraction": max(counts.values()) / len(words),
        "repeated_4gram_fraction": (
            repeated_4grams / len(ngrams) if ngrams else 0.0
        ),
    }


def _validate_natural_source(text: str) -> dict[str, Any]:
    quality = natural_text_quality(text)
    if quality["word_count"] < 1_000:
        raise ValueError("natural long-context source must contain at least 1,000 words")
    if quality["dominant_word_fraction"] > 0.12:
        raise ValueError("natural long-context source has a pathologically dominant word")
    if quality["repeated_4gram_fraction"] > 0.08:
        raise ValueError("natural long-context source has excessive repeated 4-grams")
    return quality


def load_hf_long_source(path: Path) -> tuple[str, dict[str, Any]]:
    source_file_sha256 = file_sha256(path)
    if source_file_sha256 != HF_LONG_SOURCE_FILE_SHA256:
        raise ValueError(
            f"unexpected Hugging Face source SHA-256 for {path}: "
            f"{source_file_sha256}"
        )
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read the pinned Hugging Face source parquet"
        ) from exc
    table = parquet.read_table(path, columns=["Chapter", "Text"])
    rows = table.to_pylist()
    if len(rows) != HF_LONG_SOURCE_ROWS:
        raise ValueError(
            f"expected {HF_LONG_SOURCE_ROWS} Hugging Face source rows, "
            f"found {len(rows)}"
        )
    if any(
        not isinstance(row.get("Chapter"), str)
        or not isinstance(row.get("Text"), str)
        or not row["Text"].strip()
        for row in rows
    ):
        raise ValueError("Hugging Face source rows have an unexpected schema")
    source_text = "\n\n".join(row["Text"].strip() for row in rows)
    source_text_sha256 = text_sha256(source_text)
    if source_text_sha256 != HF_LONG_SOURCE_TEXT_SHA256:
        raise ValueError(
            "normalized Hugging Face source text does not match its pinned "
            "SHA-256"
        )
    provenance = {
        "dataset_id": HF_LONG_SOURCE_DATASET_ID,
        "revision": HF_LONG_SOURCE_REVISION,
        "config": HF_LONG_SOURCE_CONFIG,
        "split": HF_LONG_SOURCE_SPLIT,
        "filename": HF_LONG_SOURCE_FILENAME,
        "file_sha256": source_file_sha256,
        "row_indices": {"start": 0, "end_inclusive": len(rows) - 1},
        "row_count": len(rows),
        "normalized_source_text_sha256": source_text_sha256,
        "license": "MIT",
        "upstream_rights": "Project Gutenberg public domain in the United States",
        "work": {
            "title": "Frankenstein; or, The Modern Prometheus",
            "author": "Mary Wollstonecraft Shelley",
        },
    }
    return source_text, provenance


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _exact_needle_rendered_index(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    needle_text: str,
    rendered_token_ids: Sequence[int],
) -> int | None:
    try:
        rendered_text = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError):
        return None
    if not isinstance(rendered_text, str):
        return None
    needle_character_index = rendered_text.find(needle_text)
    if needle_character_index < 0 or rendered_text.find(
        needle_text,
        needle_character_index + 1,
    ) >= 0:
        return None
    try:
        encoded = tokenizer(
            rendered_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None
    if isinstance(encoded, Mapping):
        encoded_ids = encoded.get("input_ids")
        offsets = encoded.get("offset_mapping")
    else:
        encoded_ids = getattr(encoded, "input_ids", None)
        offsets = getattr(encoded, "offset_mapping", None)
    try:
        normalized_ids = _normalize_token_ids(encoded_ids)
    except (TypeError, ValueError):
        return None
    if normalized_ids != list(rendered_token_ids):
        return None
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if (
        not isinstance(offsets, list)
        or len(offsets) != len(normalized_ids)
    ):
        return None
    for index, offset in enumerate(offsets):
        if (
            isinstance(offset, (list, tuple))
            and len(offset) == 2
            and type(offset[0]) is int
            and type(offset[1]) is int
            and offset[1] > offset[0]
            and offset[1] > needle_character_index
        ):
            return index
    return None


def _prompt_record(
    tokenizer: Any,
    *,
    prompt_id: str,
    label: str,
    content: str,
    validator: Mapping[str, Any],
) -> dict[str, Any]:
    messages = [{"role": "user", "content": content}]
    token_ids = render_chat_token_ids(tokenizer, messages)
    return {
        "prompt_id": prompt_id,
        "label": label,
        "content": content,
        "messages": messages,
        "content_sha256": text_sha256(content),
        "prompt_sha256": canonical_sha256(messages),
        "rendered_token_count": len(token_ids),
        "rendered_token_ids_sha256": canonical_sha256(token_ids),
        "validator": dict(validator),
    }


def build_long_prompt(
    tokenizer: Any,
    target_tokens: int,
    *,
    source_text: str,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if target_tokens < 320:
        raise ValueError("--long-rendered-tokens must be at least 320")
    if not isinstance(source_provenance, Mapping) or not source_provenance:
        raise ValueError("natural long-context source provenance is required")
    source_quality = _validate_natural_source(source_text)

    preamble = (
        "Long-context recall test using Mary Shelley's Frankenstein, sourced "
        "from a pinned Hugging Face public-domain corpus. Read the text "
        "carefully; one inserted record must be recalled after the excerpt.\n\n"
        "SOURCE START\n\n"
    )

    def rendered_count(content: str) -> int:
        return len(
            render_chat_token_ids(
                tokenizer,
                [{"role": "user", "content": content}],
            )
        )

    before_needle, _, needle_source_character = _exact_source_prefix(
        preamble,
        source_text,
        "\n\n",
        NEEDLE_TARGET_RENDERED_TOKEN,
        rendered_count,
    )
    needle_text = (
        "NEEDLE RECORD: codename BLUE-742; city Samarkand; checksum lantern."
    )
    prompt_suffix = (
        "\n\nSOURCE END\n\nReport the codename, city, and checksum from the "
        "NEEDLE RECORD in one concise sentence."
    )
    content, rendered_tokens, continuation_characters = _exact_source_prefix(
        before_needle + needle_text + "\n\n",
        source_text[needle_source_character:],
        prompt_suffix,
        target_tokens,
        rendered_count,
    )
    excerpt_character_count = needle_source_character + continuation_characters
    excerpt_text = source_text[:excerpt_character_count]
    prompt = _prompt_record(
        tokenizer,
        prompt_id="long-context-needle",
        label="12K Natural-Text Recall",
        content=content,
        validator={
            "kind": "contains_all",
            "case_sensitive": False,
            "expected_substrings": ["BLUE-742", "Samarkand", "lantern"],
        },
    )
    full_token_ids = render_chat_token_ids(tokenizer, prompt["messages"])
    prefix_token_ids = render_chat_token_ids(
        tokenizer,
        [{"role": "user", "content": before_needle}],
        add_generation_prompt=False,
    )
    approximate_needle_position = _common_prefix_length(
        prefix_token_ids,
        full_token_ids,
    )
    exact_needle_position = _exact_needle_rendered_index(
        tokenizer,
        prompt["messages"],
        needle_text,
        full_token_ids,
    )
    prompt["needle"] = {
        "target_rendered_token_index": NEEDLE_TARGET_RENDERED_TOKEN,
        "rendered_token_index": exact_needle_position,
        "rendered_token_index_approx": approximate_needle_position,
        "index_base": 0,
        "position_method": (
            "fast_tokenizer_offset_mapping"
            if exact_needle_position is not None
            else "common_prefix_before_needle"
        ),
        "facts": {
            "codename": "BLUE-742",
            "city": "Samarkand",
            "checksum": "lantern",
        },
    }
    prompt["source"] = {
        **dict(source_provenance),
        "excerpt": {
            "start_character": 0,
            "end_character_exclusive": excerpt_character_count,
            "character_count": excerpt_character_count,
            "sha256": text_sha256(excerpt_text),
            "needle_insertion_character": needle_source_character,
        },
        "quality": natural_text_quality(excerpt_text),
        "transformation": (
            "rows joined with two newlines; contiguous prefix truncated at a "
            "tokenizer-aligned boundary; synthetic needle inserted"
        ),
    }
    if rendered_tokens != target_tokens or prompt["rendered_token_count"] != target_tokens:
        raise AssertionError("long prompt did not reach the exact rendered-token target")
    measured_needle_position = (
        exact_needle_position
        if exact_needle_position is not None
        else approximate_needle_position
    )
    if abs(measured_needle_position - NEEDLE_TARGET_RENDERED_TOKEN) > 8:
        raise RuntimeError(
            "tokenizer placed the needle too far from rendered token 256: "
            f"observed index {measured_needle_position}"
        )
    return prompt


def build_classic_prompts(tokenizer: Any) -> list[dict[str, Any]]:
    specifications = (
        (
            "reasoning",
            "Reasoning",
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
            "than the ball. How much does the ball cost? Put the numeric answer "
            "first in the form `FINAL: USD X.XX`, then verify it in at most two "
            "sentences.",
            {
                "kind": "contains_all",
                "case_sensitive": False,
                "expected_substrings": ["FINAL: USD 0.05"],
            },
        ),
        (
            "code",
            "Code",
            "Write a compact Python function `is_palindrome(text)` that ignores "
            "case and non-alphanumeric characters. Return only a Python code "
            "block containing the function.",
            {
                "kind": "contains_all",
                "case_sensitive": True,
                "expected_substrings": ["def is_palindrome", "return"],
            },
        ),
        (
            "json",
            "JSON",
            "Return only this JSON object, with no Markdown fence or commentary: "
            '{"engine":"WKVM","slots":4,"status":"ready"}',
            {
                "kind": "json_equals",
                "expected_json": {
                    "engine": "WKVM",
                    "slots": 4,
                    "status": "ready",
                },
            },
        ),
        (
            "systems",
            "Systems",
            "In one sentence, explain why a bounded request queue helps an "
            "inference server remain stable. Include the exact terms "
            "`backpressure` and `overload`.",
            {
                "kind": "contains_all",
                "case_sensitive": False,
                "expected_substrings": ["backpressure", "overload"],
            },
        ),
    )
    return [
        _prompt_record(
            tokenizer,
            prompt_id=prompt_id,
            label=label,
            content=content,
            validator=validator,
        )
        for prompt_id, label, content, validator in specifications
    ]


def build_scenario(
    tokenizer: Any,
    *,
    long_source_text: str,
    long_source_provenance: Mapping[str, Any],
    long_rendered_tokens: int = 12_000,
    tokenizer_identity: str | None = None,
) -> dict[str, Any]:
    follow_up_content = (
        "Summarize your previous answer in one short sentence without adding "
        "new claims."
    )
    follow_up = {
        "prompt_id": "common-follow-up",
        "label": "Common Follow-up",
        "content": follow_up_content,
        "content_sha256": text_sha256(follow_up_content),
        "validator": {"kind": "non_empty", "min_characters": 1},
    }
    scenario: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SCENARIO_KIND,
        "deterministic": True,
        "tokenizer": tokenizer_fingerprint(
            tokenizer,
            identity=tokenizer_identity,
        ),
        "long_prompt": build_long_prompt(
            tokenizer,
            long_rendered_tokens,
            source_text=long_source_text,
            source_provenance=long_source_provenance,
        ),
        "concurrent_prompts": build_classic_prompts(tokenizer),
        "follow_up": follow_up,
        "capture_plan": {
            "offered_concurrency": OFFERED_CONCURRENCY,
            "classic_prompt_count": OFFERED_CONCURRENCY,
            "acts": ["long_prompt", "concurrency", "follow_up"],
        },
        "claim_scope": {
            "semantics": "routed_span_approximate",
            "comparison": None,
            "ten_x_proof": False,
        },
    }
    scenario["scenario_sha256"] = canonical_sha256(scenario)
    return scenario


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) and converted >= 0 else None


def _timing_seconds(result: Mapping[str, Any], name: str) -> float | None:
    timing = result.get("timing")
    containers = [timing, result] if isinstance(timing, Mapping) else [result]
    aliases = {
        "ttft": ("ttft_s", "ui_path_ttft_s", "time_to_first_token_s"),
        "e2e": ("e2e_s", "e2e_latency_s", "completion_latency_s"),
    }
    for container in containers:
        for key in aliases[name]:
            value = _number(container.get(key))
            if value is not None:
                return value
        value_ms = _number(container.get(f"{name}_ms"))
        if value_ms is not None:
            return value_ms / 1000.0
    return None


def _response_text(result: Mapping[str, Any]) -> str:
    for key in ("response_text", "output_text", "text", "response"):
        value = result.get(key)
        if isinstance(value, str):
            return value
    return ""


def _capture_error(result: Mapping[str, Any]) -> str | None:
    value = result.get("error")
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        return canonical_json(value)
    return str(value)


def _record_from_capture(
    result: Mapping[str, Any],
    *,
    prompt_id: str,
    label: str,
    phase: str,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "label": label,
        "phase": phase,
        "response_text": _response_text(result),
        "ttft_s": _timing_seconds(result, "ttft"),
        "e2e_s": _timing_seconds(result, "e2e"),
        "capture_error": _capture_error(result),
    }


def capture_records(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_requests = capture.get("requests")
    if isinstance(raw_requests, list):
        records: list[dict[str, Any]] = []
        for index, request in enumerate(raw_requests):
            if not isinstance(request, Mapping):
                raise ValueError(f"capture request {index} is not an object")
            prompt_id = request.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise ValueError(f"capture request {index} omitted prompt_id")
            label = request.get("label", prompt_id)
            phase = request.get("phase", "first_turn")
            records.append(
                _record_from_capture(
                    request,
                    prompt_id=prompt_id,
                    label=str(label),
                    phase=str(phase),
                )
            )
        return records

    acts = capture.get("acts")
    if not isinstance(acts, Mapping):
        raise ValueError("capture must contain an acts object or requests array")
    records = []
    long_result = acts.get("long_prompt")
    if isinstance(long_result, Mapping):
        records.append(
            _record_from_capture(
                long_result,
                prompt_id=str(long_result.get("prompt_id", "long-context-needle")),
                label=str(long_result.get("label", "Long Context Needle")),
                phase="long_prompt",
            )
        )

    concurrency = acts.get("concurrency")
    sessions = concurrency.get("sessions") if isinstance(concurrency, Mapping) else None
    if not isinstance(sessions, list):
        raise ValueError("capture acts.concurrency.sessions must be an array")
    for index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            raise ValueError(f"concurrency session {index} is not an object")
        prompt_id = str(session.get("prompt_id", ""))
        if not prompt_id:
            raise ValueError(f"concurrency session {index} omitted prompt_id")
        label = str(session.get("label", prompt_id))
        first_turn = session.get("first_turn")
        if not isinstance(first_turn, Mapping):
            raise ValueError(f"concurrency session {index} omitted first_turn")
        records.append(
            _record_from_capture(
                first_turn,
                prompt_id=prompt_id,
                label=label,
                phase="first_turn",
            )
        )
        follow_up = session.get("follow_up")
        if isinstance(follow_up, Mapping):
            records.append(
                _record_from_capture(
                    follow_up,
                    prompt_id=str(
                        follow_up.get("prompt_id", "common-follow-up")
                    ),
                    label=str(follow_up.get("label", "Common Follow-up")),
                    phase="follow_up",
                )
            )
    return records


def validate_response(
    response_text: str,
    validator: Mapping[str, Any],
) -> dict[str, Any]:
    kind = validator.get("kind")
    problems: list[str] = []
    if kind == "contains_all":
        case_sensitive = validator.get("case_sensitive") is True
        haystack = response_text if case_sensitive else response_text.casefold()
        expected = validator.get("expected_substrings")
        if not isinstance(expected, list) or not all(
            isinstance(item, str) and item for item in expected
        ):
            problems.append("validator has invalid expected_substrings")
        else:
            for item in expected:
                needle = item if case_sensitive else item.casefold()
                if needle not in haystack:
                    problems.append(f"missing expected substring: {item}")
    elif kind == "json_equals":
        try:
            parsed = json.loads(response_text.strip())
        except (json.JSONDecodeError, TypeError) as exc:
            problems.append(f"response is not strict JSON: {exc}")
        else:
            if parsed != validator.get("expected_json"):
                problems.append("JSON value does not match expected_json")
    elif kind == "non_empty":
        minimum = validator.get("min_characters", 1)
        if type(minimum) is not int or minimum < 1:
            problems.append("validator has invalid min_characters")
        elif len(response_text.strip()) < minimum:
            problems.append(f"response has fewer than {minimum} characters")
    else:
        problems.append(f"unsupported validator kind: {kind!r}")
    return {"passed": not problems, "problems": problems}


def _prompt_validators(scenario: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    validators: dict[str, Mapping[str, Any]] = {}
    long_prompt = scenario.get("long_prompt")
    if isinstance(long_prompt, Mapping):
        prompt_id = long_prompt.get("prompt_id")
        validator = long_prompt.get("validator")
        if isinstance(prompt_id, str) and isinstance(validator, Mapping):
            validators[prompt_id] = validator
    concurrent = scenario.get("concurrent_prompts")
    if isinstance(concurrent, list):
        for prompt in concurrent:
            if not isinstance(prompt, Mapping):
                continue
            prompt_id = prompt.get("prompt_id")
            validator = prompt.get("validator")
            if isinstance(prompt_id, str) and isinstance(validator, Mapping):
                validators[prompt_id] = validator
    follow_up = scenario.get("follow_up")
    if isinstance(follow_up, Mapping):
        prompt_id = follow_up.get("prompt_id")
        validator = follow_up.get("validator")
        if isinstance(prompt_id, str) and isinstance(validator, Mapping):
            validators[prompt_id] = validator
    return validators


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ttft = [record["ttft_s"] for record in records if record.get("ttft_s") is not None]
    e2e = [record["e2e_s"] for record in records if record.get("e2e_s") is not None]
    return {
        "request_count": len(records),
        "timed_ttft_count": len(ttft),
        "timed_e2e_count": len(e2e),
        "success_count": sum(record.get("success") is True for record in records),
        "error_count": sum(record.get("success") is not True for record in records),
        "output_tokens": sum(int(record.get("output_tokens", 0)) for record in records),
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p95_s": percentile(e2e, 0.95),
    }


def _optional_nonnegative_number(value: Any) -> int | float | None:
    if type(value) not in {int, float}:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _gpu_evidence(act: Any) -> dict[str, Any]:
    gpu = act.get("gpu") if isinstance(act, Mapping) else None
    if not isinstance(gpu, Mapping):
        gpu = {}
    raw_devices = gpu.get("devices")
    devices: list[dict[str, Any]] = []
    if isinstance(raw_devices, list):
        for raw_device in raw_devices:
            if not isinstance(raw_device, Mapping):
                continue
            devices.append(
                {
                    "index": _optional_nonnegative_int(raw_device.get("index")),
                    "name": (
                        raw_device.get("name")
                        if isinstance(raw_device.get("name"), str)
                        else None
                    ),
                    "total_mib": _optional_nonnegative_number(
                        raw_device.get("total_mib")
                    ),
                    "baseline_used_mib": _optional_nonnegative_number(
                        raw_device.get("baseline_used_mib")
                    ),
                    "peak_used_mib": _optional_nonnegative_number(
                        raw_device.get("peak_used_mib")
                    ),
                }
            )

    def whole_gpu_total(field: str) -> int | float | None:
        values = [device[field] for device in devices]
        if not values or any(value is None for value in values):
            return None
        return sum(values)

    baseline = whole_gpu_total("baseline_used_mib")
    peak = whole_gpu_total("peak_used_mib")
    return {
        "device_count": len(devices) if isinstance(raw_devices, list) else None,
        "sample_count": _optional_nonnegative_int(gpu.get("sample_count")),
        "whole_gpu_baseline_used_mib": baseline,
        "whole_gpu_peak_used_mib": peak,
        "whole_gpu_peak_increase_mib": (
            peak - baseline if baseline is not None and peak is not None else None
        ),
        "devices": devices,
        "error": gpu.get("error") if isinstance(gpu.get("error"), str) else None,
    }


def _probe_engine_metrics(probe: Any) -> Mapping[str, Any] | None:
    if not isinstance(probe, Mapping):
        return None
    metrics = probe.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    values = metrics.get("values")
    if not isinstance(values, Mapping):
        return None
    engine = values.get("engine")
    return engine if isinstance(engine, Mapping) else None


def _provider_high_water(provider: Any) -> dict[str, int | None]:
    if not isinstance(provider, Mapping):
        provider = {}
    engines = [
        engine
        for key in ("before", "after_first_turn", "after")
        if (engine := _probe_engine_metrics(provider.get(key))) is not None
    ]

    def maximum(field: str) -> int | None:
        values = [
            value
            for engine in engines
            if (value := _optional_nonnegative_int(engine.get(field))) is not None
        ]
        return max(values) if values else None

    return {
        "max_running": maximum("max_running"),
        "max_runnable_rows": maximum("max_runnable_rows"),
    }


def _provider_request_counts(delta: Any) -> dict[str, int | None]:
    server = delta.get("server") if isinstance(delta, Mapping) else None
    if not isinstance(server, Mapping):
        server = {}
    return {
        field: _optional_nonnegative_int(server.get(field))
        for field in (
            "total_requests",
            "total_errors",
            "total_cancelled",
            "total_timed_out",
        )
    }


def _follow_up_reuse(provider: Any) -> dict[str, int | None]:
    if not isinstance(provider, Mapping):
        provider = {}
    reuse = provider.get("follow_up_session_reuse_delta")
    if not isinstance(reuse, Mapping):
        reuse = {}
    follow_up_delta = provider.get("follow_up_delta")
    follow_up_engine = (
        follow_up_delta.get("engine")
        if isinstance(follow_up_delta, Mapping)
        else None
    )
    if not isinstance(follow_up_engine, Mapping):
        follow_up_engine = {}

    def reuse_value(field: str) -> int | None:
        primary = _optional_nonnegative_int(reuse.get(field))
        if primary is not None:
            return primary
        return _optional_nonnegative_int(follow_up_engine.get(field))

    return {
        "session_reuse_hits": reuse_value("session_reuse_hits"),
        "sessions_opened": _optional_nonnegative_int(
            follow_up_engine.get("sessions_opened")
        ),
        "prefix_tokens_reused": reuse_value("prefix_tokens_reused"),
    }


def _launch_semantics(
    capture: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, str | None]:
    provenance = capture.get("provenance")
    launch = provenance.get("launch") if isinstance(provenance, Mapping) else None
    capture_launch = capture.get("launch")
    claim_scope = scenario.get("claim_scope")
    candidates = (
        (
            launch.get("semantic_caveat") if isinstance(launch, Mapping) else None,
            "capture.provenance.launch.semantic_caveat",
        ),
        (
            launch.get("semantics") if isinstance(launch, Mapping) else None,
            "capture.provenance.launch.semantics",
        ),
        (
            capture_launch.get("semantic_caveat")
            if isinstance(capture_launch, Mapping)
            else None,
            "capture.launch.semantic_caveat",
        ),
        (
            capture_launch.get("semantics")
            if isinstance(capture_launch, Mapping)
            else None,
            "capture.launch.semantics",
        ),
        (
            claim_scope.get("semantics")
            if isinstance(claim_scope, Mapping)
            else None,
            "scenario.claim_scope.semantics",
        ),
    )
    for value, source in candidates:
        if isinstance(value, str) and value.strip():
            return {"value": value, "source": source}
    return {"value": None, "source": None}


def _provider_engine_config(acts: Mapping[str, Any]) -> dict[str, Any]:
    config_fields = (
        "persistent_padded_decode",
        "persistent_padded_decode_steps",
        "persistent_padded_decode_cuda_graph",
        "use_native_gemma_forward",
        "native_gemma_attention_backend",
        "native_gemma_projection_backend",
        "native_gemma_weight_backend",
        "native_gemma_checkpoint_loader",
    )
    candidates = (
        ("concurrency", "after"),
        ("concurrency", "after_first_turn"),
        ("long_prompt", "after"),
        ("concurrency", "before"),
        ("long_prompt", "before"),
    )
    for act_name, probe_name in candidates:
        act = acts.get(act_name)
        provider = act.get("provider") if isinstance(act, Mapping) else None
        probe = provider.get(probe_name) if isinstance(provider, Mapping) else None
        engine = _probe_engine_metrics(probe)
        if engine is None:
            continue
        values = {field: engine[field] for field in config_fields if field in engine}
        if values:
            return {
                "values": values,
                "source": (
                    f"capture.acts.{act_name}.provider.{probe_name}.metrics.values.engine"
                ),
            }
    return {"values": {}, "source": None}


def _capture_telemetry(
    capture: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    acts = capture.get("acts")
    if not isinstance(acts, Mapping):
        acts = {}
    long_prompt = acts.get("long_prompt")
    concurrency = acts.get("concurrency")
    long_provider = (
        long_prompt.get("provider") if isinstance(long_prompt, Mapping) else None
    )
    concurrency_provider = (
        concurrency.get("provider") if isinstance(concurrency, Mapping) else None
    )
    if not isinstance(long_provider, Mapping):
        long_provider = {}
    if not isinstance(concurrency_provider, Mapping):
        concurrency_provider = {}
    capture_summary = capture.get("summary")
    if not isinstance(capture_summary, Mapping):
        capture_summary = {}
    return {
        "long_prompt": {
            "gpu": _gpu_evidence(long_prompt),
            "provider": {
                "high_water": _provider_high_water(long_provider),
                "request_counts": _provider_request_counts(
                    long_provider.get("delta")
                ),
            },
        },
        "concurrency": {
            "gpu": _gpu_evidence(concurrency),
            "provider": {
                "high_water": _provider_high_water(concurrency_provider),
                "request_counts": _provider_request_counts(
                    concurrency_provider.get("delta")
                ),
                "first_turn_request_counts": _provider_request_counts(
                    concurrency_provider.get("first_turn_delta")
                ),
                "follow_up_request_counts": _provider_request_counts(
                    concurrency_provider.get("follow_up_delta")
                ),
                "follow_up_reuse": _follow_up_reuse(concurrency_provider),
            },
        },
        "capture": {
            "capture_errors": _optional_nonnegative_int(
                capture_summary.get("capture_errors")
            ),
            "probe_errors": _optional_nonnegative_int(
                capture_summary.get("probe_errors")
            ),
        },
        "launch": {
            "semantics": _launch_semantics(capture, scenario),
            "provider_engine_config": _provider_engine_config(acts),
        },
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _tokenizer_match(
    scenario_fingerprint: Mapping[str, Any],
    report_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    fields = ("class", "vocab_size", "chat_template_sha256", "bos_token_id", "eos_token_id")
    comparisons = {
        field: scenario_fingerprint.get(field) == report_fingerprint.get(field)
        for field in fields
        if scenario_fingerprint.get(field) is not None
    }
    return {"matched": all(comparisons.values()), "fields": comparisons}


def build_report(
    capture: Mapping[str, Any],
    scenario: Mapping[str, Any],
    tokenizer: Any,
    *,
    capture_sha256: str | None = None,
    scenario_file_sha256: str | None = None,
) -> dict[str, Any]:
    if scenario.get("kind") != SCENARIO_KIND:
        raise ValueError(f"unsupported scenario kind: {scenario.get('kind')!r}")
    declared_scenario_sha = scenario.get("scenario_sha256")
    scenario_without_hash = dict(scenario)
    scenario_without_hash.pop("scenario_sha256", None)
    computed_scenario_sha = canonical_sha256(scenario_without_hash)
    if declared_scenario_sha != computed_scenario_sha:
        raise ValueError("scenario_sha256 does not match scenario contents")

    validators = _prompt_validators(scenario)
    records = capture_records(capture)
    expected_classic_ids = {
        prompt["prompt_id"]
        for prompt in scenario.get("concurrent_prompts", [])
        if isinstance(prompt, Mapping) and isinstance(prompt.get("prompt_id"), str)
    }
    errors: list[str] = []
    enriched: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt_id = record["prompt_id"]
        validator = validators.get(prompt_id)
        validation = (
            validate_response(record["response_text"], validator)
            if validator is not None
            else {"passed": False, "problems": ["no scenario validator for prompt"]}
        )
        record_errors: list[str] = []
        if record["capture_error"]:
            record_errors.append(record["capture_error"])
        if not record["response_text"].strip():
            record_errors.append("empty response text")
        if record["ttft_s"] is None:
            record_errors.append("missing valid TTFT")
        if record["e2e_s"] is None:
            record_errors.append("missing valid E2E latency")
        if not validation["passed"]:
            record_errors.extend(validation["problems"])
        output_tokens = len(encode_text(tokenizer, record["response_text"]))
        enriched_record = {
            **record,
            "response_sha256": text_sha256(record["response_text"]),
            "response_characters": len(record["response_text"]),
            "output_tokens": output_tokens,
            "validation": validation,
            "success": not record_errors,
            "errors": record_errors,
        }
        enriched.append(enriched_record)
        errors.extend(
            f"record {index} ({prompt_id}): {problem}" for problem in record_errors
        )

    long_records = [record for record in enriched if record["phase"] == "long_prompt"]
    first_turn_records = [record for record in enriched if record["phase"] == "first_turn"]
    follow_up_records = [record for record in enriched if record["phase"] == "follow_up"]
    captured_classic_ids = {record["prompt_id"] for record in first_turn_records}
    missing_classic = sorted(expected_classic_ids - captured_classic_ids)
    extra_classic = sorted(captured_classic_ids - expected_classic_ids)
    if len(long_records) != 1:
        errors.append(f"expected one long-prompt result, found {len(long_records)}")
    if len(first_turn_records) != OFFERED_CONCURRENCY:
        errors.append(
            "expected four concurrent first-turn results, found "
            f"{len(first_turn_records)}"
        )
    if missing_classic:
        errors.append(f"missing classic prompts: {', '.join(missing_classic)}")
    if extra_classic:
        errors.append(f"unexpected classic prompts: {', '.join(extra_classic)}")

    capture_errors = capture.get("errors")
    if isinstance(capture_errors, list):
        errors.extend(f"capture: {value}" for value in capture_errors if value)

    scenario_tokenizer = scenario.get("tokenizer")
    if not isinstance(scenario_tokenizer, Mapping):
        scenario_tokenizer = {}
    report_tokenizer = tokenizer_fingerprint(tokenizer)
    tokenizer_match = _tokenizer_match(scenario_tokenizer, report_tokenizer)
    if not tokenizer_match["matched"]:
        errors.append("report tokenizer fingerprint does not match scenario")

    concurrent_metrics = _metrics(first_turn_records)
    telemetry = _capture_telemetry(capture, scenario)
    summary = {
        **concurrent_metrics,
        "offered_concurrency": OFFERED_CONCURRENCY,
        "all_request_count": len(enriched),
        "all_success_count": sum(record["success"] for record in enriched),
        "all_error_count": sum(not record["success"] for record in enriched),
        "total_output_tokens": sum(record["output_tokens"] for record in enriched),
        "capture_error_count": telemetry["capture"]["capture_errors"],
        "probe_error_count": telemetry["capture"]["probe_errors"],
        "overall_passed": not errors,
    }
    capture_scenario = capture.get("scenario")
    capture_scenario_sha = (
        capture_scenario.get("sha256")
        if isinstance(capture_scenario, Mapping)
        else None
    )
    scenario_capture_match = (
        capture_scenario_sha in {scenario_file_sha256, declared_scenario_sha}
        if capture_scenario_sha is not None
        else None
    )
    if scenario_capture_match is False:
        errors.append("capture scenario SHA-256 does not match the supplied scenario")
        summary["overall_passed"] = False
    capture_provenance = capture.get("provenance")
    if not isinstance(capture_provenance, Mapping):
        capture_provenance = {}
    capture_browser = capture.get("browser")
    if capture_browser is None:
        capture_browser = capture_provenance.get("browser")
    long_prompt = scenario.get("long_prompt")
    long_prompt_source = (
        long_prompt.get("source")
        if isinstance(long_prompt, Mapping)
        else None
    )
    if not isinstance(long_prompt_source, Mapping):
        long_prompt_source = {}

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if summary["overall_passed"] else "failed",
        "summary": summary,
        "acts": {
            "long_prompt": _metrics(long_records),
            "concurrency_first_turn": {
                **concurrent_metrics,
                "offered_concurrency": OFFERED_CONCURRENCY,
            },
            "concurrency_follow_up": {
                **_metrics(follow_up_records),
                "offered_concurrency": OFFERED_CONCURRENCY,
            },
            "all_requests": _metrics(enriched),
        },
        "telemetry": telemetry,
        "records": enriched,
        "errors": errors,
        "caveats": list(CAVEATS),
        "provenance": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scenario": {
                "declared_sha256": declared_scenario_sha,
                "file_sha256": scenario_file_sha256,
                "capture_declared_sha256": capture_scenario_sha,
                "capture_match": scenario_capture_match,
            },
            "capture": {
                "file_sha256": capture_sha256,
                "kind": capture.get("kind"),
                "schema_version": capture.get("schema_version"),
                "captured_at": capture.get("captured_at"),
                "browser": capture_browser,
            },
            "tokenizer": {
                "scenario": dict(scenario_tokenizer),
                "report": report_tokenizer,
                "match": tokenizer_match,
            },
            "long_prompt_source": dict(long_prompt_source),
            "measurement": {
                "ttft": "browser submit to first observed streamed token",
                "e2e": "browser submit to observed completion",
                "output_tokens": "response text encoded without special tokens",
                "offered_concurrency": OFFERED_CONCURRENCY,
            },
        },
    }


def _format_seconds(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f} s"


def _format_count(value: Any) -> str:
    return str(value) if type(value) is int and value >= 0 else "n/a"


def _format_mib(value: Any) -> str:
    if type(value) not in {int, float} or not math.isfinite(value):
        return "n/a"
    return f"{value:,.0f} MiB"


def _format_config_value(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in {int, float} and math.isfinite(value):
        return str(value)
    if isinstance(value, str):
        return value
    return canonical_json(value)


def report_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    acts = report["acts"]
    telemetry = report.get("telemetry")
    if not isinstance(telemetry, Mapping):
        telemetry = {}
    long_telemetry = telemetry.get("long_prompt")
    concurrency_telemetry = telemetry.get("concurrency")
    if not isinstance(long_telemetry, Mapping):
        long_telemetry = {}
    if not isinstance(concurrency_telemetry, Mapping):
        concurrency_telemetry = {}
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    long_prompt_source = provenance.get("long_prompt_source")
    if not isinstance(long_prompt_source, Mapping):
        long_prompt_source = {}
    rows = (
        ("Long context", acts["long_prompt"], "1"),
        ("Classic first turn", acts["concurrency_first_turn"], "4"),
        ("Common follow-up", acts["concurrency_follow_up"], "4"),
    )
    lines = [
        "# WKVM Open WebUI Live Demo Report",
        "",
        f"**Status:** {str(report['status']).upper()}",
        "",
        "**Offered UI concurrency:** 4 chats",
        "",
        "| Act | Offered concurrency | Success | Output tokens | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics, concurrency in rows:
        lines.append(
            f"| {label} | {concurrency} | {metrics['success_count']}/{metrics['request_count']} "
            f"| {metrics['output_tokens']} | {_format_seconds(metrics['ttft_p50_s'])} "
            f"| {_format_seconds(metrics['ttft_p95_s'])} "
            f"| {_format_seconds(metrics['e2e_p50_s'])} "
            f"| {_format_seconds(metrics['e2e_p95_s'])} |"
        )
    lines.extend(
        [
            "",
            "The four classic first turns are submitted as one synchronized UI cohort. "
            f"Their browser-observed TTFT is p50 {_format_seconds(summary['ttft_p50_s'])} "
            f"and p95 {_format_seconds(summary['ttft_p95_s'])}; E2E is p50 "
            f"{_format_seconds(summary['e2e_p50_s'])} and p95 "
            f"{_format_seconds(summary['e2e_p95_s'])}.",
            "",
            "## Runtime Evidence",
            "",
            "Provider maxima are observed lifetime high-water gauges from the "
            "act's provider probe snapshots; request fields are counter deltas.",
            "",
            "| Act | Whole-GPU baseline | Whole-GPU peak | Provider requests | Errors | Cancelled | Timed out | Max running | Max runnable rows |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, act_telemetry in (
        ("Long context", long_telemetry),
        ("Concurrency", concurrency_telemetry),
    ):
        gpu = act_telemetry.get("gpu")
        provider = act_telemetry.get("provider")
        if not isinstance(gpu, Mapping):
            gpu = {}
        if not isinstance(provider, Mapping):
            provider = {}
        counts = provider.get("request_counts")
        high_water = provider.get("high_water")
        if not isinstance(counts, Mapping):
            counts = {}
        if not isinstance(high_water, Mapping):
            high_water = {}
        lines.append(
            f"| {label} | {_format_mib(gpu.get('whole_gpu_baseline_used_mib'))} "
            f"| {_format_mib(gpu.get('whole_gpu_peak_used_mib'))} "
            f"| {_format_count(counts.get('total_requests'))} "
            f"| {_format_count(counts.get('total_errors'))} "
            f"| {_format_count(counts.get('total_cancelled'))} "
            f"| {_format_count(counts.get('total_timed_out'))} "
            f"| {_format_count(high_water.get('max_running'))} "
            f"| {_format_count(high_water.get('max_runnable_rows'))} |"
        )

    concurrency_provider = concurrency_telemetry.get("provider")
    if not isinstance(concurrency_provider, Mapping):
        concurrency_provider = {}
    lines.extend(
        [
            "",
            "| Concurrency provider phase | Requests | Errors | Cancelled | Timed out |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, field in (
        ("Classic first turn", "first_turn_request_counts"),
        ("Common follow-up", "follow_up_request_counts"),
    ):
        counts = concurrency_provider.get(field)
        if not isinstance(counts, Mapping):
            counts = {}
        lines.append(
            f"| {label} | {_format_count(counts.get('total_requests'))} "
            f"| {_format_count(counts.get('total_errors'))} "
            f"| {_format_count(counts.get('total_cancelled'))} "
            f"| {_format_count(counts.get('total_timed_out'))} |"
        )
    reuse = concurrency_provider.get("follow_up_reuse")
    if not isinstance(reuse, Mapping):
        reuse = {}
    capture_telemetry = telemetry.get("capture")
    if not isinstance(capture_telemetry, Mapping):
        capture_telemetry = {}
    launch = telemetry.get("launch")
    if not isinstance(launch, Mapping):
        launch = {}
    lines.extend(
        [
            "",
            "**Follow-up reuse:** "
            f"{_format_count(reuse.get('session_reuse_hits'))} reuse hits; "
            f"{_format_count(reuse.get('sessions_opened'))} sessions opened; "
            f"{_format_count(reuse.get('prefix_tokens_reused'))} prefix tokens reused.",
            "",
            "**Capture health:** "
            f"{_format_count(capture_telemetry.get('capture_errors'))} capture errors; "
            f"{_format_count(capture_telemetry.get('probe_errors'))} probe errors.",
        ]
    )
    semantics = launch.get("semantics")
    if not isinstance(semantics, Mapping):
        semantics = {}
    semantic_value = semantics.get("value")
    semantic_source = semantics.get("source")
    if isinstance(semantic_value, str) and semantic_value:
        source_suffix = (
            f" (source: `{semantic_source}`)"
            if isinstance(semantic_source, str) and semantic_source
            else ""
        )
        lines.extend(
            [
                "",
                f"**Launch semantic declaration:** `{semantic_value}`{source_suffix}.",
            ]
        )
    provider_config = launch.get("provider_engine_config")
    if not isinstance(provider_config, Mapping):
        provider_config = {}
    config_values = provider_config.get("values")
    config_source = provider_config.get("source")
    if isinstance(config_values, Mapping) and config_values:
        rendered_config = "; ".join(
            f"`{key}={_format_config_value(value)}`"
            for key, value in config_values.items()
        )
        source_suffix = (
            f" Source: `{config_source}`."
            if isinstance(config_source, str) and config_source
            else ""
        )
        lines.extend(
            [
                "",
                f"**Observed provider engine config:** {rendered_config}.{source_suffix}",
            ]
        )
    lines.extend(
        [
            "",
            "## Long-Context Source",
            "",
            "The 12,000-token lane uses a contiguous natural-text excerpt, "
            "not repeated filler.",
            "",
            f"- Hugging Face dataset: `{long_prompt_source.get('dataset_id', 'n/a')}`",
            f"- Revision: `{long_prompt_source.get('revision', 'n/a')}`",
            f"- License: `{long_prompt_source.get('license', 'n/a')}`",
            f"- Source text SHA-256: "
            f"`{long_prompt_source.get('normalized_source_text_sha256', 'n/a')}`",
            "",
            "## Validation",
            "",
            "| Prompt | Phase | Result | Output tokens |",
            "|---|---|---:|---:|",
        ]
    )
    for record in report["records"]:
        result = "PASS" if record["success"] else "FAIL"
        lines.append(
            f"| {record['label']} | {record['phase']} | {result} | "
            f"{record['output_tokens']} |"
        )
    lines.extend(["", "## Caveats", ""])
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def _build_command(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer_path)
    identity = _safe_tokenizer_identity(args.tokenizer_path, tokenizer)
    source_text, source_provenance = load_hf_long_source(
        args.long_source_parquet
    )
    scenario = build_scenario(
        tokenizer,
        long_source_text=source_text,
        long_source_provenance=source_provenance,
        long_rendered_tokens=args.long_rendered_tokens,
        tokenizer_identity=identity,
    )
    atomic_write_json(args.json, scenario)
    print(f"wrote deterministic scenario: {args.json}")
    print(
        "long rendered tokens: "
        f"{scenario['long_prompt']['rendered_token_count']}"
    )
    print(
        "long source: "
        f"{source_provenance['dataset_id']}@{source_provenance['revision']}"
    )
    return 0


def _report_command(args: argparse.Namespace) -> int:
    capture_path = args.capture_json
    scenario_path = args.scenario
    capture = _load_json_object(capture_path, "capture")
    scenario = _load_json_object(scenario_path, "scenario")
    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_value = scenario.get("tokenizer")
        if isinstance(tokenizer_value, Mapping):
            identity = tokenizer_value.get("identity")
            if isinstance(identity, str) and identity:
                tokenizer_path = identity
    if tokenizer_path is None:
        raise ValueError(
            "scenario does not contain a loadable tokenizer identity; pass "
            "--tokenizer-path"
        )
    tokenizer = load_tokenizer(tokenizer_path)
    report = build_report(
        capture,
        scenario,
        tokenizer,
        capture_sha256=file_sha256(capture_path),
        scenario_file_sha256=file_sha256(scenario_path),
    )
    atomic_write_json(args.json, report)
    atomic_write_text(args.markdown, report_markdown(report))
    print(f"wrote report JSON: {args.json}")
    print(f"wrote report Markdown: {args.markdown}")
    return 0 if report["status"] == "passed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build deterministic demo prompts")
    build.add_argument("--tokenizer-path", required=True)
    build.add_argument("--long-source-parquet", type=Path, required=True)
    build.add_argument("--long-rendered-tokens", type=int, default=12_000)
    build.add_argument("--json", type=Path, required=True)
    build.set_defaults(function=_build_command)

    report = subparsers.add_parser("report", help="validate and summarize a capture")
    report.add_argument("capture_json", type=Path)
    report.add_argument("--scenario", type=Path, required=True)
    report.add_argument("--tokenizer-path")
    report.add_argument("--json", type=Path, required=True)
    report.add_argument("--markdown", type=Path, required=True)
    report.set_defaults(function=_report_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.function(args))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
