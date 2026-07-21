#!/usr/bin/env python
"""Provider-direct multi-turn Gemma HTTP benchmark.

The benchmark drives ``B`` stable session identities through synchronized turn
barriers. WKVM uses its token-native streaming session contract, while vLLM
and SGLang use streaming OpenAI completions with cumulative token-ID prompts.
Stock SGLang can instead use its native ``/generate`` token-ID stream as an
autonomous trace source. Shared-history traces are loaded and validated by
``gemma_multiturn_bench`` so the HTTP and direct-engine artifacts use the same
deterministic workload.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import threading
import time
from typing import Any, Sequence
import urllib.error
import urllib.parse
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bench_prompt_utils import generated_output_fingerprint
from gemma_multiturn_bench import (
    PROMPT_TOKEN_SOURCE,
    SharedHistoryTrace,
    _append_outputs,
    _print_turn,
    _turn_prompts_and_deltas,
    atomic_write_json,
    build_shared_history_trace,
    build_workload,
    git_commit,
    git_tree_state,
    load_shared_history_trace,
    request_order_indices,
    round_or_none,
    session_id,
    shared_history_trace_metadata,
    shared_history_trace_payload,
    summarize_run,
    summarize_turn,
    workload_fingerprints,
)
from wkvm_serving_bench import (
    WholeGpuMemoryMonitor,
    build_provenance,
    collect_gpu_provenance,
    parse_json_object,
    request_error_body,
    validate_extra_body,
)


SCHEMA = "wkvm.gemma_multiturn_http_bench.v1"
TEACHER_FORCED_TOKEN_IDS_ARG = "wkvm_teacher_forced_token_ids"
DEFAULT_WKVM_ENDPOINT = "/v1/stream"
DEFAULT_OPENAI_ENDPOINT = "/v1/completions"
DEFAULT_SGLANG_NATIVE_ENDPOINT = "/generate"
MAX_EXTRA_BODY_BYTES = 1_048_576


@dataclass(frozen=True)
class TeacherForcingHook:
    field_path: tuple[str, ...] | None
    encoding: str | None
    max_tokens: int
    processor: str | None = None

    @property
    def enabled(self) -> bool:
        return self.field_path is not None

    def apply(
        self,
        body: dict[str, Any],
        token_ids: Sequence[int] | None,
    ) -> dict[str, Any]:
        updated = copy.deepcopy(body)
        if not self.enabled:
            return updated
        if token_ids is None:
            raise ValueError("teacher-forcing hook requires trace token IDs")
        normalized = [int(token_id) for token_id in token_ids]
        if not normalized or len(normalized) > self.max_tokens:
            raise ValueError(
                "teacher-forced token count must be in "
                f"[1, {self.max_tokens}], received {len(normalized)}"
            )
        if self.encoding == "array":
            encoded: Any = normalized
        elif self.encoding == "json-string":
            encoded = json.dumps(normalized, separators=(",", ":"))
        else:
            raise ValueError(f"unsupported teacher-forcing encoding {self.encoding!r}")
        assert self.field_path is not None
        _set_nested_field(updated, self.field_path, encoded)
        if self.processor is not None:
            existing = updated.get("custom_logit_processor")
            if existing is not None and existing != self.processor:
                raise ValueError(
                    "teacher processor conflicts with extra-body "
                    "custom_logit_processor"
                )
            updated["custom_logit_processor"] = self.processor
        return updated


def _set_nested_field(
    target: dict[str, Any],
    field_path: Sequence[str],
    value: Any,
) -> None:
    if not field_path or any(not part for part in field_path):
        raise ValueError("teacher-forcing field path must not be empty")
    cursor = target
    for part in field_path[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ValueError(
                f"teacher-forcing field path conflicts at {part!r}"
            )
        cursor = current
    leaf = field_path[-1]
    if leaf in cursor and cursor[leaf] != value:
        raise ValueError(
            "teacher-forcing field conflicts with a static extra-body value"
        )
    cursor[leaf] = value


def _text_or_file(raw: str | None) -> str | None:
    if raw is None:
        return None
    if raw.startswith("@"):
        value = Path(raw[1:]).read_text(encoding="utf-8").strip()
    else:
        value = raw.strip()
    if not value:
        raise ValueError("text value must not be empty")
    return value


def _optional_gpu_device(raw: str) -> str | None:
    value = raw.strip()
    return None if value.lower() in {"none", "off", "disabled"} else value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_extra_body_metadata(extra_body: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(extra_body, sort_keys=True, separators=(",", ":")).encode()
    return {
        "present": bool(extra_body),
        "top_level_fields": sorted(extra_body),
        "json_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _launch_argv() -> tuple[list[str], list[str]]:
    argv = [sys.executable, *sys.argv]
    redactions: list[str] = []
    for flag in ("--teacher-forcing-processor", "--extra-body-json"):
        try:
            index = argv.index(flag)
        except ValueError:
            continue
        value_index = index + 1
        if value_index >= len(argv):
            continue
        raw = argv[value_index]
        if flag == "--teacher-forcing-processor" and raw.startswith("@"):
            continue
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        argv[value_index] = f"<redacted sha256={digest}>"
        redactions.append(flag)
    return argv, redactions


def _resolve_teacher_hook(
    args: argparse.Namespace,
    trace: SharedHistoryTrace | None,
) -> TeacherForcingHook:
    if trace is None or args.teacher_forcing_field == "none":
        return TeacherForcingHook(
            field_path=None,
            encoding=None,
            max_tokens=args.max_teacher_forced_tokens,
        )
    raw_path = args.teacher_forcing_field
    if raw_path == "auto":
        if args.engine == "wkvm":
            raw_path = "forced_output_ids"
        elif args.engine == "vllm":
            raw_path = f"vllm_xargs.{TEACHER_FORCED_TOKEN_IDS_ARG}"
        else:
            raw_path = f"custom_params.{TEACHER_FORCED_TOKEN_IDS_ARG}"
    raw_encoding = args.teacher_forcing_encoding
    if raw_encoding == "auto":
        raw_encoding = "json-string" if args.engine == "vllm" else "array"
    processor = _text_or_file(args.teacher_forcing_processor)
    if (
        args.engine == "sglang"
        and processor is None
        and "custom_logit_processor" not in args.extra_body
    ):
        raise ValueError(
            "SGLang teacher forcing requires --teacher-forcing-processor "
            "or extra-body custom_logit_processor"
        )
    return TeacherForcingHook(
        field_path=tuple(raw_path.split(".")),
        encoding=raw_encoding,
        max_tokens=args.max_teacher_forced_tokens,
        processor=processor,
    )


def build_request_body(
    args: argparse.Namespace,
    *,
    prompt: Sequence[int],
    delta: Sequence[int],
    turn_index: int,
    request_id: str,
    stable_session_id: str,
    expected_output: Sequence[int] | None,
    teacher_hook: TeacherForcingHook,
) -> dict[str, Any]:
    if args.engine == "wkvm":
        body: dict[str, Any] = {
            "max_tokens": args.output_tokens_per_turn,
            "req_id": request_id,
            "session_id": stable_session_id,
            "timeout_s": args.request_timeout_s,
        }
        input_field = "prompt_ids" if turn_index == 0 else "delta_ids"
        input_ids = prompt if turn_index == 0 else delta
        body[input_field] = [int(token_id) for token_id in input_ids]
    elif args.sglang_native_generate:
        body = {
            "input_ids": [int(token_id) for token_id in prompt],
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": args.output_tokens_per_turn,
                "ignore_eos": True,
            },
            "stream": True,
            "rid": request_id,
        }
    else:
        body = {
            "model": args.model,
            "prompt": [int(token_id) for token_id in prompt],
            "max_tokens": args.output_tokens_per_turn,
            "min_tokens": args.output_tokens_per_turn,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "return_token_ids": True,
            "add_special_tokens": False,
            "skip_special_tokens": False,
        }
        body["rid" if args.engine == "sglang" else "request_id"] = request_id
    for key, value in args.extra_body.items():
        if key in body:
            raise ValueError(f"extra-body field {key!r} overrides a controlled field")
        body[key] = copy.deepcopy(value)
    body = teacher_hook.apply(body, expected_output)
    encoded_size = len(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    if encoded_size > args.max_request_body_bytes:
        raise ValueError(
            f"request body is {encoded_size} bytes, exceeding "
            f"--max-request-body-bytes={args.max_request_body_bytes}"
        )
    return body


def _choice_token_ids(choice: dict[str, Any]) -> list[int] | None:
    for field in ("token_ids", "output_token_ids"):
        value = choice.get(field)
        if value is not None:
            if not isinstance(value, list):
                raise ValueError(f"OpenAI choice {field} must be a list")
            return [int(token_id) for token_id in value]
    return None


def _event_cached_tokens(event: dict[str, Any]) -> int | None:
    usage = event.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens") is not None:
            return int(details["cached_tokens"])
        if usage.get("cached_tokens") is not None:
            return int(usage["cached_tokens"])
    meta_info = event.get("meta_info")
    if isinstance(meta_info, dict) and meta_info.get("cached_tokens") is not None:
        return int(meta_info["cached_tokens"])
    return None


def _sglang_native_finish_reason(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        reason = value.get("type")
        if reason is None:
            raise ValueError("SGLang native finish_reason object has no type")
        return str(reason)
    raise ValueError("SGLang native finish_reason must be a string or object")


def _sglang_native_output_delta(
    current: Sequence[int],
    event_output_ids: Sequence[int],
    *,
    completion_tokens: int | None,
    stream_mode: str | None,
) -> tuple[list[int], str | None]:
    current_ids = [int(token_id) for token_id in current]
    event_ids = [int(token_id) for token_id in event_output_ids]
    if stream_mode == "incremental":
        return event_ids, stream_mode
    if stream_mode == "cumulative":
        if (
            len(event_ids) < len(current_ids)
            or event_ids[: len(current_ids)] != current_ids
        ):
            raise ValueError("SGLang cumulative output_ids diverged from prior output")
        return event_ids[len(current_ids) :], stream_mode
    if not current_ids:
        return event_ids, None

    event_extends_current = (
        len(event_ids) >= len(current_ids)
        and event_ids[: len(current_ids)] == current_ids
    )
    if completion_tokens is not None:
        if completion_tokens == len(current_ids) + len(event_ids):
            return event_ids, "incremental"
        if completion_tokens == len(event_ids) and event_extends_current:
            return event_ids[len(current_ids) :], "cumulative"
    if event_extends_current:
        return event_ids[len(current_ids) :], "cumulative"
    return event_ids, "incremental"


def _append_error(existing: str | None, message: str) -> str:
    return message if existing is None else f"{existing}; {message}"


class _SSEEventBuffer:
    def __init__(self) -> None:
        self._partial_line = bytearray()
        self._data_lines: list[bytes] = []

    def feed(self, chunk: bytes) -> list[dict[str, Any] | str]:
        if not isinstance(chunk, bytes):
            raise TypeError("SSE transport chunks must be bytes")
        self._partial_line.extend(chunk)
        events: list[dict[str, Any] | str] = []
        while True:
            newline = self._partial_line.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(self._partial_line[:newline])
            del self._partial_line[: newline + 1]
            line = raw_line.removesuffix(b"\r")
            if not line:
                event = self._finish_event()
                if event is not None:
                    events.append(event)
                continue
            if line.startswith(b"data:"):
                data = line.removeprefix(b"data:")
                if data.startswith(b" "):
                    data = data[1:]
                self._data_lines.append(data)
        return events

    def finish(self) -> None:
        if self._partial_line.strip():
            raise ValueError("incomplete SSE line at EOF")
        if self._data_lines:
            raise ValueError("incomplete SSE event at EOF")
        self._partial_line.clear()

    def _finish_event(self) -> dict[str, Any] | str | None:
        if not self._data_lines:
            return None
        raw = b"\n".join(self._data_lines)
        self._data_lines.clear()
        try:
            data = raw.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("invalid UTF-8 in SSE data event") from exc
        if data == "[DONE]":
            return data
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in complete SSE event: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise ValueError("SSE JSON data event must be an object")
        return event


def _stream_http_request(
    args: argparse.Namespace,
    *,
    body: dict[str, Any],
    request_id: str,
    expected_output: Sequence[int] | None,
    teacher_hook: TeacherForcingHook,
) -> dict[str, Any]:
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "x-request-id": request_id,
    }
    api_key = os.environ.get(args.api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        args.request_url,
        data=data,
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first_token_time: float | None = None
    last_token_time: float | None = None
    output_token_ids: list[int] = []
    response_token_ids_seen = False
    exact_stream_token_count = 0
    text_token_signal_count = 0
    usage_output_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    server_metrics: dict[str, Any] | None = None
    response_bytes = 0
    http_status: int | None = None
    saw_finish = False
    saw_done = args.engine == "wkvm"
    sglang_native_stream_mode: str | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(
            request,
            timeout=args.request_timeout_s + 5.0,
        ) as response:
            http_status = int(response.status)
            sse_buffer = _SSEEventBuffer()
            for chunk in response:
                response_bytes += len(chunk)
                if response_bytes > args.max_response_bytes:
                    raise ValueError(
                        "stream response exceeded "
                        f"--max-response-bytes={args.max_response_bytes}"
                    )
                stop_reading = False
                for event in sse_buffer.feed(chunk):
                    now = time.perf_counter()
                    if event == "[DONE]":
                        saw_done = True
                        stop_reading = True
                        break
                    if not isinstance(event, dict):
                        continue
                    if response_id is None and event.get("id") is not None:
                        response_id = str(event["id"])
                    event_error = event.get("error")
                    if event_error:
                        if isinstance(event_error, dict):
                            event_error = json.dumps(event_error, sort_keys=True)
                        error = _append_error(error, str(event_error))
                    event_cached = _event_cached_tokens(event)
                    if event_cached is not None:
                        cached_tokens = event_cached
                    usage = event.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        usage_output_tokens = int(usage["completion_tokens"])
                    meta_info = event.get("meta_info")
                    if isinstance(meta_info, dict):
                        if meta_info.get("completion_tokens") is not None:
                            usage_output_tokens = int(meta_info["completion_tokens"])
                        if response_id is None:
                            for field in ("id", "rid"):
                                if meta_info.get(field) is not None:
                                    response_id = str(meta_info[field])
                                    break
                    if args.engine == "wkvm":
                        event_type = event.get("type")
                        if event_type == "token":
                            if first_token_time is None:
                                first_token_time = now
                            last_token_time = now
                            output_token_ids.append(int(event["token"]))
                            exact_stream_token_count += 1
                            response_token_ids_seen = True
                        elif event_type == "finish":
                            saw_finish = True
                            finish_reason = event.get("finish_reason")
                            finish_error = event.get("error")
                            if finish_error:
                                error = _append_error(error, str(finish_error))
                            metrics = event.get("metrics")
                            if isinstance(metrics, dict):
                                server_metrics = metrics
                                for field in ("reused_prefix_tokens", "cached_tokens"):
                                    if metrics.get(field) is not None:
                                        cached_tokens = int(metrics[field])
                                        break
                        elif event_type == "error":
                            error = _append_error(
                                error,
                                str(event.get("error") or "stream error"),
                            )
                        continue
                    if args.sglang_native_generate:
                        native_finish_reason = (
                            None
                            if not isinstance(meta_info, dict)
                            else meta_info.get("finish_reason")
                        )
                        if native_finish_reason is not None:
                            try:
                                finish_reason = _sglang_native_finish_reason(
                                    native_finish_reason
                                )
                            except ValueError as exc:
                                error = _append_error(error, str(exc))
                            else:
                                saw_finish = True
                        native_output_ids = event.get("output_ids")
                        if native_output_ids is not None:
                            if not isinstance(native_output_ids, list):
                                error = _append_error(
                                    error,
                                    "SGLang native output_ids must be a list",
                                )
                                continue
                            try:
                                output_delta, sglang_native_stream_mode = (
                                    _sglang_native_output_delta(
                                        output_token_ids,
                                        native_output_ids,
                                        completion_tokens=usage_output_tokens,
                                        stream_mode=sglang_native_stream_mode,
                                    )
                                )
                            except (TypeError, ValueError) as exc:
                                error = _append_error(error, str(exc))
                                continue
                            response_token_ids_seen = True
                            if output_delta:
                                if first_token_time is None:
                                    first_token_time = now
                                last_token_time = now
                                output_token_ids.extend(output_delta)
                                exact_stream_token_count += len(output_delta)
                        continue
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        error = _append_error(error, "OpenAI choice must be an object")
                        continue
                    choice_finish = choice.get("finish_reason")
                    if choice_finish is not None:
                        saw_finish = True
                        finish_reason = str(choice_finish)
                    try:
                        delta_ids = _choice_token_ids(choice)
                    except (TypeError, ValueError) as exc:
                        error = _append_error(error, str(exc))
                        delta_ids = None
                    token_signal_count = 0
                    if delta_ids is not None:
                        response_token_ids_seen = True
                        output_token_ids.extend(delta_ids)
                        exact_stream_token_count += len(delta_ids)
                        token_signal_count = len(delta_ids)
                    else:
                        logprobs = choice.get("logprobs")
                        tokens = (
                            logprobs.get("tokens")
                            if isinstance(logprobs, dict)
                            else None
                        )
                        if isinstance(tokens, list) and tokens:
                            exact_stream_token_count += len(tokens)
                            token_signal_count = len(tokens)
                        elif choice.get("text"):
                            text_token_signal_count += 1
                            token_signal_count = 1
                    if token_signal_count and first_token_time is None:
                        first_token_time = now
                    if token_signal_count:
                        last_token_time = now
                if stop_reading:
                    break
            sse_buffer.finish()
    except urllib.error.HTTPError as exc:
        http_status = int(exc.code)
        error = _append_error(error, request_error_body(exc))
    except Exception as exc:
        error = _append_error(error, str(exc).splitlines()[0])
    finished = time.perf_counter()

    expected = None if expected_output is None else [int(x) for x in expected_output]
    output_source: str | None = None
    output_ids_observed = response_token_ids_seen
    observed_count = (
        usage_output_tokens
        if usage_output_tokens is not None
        else exact_stream_token_count
        if exact_stream_token_count > 0
        else None
    )
    if response_token_ids_seen:
        output_source = "response_token_ids"
        if usage_output_tokens is not None and usage_output_tokens != len(output_token_ids):
            error = _append_error(
                error,
                "response token IDs disagree with usage: "
                f"ids={len(output_token_ids)}, usage={usage_output_tokens}",
            )
    elif (
        teacher_hook.enabled
        and expected is not None
        and observed_count == len(expected)
    ):
        output_token_ids = list(expected)
        output_source = "teacher_trace_hook_contract"
    else:
        output_token_ids = []
        if observed_count is not None:
            output_source = "count_only_without_token_ids"

    if error is None and not saw_finish:
        error = "stream ended without a finish event"
    if error is None and not saw_done:
        error = "stream ended without [DONE]"
    if error is None and finish_reason != "length":
        error = f"unexpected finish_reason {finish_reason!r}"
    if error is None and observed_count != args.output_tokens_per_turn:
        error = (
            f"expected exactly {args.output_tokens_per_turn} output tokens, "
            f"received {observed_count}"
        )
    if error is None and not output_token_ids:
        error = "stream did not expose exact output token IDs"
    if error is None and expected is not None and output_token_ids != expected:
        mismatch = next(
            (
                index
                for index, (observed, target) in enumerate(
                    zip(output_token_ids, expected, strict=False)
                )
                if observed != target
            ),
            min(len(output_token_ids), len(expected)),
        )
        error = (
            "output diverged from shared trace at token index "
            f"{mismatch}"
        )
    teacher_verification = "not_requested"
    if expected is not None:
        if response_token_ids_seen:
            teacher_verification = (
                "exact_response_token_ids"
                if output_token_ids == expected
                else "response_token_mismatch"
            )
        elif output_source == "teacher_trace_hook_contract":
            teacher_verification = "declared_hook_plus_exact_usage_count"
        else:
            teacher_verification = "unverified"
    return {
        "req_id": request_id,
        "success": error is None,
        "error": error,
        "http_status": http_status,
        "response_id": response_id,
        "finish_reason": finish_reason,
        "output_token_ids": output_token_ids if output_token_ids else None,
        "observed_output_tokens": observed_count,
        "output_token_ids_observed": output_ids_observed,
        "output_token_ids_source": output_source,
        "teacher_forcing_verification": teacher_verification,
        "ttft_s": None if first_token_time is None else first_token_time - started,
        "e2e_latency_s": (
            None if last_token_time is None else last_token_time - started
        ),
        "request_completion_latency_s": finished - started,
        "cached_tokens": cached_tokens,
        "response_bytes": response_bytes,
        "server_metrics": server_metrics,
        "stream_exact_token_signals": exact_stream_token_count,
        "stream_text_token_signals": text_token_signal_count,
    }


def _failed_request_result(request_id: str, exc: BaseException) -> dict[str, Any]:
    return {
        "req_id": request_id,
        "success": False,
        "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        "http_status": None,
        "response_id": None,
        "finish_reason": None,
        "output_token_ids": None,
        "observed_output_tokens": None,
        "output_token_ids_observed": False,
        "output_token_ids_source": None,
        "teacher_forcing_verification": "request_failed",
        "ttft_s": None,
        "e2e_latency_s": None,
        "request_completion_latency_s": None,
        "cached_tokens": None,
        "response_bytes": 0,
        "server_metrics": None,
        "stream_exact_token_signals": 0,
        "stream_text_token_signals": 0,
    }


def run_http_turns(
    args: argparse.Namespace,
    workload: Any,
    trace: SharedHistoryTrace | None,
    teacher_hook: TeacherForcingHook,
) -> dict[str, Any]:
    stable_session_ids = [session_id(index) for index in range(args.sessions)]
    histories = [list(prompt) for prompt in workload.initial_prompts]
    turn_rows: list[dict[str, Any]] = []
    generated_output_turns: list[list[list[int]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as pool:
        for turn_index in range(args.turns):
            prompts, deltas = _turn_prompts_and_deltas(
                workload,
                histories,
                turn_index,
            )
            expected_outputs = (
                None if trace is None else trace.turn_outputs[turn_index]
            )
            order = request_order_indices(
                args.sessions,
                turn_index,
                args.request_order_policy,
                args.request_order_seed,
            )
            barrier = threading.Barrier(args.sessions)
            turn_started = time.perf_counter()
            future_to_index: dict[concurrent.futures.Future, int] = {}
            request_ids: list[str] = [""] * args.sessions

            def submit_request(logical_index: int) -> dict[str, Any]:
                barrier.wait(timeout=min(args.request_timeout_s, 30.0))
                request_id = request_ids[logical_index]
                body = build_request_body(
                    args,
                    prompt=prompts[logical_index],
                    delta=deltas[logical_index],
                    turn_index=turn_index,
                    request_id=request_id,
                    stable_session_id=stable_session_ids[logical_index],
                    expected_output=(
                        None
                        if expected_outputs is None
                        else expected_outputs[logical_index]
                    ),
                    teacher_hook=teacher_hook,
                )
                return _stream_http_request(
                    args,
                    body=body,
                    request_id=request_id,
                    expected_output=(
                        None
                        if expected_outputs is None
                        else expected_outputs[logical_index]
                    ),
                    teacher_hook=teacher_hook,
                )

            for logical_index in order:
                request_ids[logical_index] = (
                    f"{args.run_id}-turn-{turn_index:02d}-"
                    f"{stable_session_ids[logical_index]}"
                )
                future = pool.submit(submit_request, logical_index)
                future_to_index[future] = logical_index
            results: list[dict[str, Any] | None] = [None] * args.sessions
            for future in concurrent.futures.as_completed(future_to_index):
                logical_index = future_to_index[future]
                try:
                    results[logical_index] = future.result()
                except BaseException as exc:
                    results[logical_index] = _failed_request_result(
                        request_ids[logical_index],
                        exc,
                    )
            wall_s = time.perf_counter() - turn_started
            complete_results = [
                result
                if result is not None
                else _failed_request_result(
                    request_ids[index],
                    RuntimeError("request worker returned no result"),
                )
                for index, result in enumerate(results)
            ]
            outputs = [result["output_token_ids"] for result in complete_results]
            errors = [result["error"] for result in complete_results]
            row = summarize_turn(
                turn_index=turn_index,
                session_ids=stable_session_ids,
                prompts=prompts,
                deltas=deltas,
                outputs=outputs,
                expected_output_tokens=args.output_tokens_per_turn,
                new_input_tokens=(
                    [len(prompt) for prompt in prompts]
                    if turn_index == 0
                    else [len(delta) for delta in deltas]
                ),
                wall_s=wall_s,
                ttft_s=[result["ttft_s"] for result in complete_results],
                e2e_s=[result["e2e_latency_s"] for result in complete_results],
                cached_tokens=[
                    result["cached_tokens"] for result in complete_results
                ],
                errors=errors,
            )
            for request_row, result in zip(
                row["requests"],
                complete_results,
                strict=True,
            ):
                request_row.update(
                    {
                        "req_id": result["req_id"],
                        "http_status": result["http_status"],
                        "response_id": result["response_id"],
                        "finish_reason": result["finish_reason"],
                        "observed_output_tokens": result["observed_output_tokens"],
                        "output_token_ids_observed": result[
                            "output_token_ids_observed"
                        ],
                        "output_token_ids_source": result[
                            "output_token_ids_source"
                        ],
                        "teacher_forcing_verification": result[
                            "teacher_forcing_verification"
                        ],
                        "response_bytes": result["response_bytes"],
                        "request_completion_latency_s": round_or_none(
                            result["request_completion_latency_s"]
                        ),
                        "stream_exact_token_signals": result[
                            "stream_exact_token_signals"
                        ],
                        "stream_text_token_signals": result[
                            "stream_text_token_signals"
                        ],
                        "server_metrics": result["server_metrics"],
                    }
                )
            exact_outputs = [
                (stable_session_ids[index], list(output))
                for index, output in enumerate(outputs)
                if output is not None
            ]
            row.update(
                {
                    "api_mode": args.api_mode,
                    "endpoint": args.endpoint,
                    "reuse_kind": (
                        "wkvm_token_native_session"
                        if args.engine == "wkvm"
                        else f"{args.engine}_prefix_cache"
                    ),
                    "request_order_policy": args.request_order_policy,
                    "request_order": [stable_session_ids[index] for index in order],
                    "response_output_fingerprint": generated_output_fingerprint(
                        exact_outputs
                    ),
                    "response_output_fingerprint_complete": (
                        len(exact_outputs) == args.sessions
                    ),
                    "response_token_ids_observed_count": sum(
                        bool(result["output_token_ids_observed"])
                        for result in complete_results
                    ),
                    "server_session_reuse": _server_session_reuse_summary(
                        complete_results
                    ),
                    "teacher_forcing": {
                        "requested": teacher_hook.enabled,
                        "trace_sha256": None if trace is None else trace.trace_sha256,
                        "expected_output_fingerprint": (
                            None
                            if trace is None
                            else trace.output_fingerprints[turn_index]
                        ),
                        "exact_response_verification_count": sum(
                            result["teacher_forcing_verification"]
                            == "exact_response_token_ids"
                            for result in complete_results
                        ),
                        "hook_contract_verification_count": sum(
                            result["teacher_forcing_verification"]
                            == "declared_hook_plus_exact_usage_count"
                            for result in complete_results
                        ),
                    },
                }
            )
            turn_rows.append(row)
            _print_turn(args.engine, row)
            if all(output is not None for output in outputs):
                generated_output_turns.append(
                    [list(output or ()) for output in outputs]
                )
            canonical_outputs: Sequence[Sequence[int] | None]
            if expected_outputs is not None:
                canonical_outputs = expected_outputs
            else:
                canonical_outputs = outputs
                if any(output is None for output in canonical_outputs):
                    raise RuntimeError(
                        "autonomous HTTP run cannot continue without exact "
                        "output token IDs for every session"
                    )
            _append_outputs(histories, canonical_outputs)
    return {
        "turns": turn_rows,
        "generated_output_turns": generated_output_turns,
    }


def _history_trace_metadata(
    trace: SharedHistoryTrace | None,
    teacher_hook: TeacherForcingHook,
) -> dict[str, Any]:
    metadata = shared_history_trace_metadata(trace)
    if trace is None:
        return metadata
    metadata["teacher_forced"] = teacher_hook.enabled
    metadata["mode"] = (
        "shared_teacher_forced_http"
        if teacher_hook.enabled
        else "shared_trace_replay_unforced"
    )
    return metadata


def _server_session_reuse_summary(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [
        result["server_metrics"]
        for result in results
        if isinstance(result.get("server_metrics"), dict)
    ]
    return {
        "metrics_available_count": len(metrics),
        "reused_prefix_tokens_total": sum(
            int(metric.get("reused_prefix_tokens") or 0) for metric in metrics
        ),
        "computed_input_tokens_total": sum(
            int(metric.get("computed_input_tokens") or 0) for metric in metrics
        ),
        "session_reuse_hits": sum(
            int(
                metric.get("session_reuse_hit")
                or bool(metric.get("session_reused"))
            )
            for metric in metrics
        ),
        "session_reuse_misses": sum(
            int(
                metric.get("session_reuse_miss")
                or (
                    metric.get("session_input_mode") == "continuation_delta"
                    and not metric.get("session_reused")
                )
            )
            for metric in metrics
        ),
        "full_reprefill_turns": sum(
            int(metric.get("full_reprefill_turn") or 0) for metric in metrics
        ),
    }


def fetch_server_metrics(url: str, timeout_s: float) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=min(timeout_s, 60.0)) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError("server metrics response must be a JSON object")
        return payload, None
    except Exception as exc:
        return None, str(exc).splitlines()[0]


def _memory_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _apply_gpu_memory_contract(
    gpu_memory: dict[str, Any],
    *,
    prelaunch_baseline_used_mib: float | None,
    memory_ceiling_mib: float | None,
) -> dict[str, Any]:
    normalized = copy.deepcopy(gpu_memory)
    request_start_baseline = _memory_number(normalized.get("baseline_used_mib"))
    normalized["request_start_baseline_used_mib"] = request_start_baseline
    normalized["request_start_baseline_source"] = (
        normalized.get("source") if request_start_baseline is not None else None
    )
    normalized["request_start_baseline_scope"] = (
        "whole_device_request_start" if request_start_baseline is not None else None
    )
    normalized["prelaunch_baseline_used_mib"] = prelaunch_baseline_used_mib
    if prelaunch_baseline_used_mib is None:
        normalized["baseline_source"] = (
            normalized.get("source") if request_start_baseline is not None else None
        )
        normalized["baseline_scope"] = (
            "whole_device_request_start" if request_start_baseline is not None else None
        )
    else:
        observed_peak = _memory_number(normalized.get("peak_used_mib"))
        effective_peak = max(
            prelaunch_baseline_used_mib,
            observed_peak
            if observed_peak is not None
            else prelaunch_baseline_used_mib,
        )
        normalized["baseline_used_mib"] = prelaunch_baseline_used_mib
        normalized["baseline_source"] = "operator_supplied_prelaunch_nvidia_smi"
        normalized["baseline_scope"] = "whole_device_pre_server_launch"
        normalized["peak_used_mib"] = effective_peak
        normalized["peak_delta_mib"] = effective_peak - prelaunch_baseline_used_mib
    peak_used_mib = _memory_number(normalized.get("peak_used_mib"))
    normalized["memory_ceiling_mib"] = memory_ceiling_mib
    normalized["memory_ceiling_scope"] = (
        "whole_device_peak_used_mib" if memory_ceiling_mib is not None else None
    )
    normalized["within_memory_ceiling"] = (
        None
        if memory_ceiling_mib is None or peak_used_mib is None
        else peak_used_mib <= memory_ceiling_mib
    )
    return normalized


def _memory_ceiling_error(gpu_memory: dict[str, Any]) -> RuntimeError | None:
    if gpu_memory.get("within_memory_ceiling") is not False:
        return None
    return RuntimeError(
        "whole-device peak GPU memory "
        f"{gpu_memory.get('peak_used_mib')} MiB exceeds ceiling "
        f"{gpu_memory.get('memory_ceiling_mib')} MiB"
    )


def _augment_memory_provenance(
    provenance: dict[str, Any],
    gpu_memory: dict[str, Any],
) -> None:
    monitor = provenance.get("gpu_memory_monitor")
    if not isinstance(monitor, dict):
        monitor = {}
        provenance["gpu_memory_monitor"] = monitor
    for field in (
        "request_start_baseline_used_mib",
        "request_start_baseline_source",
        "request_start_baseline_scope",
        "prelaunch_baseline_used_mib",
        "baseline_used_mib",
        "baseline_source",
        "baseline_scope",
        "peak_used_mib",
        "peak_delta_mib",
        "memory_ceiling_mib",
        "memory_ceiling_scope",
        "within_memory_ceiling",
    ):
        monitor[field] = gpu_memory.get(field)


def build_payload(
    args: argparse.Namespace,
    workload: Any,
    result: dict[str, Any],
    gpu_memory: dict[str, Any],
    provenance: dict[str, Any],
    teacher_hook: TeacherForcingHook,
    trace: SharedHistoryTrace | None,
    *,
    emitted_trace: SharedHistoryTrace | None = None,
    fatal_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    processor_bytes = (
        None
        if teacher_hook.processor is None
        else teacher_hook.processor.encode("utf-8")
    )
    launch_argv, launch_redactions = _launch_argv()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "engine": args.engine,
        "engine_version": args.engine_version,
        "semantic_mode": args.semantics,
        "model": args.model,
        "prompt_token_source": PROMPT_TOKEN_SOURCE,
        "api": {
            "mode": args.api_mode,
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "request_url": args.request_url,
            "streaming": True,
            "api_key_env": args.api_key_env,
            "api_key_present": bool(os.environ.get(args.api_key_env)),
        },
        "benchmark_identity": {
            "campaign_id": args.campaign_id,
            "repeat_id": args.repeat_id,
            "run_id": args.run_id,
            "memory_ceiling_mib": args.memory_ceiling_mib,
            "artifact_role": (
                "http_trace_source"
                if emitted_trace is not None
                else "http_teacher_forced_replay"
                if trace is not None and teacher_hook.enabled
                else "http_trace_replay"
                if trace is not None
                else "http_engine_generated"
            ),
        },
        "history_trace": _history_trace_metadata(trace, teacher_hook),
        "workload": {
            "sessions": args.sessions,
            "turns": args.turns,
            "initial_context_tokens": args.initial_context_tokens,
            "turn_input_tokens": args.turn_input_tokens,
            "output_tokens_per_turn": args.output_tokens_per_turn,
            "required_model_len": args.required_model_len,
            "history_policy": (
                "wkvm_token_session_initial_prompt_then_deltas"
                if args.engine == "wkvm"
                else "cumulative_full_token_history"
            ),
            "synchronized_turn_barriers": True,
            "request_order_policy": args.request_order_policy,
            "request_order_seed": args.request_order_seed,
            "fingerprints": workload_fingerprints(workload),
        },
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "max_output_tokens_per_turn": args.output_tokens_per_turn,
        },
        "teacher_forcing_hook": {
            "enabled": teacher_hook.enabled,
            "field_path": (
                None
                if teacher_hook.field_path is None
                else ".".join(teacher_hook.field_path)
            ),
            "encoding": teacher_hook.encoding,
            "max_tokens": teacher_hook.max_tokens,
            "processor_present": processor_bytes is not None,
            "processor_bytes": None if processor_bytes is None else len(processor_bytes),
            "processor_sha256": (
                None
                if processor_bytes is None
                else hashlib.sha256(processor_bytes).hexdigest()
            ),
            "verification_caveat": (
                "When an incumbent omits response token IDs, exact output IDs "
                "are attributed to the declared bounded hook only after exact "
                "usage completion_tokens matches the trace length."
                if teacher_hook.enabled
                else None
            ),
        },
        "request_extra_body": _safe_extra_body_metadata(args.extra_body),
        "gpu_memory": gpu_memory,
        "provenance": provenance,
        "git_commit": git_commit(),
        "git_tree_state": git_tree_state(),
        "launch_command": shlex.join(launch_argv),
        "launch_config": {
            "argv": launch_argv,
            "redacted_flags": launch_redactions,
            "working_directory": os.getcwd(),
        },
        "turns": result.get("turns", []),
        "summary": summarize_run(result.get("turns", []), args.turns),
    }
    if emitted_trace is not None:
        payload["emitted_history_trace"] = shared_history_trace_metadata(
            emitted_trace
        )
    if fatal_error is not None:
        payload["fatal_error"] = fatal_error
    return payload


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "sessions",
        "turns",
        "initial_context_tokens",
        "turn_input_tokens",
        "output_tokens_per_turn",
        "synthetic_vocab_size",
        "max_teacher_forced_tokens",
        "max_request_body_bytes",
        "max_response_bytes",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    if args.synthetic_vocab_size < 16:
        raise ValueError("--synthetic-vocab-size must be >= 16")
    if args.output_tokens_per_turn > args.max_teacher_forced_tokens:
        raise ValueError(
            "--output-tokens-per-turn exceeds --max-teacher-forced-tokens"
        )
    if not math.isfinite(args.request_timeout_s) or args.request_timeout_s <= 0:
        raise ValueError("--request-timeout-s must be finite and > 0")
    if args.gpu_memory_sample_interval_s <= 0:
        raise ValueError("--gpu-memory-sample-interval-s must be > 0")
    if args.gpu_memory_baseline_used_mib is not None and (
        not math.isfinite(args.gpu_memory_baseline_used_mib)
        or args.gpu_memory_baseline_used_mib < 0
    ):
        raise ValueError(
            "--gpu-memory-baseline-used-mib must be finite and >= 0"
        )
    if args.memory_ceiling_mib is not None and (
        not math.isfinite(args.memory_ceiling_mib)
        or args.memory_ceiling_mib <= 0
    ):
        raise ValueError("--memory-ceiling-mib must be finite and > 0")
    if args.gpu_memory_device is None and (
        args.gpu_memory_baseline_used_mib is not None
        or args.memory_ceiling_mib is not None
    ):
        raise ValueError(
            "GPU memory baseline and ceiling options require --gpu-memory-device"
        )
    parsed = urllib.parse.urlsplit(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--base-url must not contain credentials")
    if args.sglang_native_generate and args.engine != "sglang":
        raise ValueError("--sglang-native-generate requires --engine sglang")
    if (
        args.sglang_native_generate
        and args.shared_history_trace_json is not None
    ):
        raise ValueError(
            "--sglang-native-generate does not support shared-history "
            "forced replay"
        )
    if args.engine != "wkvm" and not args.model:
        raise ValueError("--model is required for vLLM and SGLang")
    if args.campaign_id is None and args.repeat_id is not None:
        raise ValueError("--repeat-id requires --campaign-id")
    if args.campaign_id is not None and args.repeat_id is None:
        raise ValueError("--campaign-id requires --repeat-id")
    if args.run_id is None:
        args.run_id = str(uuid.uuid4())
    else:
        try:
            args.run_id = str(uuid.UUID(str(args.run_id)))
        except ValueError as exc:
            raise ValueError("--run-id must be a UUID") from exc
    validate_extra_body(args.extra_body)
    encoded_extra = json.dumps(args.extra_body, separators=(",", ":")).encode()
    if len(encoded_extra) > MAX_EXTRA_BODY_BYTES:
        raise ValueError(
            f"--extra-body-json exceeds {MAX_EXTRA_BODY_BYTES} bytes"
        )
    args.api_mode = (
        "wkvm_token_native_session"
        if args.engine == "wkvm"
        else "sglang_native_generate"
        if args.sglang_native_generate
        else "openai_completions"
    )
    if args.endpoint is None:
        args.endpoint = (
            DEFAULT_WKVM_ENDPOINT
            if args.engine == "wkvm"
            else DEFAULT_SGLANG_NATIVE_ENDPOINT
            if args.sglang_native_generate
            else DEFAULT_OPENAI_ENDPOINT
        )
    if not args.endpoint.startswith("/"):
        raise ValueError("--endpoint must start with '/'")
    args.base_url = args.base_url.rstrip("/")
    args.request_url = f"{args.base_url}{args.endpoint}"
    args.required_model_len = (
        args.initial_context_tokens
        + args.turns * args.output_tokens_per_turn
        + (args.turns - 1) * args.turn_input_tokens
    )
    args.semantics = args.semantics or (
        "routed_span_approximate" if args.engine == "wkvm" else "full_kv"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    workload = build_workload(
        sessions=args.sessions,
        turns=args.turns,
        initial_context_tokens=args.initial_context_tokens,
        turn_input_tokens=args.turn_input_tokens,
        vocab_size=args.synthetic_vocab_size,
    )
    trace = (
        None
        if args.shared_history_trace_json is None
        else load_shared_history_trace(
            args.shared_history_trace_json,
            workload,
            sessions=args.sessions,
            turns=args.turns,
            output_tokens_per_turn=args.output_tokens_per_turn,
            vocab_size=args.synthetic_vocab_size,
        )
    )
    teacher_hook = _resolve_teacher_hook(args, trace)
    monitor = (
        None
        if args.gpu_memory_device is None
        else WholeGpuMemoryMonitor(
            args.gpu_memory_device,
            args.gpu_memory_sample_interval_s,
        )
    )
    monitor_context = monitor if monitor is not None else contextlib.nullcontext()
    result: dict[str, Any] = {"turns": [], "generated_output_turns": []}
    pending_error: BaseException | None = None
    fatal_error: dict[str, Any] | None = None
    with monitor_context:
        try:
            result = run_http_turns(args, workload, trace, teacher_hook)
        except BaseException as exc:
            pending_error = exc
            fatal_error = {
                "type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
                "phase": "http_run",
            }
    gpu_memory = (
        monitor.result()
        if monitor is not None
        else {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "enabled": False,
            "device_selector": None,
            "sample_count": 0,
        }
    )
    gpu_memory = _apply_gpu_memory_contract(
        gpu_memory,
        prelaunch_baseline_used_mib=args.gpu_memory_baseline_used_mib,
        memory_ceiling_mib=args.memory_ceiling_mib,
    )
    ceiling_error = _memory_ceiling_error(gpu_memory)
    if pending_error is None and ceiling_error is not None:
        pending_error = ceiling_error
        fatal_error = {
            "type": type(ceiling_error).__name__,
            "message": str(ceiling_error),
            "phase": "gpu_memory_ceiling",
        }
    emitted_trace: SharedHistoryTrace | None = None
    if pending_error is None and args.write_shared_history_trace_json is not None:
        try:
            error_count = sum(
                int(row.get("error_count") or 0)
                for row in result.get("turns", [])
            )
            if error_count:
                raise RuntimeError(
                    "refusing to write an autonomous trace from a run with "
                    f"{error_count} request errors"
                )
            if len(result.get("generated_output_turns", [])) != args.turns:
                raise RuntimeError(
                    "refusing to write an incomplete autonomous trace"
                )
            trace_path = Path(args.write_shared_history_trace_json)
            emitted_trace = build_shared_history_trace(
                workload,
                result["generated_output_turns"],
                sessions=args.sessions,
                turns=args.turns,
                output_tokens_per_turn=args.output_tokens_per_turn,
                vocab_size=args.synthetic_vocab_size,
                source_path=str(trace_path),
            )
            atomic_write_json(
                trace_path,
                shared_history_trace_payload(
                    emitted_trace,
                    source={
                        "engine": args.engine,
                        "engine_version": args.engine_version,
                        "model": args.model,
                        "benchmark_artifact": args.json,
                        "campaign_id": args.campaign_id,
                        "repeat_id": args.repeat_id,
                        "run_id": args.run_id,
                        "memory_ceiling_mib": args.memory_ceiling_mib,
                        "git_commit": git_commit(),
                        "git_tree_state": git_tree_state(),
                        "api_mode": args.api_mode,
                    },
                ),
            )
            print(f"WROTE {trace_path}")
        except BaseException as exc:
            pending_error = exc
            fatal_error = {
                "type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
                "phase": "history_trace_write",
            }
    gpu, gpu_probe_error = collect_gpu_provenance(args.gpu_memory_device)
    provenance = build_provenance(
        args,
        commit=git_commit(),
        gpu=gpu,
        gpu_probe_error=gpu_probe_error,
    )
    _augment_memory_provenance(provenance, gpu_memory)
    server_metrics = None
    server_metrics_error = None
    if args.server_metrics_url is not None:
        server_metrics, server_metrics_error = fetch_server_metrics(
            args.server_metrics_url,
            args.request_timeout_s,
        )
    payload = build_payload(
        args,
        workload,
        result,
        gpu_memory,
        provenance,
        teacher_hook,
        trace,
        emitted_trace=emitted_trace,
        fatal_error=fatal_error,
    )
    payload["server_metrics_after_run"] = server_metrics
    payload["server_metrics_error"] = server_metrics_error
    if args.json:
        atomic_write_json(Path(args.json), payload)
        print(f"WROTE {args.json}")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    if pending_error is not None:
        raise pending_error
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["wkvm", "vllm", "sglang"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument(
        "--sglang-native-generate",
        action="store_true",
        help=(
            "Use stock SGLang's token-ID /generate stream as an autonomous "
            "trace source instead of its OpenAI completions endpoint."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--initial-context-tokens", type=int, default=36_864)
    parser.add_argument("--turn-input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens-per-turn", type=int, default=64)
    parser.add_argument("--synthetic-vocab-size", type=int, default=262_144)
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--shared-history-trace-json", default=None)
    trace_group.add_argument("--write-shared-history-trace-json", default=None)
    parser.add_argument(
        "--teacher-forcing-field",
        default="auto",
        help=(
            "Dotted extra-body path for trace token IDs, 'auto', or 'none'. "
            "Auto uses forced_output_ids for WKVM, "
            "vllm_xargs.wkvm_teacher_forced_token_ids for vLLM, and "
            "custom_params.wkvm_teacher_forced_token_ids for SGLang."
        ),
    )
    parser.add_argument(
        "--teacher-forcing-encoding",
        choices=["auto", "array", "json-string"],
        default="auto",
        help=(
            "Auto uses a JSON-array string for vLLM and an array for "
            "WKVM/SGLang."
        ),
    )
    parser.add_argument(
        "--teacher-forcing-processor",
        default=None,
        help=(
            "Serialized SGLang custom_logit_processor, or @path containing it. "
            "The value is sent but only its hash and size are recorded."
        ),
    )
    parser.add_argument("--max-teacher-forced-tokens", type=int, default=4096)
    parser.add_argument(
        "--extra-body-json",
        dest="extra_body",
        type=parse_json_object,
        default={},
    )
    parser.add_argument("--request-order-policy", choices=["forward", "alternating", "seeded-shuffle"], default="alternating")
    parser.add_argument("--request-order-seed", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--max-request-body-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-response-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--server-metrics-url", default=None)
    parser.add_argument("--gpu-memory-device", type=_optional_gpu_device, default="0")
    parser.add_argument("--gpu-memory-sample-interval-s", type=float, default=0.1)
    parser.add_argument(
        "--gpu-memory-baseline-used-mib",
        type=float,
        default=None,
        help=(
            "Whole-device idle memory sampled before the target server starts. "
            "When provided, this becomes the artifact baseline while the "
            "request-start baseline remains recorded separately."
        ),
    )
    parser.add_argument(
        "--memory-ceiling-mib",
        type=float,
        default=None,
        help="Whole-device peak-memory ceiling enforced after artifact writing.",
    )
    parser.add_argument("--semantics", choices=["full_kv", "routed_span_approximate", "other"], default=None)
    parser.add_argument("--engine-version", default=None)
    parser.add_argument("--engine-version-source", default="operator_supplied")
    parser.add_argument("--target-server-launch-command", default=None)
    parser.add_argument(
        "--target-server-config-json",
        dest="target_server_config",
        type=parse_json_object,
        default=None,
    )
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--repeat-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--json", default=None)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
