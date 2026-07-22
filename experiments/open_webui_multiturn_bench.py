#!/usr/bin/env python
"""Open WebUI backend end-to-end multi-turn benchmark.

The measured path matches the browser backend protocol: an authenticated
``POST /api/chat/completions`` returns a task acknowledgement, while streamed
completion updates arrive on the Socket.IO ``events`` channel. The default
workload keeps 32 independent persisted chats active for eight synchronized
turns through one client connection. It is therefore a 32-logical-conversation
test, not a 32-browser-user test.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from wkvm_serving_bench import (  # noqa: E402
    WholeGpuMemoryMonitor,
    collect_gpu_provenance,
)


SCHEMA = "wkvm.open_webui_multiturn_bench.v1"
PROMPT_SOURCE = "deterministic_synthetic_text"
CLIENT_LAYOUT = "one_authenticated_socket_multiple_logical_conversations"
FILLER_ATOMS = (
    " benchmark",
    " data",
    " token",
    " context",
    " x",
    " 0",
    " a",
    "\nbenchmark",
)
SECRET_KEYS = frozenset(
    {
        "authorization",
        "bearer",
        "cookie",
        "password",
        "secret",
        "token",
        "api_key",
        "api_keys",
        "apikey",
        "key",
        "access_token",
        "refresh_token",
    }
)
SECRET_KEY_SUFFIXES = (
    "authorization",
    "_cookie",
    "_password",
    "_secret",
    "_api_key",
    "_api_keys",
    "_access_token",
    "_refresh_token",
)
PROVENANCE_PACKAGES = (
    "wkvm",
    "torch",
    "transformers",
    "python-socketio",
    "websocket-client",
)
WKVM_PARENT_TOKEN_HEADERS = {
    "X-WKVM-Stateful-Chat": "parent-token-v1",
    "X-WKVM-Assistant-Message-ID": "{{MESSAGE_ID}}",
    "X-WKVM-User-Message-ID": "{{USER_MESSAGE_ID}}",
    "X-WKVM-Parent-Message-ID": "{{USER_MESSAGE_PARENT_ID}}",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    open_webui_url: str
    model: str
    run_id: str
    sessions: int = 32
    turns: int = 8
    initial_context_tokens: int = 13_824
    turn_input_tokens: int = 32
    output_tokens_per_turn: int = 128
    request_order_policy: str = "alternating"
    request_order_seed: int = 0
    http_timeout_s: float = 30.0
    turn_timeout_s: float = 1_200.0
    socket_transport: str = "websocket"
    token_env: str = "OPEN_WEBUI_TOKEN"
    engine_name: str | None = None
    engine_version: str | None = None
    open_webui_version: str | None = None
    open_webui_commit: str | None = None
    target_server_launch_command: str | None = None
    target_server_config: Mapping[str, Any] = field(default_factory=dict)
    open_webui_config: Mapping[str, Any] = field(default_factory=dict)
    provider_metrics_url: str | None = None
    require_wkvm_session_reuse: bool = False
    configure_wkvm_parent_token_contract: bool = False
    gpu_memory_device: str | None = None
    gpu_memory_sample_interval_s: float = 0.2

    def validate(self) -> None:
        if self.sessions < 1:
            raise ValueError("sessions must be >= 1")
        if self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.initial_context_tokens < 16:
            raise ValueError("initial_context_tokens must be >= 16")
        if self.turn_input_tokens < 1:
            raise ValueError("turn_input_tokens must be >= 1")
        if self.output_tokens_per_turn < 1:
            raise ValueError("output_tokens_per_turn must be >= 1")
        if self.request_order_policy not in {
            "forward",
            "alternating",
            "seeded-shuffle",
        }:
            raise ValueError("unsupported request_order_policy")
        if self.http_timeout_s <= 0 or self.turn_timeout_s <= 0:
            raise ValueError("timeouts must be > 0")
        if self.gpu_memory_sample_interval_s <= 0:
            raise ValueError("gpu_memory_sample_interval_s must be > 0")
        if self.require_wkvm_session_reuse and not self.provider_metrics_url:
            raise ValueError(
                "require_wkvm_session_reuse requires provider_metrics_url"
            )
        if (
            self.require_wkvm_session_reuse
            and not self.configure_wkvm_parent_token_contract
        ):
            raise ValueError(
                "require_wkvm_session_reuse requires "
                "configure_wkvm_parent_token_contract"
            )


@dataclass(frozen=True)
class TextWorkload:
    initial_contents: list[str]
    turn_contents: list[list[str]]
    initial_rendered_token_counts: list[int]
    turn_content_token_counts: list[list[int]]


@dataclass
class Conversation:
    logical_session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    chat_id: str | None = None
    last_assistant_message_id: str | None = None


@dataclass(frozen=True)
class RequestPlan:
    logical_session_id: str
    session_index: int
    turn_index: int
    request_order_index: int
    message_id: str
    user_message_id: str
    expected_chat_id: str | None
    payload: dict[str, Any]
    local_prompt_tokens: int
    unique_logical_input_tokens: int
    prompt_token_ids_sha256: str
    user_content_sha256: str
    payload_sha256: str


@dataclass
class PendingRequest:
    plan: RequestPlan
    request_start_ns: int
    first_event_ns: int | None = None
    first_content_ns: int | None = None
    ack_ns: int | None = None
    terminal_ns: int | None = None
    ack_chat_id: str | None = None
    task_ids: list[str] = field(default_factory=list)
    event_count: int = 0
    content_event_count: int = 0
    event_bytes: int = 0
    output_text: str = ""
    usage: dict[str, Any] | None = None
    error: str | None = None
    done: bool = False
    terminal_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return canonical_sha256([int(token_id) for token_id in token_ids])


def session_id(index: int) -> str:
    return f"session-{index:04d}"


def request_order_indices(
    session_count: int,
    turn_index: int,
    policy: str,
    seed: int = 0,
) -> list[int]:
    if session_count < 1:
        raise ValueError("session_count must be >= 1")
    order = list(range(session_count))
    if policy == "forward":
        return order
    if policy == "alternating":
        return order if turn_index % 2 == 0 else list(reversed(order))
    if policy != "seeded-shuffle":
        raise ValueError(f"unknown request order policy {policy!r}")
    state = (
        int(seed)
        ^ ((turn_index + 1) * 0x9E3779B1)
        ^ (session_count * 0x85EBCA77)
    ) & 0xFFFF_FFFF
    for index in range(session_count - 1, 0, -1):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFF_FFFF
        swap_index = state % (index + 1)
        order[index], order[swap_index] = order[swap_index], order[index]
    return order


def _normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
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
    add_generation_prompt: bool,
) -> list[int]:
    if not messages:
        return []
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    return _normalize_token_ids(rendered)


def _exact_repeated_text(
    prefix: str,
    target_tokens: int,
    count_tokens: Callable[[str], int],
) -> tuple[str, int]:
    prefix_tokens = count_tokens(prefix)
    if prefix_tokens > target_tokens:
        raise ValueError(
            f"deterministic prefix uses {prefix_tokens} tokens, above target "
            f"{target_tokens}"
        )
    if prefix_tokens == target_tokens:
        return prefix, prefix_tokens

    for atom in FILLER_ATOMS:
        low = 0
        high = max(1, target_tokens - prefix_tokens + 8)
        high_count = count_tokens(prefix + atom * high)
        while high_count < target_tokens and high < target_tokens * 8:
            low = high
            high *= 2
            high_count = count_tokens(prefix + atom * high)
        if high_count < target_tokens:
            continue
        while low + 1 < high:
            middle = (low + high) // 2
            middle_count = count_tokens(prefix + atom * middle)
            if middle_count < target_tokens:
                low = middle
            else:
                high = middle
        for repeats in range(max(0, low - 4), high + 5):
            text = prefix + atom * repeats
            count = count_tokens(text)
            if count == target_tokens:
                return text, count
    raise RuntimeError(
        f"could not construct deterministic text with exactly {target_tokens} tokens"
    )


def exact_initial_content(
    tokenizer: Any,
    *,
    session_index: int,
    target_rendered_tokens: int,
) -> tuple[str, int]:
    nonce = sha256_text(f"open-webui-initial-{session_index}")[:20]
    prefix = (
        f"SESSION-{session_index:04d}-{nonce}\n"
        "Independent deterministic synthetic benchmark context. DATA:"
    )

    def count_tokens(content: str) -> int:
        return len(
            render_chat_token_ids(
                tokenizer,
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
            )
        )

    return _exact_repeated_text(prefix, target_rendered_tokens, count_tokens)


def exact_turn_content(
    tokenizer: Any,
    *,
    session_index: int,
    turn_index: int,
    target_content_tokens: int,
) -> tuple[str, int]:
    nonce = sha256_text(
        f"open-webui-turn-{turn_index}-session-{session_index}"
    )[:8]
    prefix = f"T{turn_index:02d}S{session_index:04d}-{nonce}:"
    return _exact_repeated_text(
        prefix,
        target_content_tokens,
        lambda content: len(encode_text(tokenizer, content)),
    )


def build_workload(
    tokenizer: Any,
    *,
    sessions: int,
    turns: int,
    initial_context_tokens: int,
    turn_input_tokens: int,
) -> TextWorkload:
    initial_contents: list[str] = []
    initial_counts: list[int] = []
    for index in range(sessions):
        content, count = exact_initial_content(
            tokenizer,
            session_index=index,
            target_rendered_tokens=initial_context_tokens,
        )
        initial_contents.append(content)
        initial_counts.append(count)

    turn_contents: list[list[str]] = []
    turn_counts: list[list[int]] = []
    for turn_index in range(1, turns):
        contents: list[str] = []
        counts: list[int] = []
        for index in range(sessions):
            content, count = exact_turn_content(
                tokenizer,
                session_index=index,
                turn_index=turn_index,
                target_content_tokens=turn_input_tokens,
            )
            contents.append(content)
            counts.append(count)
        turn_contents.append(contents)
        turn_counts.append(counts)

    return TextWorkload(
        initial_contents=initial_contents,
        turn_contents=turn_contents,
        initial_rendered_token_counts=initial_counts,
        turn_content_token_counts=turn_counts,
    )


def workload_fingerprints(workload: TextWorkload) -> dict[str, Any]:
    return {
        "prompt_source": PROMPT_SOURCE,
        "initial_content_sha256": canonical_sha256(workload.initial_contents),
        "initial_rendered_token_counts_sha256": canonical_sha256(
            workload.initial_rendered_token_counts
        ),
        "turn_content_sha256": [
            canonical_sha256(contents) for contents in workload.turn_contents
        ],
        "turn_content_token_counts_sha256": [
            canonical_sha256(counts)
            for counts in workload.turn_content_token_counts
        ],
    }


def extract_output_text(data: Mapping[str, Any]) -> str:
    output = data.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)
    content = data.get("content")
    return content if isinstance(content, str) else ""


def usage_counts(usage: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(usage, Mapping):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    prompt_tokens = prompt if type(prompt) is int and prompt >= 0 else None
    completion_tokens = (
        completion if type(completion) is int and completion >= 0 else None
    )
    return prompt_tokens, completion_tokens


def cached_token_count(usage: Mapping[str, Any] | None) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    direct = usage.get("cached_tokens")
    if type(direct) is int and direct >= 0:
        return direct
    details = usage.get("prompt_tokens_details", usage.get("input_tokens_details"))
    if isinstance(details, Mapping):
        value = details.get("cached_tokens")
        if type(value) is int and value >= 0:
            return value
    return None


def usage_total_count(usage: Mapping[str, Any] | None) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    total = usage.get("total_tokens")
    return total if type(total) is int and total >= 0 else None


def _error_text(value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("content", "message", "detail", "error"):
            if key in value:
                return _error_text(value[key])
    return str(value)


class SocketEventTracker:
    def __init__(self, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self.clock_ns = clock_ns
        self._lock = threading.Lock()
        self._requests: dict[str, PendingRequest] = {}
        self.unmatched_event_count = 0

    def register(self, request: PendingRequest) -> None:
        with self._lock:
            if request.plan.message_id in self._requests:
                raise ValueError("duplicate message ID")
            self._requests[request.plan.message_id] = request

    def handle_event(self, payload: Any) -> None:
        now_ns = self.clock_ns()
        if not isinstance(payload, Mapping):
            with self._lock:
                self.unmatched_event_count += 1
            return
        message_id = payload.get("message_id")
        with self._lock:
            request = self._requests.get(str(message_id))
            if request is None:
                self.unmatched_event_count += 1
                return
            request.event_count += 1
            request.event_bytes += len(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            if request.first_event_ns is None:
                request.first_event_ns = now_ns
            event = payload.get("data")
            if not isinstance(event, Mapping):
                return
            event_type = event.get("type")
            data = event.get("data")
            if not isinstance(data, Mapping):
                data = {}
            if event_type == "chat:completion":
                text = extract_output_text(data)
                if text:
                    request.content_event_count += 1
                    request.output_text = text
                    if request.first_content_ns is None:
                        request.first_content_ns = now_ns
                usage = data.get("usage")
                if isinstance(usage, Mapping):
                    request.usage = dict(usage)
                if data.get("error"):
                    request.error = _error_text(data["error"])
                if data.get("done") is True:
                    request.done = True
                    request.terminal_ns = request.terminal_ns or now_ns
                    request.terminal_event.set()
            elif event_type == "chat:message:error":
                request.error = _error_text(data.get("error", "chat message error"))
                request.terminal_ns = request.terminal_ns or now_ns
                request.terminal_event.set()
            elif event_type == "chat:tasks:cancel" and not request.done:
                request.error = request.error or "chat task cancelled"
                request.terminal_ns = request.terminal_ns or now_ns
                request.terminal_event.set()

    def record_ack(self, message_id: str, ack: Any, ack_ns: int) -> None:
        with self._lock:
            request = self._requests[message_id]
            request.ack_ns = ack_ns
            if not isinstance(ack, Mapping):
                request.error = "Open WebUI acknowledgement is not an object"
            elif ack.get("status") is not True:
                request.error = _error_text(ack.get("error", ack))
            else:
                chat_id = ack.get("chat_id")
                if isinstance(chat_id, str) and chat_id:
                    request.ack_chat_id = chat_id
                task_ids = ack.get("task_ids")
                if isinstance(task_ids, list):
                    request.task_ids = [str(task_id) for task_id in task_ids]
                elif ack.get("task_id") is not None:
                    request.task_ids = [str(ack["task_id"])]
                if request.ack_chat_id is None:
                    request.error = "Open WebUI acknowledgement omitted chat_id"
                elif not request.task_ids:
                    request.error = "Open WebUI acknowledgement omitted task_ids"
            if request.error and not request.done:
                request.terminal_ns = request.terminal_ns or ack_ns
                request.terminal_event.set()

    def record_http_error(self, message_id: str, error: BaseException) -> None:
        now_ns = self.clock_ns()
        with self._lock:
            request = self._requests[message_id]
            request.error = f"{type(error).__name__}: {error}"
            request.terminal_ns = request.terminal_ns or now_ns
            request.terminal_event.set()

    def record_timeout(self, request: PendingRequest) -> None:
        now_ns = self.clock_ns()
        with self._lock:
            if request.terminal_event.is_set():
                return
            request.error = "timed out waiting for terminal Socket.IO event"
            request.terminal_ns = now_ns
            request.terminal_event.set()


class SocketIOOpenWebUITransport:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        http_timeout_s: float,
        socket_transport: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.http_timeout_s = http_timeout_s
        self.socket_transport = socket_transport
        self.client: Any = None
        self.session_id: str | None = None

    def connect(self, event_handler: Callable[[Any], None]) -> None:
        try:
            import socketio
        except ImportError as exc:
            raise RuntimeError(
                "Open WebUI benchmarking requires python-socketio; install "
                "`python -m pip install 'python-socketio[client]' websocket-client`"
            ) from exc
        self.client = socketio.Client(
            reconnection=False,
            logger=False,
            engineio_logger=False,
        )
        self.client.on("events", handler=event_handler)
        self.client.connect(
            self.base_url,
            auth={"token": self.token},
            socketio_path="ws/socket.io",
            transports=[self.socket_transport],
            wait=True,
            wait_timeout=self.http_timeout_s,
        )
        self.session_id = self.client.get_sid("/") or self.client.sid
        if not self.session_id:
            raise RuntimeError("Socket.IO connected without a session ID")
        self.client.emit("user-join", {"auth": {"token": self.token}})

    def close(self) -> None:
        if self.client is not None:
            self.client.disconnect()

    def _request_json(
        self,
        path: str,
        *,
        method: str,
        body: Mapping[str, Any] | None = None,
    ) -> Any:
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            data=data,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.http_timeout_s,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Open WebUI HTTP {exc.code}: {detail}") from exc
        return json.loads(raw) if raw else {}

    def get_models(self) -> Any:
        return self._request_json("/api/models?refresh=true", method="GET")

    def get_version(self) -> Any:
        try:
            return self._request_json("/api/version", method="GET")
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def get_openai_config(self) -> Any:
        return self._request_json("/openai/config", method="GET")

    def update_openai_config(self, config: Mapping[str, Any]) -> Any:
        return self._request_json(
            "/openai/config/update",
            method="POST",
            body=config,
        )

    def post_completion(self, payload: Mapping[str, Any]) -> Any:
        return self._request_json(
            "/api/chat/completions",
            method="POST",
            body=payload,
        )


def configure_wkvm_parent_token_contract(transport: Any) -> dict[str, Any]:
    current = transport.get_openai_config()
    if not isinstance(current, Mapping):
        raise RuntimeError("Open WebUI /openai/config did not return an object")
    base_urls = current.get("OPENAI_API_BASE_URLS")
    api_keys = current.get("OPENAI_API_KEYS")
    api_configs = current.get("OPENAI_API_CONFIGS")
    if not isinstance(base_urls, list) or not base_urls:
        raise RuntimeError("Open WebUI has no provider at index 0")
    if not isinstance(api_keys, list):
        raise RuntimeError("Open WebUI config omitted OPENAI_API_KEYS")
    if not isinstance(api_configs, Mapping):
        api_configs = {}

    normalized_configs: dict[str, Any] = {}
    for index, base_url in enumerate(base_urls):
        raw_provider = api_configs.get(str(index), api_configs.get(str(base_url), {}))
        normalized_configs[str(index)] = (
            dict(raw_provider) if isinstance(raw_provider, Mapping) else {}
        )
    provider = normalized_configs["0"]
    raw_headers = provider.get("headers")
    headers = dict(raw_headers) if isinstance(raw_headers, Mapping) else {}
    headers.update(WKVM_PARENT_TOKEN_HEADERS)
    provider["headers"] = headers

    updated = transport.update_openai_config(
        {
            "ENABLE_OPENAI_API": current.get("ENABLE_OPENAI_API"),
            "OPENAI_API_BASE_URLS": list(base_urls),
            "OPENAI_API_KEYS": list(api_keys),
            "OPENAI_API_CONFIGS": normalized_configs,
        }
    )
    if not isinstance(updated, Mapping):
        raise RuntimeError("Open WebUI config update did not return an object")
    observed_configs = updated.get("OPENAI_API_CONFIGS")
    observed_provider = (
        observed_configs.get("0")
        if isinstance(observed_configs, Mapping)
        else None
    )
    observed_headers = (
        observed_provider.get("headers")
        if isinstance(observed_provider, Mapping)
        else None
    )
    configured = isinstance(observed_headers, Mapping) and all(
        observed_headers.get(name) == value
        for name, value in WKVM_PARENT_TOKEN_HEADERS.items()
    )
    if not configured:
        raise RuntimeError("Open WebUI did not retain the WKVM parent-token headers")
    return {
        "requested": True,
        "configured": True,
        "provider_index": "0",
        "header_names": sorted(WKVM_PARENT_TOKEN_HEADERS),
        "expected_headers_sha256": canonical_sha256(WKVM_PARENT_TOKEN_HEADERS),
        "observed_headers_sha256": canonical_sha256(
            {
                name: observed_headers.get(name)
                for name in sorted(WKVM_PARENT_TOKEN_HEADERS)
            }
        ),
    }


def _message_uuid(config: BenchmarkConfig, role: str, session_index: int, turn: int) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"wkvm:{config.run_id}:{role}:{session_index}:{turn}",
        )
    )


def build_request_plan(
    config: BenchmarkConfig,
    tokenizer: Any,
    conversation: Conversation,
    *,
    session_index: int,
    turn_index: int,
    request_order_index: int,
    user_content: str,
    socket_session_id: str,
) -> RequestPlan:
    user_message_id = _message_uuid(config, "user", session_index, turn_index)
    assistant_message_id = _message_uuid(
        config,
        "assistant",
        session_index,
        turn_index,
    )
    request_messages = [
        *conversation.messages,
        {"role": "user", "content": user_content},
    ]
    prompt_ids = render_chat_token_ids(
        tokenizer,
        request_messages,
        add_generation_prompt=True,
    )
    new_user_content_tokens = len(encode_text(tokenizer, user_content))
    unique_input_tokens = (
        len(prompt_ids) if turn_index == 0 else new_user_content_tokens
    )
    user_message = {
        "id": user_message_id,
        "parentId": conversation.last_assistant_message_id,
        "childrenIds": [assistant_message_id],
        "role": "user",
        "content": user_content,
        "timestamp": int(time.time()),
        "models": [config.model],
    }
    payload: dict[str, Any] = {
        "stream": True,
        "stream_options": {"include_usage": True},
        "model": config.model,
        "messages": request_messages,
        "params": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": config.output_tokens_per_turn,
            "seed": config.request_order_seed,
            "stream_delta_chunk_size": 1,
            "reasoning_tags": False,
            "function_calling": "legacy",
            "custom_params": {"ignore_eos": True},
        },
        "features": {},
        "variables": {},
        "session_id": socket_session_id,
        "id": assistant_message_id,
        "message_ids": [
            {
                "model_id": config.model,
                "message_id": assistant_message_id,
            }
        ],
        "parent_id": conversation.last_assistant_message_id,
        "user_message": user_message,
        "background_tasks": {},
    }
    if conversation.chat_id:
        payload["chat_id"] = conversation.chat_id
    return RequestPlan(
        logical_session_id=conversation.logical_session_id,
        session_index=session_index,
        turn_index=turn_index,
        request_order_index=request_order_index,
        message_id=assistant_message_id,
        user_message_id=user_message_id,
        expected_chat_id=conversation.chat_id,
        payload=payload,
        local_prompt_tokens=len(prompt_ids),
        unique_logical_input_tokens=unique_input_tokens,
        prompt_token_ids_sha256=token_ids_sha256(prompt_ids),
        user_content_sha256=sha256_text(user_content),
        payload_sha256=canonical_sha256(payload),
    )


def _dispatch_request(
    tracker: SocketEventTracker,
    transport: Any,
    plan: RequestPlan,
    clock_ns: Callable[[], int],
) -> PendingRequest:
    request = PendingRequest(plan=plan, request_start_ns=clock_ns())
    tracker.register(request)
    try:
        acknowledgement = transport.post_completion(plan.payload)
        tracker.record_ack(plan.message_id, acknowledgement, clock_ns())
    except BaseException as exc:
        tracker.record_http_error(plan.message_id, exc)
    return request


def _hashed_identifier(value: str | None) -> str | None:
    return sha256_text(value) if value else None


def pending_request_record(
    request: PendingRequest,
    *,
    run_origin_ns: int,
    expected_output_tokens: int,
) -> dict[str, Any]:
    prompt_tokens, completion_tokens = usage_counts(request.usage)
    total_tokens = usage_total_count(request.usage)
    terminal_ns = request.terminal_ns
    e2e_s = (
        (terminal_ns - request.request_start_ns) / 1e9
        if terminal_ns is not None
        else None
    )
    ttft_s = (
        (request.first_content_ns - request.request_start_ns) / 1e9
        if request.first_content_ns is not None
        else None
    )
    ack_s = (
        (request.ack_ns - request.request_start_ns) / 1e9
        if request.ack_ns is not None
        else None
    )
    expected_chat_id = request.plan.expected_chat_id
    chat_id_stable = (
        request.ack_chat_id is not None
        and (expected_chat_id is None or request.ack_chat_id == expected_chat_id)
    )
    transport_success = request.done and request.error is None
    usage_complete = (
        prompt_tokens is not None
        and completion_tokens is not None
        and total_tokens is not None
    )
    usage_sum_valid = (
        usage_complete and total_tokens == prompt_tokens + completion_tokens
    )
    accounting_valid = (
        transport_success
        and usage_sum_valid
        and completion_tokens == expected_output_tokens
        and prompt_tokens == request.plan.local_prompt_tokens
    )
    return {
        "logical_session_id": request.plan.logical_session_id,
        "session_index": request.plan.session_index,
        "turn_index": request.plan.turn_index,
        "request_order_index": request.plan.request_order_index,
        "request_start_ns": request.request_start_ns - run_origin_ns,
        "first_event_ns": (
            request.first_event_ns - run_origin_ns
            if request.first_event_ns is not None
            else None
        ),
        "first_content_ns": (
            request.first_content_ns - run_origin_ns
            if request.first_content_ns is not None
            else None
        ),
        "ack_ns": (
            request.ack_ns - run_origin_ns if request.ack_ns is not None else None
        ),
        "terminal_ns": terminal_ns - run_origin_ns if terminal_ns is not None else None,
        "ack_latency_s": round_or_none(ack_s),
        "ui_path_ttft_s": round_or_none(ttft_s),
        "e2e_latency_s": round_or_none(e2e_s),
        "transport_success": transport_success,
        "accounting_valid": accounting_valid,
        "done": request.done,
        "error": request.error,
        "event_count": request.event_count,
        "content_event_count": request.content_event_count,
        "event_bytes": request.event_bytes,
        "chat_id_sha256": _hashed_identifier(request.ack_chat_id),
        "expected_chat_id_sha256": _hashed_identifier(expected_chat_id),
        "chat_id_stable": chat_id_stable,
        "message_id_sha256": _hashed_identifier(request.plan.message_id),
        "task_ids_sha256": canonical_sha256(request.task_ids),
        "task_count": len(request.task_ids),
        "local_rendered_prompt_tokens": request.plan.local_prompt_tokens,
        "provider_prompt_tokens": prompt_tokens,
        "provider_completion_tokens": completion_tokens,
        "provider_total_tokens": total_tokens,
        "provider_cached_tokens": cached_token_count(request.usage),
        "unique_logical_input_tokens": request.plan.unique_logical_input_tokens,
        "usage_complete": usage_complete,
        "usage_sum_valid": usage_sum_valid,
        "usage": redact_secrets(request.usage),
        "output_text_chars": len(request.output_text),
        "output_text_sha256": sha256_text(request.output_text),
        "prompt_token_ids_sha256": request.plan.prompt_token_ids_sha256,
        "user_content_sha256": request.plan.user_content_sha256,
        "payload_sha256": request.plan.payload_sha256,
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_requests: int,
    synchronized_wall_s: float | None = None,
) -> dict[str, Any]:
    starts = [
        int(record["request_start_ns"])
        for record in records
        if type(record.get("request_start_ns")) is int
    ]
    terminals = [
        int(record["terminal_ns"])
        for record in records
        if type(record.get("terminal_ns")) is int
    ]
    wall_s = synchronized_wall_s
    if wall_s is None:
        wall_s = (
            (max(terminals) - min(starts)) / 1e9
            if starts and terminals and max(terminals) >= min(starts)
            else None
        )
    successful = [record for record in records if record.get("transport_success")]
    accounting_complete = (
        len(records) == expected_requests
        and len(successful) == expected_requests
        and all(record.get("accounting_valid") for record in records)
    )
    completion_tokens = sum(
        int(record["provider_completion_tokens"])
        for record in records
        if type(record.get("provider_completion_tokens")) is int
    )
    unique_input_tokens = sum(
        int(record["unique_logical_input_tokens"])
        for record in successful
        if type(record.get("unique_logical_input_tokens")) is int
    )
    application_tokens = unique_input_tokens + completion_tokens
    api_prompt_tokens = sum(
        int(record["provider_prompt_tokens"])
        for record in records
        if type(record.get("provider_prompt_tokens")) is int
    )
    api_accounted_tokens = api_prompt_tokens + completion_tokens
    e2e_values = [
        float(record["e2e_latency_s"])
        for record in successful
        if isinstance(record.get("e2e_latency_s"), (int, float))
    ]
    ttft_values = [
        float(record["ui_path_ttft_s"])
        for record in successful
        if isinstance(record.get("ui_path_ttft_s"), (int, float))
    ]
    ack_values = [
        float(record["ack_latency_s"])
        for record in records
        if isinstance(record.get("ack_latency_s"), (int, float))
    ]

    def rate(numerator: int) -> float | None:
        if not accounting_complete or wall_s is None or wall_s <= 0:
            return None
        return numerator / wall_s

    request_rate = None
    if wall_s is not None and wall_s > 0:
        request_rate = len(successful) / wall_s
    return {
        "expected_requests": expected_requests,
        "recorded_requests": len(records),
        "success_count": len(successful),
        "error_count": len(records) - len(successful),
        "accounting_valid_count": sum(
            1 for record in records if record.get("accounting_valid")
        ),
        "accounting_complete": accounting_complete,
        "synchronized_wall_s": round_or_none(wall_s),
        "generated_output_tokens": completion_tokens,
        "api_accounted_prompt_tokens": api_prompt_tokens,
        "api_accounted_total_tokens": api_accounted_tokens,
        "unique_logical_input_tokens": unique_input_tokens,
        "total_application_tokens": application_tokens,
        "e2e_generated_output_tok_s": round_or_none(rate(completion_tokens)),
        "api_accounted_total_tok_s": round_or_none(rate(api_accounted_tokens)),
        "total_application_goodput_tok_s": round_or_none(rate(application_tokens)),
        "completed_requests_per_s": round_or_none(request_rate),
        "p50_ack_latency_s": round_or_none(percentile(ack_values, 0.50)),
        "p95_ack_latency_s": round_or_none(percentile(ack_values, 0.95)),
        "p99_ack_latency_s": round_or_none(percentile(ack_values, 0.99)),
        "p50_ui_path_ttft_s": round_or_none(percentile(ttft_values, 0.50)),
        "p95_ui_path_ttft_s": round_or_none(percentile(ttft_values, 0.95)),
        "p99_ui_path_ttft_s": round_or_none(percentile(ttft_values, 0.99)),
        "p50_e2e_latency_s": round_or_none(percentile(e2e_values, 0.50)),
        "p95_e2e_latency_s": round_or_none(percentile(e2e_values, 0.95)),
        "p99_e2e_latency_s": round_or_none(percentile(e2e_values, 0.99)),
        "application_goodput_caveat": (
            "Counts each turn-0 rendered prompt, each later user-content token, "
            "and each generated output token once; it is application goodput, "
            "not model-compute throughput."
        ),
        "api_accounted_total_caveat": (
            "Sums provider usage.prompt_tokens and usage.completion_tokens for every "
            "request, so cumulative chat history is intentionally counted again on "
            "later turns; it is API accounting, not unique work or model compute."
        ),
    }


def validate_records(
    records: Sequence[Mapping[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    issues: list[str] = []
    expected = config.sessions * config.turns
    if len(records) != expected:
        issues.append(f"recorded {len(records)} requests, expected {expected}")
    by_session: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        logical_session = str(record.get("logical_session_id"))
        by_session.setdefault(logical_session, []).append(record)
        label = f"{logical_session}/turn-{record.get('turn_index')}"
        if not record.get("transport_success"):
            issues.append(f"{label}: {record.get('error') or 'transport failed'}")
        if not record.get("usage_complete"):
            issues.append(f"{label}: provider usage missing")
        elif not record.get("usage_sum_valid"):
            issues.append(f"{label}: provider total_tokens does not equal prompt + completion")
        if record.get("provider_completion_tokens") != config.output_tokens_per_turn:
            issues.append(
                f"{label}: expected {config.output_tokens_per_turn} completion tokens, "
                f"got {record.get('provider_completion_tokens')}"
            )
        if record.get("provider_prompt_tokens") != record.get(
            "local_rendered_prompt_tokens"
        ):
            issues.append(f"{label}: provider/local prompt-token mismatch")
        if not record.get("chat_id_stable"):
            issues.append(f"{label}: chat ID changed or was missing")
        if not record.get("output_text_chars"):
            issues.append(f"{label}: no assistant text reached the Socket.IO client")
        if record.get("ui_path_ttft_s") is None:
            issues.append(f"{label}: UI-path TTFT unavailable")
        if type(record.get("unique_logical_input_tokens")) is not int or int(
            record["unique_logical_input_tokens"]
        ) <= 0:
            issues.append(f"{label}: invalid unique logical input count")
        expected_unique_input = (
            config.initial_context_tokens
            if record.get("turn_index") == 0
            else config.turn_input_tokens
        )
        if record.get("unique_logical_input_tokens") != expected_unique_input:
            issues.append(
                f"{label}: expected {expected_unique_input} unique logical input "
                f"tokens, got {record.get('unique_logical_input_tokens')}"
            )
    if len(by_session) != config.sessions:
        issues.append(
            f"observed {len(by_session)} logical sessions, expected {config.sessions}"
        )
    for logical_session, session_records in by_session.items():
        chat_hashes = {
            record.get("chat_id_sha256")
            for record in session_records
            if record.get("chat_id_sha256")
        }
        if len(chat_hashes) != 1:
            issues.append(f"{logical_session}: expected one stable chat ID")
    all_chat_hashes = {
        record.get("chat_id_sha256")
        for record in records
        if record.get("chat_id_sha256")
    }
    if len(all_chat_hashes) != config.sessions:
        issues.append(
            f"observed {len(all_chat_hashes)} unique chat IDs, expected {config.sessions}"
        )
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def redact_secrets(value: Any, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in SECRET_KEYS or any(
        normalized.endswith(suffix) for suffix in SECRET_KEY_SUFFIXES
    ):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_secrets(child_value, str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [redact_secrets(item) for item in value]
    return value


def redact_command(command: str | None) -> str | None:
    if command is None:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return "<unparseable command; sha256=" + sha256_text(command) + ">"
    redacted: list[str] = []
    redact_next = False
    for part in parts:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if "=" in part:
            raw_key, _value = part.split("=", 1)
            normalized = raw_key.lstrip("-").lower().replace("-", "_")
            if normalized in SECRET_KEYS or any(
                normalized.endswith(suffix) for suffix in SECRET_KEY_SUFFIXES
            ):
                redacted.append(f"{raw_key}=<redacted>")
                continue
        normalized = part.lstrip("-").lower().replace("-", "_")
        if normalized in SECRET_KEYS or any(
            normalized.endswith(suffix) for suffix in SECRET_KEY_SUFFIXES
        ):
            redacted.append(part)
            redact_next = True
            continue
        redacted.append(part)
    return shlex.join(redacted)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_tree_state() -> dict[str, Any]:
    try:
        tracked = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        full = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {"tracked_clean": None, "clean": None}
    tracked_lines = [line for line in tracked.splitlines() if line]
    full_lines = [line for line in full.splitlines() if line]
    return {
        "tracked_clean": not tracked_lines,
        "clean": not full_lines,
        "tracked_changed_path_count": len(tracked_lines),
        "untracked_path_count": sum(line.startswith("?? ") for line in full_lines),
        "tracked_status_sha256": sha256_text(tracked),
        "status_sha256": sha256_text(full),
    }


def installed_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PROVENANCE_PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except Exception:
            versions[package] = None
    return versions


def model_ids(models_response: Any) -> list[str]:
    if not isinstance(models_response, Mapping):
        return []
    models = models_response.get("data")
    if not isinstance(models, list):
        return []
    return [
        str(model["id"])
        for model in models
        if isinstance(model, Mapping) and model.get("id") is not None
    ]


def fetch_json_snapshot(url: str, timeout_s: float) -> dict[str, Any]:
    captured_at_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
        value = json.loads(raw) if raw else {}
        if not isinstance(value, Mapping):
            raise ValueError("provider metrics response is not a JSON object")
        return {
            "captured_at_utc": captured_at_utc,
            "data": redact_secrets(dict(value)),
            "error": None,
        }
    except Exception as exc:
        return {
            "captured_at_utc": captured_at_utc,
            "data": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def validate_wkvm_session_metrics(
    snapshots: Sequence[Mapping[str, Any]],
    config: BenchmarkConfig,
) -> list[str]:
    final = next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.get("phase") == "after-run"
        ),
        None,
    )
    if not isinstance(final, Mapping) or not isinstance(final.get("data"), Mapping):
        return ["WKVM after-run provider metrics snapshot is missing"]
    data = final["data"]
    server = data.get("server")
    engine = data.get("engine")
    if not isinstance(server, Mapping) or not isinstance(engine, Mapping):
        return ["WKVM provider metrics omitted server or engine data"]
    expected_reuse = config.sessions * max(0, config.turns - 1)
    exact_prefix_hits = server.get("chat_exact_prefix_reuse_hits")
    parent_bound_hits = server.get("parent_bound_continuation_hits")
    server_reuse_hits = (
        exact_prefix_hits + parent_bound_hits
        if type(exact_prefix_hits) is int
        and exact_prefix_hits >= 0
        and type(parent_bound_hits) is int
        and parent_bound_hits >= 0
        else None
    )
    expected = {
        "server.chat_sessions": (server.get("chat_sessions"), config.sessions),
        "server.reuse_hits_total": (server_reuse_hits, expected_reuse),
        "server.parent_bound_continuation_misses": (
            server.get("parent_bound_continuation_misses"),
            0,
        ),
        "server.parent_bound_continuation_rejections": (
            server.get("parent_bound_continuation_rejections"),
            {},
        ),
        "engine.parked_sessions": (engine.get("parked_sessions"), config.sessions),
        "engine.resident_sessions": (engine.get("resident_sessions"), config.sessions),
        "engine.sessions_opened": (engine.get("sessions_opened"), config.sessions),
        "engine.sessions_closed": (engine.get("sessions_closed"), 0),
        "engine.cache_builds": (engine.get("cache_builds"), config.sessions),
        "engine.session_reuse_hits": (
            engine.get("session_reuse_hits"),
            expected_reuse,
        ),
        "engine.session_reuse_misses": (engine.get("session_reuse_misses"), 0),
        "engine.full_reprefill_turns": (engine.get("full_reprefill_turns"), 0),
    }
    return [
        f"{name} expected {wanted}, got {observed}"
        for name, (observed, wanted) in expected.items()
        if observed != wanted
    ]


def _run_turn(
    config: BenchmarkConfig,
    tokenizer: Any,
    workload: TextWorkload,
    conversations: list[Conversation],
    tracker: SocketEventTracker,
    transport: Any,
    *,
    turn_index: int,
    run_origin_ns: int,
    clock_ns: Callable[[], int],
) -> tuple[list[dict[str, Any]], bool]:
    order = request_order_indices(
        config.sessions,
        turn_index,
        config.request_order_policy,
        config.request_order_seed,
    )
    plans: list[RequestPlan] = []
    for request_order_index, session_index in enumerate(order):
        user_content = (
            workload.initial_contents[session_index]
            if turn_index == 0
            else workload.turn_contents[turn_index - 1][session_index]
        )
        plans.append(
            build_request_plan(
                config,
                tokenizer,
                conversations[session_index],
                session_index=session_index,
                turn_index=turn_index,
                request_order_index=request_order_index,
                user_content=user_content,
                socket_session_id=str(transport.session_id),
            )
        )

    requests: list[PendingRequest] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.sessions,
        thread_name_prefix="open-webui-bench",
    ) as executor:
        futures = [
            executor.submit(
                _dispatch_request,
                tracker,
                transport,
                plan,
                clock_ns,
            )
            for plan in plans
        ]
        for future in futures:
            requests.append(future.result())

    first_start_ns = min(request.request_start_ns for request in requests)
    deadline_ns = first_start_ns + int(config.turn_timeout_s * 1e9)
    for request in requests:
        remaining_s = max(0.0, (deadline_ns - clock_ns()) / 1e9)
        if not request.terminal_event.wait(remaining_s):
            tracker.record_timeout(request)

    records = [
        pending_request_record(
            request,
            run_origin_ns=run_origin_ns,
            expected_output_tokens=config.output_tokens_per_turn,
        )
        for request in requests
    ]
    request_by_session = {
        request.plan.session_index: request for request in requests
    }
    turn_ok = True
    for session_index in range(config.sessions):
        request = request_by_session[session_index]
        record = next(
            row for row in records if row["session_index"] == session_index
        )
        if not record["transport_success"] or not request.output_text:
            turn_ok = False
            continue
        conversation = conversations[session_index]
        if conversation.chat_id is None:
            conversation.chat_id = request.ack_chat_id
        elif request.ack_chat_id != conversation.chat_id:
            turn_ok = False
            continue
        conversation.messages.extend(
            [
                {
                    "role": "user",
                    "content": request.plan.payload["user_message"]["content"],
                },
                {"role": "assistant", "content": request.output_text},
            ]
        )
        conversation.last_assistant_message_id = request.plan.message_id
    return records, turn_ok


def run_benchmark(
    config: BenchmarkConfig,
    tokenizer: Any,
    transport: Any,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    config.validate()
    workload = build_workload(
        tokenizer,
        sessions=config.sessions,
        turns=config.turns,
        initial_context_tokens=config.initial_context_tokens,
        turn_input_tokens=config.turn_input_tokens,
    )
    tracker = SocketEventTracker(clock_ns=clock_ns)
    conversations = [Conversation(session_id(index)) for index in range(config.sessions)]
    gpu, gpu_probe_error = collect_gpu_provenance(config.gpu_memory_device)
    if config.gpu_memory_device is not None and gpu_probe_error:
        raise RuntimeError(f"GPU memory probe failed: {gpu_probe_error}")

    transport.connect(tracker.handle_event)
    observed_version: Any = None
    models_response: Any = None
    parent_token_contract = {
        "requested": config.configure_wkvm_parent_token_contract,
        "configured": False,
    }
    all_records: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    provider_metrics_snapshots: list[dict[str, Any]] = []
    monitor: WholeGpuMemoryMonitor | None = None
    run_origin_ns = clock_ns()
    try:
        observed_version = transport.get_version()
        if config.configure_wkvm_parent_token_contract:
            parent_token_contract = configure_wkvm_parent_token_contract(
                transport
            )
        models_response = transport.get_models()
        available_models = model_ids(models_response)
        if config.model not in available_models:
            raise RuntimeError(
                f"model {config.model!r} is absent from Open WebUI /api/models: "
                f"{available_models}"
            )
        if config.provider_metrics_url:
            provider_metrics_snapshots.append(
                {
                    "phase": "before-run",
                    **fetch_json_snapshot(
                        config.provider_metrics_url,
                        config.http_timeout_s,
                    ),
                }
            )
        monitor_context: Any = contextlib.nullcontext(None)
        if config.gpu_memory_device is not None:
            monitor = WholeGpuMemoryMonitor(
                config.gpu_memory_device,
                config.gpu_memory_sample_interval_s,
            )
            monitor_context = monitor
        with monitor_context:
            run_origin_ns = clock_ns()
            for turn_index in range(config.turns):
                records, turn_ok = _run_turn(
                    config,
                    tokenizer,
                    workload,
                    conversations,
                    tracker,
                    transport,
                    turn_index=turn_index,
                    run_origin_ns=run_origin_ns,
                    clock_ns=clock_ns,
                )
                all_records.extend(records)
                turn_rows.append(
                    {
                        "turn_index": turn_index,
                        "request_order": request_order_indices(
                            config.sessions,
                            turn_index,
                            config.request_order_policy,
                            config.request_order_seed,
                        ),
                        **summarize_records(
                            records,
                            expected_requests=config.sessions,
                        ),
                    }
                )
                if config.provider_metrics_url:
                    provider_metrics_snapshots.append(
                        {
                            "phase": f"after-turn-{turn_index}",
                            **fetch_json_snapshot(
                                config.provider_metrics_url,
                                config.http_timeout_s,
                            ),
                        }
                    )
                if not turn_ok:
                    break
            if config.provider_metrics_url:
                provider_metrics_snapshots.append(
                    {
                        "phase": "after-run",
                        **fetch_json_snapshot(
                            config.provider_metrics_url,
                            config.http_timeout_s,
                        ),
                    }
                )
    finally:
        transport.close()

    all_records.sort(key=lambda row: (row["turn_index"], row["session_index"]))
    measured_turn_wall_s = sum(
        float(row["synchronized_wall_s"])
        for row in turn_rows
        if isinstance(row.get("synchronized_wall_s"), (int, float))
    )
    summary = summarize_records(
        all_records,
        expected_requests=config.sessions * config.turns,
        synchronized_wall_s=measured_turn_wall_s,
    )
    turn_zero_records = [row for row in all_records if row["turn_index"] == 0]
    continuation_records = [row for row in all_records if row["turn_index"] > 0]
    summary["turn_0"] = summarize_records(
        turn_zero_records,
        expected_requests=config.sessions,
        synchronized_wall_s=(
            float(turn_rows[0]["synchronized_wall_s"])
            if turn_rows
            and isinstance(turn_rows[0].get("synchronized_wall_s"), (int, float))
            else None
        ),
    )
    continuation_wall_s = sum(
        float(row["synchronized_wall_s"])
        for row in turn_rows[1:]
        if isinstance(row.get("synchronized_wall_s"), (int, float))
    )
    summary["continuation_turns"] = summarize_records(
        continuation_records,
        expected_requests=config.sessions * max(0, config.turns - 1),
        synchronized_wall_s=continuation_wall_s,
    )
    validation = validate_records(all_records, config)
    metrics_errors = [
        row for row in provider_metrics_snapshots if row.get("error") is not None
    ]
    if config.provider_metrics_url and metrics_errors:
        validation["issues"].append(
            f"{len(metrics_errors)} provider metrics snapshot(s) failed"
        )
        validation["issue_count"] = len(validation["issues"])
        validation["passed"] = False
    if config.require_wkvm_session_reuse:
        validation["issues"].extend(
            validate_wkvm_session_metrics(provider_metrics_snapshots, config)
        )
        validation["issue_count"] = len(validation["issues"])
        validation["passed"] = not validation["issues"]
    return {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "passed" if validation["passed"] else "failed",
        "engine": {
            "name": config.engine_name,
            "version": config.engine_version,
            "model": config.model,
            "launch_command": redact_command(config.target_server_launch_command),
            "launch_command_sha256": (
                sha256_text(config.target_server_launch_command)
                if config.target_server_launch_command
                else None
            ),
            "config": redact_secrets(config.target_server_config),
        },
        "open_webui": {
            "base_url": config.open_webui_url,
            "version": config.open_webui_version,
            "commit": config.open_webui_commit,
            "observed_version_response": redact_secrets(observed_version),
            "models_response_sha256": canonical_sha256(models_response),
            "rest_path": "/api/chat/completions",
            "socketio_path": "/ws/socket.io",
            "socket_transport": config.socket_transport,
            "client_layout": CLIENT_LAYOUT,
            "logical_conversations": config.sessions,
            "browser_users": 1,
            "config": redact_secrets(config.open_webui_config),
            "parent_token_contract": parent_token_contract,
            "forwarded_session_contract": (
                "parent-token-v1 binds WKVM state to model/user/chat identity, current "
                "and parent message IDs, exact visible history, and parked raw tokens."
            ),
        },
        "workload": {
            "prompt_source": PROMPT_SOURCE,
            "sessions": config.sessions,
            "turns": config.turns,
            "total_requests": config.sessions * config.turns,
            "initial_rendered_prompt_tokens_per_session": (
                config.initial_context_tokens
            ),
            "continuation_user_content_tokens_per_request": (
                config.turn_input_tokens
            ),
            "output_tokens_per_request": config.output_tokens_per_turn,
            "sampling": {
                "temperature": 0,
                "top_p": 1,
                "seed": config.request_order_seed,
                "ignore_eos": True,
                "stream_options": {"include_usage": True},
            },
            "request_order_policy": config.request_order_policy,
            "request_order_seed": config.request_order_seed,
            "turn_barrier": True,
            "fingerprints": workload_fingerprints(workload),
            "history_fairness_caveat": (
                "Turn 0 uses identical text across engines. Later turns use identical "
                "user deltas but append each engine's own generated assistant text, so "
                "later full histories are autonomous rather than token-identical."
            ),
        },
        "timing_contract": {
            "clock": "time.perf_counter_ns",
            "ack_latency": "HTTP POST start to task acknowledgement",
            "ui_path_ttft": (
                "HTTP POST start to first Socket.IO chat:completion event containing "
                "non-empty assistant output"
            ),
            "e2e_latency": "HTTP POST start to terminal done/error event",
            "throughput_wall": (
                "sum of per-turn first-dispatch-to-last-terminal cohort walls; "
                "inter-turn client construction and telemetry are excluded"
            ),
            "reported_rates": {
                "e2e_generated_output_tok_s": "generated output tokens / synchronized wall",
                "api_accounted_total_tok_s": (
                    "sum of provider prompt_tokens + completion_tokens / synchronized wall"
                ),
                "total_application_goodput_tok_s": (
                    "unique turn-0 rendered prompts + later user content + generated "
                    "output / synchronized wall"
                ),
                "completed_requests_per_s": "successful terminal requests / synchronized wall",
            },
            "itl_available": False,
            "itl_caveat": (
                "Open WebUI emits cumulative and potentially coalesced content events; "
                "event boundaries are not token-exact."
            ),
        },
        "summary": summary,
        "turns": turn_rows,
        "requests": all_records,
        "resources": {
            "gpu": gpu,
            "gpu_probe_error": gpu_probe_error,
            "whole_gpu_memory": monitor.result() if monitor is not None else None,
        },
        "session_cache": {
            "provider_cached_tokens_available_count": sum(
                type(row.get("provider_cached_tokens")) is int for row in all_records
            ),
            "provider_cached_tokens_total": sum(
                int(row["provider_cached_tokens"])
                for row in all_records
                if type(row.get("provider_cached_tokens")) is int
            ),
            "stable_chat_id_sessions": sum(
                1
                for conversation in conversations
                if conversation.chat_id is not None
            ),
        },
        "provider_metrics": {
            "enabled": config.provider_metrics_url is not None,
            "url_sha256": (
                sha256_text(config.provider_metrics_url)
                if config.provider_metrics_url
                else None
            ),
            "snapshots": provider_metrics_snapshots,
            "wkvm_session_reuse_required": config.require_wkvm_session_reuse,
            "parent_token_contract_required": (
                config.configure_wkvm_parent_token_contract
            ),
        },
        "validation": validation,
        "provenance": {
            "wkvm_git_commit": git_commit(),
            "wkvm_git_tree": git_tree_state(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": installed_package_versions(),
            "token_env": config.token_env,
            "token_present": bool(os.environ.get(config.token_env)),
            "secrets_recorded": False,
        },
        "event_routing": {
            "unmatched_event_count": tracker.unmatched_event_count,
        },
    }


def load_tokenizer(path: str, *, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "exact Open WebUI workload construction requires transformers>=5.7; "
            "install the wkvm gemma-server dependencies"
        ) from exc
    return AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"must be a valid JSON object: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("must decode to a JSON object")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-webui-url", default="http://127.0.0.1:3000")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--token-env", default="OPEN_WEBUI_TOKEN")
    parser.add_argument("--run-id", default=f"open-webui-{time.time_ns()}")
    parser.add_argument("--sessions", type=int, default=32)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--initial-context-tokens", type=int, default=13_824)
    parser.add_argument("--turn-input-tokens", type=int, default=32)
    parser.add_argument("--output-tokens-per-turn", type=int, default=128)
    parser.add_argument(
        "--request-order-policy",
        choices=("forward", "alternating", "seeded-shuffle"),
        default="alternating",
    )
    parser.add_argument("--request-order-seed", type=int, default=0)
    parser.add_argument("--http-timeout-s", type=float, default=30.0)
    parser.add_argument("--turn-timeout-s", type=float, default=1_200.0)
    parser.add_argument(
        "--socket-transport",
        choices=("websocket", "polling"),
        default="websocket",
    )
    parser.add_argument("--engine-name")
    parser.add_argument("--engine-version")
    parser.add_argument("--open-webui-version")
    parser.add_argument("--open-webui-commit")
    parser.add_argument("--target-server-launch-command")
    parser.add_argument(
        "--target-server-config-json",
        type=parse_json_object,
        default={},
    )
    parser.add_argument(
        "--open-webui-config-json",
        type=parse_json_object,
        default={},
    )
    parser.add_argument("--gpu-memory-device")
    parser.add_argument("--gpu-memory-sample-interval-s", type=float, default=0.2)
    parser.add_argument(
        "--provider-metrics-url",
        help="Optional unauthenticated JSON metrics endpoint, such as WKVM /metrics.",
    )
    parser.add_argument(
        "--require-wkvm-session-reuse",
        action="store_true",
        help=(
            "Fail unless final WKVM metrics prove all sessions resident/parked, "
            "all continuations reused, and no full reprefills."
        ),
    )
    parser.add_argument(
        "--configure-wkvm-parent-token-contract",
        action="store_true",
        help=(
            "Use the authenticated Open WebUI admin API to install and verify "
            "the parent-token-v1 provider header templates at index 0."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        open_webui_url=args.open_webui_url.rstrip("/"),
        model=args.model,
        run_id=args.run_id,
        sessions=args.sessions,
        turns=args.turns,
        initial_context_tokens=args.initial_context_tokens,
        turn_input_tokens=args.turn_input_tokens,
        output_tokens_per_turn=args.output_tokens_per_turn,
        request_order_policy=args.request_order_policy,
        request_order_seed=args.request_order_seed,
        http_timeout_s=args.http_timeout_s,
        turn_timeout_s=args.turn_timeout_s,
        socket_transport=args.socket_transport,
        token_env=args.token_env,
        engine_name=args.engine_name,
        engine_version=args.engine_version,
        open_webui_version=args.open_webui_version,
        open_webui_commit=args.open_webui_commit,
        target_server_launch_command=args.target_server_launch_command,
        target_server_config=args.target_server_config_json,
        open_webui_config=args.open_webui_config_json,
        provider_metrics_url=args.provider_metrics_url,
        require_wkvm_session_reuse=args.require_wkvm_session_reuse,
        configure_wkvm_parent_token_contract=(
            args.configure_wkvm_parent_token_contract
        ),
        gpu_memory_device=args.gpu_memory_device,
        gpu_memory_sample_interval_s=args.gpu_memory_sample_interval_s,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    token = os.environ.get(config.token_env)
    if not token:
        raise SystemExit(
            f"set {config.token_env} to an authenticated Open WebUI bearer token; "
            "the benchmark never accepts or records tokens on the command line"
        )
    tokenizer = load_tokenizer(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    transport = SocketIOOpenWebUITransport(
        base_url=config.open_webui_url,
        token=token,
        http_timeout_s=config.http_timeout_s,
        socket_transport=config.socket_transport,
    )
    artifact = run_benchmark(config, tokenizer, transport)
    atomic_write_json(args.json, artifact)
    summary = artifact["summary"]
    print(f"status={artifact['status']}")
    print(f"requests={summary['success_count']}/{summary['expected_requests']}")
    print(
        "e2e_generated_output_tok_s="
        f"{summary['e2e_generated_output_tok_s']}"
    )
    print(
        "total_application_goodput_tok_s="
        f"{summary['total_application_goodput_tok_s']}"
    )
    print(f"api_accounted_total_tok_s={summary['api_accounted_total_tok_s']}")
    print(f"completed_requests_per_s={summary['completed_requests_per_s']}")
    print(f"json={args.json}")
    if artifact["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
